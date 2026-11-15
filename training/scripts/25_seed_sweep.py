"""Multi-seed replication of the baseline (windows v2, seed sweep 42..49)."""
from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import paths  # noqa: E402
from src.train import engine  # noqa: E402

BASE_CONFIG = PROJECT_ROOT / "configs" / "train_stgcnpp.yaml"
BASELINE_RUN = "runs/stgcnpp_v2"
STORED_KEYS = (
    "labels", "probs", "event_id", "video_id", "start_frame", "end_frame", "dataset",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(42, 50)))
    parser.add_argument("--out-root", default="runs/seed_sweep_v2")
    parser.add_argument("--report-root", default="reports/seed_sweep")
    parser.add_argument("--device", default="")
    return parser.parse_args()


def run_names(seeds: list[int]) -> list[str]:
    """``[42, 42, 43]`` -> ``['seed42', 'seed42_rep2', 'seed43']``."""
    seen: dict[int, int] = {}
    names = []
    for seed in seeds:
        seen[seed] = seen.get(seed, 0) + 1
        n = seen[seed]
        names.append(f"seed{seed}" if n == 1 else f"seed{seed}_rep{n}")
    return names


def seed_config(seed: int, run_name: str, out_root: str, report_root: str) -> dict:
    """Copy the baseline config and override exactly three fields."""
    cfg = copy.deepcopy(engine.load_config(BASE_CONFIG))
    cfg["train"]["seed"] = int(seed)
    cfg["output"]["run_dir"] = f"{out_root}/{run_name}"
    cfg["output"]["report"] = f"{report_root}/{run_name}.md"
    return cfg


def assert_single_variable(cfg: dict) -> None:
    """Fail loudly if the swept config drifts from the baseline anywhere else."""
    base = engine.load_config(BASE_CONFIG)
    allowed = {("train", "seed"), ("output", "run_dir"), ("output", "report")}
    problems = []
    for section in sorted(set(base) | set(cfg)):
        keys = sorted(set(base.get(section, {})) | set(cfg.get(section, {})))
        for key in keys:
            if (section, key) in allowed:
                continue
            if base.get(section, {}).get(key) != cfg.get(section, {}).get(key):
                problems.append(
                    f"{section}.{key}: baseline={base.get(section, {}).get(key)!r} "
                    f"sweep={cfg.get(section, {}).get(key)!r}"
                )
    if problems:
        raise RuntimeError("seed sweep is not single-variable:\n  " + "\n  ".join(problems))


def replay_delta(run_dir: Path, run_name: str) -> dict:
    """Compare a run against runs/stgcnpp_v2 (only meaningful for seed 42)."""
    ref_path = engine.resolve(BASELINE_RUN) / "results.json"
    if not ref_path.exists():
        return {"checked": False, "reason": f"missing {ref_path}"}
    ref = json.loads(ref_path.read_text(encoding="utf-8"))
    got = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    rows = []
    for split in ("val", "test"):
        row = {"split": split}
        for key in ("f1", "recall", "precision", "auc"):
            row[key] = float(got[split]["window"][key]) - float(ref[split]["window"][key])
        for key in ("tp", "fp", "fn", "tn"):
            row[key] = int(got[split]["window"][key]) - int(ref[split]["window"][key])
        rows.append(row)
    return {
        "checked": True,
        "run": run_name,
        "reference": BASELINE_RUN,
        "identical": all(
            abs(r[k]) < 1e-12 for r in rows for k in ("f1", "recall", "precision", "auc")
        ) and all(r[k] == 0 for r in rows for k in ("tp", "fp", "fn", "tn")),
        "best_epoch_baseline": int(ref["best_epoch"]),
        "best_epoch_rerun": int(got["best_epoch"]),
        "deltas": rows,
    }


def store_predictions(run_dir: Path, cfg: dict, device) -> dict:
    """Re-run the best checkpoint and store per-window probabilities + audit."""
    layer = engine.layer_of(cfg)
    ann_path = engine.resolve(cfg["data"]["ann_file"])
    model = engine.build_recognizer(cfg).to(device)
    checkpoint = engine.load_checkpoint(run_dir / "best.pth", device)
    model.load_state_dict(checkpoint["model"])

    recorded = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    threshold = float(cfg["eval"]["threshold"])
    out: dict[str, np.ndarray] = {}
    audit: dict = {
        "layer": layer,
        "ann_file": cfg["data"]["ann_file"],
        "ann_file_sha256": paths.sha256_file(ann_path),
        "threshold": threshold,
        "splits": {},
    }
    for split in ("val", "test"):
        dataset = engine.build_dataset(cfg, split)
        loader = engine.build_loader(cfg, split)[1]
        labels, probs = engine.predict(model, loader, device)
        got = engine.binary_metrics(labels, probs, threshold)
        ref = recorded[split]["window"]
        if got["tp"] != ref["tp"] or abs(got["f1"] - ref["f1"]) > 1e-9:
            raise RuntimeError(
                f"{split}: stored predictions disagree with results.json "
                f"(tp {got['tp']} vs {ref['tp']}, f1 {got['f1']:.9f} vs {ref['f1']:.9f})"
            )
        out[f"{split}_labels"] = labels
        out[f"{split}_probs"] = probs.astype(np.float64)
        infos = dataset.video_infos
        out[f"{split}_event_id"] = np.asarray([str(i["event_id"]) for i in infos])
        out[f"{split}_video_id"] = np.asarray([str(i["video_id"]) for i in infos])
        out[f"{split}_start_frame"] = np.asarray([int(i["start_frame"]) for i in infos], dtype=np.int64)
        out[f"{split}_end_frame"] = np.asarray([int(i["end_frame"]) for i in infos], dtype=np.int64)
        out[f"{split}_dataset"] = np.asarray([str(i["dataset"]) for i in infos])
        audit["splits"][split] = {
            "n": int(len(labels)), "n_pos": int(labels.sum()),
            "tp": got["tp"], "fp": got["fp"], "fn": got["fn"], "tn": got["tn"],
            "f1": got["f1"], "replay_matches_results_json": True,
        }
    del model
    if device.type == "cuda":
        import torch
        torch.cuda.empty_cache()

    np.savez_compressed(run_dir / "per_window.npz", **out)
    (run_dir / "audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return audit


def main() -> int:
    args = parse_args()
    import torch

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    out_root = engine.resolve(args.out_root)
    report_root = engine.resolve(args.report_root)
    out_root.mkdir(parents=True, exist_ok=True)
    report_root.mkdir(parents=True, exist_ok=True)
    names = run_names(args.seeds)

    base = engine.load_config(BASE_CONFIG)
    prereg = {
        "purpose": "measure run-to-run variance of the D-023 baseline so ablations <0.05 F1 can be judged",
        "seeds": [int(s) for s in args.seeds],
        "run_names": names,
        "repeats_note": "a seed listed more than once is a within-seed repeat; "
                        "it measures GPU run-to-run non-determinism, not seed sensitivity",
        "single_variable": "train.seed (all other config fields asserted equal to configs/train_stgcnpp.yaml)",
        "base_config": str(BASE_CONFIG.relative_to(PROJECT_ROOT)),
        "base_config_sha256": paths.sha256_file(BASE_CONFIG),
        "base_config_effective": base,
        "model_selection": "best checkpoint by val fall-class F1 at threshold 0.5 (D-023, unchanged)",
        "threshold": float(base["eval"]["threshold"]),
        "test_usage": "evaluated once per run by the training engine; analysis reads stored probabilities only",
        "device": str(device),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
    }
    (out_root / "preregistration.json").write_text(
        json.dumps(prereg, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"preregistration -> {out_root / 'preregistration.json'}", flush=True)

    manifest_rows: list[dict] = []
    replay_rows: list[dict] = []
    t_all = time.time()

    for i, (seed, name) in enumerate(zip(args.seeds, names), start=1):
        cfg = seed_config(seed, name, args.out_root, args.report_root)
        assert_single_variable(cfg)
        run_dir = engine.resolve(cfg["output"]["run_dir"])
        print(f"\n===== [{i}/{len(names)}] seed {seed} ({name}) -> {cfg['output']['run_dir']} =====",
              flush=True)

        t0 = time.time()
        results = engine.run(cfg, None)
        elapsed = time.time() - t0

        replay = replay_delta(run_dir, name)
        if replay.get("checked"):
            replay_rows.append(replay)
            if not replay["identical"]:
                t = next(r for r in replay["deltas"] if r["split"] == "test")
                print(
                    f"!! seed {seed} ({name}) does not reproduce {BASELINE_RUN}: "
                    f"test F1 {t['f1']:+.4f} R {t['recall']:+.4f} "
                    f"(tp {t['tp']:+d} fn {t['fn']:+d}) -- expected, see cudnn.deterministic",
                    flush=True,
                )

        audit = store_predictions(run_dir, cfg, device)
        row = {
            "run": name,
            "seed": seed,
            "run_dir": cfg["output"]["run_dir"],
            "best_epoch": int(results["best_epoch"]),
            "elapsed_min": round(elapsed / 60.0, 2),
            "val_f1": results["val"]["window"]["f1"],
            "val_recall": results["val"]["window"]["recall"],
            "val_precision": results["val"]["window"]["precision"],
            "val_auc": results["val"]["window"]["auc"],
            "test_f1": results["test"]["window"]["f1"],
            "test_recall": results["test"]["window"]["recall"],
            "test_precision": results["test"]["window"]["precision"],
            "test_auc": results["test"]["window"]["auc"],
            "test_tp": results["test"]["window"]["tp"],
            "test_fp": results["test"]["window"]["fp"],
            "test_fn": results["test"]["window"]["fn"],
            "test_tn": results["test"]["window"]["tn"],
            "test_event_detected": results["test"]["event"]["fall_events_detected"],
            "test_event_manifest": results["test"]["event"]["fall_events_manifest"],
            "test_false_alarms": results["test"]["event"]["nonfall_events_with_false_alarm"],
            "test_nonfall_events": results["test"]["event"]["nonfall_events"],
            "val_event_detected": results["val"]["event"]["fall_events_detected"],
            "val_event_manifest": results["val"]["event"]["fall_events_manifest"],
            "val_false_alarms": results["val"]["event"]["nonfall_events_with_false_alarm"],
            "val_nonfall_events": results["val"]["event"]["nonfall_events"],
            "ann_file_sha256": audit["ann_file_sha256"],
        }
        manifest_rows.append(row)

        with open(out_root / "runs_manifest.csv", "w", newline="", encoding="utf-8-sig") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(manifest_rows[0].keys()))
            writer.writeheader()
            writer.writerows(manifest_rows)

        print(
            f"{name} done in {elapsed / 60:.1f} min | "
            f"test F1={row['test_f1']:.4f} R={row['test_recall']:.3f} "
            f"event={row['test_event_detected']}/{row['test_event_manifest']} "
            f"FA={row['test_false_alarms']}/{row['test_nonfall_events']}",
            flush=True,
        )

    (out_root / "replay_check.json").write_text(
        json.dumps(replay_rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    total = (time.time() - t_all) / 60.0
    print(f"\nsweep complete: {len(names)} runs in {total:.1f} min", flush=True)
    print(f"manifest -> {out_root / 'runs_manifest.csv'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
