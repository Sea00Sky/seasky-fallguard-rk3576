"""Model-selection protocol experiment (, separate from the main ablation table)."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.train import engine  # noqa: E402

CKPT_CONFIG = PROJECT_ROOT / "configs" / "train_stgcnpp_ckpt.yaml"
RUN_ROOT = "runs/ms_v2"
REPORT_ROOT = "reports/model_selection/runs"
OUT_DIR = "reports/model_selection"
SEEDS = [42, 42, 42, 43, 44, 45, 46, 47, 48, 49]
LAST_N = 5
CV_FOLDS = 5
CV_SEED = 42


def run_names(seeds: list[int]) -> list[str]:
    seen: dict[int, int] = {}
    names = []
    for seed in seeds:
        seen[seed] = seen.get(seed, 0) + 1
        n = seen[seed]
        names.append(f"seed{seed}" if n == 1 else f"seed{seed}_rep{n}")
    return names


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["train", "evaluate", "all"])
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--run-root", default=RUN_ROOT)
    parser.add_argument("--out-dir", default=OUT_DIR)
    parser.add_argument("--device", default="")
    return parser.parse_args()


# ------------------------------------------------------------------ stage: train


def stage_train(args) -> int:
    """Train the 10 runs again, this time keeping every epoch's weights."""
    import copy
    import time

    import torch
    import yaml

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    base = engine.load_config(CKPT_CONFIG)
    run_root = engine.resolve(args.run_root)
    run_root.mkdir(parents=True, exist_ok=True)
    report_root = engine.resolve(REPORT_ROOT)
    report_root.mkdir(parents=True, exist_ok=True)

    prereg = {
        "purpose": "compare checkpoint-selection protocols, not models",
        "seeds": [int(s) for s in args.seeds],
        "run_names": run_names(args.seeds),
        "protocols": {
            "val_best": "epoch with highest val fall-class F1 at threshold 0.5 (D-023)",
            "last_n_mean": f"mean fall probability over the last {LAST_N} epochs",
            "event_cv": f"{CV_FOLDS}-fold event-level CV on the train split, seed {CV_SEED}",
        },
        "primary_criterion": "cross-run sd of test window F1 (lower is better)",
        "last_n": LAST_N,
        "cv_folds": CV_FOLDS,
        "cv_seed": CV_SEED,
        "ckpt_config": str(CKPT_CONFIG.relative_to(PROJECT_ROOT)),
        "base_config_effective": base,
        "test_usage": "evaluated once per protocol per run; never used to choose",
        "device": str(device),
        "torch": torch.__version__,
    }
    (run_root / "preregistration.json").write_text(
        json.dumps(prereg, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"preregistration -> {run_root / 'preregistration.json'}", flush=True)

    names = run_names(args.seeds)
    t_all = time.time()
    for i, (seed, name) in enumerate(zip(args.seeds, names), start=1):
        cfg = copy.deepcopy(base)
        cfg["train"]["seed"] = int(seed)
        cfg["output"]["run_dir"] = f"{args.run_root}/{name}"
        cfg["output"]["report"] = f"{REPORT_ROOT}/{name}.md"
        print(f"\n===== [{i}/{len(names)}] seed {seed} ({name}) =====", flush=True)
        t0 = time.time()
        engine.run(cfg, None)
        n_ckpt = len(list((engine.resolve(cfg["output"]["run_dir"]) / "epochs").glob("*.pth")))
        print(f"{name} done in {(time.time() - t0) / 60:.1f} min, {n_ckpt} epoch checkpoints",
              flush=True)
    print(f"\ntraining complete in {(time.time() - t_all) / 60:.1f} min", flush=True)
    return 0


# ------------------------------------------------------------- stage: evaluation


def make_folds(dataset, folds: int, seed: int) -> dict[str, int]:
    """Assign each event to a fold, stratified by dataset."""
    import random

    by_dataset: dict[str, list[str]] = {}
    for info in dataset.video_infos:
        by_dataset.setdefault(str(info["dataset"]), [])
        eid = str(info["event_id"])
        if eid not in by_dataset[str(info["dataset"])]:
            by_dataset[str(info["dataset"])].append(eid)

    rng = random.Random(seed)
    assignment: dict[str, int] = {}
    for dataset_name in sorted(by_dataset):
        events = sorted(by_dataset[dataset_name])
        rng.shuffle(events)
        for j, eid in enumerate(events):
            assignment[eid] = j % folds
    return assignment


def make_eval_loader(dataset, cfg):
    """A non-shuffled loader over an existing dataset."""
    from torch.utils.data import DataLoader

    import torch

    return DataLoader(
        dataset,
        batch_size=int(cfg["data"]["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["data"]["num_workers"]),
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def load_epoch_probs(model, loader, device, ckpt_dir: Path) -> tuple[dict, np.ndarray]:
    """Per-epoch probabilities on one loader, plus the labels (constant)."""
    paths = sorted(ckpt_dir.glob("epoch*.pth"))
    if not paths:
        raise RuntimeError(f"no per-epoch checkpoints in {ckpt_dir}")
    probs = {}
    labels_ref = None
    for path in paths:
        ckpt = engine.load_checkpoint(path, device)
        model.load_state_dict(ckpt["model"])
        labels, p = engine.predict(model, loader, device)
        if labels_ref is None:
            labels_ref = labels
        elif not np.array_equal(labels_ref, labels):
            raise RuntimeError(
                f"{path.name}: label order changed -- the loader is shuffling"
            )
        probs[path.stem] = p
    return probs, labels_ref


def stage_evaluate(args) -> int:
    import torch

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    cfg = engine.load_config(CKPT_CONFIG)
    threshold = float(cfg["eval"]["threshold"])
    run_root = engine.resolve(args.run_root)
    out_dir = engine.resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    names = run_names(args.seeds)

    # Non-shuffled loaders for every split: per-epoch probability vectors are
    # compared element-wise across 30 checkpoints, so the window order must be
    # fixed. The train split is normally shuffled for training; that would make
    # each epoch's vector a different permutation and silently corrupt the CV.
    train_dataset = engine.build_dataset(cfg, "train")
    train_loader = make_eval_loader(train_dataset, cfg)
    val_dataset = engine.build_dataset(cfg, "val")
    val_loader = make_eval_loader(val_dataset, cfg)
    test_dataset = engine.build_dataset(cfg, "test")
    test_loader = make_eval_loader(test_dataset, cfg)

    folds = make_folds(train_dataset, CV_FOLDS, CV_SEED)
    fold_of = np.asarray([folds[str(i["event_id"])] for i in train_dataset.video_infos])
    train_event_ids = np.asarray([str(i["event_id"]) for i in train_dataset.video_infos])
    train_labels = engine.dataset_labels(train_dataset)

    results = []
    for i, name in enumerate(names, start=1):
        run_dir = run_root / name
        ckpt_dir = run_dir / "epochs"
        recorded = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
        model = engine.build_recognizer(cfg).to(device)
        print(f"\n[{i}/{len(names)}] {name}", flush=True)

        # --- protocol A: recorded val-best (no GPU work, replay-verified below)
        best_epoch = int(recorded["best_epoch"])
        test_probs, test_labels = load_epoch_probs(model, test_loader, device, ckpt_dir)
        best_key = f"epoch{best_epoch:03d}"
        if best_key not in test_probs:
            raise RuntimeError(f"{name}: recorded best epoch {best_epoch} has no checkpoint")
        p_val_best = test_probs[best_key]
        m_val_best = engine.binary_metrics(test_labels, p_val_best, threshold)
        ref = recorded["test"]["window"]
        if m_val_best["tp"] != ref["tp"] or abs(m_val_best["f1"] - ref["f1"]) > 1e-6:
            raise RuntimeError(
                f"{name}: val_best replay mismatch "
                f"(tp {m_val_best['tp']} vs {ref['tp']}, f1 {m_val_best['f1']:.6f} "
                f"vs {ref['f1']:.6f})"
            )

        # --- protocol B: mean probability over the last N epochs
        keys = sorted(test_probs)
        last_keys = keys[-LAST_N:]
        p_last = np.mean([test_probs[k] for k in last_keys], axis=0)
        m_last = engine.binary_metrics(test_labels, p_last, threshold)

        # --- protocol C: epoch chosen by 5-fold event-level CV on train
        cv_probs, _ = load_epoch_probs(model, train_loader, device, ckpt_dir)
        cv_scores = []
        for key in sorted(cv_probs):
            p = cv_probs[key]
            fold_f1 = []
            for f in range(CV_FOLDS):
                mask = fold_of == f
                if train_labels[mask].sum() == 0 or train_labels[mask].sum() == mask.sum():
                    continue
                fold_f1.append(
                    engine.binary_metrics(train_labels[mask], p[mask], threshold)["f1"]
                )
            cv_scores.append((float(np.mean(fold_f1)), key))
        cv_scores.sort(reverse=True)
        cv_best_score, cv_best_key = cv_scores[0]
        p_cv = test_probs[cv_best_key]
        m_cv = engine.binary_metrics(test_labels, p_cv, threshold)

        row = {
            "run": name,
            "seed": int(name.split("seed")[1].split("_")[0]),
            "val_best_epoch": best_epoch,
            "val_best_test_f1": m_val_best["f1"],
            "val_best_test_recall": m_val_best["recall"],
            "val_best_test_auc": m_val_best["auc"],
            "last_n_epochs": ",".join(k.replace("epoch", "") for k in last_keys),
            "last_n_test_f1": m_last["f1"],
            "last_n_test_recall": m_last["recall"],
            "last_n_test_auc": m_last["auc"],
            "cv_best_epoch": int(cv_best_key.replace("epoch", "")),
            "cv_train_f1": cv_best_score,
            "cv_test_f1": m_cv["f1"],
            "cv_test_recall": m_cv["recall"],
            "cv_test_auc": m_cv["auc"],
        }
        results.append(row)
        print(
            f"  val_best  ep={best_epoch:2d} test F1={m_val_best['f1']:.4f}\n"
            f"  last{LAST_N}    eps={row['last_n_epochs']} test F1={m_last['f1']:.4f}\n"
            f"  event_cv  ep={row['cv_best_epoch']:2d} train F1={cv_best_score:.4f} "
            f"test F1={m_cv['f1']:.4f}",
            flush=True,
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    csv_path = out_dir / "runs.csv"
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    summary = summarize(results)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_report(out_dir / "summary.md", summary, results)
    print(f"\nwrote {csv_path}", flush=True)
    print(f"wrote {out_dir / 'summary.md'}", flush=True)
    return 0


def summarize(results: list[dict]) -> dict:
    from scipy.stats import t as student_t

    protocols = {
        "val_best": "val_best_test_f1",
        "last_n_mean": "last_n_test_f1",
        "event_cv": "cv_test_f1",
    }
    out = {"n_runs": len(results), "protocols": {}, "criterion": "cross-run sd of test window F1"}
    for name, key in protocols.items():
        values = np.asarray([r[key] for r in results], dtype=np.float64)
        sd = float(values.std(ddof=1))
        t = float(student_t.ppf(0.975, len(values) - 1))
        out["protocols"][name] = {
            "mean": float(values.mean()),
            "sd": sd,
            "min": float(values.min()),
            "max": float(values.max()),
            "range": float(values.max() - values.min()),
            "pi95_single": [float(values.mean() - t * sd * np.sqrt(1 + 1 / len(values))),
                            float(values.mean() + t * sd * np.sqrt(1 + 1 / len(values)))],
            "values": [float(v) for v in values],
        }
    base = out["protocols"]["val_best"]["sd"]
    for name, block in out["protocols"].items():
        block["sd_vs_val_best"] = (
            float(block["sd"] / base) if base > 0 else float("nan")
        )
    out["best_protocol"] = min(out["protocols"], key=lambda k: out["protocols"][k]["sd"])
    return out


def write_report(path: Path, summary: dict, results: list[dict]) -> None:
    p = summary["protocols"]
    lines = [
        "# 选模协议对照（D-037，独立实验，不并入主对照表）",
        "",
        "## 为什么单独做",
        "",
        "D-036 测出基线单次运行的 test F1 噪声底是 [0.791, 0.910]，同时发现 10 次运行的 "
        "val F1 与 test F1 的 Pearson r = +0.041（p = 0.909），best epoch 从 7 散到 28。"
        "也就是说 D-023 的「按 val F1 选 checkpoint」本身就是一个和 seed 同量级的随机源。",
        "",
        "本实验**不改模型、不改数据、不改阈值**，只换「用哪个 epoch 的权重」。",
        "",
        "## 预注册",
        "",
        f"- 训练：与 D-023 逐项相同（`configs/train_stgcnpp_ckpt.yaml` 只多 `checkpoint_every`）",
        f"- 运行：{summary['n_runs']} 次（seed 42×3 + 43–49）",
        f"- 协议 A `val_best`：val F1 最高的 epoch（D-023 现状）",
        f"- 协议 B `last_n_mean`：最后 {LAST_N} 个 epoch 的 fall 概率取均值，**不看 val**",
        f"- 协议 C `event_cv`：在 train 上做 {CV_FOLDS} 折按 event_id 分层 CV 选 epoch，test 只用一次",
        "- **主判据：test 窗口 F1 的跨运行 sd（越小越好）**",
        "- test 不参与任何选择",
        "",
        "## 结果",
        "",
        "| 协议 | mean | **sd** | 极差 | sd 相对 A | pi95_single |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for name, label in (("val_best", "A `val_best`"), ("last_n_mean", f"B `last_n_mean` (N={LAST_N})"),
                        ("event_cv", f"C `event_cv` ({CV_FOLDS} 折)")):
        b = p[name]
        lines.append(
            f"| {label} | {b['mean']:.4f} | **{b['sd']:.4f}** | {b['range']:.4f} | "
            f"{b['sd_vs_val_best']:.2f}× | [{b['pi95_single'][0]:.4f}, {b['pi95_single'][1]:.4f}] |"
        )
    lines += [
        "",
        "逐运行：",
        "",
        "| run | A: val best ep → test F1 | B: last5 → test F1 | C: CV ep → test F1 |",
        "|---|---:|---:|---:|",
    ]
    for r in results:
        lines.append(
            f"| {r['run']} | {r['val_best_epoch']} → {r['val_best_test_f1']:.4f} | "
            f"{r['last_n_test_f1']:.4f} | {r['cv_best_epoch']} → {r['cv_test_f1']:.4f} |"
        )
    winner = summary["best_protocol"]
    lines += [
        "",
        "## 判读",
        "",
        f"- 主判据胜出的是 **`{winner}`**（sd {p[winner]['sd']:.4f} vs A 的 {p['val_best']['sd']:.4f}，"
        f"比值 {p[winner]['sd_vs_val_best']:.2f}×）。",
        "- 注意：sd 只有 10 个样本，本身也不精确（L-029 的教训）；比值 <1.5× 的差异不应过度解读。",
        f"- `last_n_mean` 完全不看 val，因此不受 L-032 的选模随机性影响；"
        f"如果它的 sd 明显更低，说明「不要用 val 挑 epoch」这个方向是对的。",
        f"- `event_cv` 用 train 上的事件级 CV 代替 val 选 epoch，"
        f"若它与 A 接近，说明 val 本身样本量不足（625 窗口 / 88 正窗口）是主因，而不是选模方式。",
        "",
        "## 本实验不回答的问题",
        "",
        "- 不回答「哪个模型更好」：三个协议用的是同一批权重。",
        "- 不回答「阈值该定多少」：阈值固定 0.5，未做搜索。",
        "- 不改变 D-036 的结论：即使换选模协议，单次运行的 F1 仍然只能按带区间的形式报。",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.stage in ("train", "all"):
        rc = stage_train(args)
        if rc != 0 or args.stage == "train":
            return rc
    return stage_evaluate(args)


if __name__ == "__main__":
    raise SystemExit(main())
