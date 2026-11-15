"""Pre-registered 5-checkpoint median ensemble evaluation (inference only)."""
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

DEFAULT_RUNS = [
    "stgcnpp_v2",
    "stgcnpp_v2_cw",
    "stgcnpp_v2_aug",
    "stgcnpp_v2_onset",
    "stgcnpp_v2_pretrained",
]
SPLITS = ("val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", default=DEFAULT_RUNS)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default="")
    parser.add_argument("--out-dir", default="reports/ensemble_eval")
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def metrics_of(labels: np.ndarray, probs: np.ndarray, threshold: float) -> dict:
    return engine.binary_metrics(labels, probs, threshold)


def cluster_bootstrap_f1_diff(
    labels: np.ndarray,
    event_ids: list[str],
    probs_a: np.ndarray,
    probs_b: np.ndarray,
    threshold: float,
    rounds: int,
    seed: int,
) -> dict:
    """Percentile CI for F1(b) - F1(a), resampling events (not windows)."""
    rng = np.random.default_rng(seed)
    by_event: dict[str, list[int]] = {}
    for i, eid in enumerate(event_ids):
        by_event.setdefault(str(eid), []).append(i)
    events = list(by_event)
    idx_by_event = [np.asarray(by_event[e], dtype=np.int64) for e in events]
    diffs = []
    for _ in range(rounds):
        pick = rng.integers(0, len(events), size=len(events))
        idx = np.concatenate([idx_by_event[i] for i in pick])
        f1_a = metrics_of(labels[idx], probs_a[idx], threshold)["f1"]
        f1_b = metrics_of(labels[idx], probs_b[idx], threshold)["f1"]
        diffs.append(f1_b - f1_a)
    diffs = np.asarray(diffs)
    return {
        "point_diff": float(
            metrics_of(labels, probs_b, threshold)["f1"]
            - metrics_of(labels, probs_a, threshold)["f1"]
        ),
        "ci95_low": float(np.percentile(diffs, 2.5)),
        "ci95_high": float(np.percentile(diffs, 97.5)),
        "rounds": rounds,
        "unit": "event cluster",
    }


def event_detection_sets(
    dataset, labels: np.ndarray, probs: np.ndarray, threshold: float, split: str
) -> dict:
    """Per-event detection flags, mirroring engine.event_metrics."""
    pred = (probs >= threshold).astype(np.int64)
    by_event: dict[str, list[int]] = {}
    for i, info in enumerate(dataset.video_infos):
        by_event.setdefault(str(info["event_id"]), []).append(i)
    manifest = engine._load_manifest_events(split)
    fall_events = {r["event_id"] for r in manifest if r.get("label") == "fall"}
    nonfall_events = {r["event_id"] for r in manifest if r.get("label") == "non_fall"}
    detected = {
        eid for eid, idxs in by_event.items()
        if any(pred[i] == 1 and labels[i] == 1 for i in idxs)
    }
    far = {
        eid for eid, idxs in by_event.items()
        if eid in nonfall_events and any(pred[i] == 1 for i in idxs)
    }
    return {
        "fall_events": fall_events,
        "nonfall_events": nonfall_events,
        "detected": detected & fall_events,
        "false_alarms": far,
    }


def mcnemar_exact(a_correct: set, b_correct: set, universe: set) -> dict:
    """Exact McNemar on paired event outcomes (two-sided)."""
    b = len((a_correct - b_correct) & universe)   # A right, B wrong
    c = len((b_correct - a_correct) & universe)   # A wrong, B right
    n = b + c
    if n == 0:
        return {"b": 0, "c": 0, "p_value": 1.0}
    # two-sided exact binomial with p=0.5
    from scipy.stats import binomtest

    p = binomtest(min(b, c), n, 0.5, alternative="two-sided").pvalue
    return {"b": b, "c": c, "p_value": float(p)}


def main() -> int:
    args = parse_args()
    import torch

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    cfg = engine.load_config(None)
    threshold = float(args.threshold)

    out_dir = engine.resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    per_split: dict[str, dict] = {}
    all_rows: list[dict] = []

    for split in SPLITS:
        dataset = engine.build_dataset(cfg, split)
        loader = engine.build_loader(cfg, split)[1]
        print(f"[{split}] windows={len(dataset)}", flush=True)

        labels_ref: np.ndarray | None = None
        probs_by_run: dict[str, np.ndarray] = {}
        single_metrics: dict[str, dict] = {}

        for run in args.runs:
            model = engine.build_recognizer(cfg).to(device)
            ckpt_path = engine.resolve(f"runs/{run}/best.pth")
            ckpt = engine.load_checkpoint(ckpt_path, device)
            model.load_state_dict(ckpt["model"])
            labels, probs = engine.predict(model, loader, device)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

            if labels_ref is None:
                labels_ref = labels
            elif not np.array_equal(labels_ref, labels):
                raise RuntimeError(f"{split}: label order differs for {run}")

            probs_by_run[run] = probs
            single_metrics[run] = metrics_of(labels, probs, threshold)

            # Replay check against the recorded training result.
            recorded = json.loads(
                engine.resolve(f"runs/{run}/results.json").read_text(encoding="utf-8")
            )[split]["window"]
            got = single_metrics[run]
            if abs(got["f1"] - recorded["f1"]) > 1e-6 or got["tp"] != recorded["tp"]:
                raise RuntimeError(
                    f"{split}: {run} replay mismatch "
                    f"(got f1={got['f1']:.6f} tp={got['tp']} fp={got['fp']} fn={got['fn']}, "
                    f"recorded f1={recorded['f1']:.6f} tp={recorded['tp']} "
                    f"fp={recorded['fp']} fn={recorded['fn']})"
                )
            print(
                f"  {run:28s} f1={got['f1']:.4f} "
                f"tp/fp/fn={got['tp']}/{got['fp']}/{got['fn']} (replay ok)",
                flush=True,
            )

        # Pre-registered aggregation: median across members, threshold 0.5.
        stack = np.vstack([probs_by_run[r] for r in args.runs])
        median = np.median(stack, axis=0)
        base = probs_by_run[args.runs[0]]

        ens_window = metrics_of(labels_ref, median, threshold)
        ens_event = engine.event_metrics(dataset, labels_ref, median, threshold, split)
        base_window = single_metrics[args.runs[0]]
        base_event = engine.event_metrics(dataset, labels_ref, base, threshold, split)

        sets_base = event_detection_sets(dataset, labels_ref, base, threshold, split)
        sets_ens = event_detection_sets(dataset, labels_ref, median, threshold, split)
        mcnemar = mcnemar_exact(
            sets_base["detected"], sets_ens["detected"], sets_base["fall_events"]
        )

        boot = cluster_bootstrap_f1_diff(
            labels_ref,
            [str(i["event_id"]) for i in dataset.video_infos],
            base,
            median,
            threshold,
            args.bootstrap,
            args.seed,
        )

        per_split[split] = {
            "window": ens_window,
            "event": ens_event,
            "baseline_window": base_window,
            "baseline_event": base_event,
            "bootstrap_f1_diff_vs_baseline": boot,
            "mcnemar_event_detection": mcnemar,
            "single_model_window": single_metrics,
            "n_events_with_windows": len({str(i["event_id"]) for i in dataset.video_infos}),
        }

        for i, info in enumerate(dataset.video_infos):
            row = {
                "split": split,
                "window_id": info.get("frame_dir", ""),
                "event_id": info.get("event_id", ""),
                "video_id": info.get("video_id", ""),
                "dataset": info.get("dataset", ""),
                "start_frame": info.get("start_frame", ""),
                "end_frame": info.get("end_frame", ""),
                "coverage": info.get("coverage", ""),
                "label": int(labels_ref[i]),
                "p_median": float(median[i]),
            }
            for run in args.runs:
                row[f"p_{run}"] = float(probs_by_run[run][i])
            all_rows.append(row)

    csv_path = out_dir / "per_window_probabilities.csv"
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)

    # Pass criterion fixed in the pre-registration.
    test = per_split["test"]
    f1 = test["window"]["f1"]
    events = test["event"]["fall_events_detected"]
    far = test["event"]["nonfall_events_with_false_alarm"]
    passed = f1 >= 0.85 and events >= 26 and far <= 1
    failed = f1 < 0.84 or far > 1 or events <= 25

    summary = {
        "preregistration": {
            "members": args.runs,
            "aggregation": "per-window median",
            "threshold": threshold,
            "note": "aggregation/members/threshold fixed before evaluation; "
                    "no aggregation search on test",
        },
        "splits": per_split,
        "pass_criterion": {"pass": bool(passed), "fail": bool(failed)},
        "per_window_csv": str(csv_path.relative_to(PROJECT_ROOT)),
        "device": str(device),
    }
    json_path = out_dir / "summary.json"
    json_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    write_report(out_dir / "summary.md", summary, args)
    print(f"wrote {csv_path}", flush=True)
    print(f"wrote {json_path}", flush=True)
    print(f"wrote {out_dir / 'summary.md'}", flush=True)
    if failed:
        return 2
    return 0


def write_report(path: Path, summary: dict, args) -> None:
    test = summary["splits"]["test"]
    val = summary["splits"]["val"]
    w, e = test["window"], test["event"]
    bw, be = test["baseline_window"], test["baseline_event"]
    boot = test["bootstrap_f1_diff_vs_baseline"]
    mcn = test["mcnemar_event_detection"]

    lines = [
        "# 5-checkpoint 中位数集成（预注册，推理层）",
        "",
        "## 预注册",
        "",
        f"- 成员：{', '.join(summary['preregistration']['members'])}",
        "- 聚合：逐窗 fall 概率中位数",
        f"- 阈值：{summary['preregistration']['threshold']}（D-023；不用 val 最优阈值）",
        "- 模型、窗口、标签、切分全部不变；test 只评估这一次",
        "- 聚合方式与成员在查看任何 test 数字前固定，未在 test 上搜索",
        "",
        "## 成员复算校验",
        "",
        "| run | test 窗口 F1 | TP/FP/FN |",
        "|---|---:|---:|",
    ]
    for run, m in test["single_model_window"].items():
        lines.append(
            f"| {run} | {m['f1']:.4f} | {m['tp']}/{m['fp']}/{m['fn']} |"
        )
    lines += [
        "",
        "每个成员都先复算了 `runs/*/results.json` 的 test 窗口 F1 与 TP/FP/FN，"
        "不一致时脚本直接报错退出。",
        "",
        "## 结果（阈值 0.5）",
        "",
        "| 指标 | 基线 best.pth | 5-model median |",
        "|---|---:|---:|",
        f"| test 窗口 P | {bw['precision']:.4f} | {w['precision']:.4f} |",
        f"| test 窗口 R | {bw['recall']:.4f} | {w['recall']:.4f} |",
        f"| test 窗口 F1 | {bw['f1']:.4f} | **{w['f1']:.4f}** |",
        f"| test 窗口 AUC | {bw['auc']:.4f} | {w['auc']:.4f} |",
        f"| test 窗口 TP/FP/FN | {bw['tp']}/{bw['fp']}/{bw['fn']} | {w['tp']}/{w['fp']}/{w['fn']} |",
        f"| test 事件检出 | {be['fall_events_detected']}/{be['fall_events_manifest']} | "
        f"**{e['fall_events_detected']}/{e['fall_events_manifest']}** |",
        f"| test 事件误报 | {be['nonfall_events_with_false_alarm']}/{be['nonfall_events']} | "
        f"**{e['nonfall_events_with_false_alarm']}/{e['nonfall_events']}** |",
        f"| val 窗口 F1 | {val['baseline_window']['f1']:.4f} | {val['window']['f1']:.4f} |",
        "",
        "## 统计判读",
        "",
        f"- F1 差值（median − baseline）= {boot['point_diff']:+.4f}；"
        f"按 event 聚簇 bootstrap 的 95% CI = "
        f"[{boot['ci95_low']:+.4f}, {boot['ci95_high']:+.4f}]（{boot['rounds']} 次重采样）。",
        f"- 事件级配对 McNemar：b={mcn['b']}，c={mcn['c']}，p={mcn['p_value']:.3f}。",
        "- 即使 F1 上升，事件级只有 1 个净翻转，**不构成统计显著**；"
        "本报告只把它作为推理层规则，不并入 D-023 训练对照表。",
        "",
        "## 预注册判据",
        "",
        f"- 成功：F1 ≥ 0.85 且事件 ≥ 26/29 且 FAR ≤ 1/16 → "
        f"{'**通过**' if summary['pass_criterion']['pass'] else '未通过'}",
        f"- 失败：F1 < 0.84 或 FAR > 1/16 或事件 ≤ 25/29 → "
        f"{'**触发**' if summary['pass_criterion']['fail'] else '未触发'}",
        "",
        f"逐窗概率：`{summary['per_window_csv']}`（含 5 个成员原始概率与 median，可审计复算）",
        "",
        "## 不解决的问题",
        "",
        "- MCFD hard-view（chute03 cam8 / chute06 cam4 / chute19 cam3）五个成员概率本来就一致很低，"
        "中位数不会改善；根因是前端无骨架（L-013）。",
        "- 板端 173619 倒床漏检是 D-018 选人（L-021），集成不处理。",
        "- RK3576 上 5 模型推理是 5 倍成本；部署时需裁到 2–3 模型或蒸馏，另需批准。",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
