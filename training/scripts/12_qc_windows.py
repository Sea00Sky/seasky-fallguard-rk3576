"""QC the built windows and write reports/window_qc.md."""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import paths  # noqa: E402
from src.window import build  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="", help="alternative labels.yaml")
    parser.add_argument("--version", default="v2", choices=["v2", "v3"],
                        help="derived layer whose windows are QC'd")
    return parser.parse_args()


def _pct(v: float) -> str:
    return f"{v * 100:.1f}%"


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args()
    cfg = build.load_config(Path(args.config) if args.config else None, version=args.version)
    if not cfg.windows_npz.exists() or not cfg.windows_csv.exists():
        print("missing windows output; run scripts/11_build_windows.py first")
        return 1

    rows = paths.read_csv(cfg.windows_csv)
    with np.load(cfg.windows_npz, allow_pickle=False) as data:
        keypoints = data["keypoints"].astype(np.float32)
        scores = data["scores"].astype(np.float32)
        labels = data["labels"].astype(np.int8)
    summary = json.loads(cfg.summary_json.read_text(encoding="utf-8")) \
        if cfg.summary_json.exists() else {}

    n = len(rows)
    failures: list[str] = []
    if not (len(keypoints) == len(scores) == len(labels) == n):
        failures.append(
            f"array/row mismatch: rows={n}, kp={len(keypoints)}, "
            f"sc={len(scores)}, labels={len(labels)}"
        )
    if not np.isfinite(keypoints).all() or not np.isfinite(scores).all():
        failures.append("non-finite values in keypoints/scores")
    if set(np.unique(labels).tolist()) - {0, 1}:
        failures.append(f"unexpected labels: {sorted(set(labels.tolist()))}")
    if keypoints.shape[1] != cfg.resample_frames or keypoints.shape[2] != 17:
        failures.append(f"unexpected shape: {keypoints.shape}")
    if len({r["window_id"] for r in rows}) != n:
        failures.append("duplicate window_id")

    out_of_range = float(
        ((keypoints < -0.2) | (keypoints > 1.2)).mean()
    )
    if out_of_range > 0.01:
        failures.append(f"keypoints out of [0,1] range: {_pct(out_of_range)}")

    # Event-level split consistency (D3 / C1).
    event_splits: dict[str, set[str]] = defaultdict(set)
    event_labels: dict[str, set[int]] = defaultdict(set)
    for r, label in zip(rows, labels):
        event_splits[r["event_id"]].add(r["split"])
        if label == 1:
            event_labels[r["event_id"]].add(1)
    mixed = {e: s for e, s in event_splits.items() if len(s) > 1}
    if mixed:
        failures.append(f"events in multiple splits: {len(mixed)}")
    split_events: dict[str, set[str]] = defaultdict(set)
    for event_id, splits in event_splits.items():
        split_events[next(iter(splits))].add(event_id)
    names = sorted(split_events)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            overlap = split_events[a] & split_events[b]
            if overlap:
                failures.append(f"event leakage {a}/{b}: {len(overlap)}")

    # Coverage / score distributions.
    coverage = [float(r["coverage"]) for r in rows]
    min_cov = float(cfg.min_coverage)
    if min(coverage) < min_cov - 1e-9:
        failures.append(f"coverage below threshold: {min(coverage):.3f} < {min_cov}")

    counts = Counter()
    for r, label in zip(rows, labels):
        counts[(r["split"], "fall" if label == 1 else "non_fall")] += 1
        counts[(r["dataset"], "fall" if label == 1 else "non_fall")] += 1

    manifest = paths.read_csv(Path(cfg.manifest_dir) / "videos.csv")
    fall_events = {
        r["event_id"] for r in manifest if r["label"] == "fall" and r["split"]
    }
    positive_events = set(event_labels)
    per_split_events: dict[str, set[str]] = defaultdict(set)
    per_split_pos: dict[str, set[str]] = defaultdict(set)
    for r, label in zip(rows, labels):
        per_split_events[r["split"]].add(r["event_id"])
        if label == 1:
            per_split_pos[r["split"]].add(r["event_id"])

    stats = summary.get("stats", {})
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        f"# Phase 3 窗口构建 QC（{cfg.version}）",
        "",
        f"- 生成时间：{now}",
        f"- 窗口文件：`{paths.rel_to_project(cfg.windows_npz)}` "
        f"SHA256 `{paths.sha256_file(cfg.windows_npz)}`",
        f"- 元数据：`{paths.rel_to_project(cfg.windows_csv)}`",
        f"- 标注层：`{paths.rel_to_project(Path(cfg.annotation_dir))}`；"
        f"清单：`{paths.rel_to_project(Path(cfg.manifest_dir))}`；"
        f"关键点：`{paths.rel_to_project(Path(cfg.keypoint_index))}`",
        f"- 配置：`configs/labels.yaml` SHA256 "
        f"`{paths.sha256_file(paths.PROJECT_ROOT / 'configs' / 'labels.yaml')}`",
        f"- 窗口：{cfg.duration_s} s / 步长 {cfg.stride_s} s / "
        f"guard {cfg.negative_min_distance_s} s / T={cfg.resample_frames} / "
        f"覆盖率 ≥ {cfg.min_coverage}",
        "",
        "## 总体",
        "",
        f"- 保留窗口：{n}（fall {int((labels == 1).sum())} / "
        f"non_fall {int((labels == 0).sum())}）",
        f"- 覆盖率：中位 {_pct(statistics.median(coverage))}，"
        f"最低 {_pct(min(coverage))}",
        f"- 分数：中位 {statistics.median(scores[scores > 0]):.2f}"
        if (scores > 0).any() else "- 分数：n/a",
        f"- 构建统计：{json.dumps(stats, ensure_ascii=False)}",
        "",
        "## 分 split",
        "",
        "| split | 窗口 | fall | non_fall | 事件 | 有正窗口事件 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for split in sorted(per_split_events):
        n_split = sum(1 for r in rows if r["split"] == split)
        lines.append(
            f"| {split} | {n_split} | "
            f"{counts[(split, 'fall')]} | {counts[(split, 'non_fall')]} | "
            f"{len(per_split_events[split])} | {len(per_split_pos[split])} |"
        )

    lines += [
        "",
        "## 分数据集",
        "",
        "| dataset | 窗口 | fall | non_fall |",
        "|---|---:|---:|---:|",
    ]
    for ds in sorted({r["dataset"] for r in rows}):
        lines.append(
            f"| {ds} | {counts[(ds, 'fall')] + counts[(ds, 'non_fall')]} | "
            f"{counts[(ds, 'fall')]} | {counts[(ds, 'non_fall')]} |"
        )

    lines += ["", "## 验收", "", "| 判据 | 结果 |", "|---|---|"]
    checks = [
        ("数组与 CSV 行数一致", len(keypoints) == n),
        ("关键点/分数无 NaN", bool(np.isfinite(keypoints).all() and np.isfinite(scores).all())),
        ("shape = N x T x 17 x 2", keypoints.shape[1:] == (cfg.resample_frames, 17, 2)),
        ("标签只有 0/1", not (set(np.unique(labels).tolist()) - {0, 1})),
        ("event_id 不跨 split", not mixed),
        ("split 事件无交集", not any("leakage" in f for f in failures)),
        ("覆盖率不低于阈值", min(coverage) >= min_cov - 1e-9),
        ("坐标在合理范围", out_of_range <= 0.01),
        ("window_id 唯一", len({r["window_id"] for r in rows}) == n),
    ]
    for name, ok in checks:
        lines.append(f"| {name} | {'PASS' if ok else 'FAIL'} |")

    lines += [
        "",
        "## 已知缺口",
        "",
        f"- fall 事件总数：{len(fall_events)}；有正窗口事件：{len(positive_events)}；"
        f"无正窗口事件：{len(fall_events - set(positive_events))}",
        f"- 因覆盖率丢弃：{stats.get('dropped_coverage', 'n/a')}；"
        f"因边界丢弃：{stats.get('dropped_ambiguous', 'n/a')}；"
        f"因无轨迹丢弃：{stats.get('dropped_no_track', 'n/a')}",
        "",
        "## 判读约定",
        "",
        "- 正窗口 = fall onset 落在窗口内（D-017）；负窗口 = 与 fall 区间无重叠且距区间 ≥ guard。",
        "- 坐标为处理帧归一化坐标（x/宽、y/高）；URFD 为右半裁剪后的 320×240。",
        "- 缺失帧：≤10 帧断档线性插值，其它保持 0；窗口覆盖率按选中轨迹出现帧计。",
        "",
    ]
    report = paths.REPORTS / (
        "window_qc.md" if cfg.version == "v2" else f"window_qc_{cfg.version}.md"
    )
    report.write_text("\n".join(lines), encoding="utf-8")
    print(f"windows: {n} (fall={int((labels == 1).sum())}, non_fall={int((labels == 0).sum())})")
    print(f"report: {report}")
    print("failures:", failures or "none")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
