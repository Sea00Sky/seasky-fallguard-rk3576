"""QC the extracted keypoints and write reports/phase2_keypoints.md."""
from __future__ import annotations

import argparse
import statistics
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import paths  # noqa: E402
from src.pose import extract, qc  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="", help="alternative pose.yaml")
    parser.add_argument("--no-visuals", action="store_true")
    return parser.parse_args()


def _pct(v: float) -> str:
    return f"{v * 100:.1f}%"


def _i_safe(value) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args()
    cfg = extract.load_config(Path(args.config) if args.config else None)
    qc_cfg = qc.load_qc_config(Path(args.config) if args.config else None)

    if not cfg.index_path.exists():
        print(f"missing index: {cfg.index_path}")
        return 1
    rows = paths.read_csv(cfg.index_path)
    print(f"index rows: {len(rows)}")

    npz_failures = qc.enrich_intervals(rows)
    g = qc.global_stats(rows)
    per_ds = qc.dataset_stats(rows)
    flags = qc.collect_flags(rows, qc_cfg)

    fall_rows = [r for r in rows if r.get("label") == "fall"]
    fall_sel = [float(r["selected_coverage"]) for r in fall_rows]
    fall_sel_median = statistics.median(fall_sel) if fall_sel else 0.0
    interval_rows = [
        {"video_id": r["video_id"], "dataset": r["dataset"], "label": r.get("label", ""),
         "interval_frames": r.get("interval_frames", 0),
         "interval_covered": r.get("interval_covered", ""),
         "interval_coverage": r.get("interval_coverage", "")}
        for r in rows if _i_safe(r.get("interval_frames")) > 0
    ]
    interval_csv = cfg.output_root / "interval_coverage.csv"
    paths.write_csv(interval_csv, interval_rows)
    print(f"interval coverage rows: {len(interval_rows)} -> {interval_csv}")

    visuals: list[Path] = []
    if not args.no_visuals:
        visuals = qc.make_visuals(rows, qc_cfg, paths.REPORTS / "keypoint_qc")
        print(f"visuals: {len(visuals)}")

    score_min = float(qc_cfg.get("score_median_min", 0.50))
    interval_min = float(qc_cfg.get("interval_coverage_median_min", 0.85))
    statuses_ok = all(r.get("status") in qc.OK_STATUSES for r in rows)
    checks = [
        ("所有视频处理成功", statuses_ok, dict(Counter(r.get("status") for r in rows))),
        ("npz 帧数与 manifest 一致", not npz_failures, npz_failures[:10] or "none"),
        ("无零检测视频", g["zero_detection_videos"] == 0, g["zero_detection_videos"]),
        ("标注区间覆盖中位数", g["interval_median"] >= interval_min,
         f"{_pct(g['interval_median'])} (>= {_pct(interval_min)}, n={g['interval_videos']})"),
        ("关键点分数中位数", g["median_score"] >= score_min,
         f"{g['median_score']:.2f} (>= {score_min:.2f})"),
    ]
    failures = [name for name, ok, _ in checks if not ok]

    model_shas = sorted({r.get("model_sha256", "") for r in rows if r.get("model_sha256")})
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "# Phase 2 关键点抽取 QC",
        "",
        f"- 生成时间：{now}",
        f"- 索引：`{paths.rel_to_project(cfg.index_path)}`",
        f"- 模型：`{paths.rel_to_project(cfg.model_path)}`",
        f"- 模型 SHA256：{', '.join(f'`{s}`' for s in model_shas) or 'n/a'}",
        f"- 配置：`configs/pose.yaml` SHA256 `{paths.sha256_file(paths.POSE_CONFIG)}`",
        f"- 索引 SHA256：`{paths.sha256_file(cfg.index_path)}`",
        "",
        "## 总体",
        "",
        f"- 视频：{g['videos']}；帧：{g['frames']:,}",
        f"- 整段有检测帧覆盖率（仅参考）：{_pct(g['coverage_any'])}",
        f"- 标注区间覆盖中位数：{_pct(g['interval_median'])}"
        f"（{g['interval_videos']} 个有标注区间的视频，最低 {_pct(g['interval_min'])}）",
        f"- 零检测视频：{g['zero_detection_videos']}",
        f"- 关键点分数中位数：{g['median_score']:.2f}",
        f"- 状态：{g['status']}",
        f"- 标记复核视频：{len(flags)}",
        "",
        "## 分数据集",
        "",
        "| 数据集 | 视频 | 帧 | 整段有检测 | 区间覆盖中位数 | 区间覆盖最低 | 分数中位数 | 最低分数 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for ds, s in per_ds.items():
        lines.append(
            f"| {ds} | {s['videos']} | {s['frames']:,} | {_pct(s['coverage_any'])} | "
            f"{_pct(s['interval_median'])} (n={s['interval_videos']}) | "
            f"{_pct(s['interval_min'])} | {s['median_score']:.2f} | {s['min_score']:.2f} |"
        )

    lines += ["", "## 验收", "", "| 判据 | 结果 | 实测 |", "|---|---|---|"]
    for name, ok, detail in checks:
        lines.append(f"| {name} | {'PASS' if ok else 'FAIL'} | {detail} |")

    lines += ["", "## 复核标记", ""]
    if flags:
        lines += ["| video_id | dataset | 原因 |", "|---|---|---|"]
        for f in flags[:80]:
            lines.append(f"| {f['video_id']} | {f['dataset']} | {f['reasons']} |")
        if len(flags) > 80:
            lines.append(f"| … | | 其余 {len(flags) - 80} 条见索引 |")
    else:
        lines.append("- 无")

    if npz_failures:
        lines += ["", "## npz 校验失败", ""]
        lines += [f"- {x}" for x in npz_failures[:50]]

    if visuals:
        lines += ["", "## 骨架抽检图", "",
                  f"目录：`{paths.rel_to_project(paths.REPORTS / 'keypoint_qc')}`；共 {len(visuals)} 张。", "",
                  "| 文件 |", "|---|"]
        for p in visuals[:60]:
            lines.append(f"| `{p.name}` |")
        if len(visuals) > 60:
            lines.append(f"| … 其余 {len(visuals) - 60} 张 |")

    lines += [
        "",
        "## 判读约定",
        "",
        "- `selected_track_id` 取视频中出现帧数最多的轨迹，仅用于 QC 展示；",
        "  窗口阶段可重新选择人物，无需重抽。",
        "- 所有 npz 保留每帧最多 4 个人，因此换选人规则不需要重新推理。",
        "- 坐标位于处理后的帧坐标系；URFD 为右半裁剪，`crop_offset_x=320`。",
        "",
    ]
    report = paths.REPORTS / "phase2_keypoints.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    print(f"report: {report}")
    print("failures:", failures or "none")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
