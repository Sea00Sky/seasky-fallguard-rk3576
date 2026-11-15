"""Build manifest v2, split v2 and the v2 QC report from manual intervals."""
from __future__ import annotations

import os
import statistics
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import apply_manual, build_events, make_splits, paths  # noqa: E402

V2_ANNOTATIONS = paths.ANNOTATIONS / "v2"
V2_MANIFESTS = paths.MANIFESTS / "v2"
V2_SPLIT = paths.SPLITS / "v2.json"
QC_REPORT = paths.REPORTS / "split_v2_qc.md"

PINNED = {
    "datasets/annotations/manual/interval_annotations.csv":
        "e5f5425bce350ee790335ed631e30440583caab6ee3200f5467ed817f89139f1",
    "datasets/splits/v1.json":
        "84354f4bb2a2ac6399b595d27401e940e04d1840bfcae5c444b693e3bcc0ad68",
    "datasets/manifests/videos.csv":
        "b044b8502cf717768bffde5fa0cfd6e267818aedab0b6a3f65a4db077564e582",
    "datasets/manifests/events.csv":
        "cbb7d44cc9a7ef04483baca98d28ec4d01e959677f4f597f900e9d14dcd4311c",
    "datasets/annotations/le2i.csv":
        "b160349b607499d77d3f85a0182b241f366c40f797fe7b855ecb7f3100654c28",
    "datasets/annotations/urfd.csv":
        "22dc6df9a69078f6d6ad0cb58e62dac2148aafd125e9a7453eb2d82f9bbef8ef",
    "datasets/annotations/mcfd.csv":
        "2e1fc6d6cf7002c69a1d2d4211f16107f56fec283caecbbc53cc0d76b0143a16",
}

EXPECTED_LAYERS = {
    ("le2i", "fall"): 128,
    ("le2i", "non_fall"): 59,
    ("mcfd", "fall"): 23,
    ("urfd", "fall"): 30,
    ("urfd", "non_fall"): 40,
}
EXPECTED_SPLIT = {"train": 195, "val": 40, "test": 45}
EXPECTED_FALL_SPLIT = {"train": 126, "val": 26, "test": 29}
EXPECTED_NON_FALL_SPLIT = {"train": 69, "val": 14, "test": 16}

LE2I_STATUS = Counter({
    "valid": 96, "manual_annotated": 34, "manual_no_fall": 59, "invalid_interval": 1,
})
URFD_STATUS = Counter({
    "manual_annotated": 30, "all_negative": 40, "duplicate_conv": 3,
})
MCFD_STATUS = Counter({
    "valid": 551, "chute24_unannotated": 8, "no_video": 1,
})


def _i(value: str) -> int:
    return int(str(value).strip())


def _stats(values: list[float]) -> str:
    if not values:
        return "n/a"
    return f"{min(values):.2f} / {statistics.median(values):.2f} / {max(values):.2f}"


def _check_pinned() -> list[str]:
    failures = []
    for rel, expected in PINNED.items():
        actual = paths.sha256_file(paths.PROJECT_ROOT / rel)
        if actual != expected:
            failures.append(f"{rel}: sha256 {actual} != pinned {expected}")
    return failures


def _validate_annotations(videos: list[dict]) -> tuple[list[str], dict]:
    failures: list[str] = []
    frames = {v["video_id"]: _i(v["n_frames"]) for v in videos}

    le2i = paths.read_csv(V2_ANNOTATIONS / "le2i.csv")
    urfd = paths.read_csv(V2_ANNOTATIONS / "urfd.csv")
    mcfd = paths.read_csv(V2_ANNOTATIONS / "mcfd.csv")

    if Counter(r["status"] for r in le2i) != LE2I_STATUS:
        failures.append(f"le2i status counts: {dict(Counter(r['status'] for r in le2i))}")
    if Counter(r["status"] for r in urfd) != URFD_STATUS:
        failures.append(f"urfd status counts: {dict(Counter(r['status'] for r in urfd))}")
    if Counter(r["status"] for r in mcfd) != MCFD_STATUS:
        failures.append(f"mcfd status counts: {dict(Counter(r['status'] for r in mcfd))}")

    end_at_n = []
    for row in le2i:
        vid, status = row["video_id"], row["status"]
        n = frames.get(vid, -1)
        if status in ("valid", "manual_annotated"):
            if not row["start_frame"] or not row["end_frame"]:
                failures.append(f"{vid}: {status} without interval")
                continue
            s, e = _i(row["start_frame"]), _i(row["end_frame"])
            if not (0 <= s <= e <= n):
                failures.append(f"{vid}: interval [{s}, {e}] outside [0, {n}]")
            if e == n:
                end_at_n.append(vid)
            if _i(row["interval_frames"]) != e - s:
                failures.append(f"{vid}: interval_frames {row['interval_frames']} != {e - s}")
            if status == "manual_annotated" and e >= n:
                failures.append(f"{vid}: manual interval end {e} >= n_frames {n}")
        elif status == "manual_no_fall":
            if row["start_frame"] or row["end_frame"] or row["interval_frames"]:
                failures.append(f"{vid}: manual_no_fall carries interval")
        elif status == "invalid_interval":
            if not (_i(row["start_frame"]) >= _i(row["end_frame"])):
                failures.append(f"{vid}: invalid_interval is actually valid")
        else:
            failures.append(f"{vid}: unexpected le2i status {status}")

    for row in urfd:
        vid, label, status = row["video_id"], row["label"], row["status"]
        n = _i(row["n_frames"])
        if status == "duplicate_conv":
            if row["fall_start_frame"] or row["fall_end_frame"]:
                failures.append(f"{vid}: duplicate_conv carries interval")
            continue
        if label == "fall":
            if status != "manual_annotated":
                failures.append(f"{vid}: fall status {status}")
                continue
            s, e = _i(row["fall_start_frame"]), _i(row["fall_end_frame"])
            if not (0 <= s <= e < n):
                failures.append(f"{vid}: interval [{s}, {e}] outside [0, {n - 1}]")
        else:
            if row["fall_start_frame"] or row["fall_end_frame"]:
                failures.append(f"{vid}: non-fall carries interval")

    for row in mcfd:
        if row["status"] == "valid":
            s, e, n = _i(row["start_frame"]), _i(row["end_frame"]), _i(row["n_frames"])
            if not (0 <= s <= e < n):
                failures.append(f"mcfd chute{row['chute']}/cam{row['cam']}: [{s}, {e}] n={n}")
            if _i(row["interval_frames"]) != e - s:
                failures.append(f"mcfd chute{row['chute']}/cam{row['cam']}: interval_frames")

    return failures, {
        "le2i": le2i, "urfd": urfd, "mcfd": mcfd,
        "end_at_n": end_at_n,
    }


def _validate_events(events: list[dict], annotations: dict) -> tuple[list[str], dict]:
    failures: list[str] = []
    le2i = {r["video_id"]: r for r in annotations["le2i"]}
    urfd = {r["video_id"]: r for r in annotations["urfd"]}
    mcfd_rows: dict[str, list[dict]] = {}
    for row in annotations["mcfd"]:
        eid = f"mcfd:chute{int(row['chute']):02d}"
        mcfd_rows.setdefault(eid, []).append(row)

    eligible = [e for e in events if e["eligible_for_split"] == "True"]
    layer = Counter((e["dataset"], e["label"]) for e in eligible)
    if dict(layer) != {k: v for k, v in EXPECTED_LAYERS.items()}:
        failures.append(f"eligible layers: {dict(layer)}")

    for ev in eligible:
        eid, label, ds = ev["event_id"], ev["label"], ev["dataset"]
        if ds == "le2i":
            row = le2i[eid]
            if label == "fall" and row["status"] not in ("valid", "manual_annotated"):
                failures.append(f"{eid}: eligible fall status {row['status']}")
            if label == "fall" and not row["start_frame"]:
                failures.append(f"{eid}: eligible fall without interval")
            if label == "non_fall" and row["status"] != "manual_no_fall":
                failures.append(f"{eid}: eligible non_fall status {row['status']}")
        elif ds == "urfd":
            row = urfd[eid]
            expected = "manual_annotated" if label == "fall" else "all_negative"
            if row["status"] != expected:
                failures.append(f"{eid}: {label} status {row['status']} != {expected}")
        elif ds == "mcfd":
            rows = [r for r in mcfd_rows.get(eid, []) if r["status"] == "valid"]
            if label == "fall" and not any(_i(r["label"]) == 1 for r in rows):
                failures.append(f"{eid}: eligible fall without a label-1 camera")
        else:
            failures.append(f"{eid}: unexpected dataset {ds}")

    dups = [e for e in events if "duplicate_content" in (e.get("notes") or "")]
    if len(dups) != 3:
        failures.append(f"duplicate_content events: {len(dups)}")
    for ev in dups:
        if ev["eligible_for_split"] != "False" or ev["split"]:
            failures.append(f"{ev['event_id']}: duplicate still eligible/split")

    return failures, {
        "eligible": len(eligible),
        "fall": sum(1 for e in eligible if e["label"] == "fall"),
        "non_fall": sum(1 for e in eligible if e["label"] == "non_fall"),
        "dups": dups,
    }


def _validate_split(payload: dict, events: list[dict], videos: list[dict]) -> list[str]:
    failures = []
    bad = {k: v for k, v in payload["checks"].items() if v}
    if bad:
        failures.append(f"leakage checks: {bad}")
    if payload["counts"]["events"] != EXPECTED_SPLIT:
        failures.append(f"split events: {payload['counts']['events']}")
    if payload["counts"]["fall_events"] != EXPECTED_FALL_SPLIT:
        failures.append(f"split fall: {payload['counts']['fall_events']}")
    non_fall_split = Counter(
        ev["split"] for ev in events if ev["split"] and ev["label"] == "non_fall"
    )
    if dict(non_fall_split) != EXPECTED_NON_FALL_SPLIT:
        failures.append(f"split non_fall: {dict(non_fall_split)}")
    if len(payload["event_to_split"]) != 280:
        failures.append(f"event_to_split: {len(payload['event_to_split'])}")

    unreadable = [v["video_id"] for v in videos if v["readable"] != "True"]
    if unreadable:
        failures.append(f"unreadable videos: {unreadable}")

    for ev in events:
        if ev["eligible_for_split"] == "True" and ev["event_id"] not in payload["event_to_split"]:
            failures.append(f"{ev['event_id']}: eligible but not in split")
        if ev["event_id"] in payload["event_to_split"]:
            if ev["split"] != payload["event_to_split"][ev["event_id"]]:
                failures.append(f"{ev['event_id']}: event split mismatch")
    return failures


def _interval_stats(annotations: dict) -> dict:
    le2i = [
        (_i(r["start_frame"]), _i(r["end_frame"]))
        for r in annotations["le2i"] if r["status"] == "manual_annotated"
    ]
    urfd = [
        (_i(r["fall_start_frame"]), _i(r["fall_end_frame"]))
        for r in annotations["urfd"] if r["status"] == "manual_annotated"
    ]
    def summarize(pairs, fps):
        frames = [e - s + 1 for s, e in pairs]
        secs = [f / fps for f in frames]
        return {
            "n": len(pairs),
            "frames": (min(frames), statistics.median(frames), max(frames)),
            "seconds": (min(secs), statistics.median(secs), max(secs)),
        }
    return {"le2i": summarize(le2i, 25.0), "urfd": summarize(urfd, 30.0)}


def _write_report(merge_stats, ann_info, ev_info, payload, hashes, failures) -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    iv = _interval_stats(ann_info)
    lines = [
        "# Split v2 QC",
        "",
        f"- 生成时间：{now}",
        f"- 切分文件：`{V2_SPLIT.relative_to(paths.PROJECT_ROOT).as_posix()}`",
        f"- seed：{payload['seed']}；比例：{payload['ratios']}；单位：event_id",
        "",
        "## 冻结输入（SHA256）",
        "",
        "| 文件 | SHA256 |",
        "|---|---|",
    ]
    for rel, digest in hashes.items():
        lines.append(f"| `{rel}` | `{digest}` |")

    lines += [
        "",
        "## 人工合并",
        "",
        f"- 人工行：{merge_stats['manual_rows']}；匹配：{merge_stats['matched']}；"
        f"未匹配：{len(merge_stats['unmatched'])}；审计行：{merge_stats['audit_rows']}",
        f"- 动作：{merge_stats['actions']}",
        f"- Le2i 状态：{merge_stats['le2i_status_counts']}",
        f"- URFD 状态：{merge_stats['urfd_status_counts']}",
        "",
        "## 可切分事件",
        "",
        "| 数据集:标签 | v1 | v2 |",
        "|---|---|---|",
        "| le2i:fall | 94 | 128 |",
        "| le2i:non_fall | 0 | 59 |",
        "| mcfd:fall | 23 | 23 |",
        "| urfd:fall | 30 | 30 |",
        "| urfd:non_fall | 40 | 40 |",
        "| **合计** | **187** | **280** |",
        "",
        f"- v2：fall {ev_info['fall']} / non_fall {ev_info['non_fall']} / 合计 {ev_info['eligible']}",
        "",
        "## 切分计数",
        "",
        "| split | 全部事件 | fall | non_fall |",
        "|---|---|---|---|",
    ]
    non_fall_split = Counter(
        ev["split"] for ev in ann_info["events"] if ev["split"] and ev["label"] == "non_fall"
    )
    for split in ("train", "val", "test"):
        lines.append(
            f"| {split} | {payload['counts']['events'].get(split, 0)} | "
            f"{payload['counts']['fall_events'].get(split, 0)} | "
            f"{non_fall_split.get(split, 0)} |"
        )

    lines += [
        "",
        "## 区间统计（人工标注，闭区间帧数）",
        "",
        "| 数据集 | 条数 | 帧 min/median/max | 秒 min/median/max |",
        "|---|---|---|---|",
    ]
    for ds, s in iv.items():
        lines.append(
            f"| {ds} | {s['n']} | {_stats(list(s['frames']))} | {_stats(list(s['seconds']))} |"
        )

    lines += [
        "",
        "## 泄漏检查",
        "",
        f"- {payload['checks']}",
        "",
        "## 重复内容事件",
        "",
        "| event_id | label | eligible | split | notes |",
        "|---|---|---|---|---|",
    ]
    for ev in ev_info["dups"]:
        lines.append(
            f"| {ev['event_id']} | {ev['label']} | {ev['eligible_for_split']} | "
            f"`{ev['split']}` | {ev['notes']} |"
        )

    lines += [
        "",
        "## 已知例外",
        "",
        "- Le2i 原始 `valid` 行中 `"
        + ", ".join(ann_info["end_at_n"])
        + "` 的 `end_frame == n_frames`。原始解析允许 `end <= n_frames`，"
        "按 end 为开区间处理；人工区间全部满足 `end < n_frames`，未受影响。",
        "- 人工层 `start/end` 为 0-based 闭区间；生成层 `interval_frames` 沿用 "
        "`end - start`，与原始解析口径一致。窗口构建前需要就此单独定口径。",
        "",
        "## 验收",
        "",
        f"- failures：{failures if failures else 'none'}",
        "",
        "## 产物 SHA256",
        "",
        "| 文件 | SHA256 |",
        "|---|---|",
    ]
    for name in ("le2i.csv", "urfd.csv", "mcfd.csv", "merge_audit.csv"):
        p = V2_ANNOTATIONS / name
        lines.append(f"| `{p.relative_to(paths.PROJECT_ROOT).as_posix()}` | `{paths.sha256_file(p)}` |")
    for p in (V2_MANIFESTS / "videos.csv", V2_MANIFESTS / "events.csv", V2_SPLIT):
        lines.append(f"| `{p.relative_to(paths.PROJECT_ROOT).as_posix()}` | `{paths.sha256_file(p)}` |")

    lines += [
        "",
        "## 口径提示",
        "",
        "- v2 是重新分层洗牌的新切分，不是 v1 的增量；v1/v2 的实验结果不可混比。",
        "- v1 的 annotations、manifests、splits 全部保持字节不变。",
    ]
    QC_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    failures = _check_pinned()
    if failures:
        print("FROZEN INPUT CHECK FAILED:")
        for f in failures:
            print("  -", f)
        return 1

    merge_stats = apply_manual.apply()
    print("manual merge:", merge_stats)

    videos, events, duplicate_groups = build_events.build(annotation_dir=V2_ANNOTATIONS)
    paths.write_csv(V2_MANIFESTS / "videos.csv", videos)
    paths.write_csv(V2_MANIFESTS / "events.csv", events)
    print("build_events:", build_events.stats(videos, events))
    print("duplicate groups:", len(duplicate_groups))

    # Re-read: CSV rows carry strings ("True"/"False"), unlike the in-memory bools.
    videos = paths.read_csv(V2_MANIFESTS / "videos.csv")
    events = paths.read_csv(V2_MANIFESTS / "events.csv")

    ann_failures, ann_info = _validate_annotations(videos)
    failures.extend(ann_failures)
    ev_failures, ev_info = _validate_events(events, ann_info)
    failures.extend(ev_failures)

    payload = make_splits.make(
        version="v2",
        manifests_dir=V2_MANIFESTS,
        split_path=V2_SPLIT,
        created="2026-10-19",
    )
    videos = paths.read_csv(V2_MANIFESTS / "videos.csv")
    events = paths.read_csv(V2_MANIFESTS / "events.csv")
    ann_info["events"] = events
    failures.extend(_validate_split(payload, events, videos))

    print("split counts:", payload["counts"])
    print("leakage checks:", {k: v for k, v in payload["checks"].items() if v} or "none")
    print("failures:", failures or "none")

    if failures:
        print("V2 BUILD FAILED; report not written.")
        return 1

    hashes = {
        rel: paths.sha256_file(paths.PROJECT_ROOT / rel)
        for rel in PINNED
    }
    _write_report(merge_stats, ann_info, ev_info, payload, hashes, failures)
    print("report:", QC_REPORT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
