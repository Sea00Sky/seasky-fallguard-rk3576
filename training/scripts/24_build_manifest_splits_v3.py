"""Build manifest v3 and split v3 from the v3 annotation layer."""
from __future__ import annotations

import json
import os
import statistics
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import build_events, make_splits, paths  # noqa: E402

V3_ANNOTATIONS = paths.ANNOTATIONS_V3
V3_MANIFESTS = paths.MANIFESTS_V3
V3_SPLIT = paths.SPLITS_V3
QC_REPORT = paths.REPORTS / "split_v3_qc.md"

# Everything v3 reads must be frozen before the build starts.
PINNED = {
    "datasets/annotations/v2/le2i.csv":
        "4aaf74e68289fa54a44cdce0029979141745c7d293335fe71aa937cc6484f45f",
    "datasets/annotations/v2/mcfd.csv":
        "2e1fc6d6cf7002c69a1d2d4211f16107f56fec283caecbbc53cc0d76b0143a16",
    "datasets/annotations/v2/urfd.csv":
        "0d30ae04d2144cebec4fb74b81321d53785a23761ae4cc01591327e047f7c44b",
    "datasets/annotations/manual/mcfd_fall_intervals.csv":
        "eb047ef522ad24795e62d1a836738ce1455b096c7e5ff4ecbdf51f025aa2880c",
    "datasets/manifests/v2/videos.csv":
        "d81a4ee43facfe8b23f8044b3c117c1d16512d4f211384bf41370ca579e9b087",
    "datasets/manifests/v2/events.csv":
        "3867188ac0dca4363d99be95ebfd451bc02dc872f3783631974f389424080235",
    "datasets/splits/v2.json":
        "adaf2f45fbe2bc388542a702c787b4760d732e97b5ebdda2ec9320413c6bc2ca",
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

MCFD_STATUS_V3 = Counter({
    "valid": 288,
    "manual_annotated": 200,
    "chute24_unannotated": 8,
    "no_video": 1,
})


def _i(value) -> int:
    return int(str(value).strip())


def _check_pinned() -> list[str]:
    failures = []
    for rel, expected in PINNED.items():
        actual = paths.sha256_file(paths.PROJECT_ROOT / rel)
        if actual != expected:
            failures.append(f"{rel}: sha256 {actual} != pinned {expected}")
    return failures


def _validate_annotations() -> tuple[list[str], dict]:
    failures: list[str] = []
    mcfd = paths.read_csv(V3_ANNOTATIONS / "mcfd.csv")
    counts = Counter(r["status"] for r in mcfd)
    if counts != MCFD_STATUS_V3:
        failures.append(f"mcfd status counts: {dict(counts)}")

    for row in mcfd:
        if row["status"] not in ("valid", "manual_annotated"):
            continue
        n = _i(row["n_frames"])
        s, e = _i(row["start_frame"]), _i(row["end_frame"])
        if not (0 <= s <= e < n):
            failures.append(f"chute{row['chute']}/cam{row['cam']}: [{s}, {e}] outside [0, {n - 1}]")
        if _i(row["interval_frames"]) != e - s:
            failures.append(f"chute{row['chute']}/cam{row['cam']}: interval_frames")

    # v3 is single-variable: Le2i/URFD must be byte-identical to v2.
    for name in ("le2i.csv", "urfd.csv"):
        if paths.sha256_file(V3_ANNOTATIONS / name) != paths.sha256_file(paths.ANNOTATIONS_V2 / name):
            failures.append(f"{name} differs from v2")

    return failures, {"mcfd": mcfd}


def _validate_events(events: list[dict], mcfd_rows: list[dict]) -> tuple[list[str], dict]:
    failures: list[str] = []
    eligible = [e for e in events if e["eligible_for_split"] == "True"]
    layer = Counter((e["dataset"], e["label"]) for e in eligible)
    if dict(layer) != {k: v for k, v in EXPECTED_LAYERS.items()}:
        failures.append(f"eligible layers: {dict(layer)}")

    by_event: dict[str, list[dict]] = {}
    for row in mcfd_rows:
        by_event.setdefault(f"mcfd:chute{_i(row['chute']):02d}", []).append(row)

    for ev in eligible:
        if ev["dataset"] != "mcfd":
            continue
        rows = [r for r in by_event.get(ev["event_id"], []) if r["status"] == "manual_annotated"]
        if ev["label"] == "fall" and not rows:
            failures.append(f"{ev['event_id']}: fall event without a manual_annotated row")
        if ev["label"] == "fall" and not any(_i(r["label"]) == 1 for r in rows):
            failures.append(f"{ev['event_id']}: fall event without a label-1 row")

    dups = [e for e in events if "duplicate_content" in (e.get("notes") or "")]
    if len(dups) != 3:
        failures.append(f"duplicate_content events: {len(dups)}")

    return failures, {
        "eligible": len(eligible),
        "fall": sum(1 for e in eligible if e["label"] == "fall"),
        "non_fall": sum(1 for e in eligible if e["label"] == "non_fall"),
        "dups": dups,
    }


def _validate_split(payload: dict, v2_payload: dict, events: list[dict]) -> list[str]:
    failures: list[str] = []
    bad = {k: v for k, v in payload["checks"].items() if v}
    if bad:
        failures.append(f"leakage checks: {bad}")
    if payload["counts"]["events"] != EXPECTED_SPLIT:
        failures.append(f"split events: {payload['counts']['events']}")
    if payload["counts"]["fall_events"] != EXPECTED_FALL_SPLIT:
        failures.append(f"split fall: {payload['counts']['fall_events']}")

    # The whole point of v3: the split must be identical to v2.
    if payload["event_to_split"] != v2_payload["event_to_split"]:
        diff = [
            k for k in set(payload["event_to_split"]) | set(v2_payload["event_to_split"])
            if payload["event_to_split"].get(k) != v2_payload["event_to_split"].get(k)
        ]
        failures.append(f"v3 split differs from v2 on {len(diff)} events: {diff[:5]}")

    for ev in events:
        if ev["eligible_for_split"] == "True" and ev["event_id"] not in payload["event_to_split"]:
            failures.append(f"{ev['event_id']}: eligible but not in split")
    return failures


def _interval_stats(mcfd_rows: list[dict]) -> dict:
    spans = [
        (_i(r["start_frame"]), _i(r["end_frame"]))
        for r in mcfd_rows if r["status"] == "manual_annotated"
    ]
    frames = [e - s + 1 for s, e in spans]
    secs = [f / 120.0 for f in frames]
    return {
        "n": len(spans),
        "frames": (min(frames), statistics.median(frames), max(frames)),
        "seconds": (min(secs), statistics.median(secs), max(secs)),
    }


def _write_report(ev_info, payload, hashes, failures, iv) -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "# Split v3 QC",
        "",
        f"- 生成时间：{now}",
        f"- 切分文件：`{V3_SPLIT.relative_to(paths.PROJECT_ROOT).as_posix()}`",
        f"- seed：{payload['seed']}；比例：{payload['ratios']}；单位：event_id",
        "- v3 与 v2 的唯一差别：MCFD 跌倒区间来自人工层；切分与 v2 逐事件一致（已断言）。",
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
        "## 可切分事件",
        "",
        f"- v3：fall {ev_info['fall']} / non_fall {ev_info['non_fall']} / 合计 {ev_info['eligible']}",
        "",
        "## 切分计数",
        "",
        "| split | 全部事件 | fall | non_fall |",
        "|---|---|---|---|",
    ]
    for split in ("train", "val", "test"):
        lines.append(
            f"| {split} | {payload['counts']['events'].get(split, 0)} | "
            f"{payload['counts']['fall_events'].get(split, 0)} | "
            f"{EXPECTED_NON_FALL_SPLIT.get(split, 0)} |"
        )
    lines += [
        "",
        "## MCFD 人工区间（0-based 闭区间）",
        "",
        f"- 条数：{iv['n']}",
        f"- 帧 min/median/max：{iv['frames'][0]} / {iv['frames'][1]:.0f} / {iv['frames'][2]}",
        f"- 秒 min/median/max：{iv['seconds'][0]:.2f} / {iv['seconds'][1]:.2f} / {iv['seconds'][2]:.2f}",
        "",
        "## 泄漏检查",
        "",
        f"- {payload['checks']}",
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
    for p in (V3_ANNOTATIONS / "mcfd.csv", V3_MANIFESTS / "videos.csv",
              V3_MANIFESTS / "events.csv", V3_SPLIT):
        lines.append(f"| `{p.relative_to(paths.PROJECT_ROOT).as_posix()}` | `{paths.sha256_file(p)}` |")
    lines.append("")
    QC_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    failures = _check_pinned()
    if failures:
        print("FROZEN INPUT CHECK FAILED:")
        for f in failures:
            print("  -", f)
        return 1

    ann_failures, ann_info = _validate_annotations()
    failures.extend(ann_failures)

    videos, events, duplicate_groups = build_events.build(annotation_dir=V3_ANNOTATIONS)
    paths.write_csv(V3_MANIFESTS / "videos.csv", videos)
    paths.write_csv(V3_MANIFESTS / "events.csv", events)
    print("build_events:", build_events.stats(videos, events))
    print("duplicate groups:", len(duplicate_groups))

    videos = paths.read_csv(V3_MANIFESTS / "videos.csv")
    events = paths.read_csv(V3_MANIFESTS / "events.csv")

    ev_failures, ev_info = _validate_events(events, ann_info["mcfd"])
    failures.extend(ev_failures)

    # Reuse the v2 split instead of re-shuffling: same seed and same event set
    # would give the same assignment, but asserting it is stronger than trusting it.
    v2_payload = json.loads(paths.SPLITS_V2.read_text(encoding="utf-8"))

    payload = make_splits.make(
        version="v3",
        manifests_dir=V3_MANIFESTS,
        split_path=V3_SPLIT,
        created="2026-10-21",
    )
    events = paths.read_csv(V3_MANIFESTS / "events.csv")
    failures.extend(_validate_split(payload, v2_payload, events))

    print("split counts:", payload["counts"])
    print("leakage checks:", {k: v for k, v in payload["checks"].items() if v} or "none")
    print("failures:", failures or "none")

    if failures:
        print("V3 BUILD FAILED; report not written.")
        return 1

    hashes = {rel: paths.sha256_file(paths.PROJECT_ROOT / rel) for rel in PINNED}
    _write_report(ev_info, payload, hashes, failures, _interval_stats(ann_info["mcfd"]))
    print("report:", QC_REPORT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
