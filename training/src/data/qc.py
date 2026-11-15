"""Phase 0-1 quality gates and reports."""
from __future__ import annotations

import json
import time
from collections import Counter, defaultdict
from pathlib import Path

from . import paths

EXPECTED = {
    "le2i": {"valid": 96, "invalid_interval": 31, "missing_header": 3, "unannotated": 60},
    # Measured correction ( ): 552 CSV rows = 551 with video +
    # 1 no-video row (chute23/cam55, label 1). Label totals are 288 / 264;
    # among rows with a video they are 288 / 263.
    "mcfd": {
        "csv_rows": 552,
        "valid_rows": 551,
        "label_0": 288,
        "label_1": 263,
        "interval_frames": 30,
        "no_video": 1,
        "chute24_unannotated": 8,
    },
    "urfd": {"needs_annotation": 30, "all_negative": 40, "duplicate_conv": 3},
}


def _dir_sizes(root: Path) -> dict[str, int]:
    return {p.relative_to(root).as_posix(): p.stat().st_size for p in paths.iter_files(root)}


def _compare_dirs(src: Path, dst: Path) -> dict:
    a = _dir_sizes(src)
    b = _dir_sizes(dst) if dst.exists() else {}
    missing = sorted(set(a) - set(b))
    extra = sorted(set(b) - set(a))
    mismatch = sorted(k for k in set(a) & set(b) if a[k] != b[k])
    return {
        "src_files": len(a),
        "dst_files": len(b),
        "src_bytes": sum(a.values()),
        "dst_bytes": sum(b.values()),
        "missing": missing,
        "extra": extra,
        "size_mismatch": mismatch,
    }


def compute_hashes() -> tuple[list[dict], dict]:
    rows: list[dict] = []
    t0 = time.time()
    files = sorted(paths.iter_files(paths.RAW))
    print(f"hashing {len(files)} raw files...", flush=True)
    for i, path in enumerate(files, 1):
        stat = path.stat()
        rows.append(
            {
                "relative_path": path.relative_to(paths.RAW).as_posix(),
                "size_bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "sha256": paths.sha256_file(path),
            }
        )
        if i % 50 == 0 or i == len(files):
            print(f"  {i}/{len(files)}  {time.time() - t0:.1f}s", flush=True)
    paths.write_csv(paths.MANIFESTS / "raw_sha256.csv", rows)

    by_hash = defaultdict(list)
    for row in rows:
        by_hash[row["sha256"]].append(row["relative_path"])
    duplicates = {h: fs for h, fs in by_hash.items() if len(fs) > 1}
    return rows, duplicates


def _md_table(headers: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        out.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(out)


def main() -> int:
    import sys

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    paths.ensure_dirs()
    failures: list[str] = []

    # ---------------- raw integrity ----------------
    raw_checks = {
        "le2i": _compare_dirs(paths.SOURCE_LE2I, paths.RAW_LE2I),
        "mcfd_dataset": _compare_dirs(paths.SOURCE_MCFD / "dataset", paths.RAW_MCFD),
        "urfd": _compare_dirs(paths.SOURCE_URFD, paths.RAW_URFD),
    }
    file_checks = {}
    for name, src, dst in (
        ("mcfd_csv", paths.SOURCE_MCFD / "data_tuple3.csv", paths.RAW_MCFD_CSV),
        ("mcfd_pdf", paths.SOURCE_MCFD / "technicalReport.pdf", paths.RAW_MCFD_PDF),
    ):
        ok = dst.exists() and src.stat().st_size == dst.stat().st_size
        file_checks[name] = {"ok": ok, "src_bytes": src.stat().st_size,
                             "dst_bytes": dst.stat().st_size if dst.exists() else 0}
        if not ok:
            failures.append(f"raw file mismatch: {name}")

    for name, info in raw_checks.items():
        if info["missing"] or info["extra"] or info["size_mismatch"]:
            failures.append(f"raw directory mismatch: {name}")

    hash_rows, duplicates = compute_hashes()

    # ---------------- annotations ----------------
    le2i = paths.read_csv(paths.ANNOTATIONS / "le2i.csv")
    mcfd = paths.read_csv(paths.ANNOTATIONS / "mcfd.csv")
    urfd = paths.read_csv(paths.ANNOTATIONS / "urfd.csv")

    le2i_counts = Counter(r["status"] for r in le2i)
    for key, expected in EXPECTED["le2i"].items():
        if le2i_counts.get(key, 0) != expected:
            failures.append(f"le2i {key}: expected {expected}, got {le2i_counts.get(key, 0)}")

    mcfd_valid = [r for r in mcfd if r["status"] == "valid"]
    mcfd_label = Counter(int(r["label"]) for r in mcfd_valid)
    mcfd_intervals = Counter(int(r["interval_frames"]) for r in mcfd_valid)
    mcfd_status = Counter(r["status"] for r in mcfd)
    mcfd_csv_rows = [r for r in mcfd if r["status"] != "chute24_unannotated"]
    if len(mcfd_csv_rows) != EXPECTED["mcfd"]["csv_rows"]:
        failures.append(f"mcfd csv rows: expected 552, got {len(mcfd_csv_rows)}")
    if len(mcfd_valid) != EXPECTED["mcfd"]["valid_rows"]:
        failures.append(f"mcfd valid rows: expected 551, got {len(mcfd_valid)}")
    if mcfd_label.get(0, 0) != EXPECTED["mcfd"]["label_0"]:
        failures.append(f"mcfd label 0: expected 288, got {mcfd_label.get(0, 0)}")
    if mcfd_label.get(1, 0) != EXPECTED["mcfd"]["label_1"]:
        failures.append(f"mcfd label 1: expected 263, got {mcfd_label.get(1, 0)}")
    if mcfd_status.get("no_video", 0) != EXPECTED["mcfd"]["no_video"]:
        failures.append(f"mcfd no_video: expected 1, got {mcfd_status.get('no_video', 0)}")
    if mcfd_status.get("chute24_unannotated", 0) != EXPECTED["mcfd"]["chute24_unannotated"]:
        failures.append(
            f"mcfd chute24: expected 8, got {mcfd_status.get('chute24_unannotated', 0)}"
        )
    if set(mcfd_intervals) != {EXPECTED["mcfd"]["interval_frames"]}:
        failures.append(f"mcfd interval lengths: {dict(mcfd_intervals)}")

    urfd_status = Counter(r["status"] for r in urfd)
    for key, expected in EXPECTED["urfd"].items():
        if urfd_status.get(key, 0) != expected:
            failures.append(f"urfd {key}: expected {expected}, got {urfd_status.get(key, 0)}")
    urfd_layout = Counter(r["layout"] for r in urfd if r["status"] != "duplicate_conv")
    urfd_unreadable = [r["video_id"] for r in urfd if r["readable"] != "True"]
    if urfd_unreadable:
        failures.append(f"urfd unreadable: {urfd_unreadable}")

    # ---------------- manifests and splits ----------------
    videos = paths.read_csv(paths.MANIFESTS / "videos.csv")
    events = paths.read_csv(paths.MANIFESTS / "events.csv")
    split_payload = json.loads((paths.SPLITS / "v1.json").read_text(encoding="utf-8"))

    unreadable_videos = [v["video_id"] for v in videos if v["readable"] != "True"]
    if unreadable_videos:
        failures.append(f"manifest unreadable videos: {unreadable_videos[:10]}")

    eligible = [e for e in events if e["eligible_for_split"] == "True"]
    unassigned = [e["event_id"] for e in eligible if not e["split"]]
    if unassigned:
        failures.append(f"eligible events without split: {unassigned[:10]}")

    bad_checks = {k: v for k, v in split_payload["checks"].items() if v}
    if bad_checks:
        failures.append(f"split leakage checks: {bad_checks}")

    split_hash = paths.sha256_file(paths.SPLITS / "v1.json")

    # ---------------- reports ----------------
    raw_md = ["# Raw integrity", ""]
    for name, info in raw_checks.items():
        raw_md.append(
            f"- {name}: src_files={info['src_files']} dst_files={info['dst_files']} "
            f"src_bytes={info['src_bytes']} dst_bytes={info['dst_bytes']} "
            f"missing={len(info['missing'])} extra={len(info['extra'])} "
            f"size_mismatch={len(info['size_mismatch'])}"
        )
    for name, info in file_checks.items():
        raw_md.append(f"- {name}: ok={info['ok']} src_bytes={info['src_bytes']} dst_bytes={info['dst_bytes']}")
    raw_md.append(f"- sha256 rows: {len(hash_rows)}")
    raw_md.append(f"- duplicate hashes: {len(duplicates)}")
    for h, files in list(duplicates.items())[:20]:
        raw_md.append(f"  - {h[:16]}... -> {files}")
    raw_md.append("")
    raw_md.append("Gate: " + ("PASS" if not raw_md_has_fail(raw_checks, file_checks) else "FAIL"))
    (paths.QC / "raw_integrity.md").write_text("\n".join(raw_md) + "\n", encoding="utf-8")

    ann_md = ["# Annotation normalization", ""]
    ann_md.append("## Le2i")
    ann_md.append(_md_table(["status", "count"], sorted(le2i_counts.items())))
    ann_md.append("")
    for key in ("invalid_interval", "missing_header", "unannotated", "missing_video",
                "unreadable_video", "end_out_of_range"):
        items = [r["video_id"] for r in le2i if r["status"] == key]
        if items:
            ann_md.append(f"- {key}: {len(items)} -> {items[:10]}")
    ann_md.append("")
    ann_md.append("## MCFD")
    ann_md.append(_md_table(["status", "count"], sorted(mcfd_status.items())))
    ann_md.append("")
    ann_md.append(f"- valid rows: {len(mcfd_valid)}; label counts: {dict(mcfd_label)}; "
                  f"interval lengths: {dict(mcfd_intervals)}")
    no_video = [f"chute{r['chute']}/cam{r['cam']}" for r in mcfd if r["status"] == "no_video"]
    ann_md.append(f"- annotation rows without video: {no_video}")
    ann_md.append("")
    ann_md.append("## URFD")
    ann_md.append(_md_table(["status", "count"], sorted(urfd_status.items())))
    ann_md.append("")
    ann_md.append(_md_table(["layout", "count"], sorted(urfd_layout.items())))
    ann_md.append("")
    unknown_layout = [r["video_id"] for r in urfd
                      if r["status"] != "duplicate_conv" and not r["use_half"]]
    ann_md.append(f"- videos without a usable RGB half: {len(unknown_layout)} -> {unknown_layout}")
    ann_md.append("")
    ann_md.append("Gate: " + ("PASS" if not any(
        f.startswith(("le2i ", "mcfd ", "urfd ")) for f in failures) else "FAIL"))
    (paths.QC / "annotations.md").write_text("\n".join(ann_md) + "\n", encoding="utf-8")

    split_md = ["# Event-level splits (v1)", ""]
    split_md.append(f"- seed: {split_payload['seed']}; ratios: {split_payload['ratios']}")
    split_md.append(f"- sha256(v1.json): `{split_hash}`")
    split_md.append("")
    split_md.append("## Counts")
    split_md.append(_md_table(
        ["group", "train", "val", "test"],
        [[k, v.get("train", 0), v.get("val", 0), v.get("test", 0)]
         for k, v in split_payload["counts"]["by_dataset_label"].items()],
    ))
    split_md.append("")
    split_md.append(f"- events: {split_payload['counts']['events']}")
    split_md.append(f"- fall events: {split_payload['counts']['fall_events']}")
    split_md.append("")
    split_md.append("## Leakage checks")
    for key, value in split_payload["checks"].items():
        split_md.append(f"- {key}: {'PASS' if not value else 'FAIL ' + str(value)}")
    split_md.append("")
    split_md.append("Gate: " + ("PASS" if not bad_checks and not unassigned else "FAIL"))
    (paths.QC / "splits.md").write_text("\n".join(split_md) + "\n", encoding="utf-8")

    summary = [
        "# Phase 0-1 data foundation",
        "",
        f"- raw files hashed: {len(hash_rows)}",
        f"- duplicate raw hashes: {len(duplicates)}",
        f"- videos in manifest: {len(videos)}; unreadable: {len(unreadable_videos)}",
        f"- events: {len(events)}; eligible: {len(eligible)}",
        f"- split counts: {split_payload['counts']['events']}",
        f"- fall split counts: {split_payload['counts']['fall_events']}",
        f"- split file sha256: `{split_hash}`",
        "",
        "## Gates",
    ]
    if failures:
        summary += [f"- FAIL: {f}" for f in failures]
    else:
        summary.append("- G0 raw integrity: PASS")
        summary.append("- G1 annotation normalization: PASS")
        summary.append("- G2 events/manifest/splits: PASS")
    (paths.REPORTS / "phase0_1_foundation.md").write_text("\n".join(summary) + "\n", encoding="utf-8")

    print("\n== Gate result ==")
    if failures:
        for f in failures:
            print("FAIL:", f)
        print(f"{len(failures)} failure(s)")
        return 1
    print("0 failures")
    return 0


def raw_md_has_fail(raw_checks: dict, file_checks: dict) -> bool:
    if any(info["missing"] or info["extra"] or info["size_mismatch"] for info in raw_checks.values()):
        return True
    return not all(info["ok"] for info in file_checks.values())
