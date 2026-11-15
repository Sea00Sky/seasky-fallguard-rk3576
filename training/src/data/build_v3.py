"""Build annotation layer v3: v2 with MCFD fall intervals replaced by the manual layer."""
from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import paths  # noqa: E402

MCFD_FALL_STATUS = "manual_annotated"
MCFD_FIELDS = [
    "dataset", "chute", "cam", "start_frame", "end_frame", "interval_frames",
    "label", "status", "fps", "n_frames", "video_path", "notes",
]
AUDIT_FIELDS = [
    "video_id", "chute", "cam", "row_kind", "action",
    "start_before", "end_before", "start_after", "end_after", "notes",
]

EXPECTED = {
    "mcfd_rows": 560,
    "label0_rows": 288,
    "fall_rows": 200,
    "trimmed_label0": 48,
    "emptied_label0": 0,
    "fall_videos": 184,
    "source_videos": 184,
    "chute24_rows": 8,
    "no_video_rows": 1,
}


def _i(value) -> int:
    return int(str(value).strip())


def _vid(chute: int, cam: int) -> str:
    return f"mcfd:chute{chute:02d}:cam{cam}"


def _subtract(span: tuple[int, int], cuts: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Remove half-open `cuts` from `span`, returning the surviving pieces."""
    pieces = [span]
    for cs, ce in cuts:
        nxt: list[tuple[int, int]] = []
        for a, b in pieces:
            if ce <= a or cs >= b:
                nxt.append((a, b))
                continue
            if a < cs:
                nxt.append((a, cs))
            if ce < b:
                nxt.append((ce, b))
        pieces = nxt
    return [(a, b) for a, b in pieces if b > a]


def load_manual(manual_path: Path) -> dict[str, list[tuple[int, int]]]:
    rows = paths.read_csv(manual_path)
    if not rows:
        raise ValueError(f"manual layer is empty: {manual_path}")
    by_video: dict[str, list[tuple[int, int]]] = defaultdict(list)
    seen: set[tuple[int, int, int]] = set()
    for row in rows:
        chute, cam = _i(row["chute"]), _i(row["cam"])
        seg = _i(row["segment_index"])
        key = (chute, cam, seg)
        if key in seen:
            raise ValueError(f"duplicate manual key {key}")
        seen.add(key)
        if row["dataset"] != "mcfd":
            raise ValueError(f"{row['video_id']}: unexpected dataset {row['dataset']!r}")
        if _i(row["label"]) != 1:
            raise ValueError(f"{row['video_id']}: manual fall row with label {row['label']!r}")
        if row["status"] not in ("annotated", "propagated"):
            raise ValueError(f"{row['video_id']}: unexpected status {row['status']!r}")
        start, end = _i(row["start_frame"]), _i(row["end_frame"])
        n_frames = _i(row["n_frames"])
        if not (0 <= start <= end < n_frames):
            raise ValueError(f"{row['video_id']} seg{seg}: [{start}, {end}] outside [0, {n_frames - 1}]")
        by_video[_vid(chute, cam)].append((start, end))
    for vid, spans in by_video.items():
        spans.sort()
        for (s0, e0), (s1, _e1) in zip(spans, spans[1:]):
            if s1 <= e0:
                raise ValueError(f"{vid}: overlapping manual intervals {s0}-{e0} and {s1}-...")
    return dict(by_video)


def build_mcfd(
    v2_rows: list[dict], manual: dict[str, list[tuple[int, int]]]
) -> tuple[list[dict], list[dict], dict]:
    """Return (v3 mcfd rows, audit rows, stats)."""
    out: list[dict] = []
    audit: list[dict] = []
    stats = Counter()

    fall_videos_v2 = {
        _vid(_i(r["chute"]), _i(r["cam"]))
        for r in v2_rows
        if r["status"] == "valid" and _i(r["label"]) == 1
    }
    if fall_videos_v2 != set(manual):
        only_v2 = sorted(fall_videos_v2 - set(manual))[:5]
        only_manual = sorted(set(manual) - fall_videos_v2)[:5]
        raise ValueError(
            f"manual/v2 video sets differ: only_v2={only_v2} only_manual={only_manual}"
        )
    stats["source_videos"] = len(manual)

    for row in v2_rows:
        status = row["status"]
        chute, cam = _i(row["chute"]), _i(row["cam"])
        vid = _vid(chute, cam)
        base = {k: row.get(k, "") for k in MCFD_FIELDS}

        if status != "valid":
            out.append(base)
            stats[f"status_{status}"] += 1
            continue

        if _i(row["label"]) == 1:
            # Dropped here; the manual rows below replace this video's fall rows.
            stats["label1_dropped"] += 1
            audit.append({
                "video_id": vid, "chute": chute, "cam": cam, "row_kind": "label1",
                "action": "replaced_by_manual",
                "start_before": row["start_frame"], "end_before": row["end_frame"],
                "start_after": "", "end_after": "",
                "notes": "raw 1-based inclusive row removed",
            })
            continue

        # label 0: trim anything that overlaps a new fall interval.
        start, end = _i(row["start_frame"]), _i(row["end_frame"])
        canon = (start - 1, end)
        cuts = [(s, e + 1) for s, e in manual.get(vid, [])]
        pieces = _subtract(canon, cuts)
        stats["label0_rows"] += 1
        if pieces == [canon]:
            out.append(base)
            continue
        if not pieces:
            stats["emptied_label0"] += 1
            raise ValueError(f"{vid}: label-0 row {start}-{end} is fully covered by a fall interval")
        stats["trimmed_label0"] += 1
        stats["trim_pieces"] += len(pieces)
        for a, b in pieces:
            new = dict(base)
            # Back to the v2 row convention: 1-based inclusive [a + 1, b] with
            # interval_frames = end - start (gap count, canonical length - 1).
            new["start_frame"] = str(a + 1)
            new["end_frame"] = str(b)
            new["interval_frames"] = str(b - a - 1)
            new["notes"] = (row.get("notes") or "").strip()
            new["notes"] = (
                f"{new['notes']} | trimmed by v3 fall interval from {start}-{end}"
                if new["notes"] else f"trimmed by v3 fall interval from {start}-{end}"
            )
            out.append(new)
        audit.append({
            "video_id": vid, "chute": chute, "cam": cam, "row_kind": "label0",
            "action": "trimmed", "start_before": row["start_frame"],
            "end_before": row["end_frame"],
            "start_after": ";".join(f"{a + 1}" for a, _b in pieces),
            "end_after": ";".join(str(b) for _a, b in pieces),
            "notes": f"{len(pieces)} piece(s)",
        })

    # Emit one row per manual interval.
    for vid in sorted(manual):
        chute_s, cam_s = vid.split(":")[1][5:], vid.split(":")[2][3:]
        chute, cam = int(chute_s), int(cam_s)
        template = next(
            (r for r in v2_rows
             if r["status"] == "valid" and _i(r["chute"]) == chute and _i(r["cam"]) == cam),
            None,
        )
        if template is None:
            raise ValueError(f"{vid}: no v2 row to copy metadata from")
        for start, end in sorted(manual[vid]):
            n_frames = _i(template["n_frames"])
            if not (0 <= start <= end < n_frames):
                raise ValueError(f"{vid}: manual interval [{start}, {end}] outside [0, {n_frames - 1}]")
            out.append({
                "dataset": "mcfd",
                "chute": chute,
                "cam": cam,
                "start_frame": str(start),      # 0-based inclusive (manual convention)
                "end_frame": str(end),
                "interval_frames": str(end - start),
                "label": "1",
                "status": MCFD_FALL_STATUS,
                "fps": template["fps"],
                "n_frames": template["n_frames"],
                "video_path": template["video_path"],
                "notes": "v3 manual fall interval (0-based inclusive)",
            })
            stats["fall_rows"] += 1

    out.sort(key=lambda r: (
        _i(r["chute"]), _i(r["cam"]),
        _i(r["start_frame"]) if str(r["start_frame"]).strip() else 10**9,
    ))
    return out, audit, dict(stats)


def validate_v3(
    rows: list[dict], videos: list[dict], manual: dict, stats: dict
) -> list[str]:
    failures: list[str] = []
    stats_pieces = int(stats.get("trim_pieces", 0))
    frames = {v["video_id"]: _i(v["n_frames"]) for v in videos}
    statuses = Counter(r["status"] for r in rows)

    if len(rows) != EXPECTED["mcfd_rows"] + EXPECTED["fall_rows"] - 263:
        failures.append(f"mcfd rows {len(rows)} != expected 497")
    expected_valid = EXPECTED["label0_rows"] - EXPECTED["trimmed_label0"] + stats_pieces
    if statuses.get("valid", 0) != expected_valid:
        failures.append(
            f"valid rows {statuses.get('valid')} != {expected_valid} "
            f"(288 label-0 - 48 trimmed + {stats_pieces} pieces)"
        )
    if statuses.get(MCFD_FALL_STATUS, 0) != EXPECTED["fall_rows"]:
        failures.append(f"{MCFD_FALL_STATUS} rows {statuses.get(MCFD_FALL_STATUS)} != 200")

    for row in rows:
        if row["status"] not in ("valid", MCFD_FALL_STATUS):
            continue        # chute24_unannotated / no_video carry no interval
        vid = _vid(_i(row["chute"]), _i(row["cam"]))
        n = frames.get(vid)
        if n is None:
            failures.append(f"{vid}: not in manifest")
            continue
        if row["status"] == MCFD_FALL_STATUS:
            s, e = _i(row["start_frame"]), _i(row["end_frame"])
            if not (0 <= s <= e < n):
                failures.append(f"{vid}: manual [{s}, {e}] outside [0, {n - 1}]")
            if _i(row["interval_frames"]) != e - s:
                failures.append(f"{vid}: interval_frames {row['interval_frames']} != {e - s}")
            if _i(row["n_frames"]) != n:
                failures.append(f"{vid}: n_frames {row['n_frames']} != manifest {n}")
        elif row["status"] == "valid":
            s, e = _i(row["start_frame"]), _i(row["end_frame"])
            if not (0 <= s <= e < n):
                failures.append(f"{vid}: valid [{s}, {e}] outside [0, {n - 1}]")
            if _i(row["interval_frames"]) != e - s:
                failures.append(f"{vid}: interval_frames {row['interval_frames']} != {e - s}")

    # No label-0 row may overlap a fall interval.
    by_video_fall: dict[str, list[tuple[int, int]]] = defaultdict(list)
    by_video_ctx: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for row in rows:
        if row["status"] not in ("valid", MCFD_FALL_STATUS):
            continue
        vid = _vid(_i(row["chute"]), _i(row["cam"]))
        s, e = _i(row["start_frame"]), _i(row["end_frame"])
        if row["status"] == MCFD_FALL_STATUS:
            by_video_fall[vid].append((s, e + 1))
        elif _i(row["label"]) == 0:
            by_video_ctx[vid].append((s - 1, e))
    for vid, ctxs in by_video_ctx.items():
        for a, b in ctxs:
            for cs, ce in by_video_fall.get(vid, []):
                if max(0, min(b, ce) - max(a, cs)) > 0:
                    failures.append(f"{vid}: context [{a}, {b}) overlaps fall [{cs}, {ce})")

    # Every fall video must keep the same number of intervals as the manual layer.
    fall_counts = Counter(
        _vid(_i(r["chute"]), _i(r["cam"]))
        for r in rows if r["status"] == MCFD_FALL_STATUS
    )
    for vid, spans in manual.items():
        if fall_counts.get(vid, 0) != len(spans):
            failures.append(f"{vid}: {fall_counts.get(vid, 0)} v3 fall rows != {len(spans)} manual intervals")
    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build annotation layer v3.")
    parser.add_argument("--source-dir", type=Path, default=paths.ANNOTATIONS_V2)
    parser.add_argument("--out-dir", type=Path, default=paths.ANNOTATIONS_V3)
    parser.add_argument("--manual", type=Path, default=paths.MCFD_FALL_INTERVALS)
    parser.add_argument("--manifest-dir", type=Path, default=paths.MANIFESTS_V2)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    src = Path(args.source_dir)
    out_dir = Path(args.out_dir)
    v2_mcfd = paths.read_csv(src / "mcfd.csv")
    videos = paths.read_csv(Path(args.manifest_dir) / "videos.csv")
    manual = load_manual(Path(args.manual))

    rows, audit, stats = build_mcfd(v2_mcfd, manual)
    failures = validate_v3(rows, videos, manual, stats)
    if failures:
        print("V3 BUILD FAILED:")
        for f in failures[:20]:
            print("  -", f)
        return 1

    print(f"mcfd rows: {len(rows)} (v2 {len(v2_mcfd)})")
    print(f"stats: {stats}")
    print(f"audit rows: {len(audit)}")
    if args.dry_run:
        print("dry-run: nothing written")
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    paths.write_csv(out_dir / "mcfd.csv", rows, fieldnames=MCFD_FIELDS)
    paths.write_csv(out_dir / "mcfd_merge_audit.csv", audit, fieldnames=AUDIT_FIELDS)
    for name in ("le2i.csv", "urfd.csv"):
        paths.write_csv(out_dir / name, paths.read_csv(src / name))

    print(f"written: {out_dir}")
    print(f"  le2i.csv / urfd.csv copied from {src}")
    print(f"  mcfd.csv sha256 {paths.sha256_file(out_dir / 'mcfd.csv')}")
    print(f"  generated {datetime.now().isoformat(timespec='seconds')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
