"""Propagate hand-marked MCFD fall intervals to the other cameras of a chute."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.annotate import mcfd_segments as ms
from src.data import paths
from src.data.paths import read_csv


def _summary(rows: list[dict], metrics: dict) -> str:
    return (
        f"rows={metrics['rows']} videos={metrics['videos']} "
        f"annotated={metrics['annotated_rows']} propagated={metrics['propagated_rows']} "
        f"source_cameras={len(metrics['source_cameras'])}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fill the other seven MCFD cameras from the hand-marked source camera."
    )
    parser.add_argument("--input", type=Path, default=paths.MCFD_FALL_INTERVALS)
    parser.add_argument("--output", type=Path, default=paths.MCFD_FALL_INTERVALS)
    parser.add_argument("--chute", type=int, action="append", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--no-validate", action="store_true")
    args = parser.parse_args(argv)

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    all_rows = ms.load_mcfd_rows()
    segments = ms.merged_segments(all_rows)
    meta = ms.load_video_meta()
    existing = read_csv(args.input) if args.input.exists() else []

    if args.validate:
        stats = ms.validate_manual(existing, segments, meta, require_complete=True)
        print("valid:", _summary(existing, stats))
        return 0

    annotated = [r for r in existing if (r.get("status") or "").strip() == ms.ANNOTATED]
    if not annotated:
        print(f"没有 annotated 行：{args.input}")
        return 1
    if args.chute:
        wanted = set(args.chute)
        annotated = [r for r in annotated if ms.to_int(r.get("chute")) in wanted]
        if not annotated:
            print(f"指定 chute 没有 annotated 行：{sorted(wanted)}")
            return 1

    propagated = ms.propagate_rows(annotated, segments, meta)
    merged = ms.merge_rows(existing, propagated, chutes=set(args.chute) if args.chute else None)

    if not args.no_validate:
        stats = ms.validate_manual(merged, segments, meta, require_complete=False)
    else:
        stats = {}

    if args.dry_run:
        print(f"dry-run: would write {len(propagated)} propagated rows to {args.output}")
        for row in propagated:
            print(
                f"  {row['video_id']:<22} seg{row['segment_index']} "
                f"anchor={row['anchor_start_frame']:5d} offset={row['offset_frames']:+5d} "
                f"-> {row['start_frame']:5d}..{row['end_frame']:5d} "
                f"({row['end_frame'] - row['start_frame'] + 1:3d}f)"
            )
        if stats:
            print("merged:", _summary(merged, stats))
        return 0

    ms.write_manual_atomic(args.output, merged)
    print(f"written {args.output}: {len(propagated)} propagated rows")
    if stats:
        print("merged:", _summary(merged, stats))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
