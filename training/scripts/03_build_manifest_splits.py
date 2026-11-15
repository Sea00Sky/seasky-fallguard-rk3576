"""Build videos.csv, events.csv and the event-level split file."""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import build_events, make_splits, paths  # noqa: E402


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    paths.ensure_dirs()

    videos, events, duplicate_groups = build_events.build()
    paths.write_csv(paths.MANIFESTS / "videos.csv", videos)
    paths.write_csv(paths.MANIFESTS / "events.csv", events)
    print("build_events:", json.dumps(build_events.stats(videos, events), ensure_ascii=False, indent=2))
    print("duplicate_content groups:", len(duplicate_groups))
    for group in duplicate_groups:
        print(
            f"  {group['sha256'][:16]}... keep={group['canonical_event']} "
            f"exclude={group['excluded_events']} files={group['files']}"
        )

    payload = make_splits.make()
    print("\nsplits v1 counts:")
    print(json.dumps(payload["counts"], ensure_ascii=False, indent=2))
    bad = {k: v for k, v in payload["checks"].items() if v}
    print("leakage checks:", bad if bad else "none")

    print("\nwritten:")
    for p in (
        paths.MANIFESTS / "videos.csv",
        paths.MANIFESTS / "events.csv",
        paths.SPLITS / "v1.json",
    ):
        print(f"  {p}  {p.stat().st_size} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
