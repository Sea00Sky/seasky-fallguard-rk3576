"""Annotate fall intervals on the RK3576 board videos (evaluation-only, )."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.annotate import interval_tool  # noqa: E402
from src.data import paths  # noqa: E402

BOARD_OUTPUT = paths.ANNOTATIONS / "board" / "board_fall_intervals.csv"
BOARD_MANIFEST = paths.MANIFESTS / "board" / "videos.csv"


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not BOARD_MANIFEST.exists():
        print(f"missing board manifest: {BOARD_MANIFEST}")
        return 1
    # --scope / --manifest / --output are fixed here: this entry point exists
    # precisely so they cannot be pointed at a training layer.
    passthrough = [
        a for a in argv
        if a not in ("--scope", "board") and not a.startswith(("--scope=", "--manifest", "--output"))
    ]
    fixed = [
        "--scope", "board",
        "--manifest", str(BOARD_MANIFEST),
        "--output", str(BOARD_OUTPUT),
    ]
    return interval_tool.main(fixed + passthrough)


if __name__ == "__main__":
    raise SystemExit(main())
