"""Normalize the three annotation sources into datasets/annotations/."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import parse_le2i, parse_mcfd, parse_urfd, paths  # noqa: E402


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    paths.ensure_dirs()

    le2i = parse_le2i.parse()
    paths.write_csv(paths.ANNOTATIONS / "le2i.csv", le2i)
    print("le2i rows:", len(le2i), "status:", dict(__import__("collections").Counter(
        r["status"] for r in le2i)))

    mcfd = parse_mcfd.parse()
    paths.write_csv(paths.ANNOTATIONS / "mcfd.csv", mcfd)
    print("mcfd rows:", len(mcfd), "stats:", parse_mcfd.stats(mcfd))

    urfd = parse_urfd.parse()
    paths.write_csv(paths.ANNOTATIONS / "urfd.csv", urfd)
    print("urfd rows:", len(urfd), "stats:", parse_urfd.stats(urfd))

    print("\nwritten:")
    for name in ("le2i.csv", "mcfd.csv", "urfd.csv"):
        p = paths.ANNOTATIONS / name
        print(f"  {p}  {p.stat().st_size} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
