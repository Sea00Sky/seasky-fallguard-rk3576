"""Build fixed-length skeleton windows from extracted keypoints (v2)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.window import build  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="", help="alternative labels.yaml")
    parser.add_argument("--version", default="v2", choices=["v2", "v3"],
                        help="derived layer to read (annotations/manifests) and write (windows)")
    return parser.parse_args()


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args()
    cfg = build.load_config(Path(args.config) if args.config else None, version=args.version)
    summary = build.build(cfg)
    print(f"version: {summary.get('version', args.version)}")
    print(json.dumps(summary["stats"], ensure_ascii=False, indent=2))
    print(f"arrays: {summary['arrays']}")
    print(f"npz: {cfg.windows_npz}")
    print(f"csv: {cfg.windows_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
