"""Train ST-GCN++ on a window layer and evaluate on the held-out test split."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.train import engine  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="", help="alternative train yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", default=None, help="cuda / cpu")
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument(
        "--class-weights", choices=["balanced"], default=None,
        help="use inverse-frequency class weights (default: config value)",
    )
    parser.add_argument("--no-report", action="store_true",
                        help="skip writing the markdown report (smoke runs)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = engine.load_config(Path(args.config) if args.config else None)
    results = engine.run(cfg, args)
    print(f"saved to {engine.resolve(cfg['output']['run_dir'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
