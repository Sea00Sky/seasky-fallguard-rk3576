"""Export windows to a PYSKL PoseDataset pkl and verify the round trip."""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import paths  # noqa: E402
from src.window import build, to_pyskl  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="", help="alternative labels.yaml")
    parser.add_argument("--version", default="v2", choices=["v2", "v3"],
                        help="windows layer to export and pyskl layer to write")
    return parser.parse_args()


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args()
    cfg = build.load_config(Path(args.config) if args.config else None, version=args.version)
    out, summary = to_pyskl.export(cfg)

    with open(out, "rb") as fh:
        payload = pickle.load(fh)
    annotations = payload["annotations"]
    failures: list[str] = []
    if len(annotations) != summary["windows"]:
        failures.append("annotation count mismatch")
    ids = [a["frame_dir"] for a in annotations]
    if len(set(ids)) != len(ids):
        failures.append("duplicate frame_dir")
    for split in to_pyskl.SPLITS:
        if split not in payload["split"]:
            failures.append(f"missing split {split}")
        if len(payload["split"].get(split, [])) != summary["splits"].get(split, -1):
            failures.append(f"split size mismatch: {split}")
    for a in annotations[:50]:
        kp = a["keypoint"]
        sc = a["keypoint_score"]
        if kp.shape != (1, cfg.resample_frames, 17, 2):
            failures.append(f"bad keypoint shape at {a['frame_dir']}: {kp.shape}")
            break
        if sc.shape != (1, cfg.resample_frames, 17):
            failures.append(f"bad score shape at {a['frame_dir']}: {sc.shape}")
            break
        if not np.isfinite(kp).all() or not np.isfinite(sc).all():
            failures.append(f"non-finite values at {a['frame_dir']}")
            break
        if a["label"] not in (0, 1):
            failures.append(f"bad label at {a['frame_dir']}")
            break

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"pkl: {out}")
    print(f"pkl size: {out.stat().st_size / (1024 * 1024):.1f} MB")
    print("failures:", failures or "none")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
