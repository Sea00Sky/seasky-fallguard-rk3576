"""Build a denser YOLO calibration set: several frames per training video."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def _load_own(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, PROJECT_ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# Importing `20_convert_rknn` executes only its module top level (constants and
# function definitions), not a conversion -- the same trick `scripts/25` uses on
# `scripts/21`. `_resize_chw_to` is reused verbatim so the resampling is
# identical to the shipped calibration set.
conv20 = _load_own("convert20", "scripts/20_convert_rknn.py")
paths = conv20.paths
spec_mod = conv20.spec_mod


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-video", type=int, default=4,
                    help="frames sampled from each video, spread evenly")
    ap.add_argument("--total", type=int, default=400)
    ap.add_argument("--out", default="yolo_frames_dense")
    args = ap.parse_args()

    import cv2

    from src.deploy import yolo_export

    rows = [r for r in paths.read_csv(paths.SOURCE_MANIFEST_V2)
            if r.get("readable") == "True"]
    if not rows:
        raise SystemExit(f"no readable rows in {paths.SOURCE_MANIFEST_V2}")

    want_h, want_w = yolo_export.INPUT_SHAPE[2], yolo_export.INPUT_SHAPE[3]
    out_dir = paths.CALIB / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.npy"):
        old.unlink()

    by_ds: dict[str, list[dict]] = {}
    for r in rows:
        by_ds.setdefault(r.get("dataset", "?"), []).append(r)
    keys = sorted(by_ds)

    frames: list[np.ndarray] = []
    native = resized = 0
    skipped = 0
    i = 0
    while len(frames) < args.total:
        k = keys[i % len(keys)]
        i += 1
        if not by_ds[k]:
            if not any(by_ds[x] for x in keys):
                break
            continue
        row = by_ds[k].pop(0)
        path = paths.SOURCE_RAW_V2 / row["relative_path"]
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            print(f"  skip {row['video_id']}: cannot open {path.name}")
            skipped += 1
            continue
        total = int(row["n_frames"])
        # Spread across the clip. `total // 2` alone (the shipped rule) never
        # sees the opening or closing frames.
        picks = np.unique(
            np.rint(np.linspace(0, max(0, total - 1), args.per_video))
            .astype(int))
        for f in picks:
            if len(frames) >= args.total:
                break
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
            ok, frame = cap.read()
            if not ok:
                continue
            chw, _ = yolo_export.preprocess(frame, imgsz=spec_mod.IMGSZ)
            if chw.shape[2:] != (want_h, want_w):
                chw = conv20._resize_chw_to(chw, want_h, want_w)
                resized += 1
            else:
                native += 1
            frames.append(chw)
        cap.release()

    if not frames:
        raise SystemExit("dense calibration set came out empty")

    for j, arr in enumerate(frames):
        np.save(out_dir / f"{j:04d}.npy", arr[0])
    lst = paths.CALIB / "yolo_calib_dense.txt"
    lst.write_text("\n".join(str(out_dir / f"{j:04d}.npy")
                             for j in range(len(frames))) + "\n",
                   encoding="utf-8")

    print(f"wrote {len(frames)} frames to {out_dir}")
    print(f"  {native} native {want_h}x{want_w}, {resized} resampled, "
          f"{skipped} video(s) unreadable")
    print(f"  path list: {lst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
