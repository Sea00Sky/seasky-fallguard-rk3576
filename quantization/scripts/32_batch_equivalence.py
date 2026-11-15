"""Does PLAN §3.3 batching change any G3 number?"""

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


paths = _load_own("deploy_paths", "src/deploy/paths.py")


def _replay_detector():
    """The `ReplayDetector` from scripts/31, loaded by path."""
    spec = importlib.util.spec_from_file_location(
        "g3_script", PROJECT_ROOT / "scripts" / "31_pipeline_correctness.py")
    mod = importlib.util.module_from_spec(spec)
    # Execute only up to the definitions; `main()` is guarded by __main__.
    spec.loader.exec_module(mod)
    return mod.ReplayDetector


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--limit", type=int, default=0, help="first N videos only")
    p.add_argument("--batch", type=int, default=4, help="score_batch to test")
    return p.parse_args()


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args()

    import src.runtime as rt

    ReplayDetector = _replay_detector()

    batch_graph = paths.MODELS / "stgcnpp_v2_b4.onnx"
    if not batch_graph.exists():
        raise SystemExit(f"missing {batch_graph}; run scripts/15 first")

    rows = [r for r in paths.read_csv(paths.SOURCE_MANIFEST_BOARD)
            if r.get("readable") == "True"]
    if args.limit:
        rows = rows[:args.limit]
    print(f"{len(rows)} videos, score_batch={args.batch} vs 0", flush=True)

    per_mode: dict[int, list[tuple]] = {0: [], args.batch: []}
    for n, row in enumerate(rows, 1):
        vid = row["video_id"]
        npz = paths.keypoint_npz(vid)
        if not npz.exists():
            raise SystemExit(f"missing keypoints {npz}")

        for mode in (0, args.batch):
            det = ReplayDetector(npz)
            cfg = rt.PipelineConfig(
                yolo_onnx=paths.MODELS / "yolo26n_pose_480x640.onnx",
                stgcn_onnx=paths.MODELS / "stgcnpp_v2.onnx",
                stitch=False,          # strategy B, same as the G3 baseline
                score_batch=mode,
            )
            pipe = rt.Pipeline(cfg, detector=det)
            if mode and pipe.stgcn_batch is None:
                raise SystemExit(
                    f"score_batch={mode} was requested but no batch graph "
                    f"loaded -- the comparison would be vacuous")
            dummy = np.zeros((det.height, det.width, 3), np.uint8)
            for _ in range(det.n_frames):
                pipe.push(dummy)
            result = pipe.finish(vid)
            per_mode[mode].extend(
                (vid, w.window_id, w.track_id, w.prob_fall) for w in result.windows)
        print(f"  [{n}/{len(rows)}] {vid} done", flush=True)

    a = per_mode[0]
    b = per_mode[args.batch]
    if len(a) != len(b):
        raise SystemExit(f"window counts differ: {len(a)} vs {len(b)}")

    ids_a = [(x[0], x[1], x[2]) for x in a]
    ids_b = [(x[0], x[1], x[2]) for x in b]
    id_mismatch = sum(1 for x, y in zip(ids_a, ids_b) if x != y)

    pa = np.array([x[3] for x in a], dtype=np.float64)
    pb = np.array([x[3] for x in b], dtype=np.float64)
    d = np.abs(pa - pb)
    flips = int(((pa >= 0.5) != (pb >= 0.5)).sum())

    print(f"\nwindows: {len(a)}")
    print(f"  id mismatches:      {id_mismatch}")
    print(f"  max|dp|:            {d.max():.3e}")
    print(f"  mean|dp|:           {d.mean():.3e}")
    print(f"  bitwise identical:  {np.array_equal(pa, pb)}")
    print(f"  threshold flips:    {flips}")

    ok = id_mismatch == 0 and d.max() <= 1e-6 and flips == 0
    print(f"\n{'PASS' if ok else 'FAIL'}: batching is a speed change only")

    result = {
        "n_videos": len(rows),
        "score_batch": args.batch,
        "n_windows": len(a),
        "id_mismatches": id_mismatch,
        "max_abs_dp": float(d.max()),
        "mean_abs_dp": float(d.mean()),
        "bitwise_identical": bool(np.array_equal(pa, pb)),
        "threshold_flips": flips,
        "tolerance": 1e-6,
        "passed": bool(ok),
    }
    paths.write_json(paths.PARITY / "g1b_batch_equivalence.json", result)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
