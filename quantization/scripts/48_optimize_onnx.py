"""Run the toolkit's own ONNX optimiser before quantising."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.deploy import paths  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dump", action="store_true",
                    help="also pass inputs_for_dump to dump all tensors")
    ap.add_argument("--n-dump", type=int, default=4)
    ap.add_argument("--passes", default=None,
                    help="comma-separated passes_enable list; default all")
    args = ap.parse_args()

    import numpy as np
    from rknn.api import RKNN

    src = paths.MODELS / "yolo26n_pose_480x640_t4.onnx"
    if not src.exists():
        raise SystemExit(f"missing {src}")
    dst = paths.PARITY / "yolo26n_pose_480x640_t4_optimized.onnx"
    dst.parent.mkdir(parents=True, exist_ok=True)

    dump_inputs = None
    if args.dump:
        frames = sorted((paths.CALIB / "yolo_frames").glob("*.npy"))[:args.n_dump]
        if not frames:
            raise SystemExit("no calibration frames to dump with")
        dump_inputs = [str(f) for f in frames]
        print(f"dumping with {len(dump_inputs)} inputs", flush=True)

    passes = None
    if args.passes:
        passes = [p.strip() for p in args.passes.split(",") if p.strip()]
        print(f"passes_enable: {passes}", flush=True)

    rknn = RKNN(verbose=True)
    # `optimize_onnx` still requires `config` to have run -- without it the
    # toolkit raises "Please call rknn.config first!" from
    # `_create_ir_and_inputs_meta` (measured ). The quantisation
    # fields are irrelevant here (no build happens), but target_platform is
    # what selects the NPU op set the rewrite targets.
    rknn.config(mean_values=[[0, 0, 0]], std_values=[[1, 1, 1]],
                target_platform="rk3576", optimization_level=3)
    print(f"optimize_onnx: {src.name} -> {dst.name}", flush=True)
    rc = rknn.optimize_onnx(model=str(src), optimized_path=str(dst),
                            passes_enable=passes,
                            inputs_for_dump=dump_inputs)
    print(f"optimize_onnx returned {rc}", flush=True)
    rknn.release()
    if rc != 0:
        raise SystemExit("optimize_onnx failed -- no official graph rewrite available")

    if not dst.exists():
        raise SystemExit(f"optimize_onnx returned 0 but wrote no file at {dst}")
    print(f"wrote {dst} ({dst.stat().st_size/1e6:.2f} MB)", flush=True)

    # Compare the graphs structurally: if the optimiser changed the op mix, the
    # numeric equivalence has to be re-established before anything is built
    # from it (L-125: never verify a graph you did not build).
    import onnx
    from collections import Counter

    a = onnx.load(str(src))
    b = onnx.load(str(dst))
    ca = Counter(n.op_type for n in a.graph.node)
    cb = Counter(n.op_type for n in b.graph.node)
    diff = {k: (ca.get(k, 0), cb.get(k, 0))
            for k in sorted(set(ca) | set(cb)) if ca.get(k, 0) != cb.get(k, 0)}
    print(f"\nnodes: {len(a.graph.node)} -> {len(b.graph.node)}", flush=True)
    print(f"outputs: {[o.name for o in a.graph.output]} -> "
          f"{[o.name for o in b.graph.output]}", flush=True)
    if diff:
        print("op count changes:", flush=True)
        for k, (x, y) in diff.items():
            print(f"    {k:16s} {x:4d} -> {y:4d}", flush=True)
    else:
        print("op mix unchanged", flush=True)

    rec = {"src": str(src), "dst": str(dst), "returncode": rc,
           "size_src": src.stat().st_size, "size_dst": dst.stat().st_size,
           "n_nodes_src": len(a.graph.node), "n_nodes_dst": len(b.graph.node),
           "outputs_src": [o.name for o in a.graph.output],
           "outputs_dst": [o.name for o in b.graph.output],
           "op_diff": {k: list(v) for k, v in diff.items()},
           "dump_inputs": dump_inputs}
    out = paths.PARITY / "g5_optimize_onnx.json"
    out.write_text(json.dumps(rec, indent=2), encoding="utf-8")
    print(f"\nwrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
