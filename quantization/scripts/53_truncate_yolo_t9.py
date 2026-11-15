"""Truncate YOLO26n-pose into the official nine-head layout (t9)."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.deploy import paths  # noqa: E402

SOURCE = paths.MODELS / "yolo26n_pose_480x640.onnx"
OUT = paths.MODELS / "yolo26n_pose_480x640_t9.onnx"

# (label, tensor). Order matches the official convert.py so a reader can diff
# the two lists line by line.
HEADS: list[tuple[str, str]] = [
    ("box_1", "/one2one_cv2.0/one2one_cv2.0.2/Conv_output_0"),
    ("score_1", "/one2one_cv3.0/one2one_cv3.0.2/Conv_output_0"),
    ("pos_1", "/one2one_cv4_kpts.0/Conv_output_0"),
    ("box_2", "/one2one_cv2.1/one2one_cv2.1.2/Conv_output_0"),
    ("score_2", "/one2one_cv3.1/one2one_cv3.1.2/Conv_output_0"),
    ("pos_2", "/one2one_cv4_kpts.1/Conv_output_0"),
    ("box_3", "/one2one_cv2.2/one2one_cv2.2.2/Conv_output_0"),
    ("score_3", "/one2one_cv3.2/one2one_cv3.2.2/Conv_output_0"),
    ("pos_3", "/one2one_cv4_kpts.2/Conv_output_0"),
]
NAMES = [t for _, t in HEADS]


def _load_t24():
    spec = importlib.util.spec_from_file_location(
        "truncate24", Path(__file__).resolve().parent / "24_truncate_yolo.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build() -> onnx.ModelProto:
    t24 = _load_t24()
    model = onnx.load(str(SOURCE))

    available = {o.name for o in model.graph.output}
    for n in model.graph.node:
        available.update(n.output)
    missing = [n for n in NAMES if n not in available]
    if missing:
        raise SystemExit(f"{SOURCE.name} has no tensor(s) {missing}")

    # Prune dead nodes exactly like `scripts/24` does -- the toolkit walks the
    # graph and a half-live Concat_5 confuses its shape inference.
    reachable = t24._reachable(model.graph, set(NAMES))
    new_nodes = [n for n in model.graph.node if n.output[0] in reachable]
    del model.graph.node[:]
    model.graph.node.extend(new_nodes)

    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=False)
    shapes = {v.name: [d.dim_value or d.dim_param
                       for d in v.type.tensor_type.shape.dim]
              for v in inferred.graph.value_info}
    del model.graph.output[:]
    for name in NAMES:
        model.graph.output.append(helper.make_tensor_value_info(
            name, TensorProto.FLOAT, shapes[name]))
    onnx.checker.check_model(model)
    return model


def verify(model: onnx.ModelProto, n: int) -> dict:
    """Nine cut points must equal the full graph's intermediates bit-exactly."""
    import onnxruntime as ort

    t24 = _load_t24()
    full = onnx.load(str(SOURCE))

    # Take every original node the pruned graph dropped, i.e. the post-head
    # chain (Reshape/Concat/DFL/Sigmoid/Concat_5), and append it back.
    kept = {n.name for n in model.graph.node}
    chain = [n for n in full.graph.node if n.name not in kept]
    rebuilt = onnx.ModelProto()
    rebuilt.CopyFrom(model)
    rebuilt.graph.node.extend(chain)
    del rebuilt.graph.output[:]
    # The final Concat is named `/Concat_5` and writes the graph's own output
    # name, `output0` -- not `<node>_output_0` (measured ).
    rebuilt.graph.output.append(helper.make_tensor_value_info(
        "output0", TensorProto.FLOAT, [1, 56, 6300]))
    onnx.checker.check_model(rebuilt)

    tmp = paths.MODELS / "_tmp_t9_rebuilt.onnx"
    onnx.save(rebuilt, str(tmp))
    try:
        s_cut = ort.InferenceSession(str(tmp), providers=["CPUExecutionProvider"])
        s_full = ort.InferenceSession(str(SOURCE),
                                      providers=["CPUExecutionProvider"])
        inp = s_cut.get_inputs()[0].name
        frames = sorted((paths.CALIB / "yolo_frames").glob("*.npy"))[:n]
        worst = 0.0
        for f in frames:
            x = np.load(f)
            x = x[None] if x.ndim == 3 else x
            x = x.astype(np.float32)
            got = s_cut.run(None, {inp: x})[0]
            want = s_full.run(None, {inp: x})[0]
            worst = max(worst, float(np.abs(got - want).max()))
    finally:
        tmp.unlink(missing_ok=True)
    return {"n_frames": len(frames), "max_abs_diff_output0": worst,
            "bit_exact": worst == 0.0, "n_chain_nodes": len(chain)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--samples", type=int, default=20)
    args = ap.parse_args()
    if not (args.verify or args.write):
        ap.error("choose --verify and/or --write")

    model = build()
    print(f"built nine-head graph: {len(model.graph.node)} nodes, "
          f"{len(model.graph.output)} outputs")
    if args.verify:
        rec = verify(model, args.samples)
        print(f"rebuild vs full output0 over {rec['n_frames']} frames:")
        print(f"  chain nodes appended : {rec['n_chain_nodes']}")
        print(f"  max|d| on output0    : {rec['max_abs_diff_output0']:.3e}")
        print(f"  bit exact            : {rec['bit_exact']}")
        if not rec["bit_exact"]:
            raise SystemExit("nine-head rebuild is NOT exact; refusing to write")
    if args.write:
        onnx.save(model, str(OUT))
        print(f"wrote {OUT} ({OUT.stat().st_size} B)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
