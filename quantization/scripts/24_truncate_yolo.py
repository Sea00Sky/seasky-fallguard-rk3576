"""Truncate the YOLO26n-pose ONNX into the official multi-output layout."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.deploy import paths  # noqa: E402

SOURCE = paths.MODELS / "yolo26n_pose_480x640.onnx"

# Tensor names inside the exported graph. Confirmed by `onnx.shape_inference`
# on :
# /Mul_2_output_0 [1, 4, 6300] decoded boxes
# /Sigmoid_output_0 [1, 1, 6300] class confidence
# /Mul_3_output_0 [1, 17, 2, 6300] keypoint xy (stride-applied)
# /Sigmoid_1_output_0 [1, 17, 1, 6300] keypoint score
# /Concat_4_output_0 [1, 17, 3, 6300] = Concat(axis=2) of the two above
# /Reshape_10_output_0 [1, 51, 6300] = Reshape(Concat_4)
LAYOUTS: dict[str, list[str]] = {
    "t3": ["/Mul_2_output_0", "/Sigmoid_output_0", "/Reshape_10_output_0"],
    "t4": ["/Mul_2_output_0", "/Sigmoid_output_0",
           "/Mul_3_output_0", "/Sigmoid_1_output_0"],
}

# The op chain from the truncation point to `output0`, used only to check that
# the chosen outputs really do rebuild it.
_CONCAT5 = "Concat_5"


def truncate(layout: str) -> tuple[onnx.ModelProto, list[str]]:
    if layout not in LAYOUTS:
        raise SystemExit(f"unknown layout {layout!r}; choose {sorted(LAYOUTS)}")
    names = LAYOUTS[layout]
    model = onnx.load(str(SOURCE))

    # The exported graph carries no `value_info`, so the only place a tensor
    # name is guaranteed to exist is a node output (plus the declared outputs).
    available = {o.name for o in model.graph.output}
    for n in model.graph.node:
        available.update(n.output)
    missing = [n for n in names if n not in available]
    if missing:
        raise SystemExit(f"{SOURCE} has no tensor(s) {missing}; "
                         f"the graph changed -- re-check the node names")

    # Drop the original output declaration and re-declare the cut points.
    # Everything downstream of them is simply unreachable and onnx.checker is
    # happy with that (a graph may contain dead nodes only if they are pruned,
    # so we prune explicitly -- the toolkit walks the graph and a dangling
    # Concat_5 with one live input would confuse shape inference).
    # Ask ONNX for the shapes first: the exported graph carries no `value_info`,
    # so the new output declarations need inference to be filled in.
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=False)
    shapes = {v.name: [d.dim_value or d.dim_param
                       for d in v.type.tensor_type.shape.dim]
              for v in inferred.graph.value_info}

    keep_outputs = set(names)
    reachable = _reachable(model.graph, keep_outputs)
    new_nodes = [n for n in model.graph.node if n.output[0] in reachable]
    del model.graph.node[:]
    model.graph.node.extend(new_nodes)

    del model.graph.output[:]
    for name in names:
        model.graph.output.append(helper.make_tensor_value_info(
            name, TensorProto.FLOAT, shapes[name]))
    onnx.checker.check_model(model)
    return model, names


def _reachable(graph: onnx.GraphProto, outputs: set[str]) -> set[str]:
    """Outputs of every node that the requested tensors depend on."""
    produced: dict[str, onnx.NodeProto] = {}
    for n in graph.node:
        for o in n.output:
            produced[o] = n
    needed: set[str] = set(outputs)
    stack = list(outputs)
    seen_nodes: set[str] = set()
    while stack:
        t = stack.pop()
        n = produced.get(t)
        if n is None or n.name in seen_nodes:
            continue
        seen_nodes.add(n.name)
        for i in n.input:
            if i:
                needed.add(i)
                stack.append(i)
    return {o for n in graph.node if n.name in seen_nodes for o in n.output}


def rebuild(model: onnx.ModelProto, layout: str, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Run the truncated graph and rebuild `output0`; return (full, rebuilt)."""
    import onnxruntime as ort

    names = LAYOUTS[layout]
    tmp = paths.MODELS / f"_tmp_{layout}.onnx"
    onnx.save(model, str(tmp))
    try:
        s_cut = ort.InferenceSession(str(tmp), providers=["CPUExecutionProvider"])
        cut = s_cut.run(None, {s_cut.get_inputs()[0].name: x})
        s_full = ort.InferenceSession(str(SOURCE), providers=["CPUExecutionProvider"])
        full = s_full.run(None, {s_full.get_inputs()[0].name: x})[0]
    finally:
        tmp.unlink(missing_ok=True)

    got = dict(zip(names, cut))
    if layout == "t3":
        rebuilt = np.concatenate(
            [got["/Mul_2_output_0"], got["/Sigmoid_output_0"],
             got["/Reshape_10_output_0"]], axis=1)
    else:
        kpt = np.concatenate(
            [got["/Mul_3_output_0"], got["/Sigmoid_1_output_0"]], axis=2
        ).reshape(1, 51, -1)
        rebuilt = np.concatenate(
            [got["/Mul_2_output_0"], got["/Sigmoid_output_0"], kpt], axis=1)
    return full, rebuilt


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layout", choices=sorted(LAYOUTS), required=True)
    ap.add_argument("--verify-only", action="store_true",
                    help="check the rebuild against output0, do not write the onnx")
    ap.add_argument("--samples", type=int, default=20)
    args = ap.parse_args()

    model, names = truncate(args.layout)
    print(f"layout {args.layout}: {len(model.graph.node)} nodes, "
          f"outputs {names}", flush=True)

    # Bit-exactness against the shipped graph, on real calibration frames.
    from src.deploy import yolo_export # noqa: F401 (ensures the package imports)
    frames = sorted((paths.CALIB / "yolo_frames").glob("*.npy"))[: args.samples]
    worst = 0.0
    for f in frames:
        x = np.load(f)[None].astype(np.float32)
        full, rebuilt = rebuild(model, args.layout, x)
        worst = max(worst, float(np.abs(full - rebuilt).max()))
    print(f"rebuild check over {len(frames)} frames: max|diff| = {worst:.3e}",
          flush=True)
    if worst != 0.0:
        print("REFUSING to write: the truncation is not exact", flush=True)
        return 1

    if args.verify_only:
        return 0

    out = paths.MODELS / f"yolo26n_pose_480x640_{args.layout}.onnx"
    onnx.save(model, str(out))
    print(f"wrote {out} ({out.stat().st_size / 1e6:.2f} MB)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
