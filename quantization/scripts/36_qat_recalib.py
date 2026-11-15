"""Rebuild the QAT graph with measured activation ranges and see if it recovers."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

RUN = Path("qat_runs")
FP32 = RUN / "stgcn_qat_fp32.onnx"
RANGES = RUN / "_actual_ranges_full.npy"
OUT = PROJECT_ROOT / "experiments" / "qat" / "stgcnpp_qat_qdq_recalib.onnx"


def _load_own(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, PROJECT_ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    import onnx
    from onnx import helper, numpy_helper

    q34 = _load_own("q34", "scripts/34_export_qat_onnx.py")
    QUANT_OPS = q34.QUANT_OPS

    m = onnx.load(str(FP32))
    g = m.graph

    # The measured ranges were recorded for Conv and Gemm only. The 10 MatMul
    # nodes are the graph-convolution adjacency products
    # (`.../gcn/MatMul` with the constant `A` as second input); they have no
    # torch module behind them, so they were never in the observer table and
    # are deliberately left unquantised -- same as in scripts/34.
    conv_out = [n.output[0] for n in g.node
                if n.op_type in ("Conv", "Gemm") and n.output]
    act = np.load(str(RANGES))
    if len(act) != len(conv_out):
        raise SystemExit(f"{len(act)} measured ranges vs {len(conv_out)} Conv/Gemm")
    measured = {t: (float(lo), float(hi)) for t, (lo, hi) in zip(conv_out, act)}

    # Reuse 34's trained-range table only for the *weight* scales (those are
    # per-channel absmax of the tensor in the graph and do not drift).
    trained = q34.trained_ranges(q34.CKPT)
    folded = q34.fold_weights(m, trained)

    rename: dict[str, str] = {}
    new_nodes, new_inits, inserted = [], [], set()
    n_act = 0
    for node in g.node:
        new_nodes.append(node)
        if node.op_type not in QUANT_OPS or not node.output:
            continue
        t = node.output[0]
        if t not in measured:
            continue
        lo, hi = measured[t]
        mod = q34._module_of(node.name) or node.name
        inits, nodes, dq_out = q34._act_qdq(f"{mod}/act", t, lo, hi)
        new_inits += inits
        inserted.update(id(x) for x in nodes)
        new_nodes += nodes
        rename[t] = dq_out
        n_act += 1

    for node in new_nodes:
        if id(node) in inserted:
            continue
        for i, name in enumerate(node.input):
            if name in rename:
                node.input[i] = rename[name]
    for o in g.output:
        if o.name in rename:
            o.name = rename[o.name]

    g.ClearField("node")
    g.node.extend(new_nodes)
    g.initializer.extend(new_inits)
    onnx.checker.check_model(m)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(m, str(OUT))

    ops = [x.op_type for x in m.graph.node]
    print(f"folded {folded} weights, spliced {n_act} measured activation QDQ")
    print(f"QuantizeLinear x{ops.count('QuantizeLinear')}")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
