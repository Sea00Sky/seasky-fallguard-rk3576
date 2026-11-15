"""Export the QAT-fine-tuned ST-GCN++ to a QDQ ONNX carrying trained scales."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT.parent / "YOLOV26n-pose+ST-GCN++"
sys.path.insert(0, str(PROJECT_ROOT))


def _load_own(name: str, rel: str):
    """Load one of this project's modules by path."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(name, PROJECT_ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

CKPT = Path("qat_runs/stgcnpp_qat.pth")
OUT_DIR = Path("qat_runs")
QUANT_OPS = {"Conv", "MatMul", "Gemm"}


# ------------------------------------------------------------------ fp32 graph


def export_fp32(ckpt: Path, out: Path) -> None:
    """Export the QAT-trained float weights as a plain fp32 graph."""
    import torch

    stgcn_export = _load_own("deploy_stgcn_export", "src/deploy/stgcn_export.py")
    sys.path.insert(0, str(SOURCE_ROOT))
    from src.train import engine

    cfg = engine.load_config(None)
    model = engine.build_recognizer(cfg)
    sd = torch.load(str(ckpt), map_location="cpu")["model"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    # Every *weight* must land. Anything missing here would mean the QAT run
    # changed the architecture, which would invalidate the comparison.
    missing_w = [k for k in missing if k.endswith(".weight") or k.endswith(".bias")]
    print(f"      load: {len(missing)} missing ({len(missing_w)} weights), "
          f"{len(unexpected)} unexpected (observers)")
    if missing_w:
        raise SystemExit(f"weights missing from checkpoint: {missing_w[:5]}")
    model.eval()

    wrapper = stgcn_export.STGCNExport(model, patch=True).eval()
    print(f"      einsum->matmul patched on {wrapper.n_patched} unit_gcn blocks")
    dummy = torch.zeros(1, 1, 64, 17, 3, dtype=torch.float32)
    # opset 13, not the 12 the fp16 deliverable uses: per-channel
    # `QuantizeLinear` needs the `axis` attribute, which only exists from 13 on.
    # (opset 12 rejects it with "Unrecognized attribute: axis".) The exported
    # ops are otherwise the same set, so the comparison to the fp16 baseline
    # stays apples-to-apples.
    torch.onnx.export(
        wrapper, dummy, str(out),
        opset_version=13, input_names=["input"], output_names=["logits"],
        dynamic_axes=None, do_constant_folding=True, verbose=False)


# ------------------------------------------------------- trained observer ranges


def _obs_range(fq) -> tuple[float, float] | None:
    """(min, max) from a trained FakeQuantize, or None if it never fired."""
    obs = getattr(fq, "activation_post_process", fq)
    lo = getattr(obs, "min_val", None)
    hi = getattr(obs, "max_val", None)
    if lo is None or hi is None:
        return None
    lo_t, hi_t = np.asarray(lo).ravel(), np.asarray(hi).ravel()
    if lo_t.size == 0:
        return None
    a, b = float(lo_t.min()), float(hi_t.max())
    # An observer that never ran leaves 0/0 or an inverted range.
    if not np.isfinite(a) or not np.isfinite(b) or b <= a:
        return None
    return a, b


def trained_ranges(ckpt: Path) -> dict[str, dict]:
    """module path -> {'act': (lo,hi), 'weight': (lo,hi) or per-channel arrays}."""
    import torch
    import torch.nn as nn
    from torch.ao.quantization import (FakeQuantize, MovingAverageMinMaxObserver,
                                       PerChannelMinMaxObserver, QConfig,
                                       prepare_qat)

    sys.path.insert(0, str(SOURCE_ROOT))
    from src.train import engine

    cfg = engine.load_config(None)
    model = engine.build_recognizer(cfg)
    # `FakeQuantize.with_args`, not the bare observers -- see scripts/33. A bare
    # observer is accepted by `prepare_qat` but its module is a no-op, so the
    # "trained ranges" it records are just running min/max of untouched tensors.
    act = FakeQuantize.with_args(
        observer=MovingAverageMinMaxObserver,
        dtype=torch.qint8, qscheme=torch.per_tensor_affine, reduce_range=False)
    wt = FakeQuantize.with_args(
        observer=PerChannelMinMaxObserver,
        dtype=torch.qint8, qscheme=torch.per_channel_symmetric,
        reduce_range=False)
    model.eval()
    model.qconfig = QConfig(activation=act, weight=wt)
    for _, mod in model.named_modules():
        if isinstance(mod, (nn.Conv2d, nn.Linear)):
            mod.qconfig = model.qconfig
    model = prepare_qat(model.train(), inplace=False)

    sd = torch.load(str(ckpt), map_location="cpu")["model"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        raise SystemExit(f"QAT structure mismatch, {len(missing)} missing: "
                         f"{missing[:5]}")

    out: dict[str, dict] = {}
    n_act = n_w = 0
    for name, mod in model.named_modules():
        rec: dict = {}
        a = _obs_range(getattr(mod, "activation_post_process", None))
        if a is not None:
            rec["act"] = a
            n_act += 1
        fq = getattr(mod, "weight_fake_quant", None)
        if fq is not None:
            obs = getattr(fq, "activation_post_process", fq)
            lo = getattr(obs, "min_val", None)
            hi = getattr(obs, "max_val", None)
            if lo is not None and hi is not None:
                lo_a, hi_a = np.asarray(lo).ravel(), np.asarray(hi).ravel()
                if lo_a.size and np.all(np.isfinite(lo_a)) and np.all(hi_a > lo_a):
                    rec["weight"] = (lo_a, hi_a)
                    n_w += 1
        if rec:
            out[name] = rec
    print(f"      trained observers: {n_act} activation, {n_w} weight")
    return out


# --------------------------------------------------------------- graph surgery


def _module_of(node_name: str) -> str | None:
    """ONNX node name -> torch module path."""
    parts = [p for p in node_name.split("/") if p]
    if len(parts) < 2:
        return None
    body = parts[:-1]
    out: list[str] = []
    for seg in body:
        if out and seg.startswith(out[-1] + "."):
            out.append(seg[len(out[-1]) + 1:])
        else:
            out.append(seg)
    return ".".join(out)


def _act_qdq(prefix: str, src: str, lo: float, hi: float):
    """Per-tensor affine QDQ for one activation tensor (rknn's default)."""
    from onnx import helper, numpy_helper

    scale = np.array([max(hi - lo, 1e-8) / 255.0], dtype=np.float32)
    z = int(round(-lo / float(scale[0]))) - 128
    zp = np.array([max(-128, min(127, z))], dtype=np.int8)
    s_name, z_name = f"{prefix}_scale", f"{prefix}_zp"
    q_out, dq_out = f"{prefix}_q", f"{prefix}_dq"
    return (
        [numpy_helper.from_array(scale, s_name),
         numpy_helper.from_array(zp, z_name)],
        [helper.make_node("QuantizeLinear", [src, s_name, z_name], [q_out],
                          name=f"{prefix}_Q"),
         helper.make_node("DequantizeLinear", [q_out, s_name, z_name], [dq_out],
                          name=f"{prefix}_DQ")],
        dq_out,
    )


def fold_weights(m, ranges: dict[str, dict]) -> int:
    """Quantise-dequantise the weights in place, using the trained per-channel ranges."""
    import onnx
    from onnx import numpy_helper

    inits = {i.name: i for i in m.graph.initializer}
    n = 0
    for node in m.graph.node:
        if node.op_type not in QUANT_OPS or len(node.input) < 2:
            continue
        mod = _module_of(node.name)
        rec = ranges.get(mod) if mod else None
        if not rec or "weight" not in rec:
            continue
        init = inits.get(node.input[1])
        if init is None:          # weight is produced by a node, not a constant
            continue
        w = numpy_helper.to_array(init).astype(np.float32)
        lo, hi = rec["weight"]
        if lo.size != w.shape[0]:
            continue              # not per-output-channel; leave it alone
        # absmax of the tensor that is actually here, per output channel
        per_ch = np.abs(w).reshape(w.shape[0], -1).max(axis=1)
        # torch's symmetric per-channel FakeQuantize uses absmax/127.5 (verified
        # by exact scale match), with the int8 grid clipped to [-128, 127].
        # Using /127 here leaves a half-step bias on every weight.
        scale = np.maximum(per_ch / 127.5, 1e-8).astype(np.float32)
        shape = [w.shape[0]] + [1] * (w.ndim - 1)
        q = np.clip(np.round(w / scale.reshape(shape)), -128, 127)
        init.CopyFrom(numpy_helper.from_array(
            (q * scale.reshape(shape)).astype(np.float32), init.name))
        n += 1
    return n


def insert_act_qdq(src: Path, dst: Path, ranges: dict[str, dict]) -> dict[str, int]:
    """Splice QDQ after every quantisable op, at the position the observer watched."""
    import onnx

    m = onnx.load(str(src))
    g = m.graph
    stats = {"folded": fold_weights(m, ranges), "act": 0, "skipped": 0}

    # old tensor name -> new (dequantised) name, applied to every consumer
    rename: dict[str, str] = {}
    new_nodes, new_inits, inserted = [], [], set()
    for node in g.node:
        new_nodes.append(node)
        if node.op_type not in QUANT_OPS or not node.output:
            continue
        mod = _module_of(node.name)
        rec = ranges.get(mod) if mod else None
        if not rec or "act" not in rec:
            stats["skipped"] += 1
            continue
        lo, hi = rec["act"]
        inits, nodes, dq_out = _act_qdq(f"{mod}/act", node.output[0], lo, hi)
        new_inits += inits
        # The Q node consumes the op's raw output -- that is the whole point --
        # so it must be exempt from the rename below. Renaming it too made the
        # Q node read its own DQ output and the checker reported the graph as
        # not topologically sorted.
        inserted.update(id(n) for n in nodes)
        new_nodes += nodes
        rename[node.output[0]] = dq_out
        stats["act"] += 1

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
    onnx.save(m, str(dst))
    return stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(CKPT))
    ap.add_argument("--no-weight-qdq", action="store_true",
                    help="activation QDQ only (rknn re-quantises weights anyway)")
    args = ap.parse_args()

    import onnx

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fp32 = OUT_DIR / "stgcn_qat_fp32.onnx"
    qdq = OUT_DIR / "stgcnpp_qat_qdq.onnx"

    print("[1/3] exporting QAT float weights to fp32 ONNX", flush=True)
    export_fp32(Path(args.ckpt), fp32)
    print(f"      wrote {fp32.name} ({fp32.stat().st_size/1e6:.2f} MB)")

    print("[2/3] reading trained observer ranges from the checkpoint", flush=True)
    ranges = trained_ranges(Path(args.ckpt))

    print("[3/3] splicing QDQ", flush=True)
    stats = insert_act_qdq(fp32, qdq, ranges)
    print(f"      activation QDQ x{stats['act']}, weights folded x{stats['folded']}, "
          f"skipped {stats['skipped']}")

    m = onnx.load(str(qdq))
    ops = [x.op_type for x in m.graph.node]
    nq = ops.count("QuantizeLinear")
    print(f"      QuantizeLinear x{nq}, DequantizeLinear x{ops.count('DequantizeLinear')}")
    if nq == 0:
        raise SystemExit("no QDQ inserted -- rknn would treat this as fp32")
    onnx.checker.check_model(m)
    print(f"      wrote {qdq}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
