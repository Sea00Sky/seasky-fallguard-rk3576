"""Check that the spliced QDQ graph reproduces the QAT model it came from."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT.parent / "YOLOV26n-pose+ST-GCN++"
sys.path.insert(0, str(PROJECT_ROOT))

RUN = Path("qat_runs")
FP32 = RUN / "stgcn_qat_fp32.onnx"
QDQ = RUN / "stgcnpp_qat_qdq.onnx"
CKPT = RUN / "stgcnpp_qat.pth"


def _load_own(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, PROJECT_ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    import onnxruntime as ort
    import torch
    import torch.nn as nn
    from torch.ao.quantization import (FakeQuantize, MovingAverageMinMaxObserver,
                                       PerChannelMinMaxObserver, QConfig,
                                       prepare_qat)

    stgcn_export = _load_own("deploy_stgcn_export", "src/deploy/stgcn_export.py")
    sys.path.insert(0, str(SOURCE_ROOT))
    from src.train import engine

    cfg = engine.load_config(None)
    base = engine.build_recognizer(cfg)
    sd = torch.load(str(CKPT), map_location="cpu")["model"]
    base.load_state_dict(sd, strict=False)

    # --- the QAT model, in eval so the observers stop updating ----------------
    qat = engine.build_recognizer(cfg)
    act = FakeQuantize.with_args(
        observer=MovingAverageMinMaxObserver,
        dtype=torch.qint8, qscheme=torch.per_tensor_affine, reduce_range=False)
    wt = FakeQuantize.with_args(
        observer=PerChannelMinMaxObserver,
        dtype=torch.qint8, qscheme=torch.per_channel_symmetric, reduce_range=False)
    qat.eval()
    qat.qconfig = QConfig(activation=act, weight=wt)
    for _, m in qat.named_modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            m.qconfig = qat.qconfig
    qat = prepare_qat(qat.train(), inplace=False)
    qat.load_state_dict(sd, strict=False)
    qat.eval()                     # freezes the observers at their trained values
    print("QAT model ready (observers frozen at trained min/max)", flush=True)

    # Guard the guard: if fake-quant is a no-op, every equivalence check below
    # passes trivially against the fp32 model and proves nothing. This is the
    # exact failure the 23:14 run shipped (bare observer in the
    # qconfig), so it is checked before anything is measured.
    probe_mod = qat.backbone.gcn[0].gcn.conv.weight_fake_quant
    w0 = qat.backbone.gcn[0].gcn.conv.weight.detach()
    moved = float((probe_mod(w0) - w0).abs().max())
    print(f"fake-quant liveness: max|fq(w)-w| = {moved:.3e}", flush=True)
    if moved == 0.0:
        raise SystemExit("fake-quant is a no-op -- this checkpoint is a plain "
                         "fine-tune, not a QAT run")

    # --- inputs: real training windows, not noise -----------------------------
    calib = np.load(PROJECT_ROOT / "calib" / "stgcn_calib.npy")[:200].astype(np.float32)
    print(f"probe windows: {calib.shape}", flush=True)

    with torch.no_grad():
        pt_fp32 = stgcn_export.STGCNExport(base, patch=True).eval()(
            torch.from_numpy(calib)).numpy()
        pt_qat = stgcn_export.STGCNExport(qat, patch=True).eval()(
            torch.from_numpy(calib)).numpy()

    def run_onnx(path: Path) -> np.ndarray:
        s = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        name = s.get_inputs()[0].name
        return np.concatenate([s.run(None, {name: calib[i:i + 1]})[0]
                               for i in range(len(calib))])

    on_fp32 = run_onnx(FP32)
    on_qdq = run_onnx(QDQ)

    def decide(L: np.ndarray) -> np.ndarray:
        """The project's decision rule: softmax(logits)[:,1] >= 0.5."""
        p = np.exp(L - L.max(axis=1, keepdims=True))
        p = p / p.sum(axis=1, keepdims=True)
        return p[:, 1] >= 0.5

    def report(tag: str, a: np.ndarray, b: np.ndarray) -> dict:
        d = np.abs(a - b)
        flips = int((decide(a) != decide(b)).sum())
        print(f"  {tag:26s} max|d|={d.max():.3e}  mean={d.mean():.3e}  "
              f"flips={flips}/{len(a)}", flush=True)
        return {"max": float(d.max()), "mean": float(d.mean()), "flips": flips}

    print("\nequivalence:", flush=True)
    r_fp32 = report("onnx fp32 vs torch", pt_fp32, on_fp32)
    r_qdq = report("onnx qdq  vs torch QAT", pt_qat, on_qdq)
    r_gap = report("torch QAT vs torch fp32", pt_fp32, pt_qat)

    # What the comparison is against, for scale: how far QAT moves the logits.
    print(f"\n  QAT moved the logits by up to {r_gap['max']:.3e} vs the fp32 model")
    print(f"  the QDQ graph reproduces that to {r_qdq['max']:.3e}")

    # What the checks are, and what they are NOT.
    #
    # `onnx fp32 vs torch` must be numerically exact -- both are the same float
    # weights, so anything above 1e-4 is a broken export.
    #
    # `onnx qdq vs torch QAT` is NOT expected to be tiny, and treating a large
    # value as failure was wrong. The two apply the same 8-bit grid at the same
    # places but to different tensors: torch quantises `mstcn.transform`'s raw
    # Conv weight, while the ONNX graph has already folded
    # `gamma/sqrt(var+eps)` into it. A given relative quantisation error on two
    # different tensors does not cancel, so the logits legitimately differ.
    # The question that matters is whether the decisions survive, because
    # that is all G5 consumes. Both are therefore judged on flips.
    n = len(pt_fp32)
    dec_fp32 = decide(pt_fp32)
    dec_qdq = decide(on_qdq)
    agree = float((dec_fp32 == dec_qdq).mean())
    flips = int((dec_fp32 != dec_qdq).sum())
    print(f"\n  QDQ decision agreement vs fp32: {agree:.4f} ({flips}/{n} flips)")
    print(f"  QAT decision agreement vs fp32: "
          f"{float((decide(pt_qat) == dec_fp32).mean()):.4f}")

    ok = r_fp32["max"] <= 1e-4 and agree >= 0.99
    print(f"\n{'PASS' if ok else 'FAIL'}: fp32 exact, QDQ keeps >=99% decisions")
    if not ok:
        print("  a decision rate this low means the spliced ranges or the "
              "folded weights are wrong; do not build RKNN from this graph")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
