"""Probe: does rknn-toolkit2 accept a QDQ (fake-quantised) ONNX?"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.deploy import paths  # noqa: E402


def make_qdq_onnx(src: Path, dst: Path, calib: np.ndarray) -> None:
    """Static QDQ quantisation of `src`, using `calib` for the ranges."""
    import shutil
    import tempfile

    from onnxruntime.quantization import (CalibrationDataReader, QuantFormat,
                                          QuantType, quantize_static)

    class Reader(CalibrationDataReader):
        def __init__(self, data: np.ndarray, name: str):
            self._it = iter([{name: data[i:i + 1]} for i in range(len(data))])

        def get_next(self):
            return next(self._it, None)

    # `quantize_static` wants an input name; read it from the graph rather than
    # hard-coding "input", which is ST-GCN specific.
    import onnx
    m = onnx.load(str(src))
    input_name = m.graph.input[0].name

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        local_src = td_path / "model.onnx"
        local_dst = td_path / "model_qdq.onnx"
        shutil.copyfile(src, local_src)
        quantize_static(
            model_input=str(local_src),
            model_output=str(local_dst),
            calibration_data_reader=Reader(calib, input_name),
            quant_format=QuantFormat.QDQ,
            activation_type=QuantType.QInt8,
            weight_type=QuantType.QInt8,
            per_channel=True,
        )
        shutil.copyfile(local_dst, dst)


def count_qdq(onnx_path: Path) -> tuple[int, int]:
    import onnx
    m = onnx.load(str(onnx_path))
    ops = [n.op_type for n in m.graph.node]
    return ops.count("QuantizeLinear"), ops.count("DequantizeLinear")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="stgcn", choices=["stgcn"])
    ap.add_argument("--n-calib", type=int, default=50)
    ap.add_argument("--skip-qdq", action="store_true",
                    help="reuse an existing QDQ onnx (onnxruntime lives on the "
                         "Windows side; rknn lives in WSL, so the two halves "
                         "run in different interpreters)")
    args = ap.parse_args()

    out_dir = PROJECT_ROOT / "experiments" / "qat"
    out_dir.mkdir(parents=True, exist_ok=True)
    src = paths.MODELS / "stgcnpp_v2.onnx"
    dst = out_dir / "stgcnpp_v2_qdq.onnx"

    # Calibration windows from the frozen training pkl, the same source
    # `scripts/20.build_stgcn_calibration` uses (A11: never board data).
    npy = paths.CALIB / "stgcn_calib.npy"
    if not npy.exists():
        raise SystemExit(f"missing {npy}; run scripts/20 --precision int8 first")
    calib = np.load(npy)[: args.n_calib].astype(np.float32)
    print(f"calibration: {calib.shape} from {npy.name}", flush=True)

    if args.skip_qdq:
        if not dst.exists():
            raise SystemExit(f"missing {dst}; run without --skip-qdq on Windows")
        print(f"\n[1/4] reusing {dst.name}", flush=True)
    else:
        print(f"\n[1/4] QDQ-quantising {src.name} -> {dst.name}", flush=True)
        make_qdq_onnx(src, dst, calib)
    q, dq = count_qdq(dst)
    print(f"      QuantizeLinear x{q}, DequantizeLinear x{dq}", flush=True)
    if q == 0:
        print("      FAIL: no QDQ nodes in the graph", flush=True)
        return 2

    print("\n[2/4] loading the QDQ graph into rknn-toolkit2", flush=True)
    from rknn.api import RKNN
    rknn = RKNN(verbose=False)
    # ST-GCN's input is (1, 1, 64, 17, 3): one channel, not three. Reusing
    # YOLO's `[[0, 0, 0]]` here fails with
    # "The len of mean_values ([0, 0, 0]) for input 0 is wrong, expect 1!"
    rknn.config(mean_values=[[0]], std_values=[[1]],
                target_platform="rk3576", optimization_level=3)
    ret = rknn.load_onnx(model=str(dst))
    print(f"      load_onnx -> {ret}", flush=True)
    if ret != 0:
        print("\nRESULT: CONVERT FAIL (load_onnx rejected the QDQ graph)",
              flush=True)
        return 3

    print("\n[3/4] build (do_quantization=False: the graph is already quantised)",
          flush=True)
    ret = rknn.build(do_quantization=False)
    print(f"      build -> {ret}", flush=True)
    if ret != 0:
        print("\nRESULT: CONVERT FAIL (build rejected the QDQ graph)", flush=True)
        return 4

    rknn_path = out_dir / "stgcnpp_v2_qdq.rknn"
    rknn.export_rknn(str(rknn_path))
    print(f"      wrote {rknn_path.name} "
          f"({rknn_path.stat().st_size / 1e6:.2f} MB)", flush=True)

    print("\n[4/4] simulator (the step that decides whether QAT is measurable)",
          flush=True)
    ret = rknn.init_runtime(target=None)
    print(f"      init_runtime(target=None) -> {ret}", flush=True)
    if ret != 0:
        print("\nRESULT: CONVERT OK + SIM FAIL -- the QAT model can be built "
              "but NOT measured on this PC. Report as a limitation, not a "
              "number.", flush=True)
        rknn.release()
        return 5

    outs = rknn.inference(inputs=[calib[0]], data_format="nchw")
    if outs is None:
        print("\nRESULT: CONVERT OK + SIM FAIL (inference returned None)",
              flush=True)
        rknn.release()
        return 5
    logits = np.asarray(outs[0])
    print(f"      logits {logits.shape}: {logits.ravel()}", flush=True)
    print("\nRESULT: CONVERT OK + SIM OK -- QAT is deliverable AND measurable.",
          flush=True)
    rknn.release()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
