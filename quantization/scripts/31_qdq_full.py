"""Build a QDQ YOLO ONNX with the full calibration set."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.deploy import paths  # noqa: E402

N_CALIB = 200


def load_calib(n: int) -> np.ndarray:
    d = paths.CALIB / "yolo_frames"
    frames = sorted(d.glob("*.npy"))[:n]
    if not frames:
        raise SystemExit(f"no calibration frames in {d}")
    return np.stack([np.load(f) for f in frames]).astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-calib", type=int, default=N_CALIB)
    args = ap.parse_args()

    import onnx
    from onnxruntime.quantization import (CalibrationDataReader, QuantFormat,
                                          QuantType, quantize_static)
    from onnxruntime.quantization.shape_inference import quant_pre_process

    src = paths.MODELS / "yolo26n_pose_480x640_t4.onnx"
    out_dir = PROJECT_ROOT / "experiments" / "qat"
    out_dir.mkdir(parents=True, exist_ok=True)

    calib = load_calib(args.n_calib)
    print(f"calibration: {calib.shape}", flush=True)

    # onnxruntime's quantiser mangles non-ASCII paths when it writes its
    # `-inferred.onnx` sidecar, so all file work happens in an ASCII temp dir
    # (same fix as scripts/29).
    import shutil
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        local_pre = td_path / "pre.onnx"
        local_qdq = td_path / "qdq.onnx"

        # `quant_pre_process` is REQUIRED on this graph: without it the
        # calibrator crashes in a ReduceMax node
        # "/model.11/Constant_1_output_0_ReduceMax"
        # (the fp32 graph itself runs fine -- it is the QDQ insertion pass that
        # trips over that node). Verified .
        print("[1/3] quant_pre_process", flush=True)
        quant_pre_process(str(src), str(local_pre), skip_symbolic_shape=False)

        input_name = onnx.load(str(local_pre)).graph.input[0].name

        class Reader(CalibrationDataReader):
            def __init__(self, data: np.ndarray, name: str):
                self._it = iter([{name: data[i:i + 1]}
                                 for i in range(len(data))])

            def get_next(self):
                return next(self._it, None)

        print(f"[2/3] quantize_static QDQ over {len(calib)} frames", flush=True)
        quantize_static(
            model_input=str(local_pre),
            model_output=str(local_qdq),
            calibration_data_reader=Reader(calib, input_name),
            quant_format=QuantFormat.QDQ,
            activation_type=QuantType.QInt8,
            weight_type=QuantType.QInt8,
            per_channel=True,
        )
        dst = out_dir / "yolo_t4_qdq_full.onnx"
        shutil.copyfile(local_qdq, dst)

    m = onnx.load(str(dst))
    ops = [n.op_type for n in m.graph.node]
    print(f"[3/3] wrote {dst.name} ({dst.stat().st_size/1e6:.2f} MB)")
    print(f"      QuantizeLinear x{ops.count('QuantizeLinear')}, "
          f"DequantizeLinear x{ops.count('DequantizeLinear')}")
    print(f"      outputs: {[o.name for o in m.graph.output]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
