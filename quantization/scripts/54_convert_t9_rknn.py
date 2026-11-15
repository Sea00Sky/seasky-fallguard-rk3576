"""Convert the nine-head (t9) graph to RKNN INT8 and score it."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.deploy import paths  # noqa: E402

TARGET = "rk3576"
ONNX = paths.MODELS / "yolo26n_pose_480x640_t9.onnx"

# Same order as `scripts/53_truncate_yolo_t9.py::HEADS`, which is the same
# order as the official convert.py. The runtime maps them by name.
OUTPUTS = [
    "/one2one_cv2.0/one2one_cv2.0.2/Conv_output_0",
    "/one2one_cv3.0/one2one_cv3.0.2/Conv_output_0",
    "/one2one_cv4_kpts.0/Conv_output_0",
    "/one2one_cv2.1/one2one_cv2.1.2/Conv_output_0",
    "/one2one_cv3.1/one2one_cv3.1.2/Conv_output_0",
    "/one2one_cv4_kpts.1/Conv_output_0",
    "/one2one_cv2.2/one2one_cv2.2.2/Conv_output_0",
    "/one2one_cv3.2/one2one_cv3.2.2/Conv_output_0",
    "/one2one_cv4_kpts.2/Conv_output_0",
]


def _load_20():
    spec = importlib.util.spec_from_file_location(
        "convert20_for_t9", Path(__file__).resolve().parent / "20_convert_rknn.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["convert20_for_t9"] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--method", choices=["channel", "layer"], default="channel")
    ap.add_argument("--algo", default="normal",
                    choices=["normal", "kl_divergence", "mmse"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if not ONNX.exists():
        raise SystemExit(f"missing {ONNX}; run scripts/53_truncate_yolo_t9.py "
                         f"--verify --write on Windows first")

    from rknn.api import RKNN

    c20 = _load_20()
    out = Path(args.out) if args.out else (
        paths.MODELS / f"yolo26n_pose_480x640_int8_t9_{args.method}.rknn")

    calib = c20.build_yolo_calibration()
    import numpy as np
    paths.CALIB.mkdir(parents=True, exist_ok=True)
    np.save(paths.CALIB / "yolo_calib_t9.npy", calib)
    txt = paths.CALIB / "yolo_calib_t9.txt"
    txt.write_text("\n".join(str(p) for p in
                             sorted((paths.CALIB / "yolo_frames").glob("*.npy")))
                   + "\n", encoding="utf-8")

    rknn = RKNN(verbose=True)
    if rknn.config(mean_values=[[0, 0, 0]], std_values=[[1, 1, 1]],
                   target_platform=TARGET, quantized_method=args.method,
                   quantized_algorithm=args.algo,
                   optimization_level=3) != 0:
        raise SystemExit("config failed")
    if rknn.load_onnx(model=str(ONNX),
                      inputs=["images"],
                      input_size_list=[[1, 3, 480, 640]],
                      outputs=OUTPUTS) != 0:
        raise SystemExit("load_onnx failed")
    if rknn.build(do_quantization=True, dataset=str(txt)) != 0:
        raise SystemExit("build failed")
    if rknn.export_rknn(str(out)) != 0:
        raise SystemExit("export_rknn failed")
    rknn.release()

    rec = {"layout": "t9", "method": args.method, "algo": args.algo,
           "rknn": out.name, "size": out.stat().st_size,
           "sha256": paths.sha256_file(out), "n_outputs": len(OUTPUTS)}
    rep = paths.PARITY / f"g5_t9_build_{args.method}_{args.algo}.json"
    rep.write_text(json.dumps(rec, indent=2), encoding="utf-8")
    print(f"\nwrote {out} ({out.stat().st_size/1e6:.2f} MB)\nwrote {rep}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
