"""Convert a truncated YOLO26n-pose ONNX to INT8 RKNN and inspect its scales."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.deploy import paths  # noqa: E402

N_CALIB = 200          # same count the shipped INT8 build used
TARGET = "rk3576"


def calib_list(dense: bool = False) -> Path:
    """The toolkit wants a text file of `.npy` paths, one per line."""
    if dense:
        txt = paths.CALIB / "yolo_calib_dense.txt"
        if not txt.exists():
            raise SystemExit(f"missing {txt}; run scripts/27_dense_calibration.py")
        n = len(txt.read_text(encoding="utf-8").split())
        if n == 0:
            raise SystemExit(f"{txt} is empty")
        print(f"  calibration: dense set, {n} frames", flush=True)
        return txt

    d = paths.CALIB / "yolo_frames"
    frames = sorted(d.glob("*.npy"))[:N_CALIB]
    if not frames:
        raise SystemExit(f"no calibration frames in {d}; run scripts/20 first")
    txt = paths.CALIB / "yolo_calib.txt"
    txt.write_text("\n".join(str(f) for f in frames) + "\n", encoding="utf-8")
    return txt


def build(onnx_path: Path, out: Path, method: str, algo: str,
          quant_dtype: str = "w8a8", calib_txt: Path | None = None,
          hybrid_level: int = 0) -> None:
    from rknn.api import RKNN

    rknn = RKNN(verbose=True)
    rknn.config(mean_values=[[0, 0, 0]], std_values=[[1, 1, 1]],
                target_platform=TARGET,
                quantized_dtype=quant_dtype,
                quantized_method=method,
                quantized_algorithm=algo,
                quantized_hybrid_level=hybrid_level,
                optimization_level=3)
    # `outputs=` must NOT be passed: the graph is already truncated, and naming
    # the outputs here would re-declare them and drop the intermediate cut.
    ret = rknn.load_onnx(model=str(onnx_path))
    if ret != 0:
        raise SystemExit(f"load_onnx failed ({ret})")
    if calib_txt is None:
        calib_txt = calib_list()
    ret = rknn.build(do_quantization=True, dataset=str(calib_txt))
    if ret != 0:
        raise SystemExit(f"build failed ({ret})")
    ret = rknn.export_rknn(str(out))
    if ret != 0:
        raise SystemExit(f"export_rknn failed ({ret})")
    rknn.release()


_RANGE_RE = re.compile(
    rb"'([^']+)': \{'name': '[^']*', 'dtype': '([^']+)', 'qmethod': '([^']+)',"
    rb" 'qtype': '([^']+)', 'min': \[([^\]]*)\], 'max': \[([^\]]*)\]")


def dump_quant_table(rknn_path: Path) -> dict:
    """Pull each tensor's calibration range out of the `.rknn` container."""
    blob = rknn_path.read_bytes()
    table: dict[str, dict] = {}
    for name, dtype, qmethod, qtype, lo, hi in _RANGE_RE.findall(blob):
        n = name.decode()
        if n in table:                     # keep the first (input) occurrence
            continue
        try:
            lo_f = float(lo.decode().strip().rstrip(","))
            hi_f = float(hi.decode().strip().rstrip(","))
        except ValueError:
            continue
        table[n] = {"dtype": dtype.decode(), "qmethod": qmethod.decode(),
                    "qtype": qtype.decode(), "min": lo_f, "max": hi_f,
                    "span": hi_f - lo_f}
    return table


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layout", choices=["t3", "t4"], required=True)
    ap.add_argument("--method", choices=["channel", "layer"], default="channel")
    ap.add_argument("--algo", choices=["normal", "kl_divergence", "mmse"],
                    default="normal")
    ap.add_argument("--quant-dtype", choices=["w8a8", "w8a16", "w16a16i"],
                    default="w8a8")
    ap.add_argument("--calib", choices=["base", "dense"], default="base",
                    help="base = shipped set (1 frame/video at n//2); "
                         "dense = scripts/27's set (4 frames/video spread)")
    ap.add_argument("--hybrid-level", type=int, default=0,
                    help="quantized_hybrid_level; 0 = off (default). "
                         "Undocumented toolkit knob, level 1 = auto mixed "
                         "precision. A different mechanism from scripts/26.")
    args = ap.parse_args()

    onnx_path = paths.MODELS / f"yolo26n_pose_480x640_{args.layout}.onnx"
    if not onnx_path.exists():
        raise SystemExit(f"missing {onnx_path}; run scripts/24_truncate_yolo.py "
                         f"--layout {args.layout} first")

    tag = f"int8_{args.layout}_{args.method}"
    if args.algo != "normal":
        tag += f"_{args.algo}"
    if args.quant_dtype != "w8a8":
        tag += f"_{args.quant_dtype}"
    if args.calib != "base":
        tag += f"_{args.calib}"
    if args.hybrid_level:
        tag += f"_hyb{args.hybrid_level}"
    out = paths.MODELS / f"yolo26n_pose_480x640_{tag}.rknn"

    calib_txt = calib_list(dense=(args.calib == "dense"))
    print(f"building {out.name} from {onnx_path.name}", flush=True)
    build(onnx_path, out, args.method, args.algo, args.quant_dtype,
          calib_txt=calib_txt, hybrid_level=args.hybrid_level)

    table = dump_quant_table(out)
    print(f"\n=== quantisation ranges in {out.name} ===", flush=True)
    for name, row in table.items():
        if "output" in name or name.startswith("/"):
            print(f"  {name:26s} {row['dtype']:8s} {row['qmethod']:8s} "
                  f"{row['qtype']:5s} [{row['min']:9.4f}, {row['max']:9.4f}] "
                  f"span {row['span']:.4f}", flush=True)

    rec = {"layout": args.layout, "method": args.method, "algo": args.algo,
           "rknn": out.name, "sha256": paths.sha256_file(out),
           "size": out.stat().st_size, "quant_table": table}
    rep = paths.REPORTS / "parity" / f"g5_quant_{tag}.json"
    rep.parent.mkdir(parents=True, exist_ok=True)
    rep.write_text(json.dumps(rec, indent=2), encoding="utf-8")
    print(f"\nwrote {rep}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
