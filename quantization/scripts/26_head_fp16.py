"""G5 repair attempt: keep the saturated classification-head convs in float16."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.deploy import paths  # noqa: E402

TARGET = "rk3576"

# The three final convs of the classification head, one per stride. Names come
# from the exported graph (verified by `scripts/24_truncate_yolo.py`'s tensor
# listing and by the accuracy analysis).
CV3_FINAL = [
    "/one2one_cv3.0/one2one_cv3.0.2/Conv_output_0",
    "/one2one_cv3.1/one2one_cv3.1.2/Conv_output_0",
    "/one2one_cv3.2/one2one_cv3.2.2/Conv_output_0",
]
# Their immediate upstream activations: the collapse is at the final conv, but
# the input side of that conv is what the scale is derived against, so a
# variant that protects both sides is included.
CV3_FINAL_ACT = [
    "/one2one_cv3.0/one2one_cv3.0.1/one2one_cv3.0.1.1/act/Mul_output_0",
    "/one2one_cv3.1/one2one_cv3.1.1/one2one_cv3.1.1.1/act/Mul_output_0",
    "/one2one_cv3.2/one2one_cv3.2.1/one2one_cv3.2.1.1/act/Mul_output_0",
]

GROUPS = {
    "conv3": CV3_FINAL,
    "conv3+act": CV3_FINAL + CV3_FINAL_ACT,
}


def calib_list() -> Path:
    d = paths.CALIB / "yolo_frames"
    frames = sorted(d.glob("*.npy"))[:200]
    if not frames:
        raise SystemExit(f"no calibration frames in {d}")
    txt = paths.CALIB / "yolo_calib.txt"
    txt.write_text("\n".join(str(f) for f in frames) + "\n", encoding="utf-8")
    return txt


def run(onnx_path: Path, out: Path, protect: list[str], dtype: str,
        method: str, stem: str) -> None:
    from rknn.api import RKNN

    rknn = RKNN(verbose=False)
    rknn.config(mean_values=[[0, 0, 0]], std_values=[[1, 1, 1]],
                target_platform=TARGET, quantized_method=method,
                optimization_level=3)
    if rknn.load_onnx(model=str(onnx_path)) != 0:
        raise SystemExit("load_onnx failed")

    # Step 1 writes <stem>.quantization.cfg next to the cwd.
    if rknn.hybrid_quantization_step1(dataset=str(calib_list()),
                                      proposal=False) != 0:
        raise SystemExit("hybrid_quantization_step1 failed")

    cfg = Path(f"{stem}.quantization.cfg")
    if not cfg.exists():
        # step1 names its three artefacts after the ONNX file, not after
        # anything passed on the command line. Verified : the build
        # wrote `yolo26n_pose_480x640_t4.{model,data,quantization.cfg}` while
        # this function's `stem` said something else.
        cfg = Path(f"{onnx_path.stem}.quantization.cfg")
    if not cfg.exists():
        raise SystemExit(f"step1 did not write a .quantization.cfg for "
                         f"{onnx_path.stem}")
    stem = onnx_path.stem
    text = cfg.read_text(encoding="utf-8")

    entries = "".join(f"  {name}: {dtype}\n" for name in protect)
    if re.search(r"^custom_quantize_layers:.*$", text, flags=re.M):
        text = re.sub(r"^custom_quantize_layers:.*$",
                      "custom_quantize_layers:\n" + entries.rstrip("\n"),
                      text, flags=re.M)
    else:
        text += "\ncustom_quantize_layers:\n" + entries
    cfg.write_text(text, encoding="utf-8")
    print(f"  protected {len(protect)} layer(s) as {dtype}", flush=True)

    rknn2 = RKNN(verbose=False)
    if rknn2.hybrid_quantization_step2(
            model_input=f"{stem}.model", data_input=f"{stem}.data",
            model_quantization_cfg=str(cfg)) != 0:
        raise SystemExit("hybrid_quantization_step2 failed")
    if rknn2.export_rknn(str(out)) != 0:
        raise SystemExit("export_rknn failed")
    rknn2.release()
    rknn.release()


def dump_table(rknn_path: Path) -> dict:
    blob = rknn_path.read_bytes()
    pat = re.compile(
        rb"'([^']+)': \{'name': '[^']*', 'dtype': '([^']+)', 'qmethod': '([^']+)',"
        rb" 'qtype': '([^']+)', 'min': \[([^\]]*)\], 'max': \[([^\]]*)\]")
    table: dict[str, dict] = {}
    for name, dt, qm, qt, lo, hi in pat.findall(blob):
        n = name.decode()
        if n in table:
            continue
        try:
            lo_f = float(lo.decode().strip().rstrip(","))
            hi_f = float(hi.decode().strip().rstrip(","))
        except ValueError:
            continue
        table[n] = {"dtype": dt.decode(), "qmethod": qm.decode(),
                    "qtype": qt.decode(), "min": lo_f, "max": hi_f,
                    "span": hi_f - lo_f}
    return table


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layout", choices=["t3", "t4"], default="t4")
    ap.add_argument("--protect", choices=sorted(GROUPS), default="conv3")
    ap.add_argument("--dtype", choices=["float16", "int16"], default="float16")
    ap.add_argument("--method", choices=["channel", "layer"], default="channel")
    args = ap.parse_args()

    onnx_path = paths.MODELS / f"yolo26n_pose_480x640_{args.layout}.onnx"
    if not onnx_path.exists():
        raise SystemExit(f"missing {onnx_path}")

    tag = f"int8_{args.layout}_{args.protect}_{args.dtype}"
    out = paths.MODELS / f"yolo26n_pose_480x640_{tag}.rknn"
    stem = f"yolo26n_pose_480x640_{tag}"

    print(f"building {out.name} (protect={args.protect} as {args.dtype})",
          flush=True)
    run(onnx_path, out, GROUPS[args.protect], args.dtype, args.method, stem)

    table = dump_table(out)
    rec = {"layout": args.layout, "protect": args.protect, "dtype": args.dtype,
           "method": args.method, "rknn": out.name,
           "sha256": paths.sha256_file(out), "size": out.stat().st_size,
           "protected": GROUPS[args.protect], "quant_table": table}
    rep = paths.REPORTS / "parity" / f"g5_quant_{tag}.json"
    rep.parent.mkdir(parents=True, exist_ok=True)
    rep.write_text(json.dumps(rec, indent=2), encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size/1e6:.2f} MB)\nwrote {rep}",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
