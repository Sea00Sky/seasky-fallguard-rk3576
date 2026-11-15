"""Route 2 for multi-person INT8: keep the ENTIRE detection head in float16."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.deploy import paths  # noqa: E402

TARGET = "rk3576"


def head_layers(cfg_path: Path) -> list[str]:
    """Every `one2one_*` activation tensor name from step1's cfg."""
    import re

    text = cfg_path.read_text(encoding="utf-8")
    # Keys of `quantize_parameters` are indented 4 spaces, unlike the 2-space
    # entries under `custom_quantize_layers`.
    names = re.findall(r"^    (/one2one_[^\s:]+):$", text, flags=re.M)
    convs = [n for n in names if n.endswith("Conv_output_0")]
    if len(convs) < 30:
        raise SystemExit(
            f"only {len(convs)} one2one Conv_output_0 tensors in {cfg_path.name}; "
            f"the head naming may have changed, refusing to protect a partial "
            f"set (found {len(names)} one2one tensors in total)")
    return convs


def calib_list() -> Path:
    d = paths.CALIB / "yolo_frames"
    frames = sorted(d.glob("*.npy"))[:200]
    if not frames:
        raise SystemExit(f"no calibration frames in {d}")
    txt = paths.CALIB / "yolo_calib.txt"
    txt.write_text("\n".join(str(f) for f in frames) + "\n", encoding="utf-8")
    return txt


def build(onnx_path: Path, out: Path, dtype: str, method: str) -> int:
    """Run step1, read the real tensor names out of its cfg, patch, run step2."""
    import re

    from rknn.api import RKNN

    rknn = RKNN(verbose=False)
    rknn.config(mean_values=[[0, 0, 0]], std_values=[[1, 1, 1]],
                target_platform=TARGET, quantized_method=method,
                optimization_level=3)
    if rknn.load_onnx(model=str(onnx_path)) != 0:
        raise SystemExit("load_onnx failed")

    calib = calib_list()
    n_calib = len(calib.read_text(encoding="utf-8").split())
    print(f"  hybrid_quantization_step1 over {n_calib} frames", flush=True)
    if rknn.hybrid_quantization_step1(dataset=str(calib),
                                      proposal=False) != 0:
        raise SystemExit("hybrid_quantization_step1 failed")

    # step1 names its artefacts after the ONNX file, not after `stem`
    # (measured , documented in scripts/26).
    stem = onnx_path.stem
    cfg = Path(f"{stem}.quantization.cfg")
    if not cfg.exists():
        raise SystemExit(f"step1 did not write {cfg}")

    protect = head_layers(cfg)
    print(f"  head conv tensors to protect: {len(protect)}", flush=True)

    text = cfg.read_text(encoding="utf-8")
    entries = "".join(f"  {name}: {dtype}\n" for name in protect)
    if re.search(r"^custom_quantize_layers:.*$", text, flags=re.M):
        text = re.sub(r"^custom_quantize_layers:.*$",
                      "custom_quantize_layers:\n" + entries.rstrip("\n"),
                      text, flags=re.M)
    else:
        text += "\ncustom_quantize_layers:\n" + entries
    cfg.write_text(text, encoding="utf-8")
    print(f"  protected {len(protect)} head layer(s) as {dtype}", flush=True)

    rknn2 = RKNN(verbose=False)
    if rknn2.hybrid_quantization_step2(
            model_input=f"{stem}.model", data_input=f"{stem}.data",
            model_quantization_cfg=str(cfg)) != 0:
        raise SystemExit("hybrid_quantization_step2 failed")
    if rknn2.export_rknn(str(out)) != 0:
        raise SystemExit("export_rknn failed")
    rknn2.release()
    rknn.release()
    return len(protect)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--layout", choices=["t4"], default="t4")
    ap.add_argument("--dtype", choices=["float16", "int16"], default="float16")
    ap.add_argument("--method", choices=["channel", "layer"], default="channel")
    args = ap.parse_args()

    onnx_path = paths.MODELS / f"yolo26n_pose_480x640_{args.layout}.onnx"
    if not onnx_path.exists():
        raise SystemExit(f"missing {onnx_path}")

    tag = f"int8_{args.layout}_headall_{args.dtype}"
    out = paths.MODELS / f"yolo26n_pose_480x640_{tag}.rknn"

    print(f"building {out.name}", flush=True)
    n_protected = build(onnx_path, out, args.dtype, args.method)

    rec = {"layout": args.layout, "protect": "head_all", "dtype": args.dtype,
           "method": args.method, "rknn": out.name,
           "sha256": paths.sha256_file(out), "size": out.stat().st_size,
           "n_protected": n_protected}
    rep = paths.REPORTS / "parity" / f"g5_quant_{tag}.json"
    rep.parent.mkdir(parents=True, exist_ok=True)
    rep.write_text(json.dumps(rec, indent=2), encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size/1e6:.2f} MB)\nwrote {rep}",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
