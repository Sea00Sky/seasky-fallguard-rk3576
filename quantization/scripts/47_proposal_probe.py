"""Ask the toolkit's own proposal engine which layers to keep in float16."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.deploy import paths  # noqa: E402

TARGET = "rk3576"


def calib_list(n: int = 200) -> Path:
    d = paths.CALIB / "yolo_frames"
    frames = sorted(d.glob("*.npy"))[:n]
    if not frames:
        raise SystemExit(f"no calibration frames in {d}")
    txt = paths.CALIB / "yolo_calib.txt"
    txt.write_text("\n".join(str(f) for f in frames) + "\n", encoding="utf-8")
    return txt


def read_custom_layers(cfg: Path) -> list[tuple[str, str]]:
    """`[(tensor_name, dtype), ...]` from the cfg's custom_quantize_layers block."""
    text = cfg.read_text(encoding="utf-8")
    m = re.search(r"^custom_quantize_layers:\s*$(.*?)(?=^\S)", text,
                  flags=re.M | re.S)
    if not m:
        return []
    out = []
    for line in m.group(1).splitlines():
        mm = re.match(r"^\s+(/\S+):\s*(\S+)\s*$", line)
        if mm:
            out.append((mm.group(1), mm.group(2)))
    return out


def run(mode: str, n_calib: int) -> dict:
    from rknn.api import RKNN

    onnx_path = paths.MODELS / "yolo26n_pose_480x640_t4.onnx"
    if not onnx_path.exists():
        raise SystemExit(f"missing {onnx_path}")

    custom = None
    if mode == "head":
        # `custom_hybrid` names a subgraph by [input, output]. The head starts
        # at the three neck outputs feeding the one2one branch. Naming the
        # *final* graph outputs as the region end is the only form that is
        # unambiguous from outside the toolkit: the four t4 outputs ARE the end
        # of the head.
        custom = [
            ["/one2one_cv2.0/one2one_cv2.0.0/conv/Conv_output_0",
             "/Mul_2_output_0"],
            ["/one2one_cv3.0/one2one_cv3.0.0/one2one_cv3.0.0.0/conv/Conv_output_0",
             "/Sigmoid_output_0"],
            ["/one2one_cv4.0/one2one_cv4.0.0/conv/Conv_output_0",
             "/Mul_3_output_0"],
        ]

    rknn = RKNN(verbose=True)
    rknn.config(mean_values=[[0, 0, 0]], std_values=[[1, 1, 1]],
                target_platform=TARGET, quantized_method="channel",
                optimization_level=3)
    if rknn.load_onnx(model=str(onnx_path)) != 0:
        raise SystemExit("load_onnx failed")

    print(f"[{mode}] hybrid_quantization_step1 proposal=True "
          f"(dataset {n_calib}) ...", flush=True)
    rc = rknn.hybrid_quantization_step1(
        dataset=str(calib_list(n_calib)),
        proposal=True,
        proposal_dataset_size=min(20, n_calib),
        custom_hybrid=custom,
    )
    print(f"[{mode}] step1 returned {rc}", flush=True)
    if rc != 0:
        raise SystemExit(f"step1 failed with {rc}")

    stem = onnx_path.stem
    cfg = Path(f"{stem}.quantization.cfg")
    if not cfg.exists():
        raise SystemExit(f"step1 wrote no cfg at {cfg}")

    # Keep a copy per mode: the next invocation overwrites the same filename.
    keep = paths.PARITY / f"proposal_{mode}.quantization.cfg"
    keep.parent.mkdir(parents=True, exist_ok=True)
    keep.write_text(cfg.read_text(encoding="utf-8"), encoding="utf-8")

    chosen = read_custom_layers(cfg)
    print(f"\n[{mode}] custom_quantize_layers entries: {len(chosen)}", flush=True)
    for name, dt in chosen[:40]:
        print(f"    {dt:8s} {name}", flush=True)
    if len(chosen) > 40:
        print(f"    ... {len(chosen) - 40} more", flush=True)

    rknn.release()
    return {"mode": mode, "returncode": rc, "n_chosen": len(chosen),
            "chosen": [{"tensor": n, "dtype": d} for n, d in chosen],
            "cfg_copy": str(keep)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["global", "head"], default="global")
    ap.add_argument("--n-calib", type=int, default=200)
    args = ap.parse_args()

    rec = run(args.mode, args.n_calib)
    out = paths.PARITY / f"g5_proposal_{args.mode}.json"
    out.write_text(json.dumps(rec, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
