"""Compare the toolkit's calibrated ranges under two calibration sets."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.deploy import paths  # noqa: E402

CLIP_FLOOR = -7.0


def step1_ranges(onnx_path: Path, calib_txt: Path) -> dict[str, dict]:
    """Run step1 in an isolated cwd and return {layer: {min, max}}."""
    from rknn.api import RKNN

    with tempfile.TemporaryDirectory() as td:
        rknn = RKNN(verbose=False)
        rknn.config(mean_values=[[0, 0, 0]], std_values=[[1, 1, 1]],
                    target_platform="rk3576", quantized_method="channel",
                    optimization_level=3)
        if rknn.load_onnx(model=str(onnx_path)) != 0:
            raise SystemExit("load_onnx failed")
        import os
        prev = Path.cwd()
        try:
            os.chdir(td)
            if rknn.hybrid_quantization_step1(dataset=str(calib_txt),
                                              proposal=False) != 0:
                raise SystemExit("hybrid_quantization_step1 failed")
        finally:
            os.chdir(prev)
        rknn.release()
        cfg = Path(td) / f"{onnx_path.stem}.quantization.cfg"
        if not cfg.exists():
            raise SystemExit(f"step1 wrote no cfg in {td}")
        doc = yaml.safe_load(cfg.read_text(encoding="utf-8"))
        # Keep the cfg for the audit trail (outside the temp dir).
        keep = paths.REPORTS / "parity" / f"quantcfg_{calib_txt.stem}.yaml"
        keep.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(cfg, keep)

    out: dict[str, dict] = {}
    for name, spec in (doc.get("quantize_parameters") or {}).items():
        if not isinstance(spec, dict):
            continue

        def _lo(v):
            if v is None:
                return None
            return float(min(v)) if isinstance(v, list) else float(v)

        def _hi(v):
            if v is None:
                return None
            return float(max(v)) if isinstance(v, list) else float(v)

        # `min`/`ori_min` are per-channel lists under `qmethod: channel`; the
        # extremes are what a range comparison needs. Taking `max(ori_min)`
        # would read the *upper* end of the minima and report a degenerate
        # range -- the first version of this function did exactly that.
        lo, hi = _lo(spec.get("min")), _hi(spec.get("max"))
        if lo is None or hi is None:
            continue
        row = {"min": lo, "max": hi}
        olo, ohi = _lo(spec.get("ori_min")), _hi(spec.get("ori_max"))
        if olo is not None and ohi is not None:
            # `ori_*` is what the calibrator measured; `min`/`max` are after the
            # toolkit's clip. When they differ, the clip -- not the calibration
            # set -- is what limited the range.
            row["ori_min"] = olo
            row["ori_max"] = ohi
        out[name] = row
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layout", default="t4", choices=["t3", "t4"])
    args = ap.parse_args()

    onnx_path = paths.MODELS / f"yolo26n_pose_480x640_{args.layout}.onnx"
    if not onnx_path.exists():
        raise SystemExit(f"missing {onnx_path}")

    sets = {
        "base": paths.CALIB / "yolo_calib.txt",
        "dense": paths.CALIB / "yolo_calib_dense.txt",
    }
    for tag, p in sets.items():
        if not p.exists():
            raise SystemExit(f"missing {p}")

    got = {}
    for tag, lst in sets.items():
        print(f"step1 with {tag} set ({len(lst.read_text().split())} frames)...",
              flush=True)
        got[tag] = step1_ranges(onnx_path, lst)
        print(f"  {len(got[tag])} layers", flush=True)

    base, dense = got["base"], got["dense"]
    shared = sorted(set(base) & set(dense))
    print(f"\nlayers in both: {len(shared)}")

    b_clip = [n for n in shared if abs(base[n]["min"] - CLIP_FLOOR) < 1e-6]
    d_clip = [n for n in shared if abs(dense[n]["min"] - CLIP_FLOOR) < 1e-6]
    print(f"min == {CLIP_FLOOR}: base {len(b_clip)}, dense {len(d_clip)}")

    # Which layers moved most, by relative change of the span.
    moved = []
    for n in shared:
        b, d = base[n], dense[n]
        bs, ds = b["max"] - b["min"], d["max"] - d["min"]
        if bs <= 0:
            continue
        moved.append((abs(ds - bs) / bs, n, b, d))
    moved.sort(reverse=True)

    print("\n=== 15 largest span changes (dense vs base) ===")
    for rel, n, b, d in moved[:15]:
        print(f"  {rel*100:7.2f}%  {n:56s} "
              f"[{b['min']:9.3f},{b['max']:9.3f}] -> "
              f"[{d['min']:9.3f},{d['max']:9.3f}]")

    head = [n for n in shared if "one2one_cv3" in n and n.endswith("Conv_output_0")]
    print("\n=== classification-head final convs ===")
    print("    (ori = as measured; min/max = after the toolkit clip)")
    for n in sorted(head):
        b, d = base[n], dense[n]
        print(f"  {n}")
        for tag, r in (("base ", b), ("dense", d)):
            ori = (f"ori [{r['ori_min']:9.3f},{r['ori_max']:8.3f}]"
                   if "ori_min" in r else "ori (none)")
            print(f"      {tag}  {ori}  ->  [{r['min']:9.3f},{r['max']:8.3f}]")

    # How many layers does the toolkit clip at all, and does the dense set move
    # the *measured* range for them? This is the honest form of the question:
    # a better set can change `ori_*` and still leave `min` at -7.0.
    print("\n=== layers where the clip bites (ori != min) ===")
    for tag in ("base", "dense"):
        rows = got[tag]
        bit = [n for n in sorted(rows)
               if "ori_min" in rows[n]
               and abs(rows[n]["ori_min"] - rows[n]["min"]) > 1e-6]
        print(f"  {tag}: {len(bit)} layer(s)")
        for n in bit:
            r = rows[n]
            print(f"      {n:52s} ori_min {r['ori_min']:10.3f} "
                  f"-> min {r['min']:8.3f}")

    rec = {"layout": args.layout,
           "clip_floor": CLIP_FLOOR,
           "n_layers": {"base": len(base), "dense": len(dense)},
           "clipped": {"base": sorted(b_clip), "dense": sorted(d_clip)},
           "ranges": {tag: got[tag] for tag in got}}
    out = paths.REPORTS / "parity" / f"calib_compare_{args.layout}.json"
    out.write_text(json.dumps(rec, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
