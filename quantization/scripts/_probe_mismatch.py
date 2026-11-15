"""Throwaway: characterise the 3 mismatched frames of the t9 w16a16i run."""
import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


verify21 = load("verify21", "scripts/21_verify_rknn.py")
paths = verify21.paths
spec_mod = verify21.spec_mod

npz = sys.argv[1] if len(sys.argv) > 1 else "yolo_raw__probe_t9_w16a16i.npz"
d = np.load(paths.REPORTS / "parity" / npz)
print("npz:", npz)
raw_onnx, raw_rknn = d["raw_onnx"], d["raw_rknn"]
print("frames", raw_onnx.shape, raw_rknn.shape)

from src.deploy import yolo_export

frame = np.zeros((spec_mod.IMGSZ * 3 // 4, spec_mod.IMGSZ, 3), np.uint8)
_, params = yolo_export.preprocess(frame, imgsz=spec_mod.IMGSZ)

thr = spec_mod.CONF
margins = []
for i in range(raw_onnx.shape[0]):
    ba, _, _ = yolo_export.decode(raw_onnx[i], params, conf=thr,
                                  max_persons=spec_mod.MAX_PERSONS)
    bb, _, _ = yolo_export.decode(raw_rknn[i], params, conf=thr,
                                  max_persons=spec_mod.MAX_PERSONS)
    if len(ba) == len(bb):
        continue
    ca = np.asarray(raw_onnx[i])[0, 4]
    cb = np.asarray(raw_rknn[i])[0, 4]
    fa = np.flatnonzero(ca >= thr)
    fb = np.flatnonzero(cb >= thr)
    only_a = np.setdiff1d(fa, fb)
    only_b = np.setdiff1d(fb, fa)
    print(f"frame {i:3d}: {len(ba)} vs {len(bb)}  "
          f"only_onnx={len(only_a)} only_rknn={len(only_b)}")
    for idx in list(only_a) + list(only_b):
        ma = abs(float(ca[idx]) - thr)
        mb = abs(float(cb[idx]) - thr)
        margins.append(min(ma, mb))
        print(f"    anchor {idx:6d}  onnx_conf={ca[idx]:.6f} "
              f"rknn_conf={cb[idx]:.6f}  min|conf-thr|={min(ma, mb):.6f}")

m = np.asarray(margins) if margins else np.zeros(0)
print()
if m.size:
    print(f"flipped anchors       : {m.size}")
    print(f"min|conf-thr| median  : {np.median(m):.6f}")
    print(f"min|conf-thr| max     : {m.max():.6f}")
    print(f"within 0.001          : {int((m <= 0.001).sum())} / {m.size}")
    print(f"within 0.01           : {int((m <= 0.01).sum())} / {m.size}")
