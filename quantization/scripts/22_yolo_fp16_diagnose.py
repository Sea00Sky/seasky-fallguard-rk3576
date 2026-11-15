"""Diagnose the G4 YOLO FP16 error structure (does NOT gate anything)."""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _load_own(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, PROJECT_ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


paths = _load_own("deploy_paths", "src/deploy/paths.py")
spec_mod = _load_own("deploy_spec", "src/deploy/spec.py")
# Reuse the gate's own plumbing so this cannot drift from what G4 measures.
verify = _load_own("verify21", "scripts/21_verify_rknn.py")

CACHE = paths.REPORTS / "parity" / "yolo_raw_fp16.npz"


def gather(recompute: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from src.deploy import yolo_export

    rknn_path = verify.RKNN_FILES[("yolo", "fp16")]
    if CACHE.exists() and not recompute:
        d = np.load(CACHE, allow_pickle=False)
        if str(d["rknn_sha256"]) == paths.sha256_file(rknn_path):
            print(f"reusing {CACHE.name} ({len(d['raw_onnx'])} frames)")
            return d["raw_onnx"], d["raw_rknn"], d["frames"]
        print("cache is from a different .rknn; recomputing")

    frames = verify.load_yolo_frames(verify.N_YOLO_FRAMES)
    onnx = verify._onnx_session(verify.ONNX_FILES["yolo"])
    rknn = verify._rknn_session(rknn_path, "yolo", "fp16")
    raw_a, raw_b = [], []
    for i, chw in enumerate(frames):
        raw_a.append(verify._run(onnx, chw[None]).astype(np.float32))
        raw_b.append(verify._run(rknn, chw[None]).astype(np.float32))
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(frames)}", flush=True)
    a = np.asarray(raw_a)
    b = np.asarray(raw_b)
    np.savez_compressed(CACHE, raw_onnx=a, raw_rknn=b, frames=frames,
                        rknn_sha256=np.asarray(paths.sha256_file(rknn_path)))
    print(f"wrote {CACHE}")
    return a, b, frames


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--recompute", action="store_true")
    args = ap.parse_args()

    from src.deploy import yolo_export

    raw_a, raw_b, frames = gather(args.recompute)
    _, params = yolo_export.preprocess(
        np.zeros((spec_mod.IMGSZ * 3 // 4, spec_mod.IMGSZ, 3), np.uint8),
        imgsz=spec_mod.IMGSZ)

    errs, scores, mags = [], [], []
    mismatches = []
    swaps_total = 0
    for i in range(len(raw_a)):
        ba, ka, sa = yolo_export.decode(raw_a[i], params, conf=spec_mod.CONF,
                                        max_persons=spec_mod.MAX_PERSONS)
        bb, kb, sb = yolo_export.decode(raw_b[i], params, conf=spec_mod.CONF,
                                        max_persons=spec_mod.MAX_PERSONS)
        if len(ba) != len(bb):
            # How close to the threshold was the deciding anchor?
            ca = np.sort(raw_a[i][0, 4])
            cb = np.sort(raw_b[i][0, 4])
            near_a = ca[(ca > 0.20) & (ca < 0.30)]
            near_b = cb[(cb > 0.20) & (cb < 0.30)]
            mismatches.append({"frame": i, "n_onnx": len(ba), "n_rknn": len(bb),
                               "onnx_conf_near": near_a.tolist(),
                               "rknn_conf_near": near_b.tolist()})
            continue
        _, pairs, swaps = verify._match_boxes(ba, bb)
        swaps_total += swaps
        for ja, jb in pairs:
            d = np.linalg.norm(ka[ja] - kb[jb], axis=1)
            errs.extend(d.tolist())
            scores.extend(sa[ja].tolist())
            mags.extend(np.linalg.norm(ka[ja], axis=1).tolist())

    errs = np.asarray(errs)
    scores = np.asarray(scores)
    mags = np.asarray(mags)

    print(f"\nframes {len(raw_a)}  keypoints {len(errs)}  "
          f"order swaps {swaps_total}  count mismatches {len(mismatches)}")
    print("\n## 1. error magnitude (px)")
    for q in (50, 90, 95, 99, 99.9):
        print(f"  p{q:<5} {np.percentile(errs, q):.3f}")
    print(f"  max   {errs.max():.3f}")

    print("\n## 2. error vs keypoint confidence")
    print(f"  {'score bin':>12} {'n':>6} {'median':>8} {'p95':>8} {'max':>8}")
    for lo, hi in ((0.0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.01)):
        s = (scores >= lo) & (scores < hi)
        if s.sum():
            print(f"  [{lo:.1f},{hi:.1f})  {int(s.sum()):>6} "
                  f"{np.median(errs[s]):>8.3f} "
                  f"{np.percentile(errs[s], 95):>8.3f} {errs[s].max():>8.3f}")

    print("\n## 3. ULP scale")
    ulp640 = 640 * 2.0 ** -11
    print(f"  FP16 1 ULP at |coord|=640: {ulp640:.3f} px")
    print(f"  keypoints within 1 ULP of the model's scale: "
          f"{float((errs <= ulp640).mean()) * 100:.1f}%")
    print(f"  keypoints within 1 px: {float((errs <= 1.0).mean()) * 100:.2f}%")

    print("\n## 4. count mismatches")
    if not mismatches:
        print("  none")
    for m in mismatches:
        print(f"  frame {m['frame']}: onnx n={m['n_onnx']} rknn n={m['n_rknn']}")
        print(f"    onnx conf in (0.20,0.30): {np.round(m['onnx_conf_near'], 5)}")
        print(f"    rknn conf in (0.20,0.30): {np.round(m['rknn_conf_near'], 5)}")

    out = paths.REPORTS / "parity" / "g4_yolo_diagnosis.json"
    paths.write_json(out, {
        "n_frames": int(len(raw_a)), "n_keypoints": int(len(errs)),
        "kpt_l2": {"median": float(np.median(errs)),
                   "p95": float(np.percentile(errs, 95)),
                   "p99": float(np.percentile(errs, 99)),
                   "max": float(errs.max())},
        "frac_within_1ulp_at_640": float((errs <= ulp640).mean()),
        "frac_within_1px": float((errs <= 1.0).mean()),
        "confidence_order_swaps": swaps_total,
        "count_mismatches": mismatches})
    print(f"\nwritten to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
