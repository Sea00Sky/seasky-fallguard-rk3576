"""Score the QDQ raw outputs with the same G5 code as every other variant."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def _load_own(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, PROJECT_ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


v25 = _load_own("v25", "scripts/25_verify_truncated.py")
verify21 = v25.verify21
paths = v25.paths
spec_mod = v25.spec_mod


def main() -> int:
    ap = __import__("argparse").ArgumentParser()
    ap.add_argument("--tag", default="int8_t4_qdq")
    a = ap.parse_args()
    tag = a.tag
    cache = paths.REPORTS / "parity" / f"yolo_raw_{tag}.npz"
    ref_cache = paths.REPORTS / "parity" / "yolo_raw_int8_t4_channel.npz"
    if not cache.exists():
        raise SystemExit(f"missing {cache}; run experiments/qat/run_qdq_full.py")
    if not ref_cache.exists():
        raise SystemExit(f"missing {ref_cache} (the ONNX reference side)")

    raw_rknn = np.load(cache, allow_pickle=False)["raw_rknn"]
    ref = np.load(ref_cache, allow_pickle=False)
    raw_onnx = ref["raw_onnx"]
    if len(raw_onnx) != len(raw_rknn):
        raise SystemExit(f"frame count mismatch: onnx {len(raw_onnx)} "
                         f"vs qdq {len(raw_rknn)}")

    from src.deploy import yolo_export
    _, params = yolo_export.preprocess(
        np.zeros((spec_mod.IMGSZ * 3 // 4, spec_mod.IMGSZ, 3), np.uint8),
        imgsz=spec_mod.IMGSZ)

    ious, kpt_l2 = [], []
    count_mismatch = order_swaps = 0
    n = len(raw_onnx)
    for i in range(n):
        ba, ka, _ = yolo_export.decode(raw_onnx[i], params, conf=spec_mod.CONF,
                                       max_persons=spec_mod.MAX_PERSONS)
        bb, kb, _ = yolo_export.decode(raw_rknn[i], params, conf=spec_mod.CONF,
                                       max_persons=spec_mod.MAX_PERSONS)
        if len(ba) != len(bb):
            count_mismatch += 1
            continue
        frame_ious, pairs, swaps = verify21._match_boxes(ba, bb)
        order_swaps += swaps
        ious.extend(frame_ious)
        for ja, jb in pairs:
            kpt_l2.extend(np.linalg.norm(ka[ja] - kb[jb], axis=1).tolist())

    iou_arr = np.asarray(ious, dtype=np.float64)
    l2_arr = np.asarray(kpt_l2, dtype=np.float64)
    med = float(np.median(iou_arr)) if len(iou_arr) else 0.0
    p5 = float(np.percentile(iou_arr, 5)) if len(iou_arr) else 0.0
    l2_max = float(l2_arr.max()) if len(l2_arr) else float("inf")
    frac_1px = float((l2_arr <= 1.0).mean()) if len(l2_arr) else 0.0
    anchor = float((iou_arr >= 0.5).mean()) if len(iou_arr) else 0.0
    diag = float(np.hypot(spec_mod.IMGSZ, spec_mod.IMGSZ * 0.75))
    conf_err = float(np.abs(raw_onnx[:, 0, 4, :] - raw_rknn[:, 0, 4, :]).max())
    kpt_conf_err = float(np.abs(raw_onnx[:, 0, 55, :]
                                - raw_rknn[:, 0, 55, :]).max())
    mismatch_frac = count_mismatch / max(1, n)

    crit = {
        "anchor_ge_0.5": anchor, "anchor_min": verify21.G5_ANCHOR_MIN,
        "iou_median": med, "iou_median_min": verify21.G5_IOU_MEDIAN_INT8,
        "iou_p5": p5, "iou_p5_min": verify21.G5_IOU_P5_INT8,
        "kpt_l2_max_px": l2_max,
        "kpt_l2_max_allowed": verify21.G5_KPT_L2_DIAG_FRAC * diag,
        "kpt_frac_within_1px": frac_1px,
        "conf_abs_err_max": conf_err,
        "kpt_score_abs_err_max": kpt_conf_err,
        "count_mismatch_frames": count_mismatch,
        "count_mismatch_frac": mismatch_frac,
        "count_mismatch_frac_max": verify21.TOL_COUNT_MISMATCH_FRAC,
        "order_swaps": order_swaps,
    }
    passed = (anchor >= verify21.G5_ANCHOR_MIN
              and med >= verify21.G5_IOU_MEDIAN_INT8
              and p5 >= verify21.G5_IOU_P5_MIN
              and l2_max <= verify21.G5_KPT_L2_DIAG_FRAC * diag
              and conf_err <= verify21.TOL_CONF_ABS_FP16
              and mismatch_frac <= verify21.TOL_COUNT_MISMATCH_FRAC)

    print(f"\nG5 yolo {tag}: {'PASS' if passed else 'FAIL'}")
    for k, v in crit.items():
        print(f"    {k:32s} {v}")
    print(f"    conf channel max (rknn)          "
          f"{float(raw_rknn[:, 0, 4, :].max()):.6f}")
    print(f"    kpt score channel max (rknn)     "
          f"{float(raw_rknn[:, 0, 55, :].max()):.6f}")

    rec = {"gate": "G5", "model": "yolo", "precision": "int8_qdq",
           "variant": tag, "layout": "t4",
           "rknn": "models/yolo26n_pose_480x640_int8_t4_qdq.rknn",
           "n_frames": int(n), "criteria": crit, "passed": bool(passed)}
    out = paths.REPORTS / "parity" / f"g5_yolo_{tag}.json"
    out.write_text(json.dumps(rec, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
