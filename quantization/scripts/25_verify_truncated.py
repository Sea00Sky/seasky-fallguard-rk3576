"""G5 verification for the truncated YOLO26n-pose layouts (t3 / t4)."""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
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


verify21 = _load_own("verify21", "scripts/21_verify_rknn.py")
paths = verify21.paths
spec_mod = verify21.spec_mod

# Same tensor names as scripts/24_truncate_yolo.py. Kept here as data rather
# than imported so this script can verify an artefact built elsewhere.
LAYOUTS: dict[str, list[str]] = {
    "t3": ["/Mul_2_output_0", "/Sigmoid_output_0", "/Reshape_10_output_0"],
    "t4": ["/Mul_2_output_0", "/Sigmoid_output_0",
           "/Mul_3_output_0", "/Sigmoid_1_output_0"],
    # t9 = the official rknn3-model-zoo yolo26_pose cut: nine raw head convs,
    # one per (scale, branch), upstream of the Concat that t4 cuts downstream
    # of. Order is (box, score, kpt) x (stride 8, 16, 32), same as convert.py.
    "t9": [
        "/one2one_cv2.0/one2one_cv2.0.2/Conv_output_0",
        "/one2one_cv3.0/one2one_cv3.0.2/Conv_output_0",
        "/one2one_cv4_kpts.0/Conv_output_0",
        "/one2one_cv2.1/one2one_cv2.1.2/Conv_output_0",
        "/one2one_cv3.1/one2one_cv3.1.2/Conv_output_0",
        "/one2one_cv4_kpts.1/Conv_output_0",
        "/one2one_cv2.2/one2one_cv2.2.2/Conv_output_0",
        "/one2one_cv3.2/one2one_cv3.2.2/Conv_output_0",
        "/one2one_cv4_kpts.2/Conv_output_0",
    ],
}

# Must stay in sync with scripts/26_head_fp16.py GROUPS. Duplicated rather than
# imported because importing `26_head_fp16` executes its argparse-free module
# top level only -- but it also pulls the rknn API at import time, which this
# script only needs inside build_sim.
CV3_FINAL = [
    "/one2one_cv3.0/one2one_cv3.0.2/Conv_output_0",
    "/one2one_cv3.1/one2one_cv3.1.2/Conv_output_0",
    "/one2one_cv3.2/one2one_cv3.2.2/Conv_output_0",
]
PROTECT_GROUPS = {
    "conv3": CV3_FINAL,
    "conv3+act": CV3_FINAL + [
        "/one2one_cv3.0/one2one_cv3.0.1/one2one_cv3.0.1.1/act/Mul_output_0",
        "/one2one_cv3.1/one2one_cv3.1.1/one2one_cv3.1.1.1/act/Mul_output_0",
        "/one2one_cv3.2/one2one_cv3.2.1/one2one_cv3.2.1.1/act/Mul_output_0",
    ],
}


def _head_all() -> list[str]:
    """All `one2one_*` conv output tensor names -- the whole-head variant."""
    import onnx

    src = verify21.paths.MODELS / "yolo26n_pose_480x640_t4.onnx"
    if not src.exists():
        raise SystemExit(f"missing {src}; needed to derive the head layer list")
    m = onnx.load(str(src))
    names = [n.name + "_output_0" for n in m.graph.node
             if n.name and n.op_type in ("Conv", "MatMul")
             and "one2one" in n.name]
    if len(names) < 30:
        raise SystemExit(f"only {len(names)} one2one convs in {src.name}")
    return names


PROTECT_GROUPS["head_all"] = None      # filled lazily; see `--protect` below


# t9 head geometry: (stride, grid_h, grid_w) for 480x640 input.
_T9_SCALES = ((8, 60, 80), (16, 30, 40), (32, 15, 20))


def _t9_grid(h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
    """Pixel-centre coordinates of a (h, w) feature map, flattened row-major."""
    ys, xs = np.meshgrid(np.arange(h, dtype=np.float32),
                         np.arange(w, dtype=np.float32), indexing="ij")
    return xs.reshape(-1) + 0.5, ys.reshape(-1) + 0.5


def _t9_rebuild(outs: list[np.ndarray]) -> np.ndarray:
    """Nine raw head tensors -> `(1, 56, 6300)`."""
    boxes, confs, kpts = [], [], []
    for i, (stride, h, w) in enumerate(_T9_SCALES):
        b = np.asarray(outs[i * 3]).reshape(1, 4, -1)[0]
        c = np.asarray(outs[i * 3 + 1]).reshape(1, 1, -1)[0]
        k = np.asarray(outs[i * 3 + 2]).reshape(1, 17, 3, -1)[0]
        cx, cy = _t9_grid(h, w)
        boxes.append(np.stack([(cx - b[0]) * stride, (cy - b[1]) * stride,
                               (cx + b[2]) * stride, (cy + b[3]) * stride]))
        confs.append(1.0 / (1.0 + np.exp(-c[0])))
        kpts.append(np.stack([(cx + k[:, 0]) * stride, (cy + k[:, 1]) * stride,
                              1.0 / (1.0 + np.exp(-k[:, 2]))], axis=1))
    box = np.concatenate(boxes, axis=-1)[None].astype(np.float32)
    conf = np.concatenate(confs, axis=-1)[None, None].astype(np.float32)
    kpt = np.concatenate(kpts, axis=-1).reshape(1, 51, -1).astype(np.float32)
    return np.concatenate([box, conf, kpt], axis=1)


def reassemble(outs: list[np.ndarray], layout: str) -> np.ndarray:
    """RKNN's per-output arrays -> the `(1, 56, N)` tensor `decode` expects."""
    if layout == "t9":
        return _t9_rebuild(outs)
    if layout == "t3":
        box, conf, kpt = outs
        return np.concatenate([box, conf, kpt], axis=1).astype(np.float32)
    box, conf, xy, sc = outs
    # /Mul_3 is (1,17,2,N) and /Sigmoid_1 is (1,17,1,N); Concat_4 joined them on
    # axis 2, so undo that, then the (1,51,N) Reshape.
    kpt = np.concatenate([xy, sc], axis=2).reshape(1, 51, -1)
    return np.concatenate([box, conf, kpt], axis=1).astype(np.float32)


def build_sim(onnx_path: Path, layout: str, method: str, algo: str,
              calib_txt: Path, protected: list[str] | None = None,
              hybrid_level: int = 0, dtype: str = "w8a8"):
    """Simulator session for the truncated graph, same config as the build."""
    from rknn.api import RKNN

    rknn = RKNN(verbose=False)
    # `hybrid_level` must mirror the build. `quantized_hybrid_level=1` makes the
    # toolkit run its own `HybridProposal` pass and pick layers to promote;
    # leaving it at 0 here would simulate a *different* graph than the artefact
    # under test -- the exact mistake L-116 records for `protected`.
    rknn.config(mean_values=[[0, 0, 0]], std_values=[[1, 1, 1]],
                target_platform="rk3576", quantized_method=method,
                quantized_algorithm=algo, quantized_hybrid_level=hybrid_level,
                quantized_dtype=dtype, optimization_level=3)
    ret = rknn.load_onnx(model=str(onnx_path))
    if ret != 0:
        raise RuntimeError(f"load_onnx failed ({ret})")

    if protected:
        import os
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            prev = Path.cwd()
            try:
                os.chdir(td)
                if rknn.hybrid_quantization_step1(
                        dataset=str(calib_txt)) != 0:
                    raise RuntimeError("hybrid_quantization_step1 failed")
                cfg = Path(td) / f"{onnx_path.stem}.quantization.cfg"
                text = cfg.read_text(encoding="utf-8")
                entries = "".join(f"  {n}: float16\n" for n in protected)
                text = re.sub(r"^custom_quantize_layers:.*$",
                              "custom_quantize_layers:\n" + entries.rstrip(),
                              text, flags=re.M)
                cfg.write_text(text, encoding="utf-8")
                if rknn.hybrid_quantization_step2(
                        model_input=str(Path(td) / f"{onnx_path.stem}.model"),
                        data_input=str(Path(td) / f"{onnx_path.stem}.data"),
                        model_quantization_cfg=str(cfg)) != 0:
                    raise RuntimeError("hybrid_quantization_step2 failed")
            finally:
                os.chdir(prev)
        if rknn.init_runtime(target=None) != 0:
            raise RuntimeError("init_runtime(simulator) failed")
        return rknn

    if rknn.build(do_quantization=True, dataset=str(calib_txt)) != 0:
        raise RuntimeError("build(int8) failed")
    if rknn.init_runtime(target=None) != 0:
        raise RuntimeError("init_runtime(simulator) failed")
    return rknn


def build_and_export(onnx_path: Path, out: Path, method: str, algo: str,
                     calib_txt: Path, hybrid_level: int = 0,
                     dtype: str = "w8a8"):
    """Build once, export the `.rknn`, and return the same session for inference."""
    from rknn.api import RKNN

    rknn = RKNN(verbose=False)
    rknn.config(mean_values=[[0, 0, 0]], std_values=[[1, 1, 1]],
                target_platform="rk3576", quantized_method=method,
                quantized_algorithm=algo, quantized_hybrid_level=hybrid_level,
                quantized_dtype=dtype, optimization_level=3)
    if rknn.load_onnx(model=str(onnx_path)) != 0:
        raise RuntimeError("load_onnx failed")
    if rknn.build(do_quantization=True, dataset=str(calib_txt)) != 0:
        raise RuntimeError("build(int8) failed")
    if rknn.export_rknn(str(out)) != 0:
        raise RuntimeError("export_rknn failed")
    if rknn.init_runtime(target=None) != 0:
        raise RuntimeError("init_runtime(simulator) failed")
    return rknn


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layout", choices=sorted(LAYOUTS))
    ap.add_argument("--method", choices=["channel", "layer"], default="channel")
    ap.add_argument("--algo", choices=["normal", "kl_divergence", "mmse"],
                    default=None,
                    help="quantisation algorithm used by the build; default is "
                         "inferred from the artefact's filename tag")
    ap.add_argument("--rknn", help="verify this artefact instead of a tag-derived one")
    ap.add_argument("--n", type=int, default=verify21.N_YOLO_FRAMES)
    ap.add_argument("--recompute", action="store_true")
    ap.add_argument("--build-once", action="store_true",
                    help="build AND infer in one process (needed for mmse, "
                         "whose simulator search peaks at ~6.8 GB and cannot "
                         "be repeated; the artefact is written by this run)")
    ap.add_argument("--protect", choices=["conv3", "conv3+act", "head_all"],
                    help="the artefact was built by scripts/26 (or 46, for "
                         "head_all) with these layers held in fp16; reproduce "
                         "that conversion instead of a plain build")
    ap.add_argument("--calib", choices=["base", "dense"], default=None,
                    help="calibration list used by the build; default is "
                         "inferred from the artefact's filename tag")
    ap.add_argument("--hybrid-level", type=int, default=None,
                    help="quantized_hybrid_level used by the build; default is "
                         "inferred from the artefact's `_hybN` tag")
    ap.add_argument("--dtype", default=None,
                    choices=["w8a8", "w8a16", "w16a16i", "w16a16i_dfp"],
                    help="quantized_dtype used by the build; default is "
                         "inferred from the artefact's `_w16a16i`-style tag "
                         "(absent -> w8a8). Reproducing this matters for the "
                         "same reason as --protect: a w8a8 simulator run "
                         "against a w16a16i artefact verifies a different "
                         "graph and would report numbers that look plausible.")
    args = ap.parse_args()

    # Every setting the build used has to be reproduced here or the simulator
    # verifies a *different* graph. `--rknn` used to infer only `layout`, so
    # verifying `..._mmse.rknn` silently rebuilt with `normal` and returned
    # numbers byte-identical to the plain INT8 model ( ). The algo is
    # now inferred from the filename tag the same way `hybrid_level` is.
    if args.rknn:
        rknn_path = Path(args.rknn)
        stem = rknn_path.stem
        layout = args.layout or next((k for k in LAYOUTS if f"_{k}_" in stem), None)
        if layout is None:
            raise SystemExit("cannot infer --layout from the filename; pass it")
        algo = args.algo
        if algo is None:
            algo = next((a for a in ("kl_divergence", "mmse")
                         if f"_{a}" in stem), "normal")
    else:
        if not args.layout:
            raise SystemExit("pass --layout (or --rknn)")
        layout = args.layout
        algo = args.algo or "normal"
        tag = f"int8_{layout}_{args.method}"
        if algo != "normal":
            tag += f"_{algo}"
        if args.dtype and args.dtype != "w8a8":
            tag += f"_{args.dtype}"
        if args.calib == "dense":
            # Same class of bug as `--dtype`: `--calib` only chose which list to
            # calibrate with, so a dense build was written over the base one
            # (measured : `..._t9_channel_w16a16i.rknn` was replaced,
            # sha 26431a02 -> 98254fc8, and the 3/228 record lost its artefact).
            tag += "_dense"
        rknn_path = paths.MODELS / f"yolo26n_pose_480x640_{tag}.rknn"

    if not rknn_path.exists() and not args.build_once:
        raise SystemExit(f"missing {rknn_path}")
    onnx_path = paths.MODELS / f"yolo26n_pose_480x640_{layout}.onnx"
    if not onnx_path.exists():
        raise SystemExit(f"missing {onnx_path}; run scripts/24_truncate_yolo.py")
    calib_kind = args.calib or ("dense" if rknn_path.stem.endswith("_dense")
                                else "base")
    calib_txt = paths.CALIB / ("yolo_calib_dense.txt" if calib_kind == "dense"
                               else "yolo_calib.txt")
    if not calib_txt.exists():
        raise SystemExit(f"missing {calib_txt}; run scripts/24_convert_truncated.py")
    print(f"  calibration list: {calib_txt.name} ({calib_kind})", flush=True)

    hybrid_level = args.hybrid_level
    if hybrid_level is None:
        m = re.search(r"_hyb(\d+)$", rknn_path.stem)
        hybrid_level = int(m.group(1)) if m else 0
    if hybrid_level:
        print(f"  quantized_hybrid_level={hybrid_level}", flush=True)
    dtype = args.dtype
    if dtype is None:
        dtype = next((d for d in ("w16a16i_dfp", "w16a16i", "w8a16")
                      if f"_{d}" in rknn_path.stem), "w8a8")
    print(f"  quantized_dtype={dtype}", flush=True)
    print(f"  quantized_algorithm={algo}", flush=True)

    frames = verify21.load_yolo_frames(args.n)
    tag = rknn_path.stem.replace("yolo26n_pose_480x640_", "")
    # The cache is keyed by the artefact's sha256 as well, so a hybrid rebuild
    # cannot silently reuse the plain-INT8 cache. The tag still has to differ
    # or the two artefacts would share one file name.
    cache = paths.REPORTS / "parity" / f"yolo_raw_{tag}.npz"
    # With --build-once the artefact does not exist yet; it is written by this
    # run, so the cache is unconditionally stale and the hash comes after.
    sha = "" if args.build_once else paths.sha256_file(rknn_path)
    if args.build_once and cache.exists():
        cache.unlink()

    cached = None
    if cache.exists() and not args.recompute and not args.build_once:
        d = np.load(cache, allow_pickle=False)
        if len(d["raw_onnx"]) == len(frames) and str(d["rknn_sha256"]) == sha:
            cached = (d["raw_onnx"], d["raw_rknn"])
            print(f"  reusing {cache.name}", flush=True)

    if cached is not None:
        raw_onnx, raw_rknn = cached
    else:
        onnx = verify21._onnx_session(verify21.ONNX_FILES["yolo"])
        protected = None
        if args.protect:
            protected = (_head_all() if args.protect == "head_all"
                         else PROTECT_GROUPS[args.protect])
            print(f"  reproducing hybrid conversion, fp16: "
                  f"{len(protected)} layer(s)", flush=True)
        if args.build_once:
            print(f"  building {rknn_path.name} and inferring in one process "
                  f"(one {algo} search)", flush=True)
            rknn = build_and_export(onnx_path, rknn_path, args.method, algo,
                                    calib_txt, hybrid_level=hybrid_level,
                                    dtype=dtype)
            sha = paths.sha256_file(rknn_path)
        else:
            rknn = build_sim(onnx_path, layout, args.method, algo, calib_txt,
                             protected=protected, hybrid_level=hybrid_level,
                             dtype=dtype)
        ra, rb = [], []
        for i, chw in enumerate(frames):
            ra.append(verify21._run(onnx, chw[None]).astype(np.float32))
            outs = rknn.inference(inputs=[chw[None]], data_format="nchw")
            if outs is None:
                raise RuntimeError("RKNN inference returned None -- see toolkit log")
            rb.append(reassemble([np.asarray(o) for o in outs], layout))
            if (i + 1) % 50 == 0:
                print(f"  yolo {i + 1}/{len(frames)}", flush=True)
        raw_onnx, raw_rknn = np.asarray(ra), np.asarray(rb)
        np.savez_compressed(cache, raw_onnx=raw_onnx, raw_rknn=raw_rknn,
                            rknn_sha256=np.asarray(sha))
        print(f"  wrote {cache}", flush=True)

    # ---- identical scoring to scripts/21.verify_yolo --------------------
    from src.deploy import yolo_export

    _, params = yolo_export.preprocess(
        np.zeros((spec_mod.IMGSZ * 3 // 4, spec_mod.IMGSZ, 3), np.uint8),
        imgsz=spec_mod.IMGSZ)

    ious: list[float] = []
    kpt_l2: list[float] = []
    count_mismatch = 0
    order_swaps = 0
    n_frames = len(raw_onnx)
    for i in range(n_frames):
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
    kpt_conf_err = float(np.abs(raw_onnx[:, 0, 55, :] - raw_rknn[:, 0, 55, :]).max())
    mismatch_frac = count_mismatch / max(1, n_frames)

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
              and p5 >= verify21.G5_IOU_P5_INT8
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

    rec = {"gate": "G5", "model": "yolo", "precision": "int8",
           "variant": tag, "layout": layout, "rknn": str(rknn_path),
           "rknn_sha256": sha, "n_frames": int(n_frames),
           "criteria": crit, "passed": bool(passed)}
    out = paths.REPORTS / "parity" / f"g5_yolo_{tag}.json"
    out.write_text(json.dumps(rec, indent=2), encoding="utf-8")
    print(f"\nwrote {out}", flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
