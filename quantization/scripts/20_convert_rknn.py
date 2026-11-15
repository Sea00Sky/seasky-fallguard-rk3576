"""Phase 2 -- convert the ONNX graphs to RKNN for RK3576."""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT.parent / "YOLOV26n-pose+ST-GCN++"

# `build_yolo_calibration` does `from src.deploy import yolo_export`. Run as a
# script, `sys.path[0]` is `scripts/`, so that import fails with
# `ModuleNotFoundError: No module named 'src'`. `write_calibration_lists()`
# calls it unconditionally, so *every* `--precision int8` run died here before
# reaching the conversion. `scripts/30` does the same insert for the same reason.
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

ONNX = {
    "stgcn": paths.MODELS / "stgcnpp_v2.onnx",
    "yolo": paths.MODELS / "yolo26n_pose_480x640.onnx",
}
TARGET = "rk3576"

# Static input shapes, read from the exported graphs and asserted against them
# at load time (`scripts/20` calls `_assert_onnx_shape` before converting). The
# toolkit needs them spelled out in `load_onnx(input_size_list=...)`; it will
# not infer them from `inputs`.
STGCN_INPUT = (1, 1, 64, 17, 3)      # (N, M, T, V, C), see reference/MANIFEST_source.md
YOLO_INPUT = (1, 3, 480, 640)        # letterbox identity for 640x480 frames
# FP16 first. INT8 only after a measured latency shortfall.
DEFAULT_PRECISION = "fp16"

# Calibration sizes. Not tuned -- these are round numbers large enough to cover
# the input distribution, and the split is what matters more than the count.
N_STGCN_CALIB = 300
N_YOLO_CALIB = 200

# R3 mitigation: keep the pose head's final convolutions in fp16 so the
# confidence channel keeps its own scale instead of sharing one with the box
# coordinates (`scale = 3.008` would round `conf=0.87` to 0.0 -- see the module
# docstring).
#
# These names are derived from the graph, not typed. The first version was
# hand-written from an assumed ultralytics layout (`/one2one_cv{2,3,4}.{0,1,2}.2/Conv`
# + `cvN_kpts.2`); checked against the exported ONNX on , 11 of the
# 12 matched nothing -- the real names nest the branch one level deeper
# (`/one2one_cv2.0/one2one_cv2.0.2/Conv`), and there is one keypoint branch per
# stride, not one shared `cv4_kpts`. A silent `custom_quantize_layers` mismatch
# is exactly the INT8-conf-channel failure this mitigation exists to prevent,
# so `_yolo_head_convs()` walks the graph from `output0` back to the last Conv
# of each of the nine branches and the result is asserted to be non-empty.
#
# The toolkit matches these as substrings of node names, so the trailing
# `/Conv` is what selects the layer rather than its activation.
YOLO_HEAD_PREFIXES = tuple(f"one2one_cv{s}" for s in (2, 3, 4))


def _yolo_head_convs(onnx_path) -> list[str]:
    """Names of the Conv nodes that directly produce the three output blocks."""
    import onnx

    graph = onnx.load(str(onnx_path)).graph
    producer = {out: n for n in graph.node for out in n.output}

    def last_convs(tensor, seen=None, found=None):
        if seen is None:
            seen, found = set(), []
        if tensor in seen or tensor not in producer:
            return found
        seen.add(tensor)
        node = producer[tensor]
        if node.op_type == "Conv":
            if node.name not in found:
                found.append(node.name)
            return found
        for name in node.input:
            last_convs(name, seen, found)
        return found

    out_name = graph.output[0].name
    names = [n for n in last_convs(out_name) if n.lstrip("/").startswith(YOLO_HEAD_PREFIXES)]
    if not names:
        raise RuntimeError(
            f"no one2one head convolutions found in {onnx_path}; the INT8 "
            f"custom_quantize_layers mitigation would silently do nothing")
    return names


# NOTE: there is deliberately no module-level `YOLO_FP16_LAYERS` constant any
# more. A constant invites callers (scripts/21 did) to read a stale copy; both
# call sites now call `_yolo_head_convs(onnx_path)` so the names always come
# from the graph actually being converted.

# The toolkit's `dataset` for INT8 is a text file of image paths, not an array.
# ST-GCN's input is 5-D `(1,1,64,17,3)`, which no image loader can read, so we
# hand the toolkit a `.npy` per sample and a list of those paths. Whether the
# toolkit accepts `.npy` here is unverified (no rknn on this machine); if it
# rejects them, fall back to wrapping each sample as a 3-channel PNG stack and
# re-check G5 -- do not silently switch to noise (that is the old project's bug).
CALIB_EXT = ".npy"

DEFAULT_QUANT_ALGORITHM = "normal"

DEFAULT_QUANT_DTYPE = "w8a8"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=["stgcn", "yolo", "all"], default="all")
    p.add_argument("--precision", choices=["fp16", "int8"], default=DEFAULT_PRECISION)
    p.add_argument("--target", default=TARGET)
    p.add_argument("--quant-dtype", default=DEFAULT_QUANT_DTYPE,
                   choices=["w8a8", "w8a16", "w16a16i"],
                   help="INT8 quantisation scheme. `w8a8` quantises both weights "
                        "and activations; on ST-GCN that rounds the 2-class "
                        "logits onto a 0.078 grid, which is coarser than the "
                        "decision band (D-121). `w8a16` keeps activations in "
                        "fp16 and only quantises weights.")
    p.add_argument("--quant-algorithm", default=DEFAULT_QUANT_ALGORITHM,
                   choices=["normal", "mmse", "kl_divergence"],
                   help="toolkit's INT8 calibration algorithm. `normal` is the "
                        "toolkit default but leaves a systematic margin "
                        "compression on ST-GCN (slope 0.892, offset -0.370, see "
                        "D-120); `mmse` minimises that error directly.")
    p.add_argument("--hybrid", action="store_true",
                   help="keep the model's graph OUTPUT in fp16 via the "
                        "toolkit's two-step hybrid quantisation. MEASURED "
                        "INEFFECTIVE for ST-GCN (D-125): the protection is "
                        "accepted and step2 reports a dtype change, but the "
                        "output grid stays at 0.078125 and the decision "
                        "agreement is identical to plain INT8 to 16 digits. "
                        "Kept because it is the only API that can express the "
                        "request, and because YOLO's failure mechanism differs "
                        "(shared Concat scale) so it may still help there. "
                        "Off by default so plain INT8 stays the baseline that "
                        "the gate record refers to.")
    p.add_argument("--verify", action="store_true",
                   help="run the RKNN simulator against the ONNX reference")
    p.add_argument("--stgcn-batch", type=int, default=1, metavar="N",
                   help="PLAN §3.3: build ST-GCN with `rknn_batch_size=N` so "
                        "one forward pass scores N people. Measured "
                        "bit-identical to N separate batch-1 calls on both "
                        "fp16 and int8 (MAX_ABS = 0.0), so it is a pure speed "
                        "change. 1 keeps the delivered artefact unchanged.")
    return p.parse_args()


# --------------------------------------------------------------- calibration


def build_stgcn_calibration(n: int = N_STGCN_CALIB) -> np.ndarray:
    """Real training windows, stratified by dataset and class."""
    import pickle
    from collections import defaultdict

    pkl = SOURCE_ROOT / "datasets" / "pyskl" / "v2" / "annotations.pkl"
    with open(pkl, "rb") as fh:
        data = pickle.load(fh)
    train_ids = set(data["split"]["train"])
    pool: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for ann in data["annotations"]:
        if ann["frame_dir"] not in train_ids:
            continue
        pool[(ann.get("dataset", "?"), int(ann["label"]))].append(ann)

    # Round-robin across (dataset, label) so no stratum dominates.
    keys = sorted(pool)
    picked: list[dict] = []
    i = 0
    while len(picked) < n and any(pool[k] for k in keys):
        k = keys[i % len(keys)]
        if pool[k]:
            picked.append(pool[k].pop(0))
        i += 1
    if len(picked) < n:
        print(f"  warning: only {len(picked)} train windows available for "
              f"calibration (wanted {n})")

    xs = []
    for ann in picked:
        kp = np.asarray(ann["keypoint"], dtype=np.float32)[0]      # (T, 17, 2)
        sc = np.asarray(ann["keypoint_score"], dtype=np.float32)[0]
        h, w = ann["img_shape"]
        # annotation keypoints are in pixels of `img_shape`; the model wants
        # unit coordinates, then the [-1,1] mapping `to_model_input` applies.
        kp = kp / np.array([max(1, w), max(1, h)], dtype=np.float32)
        idx = spec_mod.resample_indices(len(kp), spec_mod.RESAMPLE_FRAMES)
        xs.append(spec_mod.to_model_input(kp[idx], sc[idx]))
    return spec_mod.stack_batch(xs)


def build_yolo_calibration(n: int = N_YOLO_CALIB) -> np.ndarray:
    """Real training-set frames through the runtime's own preprocessing."""
    import cv2

    from src.deploy import yolo_export

    rows = [r for r in paths.read_csv(paths.SOURCE_MANIFEST_V2)
            if r.get("readable") == "True"]
    if not rows:
        raise RuntimeError(
            f"no readable rows in {paths.SOURCE_MANIFEST_V2}; cannot build a "
            f"compliant YOLO calibration set. Do NOT fall back to board frames "
            f"without a new decision -- see D-110 #5.")

    want_h, want_w = yolo_export.INPUT_SHAPE[2], yolo_export.INPUT_SHAPE[3]

    # Round-robin across datasets so a single one cannot dominate the ranges.
    by_ds: dict[str, list[dict]] = {}
    for r in rows:
        by_ds.setdefault(r.get("dataset", "?"), []).append(r)
    keys = sorted(by_ds)

    out: list[np.ndarray] = []
    native = resized = 0
    i = 0
    while len(out) < n:
        k = keys[i % len(keys)]
        i += 1
        if not by_ds[k]:
            if not any(by_ds[x] for x in keys):
                break
            continue
        row = by_ds[k].pop(0)
        path = paths.SOURCE_RAW_V2 / row["relative_path"]
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            print(f"  skip {row['video_id']}: cannot open {path.name}")
            continue
        total = int(row["n_frames"])
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, total // 2))
        ok, frame = cap.read()
        cap.release()
        if not ok:
            continue
        chw, params = yolo_export.preprocess(frame, imgsz=spec_mod.IMGSZ)
        if chw.shape[2:] != (want_h, want_w):
            chw = _resize_chw_to(chw, want_h, want_w)
            resized += 1
        else:
            native += 1
        # Keep the batch dim: the graph's input is (1, 3, 480, 640). Dropping it
        # makes the toolkit warn once per calibration frame --
        # `The dims of input(ndarray) shape (3, 480, 640) is wrong, expect dims
        # is 4! Try expand dims to (1, 3, 480, 640)!` -- and then expand it
        # itself, so the numbers are identical; only the log is unusable
        # (200 warnings buried the quantisation progress). Verified equivalent
        # on .
        out.append(chw)
    if not out:
        raise RuntimeError("YOLO calibration set came out empty")
    if len(out) < n:
        print(f"  warning: only {len(out)} calibration frames (wanted {n})")
    print(f"  calibration frames: {len(out)} "
          f"({native} native {want_h}x{want_w}, {resized} resampled from other "
          f"training resolutions)")
    return np.stack(out)


def _resize_chw_to(chw: np.ndarray, h: int, w: int) -> np.ndarray:
    """`(1, 3, H, W)` float32 -> `(1, 3, h, w)`, area-averaging for downscale."""
    import cv2

    src_h, src_w = chw.shape[2], chw.shape[3]
    interp = cv2.INTER_AREA if (h * w) < (src_h * src_w) else cv2.INTER_LINEAR
    planes = [cv2.resize(chw[0, c], (w, h), interpolation=interp)
              for c in range(chw.shape[1])]
    return np.stack(planes, axis=0)[None].astype(np.float32)


# ------------------------------------------------------------------ conversion


def _assert_onnx_shape(onnx_path: Path, want: tuple) -> None:
    """Fail loudly if the declared input shape disagrees with the graph."""
    import onnx

    m = onnx.load(str(onnx_path), load_external_data=False)
    got = tuple(d.dim_value for d in m.graph.input[0].type.tensor_type.shape.dim)
    if got != tuple(want):
        raise SystemExit(
            f"{onnx_path.name}: declared input {tuple(want)} but the graph "
            f"says {got}. Fix the constant, not the toolkit call.")


HYBRID_STEM_STGCN = "stgcnpp_v2"


def _build_hybrid(rknn, stem: str, protect: list[str],
                  calib_name: str) -> None:
    """Run the toolkit's two-step hybrid quantisation, protecting `protect`."""
    cwd = Path.cwd()
    if rknn.hybrid_quantization_step1(
            dataset=str(paths.CALIB / calib_name)) != 0:
        raise SystemExit("hybrid_quantization_step1 failed")
    cfg_path = cwd / f"{stem}.quantization.cfg"
    model_path = cwd / f"{stem}.model"
    data_path = cwd / f"{stem}.data"
    for p in (cfg_path, model_path, data_path):
        if not p.exists():
            raise SystemExit(
                f"step1 did not write {p.name}; it wrote "
                f"{sorted(x.name for x in cwd.iterdir())}")

    lines = cfg_path.read_text(encoding="utf-8").splitlines()
    if not lines or not lines[0].startswith("custom_quantize_layers:"):
        raise SystemExit("cfg schema changed: no leading custom_quantize_layers")
    payload = ", ".join(f'"{n}": "float16"' for n in protect)
    lines[0] = "custom_quantize_layers: {" + payload + "}"
    cfg_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  hybrid: protecting {protect} as float16")

    if rknn.hybrid_quantization_step2(
            model_input=str(model_path), data_input=str(data_path),
            model_quantization_cfg=str(cfg_path)) != 0:
        raise SystemExit("hybrid_quantization_step2 failed")


def convert_stgcn(precision: str, target: str,
                  quant_algorithm: str = DEFAULT_QUANT_ALGORITHM,
                  quant_dtype: str = DEFAULT_QUANT_DTYPE,
                  hybrid: bool = False,
                  batch: int = 1) -> Path:
    from rknn.api import RKNN

    onnx_path = ONNX["stgcn"]
    _assert_onnx_shape(onnx_path, STGCN_INPUT)
    suffix = "" if quant_algorithm == DEFAULT_QUANT_ALGORITHM else f"_{quant_algorithm}"
    if quant_dtype != DEFAULT_QUANT_DTYPE:
        suffix += f"_{quant_dtype}"
    if hybrid:
        suffix += "_hybrid"
    if batch > 1:
        suffix += f"_b{batch}"
    out = paths.MODELS / f"stgcnpp_v2_{precision}{suffix}.rknn"
    rknn = RKNN(verbose=True)
    # ST-GCN is not an image: its input is (N, M=1, T=64, V=17, C=3). The
    # toolkit infers the channel count for mean/std from the input rank, and for
    # a 5-D tensor it expects exactly ONE value -- passing the image-style
    # `[0, 0, 0]` raises `The len of mean_values ([0, 0, 0]) for input 0 is
    # wrong, expect 1!`. Caught on the first real run ( ).
    #
    # The values themselves stay 0/1 because `spec.to_model_input` already
    # produced the model's exact input convention ([-1, 1] coordinates, score
    # channel passed through). Doing the mapping here instead would apply it
    # twice -- a silent 2x coordinate error.
    rknn.config(mean_values=[[0]], std_values=[[1]],
                target_platform=target, quantized_dtype=quant_dtype,
                quantized_algorithm=quant_algorithm, optimization_level=3)
    # `input_size_list` is mandatory whenever `inputs` is given: toolkit 2.3.0
    # raises `If 'inputs' set, the 'input_size_list' should be set also!`
    # without it. Caught on the first real run ( ) -- this path had
    # never been executed before, so the docstring's "not run" status was
    # accurate and this was simply unknown.
    rknn.load_onnx(model=str(onnx_path),
                   inputs=["input"], input_size_list=[list(STGCN_INPUT)],
                   outputs=["logits"])
    # PLAN §3.3. `rknn_batch_size` expands the *static batch-1* graph the
    # toolkit just loaded -- it does not need a dynamic-axis ONNX, which is
    # why `models/stgcnpp_v2.onnx` can stay frozen. Verified on real
    # calibration windows: batch-4 output is bit-identical to four batch-1
    # calls on both fp16 and int8 (MAX_ABS = 0.0).
    #
    # Trap: a batch-N model *accepts* a single sample and broadcasts it to all
    # N slots, returning N identical logits with only a warning. The runtime
    # must always feed exactly N rows, or pad and discard.
    bkw = {"rknn_batch_size": batch} if batch > 1 else {}
    if precision == "fp16":
        rknn.build(do_quantization=False, **bkw)
    else:
        calib = build_stgcn_calibration()
        np.save(paths.CALIB / "stgcn_calib.npy", calib)
        if hybrid:
            _build_hybrid(rknn, HYBRID_STEM_STGCN, ["logits"],
                          "stgcn_calib.txt")
        else:
            rknn.build(do_quantization=True,
                       dataset=str(paths.CALIB / "stgcn_calib.txt"))
    rknn.export_rknn(str(out))
    rknn.release()
    return out


HYBRID_STEM = "yolo26n_pose_480x640"


def _build_yolo_hybrid(rknn, onnx_path: Path, target: str, out: Path) -> None:
    """Keep `output0` in fp16 via the toolkit's two-step hybrid quantisation."""
    _build_hybrid(rknn, HYBRID_STEM, ["output0"], "yolo_calib.txt")


def convert_yolo(precision: str, target: str,
                 quant_algorithm: str = DEFAULT_QUANT_ALGORITHM,
                 quant_dtype: str = DEFAULT_QUANT_DTYPE,
                 hybrid_head: bool = False) -> Path:
    from rknn.api import RKNN

    onnx_path = ONNX["yolo"]
    _assert_onnx_shape(onnx_path, YOLO_INPUT)
    suffix = "" if quant_algorithm == DEFAULT_QUANT_ALGORITHM else f"_{quant_algorithm}"
    if quant_dtype != DEFAULT_QUANT_DTYPE:
        suffix += f"_{quant_dtype}"
    out = paths.MODELS / f"yolo26n_pose_480x640_{precision}{suffix}.rknn"
    rknn = RKNN(verbose=True)
    rknn.config(mean_values=[[0, 0, 0]], std_values=[[1, 1, 1]],
                target_platform=target, quantized_dtype=quant_dtype,
                quantized_algorithm=quant_algorithm, optimization_level=3)
    rknn.load_onnx(model=str(onnx_path),
                   inputs=["images"], input_size_list=[list(YOLO_INPUT)],
                   outputs=["output0"])
    if precision == "fp16":
        rknn.build(do_quantization=False)
    else:
        # Hazard 1 from the module docstring: the head's confidence channel is
        # destroyed by a shared scale. The fix is to keep the head's final
        # convolutions in fp16, which needs the toolkit's *two-step* hybrid
        # quantisation API -- `hybrid_quantization_step1/step2`. There is no
        # `custom_quantize_layers` argument in toolkit 2.3.0; `build()` takes
        # only `(do_quantization, dataset, rknn_batch_size)` and `config()`
        # has no such parameter either (verified by `inspect.signature` on
        # , after a run died with
        # `TypeError: RKNN.build() got an unexpected keyword argument`).
        calib = build_yolo_calibration()
        np.save(paths.CALIB / "yolo_calib.npy", calib)
        if hybrid_head:
            _build_yolo_hybrid(rknn, onnx_path, target, out)
        else:
            # Plain INT8 first: if the head survives without protection there is
            # no reason to add the hybrid machinery. `scripts/21 --precision
            # int8 --model yolo` answers that, and its `conf_abs_err_max`
            # criterion exists precisely to detect the collapse.
            rknn.build(do_quantization=True,
                       dataset=str(paths.CALIB / "yolo_calib.txt"))
    rknn.export_rknn(str(out))
    rknn.release()
    return out


def write_calibration_lists() -> None:
    """rknn-toolkit2 wants a text file of image paths, not an array."""
    paths.CALIB.mkdir(parents=True, exist_ok=True)
    for name, builder, n in (("stgcn", build_stgcn_calibration, N_STGCN_CALIB),
                             ("yolo", build_yolo_calibration, N_YOLO_CALIB)):
        arr = builder(n)
        d = paths.CALIB / f"{name}_frames"
        d.mkdir(parents=True, exist_ok=True)
        lines = []
        for i, sample in enumerate(arr):
            p = d / f"{i:04d}.npy"
            np.save(p, sample)
            lines.append(str(p))
        (paths.CALIB / f"{name}_calib.txt").write_text("\n".join(lines) + "\n",
                                                       encoding="utf-8")
        print(f"  {name}: {len(arr)} calibration samples -> {d}")


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args()

    if sys.platform != "linux":
        print("This script needs rknn-toolkit2, which publishes Linux-only "
              "wheels.\nWSL2 is currently unavailable on this machine "
              "(HCS_E_HYPERV_NOT_INSTALLED), so Phase 2 is deferred.\n"
              "Re-run inside WSL2 with python 3.10 + rknn-toolkit2 2.3.0.")
        return 2

    paths.MODELS.mkdir(parents=True, exist_ok=True)
    paths.CALIB.mkdir(parents=True, exist_ok=True)
    for name, p in ONNX.items():
        if not p.exists():
            raise SystemExit(f"missing {p}; run scripts/11 / scripts/12 first")

    if args.precision == "int8":
        print("building calibration sets ...")
        write_calibration_lists()

    wanted = ["stgcn", "yolo"] if args.model == "all" else [args.model]
    for name in wanted:
        if name == "stgcn":
            out = convert_stgcn(args.precision, args.target,
                                args.quant_algorithm, args.quant_dtype,
                                hybrid=args.hybrid, batch=args.stgcn_batch)
        else:
            out = convert_yolo(args.precision, args.target,
                               args.quant_algorithm, args.quant_dtype,
                               hybrid_head=args.hybrid)
        # L-116: exit code 0 is not evidence of success. Two silent failures
        # this session (w8a16 and a mangled `wsl ~` path) both exited 0 with no
        # artefact. Assert the artefact exists and is non-trivial.
        if not out.exists() or out.stat().st_size < 1_000_000:
            raise SystemExit(
                f"{name}: build reported success but {out} is missing or "
                f"implausibly small. Do NOT verify against a stale file.")
        print(f"{name}: {out} ({out.stat().st_size/1e6:.2f} MB)")

    if args.verify:
        print("run scripts/21_verify_rknn.py to check G4/G5")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
