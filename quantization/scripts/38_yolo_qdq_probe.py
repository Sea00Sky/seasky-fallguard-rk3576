"""Gate 0 of the INT8-QAT effort: does a QDQ graph survive the YOLO layout?"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.deploy import paths  # noqa: E402

OUT_DIR = PROJECT_ROOT / "experiments" / "qat"
SRC_ONNX = paths.MODELS / "yolo26n_pose_480x640.onnx"
QDQ_ONNX = OUT_DIR / "yolo_qdq_probe.onnx"
OUT_JSON = paths.PARITY / "g5_yolo_qdq_probe.json"


# --------------------------------------------------------------- build the QDQ


def _upgrade_to_opset(model, target: int = 13):
    """Rewrite an opset-12 graph into opset 13, node bodies included."""
    import onnx
    from onnx import version_converter

    cur = next((o.version for o in model.opset_import
                if o.domain in ("", "ai.onnx")), 12)
    if cur >= target:
        return model
    try:
        out = version_converter.convert_version(model, target)
    except Exception as exc:  # noqa: BLE001 -- surfaced verbatim on purpose
        raise SystemExit(
            f"onnx version_converter could not lift opset {cur} -> {target}: "
            f"{exc}\nThe QDQ graph needs >=13 (DequantizeLinear.axis). "
            f"Install a newer `onnx` (>=1.13) and re-run.") from exc
    print(f"  converted opset {cur} -> {target} "
          f"(attribute->input rewrites included)", flush=True)
    return out


def _check_equivalent(before, after, sample: np.ndarray) -> None:
    """Both graphs, same input, same output -- or the migration is rejected."""
    import onnxruntime as ort

    def run(model) -> np.ndarray:
        sess = ort.InferenceSession(model.SerializeToString(),
                                    providers=["CPUExecutionProvider"])
        return sess.run(None, {sess.get_inputs()[0].name: sample})[0]

    a, b = run(before), run(after)
    if a.shape != b.shape:
        raise SystemExit(f"opset migration changed the output shape: "
                         f"{a.shape} -> {b.shape}")
    d = float(np.abs(a - b).max())
    print(f"  opset migration equivalence: max|d| = {d:.3e} on one sample",
          flush=True)
    if d > 1e-4:
        raise SystemExit(
            f"opset migration changed the output (max|d| = {d:.3e}); refusing "
            f"to quantise a graph that no longer computes the same function")


def make_qdq(n_calib: int = 60) -> None:
    """Static QDQ quantisation of the YOLO graph."""
    import importlib.util
    import shutil
    import tempfile

    import onnx
    from onnxruntime.quantization import (CalibrationDataReader, QuantFormat,
                                          QuantType, quantize_static)

    spec = importlib.util.spec_from_file_location(
        "convert20", PROJECT_ROOT / "scripts" / "20_convert_rknn.py")
    convert = importlib.util.module_from_spec(spec)
    sys.modules["convert20"] = convert
    spec.loader.exec_module(convert)

    print(f"building calibration set ({n_calib} training frames) ...", flush=True)
    calib = convert.build_yolo_calibration(n_calib).astype(np.float32)
    _ = calib  # used by the reader below; the equivalence check re-reads it
    # `build_yolo_calibration` keeps the batch axis -- it returns
    # `(N, 1, 3, 480, 640)` because `scripts/20` feeds those files to the
    # toolkit, which wants the batch dim present. `quantize_static` wants one
    # sample per `get_next()` without it, so drop the redundant axis here.
    # (Passing it through raises "Invalid rank for input: images Got: 5
    # Expected: 4", which reads like a model problem but is this.)
    if calib.ndim == 5 and calib.shape[1] == 1:
        calib = calib[:, 0]
    print(f"  calibration {calib.shape}", flush=True)

    m = onnx.load(str(SRC_ONNX))
    input_name = m.graph.input[0].name

    class Reader(CalibrationDataReader):
        def __init__(self, data: np.ndarray, name: str):
            self._it = iter([{name: data[i:i + 1]} for i in range(len(data))])

        def get_next(self):
            return next(self._it, None)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # ## The opset trap (two failures, one root cause)
    #
    # The delivered graph declares opset 12 (`scripts/12` pins it because
    # RKNN wants a small, stable op set). `quantize_static` emits
    # `DequantizeLinear` with the `axis` attribute, which only exists from
    # opset 13. Left alone this fails twice, in two different places:
    #
    # load_onnx: Unrecognized attribute: axis for operator DequantizeLinear
    # -> then a segfault during `build`
    #
    # Bumping only `opset_import` afterwards is not enough and fails later:
    #
    # InvalidGraph: Unrecognized attribute: split for operator Split
    #
    # because `Split.split` moved from an attribute (<=12) to an input (>=13)
    # and the *original* nodes are still in the old form. So the migration has
    # to happen before quantisation, on the source graph, using the ONNX
    # version converter -- which rewrites every affected op, not just the stamp.
    #
    # This is a pure schema migration (no numeric change), verified by
    # `_check_equivalent` below on one calibration sample.
    m_upgraded = _upgrade_to_opset(m, target=13)
    # A schema migration that changes the output is not a migration. Check it
    # on one real sample before quantising, so a converter bug cannot be
    # mistaken for a quantisation result later.
    _check_equivalent(m, m_upgraded, calib[0:1])
    m = m_upgraded

    # `quantize_static` writes a sidecar and reopens it by path; the project
    # path has non-ASCII characters, which onnxruntime's internal path handling
    # mangles (scripts/29 documents the same failure).
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        local_src = td_path / "model.onnx"
        local_dst = td_path / "model_qdq.onnx"
        onnx.save(m, str(local_src))
        quantize_static(
            model_input=str(local_src),
            model_output=str(local_dst),
            calibration_data_reader=Reader(calib, input_name),
            quant_format=QuantFormat.QDQ,
            activation_type=QuantType.QInt8,
            weight_type=QuantType.QInt8,
            per_channel=True,
        )
        shutil.copyfile(local_dst, QDQ_ONNX)

    q = onnx.load(str(QDQ_ONNX))

    ops = [n.op_type for n in q.graph.node]
    nq, ndq = ops.count("QuantizeLinear"), ops.count("DequantizeLinear")
    print(f"wrote {QDQ_ONNX}  ({QDQ_ONNX.stat().st_size/1e6:.2f} MB)", flush=True)
    print(f"  QuantizeLinear x{nq}, DequantizeLinear x{ndq}", flush=True)
    # The output must stay a float tensor: a QDQ graph that ends in int8 would
    # hand the runtime quantised logits, which `yolo_export.decode` cannot read.
    out = q.graph.output[0]
    print(f"  graph output: {out.name} {out.type.tensor_type.elem_type}", flush=True)
    if nq == 0:
        raise SystemExit("no QuantizeLinear nodes -- the graph was not quantised")


# ------------------------------------------------------- convert + simulate (WSL)


def simulate(n_frames: int = 24) -> int:
    from rknn.api import RKNN

    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "yolo_export", PROJECT_ROOT / "src" / "deploy" / "yolo_export.py")
    ye = importlib.util.module_from_spec(spec)
    sys.modules["yolo_export"] = ye
    spec.loader.exec_module(ye)
    spec2 = importlib.util.spec_from_file_location(
        "deploy_spec", PROJECT_ROOT / "src" / "deploy" / "spec.py")
    spec_mod = importlib.util.module_from_spec(spec2)
    sys.modules["deploy_spec"] = spec_mod
    spec2.loader.exec_module(spec_mod)

    if not QDQ_ONNX.exists():
        raise SystemExit(f"missing {QDQ_ONNX}; run --make-qdq on Windows first")

    print("loading QDQ graph into rknn ...", flush=True)
    r = RKNN(verbose=False)
    # ## `optimization_level=2`, not 3
    #
    # Level 3 segfaults on QDQ graphs -- and it does so silently: `build`
    # takes the interpreter down with exit code 0 and no traceback, so the only
    # symptom is a log that stops at "rknn building ...". Measured on this YOLO
    # graph ; `experiments/qat/run_qat_stgcn.py` already carries the
    # same note for the ST-GCN QDQ graph. The toolkit even says so itself:
    #
    # The QAT model is loaded, but the config.optimization_level = 3, and
    # some performance optimization passes will affect the accuracy.
    # If you need more consistent accuracy, you can try changing
    # config.optimization_level to 2.
    #
    # `do_quantization=False` is correct for a QDQ graph: it already carries
    # its scales as QuantizeLinear/DequantizeLinear, and re-quantising would
    # discard exactly what this test is about.
    r.config(mean_values=[[0, 0, 0]], std_values=[[1, 1, 1]],
             target_platform="rk3576", optimization_level=2)
    if r.load_onnx(model=str(QDQ_ONNX)) != 0:
        raise SystemExit("load_onnx FAILED -- no delivery path for QAT")
    print("load_onnx OK (QAT model recognised)", flush=True)

    # `build` on a QDQ graph can terminate the process without raising and
    # with exit code 0 -- measured : the log ends at "rknn
    # building ..." and the interpreter is gone. L-116's pattern exactly, so
    # the success claim is made explicitly here rather than inferred from
    # "no exception".
    print("building ...", flush=True)
    rc = r.build(do_quantization=False)
    print(f"build returned {rc}", flush=True)
    if rc != 0:
        raise SystemExit("build FAILED -- no delivery path for QAT")
    print("build OK", flush=True)
    if r.init_runtime(target=None) != 0:
        print("init_runtime FAILED -- convertible but NOT measurable on PC")
        OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
        paths.write_json(OUT_JSON, {
            "step": "probe", "converted": True, "simulator": False,
            "conclusion": "QAT deliverable but not measurable on the PC simulator",
        })
        return 2
    print("convert OK, simulator OK", flush=True)

    # --- reference: the same frames through the delivered ONNX
    import onnxruntime as ort
    onnx = ort.InferenceSession(str(SRC_ONNX), providers=["CPUExecutionProvider"])
    _, params = ye.preprocess(np.zeros((480, 640, 3), np.uint8),
                              imgsz=spec_mod.IMGSZ)
    frames = _board_frames(n_frames)

    stats = []
    for i, chw in enumerate(frames):
        ra = onnx.run(None, {onnx.get_inputs()[0].name: chw})[0]
        rb = r.inference(inputs=[chw], data_format="nchw")
        if rb is None:
            raise SystemExit("simulator returned None -- see the toolkit log")
        rb = np.asarray(rb[0])
        if rb.shape != ra.shape:
            raise SystemExit(f"shape mismatch: onnx {ra.shape} vs rknn {rb.shape}")

        ba, ka, sa = ye.decode(ra, params, conf=spec_mod.CONF,
                               max_persons=spec_mod.MAX_PERSONS)
        bb, kb, sb = ye.decode(rb, params, conf=spec_mod.CONF,
                               max_persons=spec_mod.MAX_PERSONS)
        stats.append({
            "i": i,
            "conf_max_onnx": float(ra[0, 4].max()),
            "conf_max_rknn": float(rb[0, 4].max()),
            "kpt_score_max_onnx": float(ra[0, 5::3].max()),
            "kpt_score_max_rknn": float(rb[0, 5::3].max()),
            "n_det_onnx": int(len(ba)), "n_det_rknn": int(len(bb)),
            "max_abs_diff": float(np.abs(ra - rb).max()),
        })
        print(f"  frame {i}: conf {stats[-1]['conf_max_onnx']:.4f} -> "
              f"{stats[-1]['conf_max_rknn']:.4f} | det {len(ba)} -> {len(bb)}",
              flush=True)

    conf_alive = all(s["conf_max_rknn"] > 0.0 for s in stats)
    kpt_alive = all(s["kpt_score_max_rknn"] > 0.0 for s in stats)
    same_count = sum(1 for s in stats if s["n_det_onnx"] == s["n_det_rknn"])
    result = {
        "step": "probe", "converted": True, "simulator": True,
        "n_frames": len(stats),
        "conf_channel_alive": bool(conf_alive),
        "kpt_score_alive": bool(kpt_alive),
        "frames_same_detection_count": int(same_count),
        "max_abs_diff_overall": float(max(s["max_abs_diff"] for s in stats)),
        "per_frame": stats,
        "conclusion": (
            "QDQ keeps the YOLO output layout alive; QAT has a delivery path"
            if conf_alive and kpt_alive else
            "QDQ converts and runs, but the output layout still collapses -- "
            "QAT must also truncate the graph (D-127 t4 layout)"),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    paths.write_json(OUT_JSON, result)
    print(f"\nwrote {OUT_JSON}")
    print(f"conf alive: {conf_alive}   kpt score alive: {kpt_alive}")
    print(f"same detection count: {same_count}/{len(stats)} frames")
    return 0 if (conf_alive and kpt_alive) else 1


def _board_frames(n: int) -> list[np.ndarray]:
    """`n` preprocessed frames spread over the board clips."""
    import cv2

    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "yolo_export", PROJECT_ROOT / "src" / "deploy" / "yolo_export.py")
    ye = importlib.util.module_from_spec(spec)
    sys.modules["yolo_export"] = ye
    spec.loader.exec_module(ye)
    spec2 = importlib.util.spec_from_file_location(
        "deploy_spec", PROJECT_ROOT / "src" / "deploy" / "spec.py")
    spec_mod = importlib.util.module_from_spec(spec2)
    sys.modules["deploy_spec"] = spec_mod
    spec2.loader.exec_module(spec_mod)

    rows = [r for r in paths.read_csv(paths.SOURCE_MANIFEST_BOARD)
            if r.get("readable") == "True"]
    out: list[np.ndarray] = []
    per = max(1, n // max(1, len(rows)))
    for row in rows:
        cap = cv2.VideoCapture(str(paths.SOURCE_RAW_BOARD / row["relative_path"]))
        if not cap.isOpened():
            continue
        total = int(row["n_frames"])
        for f in np.linspace(0, total - 1, per).astype(int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
            ok, frame = cap.read()
            if not ok:
                continue
            chw, _ = ye.preprocess(frame, imgsz=spec_mod.IMGSZ)
            out.append(chw)
        cap.release()
        if len(out) >= n:
            break
    if not out:
        raise SystemExit("could not read any board frames")
    return out[:n]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--make-qdq", action="store_true",
                    help="Windows side: build the QDQ graph")
    ap.add_argument("--simulate", action="store_true",
                    help="WSL side: convert + run on the simulator")
    ap.add_argument("--n-calib", type=int, default=60)
    ap.add_argument("--n-frames", type=int, default=24)
    args = ap.parse_args()

    if args.make_qdq:
        make_qdq(args.n_calib)
        return 0
    if args.simulate:
        return simulate(args.n_frames)
    ap.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
