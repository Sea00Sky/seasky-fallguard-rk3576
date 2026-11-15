"""Phase 1b -- export YOLO26n-pose to ONNX and prove the decode matches (G2)."""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_own(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, PROJECT_ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


paths = _load_own("deploy_paths", "src/deploy/paths.py")
yolo = _load_own("deploy_yolo", "src/deploy/yolo_export.py")

import torch  # noqa: E402
from ultralytics import YOLO  # noqa: E402

ONNX_PATH = paths.MODELS / "yolo26n_pose_480x640.onnx"
OPSET = 12
CONF, MAX_PERSONS = 0.25, 4

# G2 criteria, as specified for Phase 1.
MIN_IOU_MEDIAN = 0.99
MIN_IOU_P5 = 0.95
MAX_KPT_L2_PX = 1.0
MIN_SCORE_CORR = 0.999


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--frames", type=int, default=240,
                   help="number of real board frames to check (>=200)")
    p.add_argument("--opset", type=int, default=OPSET)
    return p.parse_args()


def board_frames(n: int) -> list[tuple[str, int, np.ndarray]]:
    """`n` frames spread evenly over the board clips, as BGR uint8."""
    import cv2

    rows = paths.read_csv(paths.SOURCE_MANIFEST_BOARD)
    rows = [r for r in rows if r.get("readable") == "True" and int(r["n_frames"]) >= 60]
    out: list[tuple[str, int, np.ndarray]] = []
    per = max(1, n // len(rows))
    for row in rows:
        # manifest `relative_path` is relative to the source project's datasets/raw
        cap = cv2.VideoCapture(str(paths.SOURCE_RAW_BOARD / row["relative_path"]))
        if not cap.isOpened():
            continue
        total = int(row["n_frames"])
        picks = np.linspace(0, total - 1, per).astype(int)
        for f in picks:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
            ok, frame = cap.read()
            if ok:
                out.append((row["video_id"], int(f), frame))
        cap.release()
        if len(out) >= n:
            break
    return out[:n]


def ultralytics_result(model, frame):
    """ultralytics' own answer for one frame, in the same tuple layout."""
    r = model.predict(frame, imgsz=640, conf=CONF, iou=0.7, classes=[0],
                      max_det=300, verbose=False, device="cpu")[0]
    if r.boxes is None or len(r.boxes) == 0:
        return (np.zeros((0, 4), np.float32), np.zeros((0, 17, 2), np.float32),
                np.zeros((0, 17), np.float32))
    boxes = r.boxes.xyxy.cpu().numpy().astype(np.float32)
    conf = r.boxes.conf.cpu().numpy().astype(np.float32)
    kd = r.keypoints.data.cpu().numpy().astype(np.float32)
    order = np.argsort(-conf, kind="stable")[:MAX_PERSONS]
    return boxes[order], kd[order, :, :2], kd[order, :, 2]


def box_iou_one(a: np.ndarray, b: np.ndarray) -> float:
    """IoU of two xyxy boxes."""
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    bb = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = aa + bb - inter
    return float(inter / union) if union > 0 else 1.0


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args()
    paths.MODELS.mkdir(parents=True, exist_ok=True)
    paths.PARITY.mkdir(parents=True, exist_ok=True)

    want = paths.EXPECTED_SHA256["yolo26n-pose.pt"]
    got = paths.sha256_file(paths.YOLO_PT)
    print(f"weights: yolo26n-pose.pt {got[:16]}... {'OK' if got == want else 'MISMATCH'}")
    if got != want:
        raise SystemExit("wrong weight")

    model = YOLO(str(paths.YOLO_PT))
    net = model.model.eval()
    head = net.model[-1]
    print(f"head: {type(head).__name__} end2end={head.end2end} "
          f"nc={head.nc} kpt_shape={getattr(head, 'kpt_shape', None)} "
          f"max_det={head.max_det}")
    # `end2end=True` is the reason there is no NMS: ultralytics' own
    # `non_max_suppression` short-circuits to a pass-through on this flag.
    assert head.end2end, "expected the end2end (one2one) head"

    wrapper = yolo.YOLOPoseRaw(net, end2end_branch=True).eval()
    dummy = torch.zeros(*yolo.INPUT_SHAPE)
    with torch.no_grad():
        eager = wrapper(dummy)
    print(f"eager output {tuple(eager.shape)} "
          f"(expected (1, {yolo.OUT_CHANNELS}, {yolo.N_ANCHORS}))")
    assert tuple(eager.shape) == (1, yolo.OUT_CHANNELS, yolo.N_ANCHORS), \
        "unexpected export shape"

    # torch>=2.9 defaults to the dynamo exporter, which needs `onnxscript`.
    # The TorchScript path is what ultralytics itself uses (`torch2onnx`) and
    # what the opset-12 static graph was validated on, so pin it explicitly.
    torch.onnx.export(
        wrapper, dummy, str(ONNX_PATH),
        opset_version=args.opset,
        input_names=["images"], output_names=["output0"],
        dynamic_axes=None, do_constant_folding=True, verbose=False,
        dynamo=False,
    )
    print(f"wrote {ONNX_PATH} ({ONNX_PATH.stat().st_size/1e6:.2f} MB)")

    import onnx
    m = onnx.load(str(ONNX_PATH))
    onnx.checker.check_model(m)
    ops: dict[str, int] = {}
    for node in m.graph.node:
        ops[node.op_type] = ops.get(node.op_type, 0) + 1
    print(f"opset={args.opset} nodes={len(m.graph.node)} "
          f"TopK={ops.get('TopK', 0)} NonMaxSuppression={ops.get('NonMaxSuppression', 0)}")
    print("op histogram:", dict(sorted(ops.items(), key=lambda kv: -kv[1])[:12]))

    import onnxruntime as ort
    sess = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
    iname = sess.get_inputs()[0].name

    # A second graph that keeps ultralytics' own topk, so the structural check
    # compares ONNX-vs-ONNX instead of re-running the torch model, whose state
    # is not guaranteed to be stable once `fuse()` has run.
    #
    # This MUST be a fresh YOLO instance: `YOLOPoseRaw.__init__` sets
    # `net.model[-1].export = True` on the object it is handed, and `wrapper`
    # holds `net` itself. Reusing `net` here would export another no-TopK graph
    # and the structural check would compare our tensor against a copy of
    # itself -- which is exactly the 2/234 that showed up before this fix.
    full_net = YOLO(str(paths.YOLO_PT)).model.eval()
    assert full_net.model[-1].export is False, "reference graph must keep its TopK"
    full_path = paths.PARITY / "_yolo26n_pose_full300.onnx"
    torch.onnx.export(full_net, torch.zeros(*yolo.INPUT_SHAPE), str(full_path),
                      opset_version=args.opset, input_names=["images"],
                      output_names=["output0"], dynamic_axes=None,
                      do_constant_folding=True, verbose=False, dynamo=False)
    sess_full = ort.InferenceSession(str(full_path), providers=["CPUExecutionProvider"])
    full_in = sess_full.get_inputs()[0].name
    print(f"reference graph with topk: {full_path.name} "
          f"({full_path.stat().st_size/1e6:.2f} MB)")

    # ---- G2
    frames = board_frames(args.frames)
    print(f"G2 frames: {len(frames)} from {len({v for v, _, _ in frames})} clips")

    ious, kpt_l2, score_rs = [], [], []
    raw_box_d, raw_kpt_d, raw_sc_d = [], [], []
    count_mismatch, structural_ok, empty_frames = 0, 0, 0
    structural_worst = 0.0
    per_frame = []
    for vid, fidx, frame in frames:
        ref_b, ref_k, ref_s = ultralytics_result(model, frame)
        chw, params = yolo.preprocess(frame, 640, auto=True)
        assert chw.shape == yolo.INPUT_SHAPE, f"preprocess gave {chw.shape}"
        raw = sess.run(None, {iname: chw})[0]

        # Structural: ultralytics' `postprocess` is exactly `topk(300)` over the
        # tensor we export. So every one of its 300 rows must exist, bit for bit,
        # among our 6300 anchors. We look each one up rather than sorting both
        # sides: `torch.topk` and `np.argsort` disagree on the order of *tied*
        # scores, and with 6300 anchors near-zero ties are common (2 of the 300
        # rows here share a score exactly). Sorting would report a failure that
        # is a tie-break artefact, not a numerical one.
        full = sess_full.run(None, {full_in: chw})[0][0]      # (300, 57)
        # our layout is box4 + score + kpt51; full's is box4 + score + cls + kpt51
        mine57 = np.concatenate(
            [raw[0].T[:, :5], np.zeros((yolo.N_ANCHORS, 1), np.float32),
             raw[0].T[:, 5:]], axis=1)
        dist = np.abs(mine57[None, :, :] - full[:, None, :]).max(axis=2)  # (300, 6300)
        worst_here = float(dist.min(axis=1).max())
        structural_worst = max(structural_worst, worst_here)
        if worst_here <= 1e-4:
            structural_ok += 1

        on_b, on_k, on_s = yolo.decode(raw, params, CONF, MAX_PERSONS)
        if len(on_b) != len(ref_b):
            count_mismatch += 1
            per_frame.append({"video_id": vid, "frame": fidx,
                              "n_ref": int(len(ref_b)), "n_onnx": int(len(on_b))})
            continue
        if len(ref_b) == 0:
            empty_frames += 1
            continue

        # Same count and the same sort order, so match position by position --
        # no greedy pairing, which could hide a swap between two people.
        raw_box_d.append(float(np.abs(ref_b - on_b).max()))
        raw_kpt_d.append(float(np.linalg.norm(ref_k - on_k, axis=2).max()))
        raw_sc_d.append(float(np.abs(ref_s - on_s).max()))
        for i in range(len(ref_b)):
            ious.append(float(box_iou_one(ref_b[i], on_b[i])))
            kpt_l2.append(float(np.linalg.norm(ref_k[i] - on_k[i], axis=1).max()))
            a, b = ref_s[i].ravel(), on_s[i].ravel()
            if a.std() > 1e-8 and b.std() > 1e-8:
                score_rs.append(float(np.corrcoef(a, b)[0, 1]))

    ious_a = np.asarray(ious) if ious else np.zeros(1)
    l2_a = np.asarray(kpt_l2) if kpt_l2 else np.zeros(1)
    rs_a = np.asarray(score_rs) if score_rs else np.ones(1)
    iou_med = float(np.median(ious_a))
    iou_p5 = float(np.percentile(ious_a, 5))
    kpt_max = float(l2_a.max())
    rs_min = float(rs_a.min())
    raw_box_max = float(max(raw_box_d)) if raw_box_d else 0.0
    raw_kpt_max = float(max(raw_kpt_d)) if raw_kpt_d else 0.0
    raw_sc_max = float(max(raw_sc_d)) if raw_sc_d else 0.0

    ok = (iou_med >= MIN_IOU_MEDIAN and iou_p5 >= MIN_IOU_P5
          and kpt_max <= MAX_KPT_L2_PX and rs_min >= MIN_SCORE_CORR
          and count_mismatch == 0 and structural_ok == len(frames))
    result = {
        "opset": args.opset, "onnx_bytes": ONNX_PATH.stat().st_size,
        "onnx_sha256": paths.sha256_file(ONNX_PATH),
        "source_weight_sha256": want,
        "input_shape": list(yolo.INPUT_SHAPE),
        "output_shape": [1, yolo.OUT_CHANNELS, yolo.N_ANCHORS],
        "head": f"{type(head).__name__}(one2one branch, topk removed)",
        "nms": "none (end2end head; ultralytics short-circuits non_max_suppression)",
        "op_histogram": ops,
        "n_frames": len(frames), "n_clips": len({v for v, _, _ in frames}),
        "n_frames_with_person": int(len(raw_box_d)),
        "n_empty_frames": empty_frames,
        "n_matched_boxes": int(len(ious_a)),
        "box_iou_median": iou_med, "box_iou_p5": iou_p5,
        "kpt_l2_max_px": kpt_max,
        "score_corr_min": rs_min,
        "count_mismatch_frames": count_mismatch,
        "structural_topk_ok_frames": structural_ok,
        "structural_worst_row_match": structural_worst,
        "raw_max_abs_diff": {"box_px": raw_box_max, "kpt_px": raw_kpt_max,
                             "kpt_score": raw_sc_max},
        "criteria": {"iou_median": MIN_IOU_MEDIAN, "iou_p5": MIN_IOU_P5,
                     "kpt_l2_px": MAX_KPT_L2_PX, "score_corr": MIN_SCORE_CORR},
        "passed": bool(ok),
    }
    paths.write_json(paths.PARITY / "g2_yolo.json", result)

    def mark(good: bool) -> str:
        return "✅" if good else "❌"

    lines = ["# G2 — YOLO26n-pose ONNX 解码对账", "",
             f"- 源权重 `weights/yolo26n-pose.pt` sha256 `{want[:16]}…`",
             f"- 产出 `models/yolo26n_pose_480x640.onnx` "
             f"sha256 `{result['onnx_sha256'][:16]}…`，"
             f"{result['onnx_bytes']/1e6:.2f} MB，opset {args.opset}",
             f"- 输入 `{tuple(yolo.INPUT_SHAPE)}`，输出 `(1, {yolo.OUT_CHANNELS}, {yolo.N_ANCHORS})`",
             f"- 导出的头：**one2one（`end2end=True` 那条分支），去掉 `topk(300)`**；"
             f"`TopK` {ops.get('TopK', 0)} 个、`NonMaxSuppression` "
             f"{ops.get('NonMaxSuppression', 0)} 个",
             f"- **不做 NMS**：`end2end=True` 时 ultralytics 的 `non_max_suppression` "
             f"第一个分支直接返回，one2one 头本就一目标一框",
             f"- 输入 480×640 而非 640×640：`LetterBox(auto=True)` 对 640×480 帧是恒等"
             f"（`dw = dh = (640−480) % 32 = 0`），源管线送进网络的就是这个形状",
             f"- 真实板端帧 {len(frames)} 帧 / {result['n_clips']} 个视频，"
             f"其中 {len(raw_box_d)} 帧有检出、{empty_frames} 帧空，"
             f"共 {len(ious_a)} 个框逐框对齐",
             "", "| 判据 | 实测 | 门槛 | 结果 |", "|---|---:|---:|---|",
             f"| bbox IoU 中位 | {iou_med:.6f} | ≥{MIN_IOU_MEDIAN:g} "
             f"| {mark(iou_med >= MIN_IOU_MEDIAN)} |",
             f"| bbox IoU p5 | {iou_p5:.6f} | ≥{MIN_IOU_P5:g} "
             f"| {mark(iou_p5 >= MIN_IOU_P5)} |",
             f"| 关键点 L2 最大值 | {kpt_max:.3e} px | ≤{MAX_KPT_L2_PX:g} px "
             f"| {mark(kpt_max <= MAX_KPT_L2_PX)} |",
             f"| 关键点 score 相关最小值 | {rs_min:.8f} | ≥{MIN_SCORE_CORR:g} "
             f"| {mark(rs_min >= MIN_SCORE_CORR)} |",
             f"| 逐帧检出数不同 | {count_mismatch} | 0 "
             f"| {mark(count_mismatch == 0)} |",
             f"| topk(300) 结构一致帧 | {structural_ok}/{len(frames)} | 全部 "
             f"| {mark(structural_ok == len(frames))} |",
             f"| 结构检查逐行匹配最差距离 | {structural_worst:.3e} | ≤1e-4 "
             f"| {mark(structural_worst <= 1e-4)} |",
             "", "**原始差值**（不经 IoU/相关系数，直接逐框相减，"
             "反映 ONNX Runtime 与 PyTorch 的 fp32 算子顺序噪声）：", "",
             "| 量 | `max\\|Δ\\|` |", "|---|---:|",
             f"| bbox | {raw_box_max:.3e} px |",
             f"| 关键点 | {raw_kpt_max:.3e} px |",
             f"| 关键点 score | {raw_sc_max:.3e} |",
             "", f"**G2：{'通过' if ok else '未通过'}**", ""]
    (paths.PARITY / "g2_yolo.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    if per_frame:
        paths.write_json(paths.PARITY / "g2_yolo_count_mismatch.json", per_frame)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
