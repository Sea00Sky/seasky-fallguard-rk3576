"""Run the fall pipeline end-to-end on the board-recorded videos (no labels)."""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import paths  # noqa: E402
from src.pose import extract  # noqa: E402
from src.window import build as wbuild  # noqa: E402
from src.train import engine  # noqa: E402

BOARD = "board"
KEYPOINT_ROOT = paths.KEYPOINTS / BOARD
WINDOW_ROOT = paths.WINDOWS / BOARD
PYSKL_ROOT = paths.PYSKL / BOARD
REPORT_ROOT = paths.REPORTS / "board_eval"
RUN_DIR = paths.RUNS / "stgcnpp_v2"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--overwrite", action="store_true", help="re-extract keypoints")
    p.add_argument("--device", default="", help="override pose device")
    p.add_argument("--person-selection", choices=["per_person_max", "max_presence"],
                   default="per_person_max",
                   help="D-040: per_person_max scores every eligible track and takes "
                        "the max; max_presence is the D-018 baseline (reproducible)")
    p.add_argument("--no-stitch", action="store_true",
                   help="D-042: disable ID-handoff stitching (per_person_max only)")
    # ultralytics (base env) and pyskl (pyskl env) do not share an interpreter.
    p.add_argument("--stage", choices=["prepare", "infer", "all"], default="all",
                   help="prepare = keypoints+windows+pkl (base env); infer = model (pyskl env)")
    return p.parse_args()


def load_manifest() -> list[dict]:
    rows = paths.read_csv(paths.MANIFESTS / BOARD / "videos.csv")
    return [r for r in rows if r.get("readable") == "True"]


# ------------------------------------------------------------------ keypoints


def extract_keypoints(rows: list[dict], overwrite: bool, device: str) -> list[dict]:
    cfg = extract.load_config()
    if device:
        cfg.device = device
    # npz_path_for appends the dataset name, so point at keypoints/ not keypoints/board/.
    cfg.output_root = paths.KEYPOINTS
    cfg.index_path = KEYPOINT_ROOT / "extraction_index.csv"
    cfg.output_root.mkdir(parents=True, exist_ok=True)
    KEYPOINT_ROOT.mkdir(parents=True, exist_ok=True)

    todo = []
    for r in rows:
        npz = extract.npz_path_for(cfg, r)
        if overwrite or not extract.npz_is_complete(npz, int(r["n_frames"]), cfg.max_persons):
            todo.append(r)
    print(f"keypoints: {len(rows)} videos, {len(todo)} to extract")

    results = []
    if todo:
        from ultralytics import YOLO

        model = YOLO(str(cfg.model_path))
        sha = paths.sha256_file(cfg.model_path)
        for i, row in enumerate(todo, 1):
            full = {**row, "split": "", "label": ""}
            res = extract.extract_video(model, cfg, full, sha, low_conf=True)
            print(f"[{i}/{len(todo)}] {row['video_id']} frames={res['n_frames_read']} "
                  f"cov={res['selected_coverage']:.2f} {res['status']} {res['error']}",
                  flush=True)
            results.append(res)
        paths.write_csv(cfg.index_path, results, extract.INDEX_FIELDS)
    return results


# ------------------------------------------------------------------- windows


def build_windows(rows: list[dict], person_selection: str = "per_person_max",
                  stitch: bool = True) -> tuple[list[dict], np.ndarray, np.ndarray]:
    """Build the board window table."""
    cfg = wbuild.load_config()
    kp_paths = {
        r["video_id"]: extract.npz_path_for(
            _board_pose_cfg(), r
        )
        for r in rows
    }
    meta: list[dict] = []
    kps: list[np.ndarray] = []
    scs: list[np.ndarray] = []
    window_seq = 0  # counts windows, not samples (per_person_max emits several)
    n_chains = 0
    for row in rows:
        path = kp_paths[row["video_id"]]
        if not path.exists():
            continue
        with np.load(path, allow_pickle=False) as d:
            keypoints = d["keypoints"]
            scores = d["scores"]
            track_ids = d["track_ids"]
            n_persons = d["n_persons"].astype(np.int32)
            bboxes = d["bboxes"]
            width = int(d["meta_width"])
            height = int(d["meta_height"])
        fps = float(row["fps"])
        n_frames = len(n_persons)
        window = wbuild.frames_for(fps, cfg.duration_s)
        stride = wbuild.frames_for(fps, cfg.stride_s)
        if n_frames < window:
            print(f"skip {row['video_id']}: {n_frames} < {window} frames")
            continue
        groups = None
        if person_selection == "per_person_max" and stitch:
            chains = wbuild.stitch_id_handoffs(track_ids, n_persons, bboxes)
            groups = [set(c) for c in chains]
            n_chains += sum(1 for c in chains if len(c) > 1)
        for w0 in range(0, n_frames - window + 1, stride):
            w1 = w0 + window
            if person_selection == "per_person_max":
                persons = wbuild.extract_window_persons(
                    keypoints, scores, track_ids, n_persons, w0, w1,
                    cfg, width, height, groups=groups, stitch=bool(groups),
                )
                if not persons:
                    continue
                wid = f"b{window_seq:06d}"
                window_seq += 1
                for p in persons:
                    meta.append({
                        "window_id": wid,
                        "track_id": p["track_id"],
                        "members": "|".join(str(x) for x in p["members"]),
                        "video_id": row["video_id"],
                        "group": row["group"],
                        "start_frame": w0,
                        "end_frame": w1,
                        "fps": f"{fps:.6f}",
                        "width": width,
                        "height": height,
                        "coverage": f"{p['coverage']:.4f}",
                        "n_valid_frames": p["n_valid_frames"],
                    })
                    kps.append(p["keypoints"])
                    scs.append(p["scores"])
                continue

            picked = wbuild.extract_window(
                keypoints, scores, track_ids, n_persons, w0, w1, cfg, width, height
            )
            if picked is None or picked["coverage"] < cfg.min_coverage:
                continue
            meta.append({
                "window_id": f"b{window_seq:06d}",
                "track_id": picked["track_id"],
                "members": str(picked["track_id"]),
                "video_id": row["video_id"],
                "group": row["group"],
                "start_frame": w0,
                "end_frame": w1,
                "fps": f"{fps:.6f}",
                "width": width,
                "height": height,
                "coverage": f"{picked['coverage']:.4f}",
                "n_valid_frames": picked["n_valid_frames"],
            })
            window_seq += 1
            kps.append(picked["keypoints"])
            scs.append(picked["scores"])

    if not meta:
        raise RuntimeError("no board windows built")
    kp_arr = np.stack(kps).astype(np.float16)
    sc_arr = np.stack(scs).astype(np.float16)
    WINDOW_ROOT.mkdir(parents=True, exist_ok=True)
    with open(WINDOW_ROOT / "windows.npz", "wb") as fh:
        np.savez_compressed(fh, keypoints=kp_arr, scores=sc_arr)
    paths.write_csv(
        WINDOW_ROOT / "windows.csv", meta,
        ["window_id", "track_id", "members", "video_id", "group", "start_frame",
         "end_frame", "fps", "width", "height", "coverage", "n_valid_frames"],
    )
    if n_chains:
        print(f"D-042: merged {n_chains} ID-handoff chains")
    return meta, kp_arr, sc_arr


def collapse_to_windows(meta: list[dict], probs: np.ndarray) -> tuple[list[dict], np.ndarray]:
    """: one row per window id, probability = max over its person samples."""
    order: list[str] = []
    best: dict[str, tuple[float, dict]] = {}
    for i, row in enumerate(meta):
        wid = row["window_id"]
        if wid not in best:
            order.append(wid)
            best[wid] = (float(probs[i]), row)
        elif float(probs[i]) > best[wid][0]:
            best[wid] = (float(probs[i]), row)
    rows = [best[w][1] for w in order]
    arr = np.asarray([best[w][0] for w in order], dtype=np.float64)
    return rows, arr


def _board_pose_cfg() -> extract.PoseConfig:
    cfg = extract.load_config()
    cfg.output_root = paths.KEYPOINTS
    cfg.index_path = KEYPOINT_ROOT / "extraction_index.csv"
    return cfg


def export_pyskl(meta: list[dict], kp: np.ndarray, sc: np.ndarray) -> Path:
    import pickle

    anns = []
    for i, row in enumerate(meta):
        k = kp[i].astype(np.float32).copy()
        k[:, :, 0] *= float(row["width"])
        k[:, :, 1] *= float(row["height"])
        anns.append({
            "frame_dir": row["window_id"],
            "total_frames": int(k.shape[0]),
            "label": 0,  # placeholder: board clips are unlabelled
            "label_known": False,
            "keypoint": k[None],
            "keypoint_score": sc[i].astype(np.float32)[None],
            "img_shape": (int(row["height"]), int(row["width"])),
            "original_shape": (int(row["height"]), int(row["width"])),
            "video_id": row["video_id"],
            "group": row["group"],
            "start_frame": int(row["start_frame"]),
            "end_frame": int(row["end_frame"]),
            "coverage": float(row["coverage"]),
        })
    PYSKL_ROOT.mkdir(parents=True, exist_ok=True)
    out = PYSKL_ROOT / "annotations.pkl"
    with open(out, "wb") as fh:
        pickle.dump({"split": {"test": [a["frame_dir"] for a in anns]},
                     "annotations": anns}, fh, protocol=4)
    return out


# ---------------------------------------------------------------- inference


def run_model(pkl: Path, device: str) -> np.ndarray:
    import torch

    with open(RUN_DIR / "config_used.yaml", "r", encoding="utf-8-sig") as fh:
        import yaml
        cfg = yaml.safe_load(fh)
    cfg["data"]["ann_file"] = str(pkl)
    model = engine.build_recognizer(cfg)
    with open(RUN_DIR / "best.pth", "rb") as fh:
        ckpt = torch.load(fh, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(dev)
    _, loader = engine.build_loader(cfg, "test")
    labels, probs = engine.predict(model, loader, dev)
    return probs


# ------------------------------------------------------------------ reporting


def snapshots(meta: list[dict], probs: np.ndarray, threshold: float) -> None:
    import cv2

    out_dir = REPORT_ROOT / "frames"
    out_dir.mkdir(parents=True, exist_ok=True)
    by_video: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(meta):
        by_video[row["video_id"]].append(i)
    manifest = {r["video_id"]: r for r in load_manifest()}
    for vid, idxs in by_video.items():
        peak = max(idxs, key=lambda i: probs[i])
        row = meta[peak]
        frame = (int(row["start_frame"]) + int(row["end_frame"])) // 2
        src = paths.RAW / manifest[vid]["relative_path"]
        cap = cv2.VideoCapture(str(src))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ok, img = cap.read()
        cap.release()
        if not ok:
            continue
        cv2.putText(img, f"p={probs[peak]:.3f} t={frame/float(row['fps']):.1f}s",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        name = f"{vid.replace(':', '__')}__f{frame:04d}__p{probs[peak]:.3f}.jpg"
        cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 88])[1].tofile(
            str(out_dir / name)
        )


def segments(rows: list[dict], probs: np.ndarray, threshold: float) -> list[dict]:
    out: list[dict] = []
    by_video: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        by_video[r["video_id"]].append(i)
    for vid, idxs in by_video.items():
        idxs = sorted(idxs, key=lambda i: int(rows[i]["start_frame"]))
        cur = None
        for i in idxs:
            hit = probs[i] >= threshold
            if hit and cur is None:
                cur = {"start": int(rows[i]["start_frame"]),
                       "end": int(rows[i]["end_frame"]), "peak": float(probs[i]),
                       "peak_i": i}
            elif hit:
                cur["end"] = max(cur["end"], int(rows[i]["end_frame"]))
                if probs[i] > cur["peak"]:
                    cur["peak"], cur["peak_i"] = float(probs[i]), i
            elif cur is not None:
                out.append({"video_id": vid, **cur})
                cur = None
        if cur is not None:
            out.append({"video_id": vid, **cur})
    return out


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args()
    rows = load_manifest()
    if args.stage in ("prepare", "all"):
        extract_keypoints(rows, args.overwrite, args.device)
        meta, kp, sc = build_windows(rows, args.person_selection,
                                     stitch=not args.no_stitch)
        print(f"samples: {len(meta)}  "
              f"({len({m['window_id'] for m in meta})} windows, "
              f"person_selection={args.person_selection}, "
              f"stitch={not args.no_stitch})")
        pkl = export_pyskl(meta, kp, sc)
        print(f"pyskl: {pkl}")
    if args.stage == "prepare":
        return 0

    pkl = PYSKL_ROOT / "annotations.pkl"
    if not pkl.exists():
        print(f"missing {pkl}; run --stage prepare first")
        return 1
    samples = paths.read_csv(WINDOW_ROOT / "windows.csv")
    sample_probs = run_model(pkl, args.device)
    if len(sample_probs) != len(samples):
        print(f"prediction count {len(sample_probs)} != samples {len(samples)}")
        return 1
    meta, probs = collapse_to_windows(samples, sample_probs)
    print(f"windows: {len(meta)} (from {len(samples)} samples)")

    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    pred_rows = []
    for i, row in enumerate(meta):
        t0 = int(row["start_frame"]) / float(row["fps"])
        t1 = int(row["end_frame"]) / float(row["fps"])
        pred_rows.append({
            "video_id": row["video_id"], "group": row["group"],
            "window_id": row["window_id"], "start_s": f"{t0:.2f}",
            "end_s": f"{t1:.2f}", "prob_fall": f"{probs[i]:.6f}",
            "coverage": row["coverage"],
        })
    paths.write_csv(
        REPORT_ROOT / "board_predictions.csv", pred_rows,
        ["video_id", "group", "window_id", "start_s", "end_s", "prob_fall", "coverage"],
    )

    by_video: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(meta):
        by_video[r["video_id"]].append(i)
    segs = segments(meta, probs, args.threshold)

    lines = [
        "# RK3576 板端视频跌倒检测（候选告警）",
        "",
        f"- 模型：`runs/stgcnpp_v2/best.pth`（D-023 基线，阈值 {args.threshold}）",
        f"- 窗口：2.0s / stride 0.5s / 覆盖率 ≥ {wbuild.load_config().min_coverage}",
        f"- 选人：`{args.person_selection}`（D-040；`max_presence` = D-018 基线）"
        + ("" if args.person_selection != "per_person_max" or args.no_stitch
           else "，已启用换 ID 拼接（D-042）"),
        f"- 视频：{len(rows)} 个，窗口 {len(meta)} 个（{len(samples)} 个人样本）",
        "- 真值与段召回见 `reports/board_eval/board_gt_eval.md`；本表是候选告警段。",
        "- 注：表中「窗口」数以 `per_person_max` 的人样本计（同一窗口可多行）；"
        "`max_presence` 下即窗口数。",
        "",
        "| 视频 | 组 | 窗口 | 最大概率 | ≥阈值窗口 | 候选告警段 (s) |",
        "|---|---|---:|---:|---:|---|",
    ]
    for vid in sorted(by_video):
        idxs = by_video[vid]
        mx = max(probs[i] for i in idxs)
        hits = sum(probs[i] >= args.threshold for i in idxs)
        vs = [s for s in segs if s["video_id"] == vid]
        if vs:
            seg_txt = "; ".join(
                f"{s['start']/float(meta[s['peak_i']]['fps']):.1f}–"
                f"{s['end']/float(meta[s['peak_i']]['fps']):.1f} (p={s['peak']:.2f})"
                for s in vs
            )
        else:
            seg_txt = "—"
        lines.append(f"| {vid.split(':')[-1]} | {vid.split(':')[1]} | {len(idxs)} | "
                     f"{mx:.3f} | {hits} | {seg_txt} |")
    lines += [
        "",
        "## 判定",
        "",
        f"- 有候选告警的视频：**{len({s['video_id'] for s in segs})}/{len(by_video)}**",
        f"- 告警段总数：{len(segs)}",
        "- 帧图证据在 `reports/board_eval/frames/`（每个视频取概率峰值窗口的中点帧）。",
        "- 无标注，因此不能给 P/R/F1；需要人工确认这些段是否真跌倒。",
        "",
    ]
    with open(REPORT_ROOT / "board_summary.md", "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))

    snapshots(meta, probs, args.threshold)
    print("\n".join(lines))
    print(f"report: {REPORT_ROOT / 'board_summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
