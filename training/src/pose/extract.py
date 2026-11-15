"""Extract COCO-17 keypoints from manifest videos with YOLO26n-pose."""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml

from src.data import paths
from src.pose.track import IoUTracker

INDEX_FIELDS = [
    "video_id", "dataset", "event_id", "split", "label",
    "n_frames_expected", "n_frames_read", "width", "height", "crop_offset_x",
    "frames_with_person", "selected_track_id", "selected_track_frames",
    "selected_coverage", "mean_kpt_score", "median_kpt_score", "low_score_ratio",
    "elapsed_s", "status", "error", "model_sha256", "npz_path",
]


@dataclass
class PoseConfig:
    model_path: Path
    imgsz: int
    imgsz_by_dataset: dict
    conf: float
    iou: float
    device: str
    quantize: int | None
    batch_frames: int
    max_persons: int
    person_class: int
    track_iou: float
    max_age: int
    output_root: Path
    index_path: Path
    urfd_half: str
    low_score_threshold: float


def load_config(path: Path | None = None) -> PoseConfig:
    path = Path(path) if path else paths.POSE_CONFIG
    with open(path, "r", encoding="utf-8-sig") as fh:
        raw = yaml.safe_load(fh)
    model = raw["model"]
    inf = raw["inference"]
    trk = raw["tracking"]
    out = raw["output"]
    qc = raw.get("qc", {})
    return PoseConfig(
        model_path=paths.PROJECT_ROOT / model["path"],
        imgsz=int(inf["imgsz"]),
        imgsz_by_dataset={
            str(k): int(v) for k, v in (inf.get("imgsz_by_dataset") or {}).items()
        },
        conf=float(inf["conf"]),
        iou=float(inf["iou"]),
        device=str(inf["device"]),
        quantize=(int(inf["quantize"]) if inf.get("quantize") not in (None, "") else None),
        batch_frames=int(inf["batch_frames"]),
        max_persons=int(inf["max_persons"]),
        person_class=int(inf["person_class"]),
        track_iou=float(trk["iou_threshold"]),
        max_age=int(trk["max_age"]),
        output_root=paths.PROJECT_ROOT / out["root"],
        index_path=paths.PROJECT_ROOT / out["index"],
        urfd_half=str(out.get("urfd_half", "right")),
        low_score_threshold=float(qc.get("low_score_threshold", 0.20)),
    )


def sanitise_video_id(video_id: str) -> str:
    return video_id.replace(":", "__").replace("/", "_").replace("\\", "_")


def imgsz_for(cfg: PoseConfig, dataset: str) -> int:
    return int(cfg.imgsz_by_dataset.get(dataset, cfg.imgsz))


def npz_path_for(cfg: PoseConfig, row: dict) -> Path:
    return cfg.output_root / row["dataset"] / f"{sanitise_video_id(row['video_id'])}.npz"


def _empty_frame_arrays(max_persons: int) -> tuple[np.ndarray, ...]:
    return (
        np.zeros((max_persons, 17, 2), dtype=np.float16),
        np.zeros((max_persons, 17), dtype=np.float16),
        np.zeros((max_persons, 4), dtype=np.float32),
        np.full((max_persons,), -1, dtype=np.int32),
    )


def _prepare_frame(frame: np.ndarray, use_half: str, urfd_half: str) -> tuple[np.ndarray, int]:
    if use_half == urfd_half:
        offset = frame.shape[1] // 2
        return np.ascontiguousarray(frame[:, offset:]), offset
    return frame, 0


def _boxes_from_result(result) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return bboxes [n,4], keypoints [n,17,2], scores [n,17] sorted by box conf."""
    boxes = getattr(result, "boxes", None)
    kpts = getattr(result, "keypoints", None)
    if boxes is None or kpts is None or len(boxes) == 0:
        return (
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0, 17, 2), dtype=np.float32),
            np.zeros((0, 17), dtype=np.float32),
        )
    xyxy = boxes.xyxy.detach().cpu().numpy().astype(np.float32)
    conf = boxes.conf.detach().cpu().numpy().astype(np.float32)
    data = kpts.data.detach().cpu().numpy().astype(np.float32)  # [n,17,3]
    order = np.argsort(-conf, kind="stable")
    return xyxy[order], data[order, :, :2], data[order, :, 2]


def _save_npz(path: Path, arrays: dict, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    payload = {**arrays, **{f"meta_{k}": np.array(v) for k, v in meta.items()}}
    with open(tmp, "wb") as fh:
        np.savez_compressed(fh, **payload)
    tmp.replace(path)


def npz_is_complete(path: Path, n_frames: int, max_persons: int) -> bool:
    try:
        with np.load(path, allow_pickle=False) as data:
            kp = data["keypoints"]
            return (
                kp.shape == (n_frames, max_persons, 17, 2)
                and data["scores"].shape == (n_frames, max_persons, 17)
                and data["track_ids"].shape == (n_frames, max_persons)
            )
    except Exception:
        return False


def summarise(arrays: dict, low_score_threshold: float) -> dict:
    """Per-video QC numbers computed from the stored arrays."""
    n_persons = arrays["n_persons"].astype(np.int32)
    scores = arrays["scores"].astype(np.float32)
    track_ids = arrays["track_ids"]
    n_frames = len(n_persons)
    valid = np.arange(scores.shape[1])[None, :] < n_persons[:, None]
    if valid.any():
        kpt_scores = scores[valid]
        mean_score = float(kpt_scores.mean())
        median_score = float(np.median(kpt_scores))
        low_ratio = float((kpt_scores < low_score_threshold).mean())
    else:
        mean_score = median_score = 0.0
        low_ratio = 1.0

    tids = track_ids[track_ids >= 0]
    if tids.size:
        values, counts = np.unique(tids, return_counts=True)
        best = int(np.lexsort((values, -counts))[0])
        selected_id = int(values[best])
        selected_frames = int(counts[best])
    else:
        selected_id = -1
        selected_frames = 0
    return {
        "frames_with_person": int((n_persons > 0).sum()),
        "selected_track_id": selected_id,
        "selected_track_frames": selected_frames,
        "selected_coverage": round(selected_frames / n_frames, 4) if n_frames else 0.0,
        "mean_kpt_score": round(mean_score, 4),
        "median_kpt_score": round(median_score, 4),
        "low_score_ratio": round(low_ratio, 4),
    }


def extract_video(model, cfg: PoseConfig, row: dict, model_sha: str, low_conf: bool) -> dict:
    video_path = paths.PROJECT_ROOT / "datasets" / "raw" / row["relative_path"]
    expected = int(row["n_frames"])
    out_path = npz_path_for(cfg, row)
    result = {
        "video_id": row["video_id"],
        "dataset": row["dataset"],
        "event_id": row["event_id"],
        "split": row["split"],
        "label": row["label"],
        "n_frames_expected": expected,
        "n_frames_read": 0,
        "width": 0,
        "height": 0,
        "crop_offset_x": 0,
        "frames_with_person": 0,
        "selected_track_id": -1,
        "selected_track_frames": 0,
        "selected_coverage": 0.0,
        "mean_kpt_score": 0.0,
        "median_kpt_score": 0.0,
        "low_score_ratio": 0.0,
        "elapsed_s": 0.0,
        "status": "",
        "error": "",
        "model_sha256": model_sha,
        "npz_path": paths.rel_to_project(out_path),
    }

    if not low_conf and npz_is_complete(out_path, expected, cfg.max_persons):
        with np.load(out_path, allow_pickle=False) as data:
            arrays = {k: data[k] for k in ("keypoints", "scores", "bboxes", "track_ids", "n_persons")}
            result.update(summarise(arrays, cfg.low_score_threshold))
            result["n_frames_read"] = int(data["keypoints"].shape[0])
            result["width"] = int(data["meta_width"])
            result["height"] = int(data["meta_height"])
            result["crop_offset_x"] = int(data["meta_crop_offset_x"])
        result["status"] = "resumed"
        return result

    start = time.perf_counter()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        result["status"] = "read_error"
        result["error"] = "VideoCapture not opened"
        return result

    tracker = IoUTracker(cfg.track_iou, cfg.max_age)
    video_imgsz = imgsz_for(cfg, row["dataset"])
    kp_frames: list[np.ndarray] = []
    score_frames: list[np.ndarray] = []
    box_frames: list[np.ndarray] = []
    id_frames: list[np.ndarray] = []
    counts: list[int] = []
    width = height = crop_offset = 0
    frame_index = 0
    batch_frames: list[np.ndarray] = []
    batch_indices: list[int] = []

    def flush() -> None:
        nonlocal batch_frames, batch_indices
        if not batch_frames:
            return
        results = model.predict(
            batch_frames,
            imgsz=video_imgsz,
            conf=cfg.conf,
            iou=cfg.iou,
            device=cfg.device,
            quantize=cfg.quantize if str(cfg.device) != "cpu" else None,
            classes=[cfg.person_class],
            verbose=False,
        )
        for res, idx in zip(results, batch_indices):
            bboxes, kpts, scores = _boxes_from_result(res)
            if len(bboxes) > cfg.max_persons:
                bboxes, kpts, scores = (
                    bboxes[: cfg.max_persons],
                    kpts[: cfg.max_persons],
                    scores[: cfg.max_persons],
                )
            ids = tracker.update(bboxes, idx)
            kp, sc, bx, tid = _empty_frame_arrays(cfg.max_persons)
            n = len(bboxes)
            if n:
                kp[:n] = kpts.astype(np.float16)
                sc[:n] = scores.astype(np.float16)
                bx[:n] = bboxes
                tid[:n] = np.asarray(ids, dtype=np.int32)
            kp_frames.append(kp)
            score_frames.append(sc)
            box_frames.append(bx)
            id_frames.append(tid)
            counts.append(n)
        batch_frames = []
        batch_indices = []

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            prepared, offset = _prepare_frame(frame, row.get("use_half", ""), cfg.urfd_half)
            if frame_index == 0:
                crop_offset = offset
                width, height = prepared.shape[1], prepared.shape[0]
            batch_frames.append(prepared)
            batch_indices.append(frame_index)
            frame_index += 1
            if len(batch_frames) >= cfg.batch_frames:
                flush()
        flush()
    except Exception as exc:  # pragma: no cover - defensive
        result["status"] = "inference_error"
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result
    finally:
        cap.release()

    result["n_frames_read"] = frame_index
    result["width"] = width
    result["height"] = height
    result["crop_offset_x"] = crop_offset

    if frame_index != expected:
        result["status"] = "frame_count_mismatch"
        result["error"] = f"manifest {expected}, read {frame_index}"
        return result
    if frame_index == 0:
        result["status"] = "read_error"
        result["error"] = "no frames decoded"
        return result

    arrays = {
        "keypoints": np.stack(kp_frames),
        "scores": np.stack(score_frames),
        "bboxes": np.stack(box_frames),
        "track_ids": np.stack(id_frames),
        "n_persons": np.asarray(counts, dtype=np.int16),
    }
    meta = {
        "fps": row["fps"],
        "n_frames": frame_index,
        "width": width,
        "height": height,
        "source_width": int(row["width"]),
        "source_height": int(row["height"]),
        "crop_offset_x": crop_offset,
        "model_sha256": model_sha,
        "imgsz": video_imgsz,
        "conf": cfg.conf,
        "iou": cfg.iou,
        "quantize": cfg.quantize,
        "max_persons": cfg.max_persons,
        "max_age": cfg.max_age,
        "track_iou": cfg.track_iou,
    }
    _save_npz(out_path, arrays, meta)
    result.update(summarise(arrays, cfg.low_score_threshold))
    result["status"] = "ok"
    result["elapsed_s"] = round(time.perf_counter() - start, 2)
    return result


def select_rows(dataset: str = "all", limit: int = 0) -> list[dict]:
    rows = paths.read_csv(paths.MANIFESTS / "v2" / "videos.csv")
    rows = [r for r in rows if r["readable"] == "True"]
    if dataset != "all":
        rows = [r for r in rows if r["dataset"] == dataset]
    if limit > 0:
        picked: list[dict] = []
        seen: dict[str, int] = {}
        for row in rows:
            n = seen.get(row["dataset"], 0)
            if n < limit:
                picked.append(row)
                seen[row["dataset"]] = n + 1
        rows = picked
    return rows
