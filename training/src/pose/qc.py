"""Keypoint extraction QC: coverage, confidence, flags and skeleton visuals."""
from __future__ import annotations

import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml

from src.data import paths

# COCO-17 skeleton, 0-based indices.
COCO17_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]

OK_STATUSES = {"ok", "resumed"}


def load_qc_config(path: Path | None = None) -> dict:
    path = Path(path) if path else paths.POSE_CONFIG
    with open(path, "r", encoding="utf-8-sig") as fh:
        raw = yaml.safe_load(fh)
    return raw.get("qc", {})


def _f(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _i(value, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def video_coverage(row: dict) -> float:
    n = _i(row.get("n_frames_expected"))
    return _i(row.get("frames_with_person")) / n if n else 0.0


def _canonical_interval(start: int, end: int, base: int, end_inclusive: bool) -> tuple[int, int]:
    """Convert a source interval to canonical 0-based half-open [start, end)."""
    s = start - base
    e = end - base + (1 if end_inclusive else 0)
    return s, e


# Annotation statuses that carry a fall interval, per dataset layer.
# `valid` is the raw/generated layer (MCFD raw rows are 1-based inclusive);
# `manual_annotated` is the manual interval layer (0-based inclusive).
FALL_INTERVAL_STATUSES = {
    "le2i": {"valid", "manual_annotated"},
    "mcfd": {"valid", "manual_annotated"},
    "urfd": {"manual_annotated"},
}
MCFD_FALL_STATUS = "manual_annotated"


def _interval_semantics(row: dict) -> tuple[int, bool]:
    """Frame convention for one annotation row, keyed by status ()."""
    status = (row.get("status") or "").strip()
    if status == "manual_annotated":
        return 0, True     # 0-based inclusive -> [start, end + 1)
    return 1, True         # raw export: 1-based inclusive -> [start - 1, end)


def load_interval_specs(
    annotation_dir: Path | None = None,
) -> dict[str, list[tuple[int, int, str]]]:
    """Map video_id to canonical intervals from one annotation layer ()."""
    specs: dict[str, list[tuple[int, int, str]]] = defaultdict(list)
    root = Path(annotation_dir) if annotation_dir else paths.ANNOTATIONS_V2

    for row in paths.read_csv(root / "le2i.csv"):
        status = row["status"]
        if status in FALL_INTERVAL_STATUSES["le2i"]:
            base, end_inclusive = _interval_semantics(row)
            s, e = _canonical_interval(_i(row["start_frame"]), _i(row["end_frame"]), base, end_inclusive)
            specs[row["video_id"]].append((s, e, "fall"))

    for row in paths.read_csv(root / "mcfd.csv"):
        status = (row.get("status") or "").strip()
        if status not in FALL_INTERVAL_STATUSES["mcfd"]:
            continue
        vid = f"mcfd:chute{int(float(row['chute'])):02d}:cam{int(float(row['cam']))}"
        base, end_inclusive = _interval_semantics(row)
        s, e = _canonical_interval(_i(row["start_frame"]), _i(row["end_frame"]), base, end_inclusive)
        tag = "fall" if _i(row["label"]) == 1 else "context"
        specs[vid].append((s, e, tag))

    for row in paths.read_csv(root / "urfd.csv"):
        if row["status"] in FALL_INTERVAL_STATUSES["urfd"] and row["label"] == "fall":
            s, e = _canonical_interval(
                _i(row["fall_start_frame"]), _i(row["fall_end_frame"]), 0, True
            )
            specs[row["video_id"]].append((s, e, "fall"))

    return specs


def _interval_mask(specs: list[tuple[int, int, str]], n_frames: int) -> np.ndarray:
    """Union of fall intervals, or of context intervals when no fall exists."""
    if not specs:
        return np.zeros(n_frames, dtype=bool)
    tags = {tag for _s, _e, tag in specs}
    chosen = [x for x in specs if x[2] == "fall"] if "fall" in tags else specs
    mask = np.zeros(n_frames, dtype=bool)
    for s, e, _tag in chosen:
        s, e = max(0, s), min(n_frames, e)
        if e > s:
            mask[s:e] = True
    return mask


def enrich_intervals(rows: list[dict]) -> list[str]:
    """Attach interval coverage to each row; also verify npz integrity."""
    specs = load_interval_specs()
    failures: list[str] = []
    for row in rows:
        row["interval_frames"] = 0
        row["interval_covered"] = ""
        row["interval_coverage"] = ""
        rel = row.get("npz_path", "")
        if not rel:
            failures.append(f"{row.get('video_id')}: no npz_path")
            continue
        path = paths.PROJECT_ROOT / rel
        if not path.exists():
            failures.append(f"{row.get('video_id')}: missing {rel}")
            continue
        try:
            with np.load(path, allow_pickle=False) as data:
                n_persons = data["n_persons"].astype(np.int32)
                n = _i(row.get("n_frames_expected"))
                if len(n_persons) != n:
                    failures.append(f"{row.get('video_id')}: npz frames {len(n_persons)} != {n}")
        except Exception as exc:
            failures.append(f"{row.get('video_id')}: unreadable npz ({type(exc).__name__})")
            continue
        mask = _interval_mask(specs.get(row.get("video_id"), []), len(n_persons))
        if mask.any():
            row["interval_frames"] = int(mask.sum())
            row["interval_covered"] = int((n_persons[mask] > 0).sum())
            row["interval_coverage"] = round(row["interval_covered"] / row["interval_frames"], 4)
    return failures


def collect_flags(rows: list[dict], qc: dict) -> list[dict]:
    interval_flag = float(qc.get("interval_coverage_flag", 0.70))
    sel_flag = float(qc.get("coverage_selected_flag", 0.50))
    score_flag = float(qc.get("score_video_flag", 0.30))
    flags = []
    for row in rows:
        reasons = []
        if row.get("interval_coverage") not in ("", None) and \
                _f(row.get("interval_coverage")) < interval_flag:
            reasons.append(
                f"interval_coverage {_f(row.get('interval_coverage')):.2f} < {interval_flag:.2f}"
            )
        if row.get("label") == "fall" and _f(row.get("selected_coverage")) < sel_flag:
            reasons.append(
                f"selected_coverage {_f(row.get('selected_coverage')):.2f} < {sel_flag:.2f}"
            )
        if _f(row.get("median_kpt_score")) < score_flag:
            reasons.append(
                f"median_score {_f(row.get('median_kpt_score')):.2f} < {score_flag:.2f}"
            )
        if row.get("status") not in OK_STATUSES:
            reasons.append(f"status {row.get('status')}: {row.get('error', '')}")
        if reasons:
            flags.append({"video_id": row.get("video_id"), "dataset": row.get("dataset"),
                          "reasons": "; ".join(reasons)})
    return flags


def dataset_stats(rows: list[dict]) -> dict:
    out = {}
    by_dataset: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_dataset[row.get("dataset", "?")].append(row)
    for dataset, items in sorted(by_dataset.items()):
        frames = sum(_i(r.get("n_frames_expected")) for r in items)
        with_person = sum(_i(r.get("frames_with_person")) for r in items)
        scores = [_f(r.get("median_kpt_score")) for r in items]
        sel = [_f(r.get("selected_coverage")) for r in items]
        interval_cov = [
            _f(r.get("interval_coverage")) for r in items
            if r.get("interval_coverage") not in ("", None)
        ]
        out[dataset] = {
            "videos": len(items),
            "frames": frames,
            "coverage_any": (with_person / frames) if frames else 0.0,
            "median_score": statistics.median(scores) if scores else 0.0,
            "mean_score": statistics.fmean(scores) if scores else 0.0,
            "min_score": min(scores) if scores else 0.0,
            "mean_selected": statistics.fmean(sel) if sel else 0.0,
            "min_selected": min(sel) if sel else 0.0,
            "interval_videos": len(interval_cov),
            "interval_median": statistics.median(interval_cov) if interval_cov else 0.0,
            "interval_min": min(interval_cov) if interval_cov else 0.0,
            "status": dict(Counter(r.get("status", "?") for r in items)),
        }
    return out


def global_stats(rows: list[dict]) -> dict:
    frames = sum(_i(r.get("n_frames_expected")) for r in rows)
    with_person = sum(_i(r.get("frames_with_person")) for r in rows)
    scores = [_f(r.get("median_kpt_score")) for r in rows]
    interval_cov = [
        _f(r.get("interval_coverage")) for r in rows
        if r.get("interval_coverage") not in ("", None)
    ]
    return {
        "videos": len(rows),
        "frames": frames,
        "coverage_any": (with_person / frames) if frames else 0.0,
        "median_score": statistics.median(scores) if scores else 0.0,
        "interval_videos": len(interval_cov),
        "interval_median": statistics.median(interval_cov) if interval_cov else 0.0,
        "interval_min": min(interval_cov) if interval_cov else 0.0,
        "zero_detection_videos": sum(1 for r in rows if _i(r.get("frames_with_person")) == 0),
        "status": dict(Counter(r.get("status", "?") for r in rows)),
    }


def verify_npz(rows: list[dict]) -> list[str]:
    failures = []
    for row in rows:
        rel = row.get("npz_path", "")
        if not rel:
            failures.append(f"{row.get('video_id')}: no npz_path")
            continue
        path = paths.PROJECT_ROOT / rel
        if not path.exists():
            failures.append(f"{row.get('video_id')}: missing {rel}")
            continue
        try:
            with np.load(path, allow_pickle=False) as data:
                n = _i(row.get("n_frames_expected"))
                if data["keypoints"].shape[0] != n:
                    failures.append(f"{row.get('video_id')}: npz frames != {n}")
        except Exception as exc:
            failures.append(f"{row.get('video_id')}: unreadable npz ({type(exc).__name__})")
    return failures


def _draw_skeleton(frame: np.ndarray, kpts: np.ndarray, scores: np.ndarray,
                   color=(0, 220, 0), threshold: float = 0.2) -> None:
    h, w = frame.shape[:2]
    for a, b in COCO17_EDGES:
        if scores[a] < threshold or scores[b] < threshold:
            continue
        pa = (int(round(kpts[a, 0])), int(round(kpts[a, 1])))
        pb = (int(round(kpts[b, 0])), int(round(kpts[b, 1])))
        if all(0 <= v < lim for v, lim in ((pa[0], w), (pa[1], h), (pb[0], w), (pb[1], h))):
            cv2.line(frame, pa, pb, color, 2, cv2.LINE_AA)
    for i in range(len(kpts)):
        if scores[i] < threshold:
            continue
        p = (int(round(kpts[i, 0])), int(round(kpts[i, 1])))
        if 0 <= p[0] < w and 0 <= p[1] < h:
            cv2.circle(frame, p, 3, (0, 0, 255), -1, cv2.LINE_AA)


def make_visuals(rows: list[dict], qc: dict, out_dir: Path) -> list[Path]:
    per_dataset = int(qc.get("visuals_per_dataset", 8))
    per_video = int(qc.get("visuals_per_video", 4))
    threshold = float(qc.get("low_score_threshold", 0.20))
    out_dir.mkdir(parents=True, exist_ok=True)
    by_dataset: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if row.get("status") in OK_STATUSES:
            by_dataset[row["dataset"]].append(row)
    manifest = {r["video_id"]: r for r in
                paths.read_csv(paths.MANIFESTS / "v2" / "videos.csv")}

    written: list[Path] = []
    for dataset, items in sorted(by_dataset.items()):
        items = sorted(items, key=lambda r: (r.get("label") != "fall", r.get("video_id", "")))
        if len(items) <= per_dataset:
            chosen = items
        else:
            step = (len(items) - 1) / (per_dataset - 1)
            chosen = [items[int(round(k * step))] for k in range(per_dataset)]
        for row in chosen:
            npz = paths.PROJECT_ROOT / row["npz_path"]
            with np.load(npz, allow_pickle=False) as data:
                keypoints = data["keypoints"].astype(np.float32)
                scores = data["scores"].astype(np.float32)
                bboxes = data["bboxes"].astype(np.float32)
                track_ids = data["track_ids"]
                n_persons = data["n_persons"].astype(np.int32)
                crop = _i(data["meta_crop_offset_x"])
            n_frames = keypoints.shape[0]
            if n_frames == 0:
                continue
            targets = sorted({int(round(k * (n_frames - 1) / max(1, per_video - 1)))
                              for k in range(per_video)})
            video_path = paths.PROJECT_ROOT / "datasets" / "raw" / \
                manifest[row["video_id"]]["relative_path"]
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                continue
            selected_id = _i(row.get("selected_track_id"), -1)
            idx = 0
            saved = 0
            while saved < len(targets):
                ok, frame = cap.read()
                if not ok:
                    break
                if idx in targets:
                    if crop:
                        frame = frame[:, crop:]
                    for p in range(int(n_persons[idx])):
                        x1, y1, x2, y2 = bboxes[idx, p]
                        color = (0, 220, 0) if track_ids[idx, p] == selected_id else (160, 160, 160)
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 1)
                        _draw_skeleton(frame, keypoints[idx, p], scores[idx, p],
                                       color=color, threshold=threshold)
                    label = f"{row['video_id']} f={idx}"
                    cv2.putText(frame, label, (6, 18), cv2.FONT_HERSHEY_SIMPLEX,
                                0.45, (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.putText(frame, label, (6, 18), cv2.FONT_HERSHEY_SIMPLEX,
                                0.45, (0, 0, 0), 1, cv2.LINE_AA)
                    name = f"{dataset}__{_safe(row['video_id'])}__f{idx:05d}.jpg"
                    path = out_dir / name
                    # cv2.imwrite uses an ANSI fopen on Windows and fails silently
                    # for absolute paths containing non-ASCII characters.
                    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    if not ok:
                        raise RuntimeError(f"jpeg encode failed: {path}")
                    buf.tofile(str(path))
                    written.append(path)
                    saved += 1
                idx += 1
            cap.release()
    return written


def _safe(video_id: str) -> str:
    return video_id.replace(":", "__").replace("/", "_").replace("\\", "_")
