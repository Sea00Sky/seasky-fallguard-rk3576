"""Normalize URFD video-level labels and detect the side-by-side layout."""
from __future__ import annotations

import statistics
from pathlib import Path

import cv2
import numpy as np

from . import paths


def _chroma(frame: np.ndarray) -> float:
    f = frame.astype(np.float32)
    return float(np.mean(f.max(axis=2) - f.min(axis=2)))


def detect_layout(video: Path) -> dict:
    info = paths.video_info(video)
    result = {
        "layout": "unknown",
        "use_half": "",
        "chroma_left": "",
        "chroma_right": "",
        "separator_dark_cols": "",
        "layout_note": "",
    }
    if not info["readable"]:
        result["layout_note"] = info["error"]
        return result

    cap = cv2.VideoCapture(str(video))
    frames = []
    try:
        for _ in range(30):
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
    finally:
        cap.release()
    if not frames:
        result["layout_note"] = "no frames decoded"
        return result

    h, w = frames[0].shape[:2]
    if w < 2 * h or w < 500:
        result["layout"] = "single_view"
        result["use_half"] = "full"
        result["layout_note"] = f"{w}x{h} is not side-by-side"
        return result

    mid = w // 2
    left_chromas = []
    right_chromas = []
    dark_counts = []
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        col_mean = gray.mean(axis=0)
        lo, hi = max(0, mid - 14), min(w, mid + 14)
        dark_counts.append(int((col_mean[lo:hi] < 12).sum()))
        left_chromas.append(_chroma(frame[:, : mid - 6]))
        right_chromas.append(_chroma(frame[:, mid + 6 :]))

    lc = statistics.median(left_chromas)
    rc = statistics.median(right_chromas)
    dc = statistics.median(dark_counts)

    result["chroma_left"] = round(lc, 3)
    result["chroma_right"] = round(rc, 3)
    result["separator_dark_cols"] = int(dc)

    l_rgb, r_rgb = lc > 6.0, rc > 6.0
    if l_rgb and not r_rgb:
        result.update(layout="side_by_side:left_rgb_right_gray", use_half="left")
    elif r_rgb and not l_rgb:
        result.update(layout="side_by_side:left_gray_right_rgb", use_half="right")
    elif l_rgb and r_rgb:
        result.update(layout="side_by_side:both_rgb", use_half="right",
                      layout_note="both halves look RGB; right half used by convention")
    elif not l_rgb and not r_rgb:
        result.update(layout="side_by_side:both_gray", use_half="",
                      layout_note="no RGB half detected")
    else:
        result.update(layout="side_by_side:ambiguous", use_half="",
                      layout_note=f"left_chroma={lc:.2f} right_chroma={rc:.2f}")
    return result


def _rows_for_dir(root: Path, subdir: str, status: str, label: str) -> list[dict]:
    out = []
    folder = root / subdir
    if not folder.exists():
        return out
    for video in sorted(folder.glob("*.mp4")):
        info = paths.video_info(video)
        layout = detect_layout(video)
        out.append(
            {
                "video_id": f"urfd:{video.stem}",
                "split_dir": subdir,
                "label": label,
                "status": status,
                "fall_start_frame": "",
                "fall_end_frame": "",
                "video_path": video.relative_to(paths.RAW).as_posix(),
                "fps": info["fps"],
                "n_frames": info["n_frames"],
                "width": info["width"],
                "height": info["height"],
                "readable": info["readable"],
                "layout": layout["layout"],
                "use_half": layout["use_half"],
                "chroma_left": layout["chroma_left"],
                "chroma_right": layout["chroma_right"],
                "separator_dark_cols": layout["separator_dark_cols"],
                "notes": layout["layout_note"],
            }
        )
    return out


def parse() -> list[dict]:
    root = paths.RAW_URFD
    if not root.exists():
        raise FileNotFoundError(root)
    rows = []
    rows += _rows_for_dir(root, "fall", "needs_annotation", "fall")
    rows += _rows_for_dir(root, "adl", "all_negative", "non_fall")
    rows += _rows_for_dir(root / "fall", "conv", "duplicate_conv", "fall")
    rows.sort(key=lambda r: (r["split_dir"], r["video_id"]))
    return rows


def stats(rows: list[dict]) -> dict:
    from collections import Counter

    return {
        "rows_total": len(rows),
        "status_counts": dict(Counter(r["status"] for r in rows)),
        "layout_counts": dict(Counter(r["layout"] for r in rows)),
        "use_half_counts": dict(Counter(r["use_half"] for r in rows)),
        "unreadable": [r["video_id"] for r in rows if not r["readable"]],
    }
