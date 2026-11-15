"""Project paths and unicode-safe IO helpers."""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import cv2
import numpy as np

# src/data/paths.py -> parents[2] == project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

CONFIGS = PROJECT_ROOT / "configs"
SRC = PROJECT_ROOT / "src"
SCRIPTS = PROJECT_ROOT / "scripts"

DATA = PROJECT_ROOT / "datasets"
RAW = DATA / "raw"
ANNOTATIONS = DATA / "annotations"
MANUAL = ANNOTATIONS / "manual"
MCFD_FALL_INTERVALS = MANUAL / "mcfd_fall_intervals.csv"
MANIFESTS = DATA / "manifests"
SPLITS = DATA / "splits"
KEYPOINTS = DATA / "keypoints"
WINDOWS = DATA / "windows"
PYSKL = DATA / "pyskl"
QC = DATA / "qc"

# Versioned derived layers. v1 and v2 are frozen; v3 replaces the MCFD
# fall intervals with the manual layer and reuses keypoints/v2 unchanged.
ANNOTATIONS_V1 = ANNOTATIONS
ANNOTATIONS_V2 = ANNOTATIONS / "v2"
ANNOTATIONS_V3 = ANNOTATIONS / "v3"
MANIFESTS_V1 = MANIFESTS
MANIFESTS_V2 = MANIFESTS / "v2"
MANIFESTS_V3 = MANIFESTS / "v3"
SPLITS_V1 = SPLITS / "v1.json"
SPLITS_V2 = SPLITS / "v2.json"
SPLITS_V3 = SPLITS / "v3.json"
KEYPOINTS_V2 = KEYPOINTS / "v2"
KEYPOINTS_V3 = KEYPOINTS / "v3"
WINDOWS_V2 = WINDOWS / "v2"
WINDOWS_V3 = WINDOWS / "v3"
PYSKL_V2 = PYSKL / "v2"
PYSKL_V3 = PYSKL / "v3"

LAYER_VERSIONS = ("v2", "v3")


def layer(version: str) -> dict:
    """Resolve the five derived-layer roots for one pipeline version."""
    if version not in LAYER_VERSIONS:
        raise ValueError(f"unknown layer version: {version}")
    return {
        "annotations": ANNOTATIONS / version,
        "manifests": MANIFESTS / version,
        "split": SPLITS / f"{version}.json",
        "windows": WINDOWS / version,
        "pyskl": PYSKL / version,
        # v3 deliberately reuses the v2 keypoints: npz files are keyed by
        # video_id only and do not depend on annotation intervals.
        "keypoints": KEYPOINTS_V2 if version == "v3" else KEYPOINTS / version,
    }

WEIGHTS = PROJECT_ROOT / "weights"
RUNS = PROJECT_ROOT / "runs"
REPORTS = PROJECT_ROOT / "reports"
REPORTS_MCFD_INTERVALS = REPORTS / "qc_mcfd_intervals"
MEMORY = PROJECT_ROOT / "memory"

# Source datasets outside the project (read-only). 三个公开数据集不随仓库发布，
# 用 FG_SOURCE_* 指向本机下载位置；不设时按 datasets/source 下的约定目录找。
SOURCE_ROOT = Path(os.environ.get("FG_SOURCE_ROOT", PROJECT_ROOT / "datasets" / "source"))

SOURCE_LE2I = Path(os.environ.get("FG_SOURCE_LE2I", SOURCE_ROOT / "le2i" / "archive"))
SOURCE_MCFD = Path(os.environ.get("FG_SOURCE_MCFD", SOURCE_ROOT / "mcfd" / "archive"))
SOURCE_URFD = Path(os.environ.get("FG_SOURCE_URFD", SOURCE_ROOT / "urfd" / "URFD_video"))

RAW_LE2I = RAW / "le2i" / "archive"
RAW_MCFD = RAW / "mcfd" / "dataset"
RAW_MCFD_CSV = RAW / "mcfd" / "data_tuple3.csv"
RAW_MCFD_PDF = RAW / "mcfd" / "technicalReport.pdf"
RAW_URFD = RAW / "urfd"

VIDEO_EXTS = {".avi", ".mp4", ".mov", ".mkv"}

POSE_CONFIG = CONFIGS / "pose.yaml"


def ensure_dirs() -> None:
    for p in (
        ANNOTATIONS,
        MANIFESTS,
        SPLITS,
        KEYPOINTS,
        WINDOWS,
        WINDOWS_V2,
        PYSKL,
        PYSKL_V2,
        QC,
        WEIGHTS,
        RUNS,
        REPORTS,
        MEMORY,
        MEMORY / "worklog",
        MEMORY / "experiments",
    ):
        p.mkdir(parents=True, exist_ok=True)


def ensure_layer_dirs(version: str) -> None:
    """Create the writable directories of one layer; never touches frozen v2."""
    roots = layer(version)
    for key in ("annotations", "manifests", "windows", "pyskl"):
        roots[key].mkdir(parents=True, exist_ok=True)
    SPLITS.mkdir(parents=True, exist_ok=True)


def imread_unicode(path: Path | str) -> np.ndarray | None:
    """cv2.imread fails silently on some Windows Chinese paths."""
    try:
        buf = np.fromfile(str(path), dtype=np.uint8)
        if buf.size == 0:
            return None
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)
    except Exception:
        return None


def video_info(path: Path | str) -> dict:
    """Probe a video. Never raises; returns readable=False on failure."""
    path = Path(path)
    info = {
        "readable": False,
        "fps": 0.0,
        "n_frames": 0,
        "width": 0,
        "height": 0,
        "duration_s": 0.0,
        "error": "",
    }
    cap = None
    try:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            info["error"] = "VideoCapture not opened"
            return info
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        n = int(round(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        w = int(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
        h = int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        info.update(
            readable=n > 0 and w > 0 and h > 0,
            fps=fps,
            n_frames=n,
            width=w,
            height=h,
            duration_s=(n / fps) if fps > 0 else 0.0,
        )
        if not info["readable"]:
            info["error"] = "metadata incomplete"
    except Exception as exc:  # pragma: no cover - defensive
        info["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if cap is not None:
            cap.release()
    return info


def read_first_frame(path: Path | str) -> np.ndarray | None:
    cap = None
    try:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            return None
        ok, frame = cap.read()
        return frame if ok else None
    except Exception:
        return None
    finally:
        if cap is not None:
            cap.release()


def iter_files(root: Path, exts: Iterable[str] | None = None) -> Iterator[Path]:
    exts = {e.lower() for e in exts} if exts else None
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            p = Path(dirpath) / name
            if exts is None or p.suffix.lower() in exts:
                yield p


def write_csv(path: Path, rows: Sequence[dict], fieldnames: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict]:
    with open(path, "r", newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2)
        fh.write("\n")


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def human_bytes(n: int) -> str:
    value = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.2f} {unit}"
        value /= 1024
    return f"{value:.2f} TB"


def rel_to_project(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()
