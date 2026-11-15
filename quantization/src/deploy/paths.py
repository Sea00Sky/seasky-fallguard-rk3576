"""Paths for this project and read-only pointers into the source project."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC = PROJECT_ROOT / "src"
SCRIPTS = PROJECT_ROOT / "scripts"

WEIGHTS = PROJECT_ROOT / "weights"
MODELS = PROJECT_ROOT / "models"
REPORTS = PROJECT_ROOT / "reports"
REFERENCE = REPORTS / "reference"
PARITY = REPORTS / "parity"
CALIB = PROJECT_ROOT / "calib"
MEMORY = PROJECT_ROOT / "memory"

YOLO_PT = WEIGHTS / "yolo26n-pose.pt"
STGCN_PTH = WEIGHTS / "stgcnpp_v2_best.pth"
STGCN_CFG = WEIGHTS / "stgcnpp_v2_config_used.yaml"

# ------------------------------------------------------------- source project

SOURCE_ROOT = PROJECT_ROOT.parent / "YOLOV26n-pose+ST-GCN++"

SOURCE_MANIFEST_BOARD = SOURCE_ROOT / "datasets" / "manifests" / "board" / "videos.csv"
SOURCE_KEYPOINTS_BOARD = SOURCE_ROOT / "datasets" / "keypoints" / "board"
# `relative_path` in the board manifest is relative to `datasets/raw/`.
SOURCE_RAW_BOARD = SOURCE_ROOT / "datasets" / "raw"

SOURCE_MANIFEST_V2 = SOURCE_ROOT / "datasets" / "manifests" / "v2" / "videos.csv"
SOURCE_RAW_V2 = SOURCE_ROOT / "datasets" / "raw"
SOURCE_GT_BOARD = SOURCE_ROOT / "datasets" / "annotations" / "board" / "board_fall_intervals.csv"
SOURCE_WINDOWS_V2 = SOURCE_ROOT / "datasets" / "windows" / "v2"
SOURCE_REPORTS_BOARD = SOURCE_ROOT / "reports" / "board_eval"
SOURCE_REF_PER_SEGMENT = SOURCE_REPORTS_BOARD / "coverage_gate" / "per_segment.csv"
SOURCE_REF_PER_VIDEO_NOFALL = SOURCE_REPORTS_BOARD / "coverage_gate" / "per_video_nofall.csv"

# Frozen by the source project; used to prove we read the same bytes.
EXPECTED_SHA256 = {
    "yolo26n-pose.pt": "eb3bb8268828aeaf515cec23a4bfafd793944a86fe9af94ba7823609c14522a9",
    "stgcnpp_v2_best.pth": "8cd5e74e138c1fb7f57a6b9fd782ac4167e8ecabd050ef385738cfd66b7ab29d",
}


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def read_csv(path: Path) -> list[dict]:
    with open(path, "r", newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2)
        fh.write("\n")


def read_json(path: Path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def sanitise_video_id(video_id: str) -> str:
    """`board:done:clip_0831_172408` -> `board__done__clip_0831_172408`."""
    return video_id.replace(":", "__").replace("/", "_").replace("\\", "_")


def keypoint_npz(video_id: str) -> Path:
    return SOURCE_KEYPOINTS_BOARD / f"{sanitise_video_id(video_id)}.npz"


def verify_weights() -> dict:
    """Refuse to proceed on a weight whose sha256 is not the frozen one."""
    out = {}
    for name, want in EXPECTED_SHA256.items():
        path = WEIGHTS / name
        if not path.exists():
            raise FileNotFoundError(f"missing weight: {path}")
        got = sha256_file(path)
        if got != want:
            raise RuntimeError(
                f"{name}: sha256 {got} != frozen {want}; this weight is not the "
                "one the source project's numbers came from"
            )
        out[name] = {"sha256": got, "bytes": path.stat().st_size}
    return out
