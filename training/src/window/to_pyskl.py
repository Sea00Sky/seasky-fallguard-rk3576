"""Export built windows to PYSKL PoseDataset annotation format."""
from __future__ import annotations

import json
import pickle
from collections import Counter
from pathlib import Path

import numpy as np

from src.data import paths
from src.window import build

SPLITS = ("train", "val", "test")


def load_windows(cfg: build.WindowConfig) -> tuple[list[dict], np.ndarray, np.ndarray, np.ndarray]:
    rows = paths.read_csv(cfg.windows_csv)
    with np.load(cfg.windows_npz, allow_pickle=False) as data:
        keypoints = data["keypoints"].astype(np.float32)
        scores = data["scores"].astype(np.float32)
        labels = data["labels"].astype(np.int8)
    if not (len(rows) == len(keypoints) == len(scores) == len(labels)):
        raise ValueError("windows.csv and windows.npz are out of sync")
    return rows, keypoints, scores, labels


def build_annotations(cfg: build.WindowConfig) -> tuple[dict, dict]:
    rows, keypoints, scores, labels = load_windows(cfg)
    annotations: list[dict] = []
    split_ids: dict[str, list[str]] = {s: [] for s in SPLITS}
    for i, row in enumerate(rows):
        split = row["split"]
        if split not in split_ids:
            raise ValueError(f"unexpected split: {split}")
        width = float(row["width"])
        height = float(row["height"])
        kp = keypoints[i].copy()
        kp[:, :, 0] *= width
        kp[:, :, 1] *= height
        label = int(labels[i])
        if label != int(row["label"]):
            raise ValueError(f"label mismatch at {row['window_id']}")
        window_id = row["window_id"]
        annotations.append({
            "frame_dir": window_id,
            "total_frames": int(cfg.resample_frames),
            "label": label,
            "keypoint": kp[None].astype(np.float32),
            "keypoint_score": scores[i][None].astype(np.float32),
            "img_shape": (int(float(row["height"])), int(float(row["width"]))),
            "original_shape": (int(float(row["height"])), int(float(row["width"]))),
            "dataset": row["dataset"],
            "video_id": row["video_id"],
            "event_id": row["event_id"],
            "split": split,
            "start_frame": int(row["start_frame"]),
            "end_frame": int(row["end_frame"]),
            "coverage": float(row["coverage"]),
        })
        split_ids[split].append(window_id)

    payload = {"split": split_ids, "annotations": annotations}
    summary = {
        "version": cfg.version,
        "windows": len(annotations),
        "labels": dict(Counter(int(x) for x in labels)),
        "splits": {s: len(ids) for s, ids in split_ids.items()},
        "split_labels": {
            s: dict(Counter(
                a["label"] for a in annotations if a["split"] == s
            ))
            for s in SPLITS
        },
    }
    return payload, summary


def _pyskl_root(cfg: build.WindowConfig) -> Path:
    root = cfg.pyskl_root if cfg.pyskl_root is not None else paths.PYSKL_V2
    return Path(root)


def write_pyskl(cfg: build.WindowConfig, payload: dict) -> Path:
    root = _pyskl_root(cfg)
    root.mkdir(parents=True, exist_ok=True)
    out = root / "annotations.pkl"
    with open(out, "wb") as fh:
        pickle.dump(payload, fh, protocol=4)
    return out


def export(cfg: build.WindowConfig) -> tuple[Path, dict]:
    payload, summary = build_annotations(cfg)
    out = write_pyskl(cfg, payload)
    summary_path = _pyskl_root(cfg) / "export_summary.json"
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    return out, summary
