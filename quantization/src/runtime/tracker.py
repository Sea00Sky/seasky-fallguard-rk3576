"""Greedy IoU tracker -- a port of the source project's `src/pose/track.py`."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between xyxy boxes. `[N,4] x [M,4] -> [N,M]`."""
    if a.size == 0 or b.size == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    ax1, ay1, ax2, ay2 = (a[:, i][:, None] for i in range(4))
    bx1, by1, bx2, by2 = (b[:, i][None, :] for i in range(4))
    inter_w = np.clip(np.minimum(ax2, bx2) - np.maximum(ax1, bx1), 0, None)
    inter_h = np.clip(np.minimum(ay2, by2) - np.maximum(ay1, by1), 0, None)
    inter = inter_w * inter_h
    area_a = np.clip(ax2 - ax1, 0, None) * np.clip(ay2 - ay1, 0, None)
    area_b = np.clip(bx2 - bx1, 0, None) * np.clip(by2 - by1, 0, None)
    union = area_a + area_b - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(union > 0, inter / union, 0.0).astype(np.float32)


@dataclass
class _Track:
    track_id: int
    bbox: np.ndarray
    last_frame: int


class IoUTracker:
    """Deterministic greedy IoU tracker."""

    def __init__(self, iou_threshold: float = 0.30, max_age: int = 30) -> None:
        self.iou_threshold = float(iou_threshold)
        self.max_age = int(max_age)
        self._tracks: list[_Track] = []
        self._next_id = 1

    def reset(self) -> None:
        self._tracks = []
        self._next_id = 1

    def update(self, bboxes: np.ndarray, frame_index: int) -> list[int]:
        """Associate boxes with tracks; one track id per box, same order."""
        bboxes = np.asarray(bboxes, dtype=np.float32).reshape(-1, 4)
        self._tracks = [t for t in self._tracks
                        if frame_index - t.last_frame <= self.max_age]
        if len(bboxes) == 0:
            return []

        ids = [-1] * len(bboxes)
        if self._tracks:
            track_boxes = np.stack([t.bbox for t in self._tracks])
            ious = iou_matrix(bboxes, track_boxes)
            pairs = [
                (float(ious[i, j]), i, j)
                for i in range(ious.shape[0])
                for j in range(ious.shape[1])
                if ious[i, j] >= self.iou_threshold
            ]
            # Ties broken by (box index, track index) so the result does not
            # depend on Python's sort stability or dict ordering.
            pairs.sort(key=lambda p: (-p[0], p[1], p[2]))
            used_boxes: set[int] = set()
            used_tracks: set[int] = set()
            for _score, bi, tj in pairs:
                if bi in used_boxes or tj in used_tracks:
                    continue
                used_boxes.add(bi)
                used_tracks.add(tj)
                track = self._tracks[tj]
                track.bbox = bboxes[bi].copy()
                track.last_frame = frame_index
                ids[bi] = track.track_id

        for bi in range(len(bboxes)):
            if ids[bi] != -1:
                continue
            track = _Track(self._next_id, bboxes[bi].copy(), frame_index)
            self._next_id += 1
            self._tracks.append(track)
            ids[bi] = track.track_id
        return ids
