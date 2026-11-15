"""Skeleton augmentation for the v2 window pipeline."""
from __future__ import annotations

import numpy as np

SCORE_INDEX = 2
SCORE_EPS = 0.01


class SkeletonAugment:
    """Random similarity transform on every person/clip."""

    def __init__(
        self, rotate: float = 0.0, scale: float = 0.0,
        translate: float = 0.0, jitter: float = 0.0,
    ) -> None:
        if min(rotate, scale, translate, jitter) < 0:
            raise ValueError("augmentation strengths must be non-negative")
        self.rotate = float(rotate)
        self.scale = float(scale)
        self.translate = float(translate)
        self.jitter = float(jitter)

    def __repr__(self) -> str:
        return (
            f"SkeletonAugment(rotate={self.rotate}, scale={self.scale}, "
            f"translate={self.translate}, jitter={self.jitter})"
        )

    def __call__(self, results: dict) -> dict:
        keypoint = results["keypoint"]
        if keypoint.ndim != 4:
            raise ValueError(f"expected (M, T, V, C) keypoints, got {keypoint.shape}")
        num_person, _, _, channels = keypoint.shape
        if channels < 2:
            raise ValueError(f"need at least x/y channels, got {channels}")

        for m in range(num_person):
            xy = keypoint[m, ..., :2]
            if channels > SCORE_INDEX:
                valid = keypoint[m, ..., SCORE_INDEX] > SCORE_EPS
            else:
                valid = np.ones(xy.shape[:2], dtype=bool)
            if not valid.any():
                continue

            points = xy[valid]
            centre = points.mean(axis=0)
            size = float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))

            factor = 1.0 + np.random.uniform(-self.scale, self.scale)
            if self.rotate:
                theta = np.random.uniform(-self.rotate, self.rotate)
                cos, sin = np.cos(theta), np.sin(theta)
                rot = np.array([[cos, -sin], [sin, cos]])
            else:
                rot = np.eye(2)
            shift = np.random.uniform(-self.translate, self.translate, size=2)

            # Invalid joints stay at (0, 0): the score channel marks them absent,
            # and moving them would turn "missing" into a fake detected position.
            transformed = xy.copy()
            transformed[valid] = (xy[valid] - centre) @ rot.T * factor + centre + shift
            if self.jitter and size > 0:
                sigma = self.jitter * size
                transformed[valid] += np.random.normal(0.0, sigma, size=(int(valid.sum()), 2))
            keypoint[m, ..., :2] = transformed

        results["keypoint"] = keypoint
        return results
