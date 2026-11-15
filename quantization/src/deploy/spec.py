"""The executable pre-processing spec."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# --------------------------------------------------------------------- frontend
#
# configs/pose.yaml [inference]. Measured : 640 beats 1280 on the
# board-like clips (L-013). Do not change without a new data layer.
IMGSZ = 640
CONF = 0.25
IOU = 0.7
MAX_PERSONS = 4
PERSON_CLASS = 0

# configs/pose.yaml [tracking]: deterministic greedy IoU, independent of
# ultralytics' tracker.
TRACK_IOU = 0.30
MAX_AGE = 30

DURATION_S = 2.0
STRIDE_S = 0.5
GUARD_S = 0.5
NEGATIVE_MIN_DISTANCE_S = 0.5
RESAMPLE_FRAMES = 64
MIN_COVERAGE = 0.70
MAX_INTERP_GAP_FRAMES = 10
COORDINATE_NORM = "image_size"   # x / width, y / height -> [0, 1]

STITCH_GAP_MAX = 5
STITCH_OVERLAP_MAX = 1
STITCH_DIST_FACTOR = 0.25

NUM_CLASSES = 2
CLIP_LEN = 64
NUM_PERSON = 1
IN_CHANNELS = 3
NUM_JOINTS = 17
MODEL_INPUT_SHAPE = (1, NUM_PERSON, CLIP_LEN, NUM_JOINTS, IN_CHANNELS)

# PreNormalize2D(threshold=0.01, mode='fix'): a joint whose score is at or below
# this is moved to the origin *after* the [-1, 1] mapping. The score channel
# itself is passed through unchanged.
SCORE_THRESHOLD = 0.01

THRESHOLD = 0.5

# Board clips are 640x480 @ 30 fps (datasets/manifests/board/videos.csv).
BOARD_FPS = 30.0


@dataclass(frozen=True)
class WindowSpec:
    """Frame counts derived from fps. Mirrors `window.build.frames_for`."""

    fps: float
    duration_s: float = DURATION_S
    stride_s: float = STRIDE_S
    guard_s: float = GUARD_S
    resample_frames: int = RESAMPLE_FRAMES
    min_coverage: float = MIN_COVERAGE
    max_interp_gap_frames: int = MAX_INTERP_GAP_FRAMES

    @property
    def window_frames(self) -> int:
        return frames_for(self.fps, self.duration_s)

    @property
    def stride_frames(self) -> int:
        return frames_for(self.fps, self.stride_s)

    @property
    def guard_frames(self) -> int:
        return frames_for(self.fps, self.guard_s)


def frames_for(fps: float, seconds: float) -> int:
    """`max(1, int(round(fps * seconds)))` -- identical to the source project."""
    return max(1, int(round(fps * seconds)))


def window_starts(n_frames: int, spec: WindowSpec) -> range:
    """Candidate window starts. No coverage gate is applied here ()."""
    win = spec.window_frames
    if n_frames < win:
        return range(0)
    return range(0, n_frames - win + 1, spec.stride_frames)


# ------------------------------------------------------------------ transforms


def resample_indices(n_frames: int, target: int = RESAMPLE_FRAMES) -> np.ndarray:
    """`np.linspace(0, n-1, target)` then `rint` -- the source project's rule."""
    if n_frames <= 0:
        raise ValueError("empty window")
    return np.rint(np.linspace(0, n_frames - 1, target)).astype(np.int64)


def normalise_unit(keypoints: np.ndarray, width: int, height: int) -> np.ndarray:
    """Pixel coordinates -> [0, 1] by the *frame* size."""
    kp = np.array(keypoints, dtype=np.float32, copy=True)
    kp[:, :, 0] /= max(1, width)
    kp[:, :, 1] /= max(1, height)
    return kp


def to_model_input(keypoints_unit: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """[0, 1] unit coordinates -> the (T, V, 3) tensor the ONNX model eats."""
    kp = np.array(keypoints_unit, dtype=np.float32, copy=True)
    kp = kp * 2.0 - 1.0
    low = scores <= SCORE_THRESHOLD
    kp[:, :, 0][low] = 0.0
    kp[:, :, 1][low] = 0.0
    return np.concatenate([kp, scores[:, :, None]], axis=-1).astype(np.float32)


def stack_batch(samples: list[np.ndarray]) -> np.ndarray:
    """(T, V, 3) samples -> (N, 1, T, V, 3) model input, batch on axis 0."""
    arr = np.stack(samples, axis=0)                    # (N, T, V, C)
    return arr[:, None].astype(np.float32)             # (N, M=1, T, V, C)


def softmax2(logits: np.ndarray) -> np.ndarray:
    """Stable softmax over the last axis. Kept on the CPU (2 classes)."""
    z = np.asarray(logits, dtype=np.float64)
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


# ---------------------------------------------------------------- interpolation


def interpolate_gaps(
    present: np.ndarray, keypoints: np.ndarray, max_gap: int = MAX_INTERP_GAP_FRAMES
) -> tuple[np.ndarray, np.ndarray, int]:
    """Fill missing frames of one track by linear interpolation."""
    kp = np.array(keypoints, dtype=np.float32, copy=True)
    interp = np.zeros(len(present), dtype=bool)
    n = len(present)
    i = 0
    while i < n:
        if present[i]:
            i += 1
            continue
        j = i
        while j < n and not present[j]:
            j += 1
        gap = j - i
        if i > 0 and j < n and gap <= max_gap:
            for k in range(i, j):
                a = (k - i + 1) / (gap + 1)
                kp[k] = (1.0 - a) * kp[i - 1] + a * kp[j]
            interp[i:j] = True
        i = j
    return kp, interp, int(interp.sum())


# ---------------------------------------------------------------- labelling


def label_for(
    w0: int, w1: int, falls: list[tuple[int, int]], guard_frames: int
) -> int:
    """1 = fall, 0 = non_fall, -1 = ambiguous (drop). onset rule."""
    if not falls:
        return 0
    if any(w0 <= s < w1 for s, _e in falls):
        return 1
    if any(max(0, min(w1, e) - max(w0, s)) > 0 for s, e in falls):
        return -1
    distance = min(min(abs(w0 - e), abs(s - w1)) for s, e in falls)
    return 0 if distance >= guard_frames else -1


# ---------------------------------------------------------------- evaluation


def overlap(w0: int, w1: int, s: int, e: int) -> int:
    """Frames shared by window [w0, w1) and ground-truth interval [s, e]."""
    return max(0, min(w1, e + 1) - max(w0, s))
