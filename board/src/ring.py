"""Per-track rolling keypoint buffer."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import spec


@dataclass
class TrackWindow:
    """One person's view of one window, in the shape the offline path gives."""

    track_id: int
    start: int
    end: int
    keypoints: np.ndarray
    scores: np.ndarray
    present: np.ndarray
    n_interp: int

    @property
    def coverage(self) -> float:
        """Fraction of the window's frames where the person was actually seen."""
        return float(self.present.mean()) if len(self.present) else 0.0

    def model_input(self, width: int, height: int) -> np.ndarray:
        """`(1, 1, T, V, 3)` float32, ready for the ONNX graph."""
        kp = spec.normalise_unit(self.keypoints, width, height)
        idx = spec.resample_indices(len(kp), spec.RESAMPLE_FRAMES)
        sample = spec.to_model_input(kp[idx].astype(np.float32),
                                     self.scores[idx].astype(np.float32))
        return spec.stack_batch([sample])


class _Ring:
    """One track's history. Fixed size, frame-stamped slots."""

    __slots__ = ("kp", "sc", "present", "frame_at", "first", "last")

    def __init__(self, size: int) -> None:
        self.kp = np.zeros((size, 17, 2), np.float32)
        self.sc = np.zeros((size, 17), np.float32)
        self.present = np.zeros(size, bool)
        self.frame_at = np.full(size, -1, np.int64)
        self.first = -1
        self.last = -1

    def write(self, frame_index: int, kp, sc) -> None:
        """Record one frame. `kp is None` is a hole, not a sighting."""
        size = len(self.frame_at)
        slot = frame_index % size
        self.frame_at[slot] = frame_index
        if kp is not None:
            self.last = frame_index
            if self.first < 0:
                self.first = frame_index
        if kp is None:
            self.present[slot] = False
            self.kp[slot] = 0.0
            self.sc[slot] = 0.0
        else:
            self.present[slot] = True
            self.kp[slot] = kp
            self.sc[slot] = sc

    def _lo(self, start: int) -> int:
        """First frame of `[start, end)` this track can possibly know about."""
        return max(start, self.first)

    def holds(self, start: int, end: int) -> bool:
        """True when every frame of `[start, end)` is either held or pre-birth."""
        size = len(self.frame_at)
        if end - start > size:
            return False
        lo = self._lo(start)
        if lo >= end:
            # Born after the window closed: nothing of this window to serve.
            return False
        frames = np.arange(lo, end)
        return bool(np.array_equal(self.frame_at[frames % size], frames))

    def read(self, start: int, end: int):
        """Window data; frames before birth read as absent holes."""
        size = len(self.frame_at)
        n = end - start
        lo = self._lo(start)
        if lo >= end:
            return (np.zeros((n, 17, 2), np.float32),
                    np.zeros((n, 17), np.float32),
                    np.zeros(n, bool))
        slots = np.arange(lo, end) % size
        kp = np.zeros((n, 17, 2), np.float32)
        sc = np.zeros((n, 17), np.float32)
        present = np.zeros(n, bool)
        off = lo - start
        kp[off:] = self.kp[slots]
        sc[off:] = self.sc[slots]
        present[off:] = self.present[slots]
        return kp, sc, present


class RingBuffer:
    """Fixed-length per-track history."""

    def __init__(self, window_frames: int, max_age: int | None = None) -> None:
        self.window_frames = int(window_frames)
        self.max_age = int(window_frames - 1 if max_age is None else max_age)
        self._rings: dict[int, _Ring] = {}

    def reset(self) -> None:
        self._rings = {}

    # ------------------------------------------------------------------ writes

    def update(self, track_id: int, frame_index: int,
               keypoints: np.ndarray | None,
               scores: np.ndarray | None) -> None:
        """Record one frame for one track."""
        ring = self._rings.get(track_id)
        if ring is None:
            ring = self._rings[track_id] = _Ring(self.window_frames)
        ring.write(frame_index, keypoints, scores)

    def mark_missing(self, frame_index: int) -> None:
        """Write a hole for every live track not already updated this frame."""
        for ring in self._rings.values():
            if ring.last < frame_index:
                ring.write(frame_index, None, None)

    def expire(self, frame_index: int) -> list[int]:
        """Drop rings whose track has been gone longer than `max_age`."""
        dead = [tid for tid, r in self._rings.items()
                if frame_index - r.last > self.max_age]
        for tid in dead:
            del self._rings[tid]
        return dead

    # ------------------------------------------------------------------- reads

    def has(self, track_id: int) -> bool:
        return track_id in self._rings

    def track_ids(self) -> list[int]:
        return sorted(self._rings)

    def last_frame(self, track_id: int) -> int:
        return self._rings[track_id].last if track_id in self._rings else -1

    def latest(self, track_id: int):
        """最近一次真正看到的 `(kp[17,2], sc[17])`（副本）；从没看到过则 None。"""
        ring = self._rings.get(track_id)
        if ring is None or ring.last < 0:
            return None
        slot = ring.last % len(ring.frame_at)
        if ring.frame_at[slot] != ring.last or not ring.present[slot]:
            return None
        return ring.kp[slot].copy(), ring.sc[slot].copy()

    def can_take(self, track_id: int, start: int, end: int) -> bool:
        ring = self._rings.get(track_id)
        return ring is not None and ring.holds(start, end)

    def take(self, track_id: int, start: int, end: int) -> TrackWindow:
        """Extract one window, holes interpolated exactly as offline does."""
        ring = self._rings.get(track_id)
        if ring is None or not ring.holds(start, end):
            raise ValueError(
                f"track {track_id} does not hold [{start}, {end}) "
                f"(has [{ring.first if ring else -1}, "
                f"{ring.last if ring else -1}], size {self.window_frames})")
        kp, sc, present = ring.read(start, end)
        kp, _interp, n_interp = spec.interpolate_gaps(
            present, kp, spec.MAX_INTERP_GAP_FRAMES)
        return TrackWindow(track_id=track_id, start=start, end=end,
                           keypoints=kp, scores=sc, present=present,
                           n_interp=n_interp)

    def merge(self, members: set[int], start: int, end: int) -> TrackWindow | None:
        """One series over several tracks -- the online `_group_series`."""
        win = end - start
        kp = np.zeros((win, 17, 2), np.float32)
        sc = np.zeros((win, 17), np.float32)
        present = np.zeros(win, bool)
        for tid in sorted(members):
            ring = self._rings.get(tid)
            if ring is None or not ring.holds(start, end):
                continue
            k, s, p = ring.read(start, end)
            take = p & ~present
            kp[take] = k[take]
            sc[take] = s[take]
            present[take] = True
        if not present.any():
            return None
        kp, _interp, n_interp = spec.interpolate_gaps(
            present, kp, spec.MAX_INTERP_GAP_FRAMES)
        return TrackWindow(track_id=min(members), start=start, end=end,
                           keypoints=kp, scores=sc, present=present,
                           n_interp=n_interp)
