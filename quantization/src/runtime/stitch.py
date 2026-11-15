"""ID-handoff chaining, incrementally."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..deploy import spec


@dataclass
class _TrackGeom:
    first: int
    last: int
    # frame -> (centre_xy, diagonal)
    frames: dict[int, tuple[tuple[float, float], float]] = field(default_factory=dict)


def _centre_diag(box: np.ndarray) -> tuple[tuple[float, float], float]:
    b = np.asarray(box, dtype=np.float64)
    return (((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0),
            float(np.hypot(b[2] - b[0], b[3] - b[1])))


class Stitcher:
    """Incremental chaining."""

    def __init__(self, gap_max: int = spec.STITCH_GAP_MAX,
                 overlap_max: int = spec.STITCH_OVERLAP_MAX,
                 dist_factor: float = spec.STITCH_DIST_FACTOR) -> None:
        self.gap_max = int(gap_max)
        self.overlap_max = int(overlap_max)
        self.dist_factor = float(dist_factor)
        self._geom: dict[int, _TrackGeom] = {}
        self._chains: list[list[int]] = []
        # track_id -> index into `_chains`, once the track has been placed.
        self._chain_of: dict[int, int] = {}

    def reset(self) -> None:
        self.__init__(self.gap_max, self.overlap_max, self.dist_factor)

    # ------------------------------------------------------------------ writes

    def observe(self, track_id: int, frame_index: int, bbox: np.ndarray) -> None:
        """Record that `track_id` was seen at `frame_index` with this box."""
        g = self._geom.get(track_id)
        if g is None:
            g = _TrackGeom(first=frame_index, last=frame_index)
            self._geom[track_id] = g
        g.last = max(g.last, frame_index)
        g.frames[frame_index] = _centre_diag(bbox)

    # ------------------------------------------------------------------ chains

    def _handoff_distance(self, a: int, b: int) -> float | None:
        """Worst centre distance over the shared frames, or None if not a handoff."""
        fa, fb = self._geom[a].frames, self._geom[b].frames
        shared = sorted(set(fa) & set(fb))
        if not shared or len(shared) > self.overlap_max:
            return None
        worst = 0.0
        for t in shared:
            (ca, da), (cb, db) = fa[t], fb[t]
            dist = float(np.hypot(ca[0] - cb[0], ca[1] - cb[1]))
            if dist > self.dist_factor * (da + db):
                return None
            worst = max(worst, dist)
        return worst

    def _resolve(self) -> None:
        """Rebuild the chain list over every track seen so far."""
        self._chains = []
        self._chain_of = {}
        for b in sorted(self._geom, key=lambda k: (self._geom[k].first,
                                                   self._geom[k].last)):
            best: tuple[float, int] | None = None
            for ci, chain in enumerate(self._chains):
                a = chain[-1]
                gap = self._geom[b].first - self._geom[a].last
                if gap < -self.overlap_max or gap > self.gap_max:
                    continue
                dist = self._handoff_distance(a, b)
                if dist is None:
                    continue
                if best is None or dist < best[0]:
                    best = (dist, ci)
            if best is None:
                self._chains.append([b])
            else:
                self._chains[best[1]].append(b)
            self._chain_of[b] = len(self._chains) - 1 if best is None else best[1]

    def chains_so_far(self) -> list[list[int]]:
        """The chains as they stand right now. Recomputed on demand."""
        self._resolve()
        return [list(c) for c in self._chains]

    def members_of(self, track_id: int) -> set[int]:
        """All tracks currently believed to be the same body as `track_id`."""
        self._resolve()
        ci = self._chain_of.get(track_id)
        return set(self._chains[ci]) if ci is not None else {track_id}

    def root_of(self, track_id: int) -> int:
        """Smallest id in the chain -- the source project's chain label."""
        m = self.members_of(track_id)
        return min(m) if m else track_id
