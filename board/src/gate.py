"""Coverage gate and per-window probability collapse."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import spec


@dataclass
class WindowScore:
    """One window's verdict."""

    window_id: int
    start: int
    end: int
    prob_fall: float
    track_id: int
    coverage: float
    n_candidates: int

    @property
    def hit(self) -> bool:
        return self.prob_fall >= spec.THRESHOLD


def score_windows(candidates: list[tuple[int, int, int]],
                  scored: list[tuple[int, int, float, float]],
                  min_coverage: float = spec.MIN_COVERAGE) -> list[WindowScore]:
    """Collapse person scores into window verdicts."""
    best: dict[int, tuple[float, int, float]] = {}
    n_eligible: dict[int, int] = {}
    for wid, track_id, coverage, p in scored:
        if coverage < min_coverage:
            continue
        n_eligible[wid] = n_eligible.get(wid, 0) + 1
        p = float(p)
        if wid not in best or p > best[wid][0]:
            best[wid] = (p, int(track_id), float(coverage))

    out: list[WindowScore] = []
    for wid, start, end in candidates:
        p, track_id, cov = best.get(wid, (0.0, -1, 0.0))
        out.append(WindowScore(window_id=wid, start=start, end=end,
                               prob_fall=p, track_id=track_id, coverage=cov,
                               n_candidates=n_eligible.get(wid, 0)))
    return out


class Gate:
    """Stateful wrapper kept for the board loop's convenience."""

    def __init__(self, min_coverage: float = spec.MIN_COVERAGE,
                 threshold: float = spec.THRESHOLD) -> None:
        self.min_coverage = float(min_coverage)
        self.threshold = float(threshold)

    def admits(self, coverage: float) -> bool:
        return coverage >= self.min_coverage

    def decide(self, prob_fall: float) -> bool:
        return prob_fall >= self.threshold
