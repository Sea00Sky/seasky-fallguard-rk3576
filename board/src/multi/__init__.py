"""per-person 多人时序管线。"""
from __future__ import annotations

from .config import MultiConfig
from .events import Counters, Event, EventBook, FrameClock, State, TrackEventState
from .pipeline import (
    STATUS_CAPPED,
    STATUS_GATED,
    STATUS_PREGATE,
    STATUS_SCORED,
    MultiPipeline,
    MultiVideoResult,
    PersonWindow,
)

__all__ = [
    "MultiConfig",
    "Counters", "Event", "EventBook", "FrameClock", "State", "TrackEventState",
    "MultiPipeline", "MultiVideoResult", "PersonWindow",
    "STATUS_SCORED", "STATUS_GATED", "STATUS_CAPPED", "STATUS_PREGATE",
]
