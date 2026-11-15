"""Streaming fall-detection runtime, as the board will run it."""
from __future__ import annotations

from .detector import Detector, Detection
from .gate import Gate, WindowScore, score_windows
from .pipeline import Pipeline, PipelineConfig, VideoResult
from .ring import RingBuffer, TrackWindow
from .stitch import Stitcher
from .tracker import IoUTracker

__all__ = [
    "Detector", "Detection",
    "Gate", "WindowScore", "score_windows",
    "Pipeline", "PipelineConfig", "VideoResult",
    "RingBuffer", "TrackWindow",
    "Stitcher",
    "IoUTracker",
]
