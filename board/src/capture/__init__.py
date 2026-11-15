"""采集层：相机 → BGR 帧。"""
from __future__ import annotations

from .base import Frame, FrameSource
from .config import DEFAULT_DEVICE, CaptureConfig

__all__ = ["Frame", "FrameSource", "CaptureConfig", "DEFAULT_DEVICE",
           "open_source"]


def open_source(cfg: CaptureConfig) -> FrameSource:
    """按 `cfg.backend` 造一个采集源。失败就抛，不静默降级。"""
    if cfg.backend == "gst":
        from .gst_source import GstSource

        return GstSource(cfg)
    if cfg.backend == "opencv":
        from .opencv_source import OpenCvSource

        return OpenCvSource(cfg)
    raise ValueError(f"未知 backend: {cfg.backend!r}")   # __post_init__ 已挡过
