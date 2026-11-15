"""执行后端。板端只有 RKNN 一种，PC 侧对账时可以再加 ONNX 版。"""
from __future__ import annotations

from .rknn_backend import Detection, StgcnBackend, YoloBackend

__all__ = ["Detection", "YoloBackend", "StgcnBackend"]
