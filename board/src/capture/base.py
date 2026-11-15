"""采集层的公共契约：`Frame` 与 `FrameSource`。"""
from __future__ import annotations

import abc
from dataclasses import dataclass

import numpy as np


@dataclass
class Frame:
    """一帧，已转成 BGR。"""

    seq: int
    pts_ns: int | None
    bgr: np.ndarray

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.bgr.shape)


class FrameSource(abc.ABC):
    """采集源。生命周期：`start` → 若干次 `read` → `stop`。"""

    @abc.abstractmethod
    def start(self) -> None:
        """建管线并进入出帧状态。失败抛异常，不许静默降级。"""

    @abc.abstractmethod
    def read(self, timeout_s: float | None = None) -> Frame | None:
        """取一帧。超时返回 None；真错误抛异常。"""

    @abc.abstractmethod
    def stop(self) -> None:
        """停机。必须可重复调用（析构/异常路径都会走到）。"""

    @property
    @abc.abstractmethod
    def stats(self) -> dict:
        """运行计数。字段随实现不同，但丢帧类计数必须如实暴露。"""

    # ------------------------------------------------------------ 上下文管理

    def __enter__(self) -> "FrameSource":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()
