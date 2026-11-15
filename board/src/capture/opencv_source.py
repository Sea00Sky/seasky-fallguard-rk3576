"""OpenCV 采集源。"""
from __future__ import annotations

import time

from .base import Frame, FrameSource
from .config import CaptureConfig


class OpenCvSource(FrameSource):
    """`cv2.VideoCapture` 直读 V4L2 节点。"""

    def __init__(self, cfg: CaptureConfig) -> None:
        self.cfg = cfg
        self._cap = None
        self._seq = 0
        self._t_start = 0.0
        self._t_stop = 0.0
        self._fails = 0

    def start(self) -> None:
        import cv2

        if self.cfg.raw_file:
            raise ValueError(
                "raw_file 只支持 backend=gst（OpenCV 读不了裸 NV12）。"
                "逐位对拍请用 --backend gst。")
        cap = cv2.VideoCapture(self.cfg.device, cv2.CAP_V4L2)
        if not cap.isOpened():
            raise RuntimeError(
                f"打不开采集设备 {self.cfg.device}。"
                f"先查：① 节点在不在（MIPI 用 by-path，别写死编号）"
                f"② 是否被另一个进程独占（同节点 V4L2 独占，第二进程 EBUSY）")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.height)
        cap.set(cv2.CAP_PROP_FPS, self.cfg.fps)
        self._cap = cap
        self._t_start = time.monotonic()

    def read(self, timeout_s: float | None = None) -> Frame | None:
        # timeout_s 被忽略：cap.read 会阻塞到有帧。见模块 docstring。
        if self._cap is None:
            raise RuntimeError("read() 之前必须先 start()")
        ok, frame = self._cap.read()
        if not ok or frame is None:
            self._fails += 1
            raise RuntimeError(
                f"cap.read() 失败（第 {self._fails} 次）。"
                f"相机掉线 / ISP 出错 / 节点被抢都会这样")
        seq = self._seq
        self._seq += 1
        return Frame(seq=seq, pts_ns=None, bgr=frame)

    def stop(self) -> None:
        if self._cap is None:
            return
        self._t_stop = time.monotonic()
        self._cap.release()
        self._cap = None

    @property
    def stats(self) -> dict:
        dt = ((self._t_stop or time.monotonic()) - self._t_start) if self._t_start else 0.0
        return {
            "backend": "opencv",
            "read": self._seq,
            "dropped_total": None, # 拿不到，不许编一个 0
            "gaps": None,
            "max_gap_ms": None,
            "fails": self._fails,
            "seconds": round(dt, 3),
            "fps": round(self._seq / dt, 2) if dt > 0 else 0.0,
        }
