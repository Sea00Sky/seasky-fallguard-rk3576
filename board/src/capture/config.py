"""采集层配置。"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

ENV_PREFIX = "FG_CAPTURE_"

# by-path 绑定的是 rkisp-vir1（跨重启稳定），index1 = selfpath。
DEFAULT_DEVICE = "/dev/v4l/by-path/platform-rkisp-vir1-video-index1"


@dataclass
class CaptureConfig:
    """采集参数。"""

    backend: str = "gst"
    device: str = DEFAULT_DEVICE
    width: int = 640
    height: int = 480
    fps: int = 30
    io_mode: int = 2
    appsink_buffers: int = 3
    queue_buffers: int = 4
    read_timeout_s: float = 1.0
    shm_socket: str | None = None
    shm_size: int = 32_000_000
    raw_file: str | None = None

    def __post_init__(self) -> None:
        self.backend = str(self.backend).strip().lower()
        if self.backend not in ("gst", "opencv"):
            raise ValueError(
                f"backend 只能是 'gst' 或 'opencv'，收到 {self.backend!r}")
        if self.width <= 0 or self.height <= 0:
            raise ValueError(f"分辨率非法: {self.width}x{self.height}")
        if self.width % 2 or self.height % 2:
            raise ValueError(
                f"NV12 要求宽高为偶数，收到 {self.width}x{self.height}")
        if int(self.fps) <= 0:
            raise ValueError(f"fps 必须 > 0，收到 {self.fps}")
        if int(self.appsink_buffers) < 1:
            raise ValueError(f"appsink_buffers 必须 >= 1，收到 {self.appsink_buffers}")
        if int(self.queue_buffers) < 1:
            raise ValueError(f"queue_buffers 必须 >= 1，收到 {self.queue_buffers}")
        if self.io_mode not in (0, 2, 4):
            raise ValueError(
                f"io_mode={self.io_mode} 在 IMX415 上确定性失败。"
                f"只能用 0(auto) / 2(mmap) / 4(dmabuf)")
        if self.shm_socket is not None and not str(self.shm_socket).strip():
            self.shm_socket = None
        self.width = int(self.width)
        self.height = int(self.height)
        self.fps = int(self.fps)
        self.io_mode = int(self.io_mode)
        self.appsink_buffers = int(self.appsink_buffers)
        self.queue_buffers = int(self.queue_buffers)
        self.read_timeout_s = float(self.read_timeout_s)
        self.shm_size = int(self.shm_size)

    # ------------------------------------------------------------------ 工厂

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None,
                 **overrides) -> "CaptureConfig":
        """读 `FG_CAPTURE_*`，再叠显式 overrides（overrides 优先）。"""
        src = os.environ if env is None else env
        kw: dict = {}
        spec = {
            "backend": str,
            "device": str,
            "width": int,
            "height": int,
            "fps": int,
            "io_mode": int,
            "appsink_buffers": int,
            "queue_buffers": int,
            "read_timeout_s": float,
            "shm_socket": str,
            "shm_size": int,
            "raw_file": str,
        }
        for name, cast in spec.items():
            key = ENV_PREFIX + name.upper()
            raw = src.get(key)
            if raw is None or raw.strip() == "":
                continue
            try:
                kw[name] = cast(raw.strip())
            except ValueError as exc:
                raise ValueError(
                    f"环境变量 {key}={raw!r} 不是合法 {cast.__name__}") from exc
        kw.update(overrides)
        return cls(**kw)

    # ------------------------------------------------------------------ 展示

    def describe(self) -> str:
        """一行生效值。 纪律：配置外置 + 打印生效值，否则现场永远在猜。"""
        src = (f"raw_file={self.raw_file}" if self.raw_file
               else f"device={self.device} io_mode={self.io_mode}")
        shm = f"shm={self.shm_socket}" if self.shm_socket else "shm=off"
        return (f"backend={self.backend} {src} "
                f"{self.width}x{self.height}@{self.fps} "
                f"appsink={self.appsink_buffers} queue={self.queue_buffers} {shm}")

    def as_dict(self) -> dict:
        return {
            "backend": self.backend,
            "device": self.device,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "io_mode": self.io_mode,
            "appsink_buffers": self.appsink_buffers,
            "queue_buffers": self.queue_buffers,
            "read_timeout_s": self.read_timeout_s,
            "shm_socket": self.shm_socket,
            "shm_size": self.shm_size,
            "raw_file": self.raw_file,
        }
