"""推流配置。"""
from __future__ import annotations

import os
from dataclasses import dataclass

ENV_PREFIX = "FG_STREAM_"

# vir1 的 mainpath（推流专用；检测用 index1=selfpath，见 src/capture/config.py）
DEFAULT_DEVICE = "/dev/v4l/by-path/platform-rkisp-vir1-video-index0"

# 分辨率预设：名字 -> (w, h)。全部由 ISP 直接出，不缩放。
RESOLUTIONS = {
    "hd": (1920, 1080),
    "720": (1280, 720),
    "sd": (640, 480),
}


@dataclass
class StreamConfig:
    """RTSP 推流参数。"""

    device: str = DEFAULT_DEVICE
    width: int = 1920
    height: int = 1080
    fps: int = 30
    io_mode: int = 2
    port: int = 8554
    mount: str = "/live"
    bps: int = 4_000_000
    gop: int = 30
    auth: bool = True
    user: str = "admin"
    password: str = ""
    source: str = "v4l2"

    def __post_init__(self) -> None:
        self.source = str(self.source).strip().lower()
        if self.source not in ("v4l2", "test"):
            raise ValueError(f"source 只能是 v4l2 / test，收到 {self.source!r}")
        if self.width % 2 or self.height % 2:
            raise ValueError(f"NV12 要求宽高为偶数，收到 {self.width}x{self.height}")
        if self.io_mode not in (0, 2, 4):
            raise ValueError(
                f"io_mode={self.io_mode} 在 IMX415 上确定性失败，只能用 0/2/4")
        if int(self.fps) <= 0 or int(self.bps) <= 0 or int(self.gop) <= 0:
            raise ValueError("fps / bps / gop 必须 > 0")
        if not str(self.mount).startswith("/"):
            raise ValueError(f"mount 必须以 / 开头，收到 {self.mount!r}")
        if self.auth and (not self.user or not self.password):
            raise ValueError("auth=True 时 user/password 不能为空")
        self.width = int(self.width)
        self.height = int(self.height)
        self.fps = int(self.fps)
        self.io_mode = int(self.io_mode)
        self.port = int(self.port)
        self.bps = int(self.bps)
        self.gop = int(self.gop)

    # ------------------------------------------------------------------ 工厂

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None,
                 **overrides) -> "StreamConfig":
        """读 `FG_STREAM_*`，再叠显式 overrides。配错就报错，不静默回退。"""
        src = os.environ if env is None else env
        kw: dict = {}
        spec = {
            "device": str, "width": int, "height": int, "fps": int,
            "io_mode": int, "port": int, "mount": str, "bps": int, "gop": int,
            "auth": _to_bool, "user": str, "password": str, "source": str,
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

    def describe(self) -> str:
        """一行生效值。"""
        auth = f"auth={self.user}:***" if self.auth else "auth=OFF(匿名可拉流)"
        return (f"rtsp://0.0.0.0:{self.port}{self.mount} "
                f"{self.width}x{self.height}@{self.fps} "
                f"src={self.source} bps={self.bps} gop={self.gop} {auth}")

    def as_dict(self) -> dict:
        d = dict(self.__dict__)
        if d.get("auth"):
            d["password"] = "***"
        return d


def _to_bool(raw: str) -> bool:
    v = raw.strip().lower()
    if v in ("1", "true", "yes", "on"):
        return True
    if v in ("0", "false", "no", "off"):
        return False
    raise ValueError(f"不是合法布尔值: {raw!r}")
