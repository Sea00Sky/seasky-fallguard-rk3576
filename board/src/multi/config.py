"""per-track 管线的可配参数。"""
from __future__ import annotations

import os
from dataclasses import dataclass

ENV_PREFIX = "FG_MULTI_"


@dataclass
class MultiConfig:
    """per-track 判定与 ST-GCN 调度的全部可调参数。"""

    confirm_windows: int = 2
    retract_windows: int = 4
    cooldown_s: float = 30.0
    max_persons_stgcn: int = 3
    stgcn_queue_max: int = 4
    geo_veto_ar_max: float = 0.0
    geo_veto_bottom_shift: float = 0.0
    geo_veto_r_end: float = 0.0
    geo_veto_min_box: int = 20
    strong_confirm: float = 0.0
 # ---- 几何前置闸门（，默认关闭）----
    pregate: int = 0
    pregate_sample: int = 0
 # ---- 窗口长度（帧，0 = duration_s×fps）----
    window_frames: int = 0
 # ---- ② 身份桥接（，默认关闭）----
    stitch_bridge: int = 0
 # ---- ③ 强窗破冷却（，默认关闭）----
    cooldown_break_prob: float = 0.0
    cooldown_break_windows: int = 2

    def __post_init__(self) -> None:
        if int(self.confirm_windows) < 1:
            raise ValueError(
                f"confirm_windows 必须 >= 1，收到 {self.confirm_windows}。"
                f"=1 是旧口径（逐窗告警），=2 才是 1 秒确认")
        if int(self.retract_windows) < 1:
            raise ValueError(f"retract_windows 必须 >= 1，收到 {self.retract_windows}")
        if float(self.cooldown_s) < 0:
            raise ValueError(f"cooldown_s 不能为负，收到 {self.cooldown_s}")
        if int(self.max_persons_stgcn) < 1:
            raise ValueError(
                f"max_persons_stgcn 必须 >= 1，收到 {self.max_persons_stgcn}")
        if float(self.geo_veto_ar_max) < 0:
            raise ValueError(
                f"geo_veto_ar_max 不能为负（0 = 关闭），收到 {self.geo_veto_ar_max}")
        if float(self.geo_veto_bottom_shift) < 0:
            raise ValueError(
                f"geo_veto_bottom_shift 不能为负（0 = 不参与），"
                f"收到 {self.geo_veto_bottom_shift}")
        if float(self.geo_veto_r_end) < 0:
            raise ValueError(
                f"geo_veto_r_end 不能为负（0 = 不参与），收到 {self.geo_veto_r_end}")
        if int(self.geo_veto_min_box) < 0:
            raise ValueError(
                f"geo_veto_min_box 不能为负，收到 {self.geo_veto_min_box}")
        if float(self.strong_confirm) < 0:
            raise ValueError(
                f"strong_confirm 不能为负（0 = 关闭），收到 {self.strong_confirm}")
        if int(self.pregate) not in (0, 1):
            raise ValueError(f"pregate 只能是 0 或 1，收到 {self.pregate}")
        if int(self.pregate_sample) < 0:
            raise ValueError(
                f"pregate_sample 不能为负（0 = 关闭影子采样），"
                f"收到 {self.pregate_sample}")
        if int(self.window_frames) < 0:
            raise ValueError(
                f"window_frames 不能为负（0 = duration_s×fps），"
                f"收到 {self.window_frames}")
        if int(self.stitch_bridge) not in (0, 1):
            raise ValueError(
                f"stitch_bridge 只能是 0 或 1，收到 {self.stitch_bridge}")
        if float(self.cooldown_break_prob) < 0:
            raise ValueError(
                f"cooldown_break_prob 不能为负（0 = 关闭），"
                f"收到 {self.cooldown_break_prob}")
        if int(self.cooldown_break_windows) < 1:
            raise ValueError(
                f"cooldown_break_windows 必须 >= 1，"
                f"收到 {self.cooldown_break_windows}")
        self.confirm_windows = int(self.confirm_windows)
        self.retract_windows = int(self.retract_windows)
        self.cooldown_s = float(self.cooldown_s)
        self.max_persons_stgcn = int(self.max_persons_stgcn)
        self.stgcn_queue_max = int(self.stgcn_queue_max)
        self.geo_veto_ar_max = float(self.geo_veto_ar_max)
        self.geo_veto_bottom_shift = float(self.geo_veto_bottom_shift)
        self.geo_veto_r_end = float(self.geo_veto_r_end)
        self.geo_veto_min_box = int(self.geo_veto_min_box)
        self.strong_confirm = float(self.strong_confirm)
        self.pregate = int(self.pregate)
        self.pregate_sample = int(self.pregate_sample)
        self.window_frames = int(self.window_frames)
        self.stitch_bridge = int(self.stitch_bridge)
        self.cooldown_break_prob = float(self.cooldown_break_prob)
        self.cooldown_break_windows = int(self.cooldown_break_windows)

 # ------------------------------------------------------------------ 工厂

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None, **overrides) -> "MultiConfig":
        """读 `FG_MULTI_*`，再叠显式 overrides（overrides 优先）。"""
        src = os.environ if env is None else env
        kw: dict = {}
        spec = {
            "confirm_windows": int,
            "retract_windows": int,
            "cooldown_s": float,
            "max_persons_stgcn": int,
            "stgcn_queue_max": int,
            "geo_veto_ar_max": float,
            "geo_veto_bottom_shift": float,
            "geo_veto_r_end": float,
            "geo_veto_min_box": int,
            "strong_confirm": float,
            "pregate": int,
            "pregate_sample": int,
            "window_frames": int,
            "stitch_bridge": int,
            "cooldown_break_prob": float,
            "cooldown_break_windows": int,
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
        """一行生效值。纪律：配置外置 + 打印生效值，否则现场永远在猜。"""
        return (f"confirm_windows={self.confirm_windows} "
                f"retract_windows={self.retract_windows} "
                f"cooldown_s={self.cooldown_s:g} "
                f"max_persons_stgcn={self.max_persons_stgcn} "
                f"stgcn_queue_max={self.stgcn_queue_max} "
                f"geo_veto_ar_max={self.geo_veto_ar_max:g} "
                f"geo_veto_bottom_shift={self.geo_veto_bottom_shift:g} "
                f"geo_veto_r_end={self.geo_veto_r_end:g} "
                f"geo_veto_min_box={self.geo_veto_min_box} "
                f"strong_confirm={self.strong_confirm:g} "
                f"pregate={self.pregate} "
                f"pregate_sample={self.pregate_sample} "
                f"window_frames={self.window_frames} "
                f"stitch_bridge={self.stitch_bridge} "
                f"cooldown_break_prob={self.cooldown_break_prob:g} "
                f"cooldown_break_windows={self.cooldown_break_windows}")

    def as_dict(self) -> dict:
        """云端策略参数集 —— 键集是跨端契约，不要往这里加东西。"""
        return {
            "confirm_windows": self.confirm_windows,
            "retract_windows": self.retract_windows,
            "cooldown_s": self.cooldown_s,
            "max_persons_stgcn": self.max_persons_stgcn,
            "stgcn_queue_max": self.stgcn_queue_max,
            "geo_veto_ar_max": self.geo_veto_ar_max,
            "geo_veto_bottom_shift": self.geo_veto_bottom_shift,
            "geo_veto_r_end": self.geo_veto_r_end,
            "geo_veto_min_box": self.geo_veto_min_box,
            "strong_confirm": self.strong_confirm,
        }
