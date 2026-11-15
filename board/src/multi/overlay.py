"""实时渲染画布 —— 按需调用，只在有人观看时跑。"""
from __future__ import annotations

import cv2
import numpy as np

from .. import spec

# COCO-17 骨架连线（顺序与旧部署 `visualize.py:SKELETON_EDGES` 一致）
SKELETON_EDGES = [
    (0, 1), (0, 2),          # 鼻子 → 左眼/右眼
    (1, 3), (2, 4),          # 眼 → 耳
    (5, 6),                  # 锁骨
    (5, 11), (6, 12),        # 肩 → 髋
    (11, 12),                # 骨盆
    (5, 7), (7, 9),          # 左臂
    (6, 8), (8, 10),         # 右臂
    (11, 13), (13, 15),      # 左腿
    (12, 14), (14, 16),      # 右腿
]

COLOR_LIMB = (0, 255, 0)        # 绿 —— 骨骼连线（老师样式）
COLOR_KP = (0, 0, 255)          # 红 —— 关键点
COLOR_BOX = (160, 160, 160)     # 灰 —— 包围框（细，不抢骨架）
COLOR_TEXT = (255, 255, 255)    # 白
COLOR_OK = (0, 255, 0)          # 绿 —— 概率低于阈值
COLOR_ALERT = (0, 0, 255)       # 红 —— 概率过阈值 / 跌倒横幅
COLOR_TEXT_BG = (0, 0, 0)

KP_CONF = 0.3                   # 与 KP_CONF_THRESHOLD_TS 一致


def render(frame_bgr: np.ndarray, persons: list[dict], *, falling: bool = False,
           fps: float | None = None, canvas: np.ndarray | None = None
           ) -> np.ndarray:
    """把所有人画到一帧上并返回画布。不改 `frame_bgr`。"""
    if (canvas is None or canvas.shape != frame_bgr.shape
            or canvas.dtype != frame_bgr.dtype):
        canvas = np.empty_like(frame_bgr)
    np.copyto(canvas, frame_bgr)          # ← 整帧唯一一次拷贝

    for p in persons:
        _draw_person(canvas, p)

    if falling:
        _banner(canvas, "*** FALL! ***")
    if fps is not None:
        _fps(canvas, fps)
    return canvas


def _draw_person(canvas: np.ndarray, p: dict) -> None:
    """原地画一个人：细框 + 绿线 + 红点 + 左上角标签。"""
    kp = p["keypoints"]
    x1, y1, x2, y2 = p["bbox"]
    diag = max(1.0, ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5)
    radius = max(2, round(diag / 60.0))       # 随人缩放
    thickness = max(1, radius // 2)

    cv2.rectangle(canvas, (int(x1), int(y1)), (int(x2), int(y2)),
                  COLOR_BOX, 1, lineType=cv2.LINE_AA)

    for i, j in SKELETON_EDGES:
        if i >= len(kp) or j >= len(kp):
            continue
        if kp[i, 2] <= KP_CONF or kp[j, 2] <= KP_CONF:
            continue
        cv2.line(canvas, (int(kp[i, 0]), int(kp[i, 1])),
                 (int(kp[j, 0]), int(kp[j, 1])),
                 COLOR_LIMB, thickness, lineType=cv2.LINE_AA)

    for pt in kp:
        if pt[2] <= KP_CONF:
            continue
        cv2.circle(canvas, (int(pt[0]), int(pt[1])), radius, COLOR_KP, -1,
                   lineType=cv2.LINE_AA)

    _label(canvas, int(x1), int(y1), _caption(p), color=prob_color(p.get("prob")))


def _caption(p: dict) -> str:
    """`#3 p=0.87 cov=0.92` —— 纯 ASCII，`cv2.putText` 画不了中文。"""
    parts = [f"#{p['track_id']}"]
    if p.get("prob") is not None:
        parts.append(f"p={p['prob']:.2f}")
    if p.get("coverage") is not None:
        parts.append(f"cov={p['coverage']:.2f}")
    return " ".join(parts)


def _label(canvas: np.ndarray, x: int, y: int, text: str,
           color=COLOR_TEXT) -> None:
    """在 (x, y) 上方画带黑底的文字；贴边时自动往下挪，不出画。"""
    scale, thick = 0.45, 1
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    ty = y - 4
    if ty - th < 0:                       # 顶到画面外就画在框内
        ty = y + th + 4
    tx = max(0, min(x, canvas.shape[1] - tw - 4))
    cv2.rectangle(canvas, (tx - 2, ty - th - 2), (tx + tw + 2, ty + base),
                  COLOR_TEXT_BG, -1)
    cv2.putText(canvas, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, scale,
                color, thick, lineType=cv2.LINE_AA)


def _banner(canvas: np.ndarray, text: str) -> None:
    """顶部红色半透明横幅。"""
    h, w = canvas.shape[:2]
    band = canvas[0:44]
    red = np.empty_like(band)
    red[:] = COLOR_ALERT
    cv2.addWeighted(band, 0.6, red, 0.4, 0, band)   # 原地混合，不再 copy 整帧
    cv2.putText(canvas, text, (16, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                COLOR_TEXT, 2, lineType=cv2.LINE_AA)


def _fps(canvas: np.ndarray, fps: float) -> None:
    cv2.putText(canvas, f"{fps:.1f} fps", (canvas.shape[1] - 96, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TEXT, 1,
                lineType=cv2.LINE_AA)


def prob_color(prob: float | None) -> tuple[int, int, int]:
    """概率标签颜色：过阈值红、否则绿、没有概率灰。"""
    if prob is None:
        return COLOR_BOX
    return COLOR_ALERT if prob >= spec.THRESHOLD else COLOR_OK
