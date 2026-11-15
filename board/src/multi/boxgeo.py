"""检测框几何：窗口级 `ar_max`（宽高比最大值）。"""
from __future__ import annotations

from collections import deque

import numpy as np


class BoxHistory:
    """per-track 的检测框历史，只保留最近 `window_frames` 帧。"""

    __slots__ = ("window_frames", "_h", "_hbase")

    def __init__(self, window_frames: int) -> None:
        self.window_frames = int(window_frames)
        self._h: dict[int, deque] = {}
        # 该 track 至今见过的最大框高（≈这个人站立时的身高）。
        # `r_end = h_end / h_base` 就是「窗口结束时他还剩多高」——
        # 站起来了 ⇒ 接近 1；还躺在地上 ⇒ 0.3~0.5。
        self._hbase: dict[int, float] = {}

    def reset(self) -> None:
        self._h = {}
        self._hbase = {}

    # ------------------------------------------------------------------ 写

    def add(self, track_id: int, frame_index: int, box) -> None:
        """记一帧的框。`box` 是 `[x1, y1, x2, y2]`。"""
        tid = int(track_id)
        d = self._h.get(tid)
        if d is None:
            d = self._h[tid] = deque()
        d.append((int(frame_index),
                  float(box[0]), float(box[1]), float(box[2]), float(box[3])))
        h = float(box[3]) - float(box[1])
        if h > self._hbase.get(tid, 0.0):
            self._hbase[tid] = h
        cutoff = int(frame_index) - self.window_frames + 1
        while d and d[0][0] < cutoff:
            d.popleft()

    def expire(self, keep_ids) -> int:
        """丢掉已经不在 ring 里的 track。返回清掉的条数。"""
        keep = {int(t) for t in keep_ids}
        gone = [t for t in self._h if t not in keep]
        for t in gone:
            self._h.pop(t, None)
            self._hbase.pop(t, None)
        return len(gone)

    # ------------------------------------------------------------------ 读

    def window_geo(self, members, start: int, end: int):
        """窗口 `[start, end)` 内该人的 `(ar_max, bottom_shift, r_end, n_box)`。"""
        start, end = int(start), int(end)
        best_h = None
        best_ar = None
        best_bs = 0.0
        best_re = None
        best_n = 0

        for tid in members:
            tid = int(tid)
            d = self._h.get(tid)
            if not d:
                continue
            hs = []
            ars = []
            bots = []
            for (f, x1, y1, x2, y2) in d:
                if f < start or f >= end:
                    continue
                w, h = x2 - x1, y2 - y1
                if w < 1.0 or h < 1.0:
                    continue
                hs.append(h)
                ars.append(w / h)
                bots.append(y2)
            if not hs:
                continue
            h_med = float(np.median(np.asarray(hs, dtype=np.float64)))
            if best_h is None or h_med > best_h:
                h_base = self._hbase.get(tid, 0.0)
                best_h = h_med
                best_ar = max(ars)
                best_bs = max(bots) - min(bots)
                # `hs[-1]` = 窗口内最后一个有效框的高度（不一定是窗口最后一帧）
                best_re = (hs[-1] / h_base) if h_base > 1.0 else None
                best_n = len(ars)

        if best_h is None:
            return None, 0.0, None, 0
        return float(best_ar), float(best_bs), best_re, int(best_n)

    # 旧名字保留：`geo_dump2` 与已有调用点用的是 `window_ar`
    def window_ar(self, members, start: int, end: int):
        ar, _bs, _re, n = self.window_geo(members, start, end)
        return ar, n
