"""GStreamer `appsink` 采集源。"""
from __future__ import annotations

import threading
import time
from collections import deque

import numpy as np

from .base import Frame, FrameSource
from .config import CaptureConfig

try:                                    # pragma: no cover - 板端才有
    import gi

    gi.require_version("Gst", "1.0")
    from gi.repository import Gst
except Exception as _exc:               # pragma: no cover
    Gst = None
    _GST_IMPORT_ERROR = _exc
else:
    _GST_IMPORT_ERROR = None


class _DropOldestQueue:
    """有界、满则丢最旧的队列（回调侧永不阻塞）。"""

    def __init__(self, maxsize: int) -> None:
        self._q: deque[Frame] = deque()
        self._max = int(maxsize)
        self._cond = threading.Condition()
        self._closed = False
        self.dropped = 0

    def put(self, item: Frame) -> None:
        with self._cond:
            if self._closed:
                return
            if len(self._q) >= self._max:
                self._q.popleft()
                self.dropped += 1
            self._q.append(item)
            self._cond.notify()

    def get(self, timeout_s: float | None) -> Frame | None:
        deadline = None if timeout_s is None else time.monotonic() + timeout_s
        with self._cond:
            while not self._q and not self._closed:
                if deadline is None:
                    self._cond.wait()
                    continue
                left = deadline - time.monotonic()
                if left <= 0:
                    return None
                self._cond.wait(left)
            if self._closed:
                return None
            return self._q.popleft()

    def close(self) -> None:
        with self._cond:
            self._closed = True
            self._cond.notify_all()


class GstSource(FrameSource):
    """`v4l2src`（或裸 NV12 文件）→ `tee` → `appsink` → BGR 帧。"""

    def __init__(self, cfg: CaptureConfig) -> None:
        if Gst is None:                 # pragma: no cover
            raise RuntimeError(
                f"导入 GStreamer Python 绑定失败：{_GST_IMPORT_ERROR}。"
                f"板上应有 python3-gi + gir1.2-gstreamer-1.0；"
                f"`python3 -c \"import gi; gi.require_version('Gst','1.0')\"` 可自检。")
        self.cfg = cfg
        self._pipeline = None
        self._sink = None
        self._bus = None
        self._queue = _DropOldestQueue(cfg.queue_buffers)
        self._probe_pad = None
        self._probe_id = 0
        self._handler_id = 0

        # ---- 计数（回调线程写、主线程读；CPython 下 int += 是原子的够用） ----
        self._tee_count = 0         # tee 这一路发出多少帧（含随后被丢的）
        self._pulled = 0            # 回调实际收到多少帧
        self._read = 0              # 上层取走多少帧
        self._gaps = 0              # 帧间隔 > 1.5 个帧周期的次数
        self._max_gap_ns = 0
        self._last_pts_ns: int | None = None
        self._error: BaseException | None = None
        self._eos = False
        self._t_start = 0.0
        self._t_stop = 0.0

    # ------------------------------------------------------------------ 管线

    def _pipeline_desc(self) -> str:
        c = self.cfg
        if c.raw_file:
            head = (f'filesrc location="{c.raw_file}" '
                    f'! rawvideoparse format=nv12 width={c.width} '
                    f'height={c.height} framerate={c.fps}/1 '
                    f'! identity sync=true')
        else:
            head = (f"v4l2src device={c.device} io-mode={c.io_mode} "
                    f"! video/x-raw,format=NV12,width={c.width},"
                    f"height={c.height},framerate={c.fps}/1")

        parts = [
            head,
            "! tee name=t",
            # max-size-bytes/time 必须显式归零：queue 默认还有 10MB/1s 两个上限，
            # 不关掉的话「4 帧」这个意图在低分辨率下会被别的上限抢先触发。
            f"t. ! queue name=detq leaky=downstream "
            f"max-size-buffers={c.queue_buffers} max-size-bytes=0 max-size-time=0 "
            f"! appsink name=det max-buffers={c.appsink_buffers} drop=true "
            f"emit-signals=true sync=false",
        ]
        if c.shm_socket:
            parts.append(
                f"t. ! queue leaky=downstream "
                f"max-size-buffers={c.queue_buffers} max-size-bytes=0 max-size-time=0 "
                f"! shmsink socket-path={c.shm_socket} shm-size={c.shm_size} "
                f"wait-for-connection=false sync=false")
        return " ".join(parts)

    def start(self) -> None:
        if self._pipeline is not None:
            return
        Gst.init(None)
        desc = self._pipeline_desc()
        try:
            self._pipeline = Gst.parse_launch(desc)
        except Exception as exc:        # pragma: no cover
            raise RuntimeError(
                f"管线构建失败：{exc}\n描述：{desc}") from exc

        self._sink = self._pipeline.get_by_name("det")
        if self._sink is None:          # pragma: no cover
            raise RuntimeError(f"管线里找不到 appsink name=det：{desc}")
        self._sink.set_property("emit-signals", True)
        self._handler_id = self._sink.connect("new-sample", self._on_new_sample)

        # 丢帧计数：挂在 queue 的 sink pad（tee 这一路的出口），
        # 这样被 leaky queue 丢掉的帧也会被数到。
        detq = self._pipeline.get_by_name("detq")
        if detq is not None:
            self._probe_pad = detq.get_static_pad("sink")
            if self._probe_pad is not None:
                self._probe_id = self._probe_pad.add_probe(
                    Gst.PadProbeType.BUFFER, self._on_buffer)

        self._bus = self._pipeline.get_bus()
        self._t_start = time.monotonic()
        self._pipeline.set_state(Gst.State.PLAYING)

    # ------------------------------------------------------------ 回调线程

    def _on_buffer(self, pad, info):
        """tee 出口计数。只加一，不做任何 I/O（跑在 streaming thread 上）。"""
        self._tee_count += 1
        return Gst.PadProbeReturn.OK

    def _on_new_sample(self, sink):
        """异常必须自己接住 —— PyGObject 会吞掉信号回调里的异常。"""
        try:
            sample = sink.emit("pull-sample")
            if sample is None:
                return Gst.FlowReturn.OK
            buf = sample.get_buffer()
            if buf is None:
                return Gst.FlowReturn.OK

            data = buf.extract_dup(0, buf.get_size())
            h, w = self.cfg.height, self.cfg.width
            expect = w * h * 3 // 2
            if len(data) != expect:
                raise RuntimeError(
                    f"NV12 帧大小不符：拿到 {len(data)} B，按 {w}x{h} 应为 {expect} B。"
                    f"多半是 caps 没协商成 NV12（检查 v4l2src 的 format 约束）")
            nv12 = np.frombuffer(data, np.uint8).reshape(h * 3 // 2, w)
            bgr = nv12_to_bgr(nv12, h, w)

            pts = buf.pts
            pts_ns = None if pts == Gst.CLOCK_TIME_NONE else int(pts)

            seq = self._pulled
            self._pulled += 1
            self._queue.put(Frame(seq=seq, pts_ns=pts_ns, bgr=bgr))
        except BaseException as exc:    # noqa: BLE001 - 故意兜住全部
            if self._error is None:
                self._error = exc
        return Gst.FlowReturn.OK

    def _note_pts(self, pts_ns: int | None) -> None:
        """记录帧间隔。窗口按帧数切，丢帧会让一个窗覆盖的真实时间变长 —— 先只观测。"""
        if pts_ns is None:
            self._last_pts_ns = None
            return
        if self._last_pts_ns is not None:
            gap = pts_ns - self._last_pts_ns
            if gap > self._max_gap_ns:
                self._max_gap_ns = gap
            period = 1e9 / max(1, self.cfg.fps)
            if gap > 1.5 * period:
                self._gaps += 1
        self._last_pts_ns = pts_ns

    # ---------------------------------------------------------------- 主线程

    def read(self, timeout_s: float | None = None) -> Frame | None:
        if self._error is not None:
            raise self._error
        fr = self._queue.get(self.cfg.read_timeout_s if timeout_s is None else timeout_s)
        if fr is None:
            self._poll_bus()
            if self._error is not None:
                raise self._error
            return None
        self._read += 1
        # 帧间隔按消费者视角记：这才是「窗口覆盖的真实时间有没有被拉长」的证据。
        self._note_pts(fr.pts_ns)
        return fr

    def _poll_bus(self) -> None:
        """只在 `read` 超时时调用 —— 不额外开线程，也不起 GLib 主循环。"""
        if self._bus is None:
            return
        while True:
            msg = self._bus.pop_filtered(
                Gst.MessageType.ERROR | Gst.MessageType.EOS)
            if msg is None:
                return
            if msg.type == Gst.MessageType.ERROR:
                err, debug = msg.parse_error()
                self._error = RuntimeError(f"GStreamer 报错：{err.message}（{debug}）")
            elif msg.type == Gst.MessageType.EOS:
                self._eos = True

    def stop(self) -> None:
        if self._pipeline is None:
            return
        self._t_stop = time.monotonic()
        # 先摘回调和探针，再 NULL —— 否则停机过程中还会进回调
        try:
            if self._sink is not None and self._handler_id:
                self._sink.disconnect(self._handler_id)
        except Exception:               # pragma: no cover
            pass
        try:
            if self._probe_pad is not None and self._probe_id:
                self._probe_pad.remove_probe(self._probe_id)
        except Exception:               # pragma: no cover
            pass
        self._probe_id = 0
        self._handler_id = 0
        try:
            self._pipeline.set_state(Gst.State.NULL)
        finally:
            self._queue.close()
            self._pipeline = None

    # ------------------------------------------------------------------ 计数

    @property
    def stats(self) -> dict:
        dt = ((self._t_stop or time.monotonic()) - self._t_start) if self._t_start else 0.0
        gst_dropped = max(0, self._tee_count - self._pulled)
        return {
            "backend": "gst",
            "tee_count": self._tee_count,          # tee 这一路发出（含随后被丢的）
            "pulled": self._pulled,                # 回调收到
            "read": self._read,                    # 上层取走
            "gst_dropped": gst_dropped,            # leaky queue + appsink drop 丢的
            "queue_dropped": self._queue.dropped,  # 本进程队列丢的
            "dropped_total": gst_dropped + self._queue.dropped,
            "gaps": self._gaps,
            "max_gap_ms": self._max_gap_ns / 1e6,
            "seconds": round(dt, 3),
            "fps": round(self._read / dt, 2) if dt > 0 else 0.0,
            "eos": self._eos,
            "error": repr(self._error) if self._error else None,
        }


# ---------------------------------------------------------------- NV12 → BGR

def nv12_to_bgr(nv12: np.ndarray, h: int, w: int) -> np.ndarray:
    """NV12 → BGR。"""
    import cv2

    return cv2.cvtColor(nv12, cv2.COLOR_YUV2BGR_NV12)
