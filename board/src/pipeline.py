"""The streaming fall-detection pipeline."""
from __future__ import annotations

import queue
import threading
from dataclasses import dataclass, field

import numpy as np

from . import pose_filter, spec
from .gate import Gate, WindowScore, score_windows
from .ring import RingBuffer
from .stitch import Stitcher
from .tracker import IoUTracker
# `Detector`（onnxruntime 版）不导入：板端没有 onnxruntime，而
# `Pipeline.__init__` 原来是 `import onnxruntime as ort` 无条件执行，
# 会把整个模块拖垮。默认检测器改为必须由调用方注入
# （`src/backends/rknn_backend.py::YoloBackend`）。


@dataclass
class PipelineConfig:
    """Everything the loop needs. Defaults are the frozen values."""

 # 板端版本没有 ONNX 路径字段：两个后端都是注入的 RKNNLite 对象。
    fps: float = spec.BOARD_FPS
    duration_s: float = spec.DURATION_S
    stride_s: float = spec.STRIDE_S
 # ST-GCN 窗口按帧数定义（0 = 旧口径 duration_s×fps）。
 # 模型输入是骨架重采样到 64 步，训练域窗口是 48–60 源帧；按秒定义会让
 # 20fps 语料只拿到 40 帧（上采样 1.6×，B 臂事件级 6/17），而 60 帧窗
 # ≈ 训练采样（同库 PC 实测事件级 9/17，生产 veto 下）。现役 30fps 相机
 # 2s 本来就是 60 帧 ⇒ 打开它对生产直播逐位不变，只修 20fps 回放/评测。
    score_window_frames: int = 0
    conf: float = spec.CONF
    max_persons: int = spec.MAX_PERSONS
    track_iou: float = spec.TRACK_IOU
    max_age: int = spec.MAX_AGE
    min_coverage: float = spec.MIN_COVERAGE
    threshold: float = spec.THRESHOLD
 # D-042. `stitch=False` reproduces strategy B (per-track max, no ID-handoff
 # merging), which is what the source project's `--no-stitch` means.
    stitch: bool = True
 # ---- ② 身份桥接（，默认关）----
 # 跌倒本身会让框从「窄高」变「宽扁」，IoU 0.155 < TRACK_IOU 0.30 ⇒ tracker
 # 换号，而换号处必然零共享帧 ⇒ D-042 原判据永远接不上。开启后允许
 # `1 <= gap <= round(fps*gap_s)` 的零共享帧 handoff，补中心距判据。
 # 默认关 ⇒ 对既有链结果逐位不变（`p8/verify_stitch_bridge.py`）。
    stitch_bridge: bool = False
 # 桥接容忍的最大时间洞（秒）：20fps ⇒ 5 帧（原 gap_max），30fps ⇒ 8 帧。
    stitch_bridge_gap_s: float = 0.25
 # 板端固定 0：交付的 ST-GCN 图是 batch-1 静态图，`_score_batch` 逐人调用。
 # 与 PC 侧关闭批处理时的路径逐位等价（已实测）。
    score_batch: int = 0
 # ---- T3 后台线程打分----
 # 默认 False：关掉时一个线程都不建，`push` 走的是与改动前
 # 逐位相同的同步路径 ⇒ 915 窗 / 19 视频 / `p8/multi_eval.py` 基线不动。
 # 只有实时入口 `run_sensor_multi.py --async-stgcn` 会打开它。
    async_scoring: bool = False
 # 后台队列深度（最多积压几个待打分窗口）。满了阻塞等队首，不丢窗口。
    async_queue_max: int = 4
 # ---- 检出净化----
 # 改的是 tracker 的输入 ⇒ 窗口/事件/告警全部会变，不是显示层开关。
 # 要复现旧口径：`pose_min_kp=0, dedupe_contain=0, dedupe_iou=0`。
    pose_min_kp: int = 4 # 关键点里 >pose_kp_conf 的个数下限（0 = 关）
    pose_kp_conf: float = 0.3
    dedupe_contain: float = 0.55 # 被更大的框包含到这个比例就丢（0 = 关）
    dedupe_iou: float = 0.70 # 与更大的框 IoU 到这个值就丢（0 = 关）

    @property
    def window_frames(self) -> int:
        n = int(self.score_window_frames)
        return n if n > 0 else spec.frames_for(self.fps, self.duration_s)

    @property
    def stride_frames(self) -> int:
        return spec.frames_for(self.fps, self.stride_s)


@dataclass
class VideoResult:
    """What one video produced."""

    video_id: str
    n_frames: int
    width: int
    height: int
    windows: list[WindowScore] = field(default_factory=list)
 # window_id -> {"track": int, "members": list[int]} at scoring time
    window_people: dict[int, dict] = field(default_factory=dict)
    chains: list[list[int]] = field(default_factory=list)
    n_scored: int = 0 # people that passed the gate and ran the model
    n_observed: int = 0 # people the ring could serve, gated or not
    observed: list[tuple[int, int, float]] = field(default_factory=list)


@dataclass
class _Job:
    """One ST-GCN batch handed to the worker thread."""

    token: int
    items: list
    result: tuple | None = None # (probs, error)


class Pipeline:
    """Streaming pipeline. One instance per video; call `reset` to reuse."""

    def __init__(self, cfg: PipelineConfig, detector=None,
                 stgcn_session=None) -> None:
 # 板端版本：不 import onnxruntime，两个后端都必须由调用方注入。
 # 源项目在这里 `import onnxruntime as ort` 并无条件建会话，
 # 板子上没有这个包，模块导入就会失败。
        if detector is None:
            raise ValueError(
                "板端必须注入 detector（src/backends/rknn_backend.py::YoloBackend）。"
                "源项目的默认 Detector 依赖 onnxruntime，板端不可用。")
        if stgcn_session is None:
            raise ValueError(
                "板端必须注入 stgcn_session"
                "（src/backends/rknn_backend.py::StgcnBackend）。")

        self.cfg = cfg
        self.detector = detector
        self.stgcn = stgcn_session
        self.stgcn_input = self.stgcn.get_inputs()[0].name

 # 批处理在板端关闭：交付的 ST-GCN 图是 batch-1 静态图，
 # 没有动态 batch 的伴生图。`_score_batch` 会走逐人调用的路径，
 # 与 PC 侧关闭批处理时的行为一致（实测逐位等价）。
        self.stgcn_batch = None
        self.stgcn_batch_input = None
        self.batch_graph = None

 # ---- T3 后台线程的状态（必须在 reset 之前存在：reset 会先 _stop_worker）----
        self._worker: threading.Thread | None = None
        self._job_q: queue.Queue | None = None
        self._done_q: queue.Queue | None = None
        self._pending: list[tuple] = []
        self._next_token = 0
 # 本次 push 里收尾的窗口 / 产生的事件（每次 push 开头清空）
        self._just_closed: list[tuple[int, int, int]] = []
        self._events_out: list = []

        self.gate = Gate(cfg.min_coverage, cfg.threshold)
        self.reset()

    def reset(self) -> None:
 # 先停掉上一轮的后台线程（复用实例时）；`_pending` 为空时是空操作，
 # 所以 `__init__` 里第一次调用不会碰任何还没建好的状态。
        self._stop_worker()
        self._pending = []
        self._next_token = 0
        self._just_closed = []
        self._events_out = []

        self.tracker = IoUTracker(self.cfg.track_iou, self.cfg.max_age)
 # 必须传管线自己的 `max_age`：`PipelineConfig.max_age` 可被覆盖，
 # 而 `spec.MAX_AGE` 是模块级常量。Stitcher 用 `max(gap_max, max_age)` 当
 # 冻结视界，两者不一致就会重新出现「几何被 pop 但轨迹还活着」的窗口
 # ⇒ `KeyError`（ 修的就是这个）。
 # ② 身份桥接：开启时把 gap_max 至少放宽到 `stitch_bridge_gap_s` 的帧数。
 # 放大 gap_max 只影响正 gap 的桥接支：真 handoff 必须有共享帧
 # （gap <= 0），所以 D-042 原规则不受影响。关闭时用的就是原 5 帧。
        bridge_gap = spec.STITCH_GAP_MAX
        if self.cfg.stitch_bridge:
            bridge_gap = max(
                bridge_gap,
                int(round(self.cfg.fps * self.cfg.stitch_bridge_gap_s)))
        self.stitcher = Stitcher(gap_max=bridge_gap,
                                 track_max_age=self.cfg.max_age,
                                 allow_gap_bridge=self.cfg.stitch_bridge)
 # The ring outlives the tracker: a track last seen at frame 5 of a
 # 60-frame window must still serve that window. See `ring.py`.
        self.ring = RingBuffer(self.cfg.window_frames)
        self._frame = 0
 # 检出净化丢掉的数量（只做观测量，不参与任何判定）。
        self.filter_dropped_kp = 0
        self.filter_dropped_dup = 0
        self._candidates: list[tuple[int, int, int]] = []
 # (window_id, track_id, coverage, prob_fall) for every person that
 # actually went through the model. People below the coverage gate are
 # not here, because they were never scored.
        self._scored: list[tuple[int, int, float, float]] = []
 # (window_id, track_id, coverage) for every person observed, gated or
 # not -- reported for diagnosis, never used in the max rule.
        self._observed: list[tuple[int, int, float]] = []
        self._people: dict[int, dict] = {}
        self._width = 0
        self._height = 0

        if self.cfg.async_scoring:
            self._start_worker()

 # ------------------------------------------------------- T3 后台线程（打分）

    def _start_worker(self) -> None:
        """起一个 worker 线程。只跑 `_score_items`，别的什么都不碰。"""
        self._job_q = queue.Queue()
        self._done_q = queue.Queue()
        self._worker = threading.Thread(
            target=self._worker_loop, name="stgcn-t3", daemon=True)
        self._worker.start()

    def _worker_loop(self) -> None:
        while True:
            job = self._job_q.get()
            if job is None: # 停止哨兵
                return
            try:
                probs = self._score_items(job.items)
                err = None
            except BaseException as exc: # noqa: BLE001 —— 原样抛回主线程
                probs, err = None, exc
            self._done_q.put((job.token, probs, err))

    def _stop_worker(self) -> None:
        """收干净再停：先阻塞排空，再发哨兵。幂等。"""
        if self._worker is None:
            return
        self._drain(block=True) # 队列里还有窗口就先算完
        self._job_q.put(None)
        self._worker.join(timeout=5.0)
        self._worker = None
        self._job_q = None
        self._done_q = None

    def _submit(self, wid, start, end, obs, items, extra) -> None:
        """把这一窗的 ST-GCN 工作丢给 worker，立刻返回。"""
        depth = max(1, int(self.cfg.async_queue_max))
        if len(self._pending) >= depth:
            self._drain(block=True)
        self._next_token += 1
        job = _Job(token=self._next_token, items=items)
        self._pending.append((int(wid), int(start), int(end),
                              obs, items, extra, job))
        self._job_q.put(job)

    def _drain(self, block: bool) -> None:
        """按 FIFO 收结果并收尾。`block=True` 时等到队首算完。"""
        while self._pending:
            job = self._pending[0][-1]
            if job.result is None:
                try:
                    token, probs, err = (self._done_q.get() if block
                                         else self._done_q.get_nowait())
                except queue.Empty:
                    return
                if token != job.token:
                    raise RuntimeError(
                        f"ST-GCN 后台线程乱序返回：等 token={job.token}，"
                        f"拿到 {token}")
                job.result = (probs, err)

            probs, err = job.result
            rec = self._pending.pop(0)
            if err is not None:
                raise err
            self._on_window(rec[0], rec[1], rec[2], rec[3], rec[4],
                            probs, rec[5])

    def _before_score(self, wid, start, end, obs, items) -> dict:
        """主线程、窗口刚关闭时调用，早于 NPU（同步异步都在同一时刻）。"""
        return {}

    def _after_frame(self, frame_index: int) -> None:
        """主线程、`self._frame` 已自增后调用。基类不做任何事。"""
        return

    def _on_window(self, wid, start, end, obs, items, probs, extra) -> None:
        """把一个窗口的结果落到状态上（主线程，FIFO 顺序）。"""
        self._scored.extend((it[0], it[1], it[2], p)
                            for it, p in zip(items, probs))
        self._just_closed.append((int(wid), int(start), int(end)))

 # ------------------------------------------------------------------ scoring

    def _gate_items(self, items: list[tuple[int, int, float, np.ndarray]]
                    ) -> list[tuple[int, int, float, np.ndarray]]:
        """开窗后、跑 NPU 前的人数闸门。基类原样返回。"""
        return items

    def _score_batch(self, items: list[tuple[int, int, float, np.ndarray]]
                     ) -> list[float]:
        """同步路径的 NPU 调用。保留原名：`p8/verify_geo_veto.py::FakePipe`"""
        return self._score_items(items)

    def _score_items(self, items: list[tuple[int, int, float, np.ndarray]]
                     ) -> list[float]:
        """Score every person that needs it this step."""
        if not items:
            return []

 # No batch graph, or nothing to gain from one.
        if self.stgcn_batch is None or len(items) == 1:
            out: list[float] = []
            for _wid, _track, _cov, model_input in items:
                logits = self.stgcn.run(None, {self.stgcn_input: model_input})[0]
                out.append(float(spec.softmax2(logits)[0, 1]))
            return out

        out = []
        size = max(1, int(self.cfg.score_batch))
        for lo in range(0, len(items), size):
            chunk = items[lo:lo + size]
 # `model_input` is (1, 1, T, V, C) per person; concatenating on
 # axis 0 gives exactly the (n, 1, T, V, C) the graph declares.
            batch = np.concatenate([it[3] for it in chunk], axis=0)
            logits = self.stgcn_batch.run(
                None, {self.stgcn_batch_input: batch})[0]
            if len(logits) != len(chunk):
                raise RuntimeError(
                    f"batch graph returned {len(logits)} rows for "
                    f"{len(chunk)} people; a batch-N graph fed fewer samples "
                    f"broadcasts instead of erroring, so this is checked")
 # `logits` is (n, 2); `softmax2` works on the last axis, so the
 # whole block goes through in one call and row i stays row i.
            probs = spec.softmax2(logits)
            out.extend(float(p) for p in probs[:, 1])
        return out

 # -------------------------------------------------------------------- frame

    def push(self, frame_bgr: np.ndarray) -> list[WindowScore]:
        """Feed one frame; return the windows that closed on this frame."""
        cfg = self.cfg
 # 这两行必须在 `_drain` 之前：后台线程收尾的窗口要算进本次
 # push 的返回值，而不是上一次的。
        self._just_closed = []
        self._events_out = []
 # 先收后台线程已经算完的旧窗口。它们属于上一个窗口边界，
 # 在这里收掉才不会和本次新开的窗口混在一起；同步模式下
 # `_pending` 恒为空，这一句是空操作。
        self._drain(block=False)

        idx = self._frame
        h, w = frame_bgr.shape[:2]
        if idx == 0:
            self._width, self._height = w, h

        detections = self.detector.detect(frame_bgr)
 # 必须在 tracker 之前：重复框一旦进了 tracker 就会各拿一个 track_id，
 # 网页上就是「一个人两个号 / 标签乱跳」。
 # `prev_boxes` 取自 tracker 现在的框（本帧 `update` 还没跑，
 # 所以就是上一帧的）⇒ 净化按"接得上谁的身份"取舍，而不是按面积。
        detections, _d_kp, _d_dup = pose_filter.apply_to(
            detections, cfg, self.tracker.boxes())
        self.filter_dropped_kp += _d_kp
        self.filter_dropped_dup += _d_dup
        boxes = (np.stack([d.bbox for d in detections]) if detections
                 else np.zeros((0, 4), np.float32))
        ids = self.tracker.update(boxes, idx)

 # --- state: chains + per-track rings
        for det, tid in zip(detections, ids):
            det.track_id = int(tid)
            self.stitcher.observe(int(tid), idx, det.bbox)
            self.ring.update(int(tid), idx, det.keypoints, det.scores)
 # Tracks the detector missed this frame must still record a hole, or
 # coverage reads higher than the offline pipeline's.
        self.ring.mark_missing(idx)
        self.ring.expire(idx)

        self._frame += 1
        self._after_frame(idx)

 # --- did a window boundary just close?
        win, stride = cfg.window_frames, cfg.stride_frames
        end = idx + 1
        if end < win or (end - win) % stride != 0:
            return score_windows(self._just_closed, self._scored,
                                 cfg.min_coverage)
        start = end - win
        wid = len(self._candidates)
        self._candidates.append((wid, start, end))

 # --- one sample per *chain* whose ring covers this window (D-042)
 # `model_input` 在这一刻就把张量复制出来（`np.zeros` / 花式索引
 # 都是新数组），所以之后 ring 怎么翻都不会改到它 —— 这是能把它交给
 # 后台线程的前提。
        obs: list[tuple[int, int, float]] = []
        items: list[tuple[int, int, float, np.ndarray]] = []
        for root, members in self._chains_touching(start, end):
            tw = self.ring.merge(members, start, end)
            if tw is None:
                continue
            cov = tw.coverage
            self._observed.append((wid, root, cov))
            obs.append((wid, root, cov))
            if cov < self.gate.min_coverage:
                continue
            items.append((wid, root, cov, tw.model_input(self._width, self._height)))
            self._people[wid] = {"track": root, "members": sorted(members)}

 # 人数闸门 + 几何采样都跑在主线程，且都在 NPU 之前。
        items = self._gate_items(items)
        extra = self._before_score(wid, start, end, obs, items)

        if self.cfg.async_scoring:
            self._submit(wid, start, end, obs, items, extra)
 # 大多数时候还没算完；算完了就顺手收掉，降低一帧延迟。
            self._drain(block=False)
        else:
            probs = self._score_batch(items)
            self._on_window(wid, start, end, obs, items, probs, extra)

 # Report this window now, so a live consumer does not have to wait for
 # the video to end. `finish` recomputes the same numbers over the
 # whole video and is what the gates check.
        return score_windows(self._just_closed, self._scored,
                             cfg.min_coverage)

 # ------------------------------------------------------------------ chains

    def _chains_touching(self, start: int, end: int) -> list[tuple[int, set[int]]]:
        """Groups with at least one track able to serve `[start, end)`."""
        if self.cfg.stitch:
            groups = [(min(c), {t for t in c if self.ring.has(t)})
                      for c in self.stitcher.chains_so_far()]
        else:
            groups = [(t, {t}) for t in self.ring.track_ids()]

        out: list[tuple[int, set[int]]] = []
        for root, members in groups:
            if not members:
                continue
            if not any(self.ring.can_take(t, start, end) for t in members):
                continue
            out.append((root, members))
        out.sort(key=lambda kv: kv[0])
        return out

 # ------------------------------------------------------------------- finish

    def finish(self, video_id: str) -> VideoResult:
        """Close the video and collapse samples into window verdicts."""
 # 异步模式下必须先把后台线程排空再算总分：最后几个窗口的概率
 # 还在 worker 手里，漏收就会把它们报成 `prob_fall = 0.0` ——
 # 与"模型很确定没跌倒"完全同形。
        self._drain(block=True)
        self._stop_worker()
        windows = score_windows(self._candidates, self._scored,
                                self.cfg.min_coverage)
        return VideoResult(
            video_id=video_id, n_frames=self._frame,
            width=self._width, height=self._height,
            windows=windows, window_people=dict(self._people),
            chains=self.stitcher.chains_so_far(),
            n_scored=len(self._scored), n_observed=len(self._observed),
            observed=list(self._observed),
        )

 # ------------------------------------------------------------------ helpers

    def run_video(self, video_id: str, frames) -> VideoResult:
        """Convenience: push an iterable of frames, then finish."""
        for frame in frames:
            self.push(frame)
        return self.finish(video_id)
