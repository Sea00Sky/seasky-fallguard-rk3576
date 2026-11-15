"""The streaming fall-detection pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..deploy import spec
from .detector import Detector
from .gate import Gate, WindowScore, score_windows
from .ring import RingBuffer
from .stitch import Stitcher
from .tracker import IoUTracker


@dataclass
class PipelineConfig:
    """Everything the loop needs. Defaults are the frozen values."""

    yolo_onnx: Path
    stgcn_onnx: Path
    fps: float = spec.BOARD_FPS
    duration_s: float = spec.DURATION_S
    stride_s: float = spec.STRIDE_S
    conf: float = spec.CONF
    max_persons: int = spec.MAX_PERSONS
    track_iou: float = spec.TRACK_IOU
    max_age: int = spec.MAX_AGE
    min_coverage: float = spec.MIN_COVERAGE
    threshold: float = spec.THRESHOLD
    stitch: bool = True
    # PLAN §3.3. People scored at the same window boundary go through one
    # forward pass. 0 disables batching (one call per person, the pre-§3.3
    # behaviour); the value is a ceiling, not a requirement -- a boundary with
    # 2 people still sends one batch of 2.
    score_batch: int = 4
    # The dynamic-batch companion to `stgcn_onnx`. `None` means "look for
    # `<stgcn stem>_b4.onnx` next to it", and a missing file silently falls
    # back to per-person calls -- the static delivered graph cannot batch, and
    # refusing to run would make every G3/G6 script depend on scripts/15.
    stgcn_batch_onnx: Path | None = None

    def batch_onnx(self) -> Path | None:
        """The batch graph to use, or `None` for the per-person path."""
        if self.score_batch <= 1:
            return None
        if self.stgcn_batch_onnx is not None:
            return self.stgcn_batch_onnx if self.stgcn_batch_onnx.exists() else None
        guess = self.stgcn_onnx.with_name(self.stgcn_onnx.stem + "_b4.onnx")
        return guess if guess.exists() else None

    @property
    def window_frames(self) -> int:
        return spec.frames_for(self.fps, self.duration_s)

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
    n_scored: int = 0        # people that passed the gate and ran the model
    n_observed: int = 0      # people the ring could serve, gated or not
    observed: list[tuple[int, int, float]] = field(default_factory=list)


class Pipeline:
    """Streaming pipeline. One instance per video; call `reset()` to reuse."""

    def __init__(self, cfg: PipelineConfig, detector=None,
                 stgcn_session=None) -> None:
        import onnxruntime as ort

        self.cfg = cfg
        self.detector = (detector if detector is not None
                         else Detector(cfg.yolo_onnx, conf=cfg.conf,
                                       max_persons=cfg.max_persons))
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.add_session_config_entry("session.intra_op.allow_spinning", "0")
        # An injected session may be an RKNN simulator, which has no ONNX file
        # to open. When one is given, `stgcn_onnx` is never read -- but it must
        # still be a valid path for `PipelineConfig`'s own default batch lookup.
        self.stgcn = (stgcn_session if stgcn_session is not None else
                      ort.InferenceSession(
                          str(cfg.stgcn_onnx), opts,
                          providers=["CPUExecutionProvider"]))
        self.stgcn_input = self.stgcn.get_inputs()[0].name

        # PLAN §3.3. `None` here is not an error: it means the dynamic-batch
        # companion graph is absent, so `_score_batch` loops exactly as it did
        # before §3.3 and every historical number stays reproducible.
        self.stgcn_batch = None
        self.stgcn_batch_input = None
        self.batch_graph = cfg.batch_onnx()
        if self.batch_graph is not None:
            self.stgcn_batch = ort.InferenceSession(
                str(self.batch_graph), opts, providers=["CPUExecutionProvider"])
            self.stgcn_batch_input = self.stgcn_batch.get_inputs()[0].name
            shape = self.stgcn_batch.get_inputs()[0].shape
            if shape[0] != "N":
                raise ValueError(
                    f"{self.batch_graph.name} has a static batch axis "
                    f"({shape}); batching would broadcast a single sample "
                    f"across the batch instead of running N of them")

        self.gate = Gate(cfg.min_coverage, cfg.threshold)
        self.reset()

    def reset(self) -> None:
        self.tracker = IoUTracker(self.cfg.track_iou, self.cfg.max_age)
        self.stitcher = Stitcher()
        # The ring outlives the tracker: a track last seen at frame 5 of a
        # 60-frame window must still serve that window. See `ring.py`.
        self.ring = RingBuffer(self.cfg.window_frames)
        self._frame = 0
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

    # ------------------------------------------------------------------ scoring

    def _score_batch(self, items: list[tuple[int, int, float, np.ndarray]]
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
        idx = self._frame
        h, w = frame_bgr.shape[:2]
        if idx == 0:
            self._width, self._height = w, h

        detections = self.detector.detect(frame_bgr)
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

        # --- did a window boundary just close?
        win, stride = cfg.window_frames, cfg.stride_frames
        end = idx + 1
        if end < win or (end - win) % stride != 0:
            return []
        start = end - win
        wid = len(self._candidates)
        self._candidates.append((wid, start, end))

        items: list[tuple[int, int, float, np.ndarray]] = []
        for root, members in self._chains_touching(start, end):
            tw = self.ring.merge(members, start, end)
            if tw is None:
                continue
            cov = tw.coverage
            self._observed.append((wid, root, cov))
            if cov < self.gate.min_coverage:
                continue
            items.append((wid, root, cov, tw.model_input(self._width, self._height)))
            self._people[wid] = {"track": root, "members": sorted(members)}

        probs = self._score_batch(items)
        self._scored.extend((it[0], it[1], it[2], p) for it, p in zip(items, probs))

        # Report this window now, so a live consumer does not have to wait for
        # the video to end. `finish()` recomputes the same numbers over the
        # whole video and is what the gates check.
        return score_windows([(wid, start, end)], self._scored,
                             self.cfg.min_coverage)

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
