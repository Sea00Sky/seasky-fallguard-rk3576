"""MCFD multi-segment fall interval annotator (manual layer)."""
from __future__ import annotations

import argparse
import base64
import getpass
import os
import sys
import tkinter as tk
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from tkinter import messagebox, ttk

import cv2
import numpy as np

from src.annotate import mcfd_segments as ms
from src.annotate.interval_tool import FrameStore
from src.data import paths

try:  # Pillow is present in the pyskl env; Tk PhotoImage is the fallback.
    from PIL import Image, ImageTk

    _HAS_PIL = True
except ImportError:  # pragma: no cover - depends on the environment
    Image = None  # type: ignore[assignment]
    ImageTk = None  # type: ignore[assignment]
    _HAS_PIL = False

MAX_CACHED_FRAMES = 4000
SEEK_BUFFER = 16
DEFAULT_SPEED = 0.25


@dataclass(frozen=True)
class SourceCandidate:
    chute: int
    cam: int
    video_id: str
    event_id: str
    relative_path: str
    n_frames: int
    fps: float
    width: int
    height: int
    segments: tuple[ms.FallSegment, ...]
    context_spans: tuple[tuple[int, int], ...]
    coverage: float

    @property
    def abs_path(self) -> Path:
        return paths.RAW / self.relative_path


class McfdFrameStore:
    """Frame reader that never allocates 4000+ 720x480 frames."""

    def __init__(
        self,
        path: Path,
        max_cache: int = MAX_CACHED_FRAMES,
        buffer: int = SEEK_BUFFER,
    ) -> None:
        self.path = Path(path)
        self.max_cache = max(1, int(max_cache))
        self.buffer = max(1, int(buffer))
        self._cap: cv2.VideoCapture | None = None
        self._frames: list[np.ndarray] | None = None
        self._delegate: FrameStore | None = None
        self._buf: dict[int, np.ndarray] = {}
        self._next = 0
        self.fps = 0.0
        self.width = 0
        self.height = 0
        self.n_frames = 0
        self.cached = False
        self._open()

    def _open(self) -> None:
        cap = cv2.VideoCapture(str(self.path))
        if not cap.isOpened():
            raise RuntimeError(f"无法打开视频：{self.path}")
        self._cap = cap
        self.fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        self.n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        if self.n_frames <= 0:
            # Container has no frame count; delegate to the old reader.
            cap.release()
            self._cap = None
            self._delegate = FrameStore(self.path, max_cache=self.max_cache)
            self.n_frames = self._delegate.n_frames
            self.fps = self._delegate.fps or self.fps
            self.width = self._delegate.width or self.width
            self.height = self._delegate.height or self.height
            self.cached = self._delegate.cached
            return

        if self.n_frames <= self.max_cache:
            frames: list[np.ndarray] = []
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frames.append(frame)
            cap.release()
            self._cap = None
            self._frames = frames
            self.n_frames = len(frames)
            self.cached = True
            return

        self.cached = False
        self._frames = None
        self._buf = {}
        self._next = 0

    def _seek_fill(self, index: int) -> None:
        assert self._cap is not None
        start = max(0, index - 4)
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, float(start))
        self._buf.clear()
        self._next = start
        while self._next <= index:
            ok, frame = self._cap.read()
            if not ok:
                self._next = self.n_frames
                return
            self._buf[self._next] = frame
            self._next += 1
            while len(self._buf) > self.buffer:
                oldest = min(self._buf)
                del self._buf[oldest]

    def get(self, index: int) -> np.ndarray | None:
        if index < 0 or index >= self.n_frames:
            return None
        if self._delegate is not None:
            return self._delegate.get(index)
        if self._frames is not None:
            return self._frames[index]
        assert self._cap is not None
        if index in self._buf:
            return self._buf[index]
        if index == self._next and self._buf:
            ok, frame = self._cap.read()
            if not ok:
                return None
            self._buf[self._next] = frame
            self._next += 1
            while len(self._buf) > self.buffer:
                oldest = min(self._buf)
                del self._buf[oldest]
            return self._buf.get(index)
        self._seek_fill(index)
        return self._buf.get(index)

    def close(self) -> None:
        if self._delegate is not None:
            self._delegate.close()
            self._delegate = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._frames = None
        self._buf.clear()


def _force_utf8_stdout() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


# ---------------------------------------------------------------------------
# Candidates
# ---------------------------------------------------------------------------


def build_candidates(
    selected: dict[int, int],
    segments: dict[tuple[int, int], list[ms.FallSegment]],
    meta: dict[str, dict],
    coverage: dict[str, dict],
    all_rows: list[dict],
    only_chute: int | None = None,
    cam_override: int | None = None,
) -> list[SourceCandidate]:
    contexts: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for row in all_rows:
        if (row.get("status") or "").strip() != "valid":
            continue
        if ms.to_int(row.get("label")) != 0:
            continue
        key = (ms.to_int(row.get("chute")), ms.to_int(row.get("cam")))
        contexts.setdefault(key, []).append(
            (ms.to_int(row.get("start_frame")) - 1, ms.to_int(row.get("end_frame")))
        )

    out: list[SourceCandidate] = []
    for chute, cam in sorted(selected.items()):
        if only_chute is not None and chute != only_chute:
            continue
        if cam_override is not None:
            cam = cam_override
        if (chute, cam) not in segments:
            raise SystemExit(f"chute{chute:02d}/cam{cam} 没有 label-1 区间")
        vid = ms.video_id(chute, cam)
        info = meta.get(vid)
        if info is None:
            raise SystemExit(f"{vid} 不在 v2 manifest 中")
        out.append(
            SourceCandidate(
                chute=chute,
                cam=cam,
                video_id=vid,
                event_id=ms.event_id(chute),
                relative_path=info["relative_path"],
                n_frames=ms.to_int(info.get("n_frames")),
                fps=ms.to_float(info.get("fps")),
                width=ms.to_int(info.get("width")),
                height=ms.to_int(info.get("height")),
                segments=tuple(segments[(chute, cam)]),
                context_spans=tuple(sorted(contexts.get((chute, cam), []))),
                coverage=ms.to_float(coverage.get(vid, {}).get("interval_coverage")),
            )
        )
    return out


def print_candidates(candidates: list[SourceCandidate], manual: list[dict]) -> None:
    print(f"source cameras: {len(candidates)}")
    for i, cand in enumerate(candidates, 1):
        done = ms.is_camera_complete(manual, {(cand.chute, cand.cam): list(cand.segments)}, cand.chute, cand.cam)
        anchors = ", ".join(f"seg{s.index}:f{s.anchor_start}" for s in cand.segments)
        mode = "cache" if cand.n_frames <= MAX_CACHED_FRAMES else "seek"
        mark = "x" if done else " "
        print(
            f"[{mark}] {i:2d}  {cand.video_id:<22}  n={cand.n_frames:4d}  {mode:<5} "
            f"cov={cand.coverage:.3f}  {anchors}"
        )


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------


@dataclass
class SegmentState:
    segment: ms.FallSegment
    start: int | None = None
    end: int | None = None


class McfdAnnotatorApp:
    def __init__(
        self,
        root: tk.Tk,
        candidates: list[SourceCandidate],
        segments: dict[tuple[int, int], list[ms.FallSegment]],
        rows: list[dict],
        output: Path,
        annotator: str,
        start_index: int = 0,
    ) -> None:
        self.root = root
        self.candidates = candidates
        self.segments = segments
        self.rows = rows
        self.output = output
        self.annotator = annotator
        self.index = max(0, min(start_index, len(candidates) - 1))
        self.store: McfdFrameStore | None = None
        self.frame_idx = 0
        self.segment_index = 0
        self.states: list[SegmentState] = []
        self.playing = False
        self.speed = DEFAULT_SPEED
        self.zoom = 1
        self.dirty = False
        self._after_id: str | None = None
        self._syncing = False
        self._syncing_fine = False
        self._photo = None
        self._fine_window = (0, 1)
        self._seg_buttons: list[ttk.Button] = []
        self._timeline_job: str | None = None

        self._build_ui()
        self._bind_keys()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.load_index(self.index)

    # -- setup --------------------------------------------------------------

    def _build_ui(self) -> None:
        self.root.title("MCFD 跌倒区间标注器")
        self.root.geometry("1100x820")
        self.root.minsize(820, 620)

        top = ttk.Frame(self.root, padding=(8, 6, 8, 2))
        top.pack(fill=tk.X)
        self.info_var = tk.StringVar()
        ttk.Label(top, textvariable=self.info_var, anchor="w").pack(fill=tk.X)

        self.image_label = tk.Label(
            self.root, bg="#111111", fg="white", bd=1, relief=tk.SUNKEN, anchor=tk.CENTER
        )
        self.image_label.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

        segbar = ttk.Frame(self.root, padding=(8, 2))
        segbar.pack(fill=tk.X)
        self.seg_frame = ttk.Frame(segbar)
        self.seg_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)

        statusbar = ttk.Frame(self.root, padding=(8, 2))
        statusbar.pack(fill=tk.X)
        self.position_var = tk.StringVar()
        self.interval_var = tk.StringVar()
        ttk.Label(statusbar, textvariable=self.position_var, anchor="w").pack(fill=tk.X)
        ttk.Label(statusbar, textvariable=self.interval_var, anchor="w").pack(fill=tk.X)

        timeline = ttk.Frame(self.root, padding=(8, 2))
        timeline.pack(fill=tk.X)
        self.timeline = tk.Canvas(timeline, height=46, bg="#1b1b1b", highlightthickness=0)
        self.timeline.pack(fill=tk.X)
        self.timeline.bind("<Button-1>", self.on_timeline_click)
        self.timeline.bind("<Configure>", lambda _e: self.draw_timeline())

        overview = ttk.Frame(self.root, padding=(8, 2))
        overview.pack(fill=tk.X)
        ttk.Label(overview, text="全片", width=6).pack(side=tk.LEFT)
        self.scale_var = tk.IntVar(value=0)
        self.scale = tk.Scale(
            overview,
            from_=0,
            to=1,
            orient=tk.HORIZONTAL,
            variable=self.scale_var,
            showvalue=False,
            takefocus=0,
            command=self.on_scale,
        )
        self.scale.pack(side=tk.LEFT, fill=tk.X, expand=True)

        fine = ttk.Frame(self.root, padding=(8, 2))
        fine.pack(fill=tk.X)
        ttk.Label(fine, text="当前段", width=6).pack(side=tk.LEFT)
        self.fine_var = tk.IntVar(value=0)
        self.fine = tk.Scale(
            fine,
            from_=0,
            to=1,
            orient=tk.HORIZONTAL,
            variable=self.fine_var,
            showvalue=False,
            takefocus=0,
            command=self.on_fine,
        )
        self.fine.pack(side=tk.LEFT, fill=tk.X, expand=True)

        notesbar = ttk.Frame(self.root, padding=(8, 2))
        notesbar.pack(fill=tk.X)
        ttk.Label(notesbar, text="备注").pack(side=tk.LEFT)
        self.notes_var = tk.StringVar()
        ttk.Entry(notesbar, textvariable=self.notes_var).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=(4, 0)
        )

        helpbar = ttk.Frame(self.root, padding=(8, 2, 8, 6))
        helpbar.pack(fill=tk.X)
        self.message_var = tk.StringVar()
        ttk.Label(
            helpbar,
            textvariable=self.message_var,
            foreground="#8a4b00",
            anchor="w",
        ).pack(fill=tk.X)
        ttk.Label(
            helpbar,
            text=(
                "s/e 设起止 · ,/. 切段 · a 跳锚点 · 空格播放 · ←/→ 逐帧(Shift ±10) · "
                "PgUp/PgDn ±1s · [/] 速度 · +/- 缩放 · Enter 保存下一个 · Ctrl+S 保存"
            ),
            foreground="#555555",
            anchor="w",
        ).pack(fill=tk.X)

    def _bind_keys(self) -> None:
        self.root.bind_all("<Key>", self.on_key)

    # -- camera loading -----------------------------------------------------

    @property
    def candidate(self) -> SourceCandidate:
        return self.candidates[self.index]

    def _rows_for_camera(self, chute: int, cam: int) -> dict[int, dict]:
        out: dict[int, dict] = {}
        for row in self.rows:
            if ms.to_int(row.get("chute")) == chute and ms.to_int(row.get("cam")) == cam:
                out[ms.to_int(row.get("segment_index"))] = row
        return out

    def load_index(self, index: int) -> None:
        self.stop_playback()
        if self.store is not None:
            self.store.close()
            self.store = None
        self.index = max(0, min(index, len(self.candidates) - 1))
        cand = self.candidate
        self.notes_var.set("")
        try:
            self.store = McfdFrameStore(cand.abs_path)
        except Exception as exc:  # noqa: BLE001 - shown to the user
            self.store = None
            self.states = []
            self._seg_buttons = []
            for child in self.seg_frame.winfo_children():
                child.destroy()
            self.image_label.configure(image="", text=f"无法读取视频\n{exc}")
            self._photo = None
            self._update_texts()
            return

        existing = self._rows_for_camera(cand.chute, cand.cam)
        self.states = []
        for seg in cand.segments:
            row = existing.get(seg.index)
            start = end = None
            if row is not None and (row.get("status") or "").strip() == ms.ANNOTATED:
                start = ms.to_int(row.get("start_frame"), -1)
                end = ms.to_int(row.get("end_frame"), -1)
                if start < 0:
                    start = None
                if end < 0:
                    end = None
            self.states.append(SegmentState(segment=seg, start=start, end=end))
            if row is not None:
                note = (row.get("notes") or "").strip()
                if note and not self.notes_var.get():
                    self.notes_var.set(note)

        self.segment_index = 0
        for i, state in enumerate(self.states):
            if state.start is None or state.end is None:
                self.segment_index = i
                break

        if self.store is not None and not self.store.cached:
            self._message("视频较长，使用 seek 模式；逐帧精度可能 ±1 帧，QC 会复核。")

        self._rebuild_segment_buttons()
        self._update_scales()
        self.frame_idx = self.states[self.segment_index].segment.anchor_start if self.states else 0
        self.dirty = False
        self.render()

    def _rebuild_segment_buttons(self) -> None:
        for child in self.seg_frame.winfo_children():
            child.destroy()
        self._seg_buttons = []
        for i, state in enumerate(self.states):
            seg = state.segment
            mark = "▶ " if i == self.segment_index else ""
            text = f"{mark}段{i + 1} 锚点 f{seg.anchor_start} ({seg.anchor_frames}f)"
            btn = ttk.Button(
                self.seg_frame,
                text=text,
                takefocus=False,
                command=lambda idx=i: self.select_segment(idx),
            )
            btn.pack(side=tk.LEFT, padx=(0, 4))
            self._seg_buttons.append(btn)

    def _update_scales(self) -> None:
        if self.store is None:
            return
        n = self.store.n_frames
        self.scale.configure(to=max(0, n - 1))
        self._update_fine_window()

    def _update_fine_window(self) -> None:
        if self.store is None or not self.states:
            return
        state = self.states[self.segment_index]
        seg = state.segment
        fps = self.store.fps or self.candidate.fps or 120.0
        margin = int(round(2.0 * fps))
        values = [seg.anchor_start, seg.anchor_end]
        if state.start is not None:
            values.append(state.start)
        if state.end is not None:
            values.append(state.end)
        lo = max(0, min(values) - margin)
        hi = min(self.store.n_frames - 1, max(values) + margin)
        if hi - lo < int(2 * fps):
            hi = min(self.store.n_frames - 1, lo + int(2 * fps))
        if hi <= lo:
            hi = min(self.store.n_frames - 1, lo + 1)
        self._fine_window = (lo, hi)
        self._syncing_fine = True
        self.fine.configure(from_=lo, to=hi)
        self._syncing_fine = False

    # -- segment state ------------------------------------------------------

    def select_segment(self, index: int) -> None:
        if not 0 <= index < len(self.states):
            return
        self.stop_playback()
        self.segment_index = index
        self._rebuild_segment_buttons()
        self._update_fine_window()
        self.frame_idx = self.states[index].segment.anchor_start
        self.render()
        self.root.focus_set()

    def _active(self) -> SegmentState:
        return self.states[self.segment_index]

    def set_start(self) -> None:
        if self.store is None:
            return
        self._active().start = self.frame_idx
        self.dirty = True
        self._message(f"段{self.segment_index + 1} 开始 -> f{self.frame_idx}")
        self.render()

    def set_end(self) -> None:
        if self.store is None:
            return
        self._active().end = self.frame_idx
        self.dirty = True
        self._message(f"段{self.segment_index + 1} 结束 -> f{self.frame_idx}")
        self.render()

    def clear_interval(self) -> None:
        self._active().start = None
        self._active().end = None
        self.dirty = True
        self._message(f"段{self.segment_index + 1} 已清除")
        self.render()

    # -- rendering ----------------------------------------------------------

    @property
    def _n_frames(self) -> int:
        if self.store is not None:
            return self.store.n_frames
        return self.candidate.n_frames

    def _fps(self) -> float:
        if self.store is not None and self.store.fps > 0:
            return self.store.fps
        return self.candidate.fps or 120.0

    def render(self) -> None:
        if self.store is None:
            self._update_texts()
            return
        frame = self.store.get(self.frame_idx)
        if frame is None:
            self._message(f"第 {self.frame_idx} 帧读取失败。")
            return

        frame = frame.copy()
        state = self._active() if self.states else None
        color = None
        if state is not None and state.start is not None and state.end is not None:
            if state.start <= self.frame_idx <= state.end:
                color = (0, 190, 0)
            else:
                color = (0, 190, 255)
        elif state is not None and (state.start is not None or state.end is not None):
            color = (0, 190, 255)
        if color is not None:
            cv2.rectangle(frame, (0, 0), (frame.shape[1] - 1, frame.shape[0] - 1), color, 2)

        if self.zoom != 1:
            frame = cv2.resize(
                frame,
                (frame.shape[1] * self.zoom, frame.shape[0] * self.zoom),
                interpolation=cv2.INTER_NEAREST,
            )
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self._photo = self._to_photo(rgb)
        self.image_label.configure(image=self._photo, text="")

        self._syncing = True
        self.scale_var.set(self.frame_idx)
        self._syncing = False
        self._syncing_fine = True
        self.fine_var.set(self.frame_idx)
        self._syncing_fine = False
        self._update_texts()
        self.draw_timeline()

    def _to_photo(self, rgb: np.ndarray):
        if _HAS_PIL:
            return ImageTk.PhotoImage(Image.fromarray(rgb))
        ok, buffer = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        if not ok:
            raise RuntimeError("PNG encode failed")
        data = base64.b64encode(buffer.tobytes()).decode("ascii")
        return tk.PhotoImage(data=data, format="png")

    def _update_texts(self) -> None:
        cand = self.candidate
        complete = sum(
            1
            for c in self.candidates
            if ms.is_camera_complete(self.rows, self.segments, c.chute, c.cam)
        )
        self.info_var.set(
            f"{self.index + 1}/{len(self.candidates)}  |  {cand.video_id}  |  "
            f"cov={cand.coverage:.3f}  |  已完成 {complete}"
        )
        fps = self._fps()
        seconds = self.frame_idx / fps if fps > 0 else 0.0
        mode = "cache" if (self.store is not None and self.store.cached) else "seek"
        self.position_var.set(
            f"帧 {self.frame_idx}（0-based，共 {self._n_frames} 帧）  |  {seconds:.3f} s  |  "
            f"fps {fps:g}  |  {mode}"
        )

        if not self.states:
            self.interval_var.set("")
            return
        state = self._active()
        start_text = "-" if state.start is None else str(state.start)
        end_text = "-" if state.end is None else str(state.end)
        duration = ""
        if state.start is not None and state.end is not None:
            frames = state.end - state.start + 1
            duration = f"  |  {frames} 帧 = {frames / fps:.2f} s"
        self.interval_var.set(
            f"段 {self.segment_index + 1}/{len(self.states)}  "
            f"锚点 f{state.segment.anchor_start}  |  开始 {start_text}  结束 {end_text}{duration}  |  "
            f"播放 {self.speed:g}x"
        )

    def draw_timeline(self) -> None:
        if self._timeline_job is not None:
            self.root.after_cancel(self._timeline_job)
        self._timeline_job = self.root.after(30, self._draw_timeline_now)

    def _draw_timeline_now(self) -> None:
        self._timeline_job = None
        canvas = self.timeline
        width = max(1, canvas.winfo_width())
        height = max(1, canvas.winfo_height())
        canvas.delete("all")
        if not self.states:
            return
        n = max(1, self._n_frames - 1)
        cand = self.candidate

        def x(frame: int) -> float:
            return max(0.0, min(width, frame / n * width))

        # label-0 context (v2 raw rows)
        for start, end in cand.context_spans:
            canvas.create_rectangle(x(start), 4, x(max(start, end - 1)), height - 4, fill="#3a3a3a", outline="")
        # label-1 anchors
        for state in self.states:
            for start, end in state.segment.rows:
                canvas.create_rectangle(
                    x(start), 4, x(max(start, end - 1)), height - 4, fill="#b36b00", outline=""
                )
        # active edited interval
        state = self._active()
        if state.start is not None and state.end is not None:
            canvas.create_rectangle(
                x(state.start), 2, x(state.end), height - 2, outline="#00b400", width=2
            )
        # anchor marker
        ax = x(state.segment.anchor_start)
        canvas.create_line(ax, 0, ax, height, fill="#ffffff", width=1)
        # playhead
        px = x(self.frame_idx)
        canvas.create_line(px, 0, px, height, fill="#ff4040", width=2)

    # -- playback / navigation ---------------------------------------------

    def _play_delay_ms(self) -> int:
        fps = self._fps() or 30.0
        return max(20, int(1000.0 / (fps * self.speed)))

    def toggle_play(self) -> None:
        if self.store is None:
            return
        if self.playing:
            self.stop_playback()
            return
        if self.frame_idx >= self._n_frames - 1:
            self.frame_idx = 0
        self.playing = True
        self._tick()

    def _tick(self) -> None:
        if not self.playing or self.store is None:
            return
        if self.frame_idx >= self._n_frames - 1:
            self.playing = False
            self._after_id = None
            return
        self.frame_idx += 1
        self.render()
        self._after_id = self.root.after(self._play_delay_ms(), self._tick)

    def stop_playback(self) -> None:
        self.playing = False
        if self._after_id is not None:
            try:
                self.root.after_cancel(self._after_id)
            except tk.TclError:
                pass
            self._after_id = None

    def step(self, delta: int) -> None:
        self.stop_playback()
        self.frame_idx = max(0, min(self._n_frames - 1, self.frame_idx + delta))
        self.render()

    def go_to(self, index: int) -> None:
        self.stop_playback()
        self.frame_idx = max(0, min(self._n_frames - 1, index))
        self.render()

    def on_scale(self, _value) -> None:
        if self._syncing or self.store is None:
            return
        try:
            index = int(round(float(self.scale_var.get())))
        except (TypeError, ValueError):
            return
        self.go_to(index)
        self.root.focus_set()

    def on_fine(self, _value) -> None:
        if self._syncing_fine or self.store is None:
            return
        try:
            index = int(round(float(self.fine_var.get())))
        except (TypeError, ValueError):
            return
        self.go_to(index)
        self.root.focus_set()

    def on_timeline_click(self, event) -> None:
        if self.store is None:
            return
        width = max(1, self.timeline.winfo_width())
        frame = int(round(event.x / width * (self._n_frames - 1)))
        self.go_to(frame)
        self.root.focus_set()

    def _next_incomplete(self, start: int) -> int:
        total = len(self.candidates)
        for offset in range(1, total + 1):
            i = (start + offset) % total
            cand = self.candidates[i]
            if not ms.is_camera_complete(self.rows, self.segments, cand.chute, cand.cam):
                return i
        return start

    def go_next(self) -> None:
        if self.index + 1 >= len(self.candidates):
            self._message("已经是最后一个相机。")
            return
        self.load_index(self.index + 1)

    def go_prev(self) -> None:
        if self.index == 0:
            self._message("已经是第一个相机。")
            return
        self.load_index(self.index - 1)

    # -- saving -------------------------------------------------------------

    def _validate_states(self) -> str:
        n = self._n_frames
        for state in self.states:
            seg = state.segment
            if state.start is None or state.end is None:
                return f"段{seg.index + 1} 还没设起止（s / e）。"
            if not (0 <= state.start <= state.end < n):
                return f"段{seg.index + 1} 区间非法：{state.start}..{state.end}，共 {n} 帧。"
        ordered = sorted(self.states, key=lambda s: s.start if s.start is not None else -1)
        for prev, cur in zip(ordered, ordered[1:]):
            if cur.start <= prev.end:
                return f"段{prev.segment.index + 1} 与段{cur.segment.index + 1} 重叠。"
        return ""

    def save_current(self, advance: bool) -> bool:
        if self.store is None:
            return False
        error = self._validate_states()
        if error:
            self._message(error)
            return False

        cand = self.candidate
        now = datetime.now().isoformat(timespec="seconds")
        notes = self.notes_var.get().strip()
        new_rows = []
        for state in self.states:
            assert state.start is not None and state.end is not None
            seg = state.segment
            row_notes = notes
            if self.store.n_frames != cand.n_frames:
                extra = f"decoded_frames={self.store.n_frames} manifest_frames={cand.n_frames}"
                row_notes = f"{row_notes} | {extra}".strip(" |")
            new_rows.append(
                {
                    "video_id": cand.video_id,
                    "dataset": ms.DATASET,
                    "event_id": cand.event_id,
                    "chute": cand.chute,
                    "cam": cand.cam,
                    "segment_index": seg.index,
                    "segment_count": len(self.states),
                    "label": ms.LABEL,
                    "start_frame": state.start,
                    "end_frame": state.end,
                    "anchor_start_frame": seg.anchor_start,
                    "offset_frames": state.start - seg.anchor_start,
                    "n_frames": cand.n_frames,
                    "fps": f"{cand.fps:.6g}",
                    "status": ms.ANNOTATED,
                    "source_video_id": "",
                    "annotator": self.annotator,
                    "annotated_at": now,
                    "notes": row_notes,
                }
            )

        kept = [
            r
            for r in self.rows
            if not (ms.to_int(r.get("chute")) == cand.chute and ms.to_int(r.get("cam")) == cand.cam)
        ]
        rows = kept + new_rows
        try:
            ms.validate_manual(rows, self.segments, ms.load_video_meta())
        except ValueError as exc:
            self._message(f"校验失败，未写入：{exc}")
            return False
        ms.write_manual_atomic(self.output, rows)
        self.rows = rows
        self.dirty = False
        self._message(f"已保存 {cand.video_id}（{len(new_rows)} 段）-> {self.output}")
        self.root.focus_set()

        if advance:
            nxt = self._next_incomplete(self.index)
            if nxt == self.index:
                self._message("全部相机已完成。Ctrl+S 可覆盖保存。")
                self._update_texts()
            else:
                self.load_index(nxt)
        else:
            self._update_texts()
            self._rebuild_segment_buttons()
        return True

    # -- events -------------------------------------------------------------

    def _in_entry(self) -> bool:
        widget = self.root.focus_get()
        return isinstance(widget, (tk.Entry, ttk.Entry))

    def on_key(self, event: tk.Event) -> None:
        key = event.keysym
        shift = bool(event.state & 0x0001)
        ctrl = bool(event.state & 0x0004)
        if ctrl and key.lower() == "s":
            self.save_current(advance=False)
            return
        if self._in_entry() and key not in ("Return", "Escape"):
            return

        if key == "space":
            self.toggle_play()
        elif key == "Left":
            self.step(-10 if shift else -1)
        elif key == "Right":
            self.step(10 if shift else 1)
        elif key == "Home":
            self.go_to(0)
        elif key == "End":
            self.go_to(self._n_frames - 1)
        elif key == "Prior":
            self.step(-int(self._fps()))
        elif key == "Next":
            self.step(int(self._fps()))
        elif key == "s":
            self.set_start()
        elif key == "e":
            self.set_end()
        elif key == "c":
            self.clear_interval()
        elif key in ("comma", "less"):
            if self.states:
                self.select_segment((self.segment_index - 1) % len(self.states))
        elif key in ("period", "greater"):
            if self.states:
                self.select_segment((self.segment_index + 1) % len(self.states))
        elif key == "a":
            if self.states:
                self.go_to(self._active().segment.anchor_start)
        elif key == "Return":
            self.save_current(advance=True)
        elif key == "BackSpace":
            self.go_prev()
        elif key in ("plus", "equal", "KP_Add"):
            self.zoom = min(3, self.zoom + 1)
            self.render()
        elif key in ("minus", "KP_Subtract"):
            self.zoom = max(1, self.zoom - 1)
            self.render()
        elif key == "bracketleft":
            self.speed = max(0.0625, self.speed / 2)
            self._message(f"播放速度 {self.speed:g}x")
            self._update_texts()
        elif key == "bracketright":
            self.speed = min(2.0, self.speed * 2)
            self._message(f"播放速度 {self.speed:g}x")
            self._update_texts()
        elif key == "Escape":
            self.stop_playback()

    def on_close(self) -> None:
        self.stop_playback()
        if self.dirty:
            answer = messagebox.askyesnocancel("退出", "当前相机标注未保存，是否保存后退出？")
            if answer is None:
                return
            if answer and not self.save_current(advance=False):
                return
        if self.store is not None:
            self.store.close()
            self.store = None
        self.root.destroy()

    def _message(self, text: str) -> None:
        self.message_var.set(text)


# ---------------------------------------------------------------------------
# Self test
# ---------------------------------------------------------------------------


def self_test(candidates: list[SourceCandidate], limit: int) -> int:
    """Decode each selected camera and compare seek reads with sequential reads."""
    failures = 0
    for cand in candidates[:limit]:
        try:
            store = McfdFrameStore(cand.abs_path)
        except Exception as exc:  # noqa: BLE001
            print(f"[FAIL] {cand.video_id}: {exc}")
            failures += 1
            continue
        mismatch = "" if store.n_frames == cand.n_frames else f" manifest={cand.n_frames}"
        line = (
            f"[OK] {cand.video_id}: frames={store.n_frames} fps={store.fps:g} "
            f"{store.width}x{store.height} mode={'cache' if store.cached else 'seek'}{mismatch}"
        )
        if not store.cached:
            probes = sorted({0, store.n_frames // 4, store.n_frames // 2, 3 * store.n_frames // 4, store.n_frames - 1})
            cap = cv2.VideoCapture(str(cand.abs_path))
            sequential: dict[int, np.ndarray] = {}
            wanted = set(probes)
            index = 0
            while wanted:
                ok, frame = cap.read()
                if not ok:
                    break
                if index in wanted:
                    sequential[index] = frame
                    wanted.discard(index)
                index += 1
            cap.release()
            diffs = []
            exact = 0
            for idx in probes:
                got = store.get(idx)
                ref = sequential.get(idx)
                if got is None or ref is None:
                    diffs.append(999)
                    continue
                if np.array_equal(got, ref):
                    exact += 1
                else:
                    diffs.append(float(np.abs(got.astype(np.int16) - ref.astype(np.int16)).mean()))
            line += f" seek_match={exact}/{len(probes)} max_mean_abs_diff={max(diffs) if diffs else 0:.2f}"
        print(line)
        store.close()
    print(f"self-test: {min(limit, len(candidates)) - failures} ok, {failures} failed")
    return 0 if failures == 0 else 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Annotate true fall intervals on one MCFD camera per chute."
    )
    parser.add_argument("--output", type=Path, default=paths.MCFD_FALL_INTERVALS)
    parser.add_argument(
        "--annotator",
        default=os.environ.get("USERNAME") or getpass.getuser(),
    )
    parser.add_argument("--chute", type=int, default=None, help="limit to one chute (01..23)")
    parser.add_argument("--cam", type=int, default=None, help="override the selected source camera")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--self-test", type=int, default=0, metavar="N")
    parser.add_argument("--start-at", default="")
    parser.add_argument("--redo", action="store_true")
    parser.add_argument("--reset-chute", type=int, default=None)
    return parser


def _find_start(candidates: list[SourceCandidate], needle: str) -> int:
    for i, cand in enumerate(candidates):
        if cand.video_id == needle:
            return i
    matches = [i for i, c in enumerate(candidates) if needle.lower() in c.video_id.lower()]
    if len(matches) == 1:
        return matches[0]
    raise SystemExit(f"--start-at 无法唯一匹配：{needle!r}（命中 {len(matches)} 个）")


def main(argv: list[str] | None = None) -> int:
    _force_utf8_stdout()
    args = build_parser().parse_args(argv)
    if args.cam is not None and args.chute is None:
        raise SystemExit("--cam 需要同时指定 --chute")

    all_rows = ms.load_mcfd_rows()
    segments = ms.merged_segments(all_rows)
    meta = ms.load_video_meta()
    coverage = ms.load_coverage()
    selected = ms.select_source_cameras(segments, coverage, meta)
    candidates = build_candidates(
        selected,
        segments,
        meta,
        coverage,
        all_rows,
        only_chute=args.chute,
        cam_override=args.cam,
    )
    if not candidates:
        print("没有匹配的 MCFD 相机。")
        return 1

    manual = ms.load_manual(args.output)

    if args.reset_chute is not None:
        chute = args.reset_chute
        answer = input(f"输入 {chute:02d} 确认删除 chute{chute:02d} 的全部人工行：").strip()
        if answer != f"{chute:02d}":
            print("已取消。")
            return 1
        kept = [r for r in manual if ms.to_int(r.get("chute")) != chute]
        ms.write_manual_atomic(args.output, kept)
        print(f"已删除 chute{chute:02d}：{len(manual) - len(kept)} 行。")
        return 0

    if args.list:
        print_candidates(candidates, manual)
        return 0

    if args.plan:
        if args.chute is None:
            print("--plan 需要 --chute")
            return 1
        cand = candidates[0]
        print(f"chute{cand.chute:02d}: source {cand.video_id} ({cand.relative_path})")
        for seg in cand.segments:
            print(
                f"  segment {seg.index}: anchor {seg.anchor_start}..{seg.anchor_end - 1} "
                f"({seg.anchor_frames} frames, {seg.anchor_frames / cand.fps:.2f} s)"
            )
        print("  targets:")
        for cam in range(1, 9):
            if cam == cand.cam:
                continue
            tgt = segments[(cand.chute, cam)]
            spans = ", ".join(f"seg{s.index}:f{s.anchor_start}" for s in tgt)
            print(f"    cam{cam}: {spans}")
        return 0

    if args.self_test > 0:
        return self_test(candidates, args.self_test)

    if args.start_at:
        start_index = _find_start(candidates, args.start_at)
    elif args.redo:
        start_index = 0
    else:
        start_index = 0
        for i, cand in enumerate(candidates):
            if not ms.is_camera_complete(manual, segments, cand.chute, cand.cam):
                start_index = i
                break

    done = sum(
        1
        for c in candidates
        if ms.is_camera_complete(manual, segments, c.chute, c.cam)
    )
    print(
        f"source cameras={len(candidates)}  done={done}  pending={len(candidates) - done}  "
        f"output={args.output}"
    )

    root = tk.Tk()
    McfdAnnotatorApp(
        root=root,
        candidates=candidates,
        segments=segments,
        rows=manual,
        output=args.output,
        annotator=args.annotator,
        start_index=start_index,
    )
    root.mainloop()

    manual = ms.load_manual(args.output)
    done = sum(
        1
        for c in candidates
        if ms.is_camera_complete(manual, segments, c.chute, c.cam)
    )
    print(f"结束：done={done}/{len(candidates)}  output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
