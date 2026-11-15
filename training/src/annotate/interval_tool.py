"""Frame-level fall-interval annotation tool (tkinter + OpenCV)."""
from __future__ import annotations

import argparse
import base64
import csv
import getpass
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from src.data.paths import ANNOTATIONS, MANIFESTS, RAW, read_csv

try:  # Pillow is optional; Tk 8.6 can render PNG natively as a fallback.
    from PIL import Image, ImageTk

    _HAS_PIL = True
except ImportError:  # pragma: no cover - depends on the local environment
    Image = None  # type: ignore[assignment]
    ImageTk = None  # type: ignore[assignment]
    _HAS_PIL = False

import tkinter as tk
from tkinter import messagebox, ttk

OUTPUT_FIELDS = [
    "video_id",
    "dataset",
    "event_id",
    "label",
    "start_frame",
    "end_frame",
    "n_frames",
    "fps",
    "status",
    "annotator",
    "annotated_at",
    "notes",
]

# Fields appended only for segmented scopes. Keeping them out of OUTPUT_FIELDS
# leaves the hash-pinned manual layer byte-identical (see module docstring).
SEGMENT_FIELDS = ["segment_index", "segment_count"]

# Scopes where one clip may contain more than one fall. Everything else keeps
# the historical one-interval-per-video contract.
SEGMENTED_SCOPES = frozenset({"board"})

DONE_STATUSES = {"annotated", "no_fall", "unclear"}
STATUS_TEXT = {
    "": "未确认",
    "annotated": "跌倒（已标区间）",
    "no_fall": "无跌倒（可作负样本）",
    "unclear": "无法判断（排除）",
}
STATUS_COLOR = {
    "": "#444444",
    "annotated": "#0a7a28",
    "no_fall": "#1450a0",
    "unclear": "#a35a00",
}

SCOPE_ORDER = ("urfd-fall", "le2i-review", "le2i-unannotated")
SCOPE_TEXT = {
    "urfd-fall": "URFD fall（30 个，无起止）",
    "le2i-review": "Le2i 异常标注（31 invalid_interval + 3 missing_header）",
    "le2i-unannotated": "Le2i 无标注（60 个，筛负样本）",
    "board": "板端 RK3576 录像（18 个，仅评估，不进训练）",
}

DEFAULT_OUTPUT = ANNOTATIONS / "manual" / "interval_annotations.csv"

# Pre-decode frames up to this many; longer videos fall back to seek mode.
MAX_CACHED_FRAMES = 4000


# ---------------------------------------------------------------------------
# Candidates
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Candidate:
    video_id: str
    dataset: str
    event_id: str
    label: str
    annotation_status: str
    relative_path: str
    n_frames: int
    fps: float
    width: int
    height: int
    use_half: str

    @property
    def abs_path(self) -> Path:
        return RAW / self.relative_path

    @property
    def short_path(self) -> str:
        return self.relative_path


def _matches(scope: str, row: dict) -> bool:
    dataset = row.get("dataset", "")
    status = row.get("annotation_status", "")
    label = row.get("label", "")
    if scope == "urfd-fall":
        return dataset == "urfd" and label == "fall"
    if scope == "le2i-review":
        return dataset == "le2i" and status in ("invalid_interval", "missing_header")
    if scope == "le2i-unannotated":
        return dataset == "le2i" and status == "unannotated"
    if scope == "board":
        # The board manifest carries no label/annotation_status: every clip is
        # unlabelled and gets reviewed. Clips too short to window are skipped
        # here for the same reason scripts/11 skips them.
        return dataset == "board" and _to_int(row.get("n_frames", "0")) >= 10
    raise ValueError(f"unknown scope: {scope}")


def _to_int(value: str, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _to_float(value: str, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def load_candidates(scope: str, manifest: Path | None = None) -> list[Candidate]:
    """Build the candidate list from the manifest (single source of truth)."""
    if manifest is None:
        manifest = MANIFESTS / "videos.csv"
    rows = read_csv(manifest)
    scopes = SCOPE_ORDER if scope == "all" else (scope,)

    seen: set[str] = set()
    out: list[Candidate] = []
    for current in scopes:
        for row in rows:
            if not _matches(current, row):
                continue
            video_id = row.get("video_id", "")
            if not video_id or video_id in seen:
                continue
            if row.get("readable", "True") not in ("True", "true", "1"):
                print(f"[warn] unreadable, skipped: {video_id}")
                continue
            seen.add(video_id)
            out.append(
                Candidate(
                    video_id=video_id,
                    dataset=row.get("dataset", ""),
                    event_id=row.get("event_id", ""),
                    label=row.get("label", ""),
                    annotation_status=row.get("annotation_status", ""),
                    relative_path=row.get("relative_path", ""),
                    n_frames=_to_int(row.get("n_frames", "0")),
                    fps=_to_float(row.get("fps", "0")),
                    width=_to_int(row.get("width", "0")),
                    height=_to_int(row.get("height", "0")),
                    use_half=row.get("use_half", ""),
                )
            )
    return out


# ---------------------------------------------------------------------------
# Annotation CSV IO
# ---------------------------------------------------------------------------


@dataclass
class Segment:
    """One fall interval inside a clip."""

    start: int | None = None
    end: int | None = None

    @property
    def complete(self) -> bool:
        return self.start is not None and self.end is not None


def load_existing(path: Path) -> dict[str, list[dict]]:
    """Load saved rows, grouped per video."""
    if not path.exists():
        return {}
    grouped: dict[str, list[dict]] = {}
    for row in read_csv(path):
        video_id = (row.get("video_id") or "").strip()
        if not video_id:
            continue
        grouped.setdefault(video_id, []).append(row)
    for video_id, rows in grouped.items():
        rows.sort(key=lambda r: _to_int(r.get("segment_index", "0")))
    return grouped


def _has_segment_columns(path: Path) -> bool:
    """True if the file was written by a segmented scope (extra columns)."""
    if not path.exists():
        return False
    with open(path, "r", newline="", encoding="utf-8-sig") as fh:
        header = next(csv.reader(fh), [])
    return all(field in header for field in SEGMENT_FIELDS)


def write_rows(
    path: Path,
    rows: dict[str, list[dict]],
    order: Iterable[str],
    segmented: bool = False,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = [vid for vid in order if vid in rows]
    ordered_set = set(ordered)
    ordered.extend(sorted(vid for vid in rows if vid not in ordered_set))

    fields = OUTPUT_FIELDS + (SEGMENT_FIELDS if segmented else [])

    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for video_id in ordered:
            for row in rows[video_id]:
                writer.writerow(row)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Video reader
# ---------------------------------------------------------------------------


class FrameStore:
    """Decoded frames for one video."""

    def __init__(self, path: Path, max_cache: int = MAX_CACHED_FRAMES) -> None:
        self.path = path
        self.max_cache = max(1, int(max_cache))
        self._cap: cv2.VideoCapture | None = None
        self._frames: list[np.ndarray] | None = None
        self.fps = 0.0
        self.width = 0
        self.height = 0
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

        frames: list[np.ndarray] = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
            if len(frames) > self.max_cache:
                break

        if len(frames) > self.max_cache:
            # Too long to cache: drop buffers and use random seek.
            frames = []
            self._cap.release()
            self._cap = cv2.VideoCapture(str(self.path))
            self._n_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            self.cached = False
        else:
            self._frames = frames
            self._n_frames = len(frames)
            self.cached = True
            cap.release()
            self._cap = None

        if self._n_frames <= 0:
            self.close()
            raise RuntimeError(f"视频无有效帧：{self.path}")

    @property
    def n_frames(self) -> int:
        return self._n_frames

    def get(self, index: int) -> np.ndarray | None:
        if index < 0 or index >= self._n_frames:
            return None
        if self._frames is not None:
            return self._frames[index]
        assert self._cap is not None
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, float(index))
        ok, frame = self._cap.read()
        return frame if ok else None

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._frames = None


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------


class AnnotatorApp:
    def __init__(
        self,
        root: tk.Tk,
        candidates: list[Candidate],
        rows: dict[str, list[dict]],
        output: Path,
        annotator: str,
        start_index: int = 0,
        segmented: bool = False,
    ) -> None:
        self.root = root
        self.candidates = candidates
        self.rows = rows
        self.output = output
        self.annotator = annotator
        self.order = [c.video_id for c in candidates]
        self.segmented = segmented

        self.index = max(0, min(start_index, len(candidates) - 1))
        self.store: FrameStore | None = None
        self.frame_idx = 0
        self.segments: list[Segment] = [Segment()]
        self.segment_index = 0
        self.status = ""
        self.dirty = False
        self.playing = False
        self._after_id: str | None = None
        self._syncing_scale = False
        self.speed = 1.0
        self.zoom = 2
        self.show_half = True
        self._photo = None
        self._seg_signature: tuple | None = None

        self._build_ui()
        self._bind_keys()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.load_index(self.index)

    # -- UI -----------------------------------------------------------------

    def _build_ui(self) -> None:
        self.root.title("跌倒区间标注器")
        self.root.geometry("900x760")
        self.root.minsize(700, 560)

        top = ttk.Frame(self.root, padding=(8, 6, 8, 2))
        top.pack(fill=tk.X)
        self.info_var = tk.StringVar(value="")
        ttk.Label(top, textvariable=self.info_var, anchor="w").pack(fill=tk.X)

        self.image_label = tk.Label(
            self.root, bg="#111111", fg="white", bd=1, relief=tk.SUNKEN, anchor=tk.CENTER
        )
        self.image_label.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

        seek = ttk.Frame(self.root, padding=(8, 0))
        seek.pack(fill=tk.X)
        self.scale_var = tk.IntVar(value=0)
        self.scale = ttk.Scale(
            seek,
            from_=0,
            to=0,
            orient=tk.HORIZONTAL,
            variable=self.scale_var,
            command=self.on_scale,
        )
        self.scale.pack(fill=tk.X)

        self.position_var = tk.StringVar(value="")
        ttk.Label(seek, textvariable=self.position_var, anchor="w").pack(fill=tk.X)

        controls = ttk.Frame(self.root, padding=(8, 4))
        controls.pack(fill=tk.X)
        buttons = [
            ("播放/暂停", self.toggle_play),
            ("-10", lambda: self.step(-10)),
            ("-1", lambda: self.step(-1)),
            ("+1", lambda: self.step(1)),
            ("+10", lambda: self.step(10)),
            ("设为开始 (s)", self.set_start),
            ("设为结束 (e)", self.set_end),
            ("清除 (c)", self.clear_interval),
        ]
        for text, command in buttons:
            ttk.Button(controls, text=text, command=command).pack(side=tk.LEFT, padx=2)

        # Segment controls exist only for scopes that allow several falls per
        # clip; the training scopes keep the original one-interval layout.
        if self.segmented:
            segbar = ttk.Frame(self.root, padding=(8, 0))
            segbar.pack(fill=tk.X)
            ttk.Button(
                segbar, text="◀ 上一段 (,)",
                command=lambda: self.select_segment((self.segment_index - 1) % max(1, len(self.segments))),
            ).pack(side=tk.LEFT, padx=2)
            ttk.Button(
                segbar, text="下一段 ▶ (.)",
                command=lambda: self.select_segment((self.segment_index + 1) % max(1, len(self.segments))),
            ).pack(side=tk.LEFT, padx=2)
            ttk.Button(segbar, text="+ 新增段 (n)", command=self.add_segment).pack(side=tk.LEFT, padx=2)
            ttk.Button(segbar, text="− 删除当前段 (d)", command=self.delete_segment).pack(side=tk.LEFT, padx=2)
            self.seg_frame = ttk.Frame(segbar)
            self.seg_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(10, 0))

        marks = ttk.Frame(self.root, padding=(8, 0))
        marks.pack(fill=tk.X)
        ttk.Button(marks, text="1 跌倒", command=lambda: self.mark("annotated")).pack(
            side=tk.LEFT, padx=2
        )
        ttk.Button(marks, text="0 无跌倒", command=lambda: self.mark("no_fall")).pack(
            side=tk.LEFT, padx=2
        )
        ttk.Button(marks, text="9 无法判断", command=lambda: self.mark("unclear")).pack(
            side=tk.LEFT, padx=2
        )
        ttk.Button(marks, text="保存并下一个 (Enter)", command=self.save_and_next).pack(
            side=tk.RIGHT, padx=2
        )

        status = ttk.Frame(self.root, padding=(8, 4))
        status.pack(fill=tk.X)
        self.interval_var = tk.StringVar(value="")
        ttk.Label(status, textvariable=self.interval_var, anchor="w").pack(side=tk.LEFT)
        self.status_var = tk.StringVar(value="")
        self.status_label = tk.Label(
            status, textvariable=self.status_var, anchor="e", font=("Microsoft YaHei UI", 10, "bold")
        )
        self.status_label.pack(side=tk.RIGHT)

        notes = ttk.Frame(self.root, padding=(8, 2))
        notes.pack(fill=tk.X)
        ttk.Label(notes, text="备注：").pack(side=tk.LEFT)
        self.notes_var = tk.StringVar(value="")
        self.notes_entry = ttk.Entry(notes, textvariable=self.notes_var)
        self.notes_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)

        help_text = (
            "空格 播放/暂停 | ←/→ ±1 帧 | Shift+←/→ ±10 帧 | s 开始 | e 结束 | "
            "c 清除 | 1 跌倒 | 0 无跌倒 | 9 无法判断 | Enter 保存并下一个 | "
            "Backspace 上一个 | Ctrl+S 保存 | f 切换右半/全幅 | +/- 缩放 | [/] 速度"
        )
        if self.segmented:
            help_text += " | , . 切换段 | n 新增段 | d 删除段"
        ttk.Label(self.root, text=help_text, foreground="#666666", wraplength=880).pack(
            fill=tk.X, padx=8, pady=(2, 6)
        )

        self.message_var = tk.StringVar(value="")
        ttk.Label(self.root, textvariable=self.message_var, foreground="#8a4b00").pack(
            fill=tk.X, padx=8, pady=(0, 4)
        )

    def _bind_keys(self) -> None:
        self.root.bind("<KeyPress>", self.on_key)
        self.root.bind("<Control-s>", lambda _e: self.save_current(advance=False))
        self.root.bind("<Control-S>", lambda _e: self.save_current(advance=False))

    # -- state helpers ------------------------------------------------------

    @property
    def candidate(self) -> Candidate:
        return self.candidates[self.index]

    @property
    def _active(self) -> Segment:
        return self.segments[self.segment_index]

    @property
    def start(self) -> int | None:
        """Start frame of the active segment (property keeps the old call sites)."""
        return self._active.start

    @start.setter
    def start(self, value: int | None) -> None:
        self._active.start = value

    @property
    def end(self) -> int | None:
        return self._active.end

    @end.setter
    def end(self, value: int | None) -> None:
        self._active.end = value

    def _completed_segments(self) -> list[Segment]:
        return [s for s in self.segments if s.complete]

    def _in_entry(self) -> bool:
        try:
            focused = self.root.focus_get()
        except tk.TclError:
            return False
        return isinstance(focused, (tk.Entry, ttk.Entry))

    def _message(self, text: str) -> None:
        self.message_var.set(text)

    def _n_frames(self) -> int:
        return self.store.n_frames if self.store is not None else 0

    def _row_for_current(self) -> list[dict]:
        return self.rows.get(self.candidate.video_id, [])

    def _status_of(self, video_id: str) -> str:
        return status_of(self.rows, video_id)

    # -- loading ------------------------------------------------------------

    def load_index(self, index: int) -> None:
        self.stop_playback()
        if not (0 <= index < len(self.candidates)):
            self._message("没有更多视频。")
            return
        self.index = index
        if self.store is not None:
            self.store.close()
            self.store = None

        cand = self.candidate
        self.frame_idx = 0
        self.segment_index = 0
        self.segments = [Segment()]
        self._seg_signature = None
        self.status = ""
        self.notes_var.set("")
        self.dirty = False
        self.zoom = 2 if cand.width and cand.width <= 400 else 1
        self.show_half = cand.use_half == "right"

        corrupt = ""
        saved = self._row_for_current()
        if saved:
            first = saved[0]
            self.status = (first.get("status") or "").strip()
            self.notes_var.set(first.get("notes") or "")
            if self.status == "annotated":
                loaded = [
                    Segment(
                        start=self._parse_frame(row.get("start_frame")),
                        end=self._parse_frame(row.get("end_frame")),
                    )
                    for row in saved
                ]
                if loaded and all(s.complete for s in loaded):
                    self.segments = loaded
                else:
                    # Corrupt/partial row: surface it instead of silently
                    # dropping the intervals the annotator already recorded.
                    corrupt = f"已有标注行区间不完整，未载入：{cand.video_id}"

        self._message(corrupt)
        try:
            self.store = FrameStore(cand.abs_path)
        except Exception as exc:  # noqa: BLE001 - shown to the user
            self.store = None
            self.image_label.configure(image="", text=f"无法读取视频\n{exc}")
            self._photo = None
            self._update_texts()
            return

        if not self.store.cached:
            self._message("视频较长，使用随机 seek 模式，逐帧精度可能略低。")

        self.scale.configure(to=max(0, self.store.n_frames - 1))
        self._update_texts()
        self.render()

    @staticmethod
    def _parse_frame(value) -> int | None:
        text = str(value or "").strip()
        if not text:
            return None
        try:
            return int(float(text))
        except ValueError:
            return None

    # -- rendering ----------------------------------------------------------

    def render(self) -> None:
        cand = self.candidate
        if self.store is None:
            self._update_texts()
            return

        frame = self.store.get(self.frame_idx)
        if frame is None:
            self._message(f"第 {self.frame_idx} 帧读取失败。")
            return

        frame = frame.copy()
        if self.show_half and cand.use_half == "right":
            frame = frame[:, frame.shape[1] // 2 :]

        # Border colour reports the ACTIVE segment: green inside it, amber when
        # it is only half-set or the playhead is outside it. Other segments are
        # listed in the segment bar rather than drawn, to keep the frame clean.
        seg = self._active
        if seg.complete and seg.start <= self.frame_idx <= seg.end:
            color = (0, 190, 0)
        elif seg.start is not None or seg.end is not None:
            color = (0, 190, 255)
        else:
            color = None
        if color is not None:
            cv2.rectangle(frame, (0, 0), (frame.shape[1] - 1, frame.shape[0] - 1), color, 2)

        target_w = max(1, int(frame.shape[1] * self.zoom))
        target_h = max(1, int(frame.shape[0] * self.zoom))
        frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        self._photo = self._to_photo(rgb)
        self.image_label.configure(image=self._photo, text="")

        self._syncing_scale = True
        self.scale_var.set(self.frame_idx)
        self._syncing_scale = False
        self._refresh_segment_bar()
        self._update_texts()

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
        n = self._n_frames()
        reviewed = sum(1 for c in self.candidates if self._status_of(c.video_id) in DONE_STATUSES)
        count_text = ""
        if self.segmented:
            count_text = f"  |  段 {self.segment_index + 1}/{len(self.segments)}"
        self.info_var.set(
            f"{self.index + 1}/{len(self.candidates)}  |  {cand.video_id}  |  "
            f"{cand.dataset} · {cand.label} · {cand.annotation_status}  |  已标注 {reviewed}{count_text}"
        )

        if n <= 0:
            self.position_var.set("")
            self.interval_var.set("")
            self.status_label.configure(text=STATUS_TEXT.get(self.status, self.status), fg=STATUS_COLOR.get(self.status, "#444444"))
            return

        fps = self.store.fps if self.store is not None else cand.fps
        seconds = self.frame_idx / fps if fps > 0 else 0.0
        self.position_var.set(f"帧 {self.frame_idx}（0-based，共 {n} 帧）  |  {seconds:.2f} s  |  fps {fps:g}")

        seg = self._active
        if seg.start is not None or seg.end is not None:
            start_text = "-" if seg.start is None else str(seg.start)
            end_text = "-" if seg.end is None else str(seg.end)
            duration = ""
            if seg.complete and seg.end >= seg.start:
                frames = seg.end - seg.start + 1
                secs = frames / fps if fps > 0 else 0.0
                duration = f"  |  时长 {frames} 帧 = {secs:.2f} s"
            self.interval_var.set(f"开始 {start_text}  结束 {end_text}{duration}")
        else:
            self.interval_var.set("开始 -  结束 -")

        if self.segmented and len(self.segments) > 1:
            parts = []
            for i, s in enumerate(self.segments):
                mark = "▶" if i == self.segment_index else " "
                if s.complete:
                    parts.append(f"{mark}{i + 1}:[{s.start},{s.end}]")
                else:
                    parts.append(f"{mark}{i + 1}:[未完成]")
            self.interval_var.set(self.interval_var.get() + "    |    " + "  ".join(parts))

        label_text = STATUS_TEXT.get(self.status, self.status or "未确认")
        self.status_label.configure(text=f"状态：{label_text}", fg=STATUS_COLOR.get(self.status, "#444444"))

    # -- playback -----------------------------------------------------------

    def _play_delay_ms(self) -> int:
        fps = self.store.fps if self.store is not None else 30.0
        if fps <= 0:
            fps = 30.0
        return max(20, int(1000.0 / (fps * self.speed)))

    def toggle_play(self) -> None:
        if self.store is None:
            return
        if self.playing:
            self.stop_playback()
            return
        if self.frame_idx >= self.store.n_frames - 1:
            self.frame_idx = 0
        self.playing = True
        self._tick()

    def _tick(self) -> None:
        if not self.playing or self.store is None:
            return
        if self.frame_idx >= self.store.n_frames - 1:
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

    # -- navigation ---------------------------------------------------------

    def step(self, delta: int) -> None:
        if self.store is None:
            return
        self.stop_playback()
        self.frame_idx = max(0, min(self.store.n_frames - 1, self.frame_idx + delta))
        self.render()

    def go_to(self, index: int) -> None:
        if self.store is None:
            return
        self.stop_playback()
        self.frame_idx = max(0, min(self.store.n_frames - 1, index))
        self.render()

    def on_scale(self, _value) -> None:
        if self._syncing_scale or self.store is None:
            return
        self.stop_playback()
        try:
            index = int(round(float(self.scale_var.get())))
        except (TypeError, ValueError):
            return
        self.frame_idx = max(0, min(self.store.n_frames - 1, index))
        self.render()

    def go_next(self) -> None:
        if self.index + 1 >= len(self.candidates):
            self._message("已经是最后一个视频。")
            return
        self.load_index(self.index + 1)

    def go_prev(self) -> None:
        if self.index <= 0:
            self._message("已经是第一个视频。")
            return
        if self.dirty and not messagebox.askyesno("未保存", "当前标注未保存，仍然返回上一个视频？"):
            return
        self.dirty = False
        self.load_index(self.index - 1)

    # -- marking ------------------------------------------------------------

    def set_start(self) -> None:
        if self.store is None:
            return
        self.stop_playback()
        self.start = self.frame_idx
        if self.end is not None and self.end < self.start:
            self.end = None
        if self.status == "no_fall":
            self.status = ""
        self.dirty = True
        self._message(f"开始帧 = {self.start}。按 e 设结束帧，然后按 1 确认跌倒。")
        self.render()

    def set_end(self) -> None:
        if self.store is None:
            return
        if self.start is None:
            self._message("请先按 s 设开始帧。")
            return
        self.stop_playback()
        self.end = self.frame_idx
        if self.end < self.start:
            self._message("结束帧不能小于开始帧。")
            self.end = None
            self.render()
            return
        if self.status == "no_fall":
            self.status = ""
        self.dirty = True
        self._message(f"结束帧 = {self.end}。按 1 确认跌倒。")
        self.render()

    def clear_interval(self) -> None:
        self.start = None
        self.end = None
        if self.status == "annotated":
            self.status = ""
        self.dirty = True
        self._message("已清除区间。")
        self.render()

    # -- segments (SEGMENTED_SCOPES only) -----------------------------------

    def _require_segmented(self) -> bool:
        """Guard for the segment keys; a no-op on the training scopes."""
        if self.segmented:
            return True
        self._message("当前 scope 只允许一个区间（训练层 schema 已被哈希冻结）。")
        return False

    def select_segment(self, index: int) -> None:
        if not self.segmented or self.store is None:
            return
        if not 0 <= index < len(self.segments):
            return
        self.stop_playback()
        self.segment_index = index
        seg = self._active
        # Jump to the segment start so the annotator can see what they are editing.
        if seg.start is not None:
            self.frame_idx = max(0, min(self.store.n_frames - 1, seg.start))
        self._message(f"当前编辑第 {index + 1} 段。")
        self.render()

    def add_segment(self) -> None:
        if not self._require_segmented():
            return
        if self.status == "no_fall":
            self.status = ""
        self.segments.append(Segment())
        self.segment_index = len(self.segments) - 1
        self.dirty = True
        self._message(f"已新增第 {len(self.segments)} 段：按 s / e 设起止。")
        self.render()

    def _refresh_segment_bar(self) -> None:
        """One clickable chip per segment, so the active one is unmistakable."""
        if not self.segmented:
            return
        # Rebuilding on every render would flicker during playback, so only
        # rebuild when the chip labels would actually differ.
        signature = (self.segment_index, tuple((s.complete, s.start, s.end) for s in self.segments))
        if signature == self._seg_signature:
            return
        self._seg_signature = signature
        for child in self.seg_frame.winfo_children():
            child.destroy()
        for i, seg in enumerate(self.segments):
            if i == self.segment_index:
                text = f"● 段{i + 1}"
            elif seg.complete:
                text = f"段{i + 1} ✓"
            else:
                text = f"段{i + 1} …"
            ttk.Button(
                self.seg_frame, text=text, width=max(6, len(text) + 1),
                command=lambda idx=i: self.select_segment(idx),
            ).pack(side=tk.LEFT, padx=2)

    def delete_segment(self) -> None:
        if not self._require_segmented():
            return
        if len(self.segments) <= 1:
            self.clear_interval()
            return
        removed = self.segment_index
        del self.segments[removed]
        self.segment_index = max(0, min(removed, len(self.segments) - 1))
        if self.status == "annotated" and not any(s.complete for s in self.segments):
            self.status = ""
        self.dirty = True
        self._message(f"已删除第 {removed + 1} 段，现有 {len(self.segments)} 段。")
        self.render()

    def mark(self, status: str) -> None:
        if self.store is None:
            return
        self.stop_playback()
        if status == "annotated":
            if self.start is None or self.end is None:
                self._message("标记跌倒前必须先设开始帧 (s) 和结束帧 (e)。")
                return
            if not (0 <= self.start <= self.end < self.store.n_frames):
                self._message(
                    f"区间非法：start={self.start}, end={self.end}, n_frames={self.store.n_frames}。"
                )
                return
            error = self._overlap_error()
            if error:
                self._message(error)
                return
        self.status = status
        self.dirty = True
        self._message(f"状态已设为：{STATUS_TEXT[status]}。按 Enter 保存并下一个。")
        self.render()

    # -- saving -------------------------------------------------------------

    def _overlap_error(self) -> str:
        """Segments must be disjoint and in ascending order; report the clash."""
        if not self.segmented:
            return ""
        ordered = sorted(self._completed_segments(), key=lambda s: s.start or 0)
        for prev, cur in zip(ordered, ordered[1:]):
            if cur.start <= prev.end:
                return f"区间重叠：第 [{prev.start},{prev.end}] 与 [{cur.start},{cur.end}] 段交叉。"
        return ""

    def _validate(self) -> tuple[bool, list[tuple[int | str, int | str]], str]:
        if self.store is None:
            return False, [], "视频未加载，无法保存。"
        if self.status not in DONE_STATUSES:
            return False, [], "尚未选择状态：按 1 / 0 / 9 之一。"
        n = self.store.n_frames
        if self.status != "annotated":
            # no_fall / unclear still write one row with empty frames: an
            # explicit verdict is what distinguishes "reviewed, no fall" from
            # "never looked at", so returning no spans would lose the verdict.
            return True, [("", "")], ""
        if not self.segmented:
            if not self._active.complete:
                return False, [], "跌倒状态需要开始帧和结束帧。"
            spans = [(self._active.start, self._active.end)]  # type: ignore[list-item]
            if not (0 <= spans[0][0] <= spans[0][1] < n):
                return False, [], (
                    f"区间非法：start={spans[0][0]}, end={spans[0][1]}, n_frames={n}。"
                )
            return True, spans, ""
        # Segmented scope: every segment must be complete, valid and disjoint.
        if not self.segments:
            return False, [], "跌倒状态至少需要一段。"
        for i, seg in enumerate(self.segments):
            if not seg.complete:
                return False, [], f"第 {i + 1} 段还没设起止（s / e）。"
            if not (0 <= seg.start <= seg.end < n):
                return False, [], (
                    f"第 {i + 1} 段区间非法：{seg.start}..{seg.end}，共 {n} 帧。"
                )
        error = self._overlap_error()
        if error:
            return False, [], error
        spans = sorted(((s.start, s.end) for s in self.segments), key=lambda t: t[0])
        return True, spans, ""

    def save_current(self, advance: bool) -> bool:
        ok, spans, error = self._validate()
        if not ok:
            self._message(error)
            return False

        cand = self.candidate
        assert self.store is not None
        now = datetime.now().isoformat(timespec="seconds")
        notes = self.notes_var.get().strip()
        if self.store.n_frames != cand.n_frames:
            note = f"decoded_frames={self.store.n_frames} manifest_frames={cand.n_frames}"
            notes = f"{notes} | {note}".strip(" |")

        # One row per interval. For non-segmented scopes ``spans`` holds at most
        # one entry, so the historical schema and row count are unchanged.
        segment_count = len(spans)
        new_rows: list[dict] = []
        for index, (start, end) in enumerate(spans):
            row = {
                "video_id": cand.video_id,
                "dataset": cand.dataset,
                "event_id": cand.event_id,
                "label": cand.label,
                "start_frame": start,
                "end_frame": end,
                "n_frames": self.store.n_frames,
                "fps": f"{self.store.fps:.6g}" if self.store.fps > 0 else "",
                "status": self.status,
                "annotator": self.annotator,
                "annotated_at": now,
                "notes": notes,
            }
            if self.segmented:
                row["segment_index"] = index
                row["segment_count"] = segment_count
            new_rows.append(row)

        self.rows[cand.video_id] = new_rows
        write_rows(self.output, self.rows, self.order, segmented=self.segmented)
        self.dirty = False
        if self.segmented and segment_count > 1:
            self._message(f"已保存：{cand.video_id} -> {self.status}（{segment_count} 段）")
        else:
            self._message(f"已保存：{cand.video_id} -> {self.status}")

        if advance:
            self.go_next()
        return True

    def save_and_next(self) -> None:
        if self.status == "":
            # Explicit skip: advance without writing a row.
            self.dirty = False
            self.go_next()
            return
        self.save_current(advance=True)

    # -- keys / close -------------------------------------------------------

    def on_key(self, event: tk.Event) -> None:
        if self._in_entry() and event.keysym not in ("Return", "Escape"):
            return
        key = event.keysym
        shift = bool(event.state & 0x0001)

        if key == "space":
            self.toggle_play()
        elif key == "Left":
            self.step(-10 if shift else -1)
        elif key == "Right":
            self.step(10 if shift else 1)
        elif key == "Home":
            self.go_to(0)
        elif key == "End":
            self.go_to(self._n_frames() - 1)
        elif key == "Prior":
            self.step(-int(self.store.fps) if self.store else -25)
        elif key == "Next":
            self.step(int(self.store.fps) if self.store else 25)
        elif key == "s":
            self.set_start()
        elif key == "e":
            self.set_end()
        elif key == "c":
            self.clear_interval()
        elif key in ("comma", "less"):
            if self.segmented:
                self.select_segment((self.segment_index - 1) % len(self.segments))
        elif key in ("period", "greater"):
            if self.segmented:
                self.select_segment((self.segment_index + 1) % len(self.segments))
        elif key == "n":
            self.add_segment()
        elif key == "d":
            self.delete_segment()
        elif key == "1":
            self.mark("annotated")
        elif key == "0":
            self.mark("no_fall")
        elif key == "9":
            self.mark("unclear")
        elif key == "Return":
            self.save_and_next()
        elif key == "BackSpace":
            self.go_prev()
        elif key == "f":
            if self.candidate.use_half == "right":
                self.show_half = not self.show_half
                self.render()
        elif key in ("plus", "equal", "KP_Add"):
            self.zoom = min(4, self.zoom + 1)
            self.render()
        elif key in ("minus", "KP_Subtract"):
            self.zoom = max(1, self.zoom - 1)
            self.render()
        elif key == "bracketleft":
            self.speed = max(0.25, self.speed / 2)
            self._message(f"播放速度 {self.speed:g}x")
        elif key == "bracketright":
            self.speed = min(8.0, self.speed * 2)
            self._message(f"播放速度 {self.speed:g}x")
        elif key == "Escape":
            self.stop_playback()

    def on_close(self) -> None:
        self.stop_playback()
        if self.dirty:
            answer = messagebox.askyesnocancel("退出", "当前标注未保存，是否保存后退出？")
            if answer is None:
                return
            if answer and not self.save_current(advance=False):
                return
        if self.store is not None:
            self.store.close()
            self.store = None
        self.root.destroy()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def status_of(rows: dict[str, list[dict]], video_id: str) -> str:
    """Saved verdict for a video ("" when never reviewed)."""
    saved = rows.get(video_id) or []
    if not saved:
        return ""
    return (saved[0].get("status") or "").strip()


def first_unreviewed(candidates: list[Candidate], rows: dict[str, dict]) -> int:
    for i, cand in enumerate(candidates):
        status = status_of(rows, cand.video_id)
        if status not in DONE_STATUSES:
            return i
    return 0


def find_start(candidates: list[Candidate], needle: str) -> int:
    for i, cand in enumerate(candidates):
        if cand.video_id == needle:
            return i
    matches = [i for i, cand in enumerate(candidates) if needle.lower() in cand.video_id.lower()]
    if len(matches) == 1:
        return matches[0]
    raise SystemExit(f"--start-at 无法唯一匹配：{needle!r}（命中 {len(matches)} 个）")


def print_list(candidates: list[Candidate], rows: dict[str, dict]) -> None:
    done = 0
    for i, cand in enumerate(candidates):
        status = status_of(rows, cand.video_id)
        if status in DONE_STATUSES:
            done += 1
        mark = "x" if status in DONE_STATUSES else " "
        print(
            f"[{mark}] {i + 1:3d}  {cand.video_id:<42}  {cand.label:<9} "
            f"{cand.annotation_status:<18}  {cand.n_frames:4d} frames  {cand.relative_path}"
        )
    print(f"\n候选 {len(candidates)} 个；已标注 {done} 个；待标注 {len(candidates) - done} 个。")


def run_self_test(candidates: list[Candidate], limit: int) -> int:
    failures = 0
    for cand in candidates[:limit]:
        try:
            store = FrameStore(cand.abs_path)
        except Exception as exc:  # noqa: BLE001
            print(f"[FAIL] {cand.video_id}: {exc}")
            failures += 1
            continue
        mismatch = "" if store.n_frames == cand.n_frames else f"  (manifest={cand.n_frames})"
        print(
            f"[OK] {cand.video_id}: frames={store.n_frames} fps={store.fps:g} "
            f"{store.width}x{store.height} cached={store.cached} half={cand.use_half}{mismatch}"
        )
        store.close()
    print(f"self-test: {min(limit, len(candidates)) - failures} ok, {failures} failed")
    return 0 if failures == 0 else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Frame-level fall interval annotation tool (writes a manual CSV layer)."
    )
    parser.add_argument(
        "--scope",
        choices=[*SCOPE_ORDER, "board", "all"],
        default="all",
        help="which videos to annotate (default: all, in priority order)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"output CSV (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="override the manifest CSV (default: datasets/manifests/videos.csv; "
             "the board scope defaults to datasets/manifests/board/videos.csv)",
    )
    parser.add_argument(
        "--annotator",
        default=os.environ.get("USERNAME") or getpass.getuser(),
        help="annotator name stored in each row",
    )
    parser.add_argument("--redo", action="store_true", help="start at the first candidate and allow overwrite")
    parser.add_argument("--start-at", default="", help="jump to a video_id (substring allowed if unique)")
    parser.add_argument("--list", action="store_true", help="list candidates and exit")
    parser.add_argument("--self-test", type=int, default=0, metavar="N", help="decode the first N videos and exit")
    return parser


def _force_utf8_stdout() -> None:
    """Keep Chinese diagnostics readable when stdout is piped or redirected."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


def default_manifest_for(scope: str) -> Path:
    """Board clips live outside the v1/v2/v3 derived layers ()."""
    if scope == "board":
        return MANIFESTS / "board" / "videos.csv"
    return MANIFESTS / "videos.csv"


def assert_output_isolated(scope: str, output: Path) -> None:
    """Refuse to write board annotations into a layer that feeds training."""
    if scope != "board":
        return
    resolved = output.resolve()
    for version in ("v1", "v2", "v3"):
        root = (ANNOTATIONS / version).resolve()
        if resolved == root or root in resolved.parents:
            raise SystemExit(
                f"refusing to write board annotations under {root}: "
                "board data is evaluation-only and must not enter a training layer (D-038)"
            )
    # A bare file directly in annotations/ is the v1 layout; also forbidden.
    if resolved.parent == ANNOTATIONS.resolve():
        raise SystemExit(
            f"refusing to write board annotations directly into {ANNOTATIONS}: "
            "use datasets/annotations/board/ (D-038)"
        )


def assert_schema_matches(scope: str, output: Path, segmented: bool) -> None:
    """Refuse to append segmented rows to a file written in the old schema."""
    if not output.exists():
        return
    existing = _has_segment_columns(output)
    if existing == segmented:
        return
    raise SystemExit(
        f"schema mismatch: {output} "
        f"{'has' if existing else 'lacks'} {SEGMENT_FIELDS} columns, but scope "
        f"{scope!r} {'writes' if segmented else 'does not write'} them. "
        "Use a different --output."
    )


def main(argv: list[str] | None = None) -> int:
    _force_utf8_stdout()
    args = build_parser().parse_args(argv)
    manifest = args.manifest or default_manifest_for(args.scope)
    segmented = args.scope in SEGMENTED_SCOPES
    assert_output_isolated(args.scope, args.output)
    assert_schema_matches(args.scope, args.output, segmented)
    candidates = load_candidates(args.scope, manifest)
    if not candidates:
        print("没有匹配的视频。")
        return 1

    rows = load_existing(args.output)
    if args.list:
        print(f"scope={args.scope}  output={args.output}")
        print_list(candidates, rows)
        return 0
    if args.self_test > 0:
        return run_self_test(candidates, args.self_test)

    if args.start_at:
        start_index = find_start(candidates, args.start_at)
    elif args.redo:
        start_index = 0
    else:
        start_index = first_unreviewed(candidates, rows)

    pending = len(candidates) - sum(
        1 for c in candidates if status_of(rows, c.video_id) in DONE_STATUSES
    )
    print(f"scope={args.scope}  候选={len(candidates)}  待标注={pending}  输出={args.output}")
    if pending == 0 and not args.redo:
        print("全部已标注；使用 --redo 可重新覆盖。")

    root = tk.Tk()
    AnnotatorApp(
        root=root,
        candidates=candidates,
        rows=rows,
        output=args.output,
        annotator=args.annotator,
        start_index=start_index,
        segmented=segmented,
    )
    root.mainloop()

    done = {status: 0 for status in DONE_STATUSES}
    for cand in candidates:
        status = status_of(rows, cand.video_id)
        if status in done:
            done[status] += 1
    remaining = len(candidates) - sum(done.values())
    print(
        f"本次结束：annotated={done['annotated']} no_fall={done['no_fall']} "
        f"unclear={done['unclear']} 未标注={remaining}  输出={args.output}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
