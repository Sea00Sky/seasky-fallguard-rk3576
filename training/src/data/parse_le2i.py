"""Normalize Le2i annotations."""
from __future__ import annotations

import re
from pathlib import Path

from . import paths

SCENE_RE = re.compile(r"^(Coffee_room_0[12]|Home_0[12]|Lecture_room|Office)$")


def _scene_of(rel_parts: tuple[str, ...]) -> str:
    for part in rel_parts:
        if SCENE_RE.match(part):
            return part
    return rel_parts[1] if len(rel_parts) > 1 else "unknown"


def _video_for_annotation(txt: Path) -> Path:
    parts = list(txt.parts)
    for i, part in enumerate(parts):
        if part in ("Annotation_files", "Annotations_files", "Annotations_files"):
            parts[i] = "Videos"
            break
    return Path(*parts).with_suffix(".avi")


def _first_two_ints(txt: Path) -> tuple[int | None, int | None, int]:
    lines = [ln.strip() for ln in txt.read_text(encoding="utf-8", errors="replace").splitlines() if ln.strip()]
    n = len(lines)
    if n < 2:
        return None, None, n
    try:
        return int(float(lines[0])), int(float(lines[1])), n
    except ValueError:
        return None, None, n


def parse() -> list[dict]:
    root = paths.RAW_LE2I
    if not root.exists():
        raise FileNotFoundError(root)

    rows: list[dict] = []
    annotated_videos: set[str] = set()

    for txt in sorted(paths.iter_files(root, {".txt"})):
        if txt.name.lower() == "readme.txt":
            continue
        video = _video_for_annotation(txt)
        rel_video = video.relative_to(root)
        scene = _scene_of(rel_video.parts)
        stem = video.stem
        start, end, n_lines = _first_two_ints(txt)
        notes = ""

        if start is None or end is None:
            status = "missing_header"
            notes = "first two non-empty lines are not integer frame numbers"
        elif start >= end:
            status = "invalid_interval"
            notes = f"start={start} >= end={end}"
        elif not video.exists():
            status = "missing_video"
            notes = f"expected video not found: {rel_video.as_posix()}"
        else:
            info = paths.video_info(video)
            if not info["readable"]:
                status = "unreadable_video"
                notes = info["error"]
            elif end > info["n_frames"]:
                status = "end_out_of_range"
                notes = f"end={end} > n_frames={info['n_frames']}"
            else:
                status = "valid"

        if video.exists():
            annotated_videos.add(video.resolve().as_posix())

        rows.append(
            {
                "video_id": f"le2i:{scene}:{stem}",
                "scene": scene,
                "video_name": video.name,
                "video_path": rel_video.as_posix(),
                "annotation_path": txt.relative_to(root).as_posix(),
                "start_frame": "" if start is None else start,
                "end_frame": "" if end is None else end,
                "interval_frames": "" if None in (start, end) else end - start,
                "status": status,
                "n_annotation_lines": n_lines,
                "notes": notes,
            }
        )

    # Videos without any annotation file: unannotated, not negative.
    for video in sorted(paths.iter_files(root, {".avi"})):
        if video.resolve().as_posix() in annotated_videos:
            continue
        rel = video.relative_to(root)
        scene = _scene_of(rel.parts)
        rows.append(
            {
                "video_id": f"le2i:{scene}:{video.stem}",
                "scene": scene,
                "video_name": video.name,
                "video_path": rel.as_posix(),
                "annotation_path": "",
                "start_frame": "",
                "end_frame": "",
                "interval_frames": "",
                "status": "unannotated",
                "n_annotation_lines": 0,
                "notes": "no annotation file for this scene",
            }
        )

    rows.sort(key=lambda r: (r["scene"], r["video_name"]))
    return rows
