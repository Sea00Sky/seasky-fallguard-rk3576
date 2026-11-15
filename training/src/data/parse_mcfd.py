"""Normalize MCFD annotations."""
from __future__ import annotations

import csv
from collections import Counter

from . import paths


def _cam_path(chute: int, cam: int) -> Path:
    # Raw copy preserves the source nesting: mcfd/dataset/dataset/chuteNN/camN.avi
    return paths.RAW_MCFD / "dataset" / f"chute{chute:02d}" / f"cam{cam}.avi"


def parse() -> list[dict]:
    csv_path = paths.RAW_MCFD_CSV
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    rows: list[dict] = []
    with open(csv_path, "r", newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        for raw in reader:
            try:
                chute = int(float(raw["chute"]))
                cam = int(float(raw["cam"]))
                start = int(float(raw["start"]))
                end = int(float(raw["end"]))
                label = int(float(raw["label"]))
            except (KeyError, TypeError, ValueError):
                continue
            video = _cam_path(chute, cam)
            exists = video.exists()
            info = paths.video_info(video) if exists else {"readable": False, "n_frames": 0, "fps": 0.0}
            notes = ""
            status = "valid"
            if not exists:
                status = "no_video"
                notes = "annotation row has no corresponding video file"
            elif not info["readable"]:
                status = "unreadable_video"
                notes = "video could not be opened"
            elif end > info["n_frames"]:
                status = "end_out_of_range"
                notes = f"end={end} > n_frames={info['n_frames']}"

            rows.append(
                {
                    "dataset": "mcfd",
                    "chute": chute,
                    "cam": cam,
                    "start_frame": start,
                    "end_frame": end,
                    "interval_frames": end - start,
                    "label": label,
                    "status": status,
                    "fps": info["fps"],
                    "n_frames": info["n_frames"],
                    "video_path": video.relative_to(paths.RAW).as_posix() if exists else "",
                    "notes": notes,
                }
            )

    # chute24: videos exist but no annotation rows.
    chute24 = paths.RAW_MCFD / "dataset" / "chute24"
    if chute24.exists():
        for video in sorted(chute24.glob("cam*.avi")):
            try:
                cam = int(video.stem.replace("cam", ""))
            except ValueError:
                continue
            info = paths.video_info(video)
            rows.append(
                {
                    "dataset": "mcfd",
                    "chute": 24,
                    "cam": cam,
                    "start_frame": "",
                    "end_frame": "",
                    "interval_frames": "",
                    "label": "",
                    "status": "chute24_unannotated",
                    "fps": info["fps"],
                    "n_frames": info["n_frames"],
                    "video_path": video.relative_to(paths.RAW).as_posix(),
                    "notes": "chute24 has no annotation rows",
                }
            )

    rows.sort(key=lambda r: (r["chute"], r["cam"], r["start_frame"] if r["start_frame"] != "" else 10**9))
    return rows


def stats(rows: list[dict]) -> dict:
    annotated = [r for r in rows if r["status"] in ("valid", "end_out_of_range", "unreadable_video")]
    return {
        "rows_total": len(rows),
        "annotated_rows": len(annotated),
        "label_counts": dict(Counter(r["label"] for r in annotated)),
        "interval_counts": dict(Counter(r["interval_frames"] for r in annotated)),
        "status_counts": dict(Counter(r["status"] for r in rows)),
        "chutes": sorted({r["chute"] for r in rows}),
        "cams": sorted({r["cam"] for r in rows}),
    }
