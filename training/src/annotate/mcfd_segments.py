"""MCFD fall-segment helpers shared by the GUI, propagation and QC scripts."""
from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from src.data import paths

DATASET = "mcfd"
LABEL = 1
EXPECTED_FPS = 120.0

MANUAL_FIELDS = [
    "video_id",
    "dataset",
    "event_id",
    "chute",
    "cam",
    "segment_index",
    "segment_count",
    "label",
    "start_frame",
    "end_frame",
    "anchor_start_frame",
    "offset_frames",
    "n_frames",
    "fps",
    "status",
    "source_video_id",
    "annotator",
    "annotated_at",
    "notes",
]

ANNOTATED = "annotated"
PROPAGATED = "propagated"
STATUSES = {ANNOTATED, PROPAGATED}


def video_id(chute: int, cam: int) -> str:
    return f"mcfd:chute{int(chute):02d}:cam{int(cam)}"


def event_id(chute: int) -> str:
    return f"mcfd:chute{int(chute):02d}"


def to_int(value, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def to_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _strict_int(row: dict, field: str) -> int:
    value = row.get(field)
    if isinstance(value, (int, float)):
        return int(value)
    text = str(value or "").strip()
    if text == "":
        raise ValueError(f"missing {field}: {row.get('video_id', '?')}")
    return int(float(text))


@dataclass(frozen=True)
class FallSegment:
    """One disjoint physical fall on one camera."""

    chute: int
    cam: int
    index: int
    anchor_start: int  # canonical 0-based start of the first raw label-1 row
    anchor_end: int  # canonical exclusive end of the merged raw rows
    rows: tuple[tuple[int, int], ...]  # canonical (start, end) of raw rows

    @property
    def video_id(self) -> str:
        return video_id(self.chute, self.cam)

    @property
    def anchor_frames(self) -> int:
        return self.anchor_end - self.anchor_start


def load_mcfd_rows(annotation_dir: Path | None = None) -> list[dict]:
    """Read the v2 MCFD annotation (the frozen layer)."""
    path = (Path(annotation_dir) if annotation_dir else paths.ANNOTATIONS / "v2") / "mcfd.csv"
    return paths.read_csv(path)


def merged_segments(
    rows: Sequence[dict],
) -> dict[tuple[int, int], list[FallSegment]]:
    """Merge label-1 rows per camera into disjoint fall segments."""
    by_video: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for row in rows:
        if (row.get("status") or "").strip() != "valid":
            continue
        if to_int(row.get("label")) != LABEL:
            continue
        chute = to_int(row.get("chute"))
        cam = to_int(row.get("cam"))
        start = to_int(row.get("start_frame"))
        end = to_int(row.get("end_frame"))
        if start <= 0 or end < start:
            continue
        by_video.setdefault((chute, cam), []).append((start - 1, end))

    out: dict[tuple[int, int], list[FallSegment]] = {}
    for (chute, cam), spans in sorted(by_video.items()):
        spans.sort()
        merged: list[list[int]] = []
        for start, end in spans:
            if merged and start <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([start, end])
        segments = []
        for index, (start, end) in enumerate(merged):
            covered = tuple(
                (max(start, s), min(end, e)) for s, e in spans if s < end and e > start
            )
            segments.append(
                FallSegment(
                    chute=chute,
                    cam=cam,
                    index=index,
                    anchor_start=start,
                    anchor_end=end,
                    rows=covered,
                )
            )
        out[(chute, cam)] = segments
    return out


def load_video_meta(manifest_dir: Path | None = None) -> dict[str, dict]:
    """MCFD video rows keyed by video_id (v2 manifest)."""
    root = Path(manifest_dir) if manifest_dir else paths.MANIFESTS / "v2"
    rows = paths.read_csv(root / "videos.csv")
    return {
        row["video_id"]: row
        for row in rows
        if row.get("dataset") == DATASET and row.get("annotation_status") == "annotated"
    }


def load_coverage(path: Path | None = None) -> dict[str, dict]:
    path = Path(path) if path else paths.KEYPOINTS_V2 / "interval_coverage.csv"
    return {
        row["video_id"]: row
        for row in paths.read_csv(path)
        if row.get("dataset") == DATASET
    }


def select_source_cameras(
    segments: dict[tuple[int, int], list[FallSegment]],
    coverage: dict[str, dict],
    meta: dict[str, dict],
) -> dict[int, int]:
    """Pick one camera per chute: max coverage, then covered, cacheable, low cam."""
    by_chute: dict[int, list[int]] = {}
    for chute, cam in segments:
        by_chute.setdefault(chute, []).append(cam)

    chosen: dict[int, int] = {}
    for chute, cams in sorted(by_chute.items()):

        def rank(cam: int):
            vid = video_id(chute, cam)
            cov = coverage.get(vid, {})
            n_frames = to_int(meta.get(vid, {}).get("n_frames"))
            return (
                to_float(cov.get("interval_coverage")),
                to_int(cov.get("interval_covered")),
                1 if 0 < n_frames <= 4000 else 0,
                -cam,
            )

        chosen[chute] = max(cams, key=rank)
    return chosen


def load_manual(path: Path | None = None) -> list[dict]:
    path = Path(path) if path else paths.MCFD_FALL_INTERVALS
    if not path.exists():
        return []
    return paths.read_csv(path)


def write_manual_atomic(path: Path, rows: Iterable[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(
        rows,
        key=lambda r: (
            to_int(r.get("chute")),
            to_int(r.get("cam")),
            to_int(r.get("segment_index")),
        ),
    )
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=MANUAL_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in ordered:
            writer.writerow(row)
    os.replace(tmp, path)


def key_of(row: dict) -> tuple[int, int, int]:
    return (
        to_int(row.get("chute")),
        to_int(row.get("cam")),
        to_int(row.get("segment_index")),
    )


def group_manual(rows: Iterable[dict]) -> dict[tuple[int, int], list[dict]]:
    grouped: dict[tuple[int, int], list[dict]] = {}
    for row in rows:
        grouped.setdefault((to_int(row["chute"]), to_int(row["cam"])), []).append(row)
    for items in grouped.values():
        items.sort(key=lambda r: to_int(r.get("segment_index")))
    return grouped


def rows_for_camera(rows: Iterable[dict], chute: int, cam: int) -> list[dict]:
    return [r for r in rows if to_int(r.get("chute")) == chute and to_int(r.get("cam")) == cam]


def is_camera_complete(
    rows: Iterable[dict], segments: dict[tuple[int, int], list[FallSegment]], chute: int, cam: int
) -> bool:
    expected = segments.get((chute, cam), [])
    if not expected:
        return False
    have = rows_for_camera(rows, chute, cam)
    indexes = {to_int(r.get("segment_index")) for r in have}
    statuses = {(r.get("status") or "").strip() for r in have}
    return indexes == set(range(len(expected))) and statuses == {ANNOTATED}


def validate_manual(
    rows: Sequence[dict],
    segments: dict[tuple[int, int], list[FallSegment]],
    meta: dict[str, dict],
    require_complete: bool = False,
) -> dict:
    """Validate the manual layer and return summary counts."""
    errors: list[str] = []
    seen: dict[tuple[int, int, int], dict] = {}
    by_video: dict[tuple[int, int], list[dict]] = {}

    for row in rows:
        try:
            k = key_of(row)
        except Exception:  # noqa: BLE001
            errors.append(f"unparsable row: {row}")
            continue
        if k in seen:
            errors.append(f"duplicate key {k}")
            continue
        seen[k] = row
        by_video.setdefault((k[0], k[1]), []).append(row)

        chute, cam, index = k
        vid = (row.get("video_id") or "").strip()
        if vid != video_id(chute, cam):
            errors.append(f"{k}: video_id {vid!r} != {video_id(chute, cam)}")
        if (row.get("dataset") or "").strip() != DATASET:
            errors.append(f"{k}: dataset must be {DATASET}")
        if (row.get("event_id") or "").strip() != event_id(chute):
            errors.append(f"{k}: event_id must be {event_id(chute)}")
        if to_int(row.get("label")) != LABEL:
            errors.append(f"{k}: label must be 1")
        status = (row.get("status") or "").strip()
        if status not in STATUSES:
            errors.append(f"{k}: status {status!r} not in {sorted(STATUSES)}")

        expected = segments.get((chute, cam))
        if not expected:
            errors.append(f"{k}: camera has no v2 fall segment")
            continue
        if to_int(row.get("segment_count")) != len(expected):
            errors.append(
                f"{k}: segment_count {row.get('segment_count')} != {len(expected)}"
            )
        if not 0 <= index < len(expected):
            errors.append(f"{k}: segment_index out of range")
            continue
        seg = expected[index]
        if to_int(row.get("anchor_start_frame")) != seg.anchor_start:
            errors.append(
                f"{k}: anchor_start_frame {row.get('anchor_start_frame')} != {seg.anchor_start}"
            )

        try:
            start = _strict_int(row, "start_frame")
            end = _strict_int(row, "end_frame")
            n_frames = _strict_int(row, "n_frames")
            offset = _strict_int(row, "offset_frames")
        except ValueError as exc:
            errors.append(f"{k}: {exc}")
            continue
        manifest = meta.get(video_id(chute, cam))
        if manifest is None:
            errors.append(f"{k}: video missing from v2 manifest")
        else:
            if n_frames != to_int(manifest.get("n_frames")):
                errors.append(f"{k}: n_frames {n_frames} != manifest {manifest.get('n_frames')}")
        if not (0 <= start <= end < n_frames):
            errors.append(f"{k}: interval {start}..{end} outside 0..{n_frames - 1}")
        if offset != start - seg.anchor_start:
            errors.append(
                f"{k}: offset_frames {offset} != start-anchor {start - seg.anchor_start}"
            )
        fps = to_float(row.get("fps"))
        if abs(fps - EXPECTED_FPS) > 1e-6:
            errors.append(f"{k}: fps {fps:g} != {EXPECTED_FPS:g}")

        source = (row.get("source_video_id") or "").strip()
        if status == ANNOTATED and source:
            errors.append(f"{k}: annotated row must not name source_video_id")
        if status == PROPAGATED:
            if not source:
                errors.append(f"{k}: propagated row needs source_video_id")
            elif not source.startswith(event_id(chute) + ":cam"):
                errors.append(f"{k}: source {source} is not in {event_id(chute)}")
            elif source == vid:
                errors.append(f"{k}: propagated row cannot be its own source")

    for (chute, cam), items in by_video.items():
        expected = segments.get((chute, cam), [])
        indexes = sorted(to_int(r.get("segment_index")) for r in items)
        if indexes != list(range(len(expected))):
            errors.append(
                f"{video_id(chute, cam)}: indexes {indexes} != 0..{len(expected) - 1}"
            )
        ordered = sorted(items, key=lambda r: to_int(r.get("segment_index")))
        for prev, cur in zip(ordered, ordered[1:]):
            if to_int(cur.get("start_frame")) <= to_int(prev.get("end_frame")):
                errors.append(
                    f"{video_id(chute, cam)}: segments overlap at "
                    f"{prev.get('segment_index')}/{cur.get('segment_index')}"
                )

    if require_complete:
        for (chute, cam), segs in sorted(segments.items()):
            items = by_video.get((chute, cam), [])
            if len(items) != len(segs):
                errors.append(
                    f"{video_id(chute, cam)}: {len(items)} rows != {len(segs)} segments"
                )

    if errors:
        raise ValueError("manual layer validation failed:\n  " + "\n  ".join(errors))

    status_counts = {ANNOTATED: 0, PROPAGATED: 0}
    for row in rows:
        status_counts[(row.get("status") or "").strip()] += 1
    source_cameras = {
        (to_int(r["chute"]), to_int(r["cam"]))
        for r in rows
        if (r.get("status") or "").strip() == ANNOTATED
    }
    return {
        "rows": len(rows),
        "videos": len(by_video),
        "segments": sum(len(v) for v in segments.values()),
        "annotated_rows": status_counts[ANNOTATED],
        "propagated_rows": status_counts[PROPAGATED],
        "source_cameras": sorted(source_cameras),
    }


def propagate_rows(
    manual_rows: Sequence[dict],
    segments: dict[tuple[int, int], list[FallSegment]],
    meta: dict[str, dict],
) -> list[dict]:
    """Propagate hand-marked intervals to the other seven cameras of a chute."""
    annotated = [r for r in manual_rows if (r.get("status") or "").strip() == ANNOTATED]
    by_chute: dict[int, list[dict]] = {}
    for row in annotated:
        by_chute.setdefault(to_int(row.get("chute")), []).append(row)

    out: list[dict] = []
    for chute, src_rows in sorted(by_chute.items()):
        src_cams = {to_int(r.get("cam")) for r in src_rows}
        if len(src_cams) != 1:
            raise ValueError(f"chute{chute:02d}: expected exactly one source camera, got {sorted(src_cams)}")
        src_cam = next(iter(src_cams))
        src_vid = video_id(chute, src_cam)
        src_segments = segments.get((chute, src_cam), [])
        src_by_index = {to_int(r.get("segment_index")): r for r in src_rows}
        if set(src_by_index) != set(range(len(src_segments))):
            raise ValueError(
                f"{src_vid}: source rows {sorted(src_by_index)} != segments 0..{len(src_segments) - 1}"
            )

        target_cams = sorted(cam for c, cam in segments if c == chute)
        for cam in target_cams:
            if cam == src_cam:
                continue
            tgt_vid = video_id(chute, cam)
            tgt_segments = segments[(chute, cam)]
            if len(tgt_segments) != len(src_segments):
                raise ValueError(
                    f"chute{chute:02d}: {src_vid} has {len(src_segments)} segments, "
                    f"{tgt_vid} has {len(tgt_segments)}"
                )
            manifest = meta.get(tgt_vid)
            if manifest is None:
                raise ValueError(f"{tgt_vid}: missing from v2 manifest")
            n_frames = to_int(manifest.get("n_frames"))
            fps = to_float(manifest.get("fps"))
            if abs(fps - EXPECTED_FPS) > 1e-6:
                raise ValueError(f"{tgt_vid}: fps {fps:g} != {EXPECTED_FPS:g}")

            for index, tgt_seg in enumerate(tgt_segments):
                src = src_by_index[index]
                src_seg = src_segments[index]
                start = _strict_int(src, "start_frame")
                end = _strict_int(src, "end_frame")
                duration = end - start
                offset = start - src_seg.anchor_start
                dst_start = tgt_seg.anchor_start + offset
                dst_end = dst_start + duration
                clamped = []
                if dst_start < 0:
                    clamped.append(f"start {dst_start}->0")
                    dst_start = 0
                if dst_end > n_frames - 1:
                    clamped.append(f"end {dst_end}->{n_frames - 1}")
                    dst_end = n_frames - 1
                if dst_end < dst_start:
                    raise ValueError(
                        f"{tgt_vid} seg{index}: clamp produced {dst_start}..{dst_end}"
                    )
                notes = (
                    f"propagated from {src_vid} seg{index}; "
                    f"anchor={tgt_seg.anchor_start}; offset={offset}"
                )
                if clamped:
                    notes += "; clamped " + ", ".join(clamped)
                out.append(
                    {
                        "video_id": tgt_vid,
                        "dataset": DATASET,
                        "event_id": event_id(chute),
                        "chute": chute,
                        "cam": cam,
                        "segment_index": index,
                        "segment_count": len(tgt_segments),
                        "label": LABEL,
                        "start_frame": dst_start,
                        "end_frame": dst_end,
                        "anchor_start_frame": tgt_seg.anchor_start,
                        "offset_frames": dst_start - tgt_seg.anchor_start,
                        "n_frames": n_frames,
                        "fps": manifest.get("fps", ""),
                        "status": PROPAGATED,
                        "source_video_id": src_vid,
                        "annotator": (src.get("annotator") or "").strip(),
                        "annotated_at": (src.get("annotated_at") or "").strip(),
                        "notes": notes,
                    }
                )

    out.sort(key=lambda r: (to_int(r["chute"]), to_int(r["cam"]), to_int(r["segment_index"])))
    return out


def merge_rows(
    existing: Sequence[dict],
    propagated: Sequence[dict],
    chutes: set[int] | None = None,
) -> list[dict]:
    """Keep hand-marked rows, replace propagated rows for the given chutes."""
    target = {to_int(r["chute"]) for r in propagated} if chutes is None else set(chutes)
    kept = []
    for row in existing:
        status = (row.get("status") or "").strip()
        chute = to_int(row.get("chute"))
        if status == ANNOTATED:
            kept.append(row)
        elif status == PROPAGATED and chute not in target:
            kept.append(row)
    merged = kept + list(propagated)
    merged.sort(key=lambda r: (to_int(r["chute"]), to_int(r["cam"]), to_int(r["segment_index"])))
    return merged
