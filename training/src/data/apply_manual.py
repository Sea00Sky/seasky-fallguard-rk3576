"""Merge the manual interval layer into a versioned normalized annotation layer."""
from __future__ import annotations

from collections import Counter
from pathlib import Path

from . import paths

STATUS_ANNOTATED = "manual_annotated"
STATUS_NO_FALL = "manual_no_fall"

AUDIT_FIELDS = [
    "video_id", "dataset", "event_id", "source_status", "manual_status",
    "new_status", "label_before", "label_after", "start_frame", "end_frame",
    "n_frames", "action", "notes",
]


def _as_int(value: str, field: str, video_id: str) -> int:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        raise ValueError(f"{video_id}: {field} is not an integer: {value!r}") from None


def _check_interval(video_id: str, start: int, end: int, n_frames: int) -> None:
    if n_frames <= 0:
        raise ValueError(f"{video_id}: n_frames is not positive: {n_frames}")
    if not (0 <= start <= end < n_frames):
        raise ValueError(
            f"{video_id}: interval [{start}, {end}] outside [0, {n_frames - 1}]"
        )


def _le2i_label(status: str) -> str:
    if status in ("valid", STATUS_ANNOTATED):
        return "fall"
    if status == STATUS_NO_FALL:
        return "non_fall"
    return "unlabeled"


def apply(
    source_dir: Path | None = None,
    out_dir: Path | None = None,
    manual_path: Path | None = None,
    videos_manifest: Path | None = None,
) -> dict:
    source_dir = Path(source_dir) if source_dir else paths.ANNOTATIONS
    out_dir = Path(out_dir) if out_dir else paths.ANNOTATIONS / "v2"
    manual_path = Path(manual_path) if manual_path else paths.ANNOTATIONS / "manual" / "interval_annotations.csv"
    videos_manifest = Path(videos_manifest) if videos_manifest else paths.MANIFESTS / "videos.csv"

    manual = paths.read_csv(manual_path)
    manual_by_id: dict[str, dict] = {}
    for row in manual:
        vid = row["video_id"]
        if vid in manual_by_id:
            raise ValueError(f"duplicate manual row: {vid}")
        if row["status"] not in ("annotated", "no_fall"):
            raise ValueError(f"{vid}: unexpected manual status {row['status']!r}")
        manual_by_id[vid] = row

    frames_by_video = {
        row["video_id"]: _as_int(row["n_frames"], "n_frames", row["video_id"])
        for row in paths.read_csv(videos_manifest)
    }

    le2i = paths.read_csv(source_dir / "le2i.csv")
    urfd = paths.read_csv(source_dir / "urfd.csv")
    mcfd = paths.read_csv(source_dir / "mcfd.csv")

    audit: list[dict] = []
    consumed: set[str] = set()

    # ---------- Le2i ----------
    for row in le2i:
        vid = row["video_id"]
        manual_row = manual_by_id.get(vid)
        if manual_row is None:
            continue
        consumed.add(vid)

        source_status = row["status"]
        if source_status == "valid":
            raise ValueError(f"{vid}: manual row would overwrite a raw valid annotation")

        n_frames = frames_by_video.get(vid) or _as_int(
            manual_row["n_frames"], "n_frames", vid
        )
        if str(n_frames) != manual_row["n_frames"].strip():
            raise ValueError(
                f"{vid}: n_frames mismatch manifest={n_frames} manual={manual_row['n_frames']}"
            )

        label_before = _le2i_label(source_status)
        if manual_row["status"] == "annotated":
            start = _as_int(manual_row["start_frame"], "start_frame", vid)
            end = _as_int(manual_row["end_frame"], "end_frame", vid)
            _check_interval(vid, start, end, n_frames)
            row["start_frame"] = str(start)
            row["end_frame"] = str(end)
            row["interval_frames"] = str(end - start)
            row["status"] = STATUS_ANNOTATED
            row["notes"] = f"manual_annotated; previous_status={source_status}"
            action = "set_fall_interval"
        else:
            row["start_frame"] = ""
            row["end_frame"] = ""
            row["interval_frames"] = ""
            row["status"] = STATUS_NO_FALL
            row["notes"] = f"manual_no_fall; previous_status={source_status}"
            action = "set_non_fall"

        audit.append({
            "video_id": vid,
            "dataset": "le2i",
            "event_id": row["video_id"],
            "source_status": source_status,
            "manual_status": manual_row["status"],
            "new_status": row["status"],
            "label_before": label_before,
            "label_after": _le2i_label(row["status"]),
            "start_frame": row["start_frame"],
            "end_frame": row["end_frame"],
            "n_frames": n_frames,
            "action": action,
            "notes": manual_row.get("notes", ""),
        })

    # ---------- URFD ----------
    for row in urfd:
        vid = row["video_id"]
        manual_row = manual_by_id.get(vid)
        if manual_row is None:
            continue
        consumed.add(vid)

        source_status = row["status"]
        if source_status != "needs_annotation":
            raise ValueError(f"{vid}: manual row on unexpected status {source_status!r}")
        if manual_row["status"] != "annotated":
            raise ValueError(f"{vid}: URFD fall must be manually annotated, got {manual_row['status']!r}")

        n_frames = _as_int(row["n_frames"], "n_frames", vid)
        if str(n_frames) != manual_row["n_frames"].strip():
            raise ValueError(
                f"{vid}: n_frames mismatch urfd.csv={n_frames} manual={manual_row['n_frames']}"
            )
        start = _as_int(manual_row["start_frame"], "start_frame", vid)
        end = _as_int(manual_row["end_frame"], "end_frame", vid)
        _check_interval(vid, start, end, n_frames)

        row["fall_start_frame"] = str(start)
        row["fall_end_frame"] = str(end)
        row["status"] = STATUS_ANNOTATED
        row["notes"] = f"manual_annotated; previous_status={source_status}"

        audit.append({
            "video_id": vid,
            "dataset": "urfd",
            "event_id": row["video_id"],
            "source_status": source_status,
            "manual_status": manual_row["status"],
            "new_status": row["status"],
            "label_before": row["label"],
            "label_after": row["label"],
            "start_frame": row["fall_start_frame"],
            "end_frame": row["fall_end_frame"],
            "n_frames": n_frames,
            "action": "set_fall_interval",
            "notes": manual_row.get("notes", ""),
        })

    unmatched = sorted(set(manual_by_id) - consumed)
    if unmatched:
        raise ValueError(f"manual rows not matched to le2i/urfd: {unmatched}")

    # ---------- post-conditions ----------
    for row in le2i:
        if row["status"] == STATUS_ANNOTATED and not row["start_frame"]:
            raise ValueError(f"{row['video_id']}: manual_annotated without interval")
        if row["status"] == STATUS_NO_FALL and row["start_frame"]:
            raise ValueError(f"{row['video_id']}: manual_no_fall carries an interval")
    for row in urfd:
        if row["status"] == "duplicate_conv":
            continue
        if row["label"] == "fall" and row["status"] != STATUS_ANNOTATED:
            raise ValueError(f"{row['video_id']}: fall row not manually annotated")
        if row["label"] != "fall" and row["fall_start_frame"]:
            raise ValueError(f"{row['video_id']}: non-fall row carries an interval")

    paths.write_csv(out_dir / "le2i.csv", le2i)
    paths.write_csv(out_dir / "urfd.csv", urfd)
    paths.write_csv(out_dir / "mcfd.csv", mcfd)
    paths.write_csv(out_dir / "merge_audit.csv", audit, fieldnames=AUDIT_FIELDS)

    return {
        "manual_rows": len(manual),
        "matched": len(consumed),
        "unmatched": unmatched,
        "audit_rows": len(audit),
        "actions": dict(Counter(r["action"] for r in audit)),
        "le2i_status_counts": dict(Counter(r["status"] for r in le2i)),
        "urfd_status_counts": dict(Counter(r["status"] for r in urfd)),
        "mcfd_rows": len(mcfd),
        "outputs": [str(out_dir / name) for name in
                    ("le2i.csv", "urfd.csv", "mcfd.csv", "merge_audit.csv")],
    }


if __name__ == "__main__":
    import json

    print(json.dumps(apply(), ensure_ascii=False, indent=2))
