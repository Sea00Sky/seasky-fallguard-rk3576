"""Build videos.csv and events.csv from the normalized annotations."""
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

from . import paths

# Statuses that carry a supervised binary label. Raw Le2i `valid` is the
# original annotation; manual_* rows come from the manual interval layer.
# MCFD uses its own set because a v3 row can be a manual fall interval while
# the rest of the chute is still the raw `valid` export.
FALL_STATUSES = {"valid", "manual_annotated"}
NON_FALL_STATUSES = {"manual_no_fall"}
MCFD_ANNOTATED_STATUSES = {"valid", "manual_annotated"}
ELIGIBLE_LABELS = {"fall", "non_fall"}


def _probe(rel_path: str) -> dict:
    if not rel_path:
        return {"fps": "", "n_frames": "", "width": "", "height": "", "readable": False}
    video = paths.RAW / rel_path
    info = paths.video_info(video)
    return {
        "fps": info["fps"],
        "n_frames": info["n_frames"],
        "width": info["width"],
        "height": info["height"],
        "readable": info["readable"],
        "path_exists": video.exists(),
    }


def _video_row(video_id: str, dataset: str, event_id: str, rel_path: str,
               label: str, annotation_status: str, use_half: str = "", notes: str = "") -> dict:
    info = _probe(rel_path)
    return {
        "video_id": video_id,
        "dataset": dataset,
        "event_id": event_id,
        "relative_path": rel_path,
        "fps": info["fps"],
        "n_frames": info["n_frames"],
        "width": info["width"],
        "height": info["height"],
        "readable": info["readable"],
        "label": label,
        "annotation_status": annotation_status,
        "use_half": use_half,
        "split": "",
        "notes": notes,
    }


def _resolve_duplicates(videos: list[dict], events: list[dict]) -> list[dict]:
    """Find identical video content and keep only one event per duplicate group."""
    hash_path = paths.MANIFESTS / "raw_sha256.csv"
    if not hash_path.exists():
        raise FileNotFoundError(
            f"{hash_path} missing; run scripts/01_copy_raw.py first"
        )
    hashes = {r["relative_path"]: r["sha256"] for r in paths.read_csv(hash_path)}
    by_hash: dict[str, list[dict]] = defaultdict(list)
    for video in videos:
        digest = hashes.get(video["relative_path"])
        if digest:
            by_hash[digest].append(video)

    event_by_id = {e["event_id"]: e for e in events}
    groups = []
    for digest, group in sorted(by_hash.items()):
        if len(group) < 2:
            continue
        event_ids = sorted({v["event_id"] for v in group})
        if len(event_ids) == 1:
            continue  # same event, e.g. multiple MCFD cameras
        eligible_ids = [
            eid for eid in event_ids
            if event_by_id[eid]["eligible_for_split"] in (True, "True")
        ]
        canonical = eligible_ids[0] if eligible_ids else event_ids[0]
        excluded = [eid for eid in event_ids if eid != canonical]
        for eid in excluded:
            ev = event_by_id[eid]
            note = f"duplicate_content of {canonical}"
            ev["notes"] = f"{ev['notes']} | {note}" if ev["notes"] else note
            if ev["eligible_for_split"] in (True, "True"):
                ev["eligible_for_split"] = False
        for video in group:
            if video["event_id"] in excluded:
                note = f"duplicate_content of {canonical}"
                video["annotation_status"] = "duplicate_content"
                video["notes"] = f"{video['notes']} | {note}" if video["notes"] else note
        groups.append(
            {
                "sha256": digest,
                "canonical_event": canonical,
                "excluded_events": excluded,
                "files": [v["relative_path"] for v in group],
            }
        )
    return groups


def build(annotation_dir: Path | None = None) -> tuple[list[dict], list[dict], list[dict]]:
    annotation_dir = Path(annotation_dir) if annotation_dir else paths.ANNOTATIONS
    le2i = paths.read_csv(annotation_dir / "le2i.csv")
    mcfd = paths.read_csv(annotation_dir / "mcfd.csv")
    urfd = paths.read_csv(annotation_dir / "urfd.csv")

    videos: list[dict] = []
    events: list[dict] = []

    # ---------- Le2i: one video = one event ----------
    for row in le2i:
        status = row["status"]
        if status in FALL_STATUSES:
            label = "fall"
        elif status in NON_FALL_STATUSES:
            label = "non_fall"
        else:
            label = "unlabeled"
        rel = f"le2i/archive/{row['video_path']}"
        videos.append(
            _video_row(row["video_id"], "le2i", row["video_id"], rel, label, status,
                       notes=row.get("notes", ""))
        )
        events.append(
            {
                "event_id": row["video_id"],
                "dataset": "le2i",
                "group": row["scene"],
                "label": label,
                "n_videos": 1,
                "n_fall_videos": 1 if label == "fall" else 0,
                "annotation_status": status,
                "eligible_for_split": label in ELIGIBLE_LABELS,
                "split": "",
                "notes": row.get("notes", ""),
            }
        )

    # ---------- MCFD: one chute = one event ----------
    by_chute: dict[int, list[dict]] = defaultdict(list)
    for row in mcfd:
        by_chute[int(row["chute"])].append(row)

    for chute in sorted(by_chute):
        rows = by_chute[chute]
        annotated = [r for r in rows if r["status"] in MCFD_ANNOTATED_STATUSES]
        has_fall = any(int(r["label"]) == 1 for r in annotated)
        if chute == 24:
            event_label, ann_status, eligible = "unlabeled", "chute24_unannotated", False
        elif annotated:
            event_label = "fall" if has_fall else "non_fall"
            ann_status, eligible = "annotated", True
        else:
            event_label, ann_status, eligible = "unlabeled", "no_valid_rows", False

        event_id = f"mcfd:chute{chute:02d}"
        cams_with_rows = {int(r["cam"]) for r in annotated}
        for cam in range(1, 9):
            cam_rows = [r for r in annotated if int(r["cam"]) == cam]
            if cam_rows:
                cam_label = "fall" if any(int(r["label"]) == 1 for r in cam_rows) else "non_fall"
            elif chute == 24:
                cam_label = "unlabeled"
            else:
                cam_label = "unlabeled"
            rel = f"mcfd/dataset/dataset/chute{chute:02d}/cam{cam}.avi"
            videos.append(
                _video_row(
                    f"{event_id}:cam{cam}",
                    "mcfd",
                    event_id,
                    rel,
                    cam_label,
                    ann_status if cam in cams_with_rows or chute == 24 else "no_annotation_for_cam",
                    notes="" if cam in cams_with_rows else "camera has no annotation row",
                )
            )
        events.append(
            {
                "event_id": event_id,
                "dataset": "mcfd",
                "group": f"chute{chute:02d}",
                "label": event_label,
                "n_videos": 8,
                "n_fall_videos": sum(1 for r in annotated if int(r["label"]) == 1),
                "annotation_status": ann_status,
                "eligible_for_split": eligible,
                "split": "",
                "notes": "multi-camera chute; all cameras must share a split",
            }
        )

    # ---------- URFD: one file = one event; conv duplicates excluded ----------
    for row in urfd:
        if row["status"] == "duplicate_conv":
            continue
        videos.append(
            _video_row(
                row["video_id"],
                "urfd",
                row["video_id"],
                row["video_path"],
                row["label"],
                row["status"],
                use_half=row.get("use_half", ""),
                notes=row.get("notes", ""),
            )
        )
        events.append(
            {
                "event_id": row["video_id"],
                "dataset": "urfd",
                "group": row["split_dir"],
                "label": row["label"],
                "n_videos": 1,
                "n_fall_videos": 1 if row["label"] == "fall" else 0,
                "annotation_status": row["status"],
                "eligible_for_split": row["label"] in ELIGIBLE_LABELS,
                "split": "",
                "notes": row.get("notes", ""),
            }
        )

    duplicate_groups = _resolve_duplicates(videos, events)

    videos.sort(key=lambda r: (r["dataset"], r["event_id"], r["video_id"]))
    events.sort(key=lambda r: (r["dataset"], r["event_id"]))
    return videos, events, duplicate_groups


def stats(videos: list[dict], events: list[dict]) -> dict:
    return {
        "videos": len(videos),
        "events": len(events),
        "videos_by_dataset": dict(Counter(v["dataset"] for v in videos)),
        "events_by_dataset": dict(Counter(e["dataset"] for e in events)),
        "event_labels": {
            f"{ds}:{lb}": count
            for (ds, lb), count in Counter(
                (e["dataset"], e["label"]) for e in events
            ).items()
        },
        "eligible_events": sum(1 for e in events if e["eligible_for_split"]),
        "eligible_fall_events": sum(
            1 for e in events if e["eligible_for_split"] and e["label"] == "fall"
        ),
        "unreadable_videos": [v["video_id"] for v in videos if not v["readable"]],
    }
