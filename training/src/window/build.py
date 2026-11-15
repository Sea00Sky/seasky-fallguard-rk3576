"""Build fixed-length skeleton windows from extracted keypoints (v2)."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from src.data import paths
from src.pose.qc import load_interval_specs

WINDOW_FIELDS = [
    "window_id", "video_id", "dataset", "event_id", "split", "label",
    "start_frame", "end_frame", "fps", "window_frames", "width", "height",
    "coverage", "track_id", "n_valid_frames", "n_interp_frames", "mean_score",
]

NPZ_KEYS = ("keypoints", "scores", "labels")


@dataclass
class WindowConfig:
    duration_s: float
    stride_s: float
    guard_s: float
    negative_min_distance_s: float
    resample_frames: int
    min_coverage: float
    max_interp_gap_frames: int
    person_selection: str
    coordinate_norm: str
    output_root: Path
    version: str = "v2"
    manifest_dir: Path | None = None
    annotation_dir: Path | None = None
    keypoint_index: Path | None = None
    pyskl_root: Path | None = None

    @property
    def windows_npz(self) -> Path:
        return self.output_root / "windows.npz"

    @property
    def windows_csv(self) -> Path:
        return self.output_root / "windows.csv"

    @property
    def summary_json(self) -> Path:
        return self.output_root / "windows_summary.json"


def load_config(path: Path | None = None, version: str = "v2") -> WindowConfig:
    path = Path(path) if path else paths.PROJECT_ROOT / "configs" / "labels.yaml"
    with open(path, "r", encoding="utf-8-sig") as fh:
        raw = yaml.safe_load(fh)
    w = raw["window"]
    rules = " ".join(raw.get("labeling", {}).get("positive", [])).lower()
    if "onset" not in rules:
        raise ValueError(
            "D-017 requires the onset positive rule; update configs/labels.yaml"
        )
    person = str(w.get("person_selection", "max_presence"))
    if person != "max_presence":
        raise ValueError(f"unsupported person_selection: {person}")
    norm = str(w.get("coordinate_norm", "image_size"))
    if norm != "image_size":
        raise ValueError(f"unsupported coordinate_norm: {norm}")
    roots = paths.layer(version)
    return WindowConfig(
        duration_s=float(w["duration_s"]),
        stride_s=float(w["stride_s"]),
        guard_s=float(w["guard_s"]),
        negative_min_distance_s=float(w["negative_min_distance_from_fall_s"]),
        resample_frames=int(w["resample_frames"]),
        min_coverage=float(w["min_coverage"]),
        max_interp_gap_frames=int(w["max_interp_gap_frames"]),
        person_selection=person,
        coordinate_norm=norm,
        output_root=roots["windows"],
        version=version,
        manifest_dir=roots["manifests"],
        annotation_dir=roots["annotations"],
        keypoint_index=roots["keypoints"] / "extraction_index.csv",
        pyskl_root=roots["pyskl"],
    )


def frames_for(fps: float, seconds: float) -> int:
    return max(1, int(round(fps * seconds)))


def load_videos(manifest_dir: Path | None = None) -> list[dict]:
    """Manifest rows that carry a split (non-empty split only)."""
    rows = paths.read_csv((Path(manifest_dir) if manifest_dir else paths.MANIFESTS_V2) / "videos.csv")
    return [r for r in rows if r.get("split")]


def load_keypoint_paths(index_path: Path | None = None) -> dict[str, str]:
    rows = paths.read_csv(
        Path(index_path) if index_path else paths.KEYPOINTS_V2 / "extraction_index.csv"
    )
    return {
        r["video_id"]: r["npz_path"]
        for r in rows
        if r.get("status") == "ok" and r.get("npz_path")
    }


def label_for(
    w0: int, w1: int, falls: list[tuple[int, int]], guard_frames: int
) -> int:
    """Return 1 (fall), 0 (non_fall) or -1 (ambiguous, drop)."""
    if not falls:
        return 0
    if any(w0 <= s < w1 for s, _e in falls):
        return 1
    if any(max(0, min(w1, e) - max(w0, s)) > 0 for s, e in falls):
        return -1
    distance = min(min(abs(w0 - e), abs(s - w1)) for s, e in falls)
    return 0 if distance >= guard_frames else -1


def _choose_track(
    track_ids: np.ndarray, scores: np.ndarray, n_persons: np.ndarray
) -> int:
    """Track with the most appearances in the window; tie-break by mean score."""
    counts: dict[int, int] = {}
    score_sum: dict[int, float] = {}
    for t in range(len(n_persons)):
        for p in range(int(n_persons[t])):
            tid = int(track_ids[t, p])
            counts[tid] = counts.get(tid, 0) + 1
            score_sum[tid] = score_sum.get(tid, 0.0) + float(np.mean(scores[t, p]))
    if not counts:
        return -1
    return max(counts, key=lambda tid: (counts[tid], score_sum[tid] / counts[tid]))


def _track_series(
    keypoints: np.ndarray,
    scores: np.ndarray,
    track_ids: np.ndarray,
    n_persons: np.ndarray,
    track_id: int,
    max_gap: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Per-frame keypoints/scores for one track; interpolate short gaps."""
    n_frames = len(n_persons)
    kp = np.zeros((n_frames, 17, 2), dtype=np.float32)
    sc = np.zeros((n_frames, 17), dtype=np.float32)
    present = np.zeros(n_frames, dtype=bool)
    for t in range(n_frames):
        for p in range(int(n_persons[t])):
            if int(track_ids[t, p]) == track_id:
                kp[t] = keypoints[t, p].astype(np.float32)
                sc[t] = scores[t, p].astype(np.float32)
                present[t] = True
                break

    interp = np.zeros(n_frames, dtype=bool)
    i = 0
    while i < n_frames:
        if present[i]:
            i += 1
            continue
        j = i
        while j < n_frames and not present[j]:
            j += 1
        gap = j - i
        if i > 0 and j < n_frames and gap <= max_gap:
            for k in range(i, j):
                a = (k - i + 1) / (gap + 1)
                kp[k] = (1.0 - a) * kp[i - 1] + a * kp[j]
            interp[i:j] = True
        i = j
    return kp, sc, present, int(interp.sum())


def resample_indices(n_frames: int, target: int) -> np.ndarray:
    if n_frames <= 0:
        raise ValueError("empty window")
    idx = np.linspace(0, n_frames - 1, target)
    return np.rint(idx).astype(np.int64)




def _track_geometry(
    track_ids: np.ndarray, n_persons: np.ndarray, bboxes: np.ndarray
) -> tuple[dict[int, dict], dict[int, dict[int, tuple[tuple[float, float], float]]]]:
    """Per-track span plus the centre/diagonal seen at each frame."""
    meta: dict[int, dict] = {}
    frames: dict[int, dict[int, tuple[tuple[float, float], float]]] = {}
    for t in range(len(n_persons)):
        for p in range(int(n_persons[t])):
            tid = int(track_ids[t, p])
            if tid < 0:
                continue
            box = bboxes[t, p].astype(np.float64)
            centre = ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)
            diag = float(np.hypot(box[2] - box[0], box[3] - box[1]))
            m = meta.setdefault(tid, {"first": t, "last": t})
            m["last"] = t
            frames.setdefault(tid, {})[t] = (centre, diag)
    return meta, frames


def _handoff_distance(
    frames: dict, a: int, b: int, overlap_max: int
) -> float | None:
    """How far apart tracks `a` and `b` sit where they overlap."""
    shared = sorted(set(frames[a]) & set(frames[b]))
    if not shared or len(shared) > overlap_max:
        return None
    worst = 0.0
    for t in shared:
        (ca, da), (cb, db) = frames[a][t], frames[b][t]
        dist = float(np.hypot(ca[0] - cb[0], ca[1] - cb[1]))
        if dist > 0.25 * (da + db):  # 0.5 x the mean diagonal
            return None
        worst = max(worst, dist)
    return worst


def stitch_id_handoffs(
    track_ids: np.ndarray,
    n_persons: np.ndarray,
    bboxes: np.ndarray,
    gap_max: int = 5,
    overlap_max: int = 1,
) -> list[list[int]]:
    """: chain tracks that are the same body re-labelled by the tracker."""
    meta, frames = _track_geometry(track_ids, n_persons, bboxes)
    chains: list[list[int]] = []
    for b in sorted(meta, key=lambda k: (meta[k]["first"], meta[k]["last"])):
        best: tuple[float, int] | None = None
        for ci, chain in enumerate(chains):
            a = chain[-1]
            gap = meta[b]["first"] - meta[a]["last"]
            if gap < -overlap_max or gap > gap_max:
                continue
            dist = _handoff_distance(frames, a, b, overlap_max)
            if dist is None:
                continue
            if best is None or dist < best[0]:
                best = (dist, ci)
        if best is None:
            chains.append([b])
        else:
            chains[best[1]].append(b)
    return chains


def _group_series(
    keypoints: np.ndarray,
    scores: np.ndarray,
    track_ids: np.ndarray,
    n_persons: np.ndarray,
    members: set[int],
    max_gap: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Like `_track_series`, but one series over a set of merged track ids."""
    n_frames = len(n_persons)
    kp = np.zeros((n_frames, 17, 2), dtype=np.float32)
    sc = np.zeros((n_frames, 17), dtype=np.float32)
    present = np.zeros(n_frames, dtype=bool)
    for t in range(n_frames):
        for p in range(int(n_persons[t])):
            if int(track_ids[t, p]) in members:
                kp[t] = keypoints[t, p].astype(np.float32)
                sc[t] = scores[t, p].astype(np.float32)
                present[t] = True
                break

    interp = np.zeros(n_frames, dtype=bool)
    i = 0
    while i < n_frames:
        if present[i]:
            i += 1
            continue
        j = i
        while j < n_frames and not present[j]:
            j += 1
        if i > 0 and j < n_frames and (j - i) <= max_gap:
            for k in range(i, j):
                a = (k - i + 1) / (j - i + 1)
                kp[k] = (1.0 - a) * kp[i - 1] + a * kp[j]
            interp[i:j] = True
        i = j
    return kp, sc, present, int(interp.sum())


def extract_window_persons(
    keypoints: np.ndarray,
    scores: np.ndarray,
    track_ids: np.ndarray,
    n_persons: np.ndarray,
    w0: int,
    w1: int,
    cfg: WindowConfig,
    width: int,
    height: int,
    groups: list[set[int]] | None = None,
    stitch: bool = False,
) -> list[dict]:
    """Board-only ( / ): one sample per eligible person in the window."""
    if stitch:
        if groups is None:
            raise ValueError("stitch=True needs the ID-handoff chains from "
                             "stitch_id_handoffs()")
        candidates = [(min(g), g) for g in groups]
    else:
        candidates = [
            (tid, {tid})
            for tid in sorted({
                int(track_ids[t, p])
                for t in range(w0, w1)
                for p in range(int(n_persons[t]))
                if int(track_ids[t, p]) >= 0
            })
        ]

    out: list[dict] = []
    for root, members in candidates:
        kp, sc, present, n_interp = _group_series(
            keypoints[w0:w1], scores[w0:w1], track_ids[w0:w1], n_persons[w0:w1],
            members, cfg.max_interp_gap_frames,
        )
        coverage = float(present.mean())
        if coverage < cfg.min_coverage:
            continue
        kp = kp.copy()
        kp[:, :, 0] /= max(1, width)
        kp[:, :, 1] /= max(1, height)
        idx = resample_indices(w1 - w0, cfg.resample_frames)
        kp, sc = kp[idx], sc[idx]
        valid = present[idx]
        out.append({
            "track_id": root,
            "members": sorted(members),
            "n_members": len(members),
            "coverage": coverage,
            "n_valid_frames": int(present.sum()),
            "n_interp_frames": n_interp,
            "mean_score": float(sc[valid].mean()) if valid.any() else 0.0,
            "keypoints": kp.astype(np.float16),
            "scores": sc.astype(np.float16),
        })
    return out


def extract_window_per_person(
    keypoints: np.ndarray,
    scores: np.ndarray,
    track_ids: np.ndarray,
    n_persons: np.ndarray,
    w0: int,
    w1: int,
    cfg: WindowConfig,
    width: int,
    height: int,
) -> list[dict]:
    """without stitching. Thin alias kept for the call sites."""
    return extract_window_persons(
        keypoints, scores, track_ids, n_persons, w0, w1, cfg, width, height,
        stitch=False,
    )


def extract_window(
    keypoints: np.ndarray,
    scores: np.ndarray,
    track_ids: np.ndarray,
    n_persons: np.ndarray,
    w0: int,
    w1: int,
    cfg: WindowConfig,
    width: int,
    height: int,
) -> dict | None:
    """Select a track, normalise, interpolate and resample one window."""
    track_id = _choose_track(track_ids[w0:w1], scores[w0:w1], n_persons[w0:w1])
    if track_id < 0:
        return None
    kp, sc, present, n_interp = _track_series(
        keypoints[w0:w1], scores[w0:w1], track_ids[w0:w1], n_persons[w0:w1],
        track_id, cfg.max_interp_gap_frames,
    )
    coverage = float(present.mean())
    kp[:, :, 0] /= max(1, width)
    kp[:, :, 1] /= max(1, height)
    idx = resample_indices(w1 - w0, cfg.resample_frames)
    kp = kp[idx]
    sc = sc[idx]
    valid = present[idx]
    return {
        "track_id": track_id,
        "coverage": coverage,
        "n_valid_frames": int(present.sum()),
        "n_interp_frames": n_interp,
        "mean_score": float(sc[valid].mean()) if valid.any() else 0.0,
        "keypoints": kp.astype(np.float16),
        "scores": sc.astype(np.float16),
    }


def build(cfg: WindowConfig, verbose: bool = True) -> dict:
    videos = sorted(load_videos(cfg.manifest_dir), key=lambda r: r["video_id"])
    kp_paths = load_keypoint_paths(cfg.keypoint_index)
    specs = load_interval_specs(cfg.annotation_dir)

    records: list[dict] = []
    kps: list[np.ndarray] = []
    scs: list[np.ndarray] = []
    labels: list[int] = []
    stats = {
        "videos_total": len(videos),
        "videos_used": 0,
        "videos_missing_npz": 0,
        "candidate_windows": 0,
        "dropped_short_video": 0,
        "dropped_no_track": 0,
        "dropped_coverage": 0,
        "dropped_ambiguous": 0,
        "kept": 0,
        "positive": 0,
        "negative": 0,
    }

    for row in videos:
        vid = row["video_id"]
        rel = kp_paths.get(vid)
        if not rel:
            stats["videos_missing_npz"] += 1
            continue
        with np.load(paths.PROJECT_ROOT / rel, allow_pickle=False) as data:
            keypoints = data["keypoints"]
            scores = data["scores"]
            track_ids = data["track_ids"]
            n_persons = data["n_persons"].astype(np.int32)
            width = int(data["meta_width"])
            height = int(data["meta_height"])
        fps = float(row["fps"])
        n_frames = len(n_persons)
        window = frames_for(fps, cfg.duration_s)
        stride = frames_for(fps, cfg.stride_s)
        guard = frames_for(fps, cfg.negative_min_distance_s)
        if n_frames < window:
            stats["dropped_short_video"] += 1
            continue

        falls = [
            (s, e) for s, e, tag in specs.get(vid, []) if tag == "fall" and e > s
        ]
        stats["videos_used"] += 1
        for w0 in range(0, n_frames - window + 1, stride):
            w1 = w0 + window
            stats["candidate_windows"] += 1
            label = label_for(w0, w1, falls, guard)
            picked = extract_window(
                keypoints, scores, track_ids, n_persons, w0, w1, cfg, width, height
            )
            if picked is None:
                stats["dropped_no_track"] += 1
                continue
            if picked["coverage"] < cfg.min_coverage:
                stats["dropped_coverage"] += 1
                continue
            if label < 0:
                stats["dropped_ambiguous"] += 1
                continue
            window_id = f"w{len(records):06d}"
            records.append({
                "window_id": window_id,
                "video_id": vid,
                "dataset": row["dataset"],
                "event_id": row["event_id"],
                "split": row["split"],
                "label": label,
                "start_frame": w0,
                "end_frame": w1,
                "fps": f"{fps:.6f}",
                "window_frames": window,
                "width": width,
                "height": height,
                "coverage": f"{picked['coverage']:.4f}",
                "track_id": picked["track_id"],
                "n_valid_frames": picked["n_valid_frames"],
                "n_interp_frames": picked["n_interp_frames"],
                "mean_score": f"{picked['mean_score']:.4f}",
            })
            kps.append(picked["keypoints"])
            scs.append(picked["scores"])
            labels.append(label)
            stats["kept"] += 1
            stats["positive" if label == 1 else "negative"] += 1
            if verbose and stats["kept"] % 1000 == 0:
                print(f"  kept {stats['kept']} windows ...", flush=True)

    if not records:
        raise RuntimeError("no windows built; check keypoints and configs")

    cfg.output_root.mkdir(parents=True, exist_ok=True)
    keypoints_arr = np.stack(kps).astype(np.float16)
    scores_arr = np.stack(scs).astype(np.float16)
    labels_arr = np.asarray(labels, dtype=np.int8)
    with open(cfg.windows_npz, "wb") as fh:
        np.savez_compressed(
            fh, keypoints=keypoints_arr, scores=scores_arr, labels=labels_arr
        )
    paths.write_csv(cfg.windows_csv, records, WINDOW_FIELDS)

    summary = {
        "version": cfg.version,
        "config": {
            "duration_s": cfg.duration_s,
            "stride_s": cfg.stride_s,
            "guard_s": cfg.guard_s,
            "negative_min_distance_s": cfg.negative_min_distance_s,
            "resample_frames": cfg.resample_frames,
            "min_coverage": cfg.min_coverage,
            "max_interp_gap_frames": cfg.max_interp_gap_frames,
            "person_selection": cfg.person_selection,
            "coordinate_norm": cfg.coordinate_norm,
            "annotation_dir": paths.rel_to_project(Path(cfg.annotation_dir)) if cfg.annotation_dir else "",
            "manifest_dir": paths.rel_to_project(Path(cfg.manifest_dir)) if cfg.manifest_dir else "",
            "keypoint_index": paths.rel_to_project(Path(cfg.keypoint_index)) if cfg.keypoint_index else "",
        },
        "stats": stats,
        "arrays": {
            "keypoints": list(keypoints_arr.shape),
            "scores": list(scores_arr.shape),
            "labels": list(labels_arr.shape),
        },
    }
    with open(cfg.summary_json, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    return summary
