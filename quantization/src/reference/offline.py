"""The executable reference: source-project code, addressable from here."""
from __future__ import annotations

import importlib.util
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT.parent / "YOLOV26n-pose+ST-GCN++"
REFERENCE = PROJECT_ROOT / "reports" / "reference"
FROZEN_NPZ = REFERENCE / "board_candidates.npz"

EXCLUDE = {"board:squat:clip_0907_170602", "board:squat:clip_0907_170618"}


def load_spec():
    """This project's `src/deploy/spec.py`, loaded by path."""
    name = "deploy_spec"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        name, PROJECT_ROOT / "src" / "deploy" / "spec.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def require_source():
    """Make `import src.*` mean the source project, or fail loudly."""
    if not SOURCE_ROOT.exists():
        raise RuntimeError(
            f"source project not found at {SOURCE_ROOT}. This project only "
            f"*reads* it; the reference cannot be rebuilt without it.")
    ours = sys.modules.get("src")
    if ours is not None and getattr(ours, "__file__", "") and \
            Path(ours.__file__).resolve().is_relative_to(PROJECT_ROOT):
        raise RuntimeError(
            "this project's `src` package is already imported; the source "
            "project's `src.window` would be shadowed. Run the check from a "
            "script that has not imported `src.runtime` (see scripts/14), or "
            "in a fresh process.")
    if str(SOURCE_ROOT) not in sys.path:
        sys.path.insert(0, str(SOURCE_ROOT))
    return SOURCE_ROOT


def source_module(dotted: str):
    """`source_module('src.window.build')`, after `require_source()`."""
    require_source()
    import importlib

    return importlib.import_module(dotted)


# ------------------------------------------------------------------- manifest


def manifest_rows() -> list[dict]:
    """Readable board clips, in manifest order -- the order Phase 0 used."""
    here = PROJECT_ROOT / "src" / "deploy"
    spec = importlib.util.spec_from_file_location(
        "deploy_paths_for_reference", here / "paths.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    rows = mod.read_csv(mod.SOURCE_MANIFEST_BOARD)
    return [r for r in rows if r.get("readable") == "True"]


@lru_cache(maxsize=32)
def video_arrays(video_id: str) -> dict:
    """The source project's extracted keypoints for one board clip."""
    paths = _paths()
    path = paths.keypoint_npz(video_id)
    with np.load(path, allow_pickle=False) as d:
        return {
            "keypoints": d["keypoints"], "scores": d["scores"],
            "bboxes": d["bboxes"], "track_ids": d["track_ids"],
            "n_persons": d["n_persons"].astype(np.int32),
            "width": int(d["meta_width"]), "height": int(d["meta_height"]),
            "fps": float(d["meta_fps"]),
            "n_frames": int(d["meta_n_frames"]),
        }


def _paths():
    name = "deploy_paths"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        name, PROJECT_ROOT / "src" / "deploy" / "paths.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# -------------------------------------------------------------------- windows


def candidate_windows() -> list[dict]:
    """The Phase 0 candidate-window table, rebuilt from manifest + npz."""
    spec = load_spec()
    wbuild = source_module("src.window.build")
    out: list[dict] = []
    for row in manifest_rows():
        vid = row["video_id"]
        try:
            arr = video_arrays(vid)
        except FileNotFoundError:
            continue
        fps = float(row["fps"])
        n = len(arr["n_persons"])
        if n != arr["n_frames"]:
            raise RuntimeError(
                f"{vid}: npz has {n} frames but meta_n_frames says "
                f"{arr['n_frames']}; the keypoint file is truncated")
        win = wbuild.frames_for(fps, spec.DURATION_S)
        st = wbuild.frames_for(fps, spec.STRIDE_S)
        if n < win:
            continue
        for w0 in range(0, n - win + 1, st):
            out.append({"window_id": f"w{len(out):06d}", "video_id": vid,
                        "group": row["group"], "start": w0, "end": w0 + win,
                        "fps": fps})
    return out


def window_samples(video_id: str, w0: int, w1: int) -> list[dict]:
    """Every A/B/C sample of one window, with coverage -- no gate."""
    spec = load_spec()
    wbuild = source_module("src.window.build")
    arr = video_arrays(video_id)
    cfg = wbuild.load_config()
    kp, sc, tid, nper = (arr["keypoints"], arr["scores"], arr["track_ids"],
                         arr["n_persons"])
    width, height = arr["width"], arr["height"]
    out: list[dict] = []

    base = wbuild.extract_window(kp, sc, tid, nper, w0, w1, cfg, width, height)
    if base is not None:
        out.append({"kind": "A", "track": int(base["track_id"]),
                    "members": [int(base["track_id"])],
                    "coverage": float(base["coverage"])})

    ids = sorted({int(tid[t, p]) for t in range(w0, w1)
                  for p in range(int(nper[t])) if int(tid[t, p]) >= 0})
    for track in ids:
        _kp, _sc, present, _n = wbuild._track_series(
            kp[w0:w1], sc[w0:w1], tid[w0:w1], nper[w0:w1], track,
            cfg.max_interp_gap_frames)
        out.append({"kind": "B", "track": track, "members": [track],
                    "coverage": float(present.mean())})

    for chain in wbuild.stitch_id_handoffs(tid, nper, arr["bboxes"]):
        members = set(chain)
        _kp, _sc, present, _n = wbuild._group_series(
            kp[w0:w1], sc[w0:w1], tid[w0:w1], nper[w0:w1], members,
            cfg.max_interp_gap_frames)
        out.append({"kind": "C", "track": min(members),
                    "members": sorted(members),
                    "coverage": float(present.mean())})
    return out


def reference_samples() -> list[dict]:
    """The full frozen sample order: per video, per window, A then B then C."""
    out: list[dict] = []
    for w, win in enumerate(candidate_windows()):
        for s in window_samples(win["video_id"], win["start"], win["end"]):
            s["window"] = w
            out.append(s)
    return out


def stitch_chains(video_id: str) -> list[list[int]]:
    """chains as the source project computes them (offline, whole clip)."""
    wbuild = source_module("src.window.build")
    arr = video_arrays(video_id)
    return wbuild.stitch_id_handoffs(arr["track_ids"], arr["n_persons"],
                                     arr["bboxes"])


def model_input(keypoints_unit: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """(T, V, 2) unit coords + (T, V) scores -> the (1, 1, T, V, 3) graph input."""
    spec = load_spec()
    return spec.stack_batch([spec.to_model_input(
        np.asarray(keypoints_unit, dtype=np.float32),
        np.asarray(scores, dtype=np.float32))])
