"""Phase 0 -- freeze the reference outputs before touching any converter."""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT.parent / "YOLOV26n-pose+ST-GCN++"
OUT_DIR = PROJECT_ROOT / "reports" / "reference"


def _load_own(name: str, rel: str):
    """Load one of *this* project's modules under a private name."""
    spec = importlib.util.spec_from_file_location(name, PROJECT_ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_own = _load_own("deploy_paths", "src/deploy/paths.py")
OWN_WEIGHTS = _own.WEIGHTS

# The source project owns the name `src`; put it first so every `import src.*`
# below resolves to it and never to this project's package.
sys.path.insert(0, str(SOURCE_ROOT))

from src.data import paths # noqa: E402 (source project)
from src.pose import extract  # noqa: E402
from src.train import engine  # noqa: E402
from src.window import build as wbuild  # noqa: E402

BOARD = "board"
RUN_DIR = paths.RUNS / "stgcnpp_v2"
GT_CSV = paths.ANNOTATIONS / BOARD / "board_fall_intervals.csv"
# Published by the source project; the 0.70 row of the C strategy must match.
REF_PER_SEGMENT = paths.REPORTS / "board_eval" / "coverage_gate" / "per_segment.csv"
REF_PER_NOFALL = paths.REPORTS / "board_eval" / "coverage_gate" / "per_video_nofall.csv"
EXCLUDE = {"board:squat:clip_0907_170602", "board:squat:clip_0907_170618"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cpu",
                   help="must stay cpu: G0 is a CPU determinism claim")
    p.add_argument("--batch-size", type=int, default=16)
    return p.parse_args()


def load_manifest() -> list[dict]:
    rows = paths.read_csv(paths.MANIFESTS / BOARD / "videos.csv")
    return [r for r in rows if r.get("readable") == "True"]


def load_gt() -> dict[str, dict]:
    out: dict[str, dict] = {}
    for r in paths.read_csv(GT_CSV):
        rec = out.setdefault(r["video_id"], {"fps": float(r["fps"]), "segments": []})
        if r["status"] == "annotated" and r["start_frame"] != "":
            rec["segments"].append((int(r["start_frame"]), int(r["end_frame"])))
    for v in out.values():
        v["segments"].sort()
    return out


def _ann(frame_dir, kp, sc, vid, w0, w1, width, height):
    """Same annotation dict `scripts/32_coverage_gate_sweep.py` builds."""
    k = kp.copy()
    k[:, :, 0] *= float(width)
    k[:, :, 1] *= float(height)
    return {
        "frame_dir": frame_dir,
        "total_frames": int(k.shape[0]),
        "label": 0,
        "label_known": False,
        "keypoint": k[None],
        "keypoint_score": sc[None],
        "img_shape": (int(height), int(width)),
        "original_shape": (int(height), int(width)),
        "video_id": vid,
        "start_frame": int(w0),
        "end_frame": int(w1),
    }


def _series_to_sample(series, cfg, width, height):
    kp, sc, present, _ = series
    cov = float(present.mean())
    kp = kp.copy()
    kp[:, :, 0] /= max(1, width)
    kp[:, :, 1] /= max(1, height)
    idx = wbuild.resample_indices(len(present), cfg.resample_frames)
    return kp[idx].astype(np.float32), sc[idx].astype(np.float32), cov


def build_samples(cfg):
    """Every A/B/C sample of every candidate window, with coverage. No gate."""
    pose_cfg = extract.load_config()
    pose_cfg.output_root = paths.KEYPOINTS
    pose_cfg.index_path = paths.KEYPOINTS / BOARD / "extraction_index.csv"

    windows: list[dict] = []
    samples: list[dict] = []
    anns: list[dict] = []
    chains_merged = 0

    for row in load_manifest():
        vid = row["video_id"]
        path = extract.npz_path_for(pose_cfg, row)
        if not path.exists():
            print(f"  skip {vid}: no keypoints")
            continue
        with np.load(path, allow_pickle=False) as d:
            kp, sc, tid = d["keypoints"], d["scores"], d["track_ids"]
            np_ = d["n_persons"].astype(np.int32)
            bbox = d["bboxes"]
            width, height = int(d["meta_width"]), int(d["meta_height"])
        fps = float(row["fps"])
        n = len(np_)
        win = wbuild.frames_for(fps, cfg.duration_s)
        st = wbuild.frames_for(fps, cfg.stride_s)
        if n < win:
            print(f"  skip {vid}: {n} < {win} frames")
            continue

        chains = wbuild.stitch_id_handoffs(tid, np_, bbox)
        chains_merged += sum(1 for c in chains if len(c) > 1)

        for w0 in range(0, n - win + 1, st):
            w1 = w0 + win
            wid = len(windows)
            windows.append({"window_id": f"w{wid:06d}", "video_id": vid,
                            "group": row["group"], "start": w0, "end": w1,
                            "fps": fps})

            base = wbuild.extract_window(kp, sc, tid, np_, w0, w1, cfg, width, height)
            if base is not None:
                anns.append(_ann(f"a{wid:06d}", base["keypoints"], base["scores"],
                                 vid, w0, w1, width, height))
                samples.append({"window": wid, "kind": "A", "track": int(base["track_id"]),
                                "members": str(int(base["track_id"])),
                                "coverage": float(base["coverage"]), "ann": len(anns) - 1})

            ids = sorted({int(tid[t, p]) for t in range(w0, w1)
                          for p in range(int(np_[t])) if int(tid[t, p]) >= 0})
            for track in ids:
                series = wbuild._track_series(kp[w0:w1], sc[w0:w1], tid[w0:w1],
                                              np_[w0:w1], track, cfg.max_interp_gap_frames)
                k_arr, s_arr, cov = _series_to_sample(series, cfg, width, height)
                anns.append(_ann(f"b{wid:06d}_k{track:04d}", k_arr, s_arr, vid,
                                 w0, w1, width, height))
                samples.append({"window": wid, "kind": "B", "track": int(track),
                                "members": str(int(track)), "coverage": cov,
                                "ann": len(anns) - 1})

            for chain in chains:
                members = set(chain)
                series = wbuild._group_series(kp[w0:w1], sc[w0:w1], tid[w0:w1],
                                              np_[w0:w1], members, cfg.max_interp_gap_frames)
                k_arr, s_arr, cov = _series_to_sample(series, cfg, width, height)
                root = min(members)
                anns.append(_ann(f"c{wid:06d}_k{root:04d}", k_arr, s_arr, vid,
                                 w0, w1, width, height))
                samples.append({"window": wid, "kind": "C", "track": root,
                                "members": "|".join(str(x) for x in sorted(members)),
                                "coverage": cov, "ann": len(anns) - 1})

    return windows, samples, anns, chains_merged


def run_model(anns: list[dict], device: str, batch_size: int) -> np.ndarray:
    """PYSKL inference over an in-memory annotation list."""
    import pickle
    import tempfile

    import torch

    with open(RUN_DIR / "config_used.yaml", "r", encoding="utf-8-sig") as fh:
        import yaml
        cfg = yaml.safe_load(fh)
    cfg["data"]["batch_size"] = batch_size
    cfg["data"]["num_workers"] = 0

    model = engine.build_recognizer(cfg)
    with open(RUN_DIR / "best.pth", "rb") as fh:
        ckpt = torch.load(fh, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    dev = torch.device(device)
    model.to(dev).eval()

    tmp = Path(tempfile.mkstemp(suffix=".pkl")[1])
    try:
        with open(tmp, "wb") as fh:
            pickle.dump({"split": {"test": [a["frame_dir"] for a in anns]},
                         "annotations": anns}, fh, protocol=4)
        cfg["data"]["ann_file"] = str(tmp)
        _, loader = engine.build_loader(cfg, "test")
        _, probs = engine.predict(model, loader, dev)
    finally:
        # The loader's file handle may outlive the loop on Windows; a stale
        # temp file is harmless, a crash here would not be.
        try:
            tmp.unlink(missing_ok=True)
        except PermissionError:
            pass
    return np.asarray(probs, dtype=np.float64)


# ------------------------------------------------------------------ collapsing


def collapse(windows, samples, probs, kind, gate):
    """Per-window max probability over eligible samples of one strategy."""
    best: dict[int, tuple[float, dict]] = {}
    for s in samples:
        if s["kind"] != kind or s["coverage"] < gate:
            continue
        p = float(probs[s["ann"]])
        w = s["window"]
        if w not in best or p > best[w][0]:
            best[w] = (p, s)
    out = []
    for i in range(len(windows)):
        p, s = best.get(i, (0.0, None))
        out.append((p, s))
    return out


def evaluate(windows, wp, gt, threshold, exclude):
    by_video: dict[str, list[int]] = defaultdict(list)
    for i, w in enumerate(windows):
        by_video[w["video_id"]].append(i)

    seg_tot = seg_hit = pos_v = pos_h = neg_v = neg_fa = 0
    per_seg, per_video = [], []
    for vid, idxs in sorted(by_video.items()):
        if vid in exclude or vid not in gt:
            continue
        segs = gt[vid]["segments"]
        if segs:
            pos_v += 1
            any_hit = False
            for (s, e) in segs:
                cand = [i for i in idxs
                        if max(0, min(windows[i]["end"], e + 1) - max(windows[i]["start"], s)) > 0]
                best = max((wp[i][0] for i in cand), default=0.0)
                ok = best >= threshold
                seg_tot += 1
                seg_hit += int(ok)
                any_hit |= ok
                per_seg.append({"video_id": vid, "start": s, "end": e,
                                "fps": gt[vid]["fps"], "p": best, "hit": bool(ok),
                                "n_windows": len(cand)})
            pos_h += int(any_hit)
        else:
            neg_v += 1
            vmax = max((wp[i][0] for i in idxs), default=0.0)
            fa = vmax >= threshold
            neg_fa += int(fa)
            per_video.append({"video_id": vid, "max_p": vmax, "false_alarm": bool(fa),
                              "n_windows": len(idxs)})
    return {"seg_tot": seg_tot, "seg_hit": seg_hit, "pos_v": pos_v, "pos_h": pos_h,
            "neg_v": neg_v, "neg_fa": neg_fa, "per_seg": per_seg, "per_video": per_video}


def read_ref_csv(path: Path, strategy: str, gate: str) -> dict:
    rows = paths.read_csv(path)
    return {(r["video_id"], int(r["start_frame"])): r
            for r in rows if r["strategy"] == strategy and r["gate"] == gate}


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # -- weights must be the frozen ones before anything else happens
    print("weights:")
    for name, want in (("yolo26n-pose.pt", "eb3bb8268828aeaf515cec23a4bfafd793944a86fe9af94ba7823609c14522a9"),
                       ("stgcnpp_v2_best.pth", "8cd5e74e138c1fb7f57a6b9fd782ac4167e8ecabd050ef385738cfd66b7ab29d")):
        got = paths.sha256_file(OWN_WEIGHTS / name)
        flag = "OK" if got == want else "MISMATCH"
        print(f"  {name:24s} {got[:16]}... {flag}")
        if got != want:
            raise SystemExit(f"{name} is not the frozen weight")

    cfg = wbuild.load_config()
    gt = load_gt()
    print(f"config: duration={cfg.duration_s} stride={cfg.stride_s} "
          f"resample={cfg.resample_frames} min_coverage={cfg.min_coverage} "
          f"max_gap={cfg.max_interp_gap_frames}")

    t0 = time.perf_counter()
    windows, samples, anns, chains = build_samples(cfg)
    print(f"built {len(windows)} windows, {len(anns)} samples "
          f"(chains merged: {chains}) in {time.perf_counter()-t0:.1f}s")
    kinds = defaultdict(int)
    for s in samples:
        kinds[s["kind"]] += 1
    print(f"  samples by strategy: {dict(kinds)}")

    # -- G0: two identical CPU passes
    print("pass 1 ...", flush=True)
    t0 = time.perf_counter()
    p1 = run_model(anns, args.device, args.batch_size)
    t1 = time.perf_counter() - t0
    print(f"  {len(p1)} samples in {t1:.1f}s", flush=True)
    print("pass 2 ...", flush=True)
    t0 = time.perf_counter()
    p2 = run_model(anns, args.device, args.batch_size)
    t2 = time.perf_counter() - t0
    print(f"  {len(p2)} samples in {t2:.1f}s", flush=True)

    delta = float(np.max(np.abs(p1 - p2))) if len(p1) else 0.0
    n_diff = int((p1 != p2).sum())
    g0 = {
        "device": args.device,
        "n_samples": int(len(p1)),
        "max_abs_delta": delta,
        "n_samples_differing": n_diff,
        "pass1_seconds": round(t1, 2),
        "pass2_seconds": round(t2, 2),
        "passed": bool(delta == 0.0),
        "rule": "G0: max|delta p| == 0 on two identical CPU passes",
    }
    (OUT_DIR / "g0_determinism.json").write_text(
        json.dumps(g0, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"G0: max|dp|={delta:g} differing={n_diff} -> "
          f"{'PASS' if g0['passed'] else 'FAIL'}")

    # -- reference tables
    rows = []
    for i, w in enumerate(windows):
        rows.append({"window_id": w["window_id"], "video_id": w["video_id"],
                     "group": w["group"], "start_frame": w["start"],
                     "end_frame": w["end"], "fps": f"{w['fps']:.6f}"})
    with open(OUT_DIR / "board_candidates.csv", "w", newline="", encoding="utf-8-sig") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)

    sample_rows = []
    for s in samples:
        sample_rows.append({
            "window_id": windows[s["window"]]["window_id"],
            "video_id": windows[s["window"]]["video_id"],
            "strategy": s["kind"],
            "track_id": s["track"],
            "members": s["members"],
            "coverage": f"{s['coverage']:.6f}",
            "prob_fall": f"{p1[s['ann']]:.9f}",
        })
    with open(OUT_DIR / "board_samples.csv", "w", newline="", encoding="utf-8-sig") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(sample_rows[0].keys()))
        wr.writeheader()
        wr.writerows(sample_rows)

    np.savez_compressed(
        OUT_DIR / "board_candidates.npz",
        window_id=np.array([w["window_id"] for w in windows]),
        video_id=np.array([w["video_id"] for w in windows]),
        group=np.array([w["group"] for w in windows]),
        start=np.array([w["start"] for w in windows], dtype=np.int32),
        end=np.array([w["end"] for w in windows], dtype=np.int32),
        fps=np.array([w["fps"] for w in windows], dtype=np.float64),
        sample_window=np.array([s["window"] for s in samples], dtype=np.int32),
        sample_kind=np.array([s["kind"] for s in samples]),
        sample_track=np.array([s["track"] for s in samples], dtype=np.int32),
        sample_members=np.array([s["members"] for s in samples]),
        sample_coverage=np.array([s["coverage"] for s in samples], dtype=np.float64),
        sample_prob=np.asarray(p1, dtype=np.float64),
    )

    # -- validation against the source project's published table
    ref = read_ref_csv(REF_PER_SEGMENT, "C", "0.70")
    lines = ["# G0 / 参考冻结校验", "",
             f"- 模型 `runs/stgcnpp_v2/best.pth`（D-023 基线），设备 **{args.device}**",
             f"- 候选窗口 {len(windows)}，样本 {len(anns)}（A {kinds['A']} / "
             f"B {kinds['B']} / C {kinds['C']}），换 ID 合并链 {chains} 条",
             f"- G0 确定性：两次 CPU 全量前向 `max|Δp| = {delta:g}`，"
             f"不一致样本 {n_diff} → **{'通过' if g0['passed'] else '未通过'}**",
             "", "## 与源项目 `per_segment.csv`（策略 C，门 0.70）逐段对比", "",
             "| 视频 | 段起点 | 参考 p | 本次 p | Δ |", "|---|---:|---:|---:|---:|"]
    worst = 0.0
    missing = 0
    for r in evaluate(windows, collapse(windows, samples, p1, "C", 0.70),
                      gt, 0.5, EXCLUDE)["per_seg"]:
        key = (r["video_id"], r["start"])
        ref_row = ref.get(key)
        if ref_row is None:
            missing += 1
            continue
        ref_p = float(ref_row["best_prob"])
        d = abs(ref_p - r["p"])
        worst = max(worst, d)
        lines.append(f"| {r['video_id'].split(':')[-1]} | {r['start']/r['fps']:.3f} | "
                     f"{ref_p:.6f} | {r['p']:.6f} | {d:.2e} |")
    lines += ["", f"- 最大逐段偏差 **{worst:.3e}**；参考表里未匹配到的段 {missing} 个。", ""]
    (OUT_DIR / "freeze_check.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # -- summary of the three strategies, so G3 has a baseline to hit
    summary = {}
    for kind in ("A", "B", "C"):
        res = evaluate(windows, collapse(windows, samples, p1, kind, 0.70), gt, 0.5, EXCLUDE)
        summary[kind] = {"seg_hit": res["seg_hit"], "seg_tot": res["seg_tot"],
                         "pos_hit": res["pos_h"], "pos_v": res["pos_v"],
                         "neg_fa": res["neg_fa"], "neg_v": res["neg_v"]}
    print("strategies @0.70:", json.dumps(summary, ensure_ascii=False))
    (OUT_DIR / "reference_summary.json").write_text(
        json.dumps({"g0": g0, "strategies": summary, "windows": len(windows),
                    "samples": len(anns), "chains_merged": chains},
                   ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"\nwrote {OUT_DIR}")
    print(f"max per-segment delta vs published table: {worst:.3e}")
    return 0 if g0["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
