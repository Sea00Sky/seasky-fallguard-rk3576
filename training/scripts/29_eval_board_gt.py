"""Board evaluation against the adjudicated ground truth ()."""
from __future__ import annotations

import argparse
import pickle
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import paths  # noqa: E402
from src.pose import extract  # noqa: E402
from src.train import engine  # noqa: E402
from src.window import build as wbuild  # noqa: E402

BOARD = "board"
KEYPOINT_ROOT = paths.KEYPOINTS / BOARD
GT_CSV = paths.ANNOTATIONS / BOARD / "board_fall_intervals.csv"
RUN_DIR = paths.RUNS / "stgcnpp_v2"
REPORT_ROOT = paths.REPORTS / "board_eval" / "gt"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--persist", type=int, default=2,
                   help="rule C: consecutive alarmed windows by the same track")
    p.add_argument("--device", default="")
    p.add_argument("--out-dir", default=str(REPORT_ROOT))
    return p.parse_args()


# ------------------------------------------------------------------ ground truth


def load_gt() -> dict[str, dict]:
    """video_id -> {status, fps, n_frames, segments: [(s, e), ...]} (0-based incl)."""
    rows = paths.read_csv(GT_CSV)
    out: dict[str, dict] = {}
    for r in rows:
        vid = r["video_id"]
        rec = out.setdefault(vid, {
            "status": r["status"],
            "fps": float(r["fps"]),
            "n_frames": int(r["n_frames"]),
            "segments": [],
        })
        if r["status"] == "annotated" and r["start_frame"] != "":
            rec["segments"].append((int(r["start_frame"]), int(r["end_frame"])))
    for rec in out.values():
        rec["segments"].sort()
    return out


def load_manifest() -> list[dict]:
    rows = paths.read_csv(paths.MANIFESTS / BOARD / "videos.csv")
    return [r for r in rows if r.get("readable") == "True"]


# ----------------------------------------------------------------------- samples


def _ann(frame_dir: str, kp: np.ndarray, sc: np.ndarray, video_id: str,
         w0: int, w1: int, width: int, height: int) -> dict:
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
        "video_id": video_id,
        "start_frame": int(w0),
        "end_frame": int(w1),
    }


def build_samples(cfg, out_dir: Path):
    """Return (windows, persons, pkl). Every eligible track of every window."""
    rows = load_manifest()
    pose_cfg = extract.load_config()
    pose_cfg.output_root = paths.KEYPOINTS
    pose_cfg.index_path = KEYPOINT_ROOT / "extraction_index.csv"

    windows: list[dict] = []
    persons: list[dict] = []
    anns: list[dict] = []

    for row in rows:
        vid = row["video_id"]
        path = extract.npz_path_for(pose_cfg, row)
        if not path.exists():
            continue
        with np.load(path, allow_pickle=False) as d:
            kp = d["keypoints"]
            sc = d["scores"]
            tid = d["track_ids"]
            np_ = d["n_persons"].astype(np.int32)
            width = int(d["meta_width"])
            height = int(d["meta_height"])
        fps = float(row["fps"])
        n_frames = len(np_)
        window = wbuild.frames_for(fps, cfg.duration_s)
        stride = wbuild.frames_for(fps, cfg.stride_s)
        if n_frames < window:
            continue

        for w0 in range(0, n_frames - window + 1, stride):
            w1 = w0 + window
            wid = len(windows)
            base = wbuild.extract_window(kp, sc, tid, np_, w0, w1, cfg, width, height)
            windows.append({
                "window_id": f"w{wid:06d}",
                "video_id": vid, "group": row["group"],
                "start_frame": w0, "end_frame": w1, "fps": fps,
                "width": width, "height": height,
                "coverage": float(base["coverage"]) if base else 0.0,
                "baseline_track": int(base["track_id"]) if base else -1,
                "baseline_ok": bool(base and base["coverage"] >= cfg.min_coverage),
            })
            if base is not None:
                anns.append(_ann(f"b{wid:06d}", base["keypoints"].astype(np.float32),
                                 base["scores"].astype(np.float32), vid, w0, w1,
                                 width, height))
                persons.append({"window": wid, "track": int(base["track_id"]),
                                "coverage": float(base["coverage"]),
                                "ann": len(anns) - 1, "kind": "baseline"})

            ids = sorted({int(tid[t, p]) for t in range(w0, w1)
                          for p in range(int(np_[t])) if int(tid[t, p]) >= 0})
            for k in ids:
                kpw, scw, present, _ = wbuild._track_series(
                    kp[w0:w1], sc[w0:w1], tid[w0:w1], np_[w0:w1], k,
                    cfg.max_interp_gap_frames)
                cov = float(present.mean())
                if cov < cfg.min_coverage:
                    continue
                kpw = kpw.copy()
                kpw[:, :, 0] /= max(1, width)
                kpw[:, :, 1] /= max(1, height)
                idx = wbuild.resample_indices(w1 - w0, cfg.resample_frames)
                anns.append(_ann(f"t{wid:06d}_k{k:04d}", kpw[idx], scw[idx],
                                 vid, w0, w1, width, height))
                persons.append({"window": wid, "track": k, "coverage": cov,
                                "ann": len(anns) - 1, "kind": "person"})

    out_dir.mkdir(parents=True, exist_ok=True)
    pkl = out_dir / "_samples.pkl"
    with open(pkl, "wb") as fh:
        pickle.dump({"split": {"test": [a["frame_dir"] for a in anns]},
                     "annotations": anns}, fh, protocol=4)
    return windows, persons, pkl


def run_model(pkl: Path, device: str) -> np.ndarray:
    import torch
    import yaml

    with open(RUN_DIR / "config_used.yaml", "r", encoding="utf-8-sig") as fh:
        cfg = yaml.safe_load(fh)
    cfg["data"]["ann_file"] = str(pkl)
    cfg["data"]["num_workers"] = 0
    model = engine.build_recognizer(cfg)
    with open(RUN_DIR / "best.pth", "rb") as fh:
        ckpt = torch.load(fh, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(dev)
    _, loader = engine.build_loader(cfg, "test")
    _, probs = engine.predict(model, loader, dev)
    return probs


# ------------------------------------------------------------------- aggregation


def aggregate(windows: list[dict], persons: list[dict], probs: np.ndarray,
              threshold: float, persist: int) -> dict[str, dict]:
    """Return three strategies: {name: {window_id -> (p, track, members)}}."""
    by_window: dict[int, list[tuple[float, int, float]]] = defaultdict(list)
    for s in persons:
        by_window[s["window"]].append((float(probs[s["ann"]]), s["track"],
                                       s["coverage"]))

    out: dict[str, dict] = {"A_max_presence": {}, "B_per_person_max": {},
                            "C_per_person_consistent": {}}
    for w in windows:
        wid = int(w["window_id"][1:])
        cand = by_window.get(wid, [])
        base = [c for c in cand if c[1] == w["baseline_track"] and c[2] >= 0.70]
        out["A_max_presence"][w["window_id"]] = (
            base[0] if base else (0.0, w["baseline_track"], 0.0))

        # B: max over eligible persons
        elig = [c for c in cand if c[2] >= 0.70]
        out["B_per_person_max"][w["window_id"]] = (
            max(elig, key=lambda x: x[0]) if elig else (0.0, -1, 0.0))

        # C: same as B, identity persisted over consecutive windows
        out["C_per_person_consistent"][w["window_id"]] = None  # filled below

    # C: per video, walk windows in time order; a candidate alarm stands only if
    # the same track is the argmax in >= persist consecutive alarmed windows.
    by_video: dict[str, list[dict]] = defaultdict(list)
    for w in windows:
        by_video[w["video_id"]].append(w)
    for vid, ws in by_video.items():
        ws.sort(key=lambda r: r["start_frame"])
        seq = []
        for w in ws:
            p, k, cov = out["B_per_person_max"][w["window_id"]]
            seq.append((w["window_id"], p, k))
        run = []
        for i, (wid, p, k) in enumerate(seq):
            if p < threshold:
                run = []
                out["C_per_person_consistent"][wid] = (p, k, 0.0)
                continue
            if run and seq[run[-1]][2] != k:
                run = []
            run.append(i)
            if len(run) >= persist:
                out["C_per_person_consistent"][wid] = (p, k, 0.0)
            else:
                out["C_per_person_consistent"][wid] = (p, k, 0.0)
        # zero out windows whose run never reached persist
        i = 0
        while i < len(seq):
            if seq[i][1] < threshold:
                i += 1
                continue
            j = i
            while j + 1 < len(seq) and seq[j + 1][1] >= threshold \
                    and seq[j + 1][2] == seq[i][2]:
                j += 1
            if j - i + 1 < persist:
                for k2 in range(i, j + 1):
                    wid, p, k = seq[k2]
                    out["C_per_person_consistent"][wid] = (0.0, k, 0.0)
            i = j + 1
    return out


# --------------------------------------------------------------------- metrics


def overlaps(w0: int, w1: int, s: int, e: int) -> int:
    return max(0, min(w1, e + 1) - max(w0, s))


def evaluate(windows: list[dict], pred: dict[str, tuple], gt: dict[str, dict],
             threshold: float) -> dict:
    """Segment recall + video-level false alarms + window-level onset P/R/F1."""
    by_video: dict[str, list[dict]] = defaultdict(list)
    for w in windows:
        by_video[w["video_id"]].append(w)

    seg_total = seg_hit = 0
    per_seg = []
    neg_videos = pos_videos = pos_video_hit = neg_video_fa = 0
    tp = fp = fn = tn = 0
    drop = 0

    for vid, ws in sorted(by_video.items()):
        ws.sort(key=lambda r: r["start_frame"])
        rec = gt.get(vid)
        if rec is None:
            continue
        segs = rec["segments"]
        prob = {w["window_id"]: pred[w["window_id"]][0] for w in ws}

        # ---- segment level: detected if any overlapping window alarms
        if segs:
            pos_videos += 1
            vid_hit = False
            for (s, e) in segs:
                seg_total += 1
                hits = [prob[w["window_id"]] for w in ws
                        if overlaps(w["start_frame"], w["end_frame"], s, e) > 0]
                best = max(hits) if hits else 0.0
                ok = best >= threshold
                seg_hit += int(ok)
                vid_hit |= ok
                per_seg.append({"video_id": vid, "start": s, "end": e,
                                "fps": rec["fps"], "best_p": best,
                                "n_windows": len(hits), "hit": ok})
            pos_video_hit += int(vid_hit)
        else:
            neg_videos += 1
            if any(p >= threshold for p in prob.values()):
                neg_video_fa += 1

        for w in ws:
            w0, w1 = w["start_frame"], w["end_frame"]
            onset = any(w0 <= s < w1 for (s, _e) in segs)
            touch = any(overlaps(w0, w1, s, e) > 0 for (s, e) in segs)
            if onset:
                label = 1
            elif touch:
                drop += 1
                continue
            else:
                label = 0
            hit = prob[w["window_id"]] >= threshold
            if label == 1 and hit:
                tp += 1
            elif label == 1:
                fn += 1
            elif hit:
                fp += 1
            else:
                tn += 1

    prec = tp / (tp + fp) if tp + fp else 0.0
    rec_ = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec_ / (prec + rec_) if prec + rec_ else 0.0
    return {
        "seg_total": seg_total, "seg_hit": seg_hit,
        "seg_recall": seg_hit / seg_total if seg_total else 0.0,
        "pos_videos": pos_videos, "pos_video_hit": pos_video_hit,
        "neg_videos": neg_videos, "neg_video_fa": neg_video_fa,
        "window": {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
                   "precision": prec, "recall": rec_, "f1": f1,
                   "dropped_ambiguous": drop},
        "per_seg": per_seg,
    }


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args()
    cfg = wbuild.load_config()
    out_dir = paths.PROJECT_ROOT / args.out_dir
    gt = load_gt()

    windows, persons, pkl = build_samples(cfg, out_dir)
    print(f"windows={len(windows)} person-samples={len(persons)}", flush=True)
    probs = run_model(pkl, args.device)
    if len(probs) != len(persons):
        print(f"prediction count {len(probs)} != samples {len(persons)}")
        return 1

    aggs = aggregate(windows, persons, probs, args.threshold, args.persist)
    results = {}
    for name, pred in aggs.items():
        results[name] = evaluate(windows, pred, gt, args.threshold)

    lines = ["# 板端评估（对抗裁决后的真值）", ""]
    lines += [
        "- 模型 `runs/stgcnpp_v2/best.pth`（D-023 基线），阈值 "
        f"{args.threshold}，窗口 2.0 s / stride 0.5 s / 覆盖率 ≥ {cfg.min_coverage}",
        f"- 真值 `{GT_CSV.relative_to(paths.PROJECT_ROOT).as_posix()}`（D-039）："
        f"{len(gt)} 个视频，{sum(len(v['segments']) for v in gt.values())} 段跌倒，"
        f"{sum(1 for v in gt.values() if not v['segments'])} 个 no_fall",
        "- **板端数据仅用于评估，未参与训练。**",
        "- 唯一变量：窗口内的选人规则。窗口几何、模型、关键点、阈值均未改。",
        "",
        "## 汇总",
        "",
        "| 策略 | 段召回 | 有跌倒视频 | 无误报视频 | 窗口 P | 窗口 R | 窗口 F1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    label = {
        "A_max_presence": "A `max_presence`（D-018 基线）",
        "B_per_person_max": "B 逐人打分取最大",
        "C_per_person_consistent": f"C 逐人取最大 + 同一人连续 {args.persist} 窗",
    }
    for name in ("A_max_presence", "B_per_person_max", "C_per_person_consistent"):
        r = results[name]
        w = r["window"]
        lines.append(
            f"| {label[name]} | {r['seg_hit']}/{r['seg_total']} "
            f"({r['seg_recall']:.3f}) | {r['pos_video_hit']}/{r['pos_videos']} | "
            f"{r['neg_videos'] - r['neg_video_fa']}/{r['neg_videos']} | "
            f"{w['precision']:.3f} | {w['recall']:.3f} | {w['f1']:.3f} |")

    lines += ["", "## 逐段（A 基线 → B 逐人最大）", "",
              "| 视频 | 段 (s) | A best_p | B best_p | A | B |",
              "|---|---|---:|---:|:-:|:-:|"]
    a_seg = {(s["video_id"], s["start"]): s for s in results["A_max_presence"]["per_seg"]}
    for s in results["B_per_person_max"]["per_seg"]:
        a = a_seg.get((s["video_id"], s["start"]))
        name = s["video_id"].split(":")[-1]
        t0, t1 = s["start"] / s["fps"], (s["end"] + 1) / s["fps"]
        lines.append(
            f"| {name} | {t0:.1f}–{t1:.1f} | {a['best_p'] if a else 0:.3f} | "
            f"{s['best_p']:.3f} | {'✓' if a and a['hit'] else '✗'} | "
            f"{'✓' if s['hit'] else '✗'} |")

    lines += ["", "## 逐视频误报（no_fall 组）", "",
              "| 视频 | 最大概率 A | 最大概率 B | 最大概率 C |", "|---|---:|---:|---:|"]
    for vid in sorted(gt):
        if gt[vid]["segments"]:
            continue
        ws = [w["window_id"] for w in windows if w["video_id"] == vid]
        if not ws:
            continue
        row = []
        for name in ("A_max_presence", "B_per_person_max", "C_per_person_consistent"):
            p = aggs[name]
            row.append(max(p[w][0] for w in ws))
        lines.append(f"| {vid.split(':')[-1]} | {row[0]:.3f} | {row[1]:.3f} | {row[2]:.3f} |")

    report = out_dir / "summary.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nreport: {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
