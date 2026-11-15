"""Plan A (): inference-side fragment stitching on the board videos."""
from __future__ import annotations

import argparse
import pickle
import sys
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
REPORT_ROOT = paths.REPORTS / "board_eval"
BASELINE_CSV = REPORT_ROOT / "board_predictions.csv"
RUN_DIR = paths.RUNS / "stgcnpp_v2"
TARGET_SUFFIX = "clip_0831_173619"
TARGET_INTERVAL = (40.5, 43.5)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gap-max", type=int, default=10,
                   help="max frames between a track's end and the next's start")
    p.add_argument("--dist-ratio", type=float, default=0.5,
                   help="max centre displacement as a fraction of box diagonal")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--device", default="")
    p.add_argument("--out-dir", default="reports/board_eval/stitched")
    p.add_argument("--tolerance", type=float, default=1e-3,
                   help="baseline replay tolerance")
    return p.parse_args()


def load_manifest() -> list[dict]:
    rows = paths.read_csv(paths.MANIFESTS / BOARD / "videos.csv")
    return [r for r in rows if r.get("readable") == "True"]


def _board_pose_cfg() -> extract.PoseConfig:
    cfg = extract.load_config()
    cfg.output_root = paths.KEYPOINTS
    cfg.index_path = KEYPOINT_ROOT / "extraction_index.csv"
    return cfg


# ------------------------------------------------------------------- stitching


def track_meta(track_ids: np.ndarray, n_persons: np.ndarray, bboxes: np.ndarray) -> dict:
    """First/last appearance and box of every track in one video."""
    meta: dict[int, dict] = {}
    for t in range(len(n_persons)):
        for p in range(int(n_persons[t])):
            tid = int(track_ids[t, p])
            box = bboxes[t, p].astype(np.float64)
            if tid not in meta:
                meta[tid] = {"first": t, "last": t, "first_box": box, "last_box": box,
                             "frames": 0}
            meta[tid]["last"] = t
            meta[tid]["last_box"] = box
            meta[tid]["frames"] += 1
    return meta


def _centre(box: np.ndarray) -> np.ndarray:
    return np.array([(box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0])


def _diag(box: np.ndarray) -> float:
    return float(np.hypot(box[2] - box[0], box[3] - box[1]))


def stitch_chains(meta: dict, gap_max: int, ratio: float) -> list[list[int]]:
    """Greedy temporal chaining; returns lists of track ids, one per person."""
    chains: list[list[int]] = []
    for b in sorted(meta, key=lambda t: (meta[t]["first"], meta[t]["last"])):
        best: tuple[float, int] | None = None
        for ci, chain in enumerate(chains):
            a = chain[-1]
            gap = meta[b]["first"] - meta[a]["last"]
            if gap < 0 or gap > gap_max:
                continue
            dist = float(np.linalg.norm(_centre(meta[a]["last_box"])
                                        - _centre(meta[b]["first_box"])))
            limit = ratio * 0.5 * (_diag(meta[a]["last_box"]) + _diag(meta[b]["first_box"]))
            if dist <= limit and (best is None or dist < best[0]):
                best = (dist, ci)
        if best is None:
            chains.append([b])
        else:
            chains[best[1]].append(b)
    return chains


def group_series(
    keypoints: np.ndarray,
    scores: np.ndarray,
    track_ids: np.ndarray,
    n_persons: np.ndarray,
    members: set[int],
    max_gap: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Same as wbuild._track_series, but over a set of merged track ids."""
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
        gap = j - i
        if i > 0 and j < n_frames and gap <= max_gap:
            for k in range(i, j):
                a = (k - i + 1) / (gap + 1)
                kp[k] = (1.0 - a) * kp[i - 1] + a * kp[j]
            interp[i:j] = True
        i = j
    return kp, sc, present, int(interp.sum())


def resample(kp: np.ndarray, sc: np.ndarray, present: np.ndarray,
             target: int, width: int, height: int):
    kp = kp.copy()
    kp[:, :, 0] /= max(1, width)
    kp[:, :, 1] /= max(1, height)
    idx = wbuild.resample_indices(len(present), target)
    return kp[idx].astype(np.float32), sc[idx].astype(np.float32), present[idx]


# --------------------------------------------------------------------- samples


def build_samples(cfg, gap_max: int, ratio: float, out_dir: Path):
    """Return (baseline_windows, samples, pkl_path, annotations)."""
    rows = load_manifest()
    pose_cfg = _board_pose_cfg()
    baseline: list[dict] = []
    samples: list[dict] = []
    anns: list[dict] = []
    stitch_log: list[dict] = []

    for row in rows:
        path = extract.npz_path_for(pose_cfg, row)
        if not path.exists():
            continue
        with np.load(path, allow_pickle=False) as d:
            keypoints = d["keypoints"]
            scores = d["scores"]
            track_ids = d["track_ids"]
            n_persons = d["n_persons"].astype(np.int32)
            bboxes = d["bboxes"]
            width = int(d["meta_width"])
            height = int(d["meta_height"])
        fps = float(row["fps"])
        n_frames = len(n_persons)
        window = wbuild.frames_for(fps, cfg.duration_s)
        stride = wbuild.frames_for(fps, cfg.stride_s)
        if n_frames < window:
            continue

        meta = track_meta(track_ids, n_persons, bboxes)
        chains = stitch_chains(meta, gap_max, ratio)
        members_by_group: dict[int, set[int]] = {}
        for chain in chains:
            root = min(chain)
            members_by_group[root] = set(chain)
            if len(chain) > 1:
                stitch_log.append({
                    "video_id": row["video_id"], "group": root,
                    "members": chain,
                    "gaps": [meta[chain[k + 1]]["first"] - meta[chain[k]]["last"]
                             for k in range(len(chain) - 1)],
                })

        for w0 in range(0, n_frames - window + 1, stride):
            w1 = w0 + window
            picked = wbuild.extract_window(
                keypoints, scores, track_ids, n_persons, w0, w1, cfg, width, height
            )
            if picked is None or picked["coverage"] < cfg.min_coverage:
                continue
            bw = {
                "video_id": row["video_id"], "group": row["group"],
                "start_frame": w0, "end_frame": w1, "fps": fps,
                "width": width, "height": height,
                "coverage": picked["coverage"], "track_id": picked["track_id"],
            }
            baseline.append(bw)
            b_index = len(baseline) - 1
            # extract_window already normalised and resampled to 64 frames;
            # _ann converts back to pixels exactly like scripts/17 export_pyskl.
            anns.append(_ann(f"b{b_index:06d}",
                             picked["keypoints"].astype(np.float32),
                             picked["scores"].astype(np.float32),
                             row["video_id"], w0, w1, width, height))
            samples.append({"kind": "baseline", "window": b_index,
                            "track_id": picked["track_id"],
                            "coverage": picked["coverage"]})

            # --- stitched path (Plan A) ---
            for root, members in members_by_group.items():
                kp_w, sc_w, present, _ = group_series(
                    keypoints[w0:w1], scores[w0:w1], track_ids[w0:w1],
                    n_persons[w0:w1], members, cfg.max_interp_gap_frames,
                )
                coverage = float(present.mean())
                if coverage < cfg.min_coverage:
                    continue
                kp_s, sc_s, _ = resample(kp_w, sc_w, present,
                                         cfg.resample_frames, width, height)
                anns.append(_ann(f"s{len(samples):06d}", kp_s, sc_s, row["video_id"],
                                 w0, w1, width, height))
                samples.append({"kind": "stitched", "window": b_index,
                                "track_id": int(root),
                                "members": sorted(members),
                                "coverage": coverage})

    out_dir.mkdir(parents=True, exist_ok=True)
    pkl = out_dir / "_samples.pkl"
    with open(pkl, "wb") as fh:
        pickle.dump({"split": {"test": [a["frame_dir"] for a in anns]},
                     "annotations": anns}, fh, protocol=4)
    (out_dir / "_stitch_log.txt").write_text(
        "\n".join(str(s) for s in stitch_log), encoding="utf-8"
    )
    return baseline, samples, pkl, anns


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


# ------------------------------------------------------------------- inference


def run_model(pkl: Path, device: str) -> np.ndarray:
    import torch

    with open(RUN_DIR / "config_used.yaml", "r", encoding="utf-8-sig") as fh:
        import yaml
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


# ------------------------------------------------------------------ reporting


def segments(rows: list[dict], probs: np.ndarray, threshold: float) -> dict[str, list]:
    out: dict[str, list] = defaultdict(list)
    by_video: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        by_video[r["video_id"]].append(i)
    for vid, idxs in by_video.items():
        idxs = sorted(idxs, key=lambda i: rows[i]["start_frame"])
        cur = None
        for i in idxs:
            hit = probs[i] >= threshold
            if hit and cur is None:
                cur = {"start": rows[i]["start_frame"], "end": rows[i]["end_frame"],
                       "peak": float(probs[i])}
            elif hit:
                cur["end"] = max(cur["end"], rows[i]["end_frame"])
                cur["peak"] = max(cur["peak"], float(probs[i]))
            elif cur is not None:
                out[vid].append(cur)
                cur = None
        if cur is not None:
            out[vid].append(cur)
    return out


def snapshots(rows: list[dict], probs: np.ndarray, idxs: list[int],
              out_dir: Path) -> list[dict]:
    """Mid-window frames for every newly alarmed window (manual review)."""
    import cv2

    frame_dir = out_dir / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    manifest = {r["video_id"]: r for r in load_manifest()}
    out = []
    for i in idxs:
        row = rows[i]
        src = paths.RAW / manifest[row["video_id"]]["relative_path"]
        frame = (int(row["start_frame"]) + int(row["end_frame"])) // 2
        cap = cv2.VideoCapture(str(src))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ok, img = cap.read()
        cap.release()
        if not ok:
            continue
        cv2.putText(img, f"p={probs[i]:.3f} t={frame / float(row['fps']):.1f}s",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        name = (f"{row['video_id'].replace(':', '__')}"
                f"__f{frame:04d}__p{probs[i]:.3f}.jpg")
        cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 88])[1].tofile(
            str(frame_dir / name)
        )
        out.append({"window_id": f"b{i:06d}", "video_id": row["video_id"],
                    "frame": frame, "p": float(probs[i]), "path": name})
    return out


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args()
    cfg = wbuild.load_config()
    out_dir = paths.PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline, samples, pkl, anns = build_samples(cfg, args.gap_max,
                                                 args.dist_ratio, out_dir)
    print(f"windows: {len(baseline)}  samples: {len(samples)}  "
          f"(pyskl stitched={sum(s['kind'] == 'stitched' for s in samples)})",
          flush=True)
    probs = run_model(pkl, args.device)
    if len(probs) != len(samples):
        print(f"prediction count {len(probs)} != samples {len(samples)}")
        return 1

    # --- guard: replay the baseline window by window ---
    recorded = {r["window_id"]: r for r in paths.read_csv(BASELINE_CSV)}
    if len(recorded) != len(baseline):
        print(f"baseline window count mismatch: recorded {len(recorded)} "
              f"vs rebuilt {len(baseline)}")
        return 1
    base_probs = np.zeros(len(baseline), dtype=np.float64)
    worst = 0.0
    for i, s in enumerate(samples):
        if s["kind"] == "baseline":
            base_probs[s["window"]] = probs[i]
    for i, row in enumerate(baseline, 0):
        rec = recorded.get(f"b{i:06d}")
        if rec is None:
            print(f"missing recorded window b{i:06d}")
            return 1
        worst = max(worst, abs(float(rec["prob_fall"]) - base_probs[i]))
    print(f"baseline replay: max |delta| = {worst:.2e} (tolerance {args.tolerance})")
    if worst > args.tolerance:
        print("baseline replay failed; not reporting stitched numbers")
        return 1

    # --- stitched aggregation: max over eligible persons ---
    stitched = np.zeros(len(baseline), dtype=np.float64)
    best_members: dict[int, list] = {}
    probs_by_window: dict[int, list] = defaultdict(list)
    for i, s in enumerate(samples):
        if s["kind"] != "stitched":
            continue
        probs_by_window[s["window"]].append((float(probs[i]), s))
    for w, items in probs_by_window.items():
        p, s = max(items, key=lambda x: x[0])
        stitched[w] = p
        best_members[w] = s["members"]

    threshold = args.threshold
    base_segs = segments(baseline, base_probs, threshold)
    st_segs = segments(baseline, stitched, threshold)
    base_videos = {v for v, s in base_segs.items() if s}
    st_videos = {v for v, s in st_segs.items() if s}
    new_videos = sorted(st_videos - base_videos)
    lost_videos = sorted(base_videos - st_videos)

    # target window check
    tgt_idx = [i for i, r in enumerate(baseline)
               if r["video_id"].endswith(TARGET_SUFFIX)]
    tgt_hits = [
        (baseline[i]["start_frame"] / baseline[i]["fps"],
         baseline[i]["end_frame"] / baseline[i]["fps"], float(stitched[i]))
        for i in tgt_idx
        if TARGET_INTERVAL[0] <= baseline[i]["start_frame"] / baseline[i]["fps"]
        <= TARGET_INTERVAL[1]
    ]
    target_ok = any(p >= threshold for _, _, p in tgt_hits)

    # new-alarm windows for manual review
    newly = [
        i for i in range(len(baseline))
        if base_probs[i] < threshold <= stitched[i]
    ]
    lost = [i for i in range(len(baseline))
            if stitched[i] < threshold <= base_probs[i]]
    shots = snapshots(baseline, stitched, newly, out_dir)

    csv_rows = []
    for i, row in enumerate(baseline):
        csv_rows.append({
            "video_id": row["video_id"], "group": row["group"],
            "window_id": f"b{i:06d}",
            "start_s": f"{row['start_frame'] / row['fps']:.2f}",
            "end_s": f"{row['end_frame'] / row['fps']:.2f}",
            "coverage": f"{row['coverage']:.4f}",
            "baseline_track": row["track_id"],
            "p_baseline": f"{base_probs[i]:.6f}",
            "p_stitched": f"{stitched[i]:.6f}",
            "stitched_members": "|".join(str(x) for x in best_members.get(i, [])),
        })
    paths.write_csv(
        out_dir / "per_window.csv", csv_rows,
        list(csv_rows[0].keys()),
    )

    lines = [
        "# 板端推理侧碎片拼接（方案 A，D-030）",
        "",
        "## 预注册（事先固定，未在结果上调参）",
        "",
        f"- 合并规则：后一轨迹在第一轨迹结束后 ≤ {args.gap_max} 帧内开始，"
        f"且框中心位移 ≤ {args.dist_ratio} × 两框对角线均值",
        "- 每个拼接后的人单独打分，覆盖率 ≥ "
        f"{cfg.min_coverage}（configs/labels.yaml）才参与",
        f"- 窗口概率 = 合格人中的最大值；阈值 {threshold}（D-023），未按视频调参",
        "- 模型、关键点、训练窗口、pkl 全部不变；不重抽、不重训",
        "",
        "## 基线复算校验",
        "",
        f"- 重建窗口 {len(baseline)} 个，与 `board_predictions.csv` 逐窗比对，"
        f"最大绝对误差 **{worst:.2e}**（容差 {args.tolerance}）",
        "",
        "## 结果（视频级）",
        "",
        "| 指标 | 基线 max_presence | 拼接 + 逐人取最大 |",
        "|---|---:|---:|",
        f"| 告警视频 | {len(base_videos)} | {len(st_videos)} |",
        f"| 告警段总数 | {sum(len(v) for v in base_segs.values())} | "
        f"{sum(len(v) for v in st_segs.values())} |",
        "",
        f"- 新增告警视频（{len(new_videos)}）："
        + (", ".join(v.split(':')[-1] for v in new_videos) if new_videos else "无"),
        f"- 失去告警的视频："
        + (", ".join(v.split(':')[-1] for v in lost_videos) if lost_videos else "无"),
        "",
        "## 目标窗口：173619 倒床（40.5–43.5 s）",
        "",
        "| 窗口 | baseline | stitched |",
        "|---|---:|---:|",
    ]
    for t0, t1, p in tgt_hits:
        i = next(j for j, r in enumerate(baseline)
                 if r["video_id"].endswith(TARGET_SUFFIX)
                 and abs(r["start_frame"] / r["fps"] - t0) < 1e-6)
        lines.append(f"| {t0:.1f}–{t1:.1f} s | {base_probs[i]:.4f} | **{p:.4f}** |")
    lines += [
        "",
        f"- 判据：40.5–43.5 s 内至少一窗 ≥ {threshold} → "
        f"{'**通过**' if target_ok else '未通过'}",
        f"- 视频数增长 ≤ 2 → "
        f"{'**通过**' if len(new_videos) <= 2 else '未通过'}"
        f"（实测 +{len(new_videos)}）",
        "",
        "## 逐窗变化",
        "",
        f"- 新增 ≥{threshold} 的窗口：**{len(newly)}** 个"
        + (f"，涉及 {len({baseline[i]['video_id'] for i in newly})} 个视频"
           if newly else ""),
        f"- 由 ≥{threshold} 掉到 <{threshold} 的窗口：{len(lost)} 个",
        "- 新增告警窗口全部列在 `per_window.csv`（`p_baseline` < 0.5 ≤ `p_stitched`）。",
        f"- 新增告警窗口的中点帧已抽到 `{out_dir / 'frames'}`（{len(shots)} 张），"
        "供人工确认是否为真跌倒。",
        "",
        "## 说明",
        "",
        "- 板端视频无标注，以上是候选告警变化，不是准确率；新增告警视频需人工看帧。",
        "- 人工复核见 `manual_review.md`：抽查 22/30 个新增窗口，精度约 23%，"
        "方案 A 不直接可部署（D-031）。",
        "- 该脚本只影响板端推理；训练集窗口、D-023 口径与 v2 指标均未改动。",
        "- 拼接规则若要在训练/评测层落地，需走方案 B（修 tracker 并重抽），另需批准。",
        "",
    ]
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"report: {out_dir / 'summary.md'}")
    return 0 if target_ok and len(new_videos) <= 2 else 2


if __name__ == "__main__":
    raise SystemExit(main())
