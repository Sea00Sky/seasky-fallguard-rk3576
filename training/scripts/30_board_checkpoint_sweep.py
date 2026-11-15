"""板端真值上的 checkpoint 横评（ 之后）。"""
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
GT_CSV = paths.ANNOTATIONS / BOARD / "board_fall_intervals.csv"
OUT_DIR = paths.REPORTS / "board_eval" / "checkpoint_sweep"

# 用户指定的排除项：这两段是「模仿老人缓慢蹲下站不起来」，作为域偏移单独报
EXCLUDE_VIDEOS = {"board:squat:clip_0907_170602", "board:squat:clip_0907_170618"}

RUNS = [
    "stgcnpp_v2",                 # E-01 基线
    "stgcnpp_v2_cw",              # E-02 类权重
    "stgcnpp_v2_aug",             # E-03 数据增强
    "stgcnpp_v2_onset",           # E-04 onset 降权
    "stgcnpp_v2_pretrained",      # E-05 预训练 lr0.02
    "stgcnpp_v2_pretrained_lr002",   # E-08
    "stgcnpp_v2_pretrained_lr001",   # E-09
    "stgcnpp_v2_pretrained_lr0005",  # E-10
    "stgcnpp_v3",                 # E-11 v3 数据层
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--device", default="")
    return p.parse_args()


def load_gt() -> dict[str, dict]:
    rows = paths.read_csv(GT_CSV)
    out: dict[str, dict] = {}
    for r in rows:
        rec = out.setdefault(r["video_id"], {
            "status": r["status"], "fps": float(r["fps"]),
            "n_frames": int(r["n_frames"]), "segments": []})
        if r["status"] == "annotated" and r["start_frame"] != "":
            rec["segments"].append((int(r["start_frame"]), int(r["end_frame"])))
    for rec in out.values():
        rec["segments"].sort()
    return out


def load_manifest() -> list[dict]:
    return [r for r in paths.read_csv(paths.MANIFESTS / BOARD / "videos.csv")
            if r.get("readable") == "True"]


def _ann(frame_dir, kp, sc, video_id, w0, w1, width, height):
    k = kp.copy()
    k[:, :, 0] *= float(width)
    k[:, :, 1] *= float(height)
    return {"frame_dir": frame_dir, "total_frames": int(k.shape[0]),
            "label": 0, "label_known": False,
            "keypoint": k[None], "keypoint_score": sc[None],
            "img_shape": (int(height), int(width)),
            "original_shape": (int(height), int(width)),
            "video_id": video_id, "start_frame": int(w0), "end_frame": int(w1)}


def build_board_samples(cfg) -> tuple[list[dict], list[dict], Path]:
    """One sample per (window, eligible track) — identical to scripts/29."""
    pose_cfg = extract.load_config()
    pose_cfg.output_root = paths.KEYPOINTS
    pose_cfg.index_path = KEYPOINT_ROOT / "extraction_index.csv"

    windows: list[dict] = []
    persons: list[dict] = []
    anns: list[dict] = []

    for row in load_manifest():
        vid = row["video_id"]
        path = extract.npz_path_for(pose_cfg, row)
        if not path.exists():
            continue
        with np.load(path, allow_pickle=False) as d:
            kp, sc, tid = d["keypoints"], d["scores"], d["track_ids"]
            np_ = d["n_persons"].astype(np.int32)
            width, height = int(d["meta_width"]), int(d["meta_height"])
        fps = float(row["fps"])
        n_frames = len(np_)
        window = wbuild.frames_for(fps, cfg.duration_s)
        stride = wbuild.frames_for(fps, cfg.stride_s)
        if n_frames < window:
            continue
        for w0 in range(0, n_frames - window + 1, stride):
            w1 = w0 + window
            wid = len(windows)
            windows.append({"window_id": f"w{wid:06d}", "video_id": vid,
                            "group": row["group"], "start_frame": w0,
                            "end_frame": w1, "fps": fps})
            for p in wbuild.extract_window_per_person(
                    kp, sc, tid, np_, w0, w1, cfg, width, height):
                anns.append(_ann(f"w{wid:06d}_k{p['track_id']:04d}",
                                 p["keypoints"].astype(np.float32),
                                 p["scores"].astype(np.float32),
                                 vid, w0, w1, width, height))
                persons.append({"window": wid, "track": p["track_id"],
                                "ann": len(anns) - 1})

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pkl = OUT_DIR / "_samples.pkl"
    with open(pkl, "wb") as fh:
        pickle.dump({"split": {"test": [a["frame_dir"] for a in anns]},
                     "annotations": anns}, fh, protocol=4)
    return windows, persons, pkl


def score_run(run: str, pkl: Path, n_samples: int, device: str) -> np.ndarray:
    import torch
    import yaml

    with open(paths.RUNS / run / "config_used.yaml", "r", encoding="utf-8-sig") as fh:
        cfg = yaml.safe_load(fh)
    cfg["data"]["ann_file"] = str(pkl)
    cfg["data"]["num_workers"] = 0
    model = engine.build_recognizer(cfg)
    with open(paths.RUNS / run / "best.pth", "rb") as fh:
        ckpt = torch.load(fh, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(dev)
    _, loader = engine.build_loader(cfg, "test")
    _, probs = engine.predict(model, loader, dev)
    assert len(probs) == n_samples, f"{run}: {len(probs)} != {n_samples}"
    return probs


def aggregate(windows, persons, probs):
    """B: window probability = max over the window's person samples."""
    best = np.zeros(len(windows), dtype=np.float64)
    for s in persons:
        p = float(probs[s["ann"]])
        if p > best[s["window"]]:
            best[s["window"]] = p
    return best


def metrics(windows, wprob, gt, threshold, exclude):
    by_video = defaultdict(list)
    for i, w in enumerate(windows):
        by_video[w["video_id"]].append(i)

    seg_total = seg_hit = 0
    pos_vid = pos_hit = 0
    neg_vid = neg_fa = 0
    seg_rows = []
    for vid, idxs in sorted(by_video.items()):
        if vid in exclude:
            continue
        if vid not in gt:
            continue
        segs = gt[vid]["segments"]
        if segs:
            pos_vid += 1
            hit_any = False
            for (s, e) in segs:
                cand = [i for i in idxs
                        if max(0, min(windows[i]["end_frame"], e + 1)
                               - max(windows[i]["start_frame"], s)) > 0]
                best = max((wprob[i] for i in cand), default=0.0)
                seg_total += 1
                ok = best >= threshold
                seg_hit += int(ok)
                hit_any |= ok
                seg_rows.append({"video_id": vid, "start": s, "end": e,
                                 "fps": gt[vid]["fps"], "best": best, "hit": ok})
            pos_hit += int(hit_any)
        else:
            neg_vid += 1
            if any(wprob[i] >= threshold for i in idxs):
                neg_fa += 1
    return {"seg_total": seg_total, "seg_hit": seg_hit,
            "seg_recall": seg_hit / seg_total if seg_total else 0.0,
            "pos_vid": pos_vid, "pos_hit": pos_hit,
            "neg_vid": neg_vid, "neg_fa": neg_fa, "rows": seg_rows}


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args()
    cfg = wbuild.load_config()
    gt = load_gt()
    windows, persons, pkl = build_board_samples(cfg)
    print(f"windows={len(windows)} person-samples={len(persons)} "
          f"excluded={sorted(v.split(':')[-1] for v in EXCLUDE_VIDEOS)}", flush=True)

    results = {}
    for run in RUNS:
        probs = score_run(run, pkl, len(persons), args.device)
        wprob = aggregate(windows, persons, probs)
        results[run] = {"metrics": metrics(windows, wprob, gt, args.threshold,
                                           EXCLUDE_VIDEOS),
                        "wprob": wprob}
        m = results[run]["metrics"]
        print(f"  {run:34s} seg {m['seg_hit']:2d}/{m['seg_total']:2d} "
              f"({m['seg_recall']:.3f})  vid {m['pos_hit']}/{m['pos_vid']}  "
              f"FA {m['neg_fa']}/{m['neg_vid']}", flush=True)

    # --- 排序：段召回优先，其次视频检出，再次误报少
    order = sorted(RUNS, key=lambda r: (
        -results[r]["metrics"]["seg_recall"],
        -results[r]["metrics"]["pos_hit"],
        results[r]["metrics"]["neg_fa"]))

    lines = ["# 板端真值横评：9 个 checkpoint（D-040 口径）", ""]
    lines += [
        f"- 排除 `170602` / `170618`（「模仿老人缓慢蹲下站不起来」，域偏移单独报）后，"
        f"真值为 **{results[order[0]]['metrics']['seg_total']} 段 / "
        f"{results[order[0]]['metrics']['pos_vid']} 个有跌倒视频**",
        f"- 选人规则 B（逐人打分取最大，D-040），阈值 {args.threshold}，窗口 2.0 s / stride 0.5 s",
        "- 9 个 checkpoint 架构一致（2 类 / clip_len 64），只换权重",
        "- **板端数据仅用于评估，未参与任何训练**；本地只有一个种子，不构成统计显著",
        "",
        "## 排名",
        "",
        "| # | checkpoint | 段召回 | 检出视频 | 误报视频 | 备注 |",
        "|---:|---|---:|---:|---:|---|",
    ]
    note = {
        "stgcnpp_v2": "E-01 基线（D-023）",
        "stgcnpp_v2_cw": "E-02 类权重",
        "stgcnpp_v2_aug": "E-03 数据增强",
        "stgcnpp_v2_onset": "E-04 onset 降权",
        "stgcnpp_v2_pretrained": "E-05 预训练 lr 0.02",
        "stgcnpp_v2_pretrained_lr002": "E-08 预训练 lr 0.002",
        "stgcnpp_v2_pretrained_lr001": "E-09 预训练 lr 0.001",
        "stgcnpp_v2_pretrained_lr0005": "E-10 预训练 lr 0.0005",
        "stgcnpp_v3": "E-11 v3 数据层",
    }
    for rank, run in enumerate(order, 1):
        m = results[run]["metrics"]
        lines.append(f"| {rank} | `{run}` | {m['seg_hit']}/{m['seg_total']} "
                     f"({m['seg_recall']:.3f}) | {m['pos_hit']}/{m['pos_vid']} | "
                     f"{m['neg_fa']}/{m['neg_vid']} | {note[run]} |")

    lines += ["", "## 逐段（各 checkpoint 的最高概率）", "",
              "| 视频 | 段 (s) | " + " | ".join(
                  r.replace("stgcnpp_", "").replace("_pretrained", "+pt")
                  for r in order) + " |",
              "|---|---|" + "---:|" * len(order)]
    base_rows = {(r["video_id"], r["start"]): r
                 for r in results[order[0]]["metrics"]["rows"]}
    for key in sorted(base_rows, key=lambda k: (k[0], k[1])):
        r0 = base_rows[key]
        name = key[0].split(":")[-1]
        cells = []
        for run in order:
            rr = next(x for x in results[run]["metrics"]["rows"]
                      if (x["video_id"], x["start"]) == key)
            mark = "**" if rr["hit"] else ""
            cells.append(f"{mark}{rr['best']:.3f}{mark}")
        lines.append(f"| {name} | {r0['start']/r0['fps']:.1f}–"
                     f"{(r0['end']+1)/r0['fps']:.1f} | " + " | ".join(cells) + " |")

    lines += ["", "## 说明", "",
              "- 「段召回」= 至少一个与该段重叠的窗口概率 ≥ 0.5。没有合格轨迹的窗口"
              "（覆盖率 < 0.70）不参与，这部分漏检任何 checkpoint 都救不了（L-039）。",
              "- 单次运行、17 段，差异 <2 段属噪声量级；排名只能当参考，不是统计结论。",
              ""]
    report = OUT_DIR / "summary.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nreport: {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
