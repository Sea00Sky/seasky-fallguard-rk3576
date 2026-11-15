"""板端：轨迹换 ID 修复（ 候选）在真值上的效果。"""
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
RUN_DIR = paths.RUNS / "stgcnpp_v2"
OUT_DIR = paths.REPORTS / "board_eval" / "stitch_gate"
EXCLUDE = {"board:squat:clip_0907_170602", "board:squat:clip_0907_170618"}

GAP_MAX = 5
OVERLAP_MAX = 1


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--device", default="")
    return p.parse_args()


def load_gt() -> dict[str, dict]:
    out: dict[str, dict] = {}
    for r in paths.read_csv(GT_CSV):
        rec = out.setdefault(r["video_id"], {"fps": float(r["fps"]), "segments": []})
        if r["status"] == "annotated" and r["start_frame"] != "":
            rec["segments"].append((int(r["start_frame"]), int(r["end_frame"])))
    for v in out.values():
        v["segments"].sort()
    return out


def load_manifest() -> list[dict]:
    return [r for r in paths.read_csv(paths.MANIFESTS / BOARD / "videos.csv")
            if r.get("readable") == "True"]


def track_frames(tid, np_, bbox):
    meta, frames = {}, {}
    for t in range(len(np_)):
        for p in range(int(np_[t])):
            k = int(tid[t, p])
            if k < 0:
                continue
            b = bbox[t, p]
            c = ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
            diag = float(np.hypot(b[2] - b[0], b[3] - b[1]))
            m = meta.setdefault(k, {"first": t, "last": t})
            m["last"] = t
            frames.setdefault(k, {})[t] = (c, diag)
    return meta, frames


def handoff(frames, a, b):
    shared = sorted(set(frames[a]) & set(frames[b]))
    if not shared or len(shared) > OVERLAP_MAX:
        return None
    worst = 0.0
    for t in shared:
        (ca, da), (cb, db) = frames[a][t], frames[b][t]
        d = float(np.hypot(ca[0] - cb[0], ca[1] - cb[1]))
        if d > 0.5 * 0.5 * (da + db):
            return None
        worst = max(worst, d)
    return worst


def build_chains(meta, frames):
    out: list[list[int]] = []
    for b in sorted(meta, key=lambda k: (meta[k]["first"], meta[k]["last"])):
        best = None
        for ci, ch in enumerate(out):
            a = ch[-1]
            gap = meta[b]["first"] - meta[a]["last"]
            if gap < -OVERLAP_MAX or gap > GAP_MAX:
                continue
            d = handoff(frames, a, b)
            if d is None:
                continue
            if best is None or d < best[0]:
                best = (d, ci)
        if best is None:
            out.append([b])
        else:
            out[best[1]].append(b)
    return out


def _ann(frame_dir, kp, sc, vid, w0, w1, width, height):
    k = kp.copy()
    k[:, :, 0] *= float(width)
    k[:, :, 1] *= float(height)
    return {"frame_dir": frame_dir, "total_frames": int(k.shape[0]),
            "label": 0, "label_known": False,
            "keypoint": k[None], "keypoint_score": sc[None],
            "img_shape": (int(height), int(width)),
            "original_shape": (int(height), int(width)),
            "video_id": vid, "start_frame": int(w0), "end_frame": int(w1)}


def group_series(kp, sc, tid, np_, members, w0, w1, max_gap):
    n = w1 - w0
    out_kp = np.zeros((n, 17, 2), dtype=np.float32)
    out_sc = np.zeros((n, 17), dtype=np.float32)
    present = np.zeros(n, dtype=bool)
    for i, t in enumerate(range(w0, w1)):
        for p in range(int(np_[t])):
            if int(tid[t, p]) in members:
                out_kp[i] = kp[t, p]
                out_sc[i] = sc[t, p]
                present[i] = True
                break
    # interpolate short gaps (same rule as wbuild._track_series)
    i = 0
    while i < n:
        if present[i]:
            i += 1
            continue
        j = i
        while j < n and not present[j]:
            j += 1
        if i > 0 and j < n and (j - i) <= max_gap:
            for k in range(i, j):
                a = (k - i + 1) / (j - i + 1)
                out_kp[k] = (1 - a) * out_kp[i - 1] + a * out_kp[j]
        i = j
    return out_kp, out_sc, present


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args()
    cfg = wbuild.load_config()
    gt = load_gt()
    pose_cfg = extract.load_config()
    pose_cfg.output_root = paths.KEYPOINTS
    pose_cfg.index_path = KEYPOINT_ROOT / "extraction_index.csv"

    windows: list[dict] = []
    samples: list[dict] = []
    anns: list[dict] = []
    n_stitched = 0

    for row in load_manifest():
        vid = row["video_id"]
        path = extract.npz_path_for(pose_cfg, row)
        if not path.exists():
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
            continue

        meta, frames = track_frames(tid, np_, bbox)
        chains = build_chains(meta, frames)
        members_of = {root: set(c) for c in chains for root in [min(c)]}
        n_stitched += sum(1 for c in chains if len(c) > 1)

        for w0 in range(0, n - win + 1, st):
            w1 = w0 + win
            wid = len(windows)
            windows.append({"window_id": f"w{wid:06d}", "video_id": vid,
                            "start": w0, "end": w1, "fps": fps})

            base = wbuild.extract_window(kp, sc, tid, np_, w0, w1, cfg, width, height)
            if base is not None and base["coverage"] >= cfg.min_coverage:
                anns.append(_ann(f"b{wid:06d}", base["keypoints"].astype(np.float32),
                                 base["scores"].astype(np.float32), vid, w0, w1,
                                 width, height))
                samples.append({"window": wid, "kind": "base", "ann": len(anns) - 1})

            for p in wbuild.extract_window_per_person(kp, sc, tid, np_, w0, w1,
                                                      cfg, width, height):
                anns.append(_ann(f"p{wid:06d}_k{p['track_id']:04d}",
                                 p["keypoints"].astype(np.float32),
                                 p["scores"].astype(np.float32), vid, w0, w1,
                                 width, height))
                samples.append({"window": wid, "kind": "person", "ann": len(anns) - 1})

            # --- C: B + ID-handoff stitching ---
            for root, members in members_of.items():
                gkp, gsc, present = group_series(kp, sc, tid, np_, members, w0, w1,
                                                 cfg.max_interp_gap_frames)
                cov = float(present.mean())
                if cov < cfg.min_coverage:
                    continue
                gkp = gkp.copy()
                gkp[:, :, 0] /= max(1, width)
                gkp[:, :, 1] /= max(1, height)
                idx = wbuild.resample_indices(w1 - w0, cfg.resample_frames)
                anns.append(_ann(f"s{wid:06d}_k{root:04d}", gkp[idx], gsc[idx],
                                 vid, w0, w1, width, height))
                samples.append({"window": wid, "kind": "stitch", "ann": len(anns) - 1,
                                "members": sorted(members)})

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pkl = OUT_DIR / "_samples.pkl"
    with open(pkl, "wb") as fh:
        pickle.dump({"split": {"test": [a["frame_dir"] for a in anns]},
                     "annotations": anns}, fh, protocol=4)
    print(f"windows={len(windows)} samples={len(anns)} chains_merged={n_stitched}",
          flush=True)

    import torch, yaml
    with open(RUN_DIR / "config_used.yaml", "r", encoding="utf-8-sig") as fh:
        mc = yaml.safe_load(fh)
    mc["data"]["ann_file"] = str(pkl)
    mc["data"]["num_workers"] = 0
    model = engine.build_recognizer(mc)
    with open(RUN_DIR / "best.pth", "rb") as fh:
        ck = torch.load(fh, map_location="cpu")
    model.load_state_dict(ck["model"])
    dev = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(dev)
    _, loader = engine.build_loader(mc, "test")
    _, probs = engine.predict(model, loader, dev)
    assert len(probs) == len(anns), f"{len(probs)} != {len(anns)}"

    agg = {"A_baseline": np.zeros(len(windows)),
           "B_per_person": np.zeros(len(windows)),
           "C_stitched": np.zeros(len(windows))}
    key = {"base": "A_baseline", "person": "B_per_person", "stitch": "C_stitched"}
    for s in samples:
        k = key[s["kind"]]
        v = float(probs[s["ann"]])
        if v > agg[k][s["window"]]:
            agg[k][s["window"]] = v

    by_video = defaultdict(list)
    for i, w in enumerate(windows):
        by_video[w["video_id"]].append(i)

    res = {}
    for name, wp in agg.items():
        seg_tot = seg_hit = 0
        pos_v = pos_h = neg_v = neg_fa = 0
        rows = []
        for vid, idxs in sorted(by_video.items()):
            if vid in EXCLUDE or vid not in gt:
                continue
            segs = gt[vid]["segments"]
            if segs:
                pos_v += 1
                any_hit = False
                for (s, e) in segs:
                    cand = [i for i in idxs
                            if max(0, min(windows[i]["end"], e + 1)
                                   - max(windows[i]["start"], s)) > 0]
                    best = max((wp[i] for i in cand), default=0.0)
                    seg_tot += 1
                    ok = best >= args.threshold
                    seg_hit += int(ok)
                    any_hit |= ok
                    rows.append({"vid": vid.split(":")[-1], "s": s, "e": e,
                                 "fps": gt[vid]["fps"], "p": best, "hit": ok,
                                 "n": len(cand)})
                pos_h += int(any_hit)
            else:
                neg_v += 1
                if any(wp[i] >= args.threshold for i in idxs):
                    neg_fa += 1
        res[name] = {"seg_tot": seg_tot, "seg_hit": seg_hit, "pos_v": pos_v,
                     "pos_h": pos_h, "neg_v": neg_v, "neg_fa": neg_fa, "rows": rows}

    lines = ["# 板端：轨迹换 ID 修复（D-042 候选）", "",
             f"- 交接判据（预注册）：gap ∈ [-{OVERLAP_MAX}, {GAP_MAX}] 帧，同现帧数 ≤ "
             f"{OVERLAP_MAX}，同现帧两框中心距 < 0.5×对角线均值",
             f"- 触发拼接的链：**{n_stitched}** 条",
             f"- 排除 170602/170618（域偏移）后的真值：{res['C_stitched']['seg_tot']} 段 / "
             f"{res['C_stitched']['pos_v']} 个有跌倒视频 / "
             f"{res['C_stitched']['neg_v']} 个 no_fall",
             f"- 模型 `runs/stgcnpp_v2/best.pth`（D-023 基线），阈值 {args.threshold}",
             "- **板端数据仅用于评估，未参与训练。**", "",
             "## 汇总", "",
             "| 策略 | 段召回 | 检出视频 | 无误报视频 |",
             "|---|---:|---:|---:|"]
    label = {"A_baseline": "A `max_presence`（D-018）",
             "B_per_person": "B 逐人取最大（D-040，现行）",
             "C_stitched": "C B + 换 ID 拼接（本候选）"}
    for k in ("A_baseline", "B_per_person", "C_stitched"):
        r = res[k]
        lines.append(f"| {label[k]} | {r['seg_hit']}/{r['seg_tot']} "
                     f"({r['seg_hit']/max(1,r['seg_tot']):.3f}) | "
                     f"{r['pos_h']}/{r['pos_v']} | "
                     f"{r['neg_v']-r['neg_fa']}/{r['neg_v']} |")

    lines += ["", "## 逐段概率（A → B → C）", "",
              "| 视频 | 段 (s) | 可用窗 | A | B | C | B→C |",
              "|---|---|---:|---:|---:|---:|:-:|"]
    bmap = {(r["vid"], r["s"]): r for r in res["B_per_person"]["rows"]}
    amap = {(r["vid"], r["s"]): r for r in res["A_baseline"]["rows"]}
    for r in res["C_stitched"]["rows"]:
        k = (r["vid"], r["s"])
        a, b = amap.get(k), bmap.get(k)
        delta = ""
        if b and (b["hit"] != r["hit"]):
            delta = "✅ 修复" if r["hit"] else "❌ 变差"
        lines.append(f"| {r['vid']} | {r['s']/r['fps']:.1f}–{(r['e']+1)/r['fps']:.1f} | "
                     f"{r['n']} | {a['p'] if a else 0:.3f} | {b['p'] if b else 0:.3f} | "
                     f"**{r['p']:.3f}** | {delta} |")

    lines += ["", "## 未修复段的原因", "",
              "- `172939@6.0 s`：t1→t3 的换 ID 已能合并，但段内 26 帧里仍有大半帧"
              "**YOLO 完全无框**（人蹲下后被遮挡）。合并后窗口最高覆盖仍 < 0.70，"
              "这是前端漏检，不是轨迹问题。",
              "- `172408@40.1 s`：t12→t13 换 ID 合并后窗口 [1170,1230) 覆盖率 0.50→0.767，"
              "**已过门**；但 40.50–41.87 s 有 41 帧完全无框（倒地后漏检），"
              "模型能否识别取决于 39.0–41.0 s 窗口里是否含足够的下落动作。",
              ""]
    report = OUT_DIR / "summary.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nreport: {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
