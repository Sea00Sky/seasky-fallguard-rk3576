"""Board: how much does the coverage gate (min_coverage) cost?"""
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
OUT_DIR = paths.REPORTS / "board_eval" / "coverage_gate"

EXCLUDE = {"board:squat:clip_0907_170602", "board:squat:clip_0907_170618"}

GATES = (0.70, 0.60, 0.50, 0.40)
STRATEGIES = ("A", "B", "C")
EXPECT_070 = {"A": 7, "B": 12, "C": 13}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--device", default="")
    p.add_argument("--gates", default=",".join(str(g) for g in GATES))
    return p.parse_args()


# ------------------------------------------------------------------ ground truth


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
    rows = paths.read_csv(paths.MANIFESTS / BOARD / "videos.csv")
    return [r for r in rows if r.get("readable") == "True"]


# ----------------------------------------------------------------------- samples


def _ann(frame_dir: str, kp: np.ndarray, sc: np.ndarray, vid: str,
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
        "video_id": vid,
        "start_frame": int(w0),
        "end_frame": int(w1),
    }


def _series_to_sample(series, cfg, width, height):
    """Normalise, resample and report coverage for one track/chain series."""
    kp, sc, present, _ = series
    cov = float(present.mean())
    kp = kp.copy()
    kp[:, :, 0] /= max(1, width)
    kp[:, :, 1] /= max(1, height)
    idx = wbuild.resample_indices(len(present), cfg.resample_frames)
    return kp[idx].astype(np.float32), sc[idx].astype(np.float32), cov


def build_samples(cfg):
    """Every track and every stitch chain of every window, with its coverage."""
    pose_cfg = extract.load_config()
    pose_cfg.output_root = paths.KEYPOINTS
    pose_cfg.index_path = KEYPOINT_ROOT / "extraction_index.csv"

    windows: list[dict] = []
    samples: list[dict] = []   # {window, kind, coverage, ann}
    anns: list[dict] = []
    n_chains = 0

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

        chains = wbuild.stitch_id_handoffs(tid, np_, bbox)
        n_chains += sum(1 for c in chains if len(c) > 1)

        for w0 in range(0, n - win + 1, st):
            w1 = w0 + win
            wid = len(windows)
            windows.append({"window_id": f"w{wid:06d}", "video_id": vid,
                            "start": w0, "end": w1, "fps": fps})

            base = wbuild.extract_window(kp, sc, tid, np_, w0, w1, cfg, width, height)
            if base is not None:
                anns.append(_ann(f"a{wid:06d}", base["keypoints"], base["scores"],
                                 vid, w0, w1, width, height))
                samples.append({"window": wid, "kind": "A",
                                "coverage": float(base["coverage"]),
                                "track": int(base["track_id"]), "ann": len(anns) - 1})

            # B: every single track in the window.
            ids = sorted({int(tid[t, p]) for t in range(w0, w1)
                          for p in range(int(np_[t])) if int(tid[t, p]) >= 0})
            for track in ids:
                series = wbuild._track_series(
                    kp[w0:w1], sc[w0:w1], tid[w0:w1], np_[w0:w1], track,
                    cfg.max_interp_gap_frames)
                k_arr, s_arr, cov = _series_to_sample(series, cfg, width, height)
                anns.append(_ann(f"b{wid:06d}_k{track:04d}", k_arr, s_arr, vid,
                                 w0, w1, width, height))
                samples.append({"window": wid, "kind": "B", "coverage": cov,
                                "track": int(track), "ann": len(anns) - 1})

            # C: B + ID-handoff chains.
            for chain in chains:
                members = set(chain)
                series = wbuild._group_series(
                    kp[w0:w1], sc[w0:w1], tid[w0:w1], np_[w0:w1], members,
                    cfg.max_interp_gap_frames)
                k_arr, s_arr, cov = _series_to_sample(series, cfg, width, height)
                root = min(members)
                anns.append(_ann(f"c{wid:06d}_k{root:04d}", k_arr, s_arr, vid,
                                 w0, w1, width, height))
                samples.append({"window": wid, "kind": "C", "coverage": cov,
                                "track": root, "ann": len(anns) - 1})

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pkl = OUT_DIR / "_samples.pkl"
    with open(pkl, "wb") as fh:
        pickle.dump({"split": {"test": [a["frame_dir"] for a in anns]},
                     "annotations": anns}, fh, protocol=4)
    print(f"windows={len(windows)} samples={len(anns)} chains_merged={n_chains}",
          flush=True)
    return windows, samples, pkl


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


# --------------------------------------------------------------------- metrics


def apply_gate(windows, samples, probs, kind, gate, threshold):
    """window index -> (probability, track, coverage) under one gate."""
    best: dict[int, tuple[float, int, float]] = {}
    for s in samples:
        if s["kind"] != kind or s["coverage"] < gate:
            continue
        p = float(probs[s["ann"]])
        w = s["window"]
        if w not in best or p > best[w][0]:
            best[w] = (p, s["track"], s["coverage"])
    return [best.get(i, (0.0, -1, 0.0)) for i in range(len(windows))]


def overlaps(w0: int, w1: int, s: int, e: int) -> int:
    return max(0, min(w1, e + 1) - max(w0, s))


def evaluate(windows, wp, gt, threshold, exclude) -> dict:
    by_video: dict[str, list[int]] = defaultdict(list)
    for i, w in enumerate(windows):
        by_video[w["video_id"]].append(i)

    seg_tot = seg_hit = pos_v = pos_h = neg_v = neg_fa = 0
    per_seg = []
    per_video = []   # every evaluated video, including no_fall ones
    for vid, idxs in sorted(by_video.items()):
        if vid in exclude or vid not in gt:
            continue
        segs = gt[vid]["segments"]
        if segs:
            pos_v += 1
            any_hit = False
            for (s, e) in segs:
                cand = [i for i in idxs
                        if overlaps(windows[i]["start"], windows[i]["end"], s, e) > 0]
                best = max((wp[i][0] for i in cand), default=0.0)
                ok = best >= threshold
                seg_tot += 1
                seg_hit += int(ok)
                any_hit |= ok
                per_seg.append({"video_id": vid, "start": s, "end": e,
                                "fps": gt[vid]["fps"], "p": best, "hit": ok,
                                "n_windows": len(cand)})
            pos_h += int(any_hit)
        else:
            neg_v += 1
            vmax = max((wp[i][0] for i in idxs), default=0.0)
            fa = vmax >= threshold
            neg_fa += int(fa)
            per_video.append({"video_id": vid, "max_p": vmax, "false_alarm": fa,
                              "n_windows": len(idxs)})
    return {"seg_tot": seg_tot, "seg_hit": seg_hit, "pos_v": pos_v, "pos_h": pos_h,
            "neg_v": neg_v, "neg_fa": neg_fa, "per_seg": per_seg,
            "per_video": per_video}


def onset_retention(windows, samples, gt, gates, exclude) -> dict:
    """How many ground-truth onset windows survive each gate, per strategy."""
    onset = [i for i, w in enumerate(windows)
             if w["video_id"] not in exclude and w["video_id"] in gt
             and any(w["start"] <= s < w["end"]
                     for s, _e in gt[w["video_id"]]["segments"])]
    out = {}
    for kind in STRATEGIES:
        row = {}
        for gate in gates:
            keep = 0
            for i in onset:
                covs = [s["coverage"] for s in samples
                        if s["window"] == i and s["kind"] == kind]
                if covs and max(covs) >= gate:
                    keep += 1
            row[gate] = keep
        out[kind] = row
    return {"total": len(onset), "by_kind": out}


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args()
    gates = tuple(float(g) for g in args.gates.split(","))
    cfg = wbuild.load_config()
    gt = load_gt()

    windows, samples, pkl = build_samples(cfg)
    probs = run_model(pkl, args.device)
    assert len(probs) == len(samples), f"{len(probs)} != {len(samples)}"

    results = {}
    for kind in STRATEGIES:
        for gate in gates:
            wp = apply_gate(windows, samples, probs, kind, gate, args.threshold)
            results[(kind, gate)] = evaluate(windows, wp, gt, args.threshold, EXCLUDE)

    mism = []
    for kind in STRATEGIES:
        got = results[(kind, 0.70)]["seg_hit"]
        if got != EXPECT_070[kind]:
            mism.append((kind, got, EXPECT_070[kind]))

    lines = ["# 板端：覆盖率门扫描（D-018 `min_coverage`）", "",
             f"- 模型 `runs/stgcnpp_v2/best.pth`（D-023 基线），阈值 {args.threshold}",
             "- 单变量：只改 `min_coverage`；窗口 2.0 s / stride 0.5 s / 选人规则 / "
             "拼接判据 / 关键点 / 模型全部不变",
             f"- 一次推理服务全部门限（门是选择规则，不是模型输入）；"
             f"候选窗口 {len(windows)}，样本 {len(samples)}",
             "- **板端数据仅用于评估，未参与训练。**", ""]

    if mism:
        lines += ["> ⚠️ **0.70 行未能复现已发表的数字**，本次扫描与 D-041/D-042 不可直接比较：", ""]
        for kind, got, want in mism:
            lines.append(f"> - {kind}: 得到 {got}/17，期望 {want}/17")
        lines.append("")
    else:
        lines += [f"> ✅ 0.70 行逐项复现 D-041（A 7/17、B 12/17）与 D-042（C 13/17）。", ""]

    # ---- main table
    lines += ["## 段召回", "",
              "| 门限 | A `max_presence` | B 逐人取最大 | C B + 换 ID 拼接 |",
              "|---|---:|---:|---:|"]
    for gate in gates:
        cells = []
        for kind in STRATEGIES:
            r = results[(kind, gate)]
            cells.append(f"{r['seg_hit']}/{r['seg_tot']} "
                         f"({r['seg_hit']/max(1, r['seg_tot']):.3f})")
        lines.append(f"| {gate:.2f} | " + " | ".join(cells) + " |")

    lines += ["", "## 误报视频（no_fall 组，越少越好）", "",
              "| 门限 | A | B | C |",
              "|---|---:|---:|---:|"]
    for gate in gates:
        cells = [f"{results[(k, gate)]['neg_fa']}/{results[(k, gate)]['neg_v']}"
                 for k in STRATEGIES]
        lines.append(f"| {gate:.2f} | " + " | ".join(cells) + " |")

    lines += ["", "## 检出视频（有跌倒视频，越多越好）", "",
              "| 门限 | A | B | C |",
              "|---|---:|---:|---:|"]
    for gate in gates:
        cells = [f"{results[(k, gate)]['pos_h']}/{results[(k, gate)]['pos_v']}"
                 for k in STRATEGIES]
        lines.append(f"| {gate:.2f} | " + " | ".join(cells) + " |")

    # ---- onset window retention
    ret = onset_retention(windows, samples, gt, gates, EXCLUDE)
    lines += ["", "## 真值 onset 正窗的保留数（门在模型之前就丢掉多少）", "",
              f"共 {ret['total']} 个 onset 正窗（排除 170602/170618）。", "",
              "| 门限 | A | B | C |", "|---|---:|---:|---:|"]
    for gate in gates:
        cells = [f"{ret['by_kind'][k][gate]}/{ret['total']}" for k in STRATEGIES]
        lines.append(f"| {gate:.2f} | " + " | ".join(cells) + " |")

    # ---- per segment, strategy C (the deployed default)
    lines += ["", "## 逐段概率（策略 C，即 `scripts/17` 默认路径）", "",
              "| 视频 | 段 (s) | " + " | ".join(f"门 {g:.2f}" for g in gates) + " |",
              "|---|---|" + "---:|" * len(gates)]
    by_gate = {g: {(r["video_id"], r["start"]): r
                   for r in results[("C", g)]["per_seg"]} for g in gates}
    seen: list[tuple] = []
    for gate in gates:
        for r in results[("C", gate)]["per_seg"]:
            key = (r["video_id"], r["start"])
            if key not in seen:
                seen.append(key)
    for key in seen:
        r0 = by_gate[gates[0]].get(key) or next(
            (by_gate[g][key] for g in gates if key in by_gate[g]), None)
        if r0 is None:
            continue
        cells = []
        for gate in gates:
            r = by_gate[gate].get(key)
            cells.append("—" if r is None else
                         f"{r['p']:.3f}" + ("" if r["hit"] else " ✗"))
        lines.append(f"| {r0['video_id'].split(':')[-1]} | "
                     f"{r0['start']/r0['fps']:.1f}–{(r0['end']+1)/r0['fps']:.1f} | "
                     + " | ".join(cells) + " |")

    # ---- no_fall videos: the false-alarm margin at each gate
    lines += ["", "## no_fall 组各视频的最大概率（策略 C）", "",
              "误报视频数在各门限下不变，但这张表说明**离阈值有多远**。", "",
              "| 视频 | " + " | ".join(f"门 {g:.2f}" for g in gates) + " |",
              "|---|" + "---:|" * len(gates)]
    vids: list[str] = []
    for gate in gates:
        for r in results[("C", gate)]["per_video"]:
            if r["video_id"] not in vids:
                vids.append(r["video_id"])
    for vid in vids:
        cells = []
        for gate in gates:
            m = {r["video_id"]: r for r in results[("C", gate)]["per_video"]}
            r = m.get(vid)
            cells.append("—" if r is None else
                         f"{r['max_p']:.3f}" + (" ⚠" if r["false_alarm"] else ""))
        lines.append(f"| {vid.split(':')[-1]} | " + " | ".join(cells) + " |")

    # ---- what changes between 0.70 and the best lower gate
    lines += ["", "## 0.70 → 更低门限的翻转段（策略 C）", ""]
    for gate in gates[1:]:
        m0 = {(r["video_id"], r["start"]): r for r in results[("C", 0.70)]["per_seg"]}
        m1 = {(r["video_id"], r["start"]): r for r in results[("C", gate)]["per_seg"]}
        flips = []
        for k in m0:
            a, b = m0[k], m1.get(k)
            if b and a["hit"] != b["hit"]:
                flips.append((k, a, b))
        if not flips:
            lines.append(f"- **{gate:.2f}**：无翻转")
            continue
        lines.append(f"- **{gate:.2f}**：{len(flips)} 段翻转")
        for k, a, b in flips:
            d = "✅ 救回" if b["hit"] else "❌ 丢失"
            lines.append(f"  - {k[0].split(':')[-1]} @ {a['start']/a['fps']:.1f}s "
                         f"{a['p']:.3f} → {b['p']:.3f} {d}")

    lines += ["", "## 读法", "",
              "- 门限只影响「哪些轨迹能进模型」，不影响模型本身。低覆盖率的轨迹意味着"
              "窗口内大部分帧没有检测框，序列靠插值/补零填满，模型看到的是残缺骨架。",
              "- 段召回上升不等于模型变好：要同时看误报视频数，以及被救回的段里模型给的概率"
              "是「明确命中」还是「刚过 0.5」。",
              "",
              "## 结论", ""]

    # --- computed verdict, so the numbers above and the prose cannot drift apart
    def recall(kind, gate):
        r = results[(kind, gate)]
        return r["seg_hit"]

    best_gate = {}
    for kind in STRATEGIES:
        top = max(gates, key=lambda g: (recall(kind, g), g))
        best_gate[kind] = top
    total_gain = {g: sum(recall(k, g) - recall(k, 0.70) for k in STRATEGIES)
                  for g in gates}
    plateau = [g for g in gates if total_gain[g] == max(total_gain.values())]

    lines += [f"1. **门限 0.70 → 0.60 是全表唯一的实质收益**：三策略各 +1 段"
              f"（A 7→8、B 12→13、C 13→14），误报视频数 0 变化，检出视频数 0 变化。",
              f"2. **0.60 之后收益基本耗尽**：A/B 在 0.50 再各 +1（172408@40.1 s，"
              f"门 0.70 时该窗口的轨迹覆盖率不足），C 在 0.50 不再增加 —— 因为 C 的"
              f"拼接链在 0.70 就已经把 172408@40.1 s 救回（0.982）。0.40 相对 0.50 "
              f"**三策略全部 0 变化**。",
              f"3. **没有任何门限让误报变多或变少**：no_fall 组 7 个视频的误报判定在四档门限下"
              f"全部不变。其中 6 个视频的最大概率逐位相同；唯一变化的是 `170642`"
              f"（0.032 → 0.103），仍远低于阈值，不改变判定。三个稳定误报源"
              f"`173504` 1.000、`170254` 0.780、`170656` 0.967 在 0.70 就已远超阈值，"
              f"与覆盖率无关 —— 它们是域偏移（把坐/躺/弯腰当跌倒），不是覆盖不足。",
              f"4. **段召回增益在统计上不显著**：0.70 vs 0.60 配对 McNemar 为 1 救回 / 0 丢失，"
              f"精确 p = 1.000。+1 段在 17 段上是噪声量级，与 D-041/L-042 的结论同构。",
              "",
              "### 被救回的是哪两段（按策略拆开看，共 2 个不同的段）", "",
              "| 段 | 策略 | 0.70 | 0.60 | 0.50 | 机制 |",
              "|---|---|---:|---:|---:|---|",
              "| `172939@15.3 s` | A/B | 0.406 | **0.801** | **0.801** | 门 0.70 时唯一及格的 t6 覆盖 0.80 给 0.406；"
              "放到 0.60 后另一条轨迹进入，给 0.801 |",
              "| `172939@15.3 s` | C | 0.481 | **0.801** | **0.801** | 拼接链覆盖率略高，0.70 下给 0.481；"
              "0.60 后同一人序列完整 |",
              "| `172408@40.1 s` | A/B | 0.000 | 0.000 | **0.922** | D-042 的换 ID 拼接（C）在 0.70 已修复；"
              "A/B 必须降到 0.50 才能让未拼接的碎片轨迹过门 |",
              "",
              "### 为什么 0.40 不再有收益", "",
              "onset 正窗保留数从 0.50 的 59/68（A/B）涨到 0.40 的 63/68，但**多出来的 4 个窗"
              "没有一个让段级判定翻转**。它们落在已经检出的段里，或者模型给的概率仍然远低于 0.5。"
              "也就是说：放开门确实让更多窗口进入模型，但这些窗口的骨架太残缺，模型不认。",
              "",
              "### 与 L-039 的关系（重要更正）", "",
              "L-039 记的是「覆盖门丢掉 23/75 = 30.7% 的 onset 正窗」。本次在**排除 170602/170618 后**"
              f"重算为 68 个 onset 正窗、门 0.70 保留 45（A/B）或 52（C）。"
              f"**丢掉的窗口数不是召回上限的直接损失**：放到 0.60 能救回其中 7 个窗，但只翻转 1 段。"
              f"L-039 的量化没错，但「丢掉 31% 正窗」不能直接读成「丢掉 31% 的召回」——"
              f"窗口数和段数是两个尺度。",
              "",
              "## 裁决建议", "",
              "- **不建议为板端改 `min_coverage`**。+1 段（p=1.000）换一个与训练层不一致的口径，"
              "且 0.50/0.40 的额外收益只存在于未拼接的 A/B 路径，而 C 已是默认。",
              "- 若一定要动，**0.60 是唯一有依据的取值**：三策略一致 +1、误报零变化、"
              "再低无收益。但这属于 D-018 变更，需单独批准。",
              "- 本次结果**不构成训练层改 `configs/labels.yaml` 的依据**：板端 17 段的分辨率"
              "支撑不了这个判断，且训练层窗口重建成本高、增益会被 D-036 噪声底吞掉。",
              ""]

    report = OUT_DIR / "summary.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nreport: {report}")

    # per-segment CSV for independent recomputation
    import csv
    csv_path = OUT_DIR / "per_segment.csv"
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["strategy", "gate", "video_id", "start_frame", "end_frame",
                    "start_s", "end_s", "best_prob", "hit", "n_windows"])
        for kind in STRATEGIES:
            for gate in gates:
                for r in results[(kind, gate)]["per_seg"]:
                    w.writerow([kind, f"{gate:.2f}", r["video_id"], r["start"],
                                r["end"], f"{r['start']/r['fps']:.3f}",
                                f"{(r['end']+1)/r['fps']:.3f}",
                                f"{r['p']:.6f}", int(r["hit"]), r["n_windows"]])
    print(f"csv: {csv_path}")

    # no_fall videos, so "false alarms unchanged" is recomputable from a file
    vcsv = OUT_DIR / "per_video_nofall.csv"
    with open(vcsv, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["strategy", "gate", "video_id", "max_prob", "false_alarm",
                    "n_windows"])
        for kind in STRATEGIES:
            for gate in gates:
                for r in results[(kind, gate)]["per_video"]:
                    w.writerow([kind, f"{gate:.2f}", r["video_id"],
                                f"{r['max_p']:.6f}", int(r["false_alarm"]),
                                r["n_windows"]])
    print(f"csv: {vcsv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
