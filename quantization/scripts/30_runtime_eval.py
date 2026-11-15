"""Phase 3 -- replay the board clips through the streaming runtime (G3)."""
from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT.parent / "YOLOV26n-pose+ST-GCN++"
# This script runs from `scripts/`, so the project root is not on sys.path.
# It must be, for `import src.runtime`. Note the source project's package is
# also called `src`; this script never imports it (see `_offline_chains`), so
# there is no collision.
sys.path.insert(0, str(PROJECT_ROOT))


def _load_own(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, PROJECT_ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


paths = _load_own("deploy_paths", "src/deploy/paths.py")
spec_mod = _load_own("deploy_spec", "src/deploy/spec.py")

import src.runtime as rt # noqa: E402 (this project's package)

TOL_PROB = 1e-3
TOL_PIPELINE = 1e-4
EXPECT_SEGMENTS = {"A": 7, "B": 12, "C": 13}
EXPECT_VIDEOS = {"pos_h": 8, "pos_v": 9, "neg_fa": 3, "neg_v": 7}
EXCLUDE = {"board:squat:clip_0907_170602", "board:squat:clip_0907_170618"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--limit", type=int, default=0, help="first N videos only")
    p.add_argument("--no-stitch", action="store_true",
                   help="disable D-042 chaining (reproduces strategy B)")
    return p.parse_args()


def board_rows(limit: int = 0) -> list[dict]:
    rows = [r for r in paths.read_csv(paths.SOURCE_MANIFEST_BOARD)
            if r.get("readable") == "True"]
    return rows[:limit] if limit else rows


def load_reference() -> dict:
    d = np.load(paths.REFERENCE / "board_candidates.npz", allow_pickle=False)
    windows = [{"window_id": str(d["window_id"][i]),
                "video_id": str(d["video_id"][i]),
                "start": int(d["start"][i]), "end": int(d["end"][i]),
                "fps": float(d["fps"][i])}
               for i in range(len(d["window_id"]))]
    samples = [{"window": int(d["sample_window"][i]),
                "kind": str(d["sample_kind"][i]),
                "track": int(d["sample_track"][i]),
                "members": str(d["sample_members"][i]),
                "coverage": float(d["sample_coverage"][i]),
                "prob": float(d["sample_prob"][i])}
               for i in range(len(d["sample_window"]))]
    return {"windows": windows, "samples": samples}


def read_frames(path: Path):
    """Yield BGR frames one at a time -- the board's view."""
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(path)
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                return
            yield frame
    finally:
        cap.release()


def load_gt() -> dict[str, dict]:
    out: dict[str, dict] = {}
    for r in paths.read_csv(paths.SOURCE_GT_BOARD):
        vid = r["video_id"]
        rec = out.setdefault(vid, {"fps": float(r["fps"]), "segments": []})
        if r.get("status") == "no_fall" or not r.get("start_frame"):
            continue
        rec["segments"].append((int(r["start_frame"]), int(r["end_frame"])))
    return out


def load_source_reference(strategy: str, gate: float) -> tuple[dict, dict]:
    """The source project's own published numbers, for G6."""
    per_seg = {}
    for r in paths.read_csv(paths.SOURCE_REF_PER_SEGMENT):
        if r["strategy"] != strategy or float(r["gate"]) != gate:
            continue
        per_seg[(r["video_id"], int(r["start_frame"]), int(r["end_frame"]))] = {
            "prob": float(r["best_prob"]), "hit": int(r["hit"]),
            "n_windows": int(r["n_windows"]),
        }
    nofall = {}
    for r in paths.read_csv(paths.SOURCE_REF_PER_VIDEO_NOFALL):
        if r["strategy"] != strategy or float(r["gate"]) != gate:
            continue
        nofall[r["video_id"]] = {"max_prob": float(r["max_prob"]),
                                 "false_alarm": int(r["false_alarm"]),
                                 "n_windows": int(r["n_windows"])}
    return per_seg, nofall


def evaluate_no_fall(windows, probs_by_window, gt, exclude) -> list[dict]:
    """Per-video max probability on the no_fall clips (G6 criterion 3)."""
    from collections import defaultdict

    by_video: dict[str, list[int]] = defaultdict(list)
    for i, w in enumerate(windows):
        by_video[w["video_id"]].append(i)

    out = []
    for vid, idxs in sorted(by_video.items()):
        if vid in exclude or vid not in gt or gt[vid]["segments"]:
            continue
        out.append({"video_id": vid,
                    "max_prob": max((probs_by_window[i] for i in idxs), default=0.0),
                    "n_windows": len(idxs)})
    return out


def evaluate_segments(windows, probs_by_window, gt, exclude) -> dict:
    from collections import defaultdict

    by_video: dict[str, list[int]] = defaultdict(list)
    for i, w in enumerate(windows):
        by_video[w["video_id"]].append(i)

    seg_tot = seg_hit = pos_v = pos_h = neg_v = neg_fa = 0
    per_seg = []
    for vid, idxs in sorted(by_video.items()):
        if vid in exclude or vid not in gt:
            continue
        segs = gt[vid]["segments"]
        if segs:
            pos_v += 1
            any_hit = False
            for s, e in segs:
                cand = [i for i in idxs
                        if spec_mod.overlap(windows[i]["start"], windows[i]["end"], s, e) > 0]
                best = max((probs_by_window[i] for i in cand), default=0.0)
                ok = best >= spec_mod.THRESHOLD
                seg_tot += 1
                seg_hit += int(ok)
                any_hit |= ok
                per_seg.append({"video_id": vid, "start": s, "end": e,
                                "p": best, "hit": ok, "n_windows": len(cand)})
            pos_h += int(any_hit)
        else:
            neg_v += 1
            vmax = max((probs_by_window[i] for i in idxs), default=0.0)
            neg_fa += int(vmax >= spec_mod.THRESHOLD)
    return {"seg_tot": seg_tot, "seg_hit": seg_hit, "pos_v": pos_v, "pos_h": pos_h,
            "neg_v": neg_v, "neg_fa": neg_fa, "per_seg": per_seg}


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args()
    paths.REPORTS.mkdir(parents=True, exist_ok=True)

    ref = load_reference()
    gt = load_gt()
    rows = board_rows(args.limit)

    # `--no-stitch` reproduces strategy B, so the reference column must switch
    # from C to B too -- comparing B against C would report a difference that
    # is purely the stitching rule, not the runtime.
    strategy = "B" if args.no_stitch else "C"
    cfg = rt.PipelineConfig(
        yolo_onnx=paths.MODELS / "yolo26n_pose_480x640.onnx",
        stgcn_onnx=paths.MODELS / "stgcnpp_v2.onnx",
        stitch=not args.no_stitch,
    )
    print(f"strategy: {strategy} (stitch={cfg.stitch}), "
          f"expecting {EXPECT_SEGMENTS[strategy]}/17")
    for p in (cfg.yolo_onnx, cfg.stgcn_onnx):
        if not p.exists():
            raise SystemExit(f"missing {p}; run scripts/11 and scripts/12 first")

    # ---- replay every video, frame by frame
    all_windows: list[dict] = []
    all_samples: list[dict] = []
    all_probs: list[float] = []
    chain_report: list[dict] = []
    t0 = time.perf_counter()
    total_frames = 0

    for i, row in enumerate(rows, 1):
        vid = row["video_id"]
        pipe = rt.Pipeline(cfg)
        video_path = paths.SOURCE_RAW_BOARD / row["relative_path"]
        result = pipe.run_video(vid, read_frames(video_path))
        total_frames += result.n_frames

        # window ids are global; offset this video's into the global sequence
        offset = len(all_windows)
        for w in result.windows:
            all_windows.append({"window_id": f"w{offset + w.window_id:06d}",
                                "video_id": vid, "start": w.start, "end": w.end,
                                "fps": float(row["fps"])})

        # Per-sample table from the runtime's own bookkeeping, so every
        # probability stays attached to the window and person it came from.
        for wid, track, cov, p in pipe._scored:
            all_samples.append({"window": offset + wid, "kind": "C",
                                "track": int(track), "coverage": float(cov),
                                "prob": float(p)})
            all_probs.append(float(p))

        offline = _offline_chains(ref, vid) if not args.no_stitch else []
        online = [c for c in result.chains if len(c) > 1]
        chain_report.append({
            "video_id": vid,
            "n_frames": result.n_frames,
            "offline_chains": [list(c) for c in offline],
            "online_chains": [list(c) for c in online],
            "match": _chains_match(offline, online),
        })
        print(f"[{i}/{len(rows)}] {vid} frames={result.n_frames} "
              f"windows={len(result.windows)} scored={result.n_scored} "
              f"observed={result.n_observed} "
              f"chains(offline={len(offline)}, online={len(online)})",
              flush=True)

    elapsed = time.perf_counter() - t0
    print(f"\nreplayed {total_frames} frames in {elapsed:.1f}s "
          f"({total_frames/elapsed:.1f} fps)")

    # ---- compare against the reference
    cmp_windows = _compare_windows(all_windows, ref["windows"])
    wp = _collapse(all_windows, all_samples, all_probs)
    ref_wp = _ref_collapse(ref["windows"], ref["samples"], strategy)
    cmp_prob = _compare_probs(wp, ref_wp)
    seg = evaluate_segments(all_windows, wp, gt, EXCLUDE)
    nofall = evaluate_no_fall(all_windows, wp, gt, EXCLUDE)
    ref_seg, ref_nofall = load_source_reference(strategy, spec_mod.MIN_COVERAGE)

    # ---- G6-style comparisons against the source project's published tables.
    # These live here (not in a separate script) because they are the same
    # replay; splitting them would run the pipeline twice for one number.
    g6_seg = _compare_per_segment(seg["per_seg"], ref_seg)
    g6_nofall = _compare_no_fall(nofall, ref_nofall)

    print(f"\nwindows : {cmp_windows['n_mine']} mine vs {cmp_windows['n_ref']} ref, "
          f"missing {cmp_windows['n_missing']}, extra {cmp_windows['n_extra']}")
    print(f"prob    : max|dp| = {cmp_prob['max_abs_dp']:.3e} "
          f"on {cmp_prob['n_compared']} windows, "
          f"{cmp_prob['n_over_tol']} over {TOL_PROB:g} (reported, not gated)")
    print(f"segments: {seg['seg_hit']}/{seg['seg_tot']}, "
          f"videos {seg['pos_h']}/{seg['pos_v']}, "
          f"false alarms {seg['neg_fa']}/{seg['neg_v']}")
    print(f"G6 per-segment verdicts: {g6_seg['n_hit_match']}/{g6_seg['n']} agree, "
          f"max|dp| = {g6_seg['max_abs_dp']:.4f} (reported, not gated)")
    print(f"G6 no_fall per-video   : {g6_nofall['n']} videos, "
          f"max|dp| = {g6_nofall['max_abs_dp']:.3e}, "
          f"false-alarm verdicts {g6_nofall['n_verdict_match']}/{g6_nofall['n']} agree")
    n_chain_mismatch = sum(1 for c in chain_report if not c["match"])
    print(f"chains  : {n_chain_mismatch}/{len(chain_report)} videos differ "
          f"between online and offline (design property, D-106)")

    pipeline_ok = _pipeline_correctness()
    ok = (cmp_windows["n_missing"] == 0 and cmp_windows["n_extra"] == 0
          and cmp_windows["order_identical"]
          and seg["seg_hit"] == EXPECT_SEGMENTS[strategy]
          and seg["pos_h"] == EXPECT_VIDEOS["pos_h"]
          and seg["pos_v"] == EXPECT_VIDEOS["pos_v"]
          and seg["neg_fa"] == EXPECT_VIDEOS["neg_fa"]
          and seg["neg_v"] == EXPECT_VIDEOS["neg_v"]
          and pipeline_ok["ok"])
    if args.limit:
        # A partial replay cannot meet whole-dataset thresholds; it is a smoke
        # test, not a gate run. Say so rather than reporting a spurious failure.
        ok = False
        print(f"\nNOTE: --limit {args.limit} is a smoke test; "
              f"the gate verdict is only meaningful on the full 19 videos.")

    # A `--limit` run must not overwrite the full-dataset artefacts -- an
    # earlier smoke test silently replaced `g3_runtime.json` with 3-video
    # numbers. Partial runs get their own suffix.
    suffix = f"_limit{args.limit}" if args.limit else ""
    paths.write_json(paths.REPORTS / f"g3_runtime{suffix}.json", {
        "n_videos": len(rows), "n_frames": total_frames,
        "seconds": round(elapsed, 2), "fps": round(total_frames / elapsed, 2),
        "windows": cmp_windows, "probabilities": cmp_prob,
        "segments": {k: seg[k] for k in ("seg_tot", "seg_hit", "pos_v",
                                         "pos_h", "neg_v", "neg_fa")},
        "g6_per_segment": g6_seg,
        "g6_no_fall": g6_nofall,
        "chains": chain_report,
        "n_chain_mismatch": n_chain_mismatch,
        # The parameters the run actually used. Without these, the JSON alone
        # cannot tell you which gate/threshold produced its numbers -- they were
        # previously only inferable from prose in the .md.
        "params": {
            "strategy": strategy,
            "stitch": bool(cfg.stitch),
            "min_coverage": float(spec_mod.MIN_COVERAGE),
            "threshold": float(spec_mod.THRESHOLD),
        },
        "tolerance_prob": TOL_PROB,
        "criteria_ruled_D114": {
            "windows_915_of_915": cmp_windows["n_missing"] == 0
                                  and cmp_windows["n_extra"] == 0
                                  and cmp_windows["order_identical"],
            "segment_recall": f"{seg['seg_hit']}/{seg['seg_tot']}",
            "videos": f"{seg['pos_h']}/{seg['pos_v']}",
            "false_alarms": f"{seg['neg_fa']}/{seg['neg_v']}",
            "pipeline_correctness": pipeline_ok,
        },
        "deleted_criteria": ["per_window_max_abs_dp_le_1e-3",
                             "same_track_selected_per_window"],
        "expected_segments": EXPECT_SEGMENTS[strategy],
        "strategy": strategy,
        "passed": bool(ok),
    })
    paths.write_csv(paths.REPORTS / f"g3_per_segment{suffix}.csv",
                    [{"video_id": s["video_id"], "start": s["start"],
                      "end": s["end"], "prob_fall": f"{s['p']:.6f}",
                      "hit": int(s["hit"]), "n_windows": s["n_windows"]}
                     for s in seg["per_seg"]])

    _write_report(cmp_windows, cmp_prob, seg, chain_report, rows, elapsed,
                  total_frames, ok, strategy, pipeline_ok, g6_seg, g6_nofall,
                  suffix)
    return 0 if ok else 1


# ------------------------------------------------------------------- helpers


def _pipeline_correctness() -> dict:
    """Read `scripts/31`'s strategy B artefact; never silently pass."""
    path = paths.REPORTS / "g3_pipeline_correctness_B.json"
    if not path.exists():
        return {"ok": False, "reason": f"missing {path.name}; run scripts/31 "
                                       f"first (it is a G3 criterion now)"}
    import json

    d = json.loads(path.read_text(encoding="utf-8"))
    return {"ok": bool(d.get("passed") is True),
            "strategy": d.get("strategy"),
            "max_abs_dp": d.get("probability", {}).get("max_abs_dp"),
            "n_pairs": d.get("probability", {}).get("n_pairs"),
            "source": path.name}


def _compare_per_segment(per_seg: list[dict], ref: dict) -> dict:
    """G6 criterion 2, under the ruled tolerance: verdicts gated, values reported."""
    rows, n_hit_match = [], 0
    for s in per_seg:
        key = (s["video_id"], s["start"], s["end"])
        r = ref.get(key)
        if r is None:
            continue
        hit_match = int(s["hit"]) == r["hit"]
        n_hit_match += int(hit_match)
        rows.append({"video_id": s["video_id"], "start": s["start"], "end": s["end"],
                     "ref_prob": round(r["prob"], 6),
                     "runtime_prob": round(s["p"], 6),
                     "abs_dp": round(abs(s["p"] - r["prob"]), 6),
                     "ref_hit": r["hit"], "runtime_hit": int(s["hit"]),
                     "hit_match": hit_match})
    d = [r["abs_dp"] for r in rows]
    return {"n": len(rows), "n_reference_rows": len(ref),
            "n_missing_from_reference": len(ref) - len(rows),
            "n_hit_match": n_hit_match,
            "max_abs_dp": max(d) if d else 0.0,
            "mean_abs_dp": sum(d) / len(d) if d else 0.0,
            "tolerance": "verdicts must match; probabilities reported, no gate",
            "per_segment": rows}


def _compare_no_fall(mine: list[dict], ref: dict) -> dict:
    """G6 criterion 3. Same ruled tolerance as criterion 2."""
    rows, n_verdict = 0, 0
    for m in mine:
        r = ref.get(m["video_id"])
        if r is None:
            continue
        verdict_match = int(m["max_prob"] >= spec_mod.THRESHOLD) == r["false_alarm"]
        n_verdict += int(verdict_match)
        rows += 1
        m.update({"ref_max_prob": r["max_prob"], "ref_false_alarm": r["false_alarm"],
                  "abs_dp": abs(m["max_prob"] - r["max_prob"]),
                  "verdict_match": verdict_match})
    d = [m["abs_dp"] for m in mine if "abs_dp" in m]
    return {"n": rows, "n_reference_rows": len(ref),
            "n_verdict_match": n_verdict,
            "max_abs_dp": max(d) if d else 0.0,
            "mean_abs_dp": sum(d) / len(d) if d else 0.0,
            "tolerance": "verdicts must match; probabilities reported, no gate",
            "per_video": mine}


def _offline_chains(ref: dict, video_id: str) -> list[list[int]]:
    """The source project's chains for one video."""
    chains: set[tuple[int, ...]] = set()
    for s in ref["samples"]:
        if s["kind"] != "C":
            continue
        if ref["windows"][s["window"]]["video_id"] != video_id:
            continue
        members = tuple(sorted(int(x) for x in s["members"].split("|") if x))
        if len(members) > 1:
            chains.add(members)
    return [list(c) for c in sorted(chains)]


def _chains_match(offline: list[list[int]], online: list[list[int]]) -> bool:
    return {tuple(sorted(c)) for c in offline} == {tuple(sorted(c)) for c in online}


def _compare_windows(mine: list[dict], ref: list[dict]) -> dict:
    key = lambda w: (w["video_id"], w["start"], w["end"])
    mine_keys = [key(w) for w in mine]
    ref_keys = [key(w) for w in ref]
    mine_set, ref_set = set(mine_keys), set(ref_keys)
    return {
        "n_mine": len(mine), "n_ref": len(ref),
        "n_missing": len(ref_set - mine_set), "n_extra": len(mine_set - ref_set),
        "order_identical": mine_keys == ref_keys,
        "first_missing": sorted(ref_set - mine_set)[:5],
        "first_extra": sorted(mine_set - ref_set)[:5],
    }


def _collapse(windows: list[dict], samples: list[dict], probs: list[float]) -> list[float]:
    best: dict[int, float] = {}
    for s, p in zip(samples, probs):
        if s["coverage"] < spec_mod.MIN_COVERAGE:
            continue
        w = s["window"]
        if w not in best or p > best[w]:
            best[w] = float(p)
    return [best.get(i, 0.0) for i in range(len(windows))]


def _ref_collapse(windows: list[dict], samples: list[dict], kind: str) -> list[float]:
    best: dict[int, float] = {}
    for s in samples:
        if s["kind"] != kind or s["coverage"] < spec_mod.MIN_COVERAGE:
            continue
        w = s["window"]
        if w not in best or s["prob"] > best[w]:
            best[w] = s["prob"]
    return [best.get(i, 0.0) for i in range(len(windows))]


def _compare_probs(mine: list[float], ref: list[float]) -> dict:
    n = min(len(mine), len(ref))
    a = np.asarray(mine[:n]); b = np.asarray(ref[:n])
    d = np.abs(a - b)
    flips = int(((a >= spec_mod.THRESHOLD) != (b >= spec_mod.THRESHOLD)).sum())
    return {"n_compared": n, "max_abs_dp": float(d.max()) if n else 0.0,
            "mean_abs_dp": float(d.mean()) if n else 0.0,
            "n_over_tol": int((d > TOL_PROB).sum()),
            "threshold_flips": flips}


def _write_report(cmp_windows, cmp_prob, seg, chain_report, rows, elapsed,
                  total_frames, ok, strategy, pipeline_ok, g6_seg,
                  g6_nofall, suffix: str = "") -> None:
    def mark(good: bool) -> str:
        return "✅" if good else "❌"

    n_chain_mismatch = sum(1 for c in chain_report if not c["match"])
    pc = pipeline_ok
    lines = ["# G3 — 流式 runtime（ONNX）对账", "",
             f"- 视频 {len(rows)} 个，帧 {total_frames}，耗时 {elapsed:.1f}s "
             f"（{total_frames/elapsed:.1f} fps，CPU）",
             "- 逐帧喂入，状态只有：tracker、每 track 的 60 帧环、增量拼接链",
             "- 参考：`reports/reference/board_candidates.csv`（Phase 0 G0 冻结，"
             "由源项目自己的 `src.window.build` 产出）",
             "- **判据按 2026-10-22 裁决（D-114）**：两条不可达的逐值判据已删除，"
             "由「管线正确性」替代。理由见 `quantization/README.md`。",
             "", "## 判据（D-114 裁决口径）", "",
             "| 判据 | 实测 | 门槛 | 结果 |", "|---|---:|---:|---|",
             f"| 候选窗口数 | {cmp_windows['n_mine']} | ={cmp_windows['n_ref']} "
             f"| {mark(cmp_windows['n_mine'] == cmp_windows['n_ref'])} |",
             f"| 缺失 / 多余窗口 | {cmp_windows['n_missing']} / "
             f"{cmp_windows['n_extra']} | 0 / 0 "
             f"| {mark(cmp_windows['n_missing'] == 0 and cmp_windows['n_extra'] == 0)} |",
             f"| 窗口顺序完全一致 | {'是' if cmp_windows['order_identical'] else '否'} "
             f"| 是 | {mark(cmp_windows['order_identical'])} |",
             f"| 段召回 | {seg['seg_hit']}/{seg['seg_tot']} "
             f"| {EXPECT_SEGMENTS[strategy]}/17 | "
             f"{mark(seg['seg_hit'] == EXPECT_SEGMENTS[strategy])} |",
             f"| 有跌倒视频检出 | {seg['pos_h']}/{seg['pos_v']} | 8/9 "
             f"| {mark(seg['pos_h'] == EXPECT_VIDEOS['pos_h'])} |",
             f"| no_fall 视频误报 | {seg['neg_fa']}/{seg['neg_v']} | 3/7 "
             f"| {mark(seg['neg_fa'] == EXPECT_VIDEOS['neg_fa'])} |",
             f"| **管线正确性**（检测器固定为源项目输出，策略 B） "
             f"| {pc.get('max_abs_dp', float('nan')):.3e}（{pc.get('n_pairs', 0)} 对） "
             f"| ≤{TOL_PIPELINE:g} | {mark(pc['ok'])} |",
             "",
             "## 数值残差（**证据，不设门**，D-114）", "",
             "| 量 | 实测 | 说明 |", "|---|---:|---|",
             f"| `max\\|Δprob\\|`（逐窗口，策略 {strategy}） | "
             f"{cmp_prob['max_abs_dp']:.3e} | 在线 vs 离线，含检测器差异 |",
             f"| 均值 `\\|Δprob\\|` | {cmp_prob['mean_abs_dp']:.3e} | 同上 |",
             f"| 超 1e-3 的窗口 | {cmp_prob['n_over_tol']} / "
             f"{cmp_prob['n_compared']} | 同上 |",
             # `threshold_flips` is a *discrete* difference, not a tolerance one:
             # a window on opposite sides of 0.5 changes a segment's verdict. It
             # is reported separately and never folded into `n_over_tol` (L-107).
             f"| 阈值 0.5 两侧不一致的窗口 | {cmp_prob['threshold_flips']} / "
             f"{cmp_prob['n_compared']} | 同上，离散量 |",
             "",
             "这些残差的成因已在 D-108 定位到**检测器**，不是管线：参考 npz 由 "
             "CUDA FP16 生成、含 432 帧重复框，与 CPU ONNX 不是同一批检测。"
             "上表的「管线正确性」一行把检测器固定住，证明管线本身精确。",
             "",
             "## G6 对照（源项目发布的表，D-114 口径）", "",
             "| 判据 | 实测 | 门槛 | 结果 |", "|---|---:|---:|---|",
             f"| 逐段命中判定与 `per_segment.csv` 一致 "
             f"| {g6_seg['n_hit_match']}/{g6_seg['n']} | 全部一致 "
             f"| {mark(g6_seg['n_hit_match'] == g6_seg['n'] and g6_seg['n'] > 0)} |",
             f"| 逐段概率（**只列值，不设门**） | max {g6_seg['max_abs_dp']:.4f}，"
             f"均值 {g6_seg['mean_abs_dp']:.4f} | — | — |",
             f"| no_fall 逐视频误报判定一致 "
             f"| {g6_nofall['n_verdict_match']}/{g6_nofall['n']} | 全部一致 "
             f"| {mark(g6_nofall['n_verdict_match'] == g6_nofall['n'] and g6_nofall['n'] > 0)} |",
             f"| no_fall 逐视频 max 概率（**只列值**） "
             f"| max {g6_nofall['max_abs_dp']:.3e} | — | — |",
             "",
             "## 在线 vs 离线拼接链（D-042）", "",
             f"离线 `stitch_id_handoffs` 看整段视频的轨迹表，在线只能在"
             f"「窗口被评分的那一刻」用当时已知的链。{len(chain_report)} 个视频中"
             f"**{n_chain_mismatch}** 个的链集合不同。",
             ""]
    if n_chain_mismatch:
        lines += ["| 视频 | 离线链 | 在线链 |", "|---|---|---|"]
        for c in chain_report:
            if c["match"]:
                continue
            lines.append(f"| `{c['video_id']}` | {c['offline_chains']} "
                         f"| {c['online_chains']} |")
        lines.append("")
    lines += ["", "## 残差的成因（D-108，完整定位）", "",
              f"**runtime 管线本身是对的**：用源项目 npz 自己的关键点与框驱动同一个 "
              f"`Pipeline`（只替换 detector），策略 B "
              f"{pc.get('n_pairs', 0)} 对 (窗,人) 样本 "
              f"`max|Δp| = {pc.get('max_abs_dp', float('nan')):.3e}`，判据 ≤"
              f"{TOL_PIPELINE:g} {mark(pc['ok'])}。"
              f"产物：`{pc.get('source', 'g3_pipeline_correctness_B.json')}`。"
              f"剩余差异全部来自检测器输出，成因：", "",
              "1. 参考 npz 由 **CUDA FP16** 生成（`configs/pose.yaml`: "
              "`device:\"0\"`, `quantize:16`），关键点存为 float16；与 CPU ONNX "
              "差 0.104 px 标准差 —— 是量化噪声，不是解码错（G2 已证明解码差 "
              "1.5e-03 px）。",
              "2. 参考 npz 有 **432 帧**存在同一个人的重复框（IoU ≥ 0.5；分母是"
              "有 ≥2 人的 6,768 帧 → **6.38%**，占全部 14,679 帧是 2.94%）。"
              "框对分布 ≥0.5 / ≥0.9 / ≥0.95 / ≥0.99 = **448 / 150 / 105 / 15**，"
              "最大 **0.99805** —— 注意 105 是 **≥0.95** 的对数，不是「≈1.0」。"
              "`end2end=True` 让 NMS 短路，只剩 TopK，重叠锚点在 FP16 下双双入选。"
              "本 runtime 与 PyTorch CPU 都只输出一个框，且两者一致到 1e-6。"
              "→ 同一个人被拆成两个 track，**两边的轨迹数就不同**。",
              "3. 0.1 px 噪声本身只值 1.5e-02（受控实验），实测差异靠**离散翻转**："
              "0.1 px 就能让 10/19 个视频的 track ID 翻转。见 L-107。", "",
              "**关键缓解（有条件，产物见 `g3_reference_audit.json` / `scripts/40`）：**"
              "剔除**持续重叠 ≥5 帧**的重复轨迹（23 条）后重算参考，"
              "**A 7/17、B 12/17、C 13/17，与未剔除相同**。"
              "⚠️ 该「不变」**依赖 K=5 这个阈值**（裁决已接受，D-114）：144 对重叠里"
              "79 对只重叠 1 帧（正常换 ID 交接），K=2/3 时 C 会掉到 9/17。"
              "论文必须连阈值一起报。",
              ""]
    verdict = "通过" if ok else "未通过"
    lines += [f"**G3：{verdict}**（D-114 裁决口径）", ""]
    (paths.REPORTS / f"g3_runtime{suffix}.md").write_text(
        "\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    raise SystemExit(main())
