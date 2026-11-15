"""逐人跌倒判定：把「谁跌倒了」从窗口级 max 里解出来（只读分析）。"""
from __future__ import annotations

import importlib.util
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

EXCLUDE = {"board:squat:clip_0907_170602", "board:squat:clip_0907_170618"}


def _load_own(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, PROJECT_ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


paths = _load_own("deploy_paths", "src/deploy/paths.py")
spec_mod = _load_own("deploy_spec", "src/deploy/spec.py")


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ref = np.load(paths.REFERENCE / "board_candidates.npz", allow_pickle=False)

    n_win = len(ref["window_id"])
    windows = [{"video_id": str(ref["video_id"][i]),
                "start": int(ref["start"][i]), "end": int(ref["end"][i])}
               for i in range(n_win)]

    per_person: dict[tuple[int, int], float] = {}
    members: dict[tuple[int, int], str] = {}
    for i in range(len(ref["sample_window"])):
        if str(ref["sample_kind"][i]) != "C":
            continue
        if float(ref["sample_coverage"][i]) < spec_mod.MIN_COVERAGE:
            continue
        key = (int(ref["sample_window"][i]), int(ref["sample_track"][i]))
        p = float(ref["sample_prob"][i])
        if key not in per_person or p > per_person[key]:
            per_person[key] = p
        members[key] = str(ref["sample_members"][i])

    # ---- 1. 窗口级 max（今天的口径），用于证明两条路径等价
    win_max: dict[int, float] = defaultdict(float)
    for (w, _t), p in per_person.items():
        win_max[w] = max(win_max[w], p)

    # ---- 2. 段级 / 视频级：两种遍历顺序
    gt: dict[str, list[tuple[int, int]]] = {}
    for r in paths.read_csv(paths.SOURCE_GT_BOARD):
        gt.setdefault(r["video_id"], [])
        if r.get("status") == "no_fall" or not r.get("start_frame"):
            continue
        gt[r["video_id"]].append((int(r["start_frame"]), int(r["end_frame"])))

    by_video: dict[str, list[int]] = defaultdict(list)
    for i, w in enumerate(windows):
        by_video[w["video_id"]].append(i)

    seg_tot = seg_hit_windowmax = seg_hit_perperson = 0
    per_seg = []
    disagree = 0
    for vid in sorted(by_video):
        if vid in EXCLUDE or vid not in gt:
            continue
        for s, e in gt[vid]:
            cand = [i for i in by_video[vid]
                    if spec_mod.overlap(windows[i]["start"], windows[i]["end"], s, e) > 0]
            if not cand:
                continue
            # 口径 1：窗口 max，再段内 max
            a = max((win_max.get(i, 0.0) for i in cand), default=0.0)
            # 口径 2：逐人跨窗口 max，再人对人取 max
            # 同时记住取到最大值的那一窗，否则 members 查表会落空。
            people: dict[int, float] = {}
            best_win: dict[int, int] = {}
            for i in cand:
                for (w, t), p in per_person.items():
                    if w != i:
                        continue
                    if p > people.get(t, -1.0):
                        people[t] = p
                        best_win[t] = w
            b = max(people.values(), default=0.0)
            flagged = sorted((t for t, p in people.items()
                              if p >= spec_mod.THRESHOLD),
                             key=lambda t: -people[t])
            seg_tot += 1
            ok_a = a >= spec_mod.THRESHOLD
            ok_b = b >= spec_mod.THRESHOLD
            seg_hit_windowmax += int(ok_a)
            seg_hit_perperson += int(ok_b)
            if ok_a != ok_b:
                disagree += 1
            per_seg.append({
                "video_id": vid, "start": s, "end": e,
                "prob_windowmax": round(a, 6),
                "prob_perperson": round(b, 6),
                "n_people_scored": len(people),
                "n_people_flagged": len(flagged),
                "flagged": [{"track": t, "prob": round(people[t], 4),
                             "members": members.get((best_win[t], t), "")}
                            for t in flagged],
            })

    # ---- 3. no_fall 视频：有多少个「不同的人」被报跌倒
    neg_v = 0
    neg_fa_video = 0
    neg_flagged_persons: list[dict] = []
    for vid in sorted(by_video):
        if vid in EXCLUDE or vid not in gt or gt[vid]:
            continue
        neg_v += 1
        people: dict[int, float] = defaultdict(float)
        for i in by_video[vid]:
            for (w, t), p in per_person.items():
                if w == i:
                    people[t] = max(people[t], p)
        flagged = {t: p for t, p in people.items() if p >= spec_mod.THRESHOLD}
        if flagged:
            neg_fa_video += 1
        neg_flagged_persons.append({
            "video_id": vid, "n_people_scored": len(people),
            "n_people_flagged": len(flagged),
            "flagged": {str(t): round(p, 4) for t, p in flagged.items()},
        })

    # ---- 4. 没有状态机会怎样：同一个人的命中窗口是否连成一段
    contiguity = []
    for rec in per_seg:
        for f in rec["flagged"]:
            t = f["track"]
            hits = sorted(i for i in by_video[rec["video_id"]]
                          if spec_mod.overlap(windows[i]["start"], windows[i]["end"],
                                              rec["start"], rec["end"]) > 0
                          and per_person.get((i, t), 0.0) >= spec_mod.THRESHOLD)
            if not hits:
                continue
            # 相邻窗口在时间上是否重叠（窗口有 stride，通常相邻就重叠）
            runs, cur = 1, [hits[0]]
            for prev, nxt in zip(hits, hits[1:]):
                if windows[nxt]["start"] < windows[prev]["end"]:
                    cur.append(nxt)
                else:
                    runs += 1
                    cur = [nxt]
            contiguity.append({
                "video_id": rec["video_id"], "segment": [rec["start"], rec["end"]],
                "track": t, "n_hit_windows": len(hits), "n_runs": runs,
                "hit_window_span": [windows[hits[0]]["start"], windows[hits[-1]]["end"]],
            })

    multi_run = sum(1 for c in contiguity if c["n_runs"] > 1)
    out = {
        "n_segments": seg_tot,
        "segment_hit_windowmax": seg_hit_windowmax,
        "segment_hit_perperson": seg_hit_perperson,
        "n_segments_where_orders_disagree": disagree,
        "equivalence_note": (
            "max over (window, person) is order-independent, so collapsing "
            "per-window then per-segment gives exactly the same number as "
            "collapsing per-person then per-segment. Per-person aggregation "
            "adds identity, not recall."),
        "per_segment": per_seg,
        "no_fall_videos": {"n": neg_v, "n_videos_with_false_alarm": neg_fa_video,
                           "detail": neg_flagged_persons},
        "temporal_contiguity": {
            "n_flagged_person_segments": len(contiguity),
            "n_with_multiple_runs": multi_run,
            "pct_with_multiple_runs": round(100 * multi_run / len(contiguity), 1)
            if contiguity else None,
            "detail": contiguity,
        },
    }
    paths.write_json(paths.REPORTS / "per_person_decision.json", out)

    print(f"段数 {seg_tot}")
    print(f"  窗口 max 口径命中 : {seg_hit_windowmax}/{seg_tot}")
    print(f"  逐人 max  口径命中 : {seg_hit_perperson}/{seg_tot}")
    print(f"  两种遍历顺序不一致的段: {disagree}  (必然为 0)")
    print(f"\nno_fall 视频: {neg_fa_video}/{neg_v} 有报警")
    print(f"\n被报跌倒的「人·段」: {len(contiguity)}")
    print(f"  其中命中窗口分成多段的: {multi_run} "
          f"({out['temporal_contiguity']['pct_with_multiple_runs']}%) "
          f"<- 这些会重复报警，需要事件去重")
    print(f"\nwritten -> {paths.REPORTS / 'per_person_decision.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
