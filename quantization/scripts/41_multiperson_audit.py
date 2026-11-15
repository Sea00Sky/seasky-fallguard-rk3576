"""多人场景下，当前「逐人取最大」口径有多脆弱（只读分析）。"""
from __future__ import annotations

import importlib.util
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


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

    # ---- 1. 检测器每帧看到几个人（源 npz 的 n_persons，已是 max_persons=4 之后）
    files = sorted(paths.SOURCE_KEYPOINTS_BOARD.glob("*.npz"))
    per_frame: dict[int, int] = defaultdict(int)
    for f in files:
        d = np.load(f, allow_pickle=False)
        for v in d["n_persons"]:
            per_frame[int(v)] += 1
    n_frames = sum(per_frame.values())

    # ---- 2. 每个候选窗口里，过 0.70 覆盖率门的「人」数
    ref = np.load(paths.REFERENCE / "board_candidates.npz", allow_pickle=False)
    by_window: dict[int, list[tuple[int, float, float]]] = defaultdict(list)
    for i in range(len(ref["sample_window"])):
        if str(ref["sample_kind"][i]) != "C":
            continue
        by_window[int(ref["sample_window"][i])].append(
            (int(ref["sample_track"][i]), float(ref["sample_coverage"][i]),
             float(ref["sample_prob"][i])))

    gated_hist: dict[int, int] = defaultdict(int)
    raw_hist: dict[int, int] = defaultdict(int)
    margins: list[tuple[float, float]] = []
    both_fall = 0
    for w, people in by_window.items():
        raw_hist[len(people)] += 1
        gated = sorted((p for p in people if p[1] >= spec_mod.MIN_COVERAGE),
                       key=lambda t: -t[2])
        gated_hist[len(gated)] += 1
        if len(gated) >= 2:
            margins.append((gated[0][2], gated[1][2]))
            if gated[1][2] >= spec_mod.THRESHOLD:
                both_fall += 1

    m = np.asarray([a - b for a, b in margins], dtype=np.float64)
    n_win = len(by_window)

    # ---- 3. 段级：一个段内有多少个不同的「人」当过 argmax
    windows = [{"video_id": str(ref["video_id"][i]), "start": int(ref["start"][i]),
                "end": int(ref["end"][i])} for i in range(len(ref["window_id"]))]
    argmax_track: dict[int, int] = {}
    for w, people in by_window.items():
        gated = [p for p in people if p[1] >= spec_mod.MIN_COVERAGE]
        if gated:
            argmax_track[w] = max(gated, key=lambda t: t[2])[0]

    gt: dict[str, list[tuple[int, int]]] = {}
    for r in paths.read_csv(paths.SOURCE_GT_BOARD):
        gt.setdefault(r["video_id"], [])
        if r.get("status") == "no_fall" or not r.get("start_frame"):
            continue
        gt[r["video_id"]].append((int(r["start_frame"]), int(r["end_frame"])))

    seg_tracks = []
    for vid, segs in gt.items():
        for s, e in segs:
            tracks = {argmax_track[w] for w, win in enumerate(windows)
                      if win["video_id"] == vid
                      and spec_mod.overlap(win["start"], win["end"], s, e) > 0
                      and w in argmax_track}
            seg_tracks.append({"video_id": vid, "start": s, "end": e,
                               "n_distinct_argmax_tracks": len(tracks),
                               "tracks": sorted(tracks)})

    out = {
        "per_frame_person_count": {str(k): v for k, v in sorted(per_frame.items())},
        "n_frames": n_frames,
        "max_persons_in_frame": max(per_frame),
        "pct_frames_with_2plus": round(
            100 * sum(v for k, v in per_frame.items() if k >= 2) / n_frames, 2),
        "windows_with_n_gated_people": {str(k): v for k, v in sorted(gated_hist.items())},
        "windows_with_n_candidates_before_gate": {str(k): v for k, v in sorted(raw_hist.items())},
        "n_windows": n_win,
        "pct_windows_with_2plus_gated": round(100 * sum(
            v for k, v in gated_hist.items() if k >= 2) / n_win, 2),
        "multi_person_margin": {
            "n_windows": len(margins),
            "median": float(np.median(m)) if len(m) else None,
            "p10": float(np.percentile(m, 10)) if len(m) else None,
            "p90": float(np.percentile(m, 90)) if len(m) else None,
            "n_below_0.05": int((m < 0.05).sum()),
            "pct_below_0.05": round(100 * float((m < 0.05).mean()), 1) if len(m) else None,
            "n_below_0.01": int((m < 0.01).sum()),
            "pct_below_0.01": round(100 * float((m < 0.01).mean()), 1) if len(m) else None,
            "n_windows_where_2nd_also_falls": both_fall,
        },
        "segments": seg_tracks,
        "segments_with_multiple_argmax_tracks": sum(
            1 for s in seg_tracks if s["n_distinct_argmax_tracks"] > 1),
        "note": ("`per_person_max` only answers 'does *anyone* fall in this "
                 "window'. It does not answer 'who fell', and it cannot detect "
                 "two people falling at once (the 2nd person's probability is "
                 "discarded). The margin statistics show that in most "
                 "multi-person windows the ranking between the top two is "
                 "numerically unstable."),
    }
    paths.write_json(paths.REPORTS / "multiperson_audit.json", out)

    print(f"每帧人数: {dict(sorted(per_frame.items()))}")
    print(f"  ≥2 人的帧: {out['pct_frames_with_2plus']}%  (最多 {max(per_frame)} 人)")
    print(f"\n每窗过门人数: {dict(sorted(gated_hist.items()))}")
    print(f"  ≥2 个合格人的窗口: {out['pct_windows_with_2plus_gated']}%")
    print(f"\n多人窗口的 top1-top2 差距: 中位 {out['multi_person_margin']['median']:.4f}, "
          f"<0.05 占 {out['multi_person_margin']['pct_below_0.05']}%, "
          f"<0.01 占 {out['multi_person_margin']['pct_below_0.01']}%")
    print(f"  次大也 ≥0.5 的窗口: {both_fall}（= 两人同时跌倒能被报出几次）")
    print(f"\n段内有多个不同 argmax 人的段: "
          f"{out['segments_with_multiple_argmax_tracks']}/{len(seg_tracks)}")
    print(f"\nwritten -> {paths.REPORTS / 'multiperson_audit.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
