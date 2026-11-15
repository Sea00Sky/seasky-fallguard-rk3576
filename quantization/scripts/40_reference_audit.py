"""的三条定量论据，落成可复现的产物。"""
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

IOU_DUP = 0.5
EXCLUDE = {"board:squat:clip_0907_170602", "board:squat:clip_0907_170618"}
IOU_BUCKETS = (0.5, 0.8, 0.9, 0.95, 0.99, 0.999)

OVERLAP_FRAMES_MIN = 5


def iou(a: np.ndarray, b: np.ndarray) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)
    return float(inter / union) if union > 0 else 0.0


# --------------------------------------------------------------- 1. 参考体检


def audit_boxes() -> dict:
    files = sorted(paths.SOURCE_KEYPOINTS_BOARD.glob("*.npz"))
    if not files:
        raise SystemExit(f"no keypoint npz under {paths.SOURCE_KEYPOINTS_BOARD}")
    buckets = {b: 0 for b in IOU_BUCKETS}
    n_frames = n_multi = n_dup_frames = 0
    n_pairs = 0
    max_iou = 0.0
    same_id_pairs = 0
    dup_by_video: dict[str, int] = {}
    for f in files:
        d = np.load(f, allow_pickle=False)
        bb, npers, tids = d["bboxes"], d["n_persons"], d["track_ids"]
        vid = f.stem
        for t in range(len(bb)):
            n_frames += 1
            k = int(npers[t])
            if k >= 2:
                n_multi += 1
            frame_has_dup = False
            for i in range(k):
                for j in range(i + 1, k):
                    v = iou(bb[t, i], bb[t, j])
                    max_iou = max(max_iou, v)
                    if v >= IOU_DUP:
                        n_pairs += 1
                        frame_has_dup = True
                        if int(tids[t, i]) == int(tids[t, j]):
                            same_id_pairs += 1
                    for b in IOU_BUCKETS:
                        if v >= b:
                            buckets[b] += 1
            if frame_has_dup:
                n_dup_frames += 1
                dup_by_video[vid] = dup_by_video.get(vid, 0) + 1
    return {
        "n_videos": len(files),
        "n_frames": n_frames,
        "n_frames_with_2plus_persons": n_multi,
        "n_frames_with_duplicate_box": n_dup_frames,
        "pct_of_2plus_frames": round(100 * n_dup_frames / max(1, n_multi), 2),
        "pct_of_all_frames": round(100 * n_dup_frames / max(1, n_frames), 2),
        "iou_pair_counts": {str(b): buckets[b] for b in IOU_BUCKETS},
        "n_pairs_ge_0.5": n_pairs,
        "max_iou": round(max_iou, 5),
        "same_track_id_pairs": same_id_pairs,
        "duplicate_frames_by_video": dup_by_video,
    }


# ----------------------------------------------------- 2. 重复出生轨迹


def _video_id_from_stem(stem: str) -> str:
    """`board__done__clip_0831_172408` -> `board:done:clip_0831_172408`."""
    return stem.replace("board__", "board:").replace("__clip_", ":clip_")


def find_duplicates() -> tuple[set[tuple[str, int]], dict]:
    """Tracks that persistently shadow another track of the same person."""
    pairs: dict[tuple[str, int, int], int] = defaultdict(int)
    life: dict[tuple[str, int], int] = defaultdict(int)
    for f in sorted(paths.SOURCE_KEYPOINTS_BOARD.glob("*.npz")):
        d = np.load(f, allow_pickle=False)
        bb, npers, tids = d["bboxes"], d["n_persons"], d["track_ids"]
        stem = f.stem
        for t in range(len(bb)):
            k = int(npers[t])
            for i in range(k):
                tid = int(tids[t, i])
                if tid >= 0:
                    life[(stem, tid)] += 1
                for j in range(i + 1, k):
                    a, b = int(tids[t, i]), int(tids[t, j])
                    if a < 0 or b < 0 or a == b:
                        continue
                    if iou(bb[t, i], bb[t, j]) >= IOU_DUP:
                        pairs[(stem, min(a, b), max(a, b))] += 1

    dup: set[tuple[str, int]] = set()
    per_video: dict[str, int] = {}
    n_pairs_sustained = 0
    for (stem, a, b), cnt in pairs.items():
        if cnt < OVERLAP_FRAMES_MIN:
            continue
        n_pairs_sustained += 1
        short = a if life[(stem, a)] <= life[(stem, b)] else b
        vid = _video_id_from_stem(stem)
        dup.add((vid, short))
        per_video[vid] = per_video.get(vid, 0) + 1

    return dup, {
        "total_tracks": len(life),
        "n_overlapping_pairs": len(pairs),
        "n_pairs_overlap_1_frame": sum(1 for c in pairs.values() if c == 1),
        "n_pairs_sustained": n_pairs_sustained,
        "overlap_frames_min": OVERLAP_FRAMES_MIN,
        "tracks_dropped": len(dup),
        "by_video": per_video,
    }


# ------------------------------------------------- 3. 剔除后段召回


def load_gt() -> dict[str, list[tuple[int, int]]]:
    out: dict[str, list[tuple[int, int]]] = {}
    for r in paths.read_csv(paths.SOURCE_GT_BOARD):
        vid = r["video_id"]
        out.setdefault(vid, [])
        if r.get("status") == "no_fall" or not r.get("start_frame"):
            continue
        out[vid].append((int(r["start_frame"]), int(r["end_frame"])))
    return out


def segment_recall(ref: dict, drop: set[tuple[str, int]]) -> dict:
    """A/B/C recall with `drop` tracks removed from the reference samples."""
    windows = [{"video_id": str(ref["video_id"][i]), "start": int(ref["start"][i]),
                "end": int(ref["end"][i])} for i in range(len(ref["window_id"]))]
    samples = []
    for i in range(len(ref["sample_window"])):
        vid = windows[int(ref["sample_window"][i])]["video_id"]
        track = int(ref["sample_track"][i])
        if (vid.replace(":", "__"), track) in drop or (vid, track) in drop:
            continue
        samples.append((int(ref["sample_window"][i]), str(ref["sample_kind"][i]),
                        float(ref["sample_coverage"][i]), float(ref["sample_prob"][i])))
    gt = load_gt()
    by_video: dict[str, list[int]] = defaultdict(list)
    for i, w in enumerate(windows):
        by_video[w["video_id"]].append(i)

    out = {}
    for kind in ("A", "B", "C"):
        best: dict[int, float] = {}
        for w, k, cov, p in samples:
            if k != kind or cov < spec_mod.MIN_COVERAGE:
                continue
            if w not in best or p > best[w]:
                best[w] = p
        seg_tot = seg_hit = pos_v = pos_h = neg_v = neg_fa = 0
        for vid, idxs in by_video.items():
            if vid in EXCLUDE or vid not in gt:
                continue
            segs = gt[vid]
            if segs:
                pos_v += 1
                any_hit = False
                for s, e in segs:
                    cand = [i for i in idxs
                            if spec_mod.overlap(windows[i]["start"],
                                                windows[i]["end"], s, e) > 0]
                    p = max((best.get(i, 0.0) for i in cand), default=0.0)
                    ok = p >= spec_mod.THRESHOLD
                    seg_tot += 1
                    seg_hit += int(ok)
                    any_hit |= ok
                pos_h += int(any_hit)
            else:
                neg_v += 1
                vmax = max((best.get(i, 0.0) for i in idxs), default=0.0)
                neg_fa += int(vmax >= spec_mod.THRESHOLD)
        out[kind] = {"seg_hit": seg_hit, "seg_tot": seg_tot, "pos_h": pos_h,
                     "pos_v": pos_v, "neg_fa": neg_fa, "neg_v": neg_v}
    return out


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ref = np.load(paths.REFERENCE / "board_candidates.npz", allow_pickle=False)

    print("1/3 参考数据体检 ...")
    boxes = audit_boxes()
    print(f"    {boxes['n_frames_with_duplicate_box']} 帧含重复框 "
          f"({boxes['pct_of_2plus_frames']}% of 有≥2人帧, "
          f"{boxes['pct_of_all_frames']}% of 全部帧), "
          f"{boxes['n_pairs_ge_0.5']} 对 IoU≥0.5, 最大 {boxes['max_iou']}")

    print("2/3 重复轨迹（持续重叠）...")
    dup, dup_meta = find_duplicates()
    print(f"    {dup_meta['n_overlapping_pairs']} 对重叠，其中 "
          f"{dup_meta['n_pairs_overlap_1_frame']} 对只重叠 1 帧（正常交接）；"
          f"持续 ≥{OVERLAP_FRAMES_MIN} 帧的 {dup_meta['n_pairs_sustained']} 对 "
          f"-> 剔除 {dup_meta['tracks_dropped']} 条较短轨迹")

    print("3/3 剔除后重算段召回 ...")
    before = segment_recall(ref, set())
    after = segment_recall(ref, dup)
    for k in ("A", "B", "C"):
        b, a = before[k], after[k]
        print(f"    {k}: {b['seg_hit']}/{b['seg_tot']} -> {a['seg_hit']}/{a['seg_tot']}  "
              f"(视频 {b['pos_h']}/{b['pos_v']} -> {a['pos_h']}/{a['pos_v']}, "
              f"误报 {b['neg_fa']}/{b['neg_v']} -> {a['neg_fa']}/{a['neg_v']})")

    # 敏感性：结论有多依赖 `OVERLAP_FRAMES_MIN` 这个自选阈值？如果 K=2 就掉到
    # 9/17、K=5 才回到 13/17，那「不变」只在特定阈值下成立 —— 必须如实报告，
    # 否则又是一个「50 条」式的、没有定义支撑的断言。
    sensitivity = {}
    for k_min in (1, 2, 3, 5, 10, 20):
        pairs: dict[tuple[str, int, int], int] = defaultdict(int)
        life: dict[tuple[str, int], int] = defaultdict(int)
        for f in sorted(paths.SOURCE_KEYPOINTS_BOARD.glob("*.npz")):
            d = np.load(f, allow_pickle=False)
            bb, npers, tids = d["bboxes"], d["n_persons"], d["track_ids"]
            for t in range(len(bb)):
                kk = int(npers[t])
                for i in range(kk):
                    tid = int(tids[t, i])
                    if tid >= 0:
                        life[(f.stem, tid)] += 1
                    for j in range(i + 1, kk):
                        a, b = int(tids[t, i]), int(tids[t, j])
                        if a < 0 or b < 0 or a == b:
                            continue
                        if iou(bb[t, i], bb[t, j]) >= IOU_DUP:
                            pairs[(f.stem, min(a, b), max(a, b))] += 1
        drop = set()
        for (stem, a, b), cnt in pairs.items():
            if cnt >= k_min:
                short = a if life[(stem, a)] <= life[(stem, b)] else b
                drop.add((_video_id_from_stem(stem), short))
        r = segment_recall(ref, drop)
        sensitivity[str(k_min)] = {
            "tracks_dropped": len(drop),
            "C_seg_hit": r["C"]["seg_hit"], "C_seg_tot": r["C"]["seg_tot"],
            "B_seg_hit": r["B"]["seg_hit"], "A_seg_hit": r["A"]["seg_hit"],
        }
        print(f"    K={k_min:2d}: 剔除 {len(drop):3d} 条 -> "
              f"A {r['A']['seg_hit']}/17  B {r['B']['seg_hit']}/17  "
              f"C {r['C']['seg_hit']}/17")

    unchanged = all(before[k]["seg_hit"] == after[k]["seg_hit"] for k in before)
    paths.write_json(paths.REPORTS / "g3_reference_audit.json", {
        "reference_boxes": boxes,
        "duplicates": dup_meta,
        "duplicate_definition":
            f"of two tracks whose boxes overlap at IoU >= {IOU_DUP} in "
            f">= {OVERLAP_FRAMES_MIN} frames, the shorter-lived one is the "
            "duplicate and is dropped",
        "segment_recall_before": before,
        "segment_recall_after_dropping_duplicates": after,
        "recall_unchanged": unchanged,
        "threshold_sensitivity": sensitivity,
        "note": ("Quantitative backing for D-108 / the G3 ruling. The claim "
                 "'business metrics are robust to the reference's duplicate "
                 "boxes' holds ONLY at a sustained-overlap threshold >= 5 "
                 "frames; at K=2 or K=3 the C recall falls to 9/17 because "
                 "single-frame handoffs get dropped too. Report the threshold "
                 "with the claim -- the earlier '50 tracks, unchanged' had no "
                 "definition and no artefact."),
    })
    print(f"\nrecall unchanged at K={OVERLAP_FRAMES_MIN}: {unchanged}")
    print(f"written -> {paths.REPORTS / 'g3_reference_audit.json'}")
    return 0 if unchanged else 1


if __name__ == "__main__":
    raise SystemExit(main())
