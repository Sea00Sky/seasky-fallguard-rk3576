"""板端主程序：跑一个视频，输出逐窗结果和逐段判定。"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import Pipeline, PipelineConfig # noqa: E402
from src import spec # noqa: E402
from src.backends import StgcnBackend, YoloBackend # noqa: E402

# 源项目 `scripts/30_runtime_eval.py:86` 的 EXCLUDE，必须照抄。
# 这两个蹲姿视频在 GT 里各有一段，但源项目把它们排除在评测之外。
# 不排除的话分母是 19 段而不是 17 段，段召回数字与 PC 基线不可比。
# 排除理由见源项目 D-114（评测判据裁决）。
EXCLUDE = {"board:squat:clip_0907_170602", "board:squat:clip_0907_170618"}


# ---------------------------------------------------------------- RKNN 初始化


def load_rknn(path: Path, core_mask=None):
    """加载并初始化一个 RKNNLite 会话。"""
    from rknnlite.api import RKNNLite

    r = RKNNLite()
    if r.load_rknn(str(path)) != 0:
        raise RuntimeError(f"load_rknn 失败: {path}")
    kw = {} if core_mask is None else {"core_mask": core_mask}
    if r.init_runtime(**kw) != 0:
        raise RuntimeError(
            f"init_runtime 失败: {path}。"
            f"先查 librknnrt 版本："
            f"strings /usr/lib/librknnrt.so | grep -i 'librknnrt version'。"
            f"版本不匹配的报错是 'Invalid RKNN model version'，"
            f"且 RKNNLite 硬编码 /usr/lib/librknnrt.so，LD_LIBRARY_PATH 覆盖不了。")
    return r


def build_pipeline(yolo_rknn, stgcn_rknn, fps: float = spec.BOARD_FPS) -> Pipeline:
    """把两个 RKNN 会话接进管线。"""
    cfg = PipelineConfig(fps=fps)
    return Pipeline(cfg,
                    detector=YoloBackend(yolo_rknn, conf=cfg.conf,
                                         max_persons=cfg.max_persons),
                    stgcn_session=StgcnBackend(stgcn_rknn))


# ------------------------------------------------------------------ 视频读取


def read_frames(path: Path, max_frames: int = 0):
    """逐帧产出 BGR uint8。板端用 cv2.VideoCapture，与 PC 侧一致。"""
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"打不开视频: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    n = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield frame
            n += 1
            if max_frames and n >= max_frames:
                break
    finally:
        cap.release()
    return fps


def probe(path: Path) -> dict:
    import cv2

    cap = cv2.VideoCapture(str(path))
    info = {
        "fps": float(cap.get(cv2.CAP_PROP_FPS)),
        "n_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    cap.release()
    return info


# ------------------------------------------------------------------ 视频清单


def video_id_from_path(p: Path) -> str:
    """`videos/done/clip_0831_172408.mp4` -> `board:done:clip_0831_172408`。"""
    parent = p.parent.name
    mapping = {
        "done": "board:done",
        "整体测试视频": "board:integral_test",
        "蹲姿实录视频": "board:squat",
    }
    if parent not in mapping:
        raise ValueError(
            f"不认识的视频目录 {parent!r}；已知: {list(mapping)}。"
            f"目录名决定 video_id，改目录会让结果对不上 reference/videos.csv")
    return f"{mapping[parent]}:{p.stem}"


def find_videos(root: Path) -> list[Path]:
    out = []
    for d in ("done", "整体测试视频", "蹲姿实录视频"):
        out.extend(sorted((root / d).glob("*.mp4")))
    return out


# ------------------------------------------------------------------ 逐段判定


def load_fall_intervals(path: Path) -> dict[str, list[tuple[int, int]]]:
    """`board_fall_intervals.csv` -> {video_id: [(start, end), ...]}。"""
    out: dict[str, list[tuple[int, int]]] = {}
    with open(path, encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            vid = row["video_id"]
            out.setdefault(vid, [])
            s = (row.get("start_frame") or "").strip()
            e = (row.get("end_frame") or "").strip()
            if s and e:
                out[vid].append((int(s), int(e)))
    return out


def segment_hits(result, intervals: list[tuple[int, int]]) -> list[dict]:
    """每个 GT 段取覆盖它的窗口里最大的 prob_fall。"""
    rows = []
    for (s, e) in intervals:
        covering = [w for w in result.windows
                    if max(0, min(w.end, e + 1) - max(w.start, s)) > 0]
        best = max((w.prob_fall for w in covering), default=0.0)
        rows.append({
            "video_id": result.video_id, "start": s, "end": e,
            "prob_fall": round(best, 6),
            "hit": int(best >= spec.THRESHOLD),
            "n_windows": len(covering),
        })
    return rows


# ------------------------------------------------------------------ main


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=None, help="单个视频路径")
    ap.add_argument("--all", action="store_true", help="跑 videos/ 下全部")
    ap.add_argument("--videos-dir", default=str(ROOT / "videos"))
    ap.add_argument("--yolo", default=str(ROOT / "models" / "yolo26n_pose_480x640_fp16.rknn"))
    ap.add_argument("--stgcn", default=str(ROOT / "models" / "stgcnpp_v2_fp16.rknn"))
    ap.add_argument("--yolo-core", type=int, default=None,
                    help="YOLO 绑的 NPU 核心（0/1）；默认不指定")
    ap.add_argument("--stgcn-core", type=int, default=None)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "results"))
    ap.add_argument("--gt", default=str(ROOT / "reference" / "board_fall_intervals.csv"))
    ap.add_argument("--no-fall-list", default=str(
        ROOT / "reference" / "per_video_nofall.csv"))
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.video:
        videos = [Path(args.video)]
    elif args.all:
        videos = find_videos(Path(args.videos_dir))
    else:
        ap.error("给 --video 或 --all")
    if not videos:
        print(f"没找到视频: {args.videos_dir}", file=sys.stderr)
        return 2

    print(f"视频 {len(videos)} 个")
    print(f"  YOLO  {args.yolo}")
    print(f"  STGCN {args.stgcn}")

    t0 = time.time()
    yolo = load_rknn(Path(args.yolo), args.yolo_core)
    stgcn = load_rknn(Path(args.stgcn), args.stgcn_core)
    print(f"RKNN 初始化 {time.time() - t0:.1f} s", flush=True)

    intervals = load_fall_intervals(Path(args.gt))
    seg_rows: list[dict] = []
    per_video: list[dict] = []

    for i, vp in enumerate(videos, 1):
        try:
            vid = video_id_from_path(vp)
        except ValueError as exc:
            print(f"[skip] {exc}", file=sys.stderr)
            continue

        info = probe(vp)
        pipe = build_pipeline(yolo, stgcn, fps=info["fps"] or spec.BOARD_FPS)

        t = time.time()
        n = 0
        for frame in read_frames(vp, args.max_frames):
            pipe.push(frame)
            n += 1
        res = pipe.finish(vid)
        dt = time.time() - t

        fps = n / dt if dt > 0 else 0.0
        hits = [] if vid in EXCLUDE else segment_hits(res, intervals.get(vid, []))
        seg_rows.extend(hits)

        best = max((w.prob_fall for w in res.windows), default=0.0)
        row = {
            "video_id": vid, "n_frames": n, "n_windows": len(res.windows),
            "n_scored": res.n_scored, "n_observed": res.n_observed,
            "max_prob": round(best, 6),
            "seconds": round(dt, 2), "fps": round(fps, 2),
            "segments": len(hits), "hits": sum(h["hit"] for h in hits),
        }
        per_video.append(row)
        print(f"[{i}/{len(videos)}] {vid}  {n} 帧  {len(res.windows)} 窗  "
              f"{fps:.2f} fps  max_p={best:.4f}  "
              f"段 {row['hits']}/{row['segments']}", flush=True)

        (out_dir / f"{vid.replace(':', '__')}.json").write_text(
            json.dumps({
                "video_id": vid, "n_frames": n,
                "windows": [{"window_id": w.window_id, "start": w.start,
                             "end": w.end, "prob_fall": w.prob_fall,
                             "track_id": w.track_id, "coverage": w.coverage,
                             "n_candidates": w.n_candidates}
                            for w in res.windows],
                "segments": hits,
                "seconds": dt, "fps": fps,
            }, ensure_ascii=False, indent=2), encoding="utf-8")

 # ---- 汇总 -------------------------------------------------------------
    if seg_rows:
        with open(out_dir / "per_segment.csv", "w", newline="",
                  encoding="utf-8-sig") as fh:
            w = csv.DictWriter(fh, fieldnames=list(seg_rows[0]))
            w.writeheader()
            w.writerows(seg_rows)

    total_win = sum(r["n_windows"] for r in per_video)
    seg_total = sum(r["segments"] for r in per_video)
    seg_hit = sum(r["hits"] for r in per_video)

 # 有跌倒视频：GT 里有段的视频，检出 = 至少一段命中
    with_fall = {v for v in intervals if intervals[v]} - EXCLUDE
    vids_with_fall = [r for r in per_video if r["video_id"] in with_fall]
    vids_detected = [r for r in vids_with_fall if r["hits"] > 0]

 # no_fall 视频：GT 里没有段的，误报 = max_prob >= 0.5
    no_fall = [r for r in per_video
               if r["video_id"] not in with_fall and r["video_id"] not in EXCLUDE
               and r["video_id"] in intervals]
    false_alarms = [r for r in no_fall if r["max_prob"] >= spec.THRESHOLD]

    summary = {
        "n_videos": len(per_video),
        "n_windows": total_win,
        "segment_recall": f"{seg_hit}/{seg_total}",
        "segments_hit": seg_hit, "segments_total": seg_total,
        "videos_with_fall": f"{len(vids_detected)}/{len(vids_with_fall)}",
        "no_fall_false_alarms": f"{len(false_alarms)}/{len(no_fall)}",
        "pc_baseline": {
            "n_windows": 915,
            "segment_recall": "14/17",
            "videos_with_fall": "9/9",
            "no_fall_false_alarms": "3/7",
            "note": "PC 模拟器，19 视频，见 reference/g3_rknn_fp16.json；"
                    "历史 30fps 容器口径，语料已于 2026-11-12 重定时 20fps",
        },
        "per_video": per_video,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print()
    print(f"窗口 {total_win}（历史 30fps 容器基线 915；真 20fps 全量 1405）")
    print(f"段召回 {seg_hit}/{seg_total}（历史基线 14/17；真 20fps 实测 13/17）")
    print(f"有跌倒视频 {len(vids_detected)}/{len(vids_with_fall)}（历史基线 9/9；真 20fps 实测 8/9）")
    print(f"no_fall 误报 {len(false_alarms)}/{len(no_fall)}（历史基线 3/7；真 20fps 实测 4/7）")
    print(f"\n结果写到 {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
