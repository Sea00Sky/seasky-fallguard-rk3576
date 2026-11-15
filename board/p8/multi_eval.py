#!/usr/bin/env python3
"""p8/multi_eval.py — 验收（V1 / V2 / V3），不重复对账逻辑。"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import Pipeline, PipelineConfig, spec # noqa: E402
from src.multi import ( # noqa: E402
    STATUS_PREGATE,
    STATUS_SCORED,
    EventBook,
    FrameClock,
    MultiConfig,
    MultiPipeline,
    TrackEventState,
)


# ---------------------------------------------------------------- 复用 run_video

def _load_run_video():
    """按文件路径加载 `scripts/run_video.py`。"""
    path = ROOT / "scripts" / "run_video.py"
    spec_ = importlib.util.spec_from_file_location("_rv", path)
    mod = importlib.util.module_from_spec(spec_)
    spec_.loader.exec_module(mod)
    return mod


# ================================================================ selftest

def _prob_from_input(x: np.ndarray) -> float:
    """按输入张量内容算概率，与调用顺序无关。"""
    a = np.asarray(x, np.float64).reshape(-1, 3)
 # 第 0 帧第 0 个关节的 x（[-1,1] 归一化后）—— 每人固定。
 # 假检测器把人摆在 x = 10 + 120*i（画面宽 400）⇒ 归一化后 -0.7 / -0.1 / 0.5。
 # 第 0 个人给低概率、其余给高概率，这样「闸门=1 只留 id 最小的」
 # 才会真的改变窗口 max —— 否则闸门生效了也测不出来。
    v = float(a[0, 0])
    return 0.05 if v < -0.5 else 0.9


def _fake_backends(n_people_at):
    """确定性的假后端。`n_people_at(frame) -> 人数`。"""
    from src.backends import Detection

    class FakeDetector:
        def __init__(self):
            self.n_calls = 0

        def detect(self, frame):
            self.n_calls += 1
            n = n_people_at(self.n_calls - 1)
            out = []
            for i in range(n):
 # 每人一个固定不动的框 ⇒ IoU=1 ⇒ tracker 稳定续号
                x = 10.0 + 120.0 * i
                out.append(Detection(
                    bbox=np.array([x, 10.0, x + 100.0, 210.0], np.float32),
                    keypoints=np.tile(
                        np.array([[x + 50.0, 100.0]], np.float32), (17, 1)),
                    scores=np.full((17,), 0.9, np.float32)))
            return out

    class FakeStgcn:
        def __init__(self):
            self.n_calls = 0
            self._inputs = [type("I", (), {"name": "input"})()]

        def get_inputs(self):
            return self._inputs

        def run(self, _outs, feeds):
            p = float(np.clip(_prob_from_input(next(iter(feeds.values()))),
                              1e-6, 1 - 1e-6))
            self.n_calls += 1
 # softmax2 作用在最后一维，返回 (1,2)
            return [np.array([[np.log(1 - p), np.log(p)]], np.float32)]

    return FakeDetector, FakeStgcn


def cmd_selftest(_args) -> int:
    ok = True

    def check(name, got, want):
        nonlocal ok
        good = got == want
        ok = ok and good
        print(f"  {'PASS' if good else 'FAIL'}  {name}  got={got!r} want={want!r}")

    def close(name, got, want, tol=1e-9):
        nonlocal ok
        good = abs(got - want) <= tol
        ok = ok and good
        print(f"  {'PASS' if good else 'FAIL'}  {name}  got={got!r} want={want!r}")

 # ---------------------------------------------------------- 1. 等价性
    print("[1] per-person 化不改变每窗结果（MultiPipeline vs Pipeline）")
    FakeDetector, FakeStgcn = _fake_backends(lambda f: 1 + (f // 40) % 3)
    frames = [np.zeros((240, 400, 3), np.uint8) for _ in range(200)]

    def run(cls, mcfg=None):
        cfg = PipelineConfig(fps=30.0)
        det = FakeDetector()
        stg = FakeStgcn()
        if cls is MultiPipeline:
            p = cls(cfg, detector=det, stgcn_session=stg, mcfg=mcfg)
        else:
            p = cls(cfg, detector=det, stgcn_session=stg)
        for f in frames:
            p.push(f)
        return p.finish("fake")

    base = run(Pipeline)
    mcfg1 = MultiConfig(confirm_windows=1, retract_windows=1, cooldown_s=0.0,
                        max_persons_stgcn=3)
    multi = run(MultiPipeline, mcfg1)

    check("窗口数", len(multi.base.windows), len(base.windows))
    dmax = max((abs(a.prob_fall - b.prob_fall)
                for a, b in zip(base.windows, multi.base.windows)), default=0.0)
    check("max|Δp|", dmax, 0.0)
    check("track_id 全同",
          all(a.track_id == b.track_id
              for a, b in zip(base.windows, multi.base.windows)), True)
    check("coverage 全同",
          all(a.coverage == b.coverage
              for a, b in zip(base.windows, multi.base.windows)), True)

 # per_person 行数 == Σ n_candidates（过了覆盖率门的人数）
    n_scored_rows = sum(1 for r in multi.per_person if r.status == STATUS_SCORED)
    check("per_person(scored) 行数 == Σn_candidates",
          n_scored_rows, sum(w.n_candidates for w in base.windows))

 # 每窗 max 必须等于该窗 per_person(scored) 的 max
    by_wid: dict[int, float] = {}
    for r in multi.per_person:
        if r.status == STATUS_SCORED:
            by_wid[r.window_id] = max(by_wid.get(r.window_id, 0.0), r.prob_fall)
    bad = [w.window_id for w in multi.base.windows
           if abs(w.prob_fall - by_wid.get(w.window_id, 0.0)) > 0.0]
    check("每窗 max == per_person 的 max（0 例外）", bad, [])

 # ---------------------------------------------------------- 2. 人数闸门
    print("[2] MAX_PERSONS_STGCN 闸门")
    mcfg_cap = MultiConfig(confirm_windows=1, retract_windows=1, cooldown_s=0.0,
                           max_persons_stgcn=1)
    capped = run(MultiPipeline, mcfg_cap)
    n_capped = sum(1 for r in capped.per_person if r.status == "capped")
    check("确有被丢的人（否则本测试无效）", n_capped > 0, True)
    check("丢弃计数一致", capped.counters.stgcn_dropped_max_persons, n_capped)
    n_over = sum(1 for w in base.windows if w.n_candidates > 1)
    changed = [i for i, (a, b) in enumerate(zip(base.windows, capped.base.windows))
               if abs(a.prob_fall - b.prob_fall) > 0]
 # 闸门=1 只保留覆盖率最高的那个人。若被丢的人概率更低，窗口 max 不变
 # ⇒ 「变了的窗口数」只会 <= 多人窗数，不会等于它。
    check("闸门=1 确实改变了某些窗口", len(changed) > 0, True)
    check("变了的窗口都是多人窗",
          all(base.windows[i].n_candidates > 1 for i in changed), True)
    check("变了的窗口数不超过多人窗数", len(changed) <= n_over, True)

 # ---------------------------------------------------------- 3. 状态机
    print("[3] 两态状态机 / 冷却（直接喂概率序列）")

    def drive(probs, mcfg, dt=0.5):
        """按窗口序列喂状态机。返回 (事件列表, book)。"""
        book = EventBook(mcfg, clock=lambda: drive.t)
        drive.t = 0.0
        evs = []
        for i, p in enumerate(probs):
            evs.extend(book.observe(7, p, i * 15, i * 15 + 60, 1.0))
            drive.t += dt
        return evs, book

 # confirm_windows=2：单个命中不告警，连续两个才告警
    e, _b = drive([0.9, 0.1, 0.9, 0.9, 0.9],
                  MultiConfig(confirm_windows=2, cooldown_s=0.0))
    check("confirm=2 事件数", len(e), 1)
    check("confirm=2 首个 confirmed 的 window_start", e[0].window_start, 45)

 # 已 confirmed 后继续命中不再重复告警（要等 retracted 之后才可能再报）
    e, _b = drive([0.9, 0.9], MultiConfig(confirm_windows=1, cooldown_s=0.0))
    check("confirm=1 连续两窗命中 -> 只 1 条（不重复告警）", len(e), 1)

 # retract：confirmed 后连续 4 窗回落
    e, b = drive([0.9, 0.9] + [0.1] * 5,
                 MultiConfig(confirm_windows=2, retract_windows=4,
                             cooldown_s=0.0))
    check("confirmed 后 4 窗回落 -> retracted", [x.state.label() for x in e],
          ["confirmed", "retracted"])
 # 撤回带的是确认窗，不是当前窗。
 # 确认发生在 idx 1（confirm_windows=2 的第 2 窗）⇒ 撤回行沿用 window_start=15。
 # 旧期望 `5*15` 断言的是修复前的错误行为：确认 15 / 撤回 75 ⇒ 护士端
 # 两行卡片、红卡永远变不灰。期望于 随修复更正。
    check("retracted 沿用确认窗（不是当前窗）", e[1].window_start, 1 * 15)

 # 不变量：计数器必须与事件一一对应。
 # 第一版 `_retract` 漏了 `counters.retracted += 1`，结果 events 里有
 # retracted 条目、`retracted_total` 却恒为 0 —— 「事件对、计数错」最危险，
 # 看 metrics 的人会以为撤回从未发生。这条断言就是为了锁住它。
    check("计数：confirmed 与事件一致", b.counters.confirmed,
          sum(1 for x in e if x.state.label() == "confirmed"))
    check("计数：retracted 与事件一致", b.counters.retracted,
          sum(1 for x in e if x.state.label() == "retracted"))
    check("计数：retracted 不是 0（防回归）", b.counters.retracted, 1)

    e, _b = drive([0.9, 0.9] + [0.1] * 3,
                  MultiConfig(confirm_windows=2, retract_windows=4,
                              cooldown_s=0.0))
    check("只回落 3 窗 -> 不撤回", [x.state.label() for x in e], ["confirmed"])

 # ---- 冷却：撤回后 30 秒内再次命中被压掉，但要记 metric ----
 # 序列：命中(confirmed) → 回落(retracted) → 立刻再命中(被冷却压掉)
    e, b = drive([0.9, 0.1, 0.9],
                 MultiConfig(confirm_windows=1, retract_windows=1,
                             cooldown_s=30.0), dt=0.5)
    check("冷却期内不产生第二次 confirmed",
          [x.state.label() for x in e], ["confirmed", "retracted"])
    check("冷却压制记了 metric", b.counters.cooldown_suppressed, 1)
    check("按 person 分开记", list(b.counters.cooldown_suppressed_by_person), [7])

 # 冷却过后可以再次告警（二次跌倒）
    e, _b = drive([0.9, 0.1, 0.9],
                  MultiConfig(confirm_windows=1, retract_windows=1,
                              cooldown_s=0.4), dt=0.5)
    check("冷却过后二次跌倒可告警",
          [x.state.label() for x in e], ["confirmed", "retracted", "confirmed"])

 # ---------------------------------------------------------- 4. 状态隔离
    print("[4] per-track 状态隔离（生产管线的全局单份 bug）")
    book = EventBook(MultiConfig(confirm_windows=2, cooldown_s=0.0),
                     clock=lambda: 0.0)
    book.observe(1, 0.9, 0, 60, 1.0) # A 第 1 窗命中
    book.observe(2, 0.1, 0, 60, 1.0) # B 第 1 窗未命中
    ev = book.observe(1, 0.9, 15, 75, 1.0)
    check("A 第 2 窗命中即 confirmed", [x.state.label() for x in ev], ["confirmed"])
    check("A 的事件不影响 B 的 consec_hits", book.get(2).consec_hits, 0)
    check("A 的事件不影响 B 的 state", book.get(2).state.label(), "suspected")
    check("两个 track 各自一份状态", len(book), 2)

 # ---------------------------------------------------------- 5. 覆盖率门外
    print("[5] 覆盖率门外的窗口不算 miss")
    st = EventBook(MultiConfig(confirm_windows=2, retract_windows=2,
                               cooldown_s=0.0), clock=lambda: 0.0)
    st.observe(1, 0.9, 0, 60, 1.0)
 # gated 窗口由 MultiPipeline 跳过，不进 observe ⇒ consec 冻结
    check("gated 不推进 consec", st.get(1).consec_hits, 1)

 # ---- V3 的变异测试：坏实现必须被检出，否则 V3 是永远 PASS 的死测试 ----
    print("[5b] V3 变异测试：全局单份状态必须被检出")

    class _GlobalBook(EventBook):
        """故意写坏：所有 track 共用一份状态（= 生产管线那个多人 bug）。"""

        def __init__(self, cfg, clock=None):
            super().__init__(cfg, clock)
            self._one = TrackEventState(person_id=-1)

        def state_of(self, person_id):
            self._states.setdefault(person_id, self._one)
            return self._one

    synth = [{
        "video_id": "board:synth:two", "fps": 30.0,
        "per_person": [{
            "window_id": i, "person_id": pid,
            "window_start": i * 15, "window_end": i * 15 + 60,
            "prob_fall": p, "coverage": 1.0, "status": STATUS_SCORED,
        } for i in range(10) for pid, p in ((1, 0.9), (2, 0.1))],
    }]
    mc = MultiConfig(confirm_windows=2, cooldown_s=0.0)
    _m, n_p, n_bad_ok, _b, _d = isolation_check(synth, mc)
    check("好实现：检查条目 > 0", n_p > 0, True)
    check("好实现：污染 0", n_bad_ok, 0)
    _m, n_p, n_bad_bad, _b, _d = isolation_check(synth, mc, book_cls=_GlobalBook)
    check("坏实现：污染被检出（>0）", n_bad_bad > 0, True)

 # ---------------------------------------------------------- 6. V2 口径
    print("[6] V2 口径：confirm=1 时与 run_video.py 的段判定逐位等价")
    rv = _load_run_video()

    class _W:
        def __init__(self, s, e, p):
            self.start, self.end, self.prob_fall = s, e, p

    class _R:
        video_id = "fake"
        windows = [_W(0, 60, 0.1), _W(15, 75, 0.8), _W(30, 90, 0.2)]

    hits = rv.segment_hits(_R(), [(0, 59), (100, 159)])
    check("覆盖段取 max -> 命中", [h["hit"] for h in hits], [1, 0])
    close("命中段的 prob", hits[0]["prob_fall"], 0.8)

 # ---- 用真实板端基线做端到端等价性检查 ----
 # reference/board_windows.csv 是 存档的 915 窗板端结果。
 # 把它改写成「每人每窗」文档（每窗只放 1 个人 = 窗口 max 的那个人），
 # 则 confirm_windows=1 下 sweep 的段召回必须与 run_video.py 完全一致。
    base_csv = ROOT / "reference" / "board_windows.csv"
    if base_csv.exists():
        by_vid: dict[str, list[dict]] = {}
        with open(base_csv, encoding="utf-8-sig") as fh:
            for r in csv.DictReader(fh):
                by_vid.setdefault(r["video_id"], []).append(r)
        docs = [{
            "video_id": vid,
            "fps": spec.BOARD_FPS,
            "per_person": [{
                "window_id": i, "person_id": int(r["track_id"] or 0),
                "window_start": int(r["start"]), "window_end": int(r["end"]),
                "prob_fall": float(r["prob_fall"]),
                "coverage": float(r["coverage"] or 0.0),
                "status": STATUS_SCORED,
            } for i, r in enumerate(rs)],
        } for vid, rs in by_vid.items()]

        intervals = rv.load_fall_intervals(
            ROOT / "reference" / "board_fall_intervals.csv")

 # (a) run_video.py 的算法，直接跑在 CSV 上
        class _W2:
            def __init__(self, r):
                self.start = int(r["start"])
                self.end = int(r["end"])
                self.prob_fall = float(r["prob_fall"])

        class _R2:
            def __init__(self, vid, rs):
                self.video_id = vid
                self.windows = [_W2(r) for r in rs]

        ref_hit = ref_tot = 0
        ref_wf = ref_det = 0
        ref_nf = ref_fa = 0
        for vid, rs in by_vid.items():
            if vid in rv.EXCLUDE:
                continue
            res = _R2(vid, rs)
            hs = rv.segment_hits(res, intervals.get(vid, []))
            ref_hit += sum(h["hit"] for h in hs)
            ref_tot += len(hs)
            mx = max((w.prob_fall for w in res.windows), default=0.0)
            if intervals.get(vid):
                ref_wf += 1
                ref_det += 1 if any(h["hit"] for h in hs) else 0
            elif vid in intervals:
                ref_nf += 1
                ref_fa += 1 if mx >= spec.THRESHOLD else 0

        got = evaluate(docs, intervals,
                       MultiConfig(confirm_windows=1, cooldown_s=0.0),
                       rv.EXCLUDE)
        check("段召回（真实基线 915 窗）",
              got["segment_recall"], f"{ref_hit}/{ref_tot}")
        check("有跌倒视频", got["videos_with_fall"], f"{ref_det}/{ref_wf}")
        check("no_fall 误报", got["no_fall_false_alarms"], f"{ref_fa}/{ref_nf}")
 # 与 Q3 存档的板端数字对照（docs/上板实测报告.md）
        check("与 Q3 板端存档一致（段召回 11/17）",
              got["segment_recall"], "11/17")
    else:
        print("  SKIP  没找到 reference/board_windows.csv")

    print()
    print("SELFTEST", "PASS" if ok else "FAIL")
    return 0 if ok else 1


# ================================================================ all（板端）

def cmd_all(args) -> int:
    rv = _load_run_video()
    out = Path(args.out)
    raw = out / "raw"
    raw.mkdir(parents=True, exist_ok=True)

    videos = ([Path(args.video)] if args.video
              else rv.find_videos(Path(args.videos_dir)))
    if not videos:
        print(f"没找到视频: {args.videos_dir}", file=sys.stderr)
        return 2

    _pm = {}
    if getattr(args, "pregate", None) is not None:
        _pm["pregate"] = args.pregate
    if getattr(args, "pregate_sample", None) is not None:
        _pm["pregate_sample"] = args.pregate_sample
    mcfg = MultiConfig.from_env(
        confirm_windows=args.confirm_windows,
        retract_windows=args.retract_windows,
        cooldown_s=args.cooldown_s,
        max_persons_stgcn=args.max_persons_stgcn,
        **_pm,
    )
    print(f"视频 {len(videos)} 个")
    print(f"  YOLO  {args.yolo}")
    print(f"  STGCN {args.stgcn}")
    print(f"  multi {mcfg.describe()}")
    if mcfg.pregate:
        print("   几何前置闸门开启：被闸窗口的 prob_fall 不再是模型原值 ⇒ "
              "915/1405 窗逐位基线（V1）只在闸门关闭时成立。", flush=True)

    yolo = rv.load_rknn(Path(args.yolo))
    stgcn = rv.load_rknn(Path(args.stgcn))

    intervals = rv.load_fall_intervals(Path(args.gt))
    per_video: list[dict] = []
    seg_rows: list[dict] = []
    dump_rows: list[tuple] = []

    for i, vp in enumerate(videos, 1):
        try:
            vid = rv.video_id_from_path(vp)
        except ValueError as exc:
            print(f"[skip] {exc}", file=sys.stderr)
            continue

        info = rv.probe(vp)
 # `--fps` 覆盖容器元数据。 之前，板端测试视频是
 # `record_board_demo.py` 用硬编码 30 fps 的 VideoWriter 写的，
 # 而 USB 摄像头实际只交付 ~20 fps（文件创建→mtime 与帧数对出来
 # 19.976–20.035）⇒ 容器全部谎报 30（60 帧窗 = 3 真秒、动作快 1.5×）。
 # 语料已用 `scripts/retime_videos_fps.py` 重定时到 20 fps（帧内容
 # 逐帧校验不变）⇒ 默认读容器就是真时间轴；
 # 只有复现旧的 915 窗基线时才传 `--fps 30`。
        fps_eff = float(args.fps) if getattr(args, "fps", None) else (
            info["fps"] or spec.BOARD_FPS)
 # `--no-pose-filter` 用来复现旧口径（三道闸全关）。
        _pf = (dict(pose_min_kp=0, dedupe_contain=0.0, dedupe_iou=0.0)
               if getattr(args, "no_pose_filter", False) else {})
        cfg = PipelineConfig(fps=fps_eff,
                             score_window_frames=int(args.window_frames),
                             stitch_bridge=bool(getattr(args, "stitch_bridge",
                                                        False)),
                             async_scoring=bool(args.async_stgcn),
                             async_queue_max=int(args.async_queue_max),
                             **_pf)
        from src.backends import StgcnBackend, YoloBackend
 # 离线默认帧时间时钟：墙钟会让冷却随板子负载漂，
 # 同一份视频跑两遍事件不同 ⇒ 逐事件对账无法成立。
        clock = FrameClock(cfg.fps) if args.clock == "frame" else None
        pipe = MultiPipeline(
            cfg,
            detector=YoloBackend(yolo, conf=cfg.conf, max_persons=cfg.max_persons),
            stgcn_session=StgcnBackend(stgcn),
            mcfg=mcfg,
            clock=clock,
        )
        if getattr(args, "stitch_bridge", False):
            print(f"[stitch] allow_gap_bridge=True gap_max="
                  f"{pipe.stitcher.gap_max}（仅本臂）", flush=True)

        n = 0
        dump = dump_rows if getattr(args, "dump_tracks", "") else None
        for frame in rv.read_frames(vp, args.max_frames):
            pipe.push(frame)
            n += 1
            if dump is not None:
                idx = pipe._frame - 1
                for t in pipe.tracker._tracks:
                    if t.last_frame == idx:
                        b = t.bbox
                        dump.append((vid, idx, int(t.track_id),
                                     float(b[0]), float(b[1]),
                                     float(b[2]), float(b[3])))
        res = pipe.finish(vid)

        hits = [] if vid in rv.EXCLUDE else rv.segment_hits(
            res.base, intervals.get(vid, []))
        seg_rows.extend(hits)

        best = max((w.prob_fall for w in res.windows), default=0.0)
        row = {
            "video_id": vid, "n_frames": n, "n_windows": len(res.windows),
            "n_scored": res.base.n_scored, "n_observed": res.base.n_observed,
            "max_prob": round(best, 6),
            "segments": len(hits), "hits": sum(h["hit"] for h in hits),
            "n_persons": res.n_persons,
            "n_confirmed": len(res.confirmed()),
            "n_retracted": len(res.retracted()),
        }
        per_video.append(row)
        print(f"[{i}/{len(videos)}] {vid}  {n} 帧  {len(res.windows)} 窗  "
              f"人 {res.n_persons}  事件 {row['n_confirmed']}C/"
              f"{row['n_retracted']}R  段 {row['hits']}/{row['segments']}",
              flush=True)

 # 字段与 scripts/run_video.py 完全一致 ⇒ 可直接喂 p8/regress.py
        (raw / f"{vid.replace(':', '__')}.json").write_text(json.dumps({
            "video_id": vid, "n_frames": n, "clock": args.clock,
            "windows": [{"window_id": w.window_id, "start": w.start,
                         "end": w.end, "prob_fall": w.prob_fall,
                         "track_id": w.track_id, "coverage": w.coverage,
                         "n_candidates": w.n_candidates}
                        for w in res.windows],
            "segments": hits,
 # ---- 新增（regress.py 会忽略未知字段）----
            "fps": fps_eff,
            "container_fps": info["fps"],
            "per_person": [{
                "window_id": r.window_id, "person_id": r.person_id,
 # `window_start`/`window_end` 必须显式写出来。
 # 第一版漏了，`sweep` 只能靠 `window_id == windows[] 下标`
 # 这个隐含约定去猜 —— 那是脆的：一旦窗口表被过滤/重排就静默出错。
                "window_start": r.window_start, "window_end": r.window_end,
                "prob_fall": r.prob_fall, "coverage": r.coverage,
                "status": r.status, "members": r.members,
 # 检测框几何：否决关闭时是 None/0/False。
 # `prob_fall` 是模型原值，`vetoed` 才是「喂给状态机的那个数
 # 被换成 0」—— `_replay` 认这个键。
                "geo_ar_max": r.geo_ar_max,
                "geo_bottom_shift": r.geo_bottom_shift,
                "geo_r_end": r.geo_r_end,
                "geo_n_box": r.geo_n_box,
                "vetoed": r.vetoed,
 # 几何前置闸门（默认关）：关闭时恒为 None / None。
 # `_replay` 认 `pregate=="feed0"`（喂 0）；`skip` 走「看不见」分支。
                "pregate": r.pregate,
                "pregate_prob": r.pregate_prob,
            } for r in res.per_person],
            "events": [{
                "person_id": e.person_id, "state": e.state.label(),
                "window_start": e.window_start, "window_end": e.window_end,
                "prob_fall": e.prob_fall, "coverage": e.coverage,
                "consec_hits": e.consec_hits, "consec_miss": e.consec_miss,
            } for e in res.events],
            "counters": res.counters.as_dict(),
            "multi_cfg": res.multi_cfg,
        }, ensure_ascii=False, indent=2), encoding="utf-8")

    if seg_rows:
        with open(raw / "per_segment.csv", "w", newline="",
                  encoding="utf-8-sig") as fh:
            w = csv.DictWriter(fh, fieldnames=list(seg_rows[0]))
            w.writeheader()
            w.writerows(seg_rows)

    if getattr(args, "dump_tracks", ""):
        dump_path = Path(args.dump_tracks)
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        with open(dump_path, "w", newline="", encoding="utf-8") as fh:
            dw = csv.writer(fh)
            dw.writerow(["video_id", "frame", "track_id",
                         "x1", "y1", "x2", "y2"])
            dw.writerows(dump_rows)
        print(f"写出 {dump_path}（{len(dump_rows)} 行）", flush=True)

    with_fall = {v for v in intervals if intervals[v]} - rv.EXCLUDE
    vids_with_fall = [r for r in per_video if r["video_id"] in with_fall]
    vids_detected = [r for r in vids_with_fall if r["hits"] > 0]
    no_fall = [r for r in per_video
               if r["video_id"] not in with_fall
               and r["video_id"] not in rv.EXCLUDE and r["video_id"] in intervals]
    false_alarms = [r for r in no_fall if r["max_prob"] >= spec.THRESHOLD]

    summary = {
        "n_videos": len(per_video),
        "n_windows": sum(r["n_windows"] for r in per_video),
        "segment_recall": f"{sum(r['hits'] for r in per_video)}/"
                          f"{sum(r['segments'] for r in per_video)}",
        "videos_with_fall": f"{len(vids_detected)}/{len(vids_with_fall)}",
        "no_fall_false_alarms": f"{len(false_alarms)}/{len(no_fall)}",
 # 与 per-video JSON 的 `res.multi_cfg` 同口径：补上闸门开关，
 # 让 summary 也能自证「这批产物是闸门开还是关跑出来的」。
        "multi_cfg": {**mcfg.as_dict(),
                      "pregate": int(mcfg.pregate),
                      "pregate_sample": int(mcfg.pregate_sample)},
        "per_video": per_video,
    }
    (raw / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print()
 # ：语料已重定时到真实 20 fps ⇒ 默认是 1405 窗。
 # 915 / 14/17 / 9/9 / 3/7 是历史 30 fps 容器口径的基线，
 # 只在 `--fps 30` 时才有可比性；别把默认跑出来的数字往这四行上硬套。
    print(f"窗口 {summary['n_windows']}（历史 30fps 容器基线 915；真 20fps 实测 1405）")
    print(f"段召回 {summary['segment_recall']}（历史基线 14/17；真 20fps 实测 13/17）")
    print(f"有跌倒视频 {summary['videos_with_fall']}（历史基线 9/9；真 20fps 实测 8/9）")
    print(f"no_fall 误报 {summary['no_fall_false_alarms']}（历史基线 3/7；真 20fps 实测 4/7）")
    print(f"\n产物 {raw}")
    print("下一步（V1 硬门）：")
    print(f"  python p8/regress.py snap --from {raw} --out {out / 'r'}")
    print(f"  python p8/regress.py diff --new {out / 'r'} "
          f"--base reference/board_windows.csv --expect-zero")
    return 0


# ================================================================ sweep（V2）

def _fill_window_bounds(doc: dict) -> int:
    """老产物没有 `window_start`/`window_end` 时，从 `windows[]` 按 `window_id` 补齐。"""
    bounds = {int(w["window_id"]): (int(w["start"]), int(w["end"]))
              for w in doc.get("windows", [])}
    n = 0
    for r in doc.get("per_person", []):
        if "window_start" in r and "window_end" in r:
            continue
        b = bounds.get(int(r["window_id"]))
        if b is None:
            raise SystemExit(
                f"[FATAL] {doc.get('video_id')} window_id={r['window_id']} "
                f"在 windows[] 里找不到，无法补齐窗口边界")
        r["window_start"], r["window_end"] = b
        n += 1
    return n


def _replay(per_person, fps, mcfg):
    """在已记录的概率序列上重放状态机。时钟 = `window_end / fps`（确定性）。"""
    book = EventBook(mcfg, clock=lambda: _replay.t)
    _replay.t = 0.0
    events = []
    for r in per_person:
        status = r["status"]
        if status == STATUS_PREGATE:
 # 几何前置闸门（`FG_MULTI_PREGATE`）：`skip` 状态下状态机完全
 # 看不见这个窗口；`feed0` 与 veto 相同 —— 喂 0.0。
 # 老 dump 没有 pregate 键也走不到这里（status 不会是 "pregate"）。
            if r.get("pregate") != "feed0":
                continue
            prob = 0.0
        elif status == STATUS_SCORED:
 # 几何否决（`MultiConfig.geo_veto_ar_max`）在管线里是把喂给状态机的
 # 概率换成 0，而 `prob_fall` 保留模型原值供审计。
 # ⇒ 回放必须认 `vetoed`，否则离线评测和生产会漂移。
 # 老 dump 没有这个键 ⇒ `get` 返回 False ⇒ 行为不变。
            prob = 0.0 if r.get("vetoed") else float(r["prob_fall"])
        else:
            continue
        _replay.t = float(r.get("window_end", 0)) / max(1e-6, fps)
        events.extend(book.observe(int(r["person_id"]), prob,
                                   int(r["window_start"]),
                                   int(r["window_end"]),
                                   float(r["coverage"])))
    return events, book


def _group_by_window(per_person) -> dict[int, list[dict]]:
    """`per_person` -> `{window_id: [被评分的人, ...]}`。"""
    out: dict[int, list[dict]] = {}
    for r in per_person:
        if r["status"] == STATUS_SCORED:
            out.setdefault(int(r["window_id"]), []).append(r)
    return out


def evaluate(docs, intervals, mcfg, exclude=()) -> dict:
    """在一批 per-video 文档上评估一个 `MultiConfig`。"""
    seg_hit = seg_tot = 0
    with_fall = det = 0
    nofall = fa = 0
    n_events = n_conf = 0

    for d in docs:
        vid = d["video_id"]
        fps = float(d.get("fps") or spec.BOARD_FPS)
        events, _book = _replay(d["per_person"], fps, mcfg)
        n_events += len(events)
        conf = [e for e in events if e.state.label() == "confirmed"]
        n_conf += len(conf)

        if vid in exclude:
            continue

        vid_hit = 0
        for (s, e_) in intervals.get(vid, []):
            seg_tot += 1
            best = 0.0
            for ev in conf:
                if max(0, min(ev.window_end, e_ + 1) - max(ev.window_start, s)) > 0:
                    best = max(best, ev.prob_fall)
            if best >= spec.THRESHOLD:
                seg_hit += 1
                vid_hit += 1

        if intervals.get(vid):
            with_fall += 1
 # 与 run_video.py 同口径：段级命中（hits > 0），
 # 不是「有任何 confirmed 事件」。两者在有跌倒视频上会差 2 个
 # （9/9 vs 7/9）—— 后者把「告警了但没盖住 GT 段」也算检出。
            det += 1 if vid_hit > 0 else 0
        elif vid in intervals:
            nofall += 1
            if any(ev.prob_fall >= spec.THRESHOLD for ev in conf):
                fa += 1

    return {
        "confirm_windows": mcfg.confirm_windows,
        "segment_recall": f"{seg_hit}/{seg_tot}",
        "seg_hit": seg_hit, "seg_tot": seg_tot,
        "videos_with_fall": f"{det}/{with_fall}",
        "no_fall_false_alarms": f"{fa}/{nofall}",
        "events": n_events, "confirmed": n_conf,
    }


def cmd_sweep(args) -> int:
    rv = _load_run_video()
    src = Path(args.from_dir) / "raw"
    files = sorted(p for p in src.glob("*.json")
                   if p.name not in ("summary.json", "digest.json",
                                     "run_meta.json"))
    if not files:
        raise SystemExit(f"[FATAL] {src} 下没有 per-video JSON，先跑 all")

    intervals = rv.load_fall_intervals(
        Path(args.gt) if args.gt else ROOT / "reference" / "board_fall_intervals.csv")
    docs = [json.loads(p.read_text(encoding="utf-8")) for p in files]
    docs = [d for d in docs if "per_person" in d]
    n_filled = sum(_fill_window_bounds(d) for d in docs)
    if n_filled:
        print(f"[compat] 老产物缺 window_start/window_end，已从 windows[] 补齐 "
              f"{n_filled} 条（建议用新版 `all` 重出产物）")

 # 守卫：如果没有任何窗口有 2 个以上被评分的人，说明喂进来的是
 # 「每窗一行」的折叠产物（比如由 reference/board_windows.csv 伪造的）。
 # 那种输入下 confirm_windows 只作用于「窗口序列」，看不到多人，
 # 数字不能当 V2 结果 —— 必须喊出来，不能静默出一张看起来正常的表。
    n_multi_win = sum(
        1 for d in docs
        for _wid, rows in _group_by_window(d["per_person"]).items()
        if len(rows) >= 2)
    collapsed = n_multi_win == 0

    rows = []
    for cw in args.confirm_windows:
        mcfg = MultiConfig(confirm_windows=cw,
                           retract_windows=args.retract_windows,
                           cooldown_s=args.cooldown_s,
                           max_persons_stgcn=args.max_persons_stgcn)
        rows.append(evaluate(docs, intervals, mcfg, rv.EXCLUDE))

    out = Path(args.from_dir) / "sweep.md"
    L = ["# V2 · `CONFIRM_WINDOWS` 扫描（P8 §S2c）", ""]
    L.append(">  **`=1` 才等价于旧口径**（每个命中窗口立即告警）。"
             "`=2`/`=3` 是**收紧**，召回必然不升 —— 这是用召回换误报，"
             "**是主动选择，不是 bug**。新基线必须明确写出来，不许写「持平」。")
    L.append("")
    L.append(">  重放时钟 = `window_end / fps`，只用于**比较相对效果**；"
             "`cooldown_s` 的**绝对值**不能用回放时间轴标定（P8 §5.2(c)）。")
    L.append("")
    L.append("| confirm_windows | 段召回 | 有跌倒视频 | no_fall 误报 | "
             "confirmed 事件 | 全部事件 |")
    L.append("|---:|---|---|---|---:|---:|")
    for r in rows:
        L.append(f"| {r['confirm_windows']} | {r['segment_recall']} | "
                 f"{r['videos_with_fall']} | {r['no_fall_false_alarms']} | "
                 f"{r['confirmed']} | {r['events']} |")
    L.append("")
    L.append(f"cooldown_s = {args.cooldown_s:g}　"
             f"retract_windows = {args.retract_windows}　"
             f"多人窗口 = {n_multi_win}")
    if collapsed:
        L.append("")
        L.append(">  **本次结果不可作为 V2 结论**：输入里**没有任何多人窗口**"
                 "（`n_multi_win = 0`）。这说明喂进来的是「每窗一行」的折叠产物"
                 "（例如由 `reference/board_windows.csv` 伪造的）。")
        L.append("> 那种输入下 `confirm_windows` 只作用在窗口序列上，"
                 "**看不到多人**，`=1` 那一行虽然能复现板端 11/17，"
                 "但整张表不能用来论证确认窗口的效果。")
        L.append("> ⇒ **V2 必须用 `multi_eval.py all` 在板上跑出的产物。**")
    out.write_text("\n".join(L), encoding="utf-8")

    print("\n".join(L))
    print(f"\n-> {out}")
    return 0


# ================================================================ isolation（V3）

def isolation_check(docs, mcfg, book_cls=None):
    """V3 的核心比较。返回 (多人窗口数, 检查条目数, 坏条目数, 同时命中窗数, 明细)。"""
    Book = book_cls or EventBook
    n_multi = n_person = n_bad = both_fall = 0
    details: list[str] = []

    def replay(rows, fps):
        book = Book(mcfg, clock=lambda: replay.t)
        replay.t = 0.0
        evs = []
        for r in rows:
            if r["status"] != STATUS_SCORED:
                continue
            replay.t = float(r.get("window_end", 0)) / max(1e-6, fps)
            evs.extend(book.observe(int(r["person_id"]), float(r["prob_fall"]),
                                    int(r["window_start"]),
                                    int(r["window_end"]),
                                    float(r["coverage"])))
        return evs

    key = lambda e: (e.person_id, e.state.label(), # noqa: E731
                     e.window_start, round(e.prob_fall, 12))

    for d in docs:
        fps = float(d.get("fps") or spec.BOARD_FPS)
        pp = d["per_person"]
        by_win: dict[int, list[dict]] = {}
        for r in pp:
            if r["status"] == STATUS_SCORED:
                by_win.setdefault(int(r["window_id"]), []).append(r)
        if not any(len(v) >= 2 for v in by_win.values()):
            continue

 # 完整重放每个视频只算一次（第一版放在人循环里 ⇒ O(n²) 重放）
        full_by_pid: dict[int, list] = {}
        for e in replay(pp, fps):
            full_by_pid.setdefault(e.person_id, []).append(key(e))

        for wid, rows in by_win.items():
            if len(rows) < 2:
                continue
            n_multi += 1
            if sum(1 for r in rows if r["prob_fall"] >= spec.THRESHOLD) >= 2:
                both_fall += 1
            for r in rows:
                n_person += 1
                pid = int(r["person_id"])
                a = sorted(full_by_pid.get(pid, []))
                b = sorted(key(e) for e in
                           replay([x for x in pp if x["person_id"] == pid], fps))
                if a != b:
                    n_bad += 1
                    if len(details) < 10:
                        details.append(f"{d['video_id']} win={wid} person={pid}: "
                                       f"full={a} solo={b}")
    return n_multi, n_person, n_bad, both_fall, details


def cmd_isolation(args) -> int:
    """V3：多人窗口里逐 track 的状态迁移互不影响。"""
    src = Path(args.from_dir) / "raw"
    docs = [json.loads(p.read_text(encoding="utf-8"))
            for p in sorted(src.glob("*.json")) if p.name != "summary.json"]
    docs = [d for d in docs if "per_person" in d]
    for d in docs:
        _fill_window_bounds(d)
    if not docs:
        raise SystemExit(f"[FATAL] {src} 下没有 per_person，先跑 all")

    mcfg = MultiConfig.from_env(confirm_windows=args.confirm_windows,
                                cooldown_s=args.cooldown_s)
    n_multi, n_person, n_bad, both_fall, details = isolation_check(docs, mcfg)

    print(f"多人窗口 {n_multi}　被检查的 (窗口,人) {n_person}")
    print(f"状态被他人影响的条目 **{n_bad}**（应为 0）")
    print(f" 其中「两人同时命中」的窗口 **{both_fall}**"
          f"（现有数据实测 0 ⇒ 只能验旁观者不被污染，"
          f"验不了「两人同时跌倒」）")
    for s in details:
        print("  !", s)
    print()

 # 样本为空时绝不能报 PASS —— 那是假绿（0 条检查当然 0 条失败）。
 # 典型成因：喂进来的是 `reference/board_windows.csv` 那种「每窗一行」
 # 的折叠产物，没有 per-person 维度。V3 必须用 `all` 的产物。
    if n_person == 0:
        print("V3 SKIP —— **没有任何多人窗口可检查**，本次未验证隔离性。")
        print("      V3 需要 `multi_eval.py all` 的产物（含 per_person），"
              "不能用折叠后的 windows.csv。")
        return 2
    print("V3", "PASS" if n_bad == 0 else "FAIL")
    return 0 if n_bad == 0 else 1


# ================================================================ main

def main() -> int:
    ap = argparse.ArgumentParser(description="S2c 验收：V1 / V2 / V3")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("selftest", help="PC：不碰 NPU/视频，验 per-person 化与状态机")
    p.set_defaults(func=cmd_selftest)

    p = sub.add_parser("all", help="板端：跑视频，产物与 run_video.py 同 schema")
    p.add_argument("--video", default=None)
    p.add_argument("--videos-dir", default=str(ROOT / "videos"))
    p.add_argument("--yolo", default=str(ROOT / "models" / "yolo26n_pose_480x640_fp16.rknn"))
    p.add_argument("--stgcn", default=str(ROOT / "models" / "stgcnpp_v2_fp16.rknn"))
    p.add_argument("--gt", default=str(ROOT / "reference" / "board_fall_intervals.csv"))
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--window-frames", type=int, default=0,
                   help="ST-GCN 窗口固定帧数（0 = duration_s×fps）。"
                        "60 = 训练域采样；30fps 相机下等于现行 2s。")
    p.add_argument("--fps", type=float, default=None,
                   help="覆盖容器元数据里的帧率（窗口长度按它算）。\n"
                        "语料已于 2026-11-12 重定时到真实 20 fps（帧内容不变），\n"
                        "不传 = 用 cv2 读到的容器值，就是真时间轴。\n"
                        "只有复现旧 915 窗基线时才传 30。")
    p.add_argument("--out", default=str(ROOT / "results" / "multi"))
    p.add_argument("--no-pose-filter", action="store_true",
                   help="关掉检出净化（复现第106轮及以前的口径，见 src/pose_filter.py）")
    p.add_argument("--confirm-windows", type=int, default=2)
    p.add_argument("--retract-windows", type=int, default=4)
    p.add_argument("--cooldown-s", type=float, default=30.0)
    p.add_argument("--max-persons-stgcn", type=int, default=3,
                   help=" V1 对账必须 >= 3，否则每窗 max 会变")
    p.add_argument("--clock", choices=["frame", "wall"], default="frame",
                   help="事件冷却用的时钟。frame=帧时间（离线默认，可复现）；\n"
                        "wall=time.monotonic（旧行为，事件不可复现，只用于对照）")
    p.add_argument("--async-stgcn", action="store_true",
                   help="T3：ST-GCN 放后台线程。**判据**：与不带本开关的同一次跑\n"
                        "产物逐位相同（`--out` 指到不同目录再 diff raw/*.json）。\n"
                        " 默认关 —— 关掉时就是碰撞前的代码路径。")
    p.add_argument("--async-queue-max", type=int, default=4)
    p.add_argument("--pregate", type=int, default=None, choices=[0, 1],
                   help="几何前置闸门（reuse geo_veto_* 阈值）。默认 None = 走 "
                        "FG_MULTI_PREGATE / MultiConfig 默认（关）。\n"
                        " 打开后窗口级 prob_fall 不再是模型原值，V1 对账必须关闭。")
    p.add_argument("--pregate-sample", type=int, default=None,
                   help="影子采样：每 N 个被闸条目跑一次模型做反事实审计（推荐 8）")
    p.add_argument("--stitch-bridge", action="store_true",
                   help="② 身份桥接：Stitcher 允许零共享帧 handoff（默认关；"
                        "只影响本臂，不改生产默认）")
    p.add_argument("--dump-tracks", default="",
                   help="把逐帧 (video, frame, track_id, bbox) 写到该 CSV"
                        "（链级 A/B 用；默认不写）")
    p.set_defaults(func=cmd_all)

    p = sub.add_parser("sweep", help="V2：扫 CONFIRM_WINDOWS（不重跑 NPU）")
    p.add_argument("--from", dest="from_dir", required=True)
    p.add_argument("--confirm-windows", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--retract-windows", type=int, default=4)
    p.add_argument("--cooldown-s", type=float, default=0.0,
                   help="扫描时默认 0：只看确认窗口的效果，不混入冷却")
    p.add_argument("--max-persons-stgcn", type=int, default=3)
    p.add_argument("--gt", default=None)
    p.set_defaults(func=cmd_sweep)

    p = sub.add_parser("isolation", help="V3：多人状态隔离")
    p.add_argument("--from", dest="from_dir", required=True)
    p.add_argument("--confirm-windows", type=int, default=2)
    p.add_argument("--cooldown-s", type=float, default=0.0)
    p.set_defaults(func=cmd_isolation)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
