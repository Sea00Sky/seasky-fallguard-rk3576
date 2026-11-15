#!/usr/bin/env python3
"""T3 后台线程打分的等价性验证（PC，不需要 NPU / 摄像头 / 视频文件）。"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import PipelineConfig # noqa: E402
from src.backends import Detection # noqa: E402
from src.multi import ( # noqa: E402
    FrameClock,
    MultiConfig,
    MultiPipeline,
)

FPS = 30.0


# --------------------------------------------------------------------------- 假体
class _Inp:
    name = "input"


class FakeStgcn:
    """分数 = 输入张量的确定性函数。同一张量 ⇒ 逐位同一分数。"""

    def __init__(self, sleep_s: float = 0.0) -> None:
        self.sleep_s = float(sleep_s)
        self.n_calls = 0
        self._inp = _Inp()
        self._w = None

    def get_inputs(self):
        return [self._inp]

    def run(self, _names, feeds):
        x = np.asarray(next(iter(feeds.values())), np.float64)
        if self.sleep_s:
            time.sleep(self.sleep_s) # 模拟 NPU 延迟，把后台队列逼满
        if self._w is None or self._w.size != x.size:
            self._w = np.linspace(0.5, 1.5, x.size)
        s = float(np.dot(x.ravel(), self._w))
        p = (math.sin(s * 12.9898) * 43758.5453) % 1.0
        p = min(max(p, 1e-6), 1.0 - 1e-6)
        self.n_calls += 1
        return [np.array([[math.log1p(-p), math.log(p)]], np.float64)]


class Actor:
    """一条合成轨迹：`[birth, death)` 内每帧出一个框，`visible(i)` 决定可见性。"""

    def __init__(self, birth, death, box, visible=None) -> None:
        self.birth = int(birth)
        self.death = int(death)
        self.box = box
        self.visible = visible or (lambda i: True)


class SynthDetector:
    """无状态、按帧号确定的检测器 —— 两次跑喂进去的框逐位相同。"""

    def __init__(self, actors) -> None:
        self.actors = list(actors)
        self.i = 0

    def detect(self, _frame_bgr):
        i = self.i
        self.i += 1
        out = []
        for a in self.actors:
            if not (a.birth <= i < a.death) or not a.visible(i):
                continue
            x1, y1, x2, y2 = a.box(i)
            kp = np.empty((17, 2), np.float32)
            kp[:, 0] = x1 + (np.arange(17) / 16.0) * (x2 - x1)
            kp[:, 1] = y1 + ((np.arange(17) * 7 % 17) / 16.0) * (y2 - y1)
            sc = np.full(17, 0.75, np.float32)
            out.append(Detection(np.array([x1, y1, x2, y2], np.float32), kp, sc))
        return out


# ----------------------------------------------------------------------- 场景
def _crowd_actors():
    """4 个人，横向排开、缓慢移动；互不重叠到 IoU 混淆的程度。"""
    return [
        Actor(0, 10 ** 9, lambda i: (60 + 0.20 * i, 100, 140 + 0.20 * i, 340)),
        Actor(0, 10 ** 9, lambda i: (180 - 0.15 * i, 120, 260 - 0.15 * i, 360)),
        Actor(0, 10 ** 9, lambda i: (320 + 0.10 * i, 90, 400 + 0.10 * i, 330)),
        Actor(0, 10 ** 9, lambda i: (450 - 0.12 * i, 110, 530 - 0.12 * i, 350)),
    ]


def _gap_actors():
    """一个几乎全程可见，一个 5 帧可见 / 9 帧丢失（覆盖率掉到门外）。"""
    return [
        Actor(0, 10 ** 9, lambda i: (80 + 0.1 * i, 100, 160 + 0.1 * i, 340)),
        Actor(0, 10 ** 9, lambda i: (300, 120, 380, 360),
              visible=lambda i: (i % 14) < 5),
    ]


GEO = dict(geo_veto_ar_max=1.8, geo_veto_bottom_shift=25.0,
           geo_veto_r_end=0.81, geo_veto_min_box=20)

SCENARIOS = [
    ("single", dict(
        sleep=0.0, queue=4, actors=_crowd_actors()[:1],
        mcfg=MultiConfig(confirm_windows=2, retract_windows=4, cooldown_s=0.0,
                         max_persons_stgcn=3))),
    ("crowd-cap2", dict(
        sleep=0.0, queue=4, actors=_crowd_actors(),
        mcfg=MultiConfig(confirm_windows=2, retract_windows=4, cooldown_s=0.0,
                         max_persons_stgcn=2, **GEO))),
    ("gaps", dict(
        sleep=0.0, queue=4, actors=_gap_actors(),
        mcfg=MultiConfig(confirm_windows=2, retract_windows=4, cooldown_s=0.0,
                         max_persons_stgcn=3, **GEO))),
    ("slow-q1", dict(
        sleep=0.05, queue=1, actors=_crowd_actors()[:3],
        mcfg=MultiConfig(confirm_windows=2, retract_windows=4, cooldown_s=0.0,
                         max_persons_stgcn=3))),
 # 几何前置闸门 + 影子采样：`_pregate` / `_shadow` / `_geo_cache` 都是
 # 主线程状态，异步下 `_on_window` 晚 1–2 帧 ⇒ 同步/异步必须逐位相同。
    ("crowd-pregate", dict(
        sleep=0.0, queue=4, actors=_crowd_actors(),
        mcfg=MultiConfig(confirm_windows=2, retract_windows=4, cooldown_s=0.0,
                         max_persons_stgcn=3, pregate=1, pregate_sample=2, **GEO))),
]


# ----------------------------------------------------------------------- 跑与比
class _RecordingPipe(MultiPipeline):
    """记下每个窗口「结果比窗口晚了几帧落地」。"""

    def __init__(self, *a, **k) -> None:
        self.defer_by = []
        super().__init__(*a, **k)

    def _on_window(self, wid, start, end, obs, items, probs, extra):
        super()._on_window(wid, start, end, obs, items, probs, extra)
        self.defer_by.append(self._frame - int(end))


def run_once(scn, n_frames, async_scoring):
    det = SynthDetector(scn["actors"])
    sg = FakeStgcn(scn["sleep"])
    pcfg = PipelineConfig(fps=FPS, async_scoring=async_scoring,
                          async_queue_max=scn["queue"])
    pipe = _RecordingPipe(pcfg, detector=det, stgcn_session=sg,
                          mcfg=scn["mcfg"], clock=FrameClock(FPS))
    frame = np.zeros((480, 640, 3), np.uint8)
    t0 = time.perf_counter()
    for _ in range(n_frames):
        pipe.push(frame)
    res = pipe.finish("synth")
    return pipe, res, sg, time.perf_counter() - t0


def snapshot(pipe, res):
    """把结果摊平成可逐位比较的结构。不做 round —— 判据就是逐位相同。"""
    return {
        "n_frames": res.base.n_frames,
        "windows": [(w.window_id, w.start, w.end, w.prob_fall, w.track_id,
                     w.coverage, w.n_candidates) for w in res.base.windows],
        "per_person": [(p.window_id, p.window_start, p.window_end, p.person_id,
                        p.prob_fall, p.coverage, p.status, tuple(p.members),
                        p.geo_ar_max, p.geo_bottom_shift, p.geo_r_end,
                        p.geo_n_box, p.vetoed, p.pregate, p.pregate_prob)
                       for p in res.per_person],
        "events": [(e.person_id, e.state.label(), e.window_start, e.window_end,
                    e.prob_fall, e.coverage, e.consec_hits, e.consec_miss)
                   for e in res.events],
        "counters": {k: v for k, v in vars(res.counters).items()},
        "scored": [(a, b, c, d) for a, b, c, d in pipe._scored],
        "observed": [(a, b, c) for a, b, c in pipe._observed],
        "n_scored": res.base.n_scored,
        "n_observed": res.base.n_observed,
        "n_persons": res.n_persons,
        "chains": [list(c) for c in res.base.chains],
        "people": {int(k): (v["track"], tuple(v["members"]))
                   for k, v in res.base.window_people.items()},
    }


def diff(sa, sb):
    """返回第一处不同的 (字段, A, B)，全同则 None。"""
    for key in sa:
        if sa[key] == sb[key]:
            continue
        a, b = sa[key], sb[key]
        if isinstance(a, list) and isinstance(b, list):
            for i, (x, y) in enumerate(zip(a, b)):
                if x != y:
                    return key, f"[{i}] {x}", f"[{i}] {y}"
            return key, f"len={len(a)}", f"len={len(b)}"
        return key, a, b
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description="T3 后台线程等价性验证")
    ap.add_argument("--frames", type=int, default=600,
                    help="每个场景的帧数（默认 600 = 20 个窗口）")
    args = ap.parse_args()

    fails = 0
    for name, scn in SCENARIOS:
        print(f"\n[{name}] {args.frames} 帧  sleep={scn['sleep']}s  "
              f"queue={scn['queue']}  "
              f"cap={scn['mcfg'].max_persons_stgcn}  "
              f"geo={'on' if scn['mcfg'].geo_veto_ar_max > 0 else 'off'}",
              flush=True)

        p_s, r_s, g_s, t_s = run_once(scn, args.frames, False)
        p_a, r_a, g_a, t_a = run_once(scn, args.frames, True)

        sa, sb = snapshot(p_s, r_s), snapshot(p_a, r_a)
        d = diff(sa, sb)

        n_win = len(sa["windows"])
        n_pp = len(sa["per_person"])
        n_ev = len(sa["events"])
        n_cap = sum(1 for p in r_s.per_person if p.status == "capped")
        n_gate = sum(1 for p in r_s.per_person if p.status == "gated")
        n_pre = sum(1 for p in r_s.per_person if p.status == "pregate")
        print(f"  窗口 {n_win} · 逐人记录 {n_pp}（scored/gated/capped/pregate = "
              f"{n_pp - n_cap - n_gate - n_pre}/{n_gate}/{n_cap}/{n_pre}）· "
              f"事件 {n_ev} · "
              f"stgcn 调用 {g_s.n_calls} vs {g_a.n_calls} · "
              f"耗时 {t_s:.1f}s vs {t_a:.1f}s", flush=True)

 # 迟到帧数：同步必须全 0；异步必须不全 0，否则这条 PASS 是空的。
        ds, da = p_s.defer_by, p_a.defer_by
        late_a = sum(1 for x in da if x > 0)
        print(f"  迟到帧数：同步 max={max(ds, default=0)} 全零={all(x == 0 for x in ds)}"
              f" · 异步 max={max(da, default=0)} 迟到窗 {late_a}/{len(da)}",
              flush=True)

        if d is not None:
            fails += 1
            print(f"  FAIL  字段 `{d[0]}`\n        同步: {d[1]}\n"
                  f"        异步: {d[2]}", flush=True)
        elif any(x != 0 for x in ds):
            fails += 1
            print("  FAIL  同步模式出现了迟到窗口（说明异步分支被误触发）", flush=True)
        elif late_a == 0:
            fails += 1
            print("  FAIL  异步模式一个迟到窗口都没有 —— 后台线程没被真正用上，"
                  "这条 PASS 是空的", flush=True)
        else:
            print("  PASS  同步 ≡ 异步（逐位），且异步确实延迟落地", flush=True)

    print()
    if fails:
        print(f"=== {fails} 个场景不一致 ===")
        return 1
    print("=== 全部场景 同步 ≡ 异步 ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
