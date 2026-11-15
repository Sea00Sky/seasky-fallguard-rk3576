#!/usr/bin/env python3
"""几何否决通道的端到端自测（合成数据，不碰 NPU / 视频）。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import PipelineConfig                      # noqa: E402
from src.backends import Detection                  # noqa: E402
from src.multi import MultiConfig, MultiPipeline    # noqa: E402


class FakeDetector:
    """每帧返回一个人，框由 `box_fn(frame_index)` 决定。"""

    def __init__(self, box_fn):
        self.box_fn = box_fn
        self.i = -1

    def detect(self, frame_bgr):
        self.i += 1
        x1, y1, x2, y2 = self.box_fn(self.i)
        kp = np.zeros((17, 2), np.float32)
        sc = np.ones(17, np.float32)
        return [Detection(np.asarray([x1, y1, x2, y2], np.float32), kp, sc)]


class _Inp:
    name = "input"


class DummyStgcn:
    """父类 `__init__` 只读 `get_inputs[0].name`；`_score_batch` 被下面覆盖。"""

    def get_inputs(self):
        return [_Inp()]


class FakePipe(MultiPipeline):
    """绕过 NPU：直接给每个人 0.9。"""

    def _score_batch(self, items):
        return [0.9] * len(items)


def run(box_fn, mcfg, n_frames=120):
    cfg = PipelineConfig(fps=30.0)
    pipe = FakePipe(cfg, detector=FakeDetector(box_fn), stgcn_session=DummyStgcn(),
                    mcfg=mcfg)
    frame = np.zeros((720, 1280, 3), np.uint8)
    for _ in range(n_frames):
        pipe.push(frame)
    res = pipe.finish("synth")
    return res, pipe


UPRIGHT = lambda f: (100, 100, 150, 300)        # 50x200 → ar 0.25
LYING = lambda f: (100, 100, 300, 160)          # 200x60 → ar 3.33


def main():
    fails = []

    def check(name, got, want):
        ok = got == want
        print(f"  {'PASS' if ok else 'FAIL'}  {name}  got={got} want={want}")
        if not ok:
            fails.append(name)

    print("[1] 默认关闭：一个代码路径都不执行")
    r_off, p_off = run(UPRIGHT, MultiConfig())
    check("geo_ar_max 全 None", all(x.geo_ar_max is None for x in r_off.per_person), True)
    check("vetoed 全 False", all(not x.vetoed for x in r_off.per_person), True)
    check("_boxhist 是 None", p_off._boxhist is None, True)
    n_conf_off = len([e for e in r_off.events if e.state.label() == "confirmed"])

    print("[2] 关闭 vs 关闭：两次跑逐位相同")
    r_off2, _ = run(UPRIGHT, MultiConfig())
    same = [(e.person_id, e.state.label(), e.window_start, e.prob_fall)
            for e in r_off.events] == [(e.person_id, e.state.label(), e.window_start, e.prob_fall)
                                       for e in r_off2.events]
    check("事件序列相同", same, True)
    check("直立也有人确认（关闭时不该被否决）", n_conf_off > 0, True)

    print("[3] 开启后：直立被否决，躺下不被否决")
    r_up, _ = run(UPRIGHT, MultiConfig(geo_veto_ar_max=0.90))
    check("直立：全部被否决", all(x.vetoed for x in r_up.per_person if x.geo_ar_max is not None), True)
    check("直立：一条确认都没有", len([e for e in r_up.events if e.state.label() == "confirmed"]), 0)
    check("直立：geo_vetoed 计数 > 0", r_up.counters.geo_vetoed > 0, True)

    r_ly, _ = run(LYING, MultiConfig(geo_veto_ar_max=0.90))
    check("躺下：一条都没被否决", any(x.vetoed for x in r_ly.per_person), False)
    check("躺下：仍有确认", len([e for e in r_ly.events if e.state.label() == "confirmed"]) > 0, True)

    print("[4] 安全阀：框太少不否决")
    # 只有前 5 帧有框，其余帧 detector 返回一个 1x1 的退化框（会被 w<1/h<1 过滤）
    def sparse(f):
        return (100, 100, 150, 300) if f < 5 else (0, 0, 0, 0)
    r_sp, _ = run(sparse, MultiConfig(geo_veto_ar_max=0.90, geo_veto_min_box=20))
    vetoed_sp = [x for x in r_sp.per_person if x.vetoed]
    check("稀疏框：没有被否决", len(vetoed_sp), 0)
    check("稀疏框：n_box < 20", all(x.geo_n_box < 20 for x in r_sp.per_person), True)

    print("[5] 组合规则 ar_max<2.00 且 bottom_shift<25：三种形态")
    # 三种形态都逐帧渐变，不能跳变 —— 跳变会让 IoU tracker 换 track id，
    # 窗口就只看得到其中一段，测的就不是几何判据了。

    # 跌倒·向外倒：框从竖变横，框底钉在 300
    def fall_out(f):
        t = min(1.0, f / 59.0)
        h = 200 * (1 - t) + 60 * t
        w = 50 * (1 - t) + 200 * t
        return (100, 300 - h, 100 + w, 300)
    r_fo, _ = run(fall_out, MultiConfig(geo_veto_ar_max=2.00,
                                        geo_veto_bottom_shift=25))
    check("向外倒：不被否决（ar 大）", any(x.vetoed for x in r_fo.per_person), False)
    check("向外倒：仍有确认", len([e for e in r_fo.events
                                   if e.state.label() == "confirmed"]) > 0, True)

    # 跌倒·滑出：框大小不变，框底持续从 300 滑到 400
    # 必须让整个 120 帧都在滑 —— 只滑前 60 帧的话，后面的窗口是静止的，
    # bshift=0 会被正当否决，测出来的是"静止的人"不是"滑出"。
    def fall_slide(f):
        t = min(1.0, f / 119.0)
        y2 = 300 + 100 * t
        return (100, y2 - 100, 180, y2)
    r_fs, _ = run(fall_slide, MultiConfig(geo_veto_ar_max=2.00,
                                          geo_veto_bottom_shift=25))
    check("滑出：不被否决（bshift 大）", any(x.vetoed for x in r_fs.per_person), False)

    # 下蹲：框从竖变扁变宽（末态 ar≈1.3），框底钉在 300
    def squat(f):
        t = min(1.0, f / 59.0)
        h = 200 * (1 - t) + 100 * t
        w = 50 * (1 - t) + 130 * t
        return (100, 300 - h, 100 + w, 300)
    r_sq, _ = run(squat, MultiConfig(geo_veto_ar_max=2.00,
                                     geo_veto_bottom_shift=25))
    check("下蹲：被否决", all(x.vetoed for x in r_sq.per_person
                              if x.geo_ar_max is not None), True)
    check("下蹲：一条确认都没有",
          len([e for e in r_sq.events if e.state.label() == "confirmed"]), 0)

    print("[6] 只开一个条件时的行为")
    r_a, _ = run(squat, MultiConfig(geo_veto_ar_max=2.00))
    check("只开 ar_max=2.00：下蹲仍被否决", any(x.vetoed for x in r_a.per_person), True)
    r_b, _ = run(squat, MultiConfig(geo_veto_bottom_shift=25))
    check("只开 bshift=25：下蹲仍被否决", any(x.vetoed for x in r_b.per_person), True)
    r_n, _ = run(squat, MultiConfig())
    check("两个都关：不否决", any(x.vetoed for x in r_n.per_person), False)

    print("[7] 时序条件 r_end：窗口结束时人站没站起来")
    # 跌倒后 1.5 秒内自己站起来：每个窗口结束时都是直立的 ⇒ r_end≈1.0
    def fall_then_up(f):
        h = 200 if (f < 15 or f >= 45) else 100
        return (100, 300 - h, 150, 300)
    r_up, _ = run(fall_then_up, MultiConfig(geo_veto_r_end=0.81))
    check("站起来了：被否决", any(x.vetoed for x in r_up.per_person), True)
    check("站起来了：没有确认",
          len([e for e in r_up.events if e.state.label() == "confirmed"]), 0)

    # 倒下后一直躺着：窗口结束时仍是 0.5 ⇒ 不否决
    def stay_down(f):
        h = 200 if f < 15 else 100
        return (100, 300 - h, 150, 300)
    r_dn, _ = run(stay_down, MultiConfig(geo_veto_r_end=0.81))
    check("还躺着：不被否决", any(x.vetoed for x in r_dn.per_person), False)
    check("还躺着：仍有确认", len([e for e in r_dn.events
                                   if e.state.label() == "confirmed"]) > 0, True)

    print("[8] r_end 与几何组是 OR：任一成立就否决")
    # 框很窄(ar=0.25)但人站着 —— 几何组和时序组都成立
    r_or, _ = run(lambda f: (100, 100, 150, 300),
                  MultiConfig(geo_veto_ar_max=1.80, geo_veto_bottom_shift=25,
                              geo_veto_r_end=0.81))
    check("几何+时序都成立：否决", any(x.vetoed for x in r_or.per_person), True)

    print()
    if fails:
        print(f"FAILED: {fails}")
        return 1
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
