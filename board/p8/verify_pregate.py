#!/usr/bin/env python3
"""几何前置闸门（`FG_MULTI_PREGATE`）的端到端自测（合成数据，不碰 NPU / 视频）。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import PipelineConfig # noqa: E402
from src.backends import Detection # noqa: E402
from src.multi import ( # noqa: E402
    STATUS_PREGATE,
    MultiConfig,
    MultiPipeline,
)

N_FRAMES = 120 # 60 帧窗 / 15 帧 stride ⇒ 5 个窗口


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


class CountingStgcn:
    """固定概率 0.9，数调用次数 —— 闸门的全部意义就是把这个数压下去。"""

    def __init__(self, p: float = 0.9):
        self.p = float(p)
        self.n_calls = 0
        self._inputs = [type("I", (), {"name": "input"})()]

    def get_inputs(self):
        return self._inputs

    def run(self, _outs, feeds):
        self.n_calls += 1
        p = float(np.clip(self.p, 1e-6, 1 - 1e-6))
        return [np.array([[np.log(1 - p), np.log(p)]], np.float32)]


def run(box_fn, mcfg):
    cfg = PipelineConfig(fps=30.0)
    stg = CountingStgcn()
    pipe = MultiPipeline(cfg, detector=FakeDetector(box_fn), stgcn_session=stg,
                         mcfg=mcfg)
    frame = np.zeros((720, 1280, 3), np.uint8)
    for _ in range(N_FRAMES):
        pipe.push(frame)
    res = pipe.finish("synth")
    return res, pipe, stg


# 形态（与 p8/verify_geo_veto.py 同一套）
UPRIGHT = lambda f: (100, 100, 150, 300) # 50x200 → ar 0.25, bs 0, r_end 1.0
WIDE = lambda f: (100, 100, 300, 160) # 200x60 → ar 3.33, bs 0, r_end 1.0

THR = dict(geo_veto_ar_max=1.80, geo_veto_bottom_shift=25.0,
           geo_veto_r_end=0.81)
# 只开时序组：用来构造「不该被闸」的人（UPRIGHT 的 ar/bs 恒命中几何组）
THR_RE = dict(geo_veto_r_end=0.81)


def stay_down(f):
    """前 15 帧站着（200 高），之后一直躺着（100 高）⇒ r_end = 0.5，不闸。"""
    h = 200 if f < 15 else 100
    return (100, 300 - h, 150, 300)


def ev_sig(res):
    return [(e.person_id, e.state.label(), e.window_start, e.window_end,
             round(e.prob_fall, 9)) for e in res.events]


def main() -> int:
    fails = []

    def check(name, got, want):
        ok = got == want
        print(f"  {'PASS' if ok else 'FAIL'}  {name}  got={got!r} want={want!r}")
        if not ok:
            fails.append(name)

 # ---------------------------------------------------------------- 1
    print("[1] 默认关闭：一个代码路径都不执行")
    r_off, p_off, s_off = run(UPRIGHT, MultiConfig())
    check("_boxhist 是 None", p_off._boxhist is None, True)
    check("pregate 字段全 None", all(x.pregate is None for x in r_off.per_person), True)
    check("pregate 计数器全 0",
          (r_off.counters.pregate_skipped, r_off.counters.pregate_feed0,
           r_off.counters.pregate_sampled), (0, 0, 0))
    check("模型照跑（5 窗）", s_off.n_calls, 5)
    check("有确认（0.9 >= 0.5）",
          len([e for e in r_off.events if e.state.label() == "confirmed"]) > 0, True)

 # ---------------------------------------------------------------- 2
    print("[2] pregate=1 但阈值全 0：与关闭逐位相同")
    r_z, _p_z, s_z = run(UPRIGHT, MultiConfig(pregate=1))
    check("模型调用数相同", s_z.n_calls, s_off.n_calls)
    check("事件逐位相同", ev_sig(r_z), ev_sig(r_off))
    check("pregate 字段仍全 None",
          all(x.pregate is None for x in r_z.per_person), True)

 # ---------------------------------------------------------------- 3
    print("[3] 几何组 → skip：不跑模型，状态机看不见")
 # 先拿到「veto 开、闸门关」的参照：每行都会被 veto
    r_veto, p_veto, s_veto = run(UPRIGHT, MultiConfig(**THR))
    n_vetoed = sum(1 for x in r_veto.per_person if x.vetoed)
    check("参照：确有被 veto 的行（否则本测试无效）", n_vetoed > 0, True)
    check("参照：模型照跑", s_veto.n_calls, 5)

    r_skip, p_skip, s_skip = run(UPRIGHT, MultiConfig(pregate=1, **THR))
    rows = [x for x in r_skip.per_person]
    check("模型 0 次调用", s_skip.n_calls, 0)
    check("行数不变（观测记录还在）", len(rows), len(r_veto.per_person))
    check("全部 status=pregate",
          all(x.status == STATUS_PREGATE for x in rows), True)
    check("全部 pregate=skip", all(x.pregate == "skip" for x in rows), True)
    check("prob_fall 全是 0.0（没测）", all(x.prob_fall == 0.0 for x in rows), True)
    check("pregate_skipped == 参照 vetoed 行数",
          r_skip.counters.pregate_skipped, n_vetoed)
    check("geo_vetoed 温度计与参照相同",
          r_skip.counters.geo_vetoed, r_veto.counters.geo_vetoed)
    check("几何证据仍写进 dump",
          all(x.geo_ar_max is not None and x.vetoed for x in rows), True)
    check("事件数 0（skip = 状态机看不见）", len(r_skip.events), 0)
    check("参照事件数也是 0（veto 喂 0）", len(r_veto.events), 0)
    check("persons_scored：skip 不记 / veto 照记",
          (r_skip.counters.persons_scored, r_veto.counters.persons_scored),
          (0, 5))
 # 窗口级按设计会变：skip 的窗口没有模型分数
    w_on = max(w.prob_fall for w in r_skip.windows)
    w_off = max(w.prob_fall for w in r_veto.windows)
    check("窗口级按设计改变（skip 窗口没有模型分数）",
          (w_on == 0.0 and abs(w_off - 0.9) < 1e-6), True)

 # ---------------------------------------------------------------- 4
    print("[4] r_end 组 → feed0：不跑模型，事件与 veto 逐位相同")
    r_vf, _p_vf, s_vf = run(WIDE, MultiConfig(**THR))
    r_f0, _p_f0, s_f0 = run(WIDE, MultiConfig(pregate=1, **THR))
    check("参照：全部被 veto", all(x.vetoed for x in r_vf.per_person), True)
    check("闸门：模型 0 次调用", s_f0.n_calls, 0)
    check("闸门：全部 pregate=feed0",
          all(x.pregate == "feed0" for x in r_f0.per_person), True)
    check("闸门：pregate_feed0 计数",
          r_f0.counters.pregate_feed0, len(r_vf.per_person))
    check("事件与 veto 参照逐位相同", ev_sig(r_f0), ev_sig(r_vf))
    check("persons_scored 与 veto 参照相同（feed0 = 喂 0）",
          r_f0.counters.persons_scored, r_vf.counters.persons_scored)

 # ---------------------------------------------------------------- 5
    print("[5] 未被闸的人：模型照跑，事件不变")
 # 只开 r_end：UPRIGHT 形态的 ar/bs 恒命中几何组，测不出「不闸」。
    r_nd_off, _p_nd, s_nd_off = run(stay_down, MultiConfig(**THR_RE))
    r_nd_on, _p_nd2, s_nd_on = run(stay_down, MultiConfig(pregate=1, **THR_RE))
    check("参照：没有行被 veto", any(x.vetoed for x in r_nd_off.per_person), False)
    check("闸门：模型调用数相同", s_nd_on.n_calls, s_nd_off.n_calls)
    check("闸门：没有行被 pregate",
          any(x.pregate is not None for x in r_nd_on.per_person), False)
    check("闸门：事件逐位相同", ev_sig(r_nd_on), ev_sig(r_nd_off))
    check("参考：确有确认（否则本测试无效）",
          len([e for e in r_nd_off.events if e.state.label() == "confirmed"]) > 0,
          True)

 # ---------------------------------------------------------------- 6
    print("[6] 影子采样 1/2：只跑反事实，不改任何判定")
    r_sh, _p_sh, s_sh = run(UPRIGHT, MultiConfig(pregate=1, pregate_sample=2, **THR))
    n_gated = len([x for x in r_sh.per_person if x.pregate is not None])
    sampled = [x for x in r_sh.per_person if x.pregate_prob is not None]
    check("被闸条目数 = 5", n_gated, 5)
    check("采样数 = ceil(5/2) = 2", len(sampled), 2)
    check("模型调用数 = 采样数", s_sh.n_calls, len(sampled))
    check("pregate_sampled 计数", r_sh.counters.pregate_sampled, len(sampled))
    check("反事实分数 = 0.9", all(abs(x.pregate_prob - 0.9) < 1e-6 for x in sampled),
          True)
    check("prob_fall 仍是 0.0（反事实不喂状态机）",
          all(x.prob_fall == 0.0 for x in r_sh.per_person), True)
    check("事件与 sample=0 完全相同", ev_sig(r_sh), ev_sig(r_skip))
    check("影子样本不进 _scored（窗口级仍是 0）",
          max(w.prob_fall for w in r_sh.windows), 0.0)

 # ---------------------------------------------------------------- 7
    print("[7] 配置校验")
    try:
        MultiConfig(pregate=2)
        check("pregate=2 报错", False, True)
    except ValueError:
        check("pregate=2 报错", True, True)
    try:
        MultiConfig(pregate_sample=-1)
        check("pregate_sample=-1 报错", False, True)
    except ValueError:
        check("pregate_sample=-1 报错", True, True)
    check("describe 含 pregate",
          "pregate=1" in MultiConfig(pregate=1, pregate_sample=8).describe(), True)
    check("as_dict 不含 pregate（跨端哈希契约）",
          "pregate" in MultiConfig(pregate=1).as_dict(), False)

    print()
    if fails:
        print(f"FAILED: {fails}")
        return 1
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
