#!/usr/bin/env python3
"""② 身份桥接 · 单元验收（`Stitcher.allow_gap_bridge`，默认关）。"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load(path: Path, name: str):
    s = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


ME = _load(ROOT / "p8" / "multi_eval.py", "_me")
from src.multi.config import MultiConfig # noqa: E402
from src.pipeline import Pipeline, PipelineConfig # noqa: E402
from src.stitch import Stitcher # noqa: E402

FAILS = []


def check(name, got, want):
    ok = got == want
    print(f"  {'PASS' if ok else 'FAIL'}  {name}  got={got!r} want={want!r}")
    if not ok:
        FAILS.append(name)


def box(cx, cy, w=40.0, h=60.0):
    return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])


def feed(st, track_id, frames, cx=100.0, cy=100.0):
    for f in frames:
        st.observe(track_id, f, box(cx, cy))


def main() -> int:
    print("[1] 默认关：零共享帧不接")
    st = Stitcher()
    check("默认 flag", st.allow_gap_bridge, False)
    feed(st, 1, range(0, 10))
    feed(st, 2, range(13, 20)) # gap = 3，中心完全重合
    check("默认链", st.chains_so_far(), [[1], [2]])

    print("\n[2] 开启后 gap=1..5 接上")
    for gap, want in ((1, [[1, 2]]), (3, [[1, 2]]), (5, [[1, 2]]), (6, [[1], [2]])):
        st = Stitcher(allow_gap_bridge=True)
        feed(st, 1, range(0, 10)) # last = 9
        feed(st, 2, range(9 + gap, 20)) # first = 9 + gap
        check(f"gap={gap}", st.chains_so_far(), want)

    print("\n[3] 距离超限不接")
    st = Stitcher(allow_gap_bridge=True)
    feed(st, 1, range(0, 10), cx=100.0)
    feed(st, 2, range(12, 20), cx=1000.0)
    check("远距离", st.chains_so_far(), [[1], [2]])

    print("\n[4] 共享帧那支不变")
    st = Stitcher(allow_gap_bridge=True)
    feed(st, 1, range(0, 11))
    feed(st, 2, range(10, 20)) # 共享 1 帧（f=10），重复框
    check("共享 1 帧照旧接", st.chains_so_far(), [[1, 2]])
    st = Stitcher(allow_gap_bridge=True)
    feed(st, 1, range(0, 12))
    feed(st, 2, range(10, 20)) # 共享 2 帧 > overlap_max
    check("共享 2 帧照旧拒", st.chains_so_far(), [[1], [2]])

    print("\n[5] 真 handoff 优先")
    st = Stitcher(allow_gap_bridge=True)
    feed(st, 1, range(0, 6)) # 链 A 尾，last=5，桥接到 3 号
    feed(st, 2, range(0, 11)) # 链 B 尾，与 3 号共享 f=10
    feed(st, 3, range(10, 16))
    check("3 号跟共享帧链", st.chains_so_far(), [[1], [2, 3]])
    st2 = Stitcher(allow_gap_bridge=False)
    feed(st2, 1, range(0, 6))
    feed(st2, 2, range(0, 11))
    feed(st2, 3, range(10, 16))
    check("关时同结果", st2.chains_so_far(), [[1], [2, 3]])

    print("\n[6] _trim 裁帧后仍能桥")
    st = Stitcher(allow_gap_bridge=True)
    feed(st, 1, range(0, 200))
    feed(st, 2, range(202, 220)) # gap=2，1 号几何已被裁到只剩头/尾
    check("裁帧后", st.chains_so_far(), [[1, 2]])

    print("\n[7] reset 保留开关")
    st = Stitcher(allow_gap_bridge=True)
    st.reset()
    check("reset 后 flag", st.allow_gap_bridge, True)
    st.reset()
    st.reset()
    check("reset 三次仍 True", st.allow_gap_bridge, True)

    print("\n[8] 管线接线：帧率 → gap_max")
    FakeDetector, FakeStgcn = ME._fake_backends(lambda _f: 1)

    def stitcher_of(fps, on):
        pipe = Pipeline(PipelineConfig(fps=fps, stitch_bridge=on),
                        detector=FakeDetector(), stgcn_session=FakeStgcn())
        return pipe.stitcher

    check("默认关", PipelineConfig().stitch_bridge, False)
    s30 = stitcher_of(30.0, True)
    check("30fps 开 → gap 8", (s30.gap_max, s30.allow_gap_bridge), (8, True))
    s20 = stitcher_of(20.0, True)
    check("20fps 开 → gap 5", (s20.gap_max, s20.allow_gap_bridge), (5, True))
    s30o = stitcher_of(30.0, False)
    check("30fps 关 → 原 5", (s30o.gap_max, s30o.allow_gap_bridge), (5, False))

    print("\n[9] MultiConfig 环境变量")
    mc = MultiConfig.from_env({"FG_MULTI_STITCH_BRIDGE": "1"})
    check("from_env 读到 1", mc.stitch_bridge, 1)
    check("describe 打印", "stitch_bridge=1" in mc.describe(), True)
    check("as_dict 不含（云端契约不变）", "stitch_bridge" in mc.as_dict(), False)
    check("默认 0", MultiConfig().stitch_bridge, 0)
    try:
        MultiConfig.from_env({"FG_MULTI_STITCH_BRIDGE": "2"})
        check("只允许 0/1", "no error", "ValueError")
    except ValueError:
        check("只允许 0/1", "ValueError", "ValueError")

    print()
    print("STITCH-BRIDGE", "PASS" if not FAILS else f"FAIL ({len(FAILS)})")
    return 0 if not FAILS else 1


if __name__ == "__main__":
    raise SystemExit(main())
