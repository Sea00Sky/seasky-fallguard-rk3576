#!/usr/bin/env python3
"""验收 · 窗口按帧数定义（`score_window_frames`）不碰旧口径。"""
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

FAILS = []


def check(name, got, want):
    ok = got == want
    print(f"  {'PASS' if ok else 'FAIL'}  {name}  got={got!r} want={want!r}")
    if not ok:
        FAILS.append(name)


def run(fps: float, window_frames: int, n: int = 200):
    FakeDetector, FakeStgcn = ME._fake_backends(lambda _f: 1)
    cfg = PipelineConfig(fps=fps, score_window_frames=window_frames)
    pipe = Pipeline(cfg, detector=FakeDetector(), stgcn_session=FakeStgcn())
    for _ in range(n):
        pipe.push(np.zeros((240, 400, 3), np.uint8))
    return cfg, pipe.finish("fake")


def main() -> int:
    print("[1] PipelineConfig 窗口帧数")
    check("30fps 默认 2s = 60 帧", PipelineConfig(fps=30.0).window_frames, 60)
    check("20fps 默认 2s = 40 帧", PipelineConfig(fps=20.0).window_frames, 40)
    check("20fps 固定 60 帧", PipelineConfig(
        fps=20.0, score_window_frames=60).window_frames, 60)
    check("30fps 固定 60 帧 == 默认", PipelineConfig(
        fps=30.0, score_window_frames=60).window_frames,
        PipelineConfig(fps=30.0).window_frames)
    check("20fps stride 仍是 0.5s = 10 帧",
          PipelineConfig(fps=20.0, score_window_frames=60).stride_frames, 10)

    print("\n[2] 端到端窗口表（200 帧 @ 20fps）")
    _, r40 = run(20.0, 0)
    _, r60 = run(20.0, 60)
    check("旧口径 40 帧窗数", len(r40.windows), 17)
    check("新口径 60 帧窗数", len(r60.windows), 15)
    check("旧口径首窗", (r40.windows[0].start, r40.windows[0].end), (0, 40))
    check("新口径首窗", (r60.windows[0].start, r60.windows[0].end), (0, 60))
    check("新口径末窗", (r60.windows[-1].start, r60.windows[-1].end), (140, 200))

    print("\n[3] 30fps 直播口径：60 帧开关必须逐位不变")
    _, a = run(30.0, 0)
    _, b = run(30.0, 60)
    check("窗口数相同", len(a.windows), len(b.windows))
    check("窗口边界全同",
          [(w.start, w.end, w.track_id) for w in a.windows],
          [(w.start, w.end, w.track_id) for w in b.windows])
    check("概率全同",
          [w.prob_fall for w in a.windows],
          [w.prob_fall for w in b.windows])
    check("coverage 全同",
          [w.coverage for w in a.windows],
          [w.coverage for w in b.windows])

    print("\n[4] MultiConfig 环境变量")
    mc = MultiConfig.from_env({"FG_MULTI_WINDOW_FRAMES": "60"})
    check("from_env 读到 60", mc.window_frames, 60)
    check("describe 打印", "window_frames=60" in mc.describe(), True)
    check("as_dict 不含（云端契约不变）", "window_frames" in mc.as_dict(), False)
    check("默认 0", MultiConfig().window_frames, 0)
    try:
        MultiConfig.from_env({"FG_MULTI_WINDOW_FRAMES": "-1"})
        check("负数必须报错", "no error", "ValueError")
    except ValueError:
        check("负数必须报错", "ValueError", "ValueError")

    print()
    print("WINDOW-FRAMES", "PASS" if not FAILS else f"FAIL ({len(FAILS)})")
    return 0 if not FAILS else 1


if __name__ == "__main__":
    raise SystemExit(main())
