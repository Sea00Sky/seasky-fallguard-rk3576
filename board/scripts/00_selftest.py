"""在 PC 上验证板端代码的管线逻辑，不需要 rknn、不需要板子。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import Pipeline, PipelineConfig          # noqa: E402
from src import spec                             # noqa: E402
from src.gate import score_windows               # noqa: E402
from src.yolo_decode import INPUT_SHAPE, preprocess   # noqa: E402

FAILS: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    mark = "OK  " if cond else "FAIL"
    print(f"[{mark}] {name}" + (f"  {detail}" if detail else ""))
    if not cond:
        FAILS.append(name)


# --------------------------------------------------------------- 假后端


class FakeDetection:
    __slots__ = ("bbox", "keypoints", "scores", "track_id")

    def __init__(self, bbox, keypoints, scores, track_id=-1):
        self.bbox = bbox
        self.keypoints = keypoints
        self.scores = scores
        self.track_id = track_id


class FakeDetector:
    """每帧在固定位置返回一个人，关键点随帧号平移。"""

    def __init__(self, present_frames: set[int] | None = None) -> None:
        # None = 每帧都在
        self.present = present_frames
        self.n_calls = 0

    def detect(self, frame_bgr):
        i = self.n_calls
        self.n_calls += 1
        if self.present is not None and i not in self.present:
            return []
        kp = np.zeros((17, 2), np.float32)
        for j in range(17):
            kp[j] = (100 + 10 * j + i * 0.1, 200 + 5 * j)
        return [FakeDetection(
            bbox=np.array([100, 200, 200, 400], np.float32),
            keypoints=kp,
            scores=np.full(17, 0.9, np.float32))]


class FakeStgcn:
    """返回固定 logits 的假 ST-GCN。"""

    class _In:
        name = "input"

    def __init__(self, logit=(0.0, 2.0)) -> None:
        self.logit = np.array(logit, np.float32)
        self.n_calls = 0

    def get_inputs(self):
        return [self._In()]

    def run(self, _names, feeds):
        self.n_calls += 1
        x = next(iter(feeds.values()))
        assert x.shape[1:] == (1, spec.CLIP_LEN, spec.NUM_JOINTS, spec.IN_CHANNELS), \
            f"ST-GCN 输入形状 {x.shape} 不对"
        return [np.tile(self.logit, (x.shape[0], 1))]


# --------------------------------------------------------------- 各项检查


def t_imports() -> None:
    import src
    check("包可导入（无 torch/onnxruntime/rknn）", True)
    for mod in ("torch", "onnxruntime", "rknnlite"):
        check(f"未依赖 {mod}", mod not in sys.modules,
              "（若已装则此条无意义）")


def t_preprocess_identity() -> None:
    frame = np.random.randint(0, 255, (480, 640, 3), np.uint8)
    chw, params = preprocess(frame, imgsz=spec.IMGSZ)
    check("preprocess 输出形状", chw.shape == INPUT_SHAPE, f"{chw.shape}")
    check("letterbox 对 640x480 是恒等", params["ratio"] == 1.0
          and params["pad"] == (0, 0), f"r={params['ratio']} pad={params['pad']}")
    check("preprocess 值域 [0,1]", 0.0 <= chw.min() and chw.max() <= 1.0,
          f"[{chw.min():.3f}, {chw.max():.3f}]")
    # BGR -> RGB 检查：纯蓝帧应该变成 R=0,G=0,B=1 在 CHW 的最后一个通道
    blue = np.zeros((480, 640, 3), np.uint8)
    blue[:, :, 0] = 255
    chw_b, _ = preprocess(blue, imgsz=spec.IMGSZ)
    check("BGR->RGB 通道顺序正确",
          chw_b[0, 2].max() == 1.0 and chw_b[0, 0].max() == 0.0,
          f"R={chw_b[0,0].max():.1f} B={chw_b[0,2].max():.1f}")


def t_window_geometry() -> None:
    check("窗口帧数 = 60", spec.frames_for(30.0, spec.DURATION_S) == 60,
          f"{spec.frames_for(30.0, spec.DURATION_S)}")
    check("步长帧数 = 15", spec.frames_for(30.0, spec.STRIDE_S) == 15,
          f"{spec.frames_for(30.0, spec.STRIDE_S)}")
    check("CLIP_LEN = 64", spec.CLIP_LEN == 64)
    check("MODEL_INPUT_SHAPE", spec.MODEL_INPUT_SHAPE == (1, 1, 64, 17, 3),
          f"{spec.MODEL_INPUT_SHAPE}")


def t_pipeline_basic() -> None:
    """跑 300 帧，一个人一直在。"""
    cfg = PipelineConfig(fps=30.0)
    pipe = Pipeline(cfg, detector=FakeDetector(), stgcn_session=FakeStgcn())
    n = 300
    closed = 0
    for i in range(n):
        closed += len(pipe.push(np.zeros((480, 640, 3), np.uint8)))
    res = pipe.finish("synthetic")

    win, stride = cfg.window_frames, cfg.stride_frames
    expect = len(range(0, n - win + 1, stride))
    check("窗口数 = range(0, n-win+1, stride)", len(res.windows) == expect,
          f"{len(res.windows)} vs {expect}")

    check("窗口边界连续", all(res.windows[i].start == i * stride
                              for i in range(len(res.windows))))
    check("每窗都有 1 个候选", all(w.n_candidates == 1 for w in res.windows))
    # logit (0,2) -> softmax -> p(fall)=0.8808
    p = 1 / (1 + np.exp(-2.0))
    check("概率 = softmax(logits)[1]", abs(res.windows[0].prob_fall - p) < 1e-6,
          f"{res.windows[0].prob_fall:.6f} vs {p:.6f}")


def t_coverage_gate() -> None:
    """只在前 40 帧出现 -> 覆盖率 40/60 < 0.70 -> 该窗被门挡住。"""
    cfg = PipelineConfig(fps=30.0)
    present = set(range(0, 40))
    pipe = Pipeline(cfg, detector=FakeDetector(present),
                    stgcn_session=FakeStgcn())
    for i in range(60):
        pipe.push(np.zeros((480, 640, 3), np.uint8))
    res = pipe.finish("gate")

    check("覆盖率门生效（40/60=0.667 < 0.70）",
          res.windows[0].prob_fall == 0.0 and res.windows[0].n_candidates == 0,
          f"p={res.windows[0].prob_fall} n={res.windows[0].n_candidates}")

    # 42/60 = 0.70 恰好过门（>= 不是 >）
    cfg2 = PipelineConfig(fps=30.0)
    pipe2 = Pipeline(cfg2, detector=FakeDetector(set(range(42))),
                     stgcn_session=FakeStgcn())
    for i in range(60):
        pipe2.push(np.zeros((480, 640, 3), np.uint8))
    res2 = pipe2.finish("gate2")
    check("覆盖率门用 >=（42/60=0.70 通过）",
          res2.windows[0].n_candidates == 1,
          f"n={res2.windows[0].n_candidates}")


def t_max_rule() -> None:
    """两个人同窗，取较大的那个。"""
    scored = [(0, 1, 0.9, 0.2), (0, 2, 0.9, 0.8), (0, 3, 0.5, 0.99)]
    out = score_windows([(0, 0, 60)], scored, min_coverage=0.70)
    check("每窗取最大（忽略低于门的 0.99）",
          abs(out[0].prob_fall - 0.8) < 1e-9 and out[0].track_id == 2,
          f"p={out[0].prob_fall} track={out[0].track_id}")
    check("n_candidates 只数过门的", out[0].n_candidates == 2,
          f"{out[0].n_candidates}")


def t_threshold() -> None:
    """0.5 阈值用 >=。"""
    from src.gate import WindowScore
    w = WindowScore(0, 0, 60, 0.5, 1, 0.9, 1)
    check("阈值 0.5 用 >=（0.5 算命中）", w.hit is True)
    w2 = WindowScore(0, 0, 60, 0.4999, 1, 0.9, 1)
    check("0.4999 不算命中", w2.hit is False)


def t_no_model_deps() -> None:
    """确认搬过来的模块没有偷偷 import 重依赖。"""
    import ast

    heavy = {"torch", "onnxruntime", "rknn", "rknnlite", "ultralytics"}
    offenders = []
    for p in sorted((ROOT / "src").rglob("*.py")):
        tree = ast.parse(p.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    if a.name.split(".")[0] in heavy:
                        offenders.append(f"{p.name}: import {a.name}")
            elif isinstance(node, ast.ImportFrom) and node.module:
                if node.module.split(".")[0] in heavy:
                    offenders.append(f"{p.name}: from {node.module}")
    check("src/ 无重依赖 import（rknnlite 除外，它在函数内）",
          not offenders, "; ".join(offenders) if offenders else "")


def main() -> int:
    print("=" * 70)
    print("板端代码自检（PC 侧，不需要 rknn）")
    print("=" * 70)

    print("\n-- 导入 --")
    t_imports()
    print("\n-- 预处理 --")
    t_preprocess_identity()
    print("\n-- 窗口几何 --")
    t_window_geometry()
    print("\n-- 管线 --")
    t_pipeline_basic()
    print("\n-- 覆盖率门 --")
    t_coverage_gate()
    print("\n-- 每窗取最大 --")
    t_max_rule()
    print("\n-- 阈值 --")
    t_threshold()
    print("\n-- 依赖 --")
    t_no_model_deps()

    print("\n" + "=" * 70)
    if FAILS:
        print(f"FAIL: {len(FAILS)} 项")
        for f in FAILS:
            print(f"  - {f}")
        return 1
    print("全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
