"""板端时延测量：微基准 + 端到端逐帧 trace。"""
from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import Pipeline, PipelineConfig            # noqa: E402
from src import spec                                # noqa: E402
from src.backends import StgcnBackend, YoloBackend  # noqa: E402
from src.backends.rknn_backend import Detection     # noqa: E402
from src.yolo_decode import INPUT_SHAPE, preprocess  # noqa: E402

STGCN_INPUT_SHAPE = (1, 1, spec.CLIP_LEN, spec.NUM_JOINTS, spec.IN_CHANNELS)


# ------------------------------------------------------------------ 统计


def stats(samples_ns: list[int]) -> dict:
    """纳秒样本 -> 毫秒统计量。空样本返回 n=0，不抛异常。"""
    if not samples_ns:
        return {"n": 0, "p50": None, "p95": None, "p99": None,
                "mean": None, "max": None, "min": None}
    ms = sorted(s / 1e6 for s in samples_ns)
    n = len(ms)
    return {
        "n": n,
        "p50": round(statistics.median(ms), 4),
        "p95": round(ms[min(n - 1, int(round(0.95 * (n - 1))))], 4),
        "p99": round(ms[min(n - 1, int(round(0.99 * (n - 1))))], 4),
        "mean": round(sum(ms) / n, 4),
        "max": round(ms[-1], 4),
        "min": round(ms[0], 4),
    }


def _fmt(s: dict) -> str:
    if not s.get("n"):
        return "n=0"
    return (f"p50={s['p50']:.2f} p95={s['p95']:.2f} p99={s['p99']:.2f} "
            f"mean={s['mean']:.2f} max={s['max']:.2f} ms (n={s['n']})")


# ------------------------------------------------------- 计时代理（核心）


class _TimedRknn:
    """转发 `inference` 并累加纳秒的 RKNN 代理。"""

    __slots__ = ("inner", "samples_ns", "n_calls")

    def __init__(self, inner) -> None:
        self.inner = inner
        self.samples_ns: list[int] = []
        self.n_calls = 0

    def reset(self) -> None:
        self.samples_ns = []
        self.n_calls = 0

    def inference(self, **kwargs):
        t0 = time.perf_counter_ns()
        out = self.inner.inference(**kwargs)
        self.samples_ns.append(time.perf_counter_ns() - t0)
        self.n_calls += 1
        return out

    def __getattr__(self, name):
        # `__getattr__` 只在正常查找失败时触发，所以不会遮蔽上面两个属性。
        return getattr(self.inner, name)


class _TimedBackend:
    """在 `detect` / `run` 外面套一层总时延，用于求 CPU 侧成本。"""

    def __init__(self, inner, timed_rknn: _TimedRknn) -> None:
        self.inner = inner
        self.timed_rknn = timed_rknn
        self.samples_ns: list[int] = []

    def reset(self) -> None:
        self.samples_ns = []
        self.timed_rknn.reset()

    def detect(self, frame_bgr):
        t0 = time.perf_counter_ns()
        out = self.inner.detect(frame_bgr)
        self.samples_ns.append(time.perf_counter_ns() - t0)
        return out

    def get_inputs(self):
        return self.inner.get_inputs()

    def run(self, output_names, feeds):
        t0 = time.perf_counter_ns()
        out = self.inner.run(output_names, feeds)
        self.samples_ns.append(time.perf_counter_ns() - t0)
        return out

    def __getattr__(self, name):
        return getattr(self.inner, name)


# ------------------------------------------------------------------ 假后端


class FakeRKNN:
    """PC 自检用。返回固定形状、固定数值，不碰 NPU。"""

    def __init__(self, out_shape, delay_ms: float = 0.5,
                 in_shape: tuple | None = None) -> None:
        self.out_shape = out_shape
        self.in_shape = in_shape
        self.delay_s = delay_ms / 1000.0
        self.n = 0

    def inference(self, inputs=None, data_format=None):
        assert data_format == "nchw", f"data_format 必须是 nchw，收到 {data_format!r}"
        x = np.asarray(inputs[0])
        if self.in_shape is not None:
            assert x.shape == self.in_shape, \
                f"输入形状 {x.shape}，期望 {self.in_shape}"
        time.sleep(self.delay_s)
        self.n += 1
        out = np.zeros(self.out_shape, np.float32)
        if self.out_shape[1:] == (56, 6300):
            # YOLO 布局：4 box + 1 conf + 17×3 kpt。
            # 只让前 3 个 anchor 过 conf>0.25，其余为 0——否则 decode 会对
            # 6300 个框做排序，dry-run 变成在测 CPU 排序而不是在验代码路径。
            out[0, 4, :3] = [0.9, 0.8, 0.7]
            # 布局是 (1, 56, 6300) ⇒ 每行是一个属性，每列是一个 anchor。
            out[0, 0:4, :3] = [[100, 110, 120],    # x1
                               [200, 210, 220],    # y1
                               [300, 310, 320],    # x2
                               [400, 410, 420]]    # y2
            out[0, 5:, :3] = 0.8
        else:
            out[...] = float(x.mean())
        return [out]


class FakeStgcnRKNN(FakeRKNN):
    """ST-GCN 的假会话：输出 (1,2) logits。"""

    def __init__(self, delay_ms: float = 0.3) -> None:
        super().__init__((1, 2), delay_ms)

    def inference(self, inputs=None, data_format=None):
        assert data_format == "nchw"
        time.sleep(self.delay_s)
        self.n += 1
        # logits (0, 2) -> p(fall) = 0.8808，与 00_selftest 的期望一致
        return [np.array([[0.0, 2.0]], np.float32)]


# ------------------------------------------------------------------ RKNN


def load_rknn(path: Path, core_mask=None):
    """加载 + 初始化。与 `run_video.py::load_rknn` 同构。"""
    from rknnlite.api import RKNNLite

    r = RKNNLite()
    if r.load_rknn(str(path)) != 0:
        raise RuntimeError(f"load_rknn 失败: {path}")
    kw = {} if core_mask is None else {"core_mask": core_mask}
    if r.init_runtime(**kw) != 0:
        raise RuntimeError(
            f"init_runtime 失败: {path}（core_mask={core_mask}）。"
            f"先查 librknnrt 版本，再查 core_mask 是不是合法常量。")
    return r


def print_core_constants() -> dict:
    """打印 rknnlite 真实的 NPU_CORE_* 常量，并返回字典。"""
    try:
        from rknnlite.api import RKNNLite
    except ImportError:
        print("PC 上没有 rknnlite（这是正常的）。"
              "这条命令要在板上跑：python3 scripts/03_latency.py --print-cores",
              file=sys.stderr)
        return {}
    table = {k: getattr(RKNNLite, k) for k in dir(RKNNLite)
             if k.startswith("NPU")}
    print("RKNNLite 常量表（写报告时按这张表解释 core_mask）：")
    for k, v in sorted(table.items(), key=lambda kv: str(kv[1])):
        print(f"  {k:24s} = {v}")
    return table


# ------------------------------------------------------------------ 微基准


def bench_yolo(rknn, frame_bgr: np.ndarray, n: int, warmup: int) -> dict:
    """N1（NPU-only）与 N1c（含前后处理）。固定同一张帧。"""
    chw, _params = preprocess(frame_bgr, imgsz=spec.IMGSZ)
    assert chw.shape == INPUT_SHAPE, f"预处理产出 {chw.shape}，图要的是 {INPUT_SHAPE}"

    timed = _TimedRknn(rknn)
    backend = YoloBackend(timed, conf=spec.CONF, max_persons=spec.MAX_PERSONS)

    for _ in range(warmup):
        backend.detect(frame_bgr)
    timed.reset()
    outer = _TimedBackend(backend, timed)
    outer.reset()
    for _ in range(n):
        outer.detect(frame_bgr)

    npu = stats(timed.samples_ns)
    total = stats(outer.samples_ns)
    cpu = None
    if npu["n"] and total["n"]:
        cpu = {k: round(total[k] - npu[k], 4)
               for k in ("p50", "p95", "p99", "mean")}
    return {"npu_only": npu, "with_prepost": total, "cpu_prepost": cpu,
            "n_detections": len(backend.detect(frame_bgr))}


def bench_stgcn(rknn, window_tensor: np.ndarray, n: int, warmup: int) -> dict:
    """N2（NPU-only）与 N2c（含 softmax）。固定同一个窗口张量。"""
    assert window_tensor.shape == STGCN_INPUT_SHAPE, \
        f"窗口张量 {window_tensor.shape}，图要的是 {STGCN_INPUT_SHAPE}"

    timed = _TimedRknn(rknn)
    backend = StgcnBackend(timed)
    name = backend.get_inputs()[0].name

    for _ in range(warmup):
        backend.run(None, {name: window_tensor})
    timed.reset()
    outer = _TimedBackend(backend, timed)
    outer.reset()
    for _ in range(n):
        outer.run(None, {name: window_tensor})

    npu = stats(timed.samples_ns)
    total = stats(outer.samples_ns)
    cpu = None
    if npu["n"] and total["n"]:
        cpu = {k: round(total[k] - npu[k], 4)
               for k in ("p50", "p95", "p99", "mean")}
    return {"npu_only": npu, "with_softmax": total, "softmax_cost": cpu}


def bench_window_input(n: int, warmup: int, w: int = 640, h: int = 480) -> dict:
    """N2i：`TrackWindow.model_input` 的纯 CPU 成本（每人每窗一次）。"""
    from src.ring import TrackWindow

    rng = np.random.default_rng(0)
    kp = rng.uniform(0, 1, (spec.frames_for(30.0, spec.DURATION_S), 17, 2)).astype(np.float32) * [w, h]
    sc = rng.uniform(0.3, 1.0, (len(kp), 17)).astype(np.float32)
    tw = TrackWindow(track_id=0, start=0, end=len(kp), keypoints=kp,
                     scores=sc, present=np.ones(len(kp), bool), n_interp=0)

    for _ in range(warmup):
        tw.model_input(w, h)
    samples = []
    for _ in range(n):
        t0 = time.perf_counter_ns()
        tw.model_input(w, h)
        samples.append(time.perf_counter_ns() - t0)
    return stats(samples)


# ------------------------------------------------------------------ 端到端


def _probe_frames(video: Path) -> int:
    """只读元数据取帧数，不解码。用于筛掉太短的视频。"""
    import cv2

    cap = cv2.VideoCapture(str(video))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else 0
    cap.release()
    return n


def _frame_sample(video: Path, index: int = 60):
    """取一段真实视频里的某一帧，作为微基准的固定输入。"""
    import cv2

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"打不开视频: {video}")
    frame = None
    for i in range(index + 1):
        ok, f = cap.read()
        if not ok:
            break
        frame = f
    cap.release()
    if frame is None:
        raise RuntimeError(f"{video} 读不到第 {index} 帧")
    return frame


def _window_from_video(video: Path, yolo_backend, start: int = 0):
    """跑一小段真实视频，取一个真实的 ST-GCN 窗口张量。"""
    import cv2

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"打不开视频: {video}")
    frames = []
    for _ in range(start + spec.frames_for(30.0, spec.DURATION_S) + 15):
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()

    win = spec.frames_for(30.0, spec.DURATION_S)
    kp = np.zeros((win, 17, 2), np.float32)
    sc = np.zeros((win, 17), np.float32)
    present = np.zeros(win, bool)
    for i, f in enumerate(frames[:win]):
        dets = yolo_backend.detect(f)
        if dets:
            d = dets[0]
            kp[i], sc[i], present[i] = d.keypoints, d.scores, True

    if not present.any():
        raise RuntimeError("这一段里没检出人，取不到真实窗口张量；换一个视频或 --window-start")

    kp, _i, _n = spec.interpolate_gaps(present, kp, spec.MAX_INTERP_GAP_FRAMES)
    from src.ring import TrackWindow
    tw = TrackWindow(track_id=0, start=0, end=win, keypoints=kp, scores=sc,
                     present=present, n_interp=0)
    return tw.model_input(640, 480)


def run_e2e(yolo_rknn, stgcn_rknn, videos: list[Path], out_dir: Path,
            core_note: str) -> dict:
    """N3：逐帧 trace + 聚合 fps。含视频解码。"""
    import cv2

    per_video = []
    trace_all: list[dict] = []

    for vp in videos:
        pipe = Pipeline(
            PipelineConfig(fps=spec.BOARD_FPS),
            detector=YoloBackend(yolo_rknn, conf=spec.CONF,
                                 max_persons=spec.MAX_PERSONS),
            stgcn_session=StgcnBackend(stgcn_rknn),
        )
        cap = cv2.VideoCapture(str(vp))
        if not cap.isOpened():
            raise RuntimeError(f"打不开视频: {vp}")

        n = 0
        t_read = t_push = 0
        t_start = time.perf_counter_ns()
        while True:
            a = time.perf_counter_ns()
            ok, frame = cap.read()
            b = time.perf_counter_ns()
            if not ok:
                break
            closed = pipe.push(frame)
            c = time.perf_counter_ns()
            t_read += b - a
            t_push += c - b
            trace_all.append({
                "video": vp.name, "frame": n,
                "t_read_ms": round((b - a) / 1e6, 4),
                "t_push_ms": round((c - b) / 1e6, 4),
                "n_closed": len(closed),
            })
            n += 1
        t_end = time.perf_counter_ns()
        cap.release()
        res = pipe.finish(vp.stem)

        dt_s = (t_end - t_start) / 1e9
        per_video.append({
            "video": vp.name, "n_frames": n, "n_windows": len(res.windows),
            "n_scored": res.n_scored, "n_observed": res.n_observed,
            "seconds": round(dt_s, 3),
            "fps": round(n / dt_s, 3) if dt_s > 0 else 0.0,
            "read_ms_total": round(t_read / 1e6, 2),
            "push_ms_total": round(t_push / 1e6, 2),
            "max_prob": round(max((w.prob_fall for w in res.windows),
                                  default=0.0), 6),
        })
        print(f"  {vp.name:34s} {n:5d} 帧 {len(res.windows):4d} 窗 "
              f"{per_video[-1]['fps']:7.2f} fps", flush=True)

    # 聚合：Σn / Σdt，不用每视频 fps 的算术平均（短视频权重会被放大）
    total_n = sum(r["n_frames"] for r in per_video)
    total_dt = sum(r["seconds"] for r in per_video)
    total_push = sum(r["push_ms_total"] for r in per_video) / 1000.0
    f_e2e = total_n / total_dt if total_dt > 0 else 0.0
    f_compute = total_n / total_push if total_push > 0 else 0.0

    if trace_all:
        with open(out_dir / "e2e_trace.csv", "w", newline="",
                  encoding="utf-8-sig") as fh:
            import csv
            w = csv.DictWriter(fh, fieldnames=list(trace_all[0]))
            w.writeheader()
            w.writerows(trace_all)

    read_stats = stats([int(r["t_read_ms"] * 1e6) for r in trace_all])
    push_stats = stats([int(r["t_push_ms"] * 1e6) for r in trace_all])

    return {
        "core_note": core_note,
        "per_video": per_video,
        "total_frames": total_n,
        "total_windows": sum(r["n_windows"] for r in per_video),
        "total_seconds": round(total_dt, 3),
        "fps_e2e": round(f_e2e, 3),
        "fps_compute_no_decode": round(f_compute, 3),
        "t_read_per_frame": read_stats,
        "t_push_per_frame": push_stats,
        "budget_note": (
            f"30 fps 预算 = 33.3 ms/帧。实测 {f_e2e:.2f} fps ⇒ "
            f"每帧 {1000 / f_e2e:.1f} ms。" if f_e2e else "无数据"),
    }


# ------------------------------------------------------------------ 环境


def collect_env(core_yolo=None, core_stgcn=None) -> dict:
    """采集报告必填的环境量。板上读 `/sys`，PC 上自动跳过。"""
    def read(p):
        try:
            return Path(p).read_text().strip()
        except Exception:
            return None

    devfreq = {}
    try:
        for d in sorted(Path("/sys/class/devfreq").iterdir()):
            name = (d / "name").read_text().strip() if (d / "name").exists() else d.name
            devfreq[name] = {
                "cur_freq": read(d / "cur_freq"),
                "governor": read(d / "governor"),
                "available_frequencies": read(d / "available_frequencies"),
            }
    except Exception:
        pass

    thermal = {}
    try:
        for z in sorted(Path("/sys/class/thermal").glob("thermal_zone*")):
            t = read(z / "type")
            v = read(z / "temp")
            if t and v:
                thermal[t] = v
    except Exception:
        pass

    return {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": platform.node(),
        "machine": platform.machine(),
        "kernel": platform.release(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "cv2": _cv2_version(),
        "loadavg": read("/proc/loadavg"),
        "uptime_s": read("/proc/uptime"),
        "devfreq": devfreq,
        "thermal": thermal,
        "core_mask_yolo": core_yolo,
        "core_mask_stgcn": core_stgcn,
        "zero_copy": False,          # RKNNLite 无零拷贝通道，恒 false
        "two_model_concurrent": False,  # Pipeline.push 单线程串行
    }


def _cv2_version() -> str | None:
    try:
        import cv2
        return cv2.__version__
    except Exception:
        return None


# ------------------------------------------------------------------ main


def main() -> int:
    ap = argparse.ArgumentParser(
        description="板端时延测量（微基准 + 端到端）。不改 src/ 一行。")
    ap.add_argument("--out", default=str(ROOT / "results" / "latency" / "run1"))
    ap.add_argument("--yolo", default=str(ROOT / "models" / "yolo26n_pose_480x640_fp16.rknn"))
    ap.add_argument("--stgcn", default=str(ROOT / "models" / "stgcnpp_v2_fp16.rknn"))
    ap.add_argument("--yolo-core", type=int, default=None,
                    help="直接传 NPU_CORE_* 常量值；先跑 --print-cores 确认")
    ap.add_argument("--stgcn-core", type=int, default=None)
    ap.add_argument("--micro-only", action="store_true")
    ap.add_argument("--e2e-only", action="store_true")
    ap.add_argument("--videos", default=None,
                    help="逗号分隔的视频文件名（不含 .mp4）；默认全部 19 个")
    ap.add_argument("--n-micro", type=int, default=300)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--window-start", type=int, default=0,
                    help="构造 ST-GCN 固定张量时从第几帧开始")
    ap.add_argument("--no-gc-disable", action="store_true",
                    help="默认测量期 gc.disable()；加此参数保留 GC")
    ap.add_argument("--print-cores", action="store_true",
                    help="只打印 RKNNLite 常量表后退出")
    ap.add_argument("--dry-run", action="store_true",
                    help="PC 自检：用假后端，不需要 rknnlite")
    args = ap.parse_args()

    if args.print_cores:
        return 0 if print_core_constants() else 3

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- 会话 ----------------------------------------------------------
    if args.dry_run:
        yolo_rknn = FakeRKNN((1, 56, 6300), in_shape=INPUT_SHAPE)
        stgcn_rknn = FakeStgcnRKNN()
        print("[dry-run] 使用假后端，不碰 NPU")
    else:
        yolo_rknn = load_rknn(Path(args.yolo), args.yolo_core)
        stgcn_rknn = load_rknn(Path(args.stgcn), args.stgcn_core)

    env = collect_env(args.yolo_core, args.stgcn_core)
    result = {"env": env, "args": vars(args)}

    if not args.no_gc_disable:
        gc.disable()
        result["gc_disabled"] = True

    try:
        # ---- 微基准 ----------------------------------------------------
        if not args.e2e_only:
            print("\n== N1 YOLO 微基准 ==")
            videos = sorted((ROOT / "videos").rglob("*.mp4"))
            if not videos:
                print("videos/ 下没有 mp4", file=sys.stderr)
                return 2
            frame = _frame_sample(videos[0])
            r = bench_yolo(yolo_rknn, frame, args.n_micro, args.warmup)
            result["N1_yolo_npu_only"] = r["npu_only"]
            result["N1c_yolo_with_prepost"] = r["with_prepost"]
            result["N1_cpu_prepost"] = r["cpu_prepost"]
            result["N1_n_detections"] = r["n_detections"]
            print(f"  NPU-only    {_fmt(r['npu_only'])}")
            print(f"  含前后处理  {_fmt(r['with_prepost'])}")
            print(f"  CPU 侧成本  {r['cpu_prepost']}")

            print("\n== N2 ST-GCN 微基准 ==")
            yb = YoloBackend(yolo_rknn, conf=spec.CONF, max_persons=spec.MAX_PERSONS)
            win = _window_from_video(videos[0], yb, args.window_start)
            r2 = bench_stgcn(stgcn_rknn, win, args.n_micro, args.warmup)
            result["N2_stgcn_npu_only"] = r2["npu_only"]
            result["N2c_stgcn_with_softmax"] = r2["with_softmax"]
            result["N2_softmax_cost"] = r2["softmax_cost"]
            print(f"  NPU-only    {_fmt(r2['npu_only'])}")
            print(f"  含 softmax  {_fmt(r2['with_softmax'])}")

            print("\n== N2i 窗口张量构造（纯 CPU）==")
            r3 = bench_window_input(args.n_micro, args.warmup)
            result["N2i_window_input"] = r3
            print(f"  {_fmt(r3)}")

        # ---- 端到端 ----------------------------------------------------
        if not args.micro_only:
            print("\n== N3 端到端 ==")
            all_videos = sorted((ROOT / "videos").rglob("*.mp4"))
            if args.videos:
                want = {v.strip() for v in args.videos.split(",")}
                all_videos = [v for v in all_videos if v.stem in want]
            # 剔除帧数不足一个窗口的视频（`clip_0831_173416` 只有 3 帧）。
            # 它产生 0 窗口，单视频 fps 无意义，留着会污染聚合。
            # 用帧数而不是文件大小判断：大小是经验阈值，帧数是几何事实。
            all_videos = [v for v in all_videos
                          if _probe_frames(v) >= spec.frames_for(
                              spec.BOARD_FPS, spec.DURATION_S)]
            if not all_videos:
                print("没有可跑的视频", file=sys.stderr)
                return 2
            core_note = f"yolo={args.yolo_core} stgcn={args.stgcn_core}"
            e2e = run_e2e(yolo_rknn, stgcn_rknn, all_videos, out_dir, core_note)
            result["N3_e2e"] = e2e
            print(f"\n  合计 {e2e['total_frames']} 帧 / "
                  f"{e2e['total_windows']} 窗口 / {e2e['total_seconds']} s")
            print(f"  F_e2e     = {e2e['fps_e2e']} fps（含解码）")
            print(f"  F_compute = {e2e['fps_compute_no_decode']} fps（剔除解码）")
            print(f"  {e2e['budget_note']}")
            if e2e["total_windows"] != 915 and len(all_videos) == 19:
                print("   窗口数不是 915 —— 先查 tracker/覆盖率门/拼接，"
                      "别查模型（见 run_video.py 头部注释）")
    finally:
        if not args.no_gc_disable:
            gc.enable()

    (out_dir / "latency.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果写到 {out_dir / 'latency.json'}")

    if not args.dry_run:
        for obj in (yolo_rknn, stgcn_rknn):
            try:
                obj.release()
            except Exception:
                pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
