""" 验收：GStreamer appsink 采集层（分支 γ：selfpath 检测 / mainpath 推流）。"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

RAW = Path("/tmp/fg_s4_raw.nv12")
SHM = "/tmp/fg_s4_shm"
BY_PATH_SELF = "/dev/v4l/by-path/platform-rkisp-vir1-video-index1"
BY_PATH_MAIN = "/dev/v4l/by-path/platform-rkisp-vir1-video-index0"

RESULTS: list[tuple[str, str, str]] = []       # (case, verdict, detail)


def record(case: str, ok: bool, detail: str) -> None:
    RESULTS.append((case, "PASS" if ok else "FAIL", detail))
    print(f"  → {' PASS' if ok else ' FAIL'}  {detail}", flush=True)


# ------------------------------------------------------------------ 前置检查


def preflight() -> None:
    if os.geteuid() == 0:
        raise SystemExit(
            "不要用 sudo 跑：sudo 把 HOME 换成 /root，会丢掉 ~/.local 里的 numpy；"
            "而且会在 /tmp 留下 root 所有的测试文件。")
    if not Path(BY_PATH_SELF).exists():
        raise SystemExit(
            f"找不到 {BY_PATH_SELF}。"
            f"判据是「by-path 存在 **且** 能出帧」——"
            f"别用 /dev/video-camera0（它指向不能出帧的 video51）")
    Path(SHM).unlink(missing_ok=True)


# ------------------------------------------------------------------ 模型/管线


class Models:
    """懒加载两个 RKNN 会话（L1 的管线对拍才用得上）。"""

    def __init__(self, args):
        self.args = args
        self._yolo = None
        self._stgcn = None

    def _load(self):
        if self._yolo is not None:
            return
        from run_video import load_rknn

        t0 = time.monotonic()
        self._yolo = load_rknn(Path(self.args.yolo), self.args.yolo_core)
        self._stgcn = load_rknn(Path(self.args.stgcn), self.args.stgcn_core)
        print(f"  [rknn] 初始化 {time.monotonic() - t0:.1f} s", flush=True)

    def make_pipe(self, fps: int):
        self._load()
        from src import PipelineConfig
        from src.backends import StgcnBackend, YoloBackend
        from src.multi import MultiConfig, MultiPipeline

        cfg = PipelineConfig(fps=fps)
        return MultiPipeline(
            cfg,
            detector=YoloBackend(self._yolo, conf=cfg.conf, max_persons=cfg.max_persons),
            stgcn_session=StgcnBackend(self._stgcn),
            # confirm_windows 必须 =1：=1 才等价旧口径，逐窗对拍才有意义
            mcfg=MultiConfig(confirm_windows=1, max_persons_stgcn=3))


# ------------------------------------------------------------------ 工具


def record_raw(dev: str, n_frames: int, w: int, h: int, fps: int) -> None:
    """从相机录一段裸 NV12，作为 L1 的唯一源。"""
    print(f"  录制 {n_frames} 帧裸 NV12 → {RAW}", flush=True)
    cmd = [
        "gst-launch-1.0", "-q",
        "v4l2src", f"device={dev}", "io-mode=2", f"num-buffers={n_frames}",
        "!", f"video/x-raw,format=NV12,width={w},height={h},framerate={fps}/1",
        "!", "filesink", f"location={RAW}",
    ]
    t0 = time.monotonic()
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0 or not RAW.exists():
        raise RuntimeError(f"录制失败 rc={r.returncode}\n{r.stderr[-800:]}")
    size = RAW.stat().st_size
    print(f"  {size} B / {time.monotonic() - t0:.1f} s", flush=True)


def drain(src, n: int, timeout_s: float = 5.0) -> list:
    """从采集源取 n 帧（或直到超时）。"""
    out = []
    t_end = time.monotonic() + timeout_s
    while len(out) < n and time.monotonic() < t_end:
        fr = src.read(timeout_s=1.0)
        if fr is None:
            continue
        out.append(fr.bgr)
    return out


# ------------------------------------------------------------------ L1


def case_l1(args, models: Models) -> None:
    w, h, fps = args.width, args.height, args.fps
    fsz = w * h * 3 // 2
    if not RAW.exists():
        record_raw(args.device, args.raw_frames, w, h, fps)
    raw = RAW.read_bytes()
    n = len(raw) // fsz
    if n <= 0:
        record("L1", False, f"裸文件太小：{len(raw)} B")
        return
    print(f"  L1 源：{n} 帧 × {fsz} B", flush=True)

    # ---- 路径 A：GStreamer（filesrc ! rawvideoparse ! tee ! appsink） ----
    from src.capture import CaptureConfig, open_source

    src = open_source(CaptureConfig(backend="gst", raw_file=str(RAW),
                                    width=w, height=h, fps=fps))
    src.start()
    try:
        fa = drain(src, n, timeout_s=10.0)
    finally:
        src.stop()

    # ---- 路径 B：numpy 直读同一份字节 + 同一个转换函数 ----
    from src.capture.gst_source import nv12_to_bgr

    fb = []
    for i in range(len(fa)):
        nv12 = np.frombuffer(raw[i * fsz:(i + 1) * fsz], np.uint8).reshape(h * 3 // 2, w)
        fb.append(nv12_to_bgr(nv12, h, w))

    if len(fa) != n:
        record("L1", False, f"GStreamer 路径只拿到 {len(fa)}/{n} 帧")
        return

    dmax = 0
    for a, b in zip(fa, fb):
        dmax = max(dmax, int(np.abs(a.astype(np.int16) - b.astype(np.int16)).max()))
    print(f"  L1 像素：{len(fa)} 帧逐帧比对，max|Δ| = {dmax}", flush=True)
    if dmax != 0:
        record("L1", False, f"像素不一致 max|Δ|={dmax}（采集层篡改了数据）")
        return

    # ---- 两条路径各自过一遍管线，比窗口 ----
    pa = models.make_pipe(fps)
    for f in fa:
        pa.push(f)
    ra = pa.finish("l1_gst")

    pb = models.make_pipe(fps)
    for f in fb:
        pb.push(f)
    rb = pb.finish("l1_np")

    wa = [(x.window_id, x.start, x.end, x.prob_fall, x.track_id, x.coverage)
          for x in ra.windows]
    wb = [(x.window_id, x.start, x.end, x.prob_fall, x.track_id, x.coverage)
          for x in rb.windows]
    if len(wa) != len(wb):
        record("L1", False, f"窗口数不同 {len(wa)} vs {len(wb)}")
        return
    maxdp = max((abs(a[3] - b[3]) for a, b in zip(wa, wb)), default=0.0)
    track_diff = sum(1 for a, b in zip(wa, wb) if a[4] != b[4])
    cov_diff = max((abs(a[5] - b[5]) for a, b in zip(wa, wb)), default=0.0)
    ok = (maxdp == 0.0) and (track_diff == 0) and (cov_diff == 0.0)
    record("L1", ok,
           f"像素 max|Δ|=0；窗口 {len(wa)} 个，max|Δp|={maxdp:.10f}，"
           f"track_diff={track_diff}，max|Δcov|={cov_diff:.10f}")


# ------------------------------------------------------------------ L2


def case_l2(args, _models) -> None:
    from src.capture import CaptureConfig, open_source

    src = open_source(CaptureConfig(backend="gst", device=args.device,
                                    width=args.width, height=args.height,
                                    fps=args.fps,
                                    queue_buffers=args.queue_buffers))
    seqs = []
    fresh_ok = False
    fresh_detail = "未做"
    t0 = time.monotonic()
    src.start()
    try:
        while time.monotonic() - t0 < args.slow_seconds:
            fr = src.read(timeout_s=1.0)
            if fr is None:
                continue
            seqs.append(fr.seq)
            time.sleep(args.slow)          # 人为让消费者变慢

        # ---- 证明丢的是「最旧」而不是「最新」----
        # 停读 1s（约 30 帧到达，队列只留最后 4 帧），再取一帧：
        # 拿到 seq 应该贴近 pulled-1；若拿到 last_read+1 说明丢的是新帧。
        time.sleep(1.0)
        fr = src.read(timeout_s=2.0)
        newest = src.stats["pulled"] - 1
        if fr is None:
            fresh_detail = "停读 1s 后读不到帧"
        else:
            fresh_ok = fr.seq >= newest - (args.queue_buffers + 2)
            fresh_detail = f"停读 1s 后取到 seq={fr.seq}，最新={newest}"
    finally:
        src.stop()
    st = src.stats
    skips = sum(1 for a, b in zip(seqs, seqs[1:]) if b != a + 1)
    lag = st["tee_count"] - (st["read"] + st["dropped_total"])
    print(f"  L2 采 {st['read']} 帧 / {st['seconds']}s = {st['fps']} fps；"
          f"tee={st['tee_count']} gst_drop={st['gst_dropped']} "
          f"q_drop={st['queue_dropped']} gaps={st['gaps']}", flush=True)
    ok = (st["dropped_total"] > 0 and skips > 0 and st["gaps"] > 0
          and 0 <= lag < 16 and fresh_ok)
    record("L2", ok,
           f"dropped_total={st['dropped_total']}，seq 跳号 {skips} 次，"
           f"PTS 间隔超标 {st['gaps']} 次，tee-(read+drop)={lag}；{fresh_detail}")


# ------------------------------------------------------------------ L3


def case_l3(args, _models) -> None:
    from src.capture import CaptureConfig, open_source

    Path(SHM).unlink(missing_ok=True)
    src = open_source(CaptureConfig(backend="gst", device=args.device,
                                    width=args.width, height=args.height,
                                    fps=args.fps, shm_socket=SHM))
    counts = []
    src.start()
    try:
        # 阶段 1：没有任何消费者
        t0 = time.monotonic()
        n1 = 0
        while time.monotonic() - t0 < args.iso_seconds:
            if src.read(timeout_s=1.0) is not None:
                n1 += 1
        counts.append(("无消费者", n1, time.monotonic() - t0))

        # 阶段 2：起一个真消费者，跑 1s 后 kill -9
        cons = subprocess.Popen(
            ["gst-launch-1.0", "-q", "shmsrc", f"socket-path={SHM}", "is-live=true",
             "!", f"video/x-raw,format=NV12,width={args.width},"
             f"height={args.height},framerate={args.fps}/1",
             "!", "fakesink", "sync=false"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(1.0)
        cons.kill() # 用 pid，不用 pkill -f
        cons.wait(timeout=5)

        # 阶段 3：消费者已死，继续看帧数
        t0 = time.monotonic()
        n2 = 0
        while time.monotonic() - t0 < args.iso_seconds:
            if src.read(timeout_s=1.0) is not None:
                n2 += 1
        counts.append(("消费者被 kill -9 后", n2, time.monotonic() - t0))
    finally:
        src.stop()

    fps1 = counts[0][1] / counts[0][2]
    fps2 = counts[1][1] / counts[1][2]
    print(f"  L3 无消费者 {fps1:.2f} fps / 杀掉消费者后 {fps2:.2f} fps", flush=True)
    ok = fps1 >= args.min_fps and fps2 >= args.min_fps
    record("L3", ok,
           f"无消费者 {fps1:.2f} fps，kill -9 消费者后 {fps2:.2f} fps"
           f"（门 {args.min_fps}）—— 判据不看 shmsink 的 errno:16")


# ------------------------------------------------------------------ L4


def case_l4(args, _models) -> dict:
    from src.capture import CaptureConfig, open_source

    src = open_source(CaptureConfig(backend="opencv", device=args.device,
                                    width=args.width, height=args.height,
                                    fps=args.fps))
    src.start()
    t0 = time.monotonic()
    n = 0
    try:
        while time.monotonic() - t0 < args.iso_seconds:
            if src.read(timeout_s=1.0) is not None:
                n += 1
    finally:
        src.stop()
    st = src.stats
    ok = n > 0 and st["fps"] >= args.min_fps
    record("L4", ok, f"opencv 后端 {n} 帧 / {st['seconds']}s = {st['fps']} fps")
    return st


# ------------------------------------------------------------------ L6


def case_l6(args, _models) -> None:
    """γ 的核心主张：推流端挂了，检测端不受影响。"""
    from src.capture import CaptureConfig, open_source

    streamer = subprocess.Popen(
        ["gst-launch-1.0", "-q", "v4l2src", f"device={BY_PATH_MAIN}", "io-mode=2",
         "!", f"video/x-raw,format=NV12,width=1920,height=1080,framerate={args.fps}/1",
         "!", "fakesink", "sync=false"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(1.0)                       # 让 ISP 先按 1080p mainpath 配好

    src = open_source(CaptureConfig(backend="gst", device=args.device,
                                    width=args.width, height=args.height,
                                    fps=args.fps))
    rates = []
    try:
        src.start()
        for phase in ("推流端在跑", "推流端被 kill -9 后"):
            t0 = time.monotonic()
            n = 0
            while time.monotonic() - t0 < args.iso_seconds:
                if src.read(timeout_s=1.0) is not None:
                    n += 1
            rates.append((phase, n / (time.monotonic() - t0)))
            if phase == "推流端在跑":
                streamer.kill()
                streamer.wait(timeout=5)
    finally:
        src.stop()
        if streamer.poll() is None:
            streamer.kill()
            streamer.wait(timeout=5)

    print(f"  L6 推流端在跑 {rates[0][1]:.2f} fps / 杀掉后 {rates[1][1]:.2f} fps",
          flush=True)
    ok = all(r >= args.min_fps for _p, r in rates)
    record("L6", ok,
           f"推流端在跑 {rates[0][1]:.2f} fps，kill -9 后 {rates[1][1]:.2f} fps"
           f"（门 {args.min_fps}）—— γ 的隔离性")


# ------------------------------------------------------------------ L5


def case_l5(args, models) -> None:
    """只报不判：两个后端的 fps 量级。"""
    from src.capture import CaptureConfig, open_source

    out = {}
    for backend in ("gst", "opencv"):
        src = open_source(CaptureConfig(backend=backend, device=args.device,
                                        width=args.width, height=args.height,
                                        fps=args.fps))
        src.start()
        t0 = time.monotonic()
        n = 0
        try:
            while time.monotonic() - t0 < args.iso_seconds:
                if src.read(timeout_s=1.0) is not None:
                    n += 1
        finally:
            src.stop()
        out[backend] = src.stats
    print(f"  L5 gst={out['gst']['fps']} fps  opencv={out['opencv']['fps']} fps"
          f"（**只报不判**：解码/转换路线不同）", flush=True)
    RESULTS.append(("L5", "REPORT",
                    f"gst={out['gst']['fps']} fps / opencv={out['opencv']['fps']} fps"))


# ------------------------------------------------------------------ main


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--case", choices=["l1", "l2", "l3", "l4", "l5", "l6"],
                    default=None)
    ap.add_argument("--device", default=BY_PATH_SELF)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--raw-frames", type=int, default=100)
    ap.add_argument("--queue-buffers", type=int, default=4,
                    help="L2 用的 leaky queue 深度（新鲜度判据要用它）")
    ap.add_argument("--slow", type=float, default=0.05, help="L2 每帧额外 sleep 秒")
    ap.add_argument("--slow-seconds", type=float, default=6.0)
    ap.add_argument("--iso-seconds", type=float, default=5.0)
    ap.add_argument("--min-fps", type=float, default=25.0)
    ap.add_argument("--yolo", default=str(ROOT / "models" / "yolo26n_pose_480x640_fp16.rknn"))
    ap.add_argument("--stgcn", default=str(ROOT / "models" / "stgcnpp_v2_fp16.rknn"))
    ap.add_argument("--yolo-core", type=int, default=None)
    ap.add_argument("--stgcn-core", type=int, default=None)
    args = ap.parse_args()

    preflight()
    models = Models(args)
    cases = (["l1", "l2", "l3", "l4", "l6", "l5"]
             if (args.all or not args.case) else [args.case])

    print(f"验收  设备={args.device}  {args.width}x{args.height}@{args.fps}",
          flush=True)
    for c in cases:
        print(f"\n[{c.upper()}]", flush=True)
        t0 = time.monotonic()
        try:
            {"l1": lambda: case_l1(args, models),
             "l2": lambda: case_l2(args, models),
             "l3": lambda: case_l3(args, models),
             "l4": lambda: case_l4(args, models),
             "l5": lambda: case_l5(args, models),
             "l6": lambda: case_l6(args, models)}[c]()
        except Exception as exc:                       # noqa: BLE001
            record(c.upper(), False, f"异常：{type(exc).__name__}: {exc}")
        print(f"  （{time.monotonic() - t0:.1f}s）", flush=True)

    print("\n================ 汇总 ================")
    for name, verdict, detail in RESULTS:
        print(f"{name:4s} {verdict:6s} {detail}")
    hard = [r for r in RESULTS if r[1] != "REPORT"]
    bad = [r for r in hard if r[1] != "PASS"]
    print(f"\n硬门 {len(hard) - len(bad)}/{len(hard)} PASS"
          f"（L5 只报不判）")
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
