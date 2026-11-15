"""板端实时入口：相机 → 采集层 → per-person 管线 → 打印事件。"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src import PipelineConfig                      # noqa: E402
from src.backends import StgcnBackend, YoloBackend  # noqa: E402
from src.capture import CaptureConfig, open_source  # noqa: E402
from src.multi import MultiConfig, MultiPipeline    # noqa: E402
from run_video import load_rknn                     # noqa: E402


def build_config(args) -> CaptureConfig:
    """只把显式给了的参数当 overrides，其余走 `FG_CAPTURE_*` / 默认值。"""
    over = {}
    for name in ("backend", "device", "width", "height", "fps", "io_mode",
                 "appsink_buffers", "queue_buffers", "shm_socket", "raw_file"):
        val = getattr(args, name, None)
        if val is not None:
            over[name] = val
    return CaptureConfig.from_env(**over)


def fmt_event(evt, t0: float) -> str:
    return (f"  [{time.monotonic() - t0:7.2f}s] "
            f"{evt.state.label():9s} person={evt.person_id:<3d} "
            f"window={evt.window_start}..{evt.window_end} "
            f"p={evt.prob_fall:.3f} cov={evt.coverage:.2f} "
            f"hits={evt.consec_hits} miss={evt.consec_miss}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["gst", "opencv"], default=None)
    ap.add_argument("--device", default=None,
                    help="默认 selfpath 的 by-path（检测用）")
    ap.add_argument("--width", type=int, default=None)
    ap.add_argument("--height", type=int, default=None)
    ap.add_argument("--fps", type=int, default=None)
    ap.add_argument("--io-mode", type=int, default=None)
    ap.add_argument("--appsink-buffers", type=int, default=None)
    ap.add_argument("--queue-buffers", type=int, default=None)
    ap.add_argument("--shm-socket", default=None,
                    help="给了才开 shmsink 分支（推流用）")
    ap.add_argument("--raw-file", default=None, help="裸 NV12 文件（对拍用）")
    ap.add_argument("--seconds", type=float, default=30.0, help="0 = 一直跑")
    ap.add_argument("--yolo", default=str(ROOT / "models" / "yolo26n_pose_480x640_fp16.rknn"))
    ap.add_argument("--stgcn", default=str(ROOT / "models" / "stgcnpp_v2_fp16.rknn"))
    ap.add_argument("--yolo-core", type=int, default=None)
    ap.add_argument("--stgcn-core", type=int, default=None)
    ap.add_argument("--confirm-windows", type=int, default=None)
    ap.add_argument("--quiet", action="store_true", help="只打事件与汇总")
    args = ap.parse_args()

    ccfg = build_config(args)
    print(f"[capture] {ccfg.describe()}", flush=True)

    over = {}
    if args.confirm_windows is not None:
        over["confirm_windows"] = args.confirm_windows
    mcfg = MultiConfig.from_env(**over)
    print(f"[multi]   {mcfg.describe()}", flush=True)

    src = open_source(ccfg)
    t_init = time.monotonic()
    yolo = load_rknn(Path(args.yolo), args.yolo_core)
    stgcn = load_rknn(Path(args.stgcn), args.stgcn_core)
    print(f"[rknn]    初始化 {time.monotonic() - t_init:.1f} s", flush=True)

    pcfg = PipelineConfig(fps=ccfg.fps)
    pipe = MultiPipeline(
        pcfg,
        detector=YoloBackend(yolo, conf=pcfg.conf, max_persons=pcfg.max_persons),
        stgcn_session=StgcnBackend(stgcn),
        mcfg=mcfg)

    src.start()
    t0 = time.monotonic()
    n_evt = 0
    last_report = t0
    try:
        while True:
            if args.seconds and time.monotonic() - t0 >= args.seconds:
                break
            fr = src.read(timeout_s=1.0)
            if fr is None:
                continue
            for evt in pipe.push(fr.bgr):
                n_evt += 1
                print(fmt_event(evt, t0), flush=True)
            now = time.monotonic()
            if not args.quiet and now - last_report >= 5.0:
                last_report = now
                st = src.stats
                print(f"  ... {now - t0:6.1f}s  帧 {st['read']}  "
                      f"{st['fps']:.1f} fps  丢 {st['dropped_total']}  "
                      f"事件 {n_evt}", flush=True)
    except KeyboardInterrupt:
        print("\n[stop] Ctrl-C", flush=True)
    finally:
        src.stop()

    res = pipe.finish("live")
    st = src.stats
    print()
    print("================ 汇总 ================")
    print(f"采集      {st['backend']}  帧 {st['read']}  {st['fps']:.2f} fps  "
          f"({st['seconds']:.1f}s)")
    if st.get("dropped_total") is not None:
        print(f"丢帧      gst {st['gst_dropped']} + queue {st['queue_dropped']} "
              f"= {st['dropped_total']}   "
              f"跳号间隔 {st['gaps']} 次（最大 {st['max_gap_ms']:.1f} ms）")
    # n_scored / n_observed 在 base（VideoResult）上，不在 MultiVideoResult 上
    print(f"窗口      {len(res.windows)}   评分 {res.base.n_scored}   "
          f"观测 {res.base.n_observed}   人数 {res.n_persons}")
    print(f"事件      {len(res.events)} 条（confirmed "
          f"{len(res.confirmed())} / retracted {len(res.retracted())}）")
    print(f"计数器    {res.counters}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
