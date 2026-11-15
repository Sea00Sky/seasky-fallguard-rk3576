#!/usr/bin/env python3
"""ann_bridge.py —— 演示旁路：把检测进程的标注画面（MJPEG）转成 RTSP。"""
from __future__ import annotations

import argparse
import signal
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.stream import RtspServer, StreamConfig  # noqa: E402

DEFAULT_SOURCE = "http://127.0.0.1:8888/stream"


def build_launch(src_url: str, width: int, height: int, fps: int, bps: int) -> str:
    """拼 `RTSPMediaFactory.set_launch` 用的管线串（见模块 docstring）。"""
    caps = f"video/x-raw,format=NV12,width={width},height={height},framerate={fps}/1"
    return (
        f"( souphttpsrc location={src_url} is-live=true do-timestamp=true "
        f"! multipartdemux "
        f"! jpegdec "
        f"! videoconvert "
        f"! videorate "
        f"! {caps} "
        f"! mpph264enc rc-mode=cbr bps={bps} gop={fps} "
        f"! h264parse config-interval=-1 "
        f"! rtph264pay name=pay0 pt=96 )"
    )


def probe(src_url: str, timeout_s: float = 15.0) -> int:
    """只验「能不能从 MJPEG 取到帧」，不起 RTSP。"""
    import gi

    gi.require_version("Gst", "1.0")
    from gi.repository import Gst

    Gst.init(None)
    pipeline = Gst.parse_launch(
        f"souphttpsrc location={src_url} is-live=true do-timestamp=true "
        f"! multipartdemux ! jpegdec "
        f"! appsink name=sink max-buffers=2 drop=true emit-signals=false"
    )
    appsink = pipeline.get_by_name("sink")
    deadline = time.monotonic() + timeout_s
    sizes: list[tuple[int, int]] = []
    pipeline.set_state(Gst.State.PLAYING)
    try:
        while time.monotonic() < deadline and len(sizes) < 5:
            msg = pipeline.get_bus().pop_filtered(Gst.MessageType.ERROR)
            if msg is not None:
                err, dbg = msg.parse_error()
                print(f"[probe] 源错误：{err.message}（{dbg}）", flush=True)
                return 1
            sample = appsink.emit("try-pull-sample", int(0.5 * Gst.SECOND))
            if sample is None:
                continue
            caps = sample.get_caps().get_structure(0)
            sizes.append((caps.get_value("width"), caps.get_value("height")))
    finally:
        pipeline.set_state(Gst.State.NULL)

    if not sizes:
        print(f"[probe] {timeout_s:.0f} 秒内没取到任何帧 —— "
              f"检查 8888 是否在跑、通道名是否为 ann", flush=True)
        return 1
    w, h = sizes[-1]
    print(f"[probe] OK 取到 {len(sizes)} 帧，尺寸 {w}×{h}，源 = {src_url}", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="MJPEG 标注画面 → RTSP（演示旁路）")
    ap.add_argument("--source-url", default=DEFAULT_SOURCE,
                    help=f"MJPEG 地址（默认 {DEFAULT_SOURCE}；/raw 是裸帧）")
    ap.add_argument("--port", type=int, default=None,
                    help="RTSP 端口；不给则用 FG_STREAM_PORT（默认 8554）")
    ap.add_argument("--mount", default=None,
                    help="挂载点；不给则用 FG_STREAM_MOUNT（默认 /live）")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=15,
                    help="规整后的帧率，应与 sensor 的 --web-fps 一致")
    ap.add_argument("--bps", type=int, default=1_500_000,
                    help="CBR 目标码率；640×480 给 1.5 Mbps 已经很富裕")
    ap.add_argument("--no-auth", action="store_true",
                    help=" 关鉴权，局域网内任何人都能拉流")
    ap.add_argument("--probe", action="store_true", help="只验能否取帧，不起 RTSP")
    args = ap.parse_args()

    if args.probe:
        return probe(args.source_url)

    cfg = StreamConfig.from_env()
    if args.port is not None:
        cfg.port = args.port
    if args.mount is not None:
        cfg.mount = args.mount
    if args.no_auth:
        cfg.auth = False
        print("[bridge]  已关闭鉴权：同网段任何人都能拉流", flush=True)

    launch = build_launch(args.source_url, args.width, args.height, args.fps, args.bps)
    print(f"[bridge] 源     = {args.source_url}", flush=True)
    print(f"[bridge] 输出   = rtsp://{cfg.user}:***@{cfg.port}{cfg.mount}", flush=True)
    print(f"[bridge] launch = {launch}", flush=True)

    srv = RtspServer(cfg, launch=launch)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    try:
        srv.run_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.stop()
        print("[bridge] 已停止", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
