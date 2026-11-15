"""板端推流入口：mainpath → H.264 → RTSP。"""
from __future__ import annotations

import argparse
import os
import signal
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.stream import RESOLUTIONS, RtspServer, StreamConfig  # noqa: E402


def build_config(args) -> StreamConfig:
    """只把显式给了的参数当 overrides，其余走 `FG_STREAM_*` / 默认值。"""
    over = {}
    if args.res:
        w, h = RESOLUTIONS[args.res]
        over["width"], over["height"] = w, h
    if args.width is not None:
        over["width"] = args.width
    if args.height is not None:
        over["height"] = args.height
    if args.gop is None and args.fps is not None:
        over["gop"] = args.fps                 # 1 秒一个 I 帧
    for name in ("device", "fps", "io_mode", "port", "mount", "bps", "gop",
                 "user", "password", "source"):
        val = getattr(args, name, None)
        if val is not None:
            over[name] = val
    if args.no_auth:
        over["auth"] = False
    return StreamConfig.from_env(**over)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", choices=sorted(RESOLUTIONS), default=None,
                    help="分辨率预设：hd=1920x1080 / 720=1280x720 / sd=640x480")
    ap.add_argument("--width", type=int, default=None)
    ap.add_argument("--height", type=int, default=None)
    ap.add_argument("--fps", type=int, default=None)
    ap.add_argument("--io-mode", type=int, default=None)
    ap.add_argument("--device", default=None, help="默认 mainpath 的 by-path")
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--mount", default=None)
    ap.add_argument("--bps", type=int, default=None, help="CBR 目标码率 bit/s")
    ap.add_argument("--gop", type=int, default=None)
    ap.add_argument("--source", choices=["v4l2", "test"], default=None)
    ap.add_argument("--user", default=None)
    ap.add_argument("--password", default=None)
    ap.add_argument("--no-auth", action="store_true",
                    help=" 关闭 RTSP 鉴权（匿名可拉流）")
    ap.add_argument("--print-launch", action="store_true", help="只打印管线串后退出")
    args = ap.parse_args()

    cfg = build_config(args)
    srv = RtspServer(cfg)

    if args.print_launch:
        print(srv.launch)
        return 0

    print(f"[stream] {cfg.describe()}", flush=True)
    print(f"[stream] launch = {srv.launch}", flush=True)

    if not cfg.auth:
        print("=" * 68, flush=True)
        print("    RTSP 鉴权已关闭 —— 同一网段内**任何人**都能拉取病房画面。",
              flush=True)
        print("    只在隔离网段/单机自测时使用。", flush=True)
        print("=" * 68, flush=True)

    if cfg.source == "v4l2":
        if not os.path.exists(cfg.device):
            print(f"[stream] FATAL: 设备不存在 {cfg.device}", flush=True)
            print("[stream] 提示：`ls -l /dev/v4l/by-path/` 看实际名字；"
                  "推流用 index0(mainpath)，检测用 index1(selfpath)", flush=True)
            return 2

    srv.start()

    def on_client_connected(_server, client):
        try:
            print(f"[stream] client +1  {client.get_connection().get_ip() if client.get_connection() else '?'}",
                  flush=True)
            client.connect("closed", lambda c: print("[stream] client -1", flush=True))
        except Exception as exc:                  # 信号回调异常会被吞掉
            print(f"[stream] client 记录失败: {exc}", flush=True)

    def on_media_configure(_factory, media):
        try:
            print("[stream] media 已构建（首个客户端）", flush=True)
        except Exception:
            pass

    srv.server.connect("client-connected", on_client_connected)
    srv.factory.connect("media-configure", on_media_configure)

    def _quit(*_):
        print("\n[stream] 收到信号，停止", flush=True)
        srv.stop()

    signal.signal(signal.SIGINT, _quit)
    signal.signal(signal.SIGTERM, _quit)

    # start 已在后台线程跑 GLib 主循环；主线程只等信号
    try:
        signal.pause()
    except AttributeError:                        # Windows 没有 pause
        import time
        while True:
            time.sleep(1.0)
    print("[stream] stopped", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
