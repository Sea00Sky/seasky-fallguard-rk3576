"""板端云出口入口：本地 RTSP → SRT → 云端 mediamtx。"""
from __future__ import annotations

import argparse
import os
import signal
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.stream import StreamConfig  # noqa: E402
from src.stream.cloud import (  # noqa: E402
    CloudConfig, CloudSender, local_rtsp_from_stream, redact_launch)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--uri", default=None,
                    help="云端 SRT 基址，如 srt://1.2.3.4:8890")
    ap.add_argument("--streamid", default=None,
                    help="mediamtx 发布标识（默认 publish:live）")
    ap.add_argument("--passphrase", default=None, help="SRT AES 口令")
    ap.add_argument("--latency", type=int, default=None,
                    help="SRT 缓冲毫秒（默认 200）")
    ap.add_argument("--local-rtsp", default=None,
                    help="本地拉流地址；默认由 FG_STREAM_* 拼（含凭据）")
    ap.add_argument("--print-launch", action="store_true",
                    help="只打印管线串后退出")
    ap.add_argument("--check", action="store_true",
                    help="只做静态解析（挡 launch 串写错）后退出")
    args = ap.parse_args()

    over: dict = {}
    if args.uri is not None:
        over["uri"] = args.uri
    if args.streamid is not None:
        over["streamid"] = args.streamid
    if args.passphrase is not None:
        over["passphrase"] = args.passphrase
    if args.latency is not None:
        over["latency_ms"] = args.latency
    if args.local_rtsp is not None:
        over["local_rtsp"] = args.local_rtsp
    elif not os.environ.get("FG_CLOUD_LOCAL_RTSP"):
        # 没显式给 ⇒ 跟着本地推流的端口/挂载点/凭据走，避免两处配置漂移
        over["local_rtsp"] = local_rtsp_from_stream(StreamConfig.from_env())

    cfg = CloudConfig.from_env(**over)

    if not cfg.enabled:
        # V1.2：预留但未启用 ⇒ 静默 exit 0（systemd Restart=on-failure 不会拉起）
        return 0

    sender = CloudSender(cfg)

    if args.print_launch:
        # 脱敏后打印：这条命令常被贴进文档/聊天，口令不该跟着走
        print(redact_launch(sender.launch))
        return 0

    if args.check:
        try:
            sender.preflight()
        except RuntimeError as exc:
            print(f"[cloud]  {exc}", flush=True)
            return 2
        print(f"[cloud]  launch 串解析通过\n"
              f"[cloud] {redact_launch(sender.launch)}", flush=True)
        return 0

    def _quit(*_):
        print("\n[cloud] 收到信号，停止", flush=True)
        sender.stop()

    signal.signal(signal.SIGINT, _quit)
    signal.signal(signal.SIGTERM, _quit)

    return sender.run_forever()


if __name__ == "__main__":
    raise SystemExit(main())
