#!/usr/bin/env python3
"""p8/verify_gate.py — 任务 5 验收：低分事件必须送达"""
import argparse
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

BOARD_ROOT = Path("/opt/fallguard/deploy")
sys.path.insert(0, str(BOARD_ROOT))

import numpy as np  # noqa: E402

RECEIVED: list[dict] = []


class _Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(n) if n else b""
        try:
            RECEIVED.append(json.loads(raw.decode("utf-8", "replace")))
        except Exception:
            RECEIVED.append({"__raw__": raw.decode("utf-8", "replace")})
        self.send_response(200)
        self.send_header("Content-Length", "2")
        self.end_headers()
        self.wfile.write(b"ok")

    def log_message(self, *_a):
        pass          # 静音：一次验收不要被 access log 刷屏


class _FakeHybrid:
    """`push_fall_event` 只读 `_last_ml_result.confidence`。"""
    _last_ml_result = {"confidence": 0.87}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=9099)
    ap.add_argument("--timeout", type=float, default=30.0)
    ap.add_argument("--score", type=float, default=0.16,
                    help="故意造的 hybrid_score（必须 < push_threshold）")
    args = ap.parse_args()

    from shm_adapter import SensorShmWriter  # noqa: E402

    srv = HTTPServer(("0.0.0.0", args.port), _Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    print(f"[recv] 监听 0.0.0.0:{args.port}")

    adapter = SensorShmWriter()
    cfg = adapter._cfg
    thr = float(cfg.get("push_threshold", -1))
    print(f"[cfg] push_threshold={thr}  push_enabled={cfg.get('push_enabled')}  "
          f"push_url={cfg.get('push_url')}")
    print(f"[cfg] 本次造的事件 hybrid_score={args.score}"
          f"  {'< 阈值 ⇒ 改动前必被拒' if args.score < thr else '>= 阈值（失去检验意义）'}")

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[:, :320] = (80, 80, 160)
    frame[:, 320:] = (30, 160, 60)

    event = {"reason": "dual_confirm", "features": {"hybrid_score": args.score}}
    person = {"person_id": 1, "bbox": [100.0, 100.0, 300.0, 400.0]}

    print("\n=== 推送一条低分事件 ===")
    seq = adapter.push_fall_event(event, person, _FakeHybrid(), frame)
    print(f"[push] seq={seq}")

    print(f"\n=== 等网关推送（最多 {args.timeout:.0f}s）===")
    t0 = time.time()
    while time.time() - t0 < args.timeout and not RECEIVED:
        time.sleep(0.3)

    ok = bool(RECEIVED)
    if ok:
        for i, body in enumerate(RECEIVED, 1):
            ev = (body or {}).get("event", {})
            print(f"[recv] #{i} person_id={ev.get('person_id')} "
                  f"hybrid_score={ev.get('hybrid_score')} "
                  f"reason={ev.get('reason')} state={ev.get('state', '<无此字段>')}")
        got = (RECEIVED[0].get("event") or {}).get("hybrid_score")
        ok = got is not None and abs(float(got) - args.score) < 1e-6
        print(f"\n[判据] 送达分数 {got} == 造的 {args.score} ? "
              f"{'YES' if ok else 'NO'}")
    else:
        print("[recv] 超时：一条都没收到")

    print(f"\n{'PASS' if ok else 'FAIL'}  "
          f"（hybrid_score={args.score} < threshold={thr}，"
          f"{'送达' if ok else '未送达'}）")
    srv.shutdown()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
