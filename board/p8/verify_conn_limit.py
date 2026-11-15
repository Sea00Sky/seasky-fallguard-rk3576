#!/usr/bin/env python3
"""p8/verify_conn_limit.py — 单独验 MAX_CONN（时序要可控）"""
import socket
import subprocess
import sys
import time

HTTP_PORT = 8080
MAX_CONN = 32
N = 45


def main() -> int:
    socks, ok_conn = [], 0
    for _ in range(N):
        s = socket.socket()
        s.settimeout(3)
        try:
            s.connect(("127.0.0.1", HTTP_PORT))
            s.sendall(b"GET /api/live.mjpg HTTP/1.1\r\nHost: x\r\n\r\n")
            socks.append(s)
        except Exception:
            s.close()
    time.sleep(2.0)

    # 数「拿到 200 响应头」的：被拒的连接会被网关 close ⇒ recv 返回 b""
    served = 0
    for s in socks:
        try:
            s.settimeout(2)
            data = s.recv(64)
            if data.startswith(b"HTTP/1.1 200"):
                served += 1
        except Exception:
            pass
    for s in socks:
        try:
            s.close()
        except Exception:
            pass

    print(f"发起 {N} 个连接，拿到 HTTP 200 的 {served} 个（MAX_CONN={MAX_CONN}）")
    ok = served <= MAX_CONN
    print(f"判据 served <= {MAX_CONN}: {'PASS' if ok else 'FAIL'}"
          f"{'' if ok else '  ⇒ MAX_CONN 没生效！'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
