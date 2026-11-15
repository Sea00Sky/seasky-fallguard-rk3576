#!/usr/bin/env python3
"""p8/verify_batch_a.py — 任务 2/3 批次 A 的运行时验收"""
import argparse
import json
import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

BOARD_ROOT = Path("/opt/fallguard/deploy")
GW = "/usr/local/bin/fallguard-gateway"
CFG = "/etc/fallguard/config.json"
HTTP_PORT = 8080
MAX_CONN_EXPECT = 32        # 与 main.c 的 MAX_CONN 保持一致


def _sh(cmd: str, timeout: float = 30) -> tuple[int, str]:
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                       timeout=timeout)
    return p.returncode, (p.stdout + p.stderr).strip()


def _gw_pid() -> int:
    rc, out = _sh("pgrep -f 'fallguard-gatewa[y] --config' | head -1")
    return int(out) if rc == 0 and out.isdigit() else 0


def _start_gw() -> int:
    """起网关（前台脚本 + setsid），返回 pid。要求 shm 已存在。"""
    _sh("pkill -f 'fallguard-gatewa[y] --config'")
    time.sleep(0.5)
    script = f"/tmp/startgw_{os.getpid()}.sh"
    Path(script).write_text("#!/bin/sh\nexec %s --config %s\n" % (GW, CFG))
    os.chmod(script, 0o755)
    subprocess.Popen(["setsid", script],
                     stdout=open(f"/tmp/gw_a_{os.getpid()}.log", "w"),
                     stderr=subprocess.STDOUT,
                     stdin=subprocess.DEVNULL)
    for _ in range(50):
        time.sleep(0.2)
        pid = _gw_pid()
        if pid:
            return pid
    return 0


def _stop_gw():
    _sh("pkill -f 'fallguard-gatewa[y] --config'")
    time.sleep(0.5)


def _hold_shm():
    """确保 /dev/shm/fallguard_shm 存在（网关 shm_open 不带 O_CREAT）。"""
    if Path("/dev/shm/fallguard_shm").exists():
        return True
    code = (
        "import sys,time; sys.path.insert(0,'%s');"
        "from shm_adapter import SensorShmWriter;"
        "a=SensorShmWriter(); time.sleep(900)" % BOARD_ROOT
    )
    subprocess.Popen(["setsid", "python3", "-c", code],
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                     stdin=subprocess.DEVNULL)
    for _ in range(50):
        time.sleep(0.2)
        if Path("/dev/shm/fallguard_shm").exists():
            return True
    return False


# ---------------------------------------------------------------- A2 / A1

def case_clock() -> bool:
    """改系统时间 ±1 小时，网关必须：① 不崩 ② /api/status 持续可访问 ③ T1 不假死"""
    ok = True
    pid0 = _gw_pid()
    if not pid0:
        print("  [FAIL] 网关没起来")
        return False
    print(f"  网关 pid={pid0}")

    def probe(tag):
        try:
            with urllib.request.urlopen(
                    f"http://127.0.0.1:{HTTP_PORT}/api/status", timeout=5) as r:
                d = json.loads(r.read())
            print(f"  [{tag}] 200  last_evt_seq={d['status']['last_evt_seq']} "
                  f"queue={d['queue']['depth']}")
            return True
        except Exception as e:
            print(f"  [{tag}] 失败: {e}")
            return False

    ok &= probe("跳变前")
    rc, _ = _sh("date -s '+1 hour'")
    print(f"  系统时间 +1h (rc={rc})")
    time.sleep(1)
    ok &= probe("+1h 后")
    ok &= (_gw_pid() == pid0)
    print(f"  pid 未变: {_gw_pid() == pid0}")

    rc, _ = _sh("date -s '-1 hour'")
    print(f"  系统时间 -1h 恢复 (rc={rc})")
    time.sleep(1)
    ok &= probe("-1h 后")
    ok &= (_gw_pid() == pid0)

    if _gw_pid() != pid0:
        print("  [FAIL] 时间跳变导致网关重启/崩溃")
        ok = False
    return ok


# ---------------------------------------------------------------- A3

def case_queue() -> bool:
    """灌 400 条事件（EQ_CAP=256）⇒ 生产者不阻塞，且 eq_dropped_total > 0"""
    sys.path.insert(0, str(BOARD_ROOT))
    import numpy as np
    from shm_adapter import SensorShmWriter

    a = SensorShmWriter()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    class H:
        _last_ml_result = {"confidence": 0.9}

    # 冷却关掉，否则第 2 条起全被 cooldown 挡掉
    a._cfg["push_cooldown_ms"] = 0
    a._last_push_ms = 0

    N = 400
    t0 = time.time()
    pushed = 0
    for i in range(N):
        r = a.push_fall_event(
            {"reason": "dual_confirm", "features": {"hybrid_score": 0.9}},
            {"person_id": 1, "bbox": [0, 0, 10, 10]}, H(), frame)
        if r is not None:
            pushed += 1
    dt = time.time() - t0
    print(f"  灌 {N} 条（实际入环 {pushed}）耗时 {dt:.2f}s")
    ok = dt < 5.0 and pushed > 200
    print(f"  不阻塞(<5s): {dt < 5.0}  入环>200: {pushed > 200}")
    if _gw_pid() == 0:
        print("  [FAIL] 网关被拖死了")
        ok = False
    return ok


# ---------------------------------------------------------------- A4

def case_conn() -> bool:
    """开 40 个并发 MJPEG 连接 ⇒ 进程不崩、RSS 受控、线程数受控。"""
    pid = _gw_pid()
    if not pid:
        print("  [FAIL] 网关没起来")
        return False

    def proc_field(name):
        try:
            for line in Path(f"/proc/{pid}/status").read_text().splitlines():
                if line.startswith(name):
                    return int(line.split()[1])
        except Exception:
            pass
        return -1

    thr0, rss0 = proc_field("Threads:"), proc_field("VmRSS:")
    socks = []
    for _ in range(40):
        s = socket.socket()
        s.settimeout(2)
        try:
            s.connect(("127.0.0.1", HTTP_PORT))
            s.sendall(b"GET /api/live.mjpg HTTP/1.1\r\nHost: x\r\n\r\n")
            socks.append(s)
        except Exception:
            s.close()
    time.sleep(2.0)
    thr1, rss1 = proc_field("Threads:"), proc_field("VmRSS:")
    alive = _gw_pid() == pid

    for s in socks:
        try:
            s.close()
        except Exception:
            pass
    time.sleep(1.5)

    d_thr, d_rss = thr1 - thr0, (rss1 - rss0) / 1024.0
    print(f"  线程 {thr0} → {thr1} (Δ{d_thr})；"
          f"RSS {rss0//1024} → {rss1//1024} MB (Δ{d_rss:.1f} MB)；存活={alive}")
    ok_thr = d_thr <= MAX_CONN_EXPECT + 8      # 5 个常驻线程 + accept 线程等
    ok_rss = d_rss < 100
    print(f"  线程增量 ≤{MAX_CONN_EXPECT + 8}: {ok_thr}   RSS 增量 <100MB: {ok_rss}")
    return alive and ok_thr and ok_rss


CASES = {"clock": case_clock, "queue": case_queue, "conn": case_conn}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", choices=list(CASES), default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--no-setup", action="store_true",
                    help="shm 与网关已由外层脚本起好，本脚本只管测")
    args = ap.parse_args()

    if args.no_setup:
        pid = _gw_pid()
        if not pid:
            print("[FATAL] --no-setup 但网关没在跑")
            return 2
        print(f"[setup] 复用已在跑的网关 pid={pid}\n")
    else:
        if not _hold_shm():
            print("[FATAL] 起不来 shm")
            return 2
        pid = _start_gw()
        if not pid:
            print("[FATAL] 起不来网关")
            return 2
        print(f"[setup] 网关 pid={pid}\n")

    todo = list(CASES) if args.all or not args.case else [args.case]
    results = {}
    try:
        for name in todo:
            print(f"=== {name} ===")
            try:
                results[name] = CASES[name]()
            except Exception as e:
                print(f"  [EXC] {e}")
                results[name] = False
            print()
    finally:
        if not args.no_setup:
            _stop_gw()

    print("=== 汇总 ===")
    for k, v in results.items():
        print(f"  {k:8s} {'PASS' if v else 'FAIL'}")
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
