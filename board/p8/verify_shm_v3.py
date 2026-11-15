#!/usr/bin/env python3
"""p8/verify_shm_v3.py — 任务 1（shm v3）板上验收"""
import argparse
import ctypes
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

BOARD_ROOT = Path("/opt/fallguard/deploy")
SHM_PROTO = Path("/opt/fallguard/gateway/shm_proto")
GW = "/usr/local/bin/fallguard-gateway"
CFG = "/etc/fallguard/config.json"
HTTP_PORT = 8080
SHM_NAME = "/fallguard_shm"
SHM_PATH = "/dev/shm/fallguard_shm"
SEMS = ["/dev/shm/sem.fg_evt_sem", "/dev/shm/sem.fg_live_sem",
        "/dev/shm/sem.fg_shm_mutex"]

sys.path.insert(0, str(SHM_PROTO))
sys.path.insert(0, str(BOARD_ROOT))


def _sh(cmd: str, timeout: float = 60) -> tuple[int, str]:
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                       timeout=timeout)
    return p.returncode, (p.stdout + p.stderr).strip()


def _gw_pid() -> int:
    rc, out = _sh("pgrep -f 'fallguard-gatewa[y] --config' | head -1")
    return int(out) if rc == 0 and out.isdigit() else 0


def _clean_shm():
    _sh("pkill -9 -f 'fallguard-gatewa[y]'")
    _sh("pkill -9 -f 'SensorShmWriter'")
    time.sleep(0.6)
    for p in [SHM_PATH] + SEMS:
        try:
            os.unlink(p)
        except OSError:
            pass


def _start_gw(expect_fail: bool = False) -> tuple[int, str]:
    """起网关，返回 (pid, 日志)。expect_fail=True 时返回退出码而非 pid。"""
    _sh("pkill -f 'fallguard-gatewa[y] --config'")
    time.sleep(0.4)
    log = "/tmp/gw_v3_%d.log" % os.getpid()
    script = "/tmp/startgw_v3_%d.sh" % os.getpid()
    Path(script).write_text("#!/bin/sh\nexec %s --config %s\n" % (GW, CFG))
    os.chmod(script, 0o755)
    subprocess.Popen(["setsid", script],
                     stdout=open(log, "w"), stderr=subprocess.STDOUT,
                     stdin=subprocess.DEVNULL)
    for _ in range(30):
        time.sleep(0.2)
        if _gw_pid():
            return _gw_pid(), Path(log).read_text(errors="replace")
    return 0, Path(log).read_text(errors="replace") if Path(log).exists() else ""


def _http(path: str, timeout: float = 6):
    """返回 (status, body_bytes)；连接失败返回 (None, 异常文本)。"""
    try:
        with urllib.request.urlopen(
                "http://127.0.0.1:%d%s" % (HTTP_PORT, path), timeout=timeout) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()
    except Exception as e:
        return None, str(e).encode()


# ------------------------------------------------------------------ V1

def case_layout() -> bool:
    """V1：C 断言（编译期已过）+ Python 断言 + 跨语言 shm_size 互认"""
    import shm_ipc
    print("  Python: FgEvent=%d FgFrame=%d FgShm=%d ver=%d cap=%d"
          % (ctypes.sizeof(shm_ipc.FgEvent), ctypes.sizeof(shm_ipc.FgFrame),
             ctypes.sizeof(shm_ipc.FgShm), shm_ipc.FG_SHM_VERSION,
             shm_ipc.FG_EVT_RING_CAP))
    ok = (ctypes.sizeof(shm_ipc.FgShm) == 414344
          and ctypes.sizeof(shm_ipc.FgEvent) == 144
          and shm_ipc.FG_SHM_VERSION == 3
          and shm_ipc.FG_EVT_RING_CAP == 32)
    print("  Python 侧常量与 v3 期望一致: %s" % ok)

    # C 侧的编译期断言只能通过「重新编译」来证明它还在
    rc, out = _sh("cd /opt/fallguard/gateway/edge_gateway && "
                  "touch include/fallguard_shm.h && make 2>&1 | tail -6")
    c_ok = rc == 0 and "error" not in out.lower()
    print("  C 侧重新编译（12 条静态断言生效）: %s" % c_ok)
    if not c_ok:
        print("    " + out.replace("\n", "\n    "))
    return ok and c_ok


# ------------------------------------------------------------------ V2

def case_triple_check() -> bool:
    """V2：伪造 version=2 与 shm_size 错的 shm，网关必须拒启动"""
    import shm_ipc
    ok = True

    for tag, mutate in (
        ("version=2", lambda s: setattr(s, "version", 2)),
        ("shm_size 错", lambda s: setattr(s, "shm_size", 820320)),
    ):
        _clean_shm()
        # 造一个合法 v3 shm，再篡改其中一个校验字段
        w = shm_ipc.FallGuardShm(create=True)
        mutate(w._shm)
        w.close()
        t0 = time.time()
        pid, log = _start_gw(expect_fail=True)
        # 网关应当连上就退出（或根本没起来）
        time.sleep(1.2)
        still = _gw_pid()
        rejected = (still == 0)
        # `log_status` 走 syslog（log.c: vsyslog），不写 stdout ⇒
        # 读 /tmp/gw_*.log 永远是空的。必须查 journal。
        _, jl = _sh("journalctl -t fallguard-gw --since '@%d' --no-pager -o cat "
                    "2>/dev/null | tail -20" % int(t0 - 2))
        if not jl.strip():
            _, jl = _sh("journalctl --since '@%d' --no-pager -o cat 2>/dev/null "
                        "| grep -i fallguard | tail -20" % int(t0 - 2))
        hit = any(k in jl for k in ("版本不匹配", "结构漂移", "拒绝启动",
                                    "版本", "漂移"))
        print("  %-12s 网关被拒: %-5s 日志命中关键句: %s"
              % (tag, rejected, hit))
        if not rejected or not hit:
            print("    stdout: " + log.strip().replace("\n", " | ")[-160:])
            print("    journal: " + jl.strip().replace("\n", " | ")[-260:])
            ok = False
        _sh("pkill -9 -f 'fallguard-gatewa[y]'")

    _clean_shm()
    return ok


# ------------------------------------------------------------------ V3

def case_ring() -> bool:
    """V3：灌 40 条事件，环里必须保留 32 条（v2 只能留 8）"""
    import shm_ipc
    from shm_adapter import SensorShmWriter
    import numpy as np

    _clean_shm()
    w = shm_ipc.FallGuardShm(create=True) # 建 shm，但不起网关
    w.close()

    a = SensorShmWriter()
    a._cfg["push_cooldown_ms"] = 0
    a._last_push_ms = 0
    frame = np.zeros((240, 320, 3), dtype=np.uint8)

    class H:
        _last_ml_result = {"confidence": 0.9}

    N = 40
    for i in range(N):
        a.push_fall_event(
            {"reason": "dual_confirm", "features": {"hybrid_score": 0.9},
             "window_start": i, "window_end": i + 50, "prob_fall": 0.8,
             "coverage": 0.75, "model_ver": 0xABCD},
            {"person_id": (i % 3) + 1, "bbox": [0, 0, 10, 10]}, H(), frame)

    r = shm_ipc.FallGuardShm(create=False)
    cnt = r._shm.evt_count
    cap = shm_ipc.FG_EVT_RING_CAP
    print("  灌 %d 条 ⇒ 环内 evt_count=%d（容量 %d）" % (N, cnt, cap))
    ok_cnt = (cnt == cap)
    print("  环被填满且不超容: %s" % ok_cnt)

    # 新字段必须真的落盘（不是只扩了结构没填值）
    evs = r.peek_events()
    last = evs[-1] if evs else None
    if last:
        print("  末条: state=%d win=%d..%d pf=%.2f cov=%.2f mv=0x%X"
              % (last["state"], last["window_start"], last["window_end"],
                 last["prob_fall"], last["coverage"], last["model_ver"]))
    ok_fields = (last is not None
                 and last["state"] == shm_ipc.FG_EVT_ST_CONFIRMED
                 and last["window_end"] - last["window_start"] == 50
                 and abs(last["prob_fall"] - 0.8) < 1e-3
                 and abs(last["coverage"] - 0.75) < 1e-3
                 and last["model_ver"] == 0xABCD)
    print("  v3 新字段落盘正确: %s" % ok_fields)
    # 覆盖语义：40 条灌进 32 深环 ⇒ 最旧的 8 条必须已被顶掉
    seqs = [e["seq"] for e in evs]
    oldest_kept = min(seqs) if seqs else -1
    print("  环内 seq 范围 %d..%d（应保留最后 32 条，最旧 8 条被顶掉）"
          % (oldest_kept, max(seqs) if seqs else -1))
    ok_overwrite = (len(seqs) == cap and oldest_kept == N - cap + 1)
    print("  丢最旧、留最新: %s" % ok_overwrite)

    r.close()
    _clean_shm()
    return ok_cnt and ok_fields and ok_overwrite


# ------------------------------------------------------------------ V4

def case_snap() -> bool:
    """V4：快照通路 —— /api/snap 返回真 JPEG（SOI/EOI 标记齐全）"""
    import shm_ipc
    from shm_adapter import SensorShmWriter
    import numpy as np

    _clean_shm()
    w = shm_ipc.FallGuardShm(create=True)
    w.close()
    pid, _ = _start_gw()
    if not pid:
        print("  [FAIL] 网关没起来")
        return False
    time.sleep(0.8)

    a = SensorShmWriter()
    a._cfg["push_cooldown_ms"] = 0
    a._last_push_ms = 0
    # 造一张有内容的图，避免全黑 JPEG 过小掩盖问题
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[100:380, 150:490] = (40, 180, 220)

    class H:
        _last_ml_result = {"confidence": 0.9}

    a.push_fall_event({"reason": "dual_confirm", "features": {"hybrid_score": 0.9}},
                      {"person_id": 1, "bbox": [10, 10, 50, 50]}, H(), frame)

    st, body = _http("/api/snap")
    print("  /api/snap → HTTP %s, %d B" % (st, len(body) if body else 0))
    ok_st = (st == 200)
    ok_jpeg = bool(body) and body[:2] == b"\xff\xd8" and body[-2:] == b"\xff\xd9"
    print("  是完整 JPEG (SOI+EOI): %s" % ok_jpeg)

    st2, body2 = _http("/api/status")
    ok_status = (st2 == 200)
    if ok_status:
        d = json.loads(body2)
        print("  /api/status → 200, people=%s" % d.get("status", {}).get("people"))
    else:
        print("  /api/status → %s" % st2)

    _sh("pkill -9 -f 'fallguard-gatewa[y]'")
    _clean_shm()
    return ok_st and ok_jpeg and ok_status


# ------------------------------------------------------------------ V5

def case_live_gone() -> bool:
    """V5：/api/live.mjpg 必须 410（v3 起直播不在 shm 里）"""
    import shm_ipc
    _clean_shm()
    w = shm_ipc.FallGuardShm(create=True)
    w.close()
    pid, _ = _start_gw()
    if not pid:
        print("  [FAIL] 网关没起来")
        return False
    time.sleep(0.6)

    st, body = _http("/api/live.mjpg")
    print("  /api/live.mjpg → HTTP %s" % st)
    print("  body: %s" % (body[:70].decode("utf-8", "replace") if body else ""))
    ok = (st == 410)
    print("  是 410 Gone（不是 404/200）: %s" % ok)

    st2, _ = _http("/api/live")
    print("  /api/live → HTTP %s（同 410）: %s" % (st2, st2 == 410))

    _sh("pkill -9 -f 'fallguard-gatewa[y]'")
    _clean_shm()
    return ok and st2 == 410


CASES = {"layout": case_layout, "check": case_triple_check, "ring": case_ring,
         "snap": case_snap, "live": case_live_gone}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", choices=list(CASES), default=None)
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()

    todo = list(CASES) if args.all or not args.case else [args.case]
    results = {}
    for name in todo:
        print("=== %s ===" % name)
        try:
            results[name] = CASES[name]()
        except Exception as e:
            import traceback
            traceback.print_exc()
            results[name] = False
        print()

    print("=== 汇总 ===")
    for k, v in results.items():
        print("  %-8s %s" % (k, "PASS" if v else "FAIL"))
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
