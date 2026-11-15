#!/usr/bin/env python3
"""p8/verify_s3_state_upsert.py — 任务 4 / 5(b)(c) 板上验收"""
import argparse
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

BOARD_ROOT = Path("/opt/fallguard/deploy")
SHM_PROTO = Path("/opt/fallguard/gateway/shm_proto")
GW_SRC = "/opt/fallguard/gateway/edge_gateway"
GW = "/usr/local/bin/fallguard-gateway"

SHM_PATH = "/dev/shm/fallguard_shm"
SEMS = ["/dev/shm/sem.fg_evt_sem", "/dev/shm/sem.fg_live_sem",
        "/dev/shm/sem.fg_shm_mutex"]

TEST_ROOT = Path("/tmp/fg_s3_test")
CFG = TEST_ROOT / "config.json"
DATA = TEST_ROOT / "data"
DB = DATA / "events.db"
HTTP_PORT = 18080
PUSH_PORT = 19099

sys.path.insert(0, str(SHM_PROTO))
sys.path.insert(0, str(BOARD_ROOT))

PUSHES = []          # 收到的推送体（dict）
_pushes_lk = threading.Lock()


# ------------------------------------------------------------------ 基础设施

def _sh(cmd, timeout=120):
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    return p.returncode, (p.stdout + p.stderr).strip()


def _gw_pid():
    rc, out = _sh("pgrep -f 'fallguard-gatewa[y] --config' | head -1")
    return int(out) if rc == 0 and out.isdigit() else 0


def _clean_proc():
    _sh("pkill -9 -f 'fallguard-gatewa[y]'")
    time.sleep(0.4)
    for p in [SHM_PATH] + SEMS:
        try:
            os.unlink(p)
        except OSError:
            pass


def _write_cfg(cooldown_ms):
    TEST_ROOT.mkdir(parents=True, exist_ok=True)
    CFG.write_text(json.dumps({
        "shm_name": "/fallguard_shm",
        "http_port": HTTP_PORT,
        "sensitivity": "medium",
        "push_threshold": 0.60,
        "push_cooldown_ms": cooldown_ms,
        "push_enabled": True,
        "push_url": "http://127.0.0.1:%d/push" % PUSH_PORT,
        "data_dir": str(DATA),
    }, indent=2))
    os.chmod(CFG, 0o644)


def _make_shm():
    """先建 shm 再起网关：网关是读端，shm 不存在时它连不上会直接退出。"""
    import shm_ipc
    w = shm_ipc.FallGuardShm(create=True)
    w.close()


def _start_gw():
    _sh("pkill -9 -f 'fallguard-gatewa[y]'")
    time.sleep(0.3)
    log = str(TEST_ROOT / "gw.log")
    script = str(TEST_ROOT / "startgw.sh")
    Path(script).write_text("#!/bin/sh\nexec %s --config %s\n" % (GW, CFG))
    os.chmod(script, 0o755)
    subprocess.Popen(["setsid", script], stdout=open(log, "w"),
                     stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
    for _ in range(40):
        time.sleep(0.2)
        if _gw_pid():
            return _gw_pid()
    return 0


class _PushHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(n)
        try:
            with _pushes_lk:
                PUSHES.append(json.loads(raw.decode("utf-8", "replace")))
        except Exception as e:
            with _pushes_lk:
                PUSHES.append({"__parse_error__": str(e), "raw": raw[:200].decode("utf-8", "replace")})
        self.send_response(200)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def log_message(self, *a):
        pass


def _start_receiver():
    srv = ThreadingHTTPServer(("127.0.0.1", PUSH_PORT), _PushHandler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    return srv


def _new_writer():
    from shm_adapter import SensorShmWriter
    return SensorShmWriter()


def _push(w, evt, person=None, frame=None):
    """走生产入口 push_fall_event（含 Python 侧白名单/冷却闸门）。"""
    import numpy as np
    class H:
        _last_ml_result = {"confidence": 0.9}
    person = person or {"person_id": 1, "bbox": [0, 0, 10, 10]}
    frame = frame if frame is not None else np.zeros((240, 320, 3), dtype=np.uint8)
    return w.push_fall_event(evt, person, H(), frame)


def _push_raw(w, evt):
    """绕过 push_fall_event，直接写 shm —— 用于指定 seq / state 的底层用例。"""
    return w._shm.push_event(evt)


def _wait_pushes(n, timeout=5.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        with _pushes_lk:
            if len(PUSHES) >= n:
                return True
        time.sleep(0.05)
    return False


def _db_rows(where="1=1", params=()):
    for _ in range(30):
        if DB.exists():
            break
        time.sleep(0.1)
    con = sqlite3.connect("file:%s?mode=ro" % DB, uri=True, timeout=5)
    try:
        con.row_factory = sqlite3.Row
        return [dict(r) for r in con.execute(
            "SELECT * FROM events WHERE " + where, params).fetchall()]
    finally:
        con.close()


def _fresh(cooldown_ms=0, keep_db=False):
    """清进程 + 清库 + 起网关 + 起 writer。"""
    _clean_proc()
    if not keep_db and DATA.exists():
        shutil.rmtree(DATA)
    DATA.mkdir(parents=True, exist_ok=True)
    _write_cfg(cooldown_ms)
    _make_shm()                     # 必须在起网关之前
    with _pushes_lk:
        PUSHES.clear()
    pid = _start_gw()
    if not pid:
        raise RuntimeError("网关没起来")
    time.sleep(0.3)
    return _new_writer()


# ------------------------------------------------------------------ 用例

def case_migrate():
    """V1：v2 老库就地迁移，老行 state 默认 1"""
    _clean_proc()
    if DATA.exists():
        shutil.rmtree(DATA)
    DATA.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(DB))
    con.executescript(
        "CREATE TABLE events("
        " seq INTEGER PRIMARY KEY, ts_sec INTEGER, ts_ms INTEGER,"
        " person_id INTEGER, class_id INTEGER, confidence REAL,"
        " bbox_x1 REAL, bbox_y1 REAL, bbox_x2 REAL, bbox_y2 REAL,"
        " hybrid_score REAL, reason TEXT, snapshot_seq INTEGER, snap_file TEXT,"
        " pushed INTEGER DEFAULT -1,"
        " created_at TEXT DEFAULT (datetime('now','localtime')));")
    con.execute("INSERT INTO events(seq,ts_sec,ts_ms,person_id,class_id,confidence,"
                "bbox_x1,bbox_y1,bbox_x2,bbox_y2,hybrid_score,reason,snapshot_seq,pushed)"
                " VALUES(1,1700000000,123,1,5,0.9,0,0,10,10,0.8,'dual_confirm',0,1)")
    con.commit()
    con.close()

    _write_cfg(0)
    _make_shm()
    pid = _start_gw()
    time.sleep(0.6)
    ok_pid = (pid != 0)

    con = sqlite3.connect("file:%s?mode=ro" % DB, uri=True)
    con.row_factory = sqlite3.Row
    cols = {r[1] for r in con.execute("PRAGMA table_info(events)")}
    need = {"state", "window_start", "window_end", "model_ver", "prob_fall", "coverage"}
    ok_cols = need.issubset(cols)
    row = dict(con.execute("SELECT * FROM events WHERE seq=1").fetchone())
    idx = [r[0] for r in con.execute(
        "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_events_idem'")]
    con.close()
    ok_legacy = (row.get("state") == 1)
    ok_idx = ("idx_events_idem" in idx)
    print("  网关启动: %-5s 新列齐全: %-5s 老行 state=%s 幂等索引: %s"
          % (ok_pid, ok_cols, row.get("state"), ok_idx))
    return ok_pid and ok_cols and ok_legacy and ok_idx


def case_gate():
    """V2：state=0 拒推但落盘；state=1/2 放行"""
    w = _fresh(cooldown_ms=0)
    ok = True

    # state=0（suspected）：应落盘但不推
    _push(w, {"reason": "dual_confirm", "state": 0,
              "features": {"hybrid_score": 0.9}, "window_start": 9001,
              "window_end": 9051, "prob_fall": 0.7, "coverage": 0.8, "model_ver": 7})
    time.sleep(0.8)
    with _pushes_lk:
        n0 = len(PUSHES)
    rows0 = _db_rows("window_start=9001")
    ok0 = (n0 == 0 and len(rows0) == 1 and rows0[0]["state"] == 0)
    print("  state=0: 推送=%d（应 0） 落盘=%d 行 state=%s"
          % (n0, len(rows0), rows0[0]["state"] if rows0 else "–"))
    ok &= ok0

    # state=1：应推
    _push(w, {"reason": "dual_confirm", "state": 1,
              "features": {"hybrid_score": 0.9}, "window_start": 9002,
              "window_end": 9052, "prob_fall": 0.9, "coverage": 0.8, "model_ver": 7})
    ok1 = _wait_pushes(1, 4)
    with _pushes_lk:
        body1 = PUSHES[-1] if PUSHES else {}
    ev1 = body1.get("event", {})
    print("  state=1: 推送=%s body.state=%s" % (ok1, ev1.get("state")))
    ok &= ok1 and ev1.get("state") == 1

    # state=2（同键撤回）：应推
    _push(w, {"reason": "retract_low_prob", "state": 2,
              "features": {"hybrid_score": 0.1}, "window_start": 9002,
              "window_end": 9102, "prob_fall": 0.2, "coverage": 0.8, "model_ver": 7})
    ok2 = _wait_pushes(2, 4)
    with _pushes_lk:
        body2 = PUSHES[-1] if PUSHES else {}
    ev2 = body2.get("event", {})
    print("  state=2: 推送=%s body.state=%s（撤回 reason 不在白名单也必须过）"
          % (ok2, ev2.get("state")))
    ok &= ok2 and ev2.get("state") == 2
    return ok


def case_idem():
    """V3：同键 confirmed → retracted ⇒ 1 行、终态 retracted、同一 seq"""
    w = _fresh(cooldown_ms=0)
    _push(w, {"reason": "dual_confirm", "state": 1,
              "features": {"hybrid_score": 0.9}, "window_start": 1001,
              "window_end": 1051, "prob_fall": 0.9, "coverage": 0.8, "model_ver": 7},
          person={"person_id": 2, "bbox": [0, 0, 10, 10]})
    _wait_pushes(1, 4)
    _push(w, {"reason": "retract_low_prob", "state": 2,
              "features": {"hybrid_score": 0.1}, "window_start": 1001,
              "window_end": 1101, "prob_fall": 0.2, "coverage": 0.8, "model_ver": 7},
          person={"person_id": 2, "bbox": [0, 0, 10, 10]})
    _wait_pushes(2, 4)
    time.sleep(0.3)

    rows = _db_rows("person_id=2 AND window_start=1001 AND model_ver=7")
    with _pushes_lk:
        seqs = [p.get("event", {}).get("seq") for p in PUSHES[:2]]
        states = [p.get("event", {}).get("state") for p in PUSHES[:2]]
    ok_rows = (len(rows) == 1 and rows[0]["state"] == 2)
    ok_seq = (len(seqs) == 2 and seqs[0] is not None and seqs[0] == seqs[1])
    print("  DB 行数=%d（应 1） 终态 state=%s（应 2）"
          % (len(rows), rows[0]["state"] if rows else "–"))
    print("  推送 seq=%s（应相同） state=%s" % (seqs, states))
    return ok_rows and ok_seq


def case_replay():
    """V4：同一 seq 再投一次 ⇒ 仍 1 行（网关重启重放事件环）"""
    w = _fresh(cooldown_ms=0)
    evt = {"seq": 500, "person_id": 3, "class_id": 5, "confidence": 0.9,
           "bbox": [0, 0, 10, 10], "hybrid_score": 0.9, "reason": "dual_confirm",
           "state": 1, "window_start": 2001, "window_end": 2051,
           "prob_fall": 0.9, "coverage": 0.8, "model_ver": 9}
    _push_raw(w, evt)
    time.sleep(0.6)
    _push_raw(w, evt)          # 完全同一条重放
    time.sleep(0.6)
    rows = _db_rows("seq=500")
    print("  同 seq 投两次 ⇒ DB 行数=%d（应 1）" % len(rows))
    return len(rows) == 1


def case_nomerge():
    """V5：window_start=0 不参与幂等键 —— 两条事件必须留两行"""
    w = _fresh(cooldown_ms=0)
    for i in range(2):
        _push(w, {"reason": "dual_confirm", "state": 1,
                  "features": {"hybrid_score": 0.9}, "window_start": 0,
                  "window_end": 0, "prob_fall": 0.9, "coverage": 0.8, "model_ver": 0},
              person={"person_id": 4, "bbox": [0, 0, 10, 10]})
        time.sleep(0.5)
    rows = _db_rows("person_id=4")
    print("  window_start=0 两条 ⇒ DB 行数=%d（应 2，不能合并）" % len(rows))
    return len(rows) == 2


def case_cooldown():
    """V6：confirmed 吃冷却，retracted 绕过冷却"""
    w = _fresh(cooldown_ms=30000)
    ok = True
    _push(w, {"reason": "dual_confirm", "state": 1,
              "features": {"hybrid_score": 0.9}, "window_start": 3001,
              "window_end": 3051, "prob_fall": 0.9, "coverage": 0.8, "model_ver": 5})
    ok1 = _wait_pushes(1, 4)
    with _pushes_lk:
        n1 = len(PUSHES)
    # 冷却期内第二条 confirmed：Python 侧就该拦掉
    _push(w, {"reason": "dual_confirm", "state": 1,
              "features": {"hybrid_score": 0.9}, "window_start": 3002,
              "window_end": 3052, "prob_fall": 0.9, "coverage": 0.8, "model_ver": 5})
    time.sleep(0.8)
    with _pushes_lk:
        n2 = len(PUSHES)
    ok_cd = (n1 == 1 and n2 == 1)
    # 冷却期内撤回：必须放行
    _push(w, {"reason": "retract_low_prob", "state": 2,
              "features": {"hybrid_score": 0.1}, "window_start": 3001,
              "window_end": 3101, "prob_fall": 0.1, "coverage": 0.8, "model_ver": 5})
    ok3 = _wait_pushes(2, 4)
    print("  首条推送=%s 冷却内第二条被拦=%s 冷却内撤回放行=%s"
          % (ok1, ok_cd, ok3))
    ok &= ok1 and ok_cd and ok3
    return ok


CASES = {
    "migrate": case_migrate,
    "gate": case_gate,
    "idem": case_idem,
    "replay": case_replay,
    "nomerge": case_nomerge,
    "cooldown": case_cooldown,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", choices=list(CASES), action="append")
    ap.add_argument("--build", action="store_true", help="先 make 再验")
    a = ap.parse_args()

    # 不要用 sudo 跑：sudo 把 HOME 换成 /root，会丢掉 ~/.local 里的 numpy，
    # push_fall_event 需要 numpy 帧。网关写 /dev/shm（1777）和 /tmp 都不需要 root。
    if os.geteuid() == 0:
        print(" 建议用普通用户跑（sudo 会丢 numpy）。继续……")

    if a.build:
        print("== 编译 ==")
        rc, out = _sh("cd %s && make 2>&1 | tail -8" % GW_SRC)
        print(out)
        if rc != 0:
            print("编译失败")
            return 2

    _start_receiver()
    names = a.case or list(CASES)
    results = {}
    try:
        for n in names:
            print("\n== %s ==" % n)
            try:
                results[n] = bool(CASES[n]())
            except Exception as e:
                print("  EXCEPTION: %s: %s" % (type(e).__name__, e))
                results[n] = False
            print("  => %s" % ("PASS" if results[n] else "FAIL"))
    finally:
        _clean_proc()
        if DATA.exists():
            shutil.rmtree(DATA, ignore_errors=True)

    print("\n== 汇总 ==")
    for n in names:
        print("  %-10s %s" % (n, "PASS" if results.get(n) else "FAIL"))
    allok = all(results.values())
    print("\n%s (%d/%d)" % ("ALL PASS" if allok else "FAILED",
                            sum(1 for v in results.values() if v), len(results)))
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())
