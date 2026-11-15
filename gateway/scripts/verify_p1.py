#!/usr/bin/env python3
"""verify_p1.py — 验收：回放 hospital_ward 场景（加速），读回共享内存，
逐字段与 scenario 对比，并验证环形队列/状态/帧语义。全过则打印 PASS。"""

import json
import os
import subprocess
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "shm_proto"))
from shm_ipc import FallGuardShm, FG_EVT_RING_CAP  # noqa: E402

SCENARIO = os.path.join(BASE, "tests", "scenario_hospital_ward.json")
RING_CAP = FG_EVT_RING_CAP


def main():
    with open(SCENARIO, encoding="utf-8") as f:
        events = json.load(f)["events"]
    n = len(events)  # 11

    # 1) 回放场景（10 倍加速，跑完保留 shm）
    mock = os.path.join(BASE, "sensor", "mock_fall_generator.py")
    cmd = [sys.executable, mock, "--scenario", SCENARIO,
           "--mode", "once", "--scale", "10", "--no-clean", "--fresh", "--quiet"]
    print(">", " ".join(cmd))
    p = subprocess.run(cmd, capture_output=True, text=True)
    print(p.stdout)
    if p.returncode != 0:
        print("mock 异常退出，stderr:\n", p.stderr)
        return 1

    shm = FallGuardShm(create=False)
    got = shm.peek_events()
    ring = shm._shm
    checks = []

    def ck(name, ok, a, b):
        checks.append((name, bool(ok), a, b))

    # 2) 环形队列：写入 n 条，容量 RING_CAP；超出容量则覆盖最旧，只留最后 keep 条
    keep = min(n, RING_CAP)
    ck("evt_count==keep", ring.evt_count == keep, ring.evt_count, keep)
    ck("head 前移 %d" % (n - keep), ring.evt_head == (n - keep) % RING_CAP,
       ring.evt_head, (n - keep) % RING_CAP)
    if n >= RING_CAP:
        ck("head==tail (满)", ring.evt_head == ring.evt_tail, ring.evt_head, ring.evt_tail)
    ck("幸存事件数==keep", len(got) == keep, len(got), keep)
    seqs = [e["seq"] for e in got]
    ck("seq 单调 %d..%d" % (n - keep + 1, n), seqs == list(range(n - keep + 1, n + 1)),
       seqs, list(range(n - keep + 1, n + 1)))

    # 3) 幸存事件逐字段对比 scenario（scenario 下标 n-keep .. n-1）
    for i, (exp, g) in enumerate(zip(events[n - keep:], got)):
        tag = "evt%d" % (i + 1 + n - keep)
        ck("%s class_id" % tag, g["class_id"] == exp["class_id"], g["class_id"], exp["class_id"])
        ck("%s person_id" % tag, g["person_id"] == exp["person_id"], g["person_id"], exp["person_id"])
        ck("%s hybrid(%.2f)" % (tag, exp["hybrid_score"]),
           abs(g["hybrid_score"] - exp["hybrid_score"]) < 1e-3, g["hybrid_score"], exp["hybrid_score"])
        ck("%s reason" % tag, g["reason"] == exp["reason"], g["reason"], exp["reason"])
        ck("%s bbox" % tag, all(abs(a - b) < 1e-3 for a, b in zip(g["bbox"], exp["bbox"])),
           g["bbox"], exp["bbox"])
        exp_snap = bool(exp.get("snapshot", True))
        ck("%s snapshot_seq>0==%s" % (tag, exp_snap),
           (g["snapshot_seq"] > 0) == exp_snap, g["snapshot_seq"], exp_snap)

    # 4) 状态区 / 帧
    ck("status_seq>0", ring.status_seq > 0, ring.status_seq, 0)
    ck("frame_count 随状态递增", ring.frame_count >= 10, ring.frame_count, 10)
    ck("last_evt_seq==11", ring.last_evt_seq == n, ring.last_evt_seq, n)
    snap_new = max(ring.alarm_snap[0].frame_seq, ring.alarm_snap[1].frame_seq)
    last_snap_evt = [g for g in got if g["snapshot_seq"] > 0][-1]
    ck("最新快照 frame_seq==最后快照事件 snapshot_seq",
       snap_new == last_snap_evt["snapshot_seq"], snap_new, last_snap_evt["snapshot_seq"])
    # v3 起 shm 只剩告警快照一块帧区，直播帧不再进共享内存，这里只查快照可读
    ck("告警快照 JPEG 长度>0",
       max(ring.alarm_snap[0].jpeg_len, ring.alarm_snap[1].jpeg_len) > 0,
       max(ring.alarm_snap[0].jpeg_len, ring.alarm_snap[1].jpeg_len), 0)

    # 5) dump 快照（人工复核用）
    print("\n--- dump_shm.py ---")
    print(shm.dump())
    shm.cleanup()

    all_ok = all(ok for _, ok, _, _ in checks)
    for name, ok, a, b in checks:
        print("  [%s] %s" % ("OK" if ok else "FAIL", name))
    print("\nVERIFY_P1", "PASS" if all_ok else "FAIL")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
