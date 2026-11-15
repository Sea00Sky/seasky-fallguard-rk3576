#!/usr/bin/env python3
"""mock_fall_generator.py — 模拟事件源：按 scenario 场景脚本写共享内存。"""

import argparse
import base64
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "shm_proto"))
from shm_ipc import FallGuardShm  # noqa: E402

# 1x1 白色 JPEG 兜底（cv2 不可用时的占位帧；板上感知本来就有 cv2，走画帧）
PLACEHOLDER_JPEG_B64 = (
    "/9j/4AAQSkZJRgABAQEAYABgAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0a"
    "HBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/wAALCAABAAEBAREA/8QAFAABAAAAAAAA"
    "AAAAAAAAAAAACf/EABQQAQAAAAAAAAAAAAAAAAAAAAD/2gAIAQEAAD8AVN//2Q=="
)
PLACEHOLDER_JPEG = base64.b64decode(PLACEHOLDER_JPEG_B64)

# 三档灵敏度预设（与网关 config.c 的默认档位一致）
SENS_PRESETS = {
    "low":    {"level": 0, "push_threshold": 0.75, "push_cooldown_ms": 6000},
    "medium": {"level": 1, "push_threshold": 0.60, "push_cooldown_ms": 3000},
    "high":   {"level": 2, "push_threshold": 0.45, "push_cooldown_ms": 1500},
}


def _make_jpeg(seq, text=""):
    """优先用 cv2 画带帧号/文字的图；否则占位 JPEG。"""
    try:
        import cv2
        import numpy as np
        img = np.full((240, 320, 3), (40, 40, 60), dtype=np.uint8)
        cv2.putText(img, "fallguard mock #%d" % seq, (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        if text:
            cv2.putText(img, text, (20, 200), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (180, 255, 180), 2)
        ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        if ok:
            return buf.tobytes()
    except Exception:
        pass
    return PLACEHOLDER_JPEG


def load_scenario(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data, data.get("events", [])


def self_test():
    """写一条已知事件 → 读回 → 逐字段断言一致。"""
    shm = FallGuardShm()
    evt = {
        "seq": 42, "person_id": 3, "class_id": 5, "confidence": 0.92,
        "bbox": [12.5, 30.0, 198.25, 233.5], "hybrid_score": 0.85,
        "reason": "dual_confirm", "ts_sec": 1700000000, "ts_ms": 123,
    }
    snap = _make_jpeg(1, "self-test")
    ev_seq = shm.push_event(evt, snap, snap_wh=(320, 240))
    got = shm.take_event()
    fr = shm.pick_latest_frame("snap")

    checks = []

    def ck(name, a, b, tol=0.0):
        if isinstance(a, float) or isinstance(b, float):
            ok = abs(float(a) - float(b)) <= (tol if tol else 1e-4)
        else:
            ok = (a == b)
        checks.append((name, ok, a, b))

    ck("seq", got["seq"], 42)
    ck("ts_sec", got["ts_sec"], 1700000000)
    ck("ts_ms", got["ts_ms"], 123)
    ck("person_id", got["person_id"], 3)
    ck("class_id", got["class_id"], 5)
    ck("confidence", got["confidence"], 0.92, tol=1e-3)
    ck("bbox", got["bbox"], [12.5, 30.0, 198.25, 233.5])
    ck("hybrid_score", got["hybrid_score"], 0.85, tol=1e-3)
    ck("reason", got["reason"], "dual_confirm")
    ck("ev_seq==42", ev_seq, 42)
    ck("snapshot_seq==frame_seq", got["snapshot_seq"], fr["frame_seq"])
    ck("snap jpeg_len", fr["jpeg_len"], len(snap))
    ck("snap jpeg bytes", fr["jpeg"] == snap, True)
    ck("snap w/h", (fr["w"], fr["h"]), (320, 240))
    ck("status_seq>0", shm._shm.status_seq > 0, True)
    ck("last_evt_seq==42", shm._shm.last_evt_seq, 42)
    ck("ring empty after take", shm._shm.evt_count, 0)

    all_ok = all(ok for _, ok, _, _ in checks)
    for name, ok, a, b in checks:
        print("  [%s] %s: %r == %r" % ("OK" if ok else "FAIL", name, a, b))
    shm.cleanup()
    print("SELF-TEST", "PASS" if all_ok else "FAIL")
    return 0 if all_ok else 1


def _fresh():
    """启动前清理已有 shm/信号量（重跑/调试用，避免残留污染断言）。"""
    import shm_ipc
    shm_ipc._libc.shm_unlink(shm_ipc.FallGuardShm.SHM_DEFAULT.encode())
    for n in (shm_ipc.FallGuardShm.SEM_MUTEX, shm_ipc.FallGuardShm.SEM_EVT,
              shm_ipc.FallGuardShm.SEM_LIVE):
        shm_ipc._libc.sem_unlink(n.encode())
    print("[mock] --fresh 已清理残留 shm/sem", flush=True)


def run(args):
    if args.fresh:
        _fresh()
    data, events = load_scenario(args.scenario)
    scale = float(args.scale) if args.scale and args.scale > 0 else 1.0
    interval_ms = int(args.interval) if args.interval else int(data.get("interval_ms", 1000))
    interval_ms = int(interval_ms / scale)
    snap_enabled = args.snapshot if args.snapshot is not None else bool(data.get("snapshot", True))
    sens = SENS_PRESETS.get(str(data.get("sensitivity", "medium")), SENS_PRESETS["medium"])

    shm = FallGuardShm()
    # 配置区按 scenario 预设播种（真实系统由网关 T4 写；此处先播种便于验证）
    shm._shm.cfg_seq += 1
    shm._shm.sensitivity_level = sens["level"]
    shm._shm.push_threshold = sens["push_threshold"]
    shm._shm.push_cooldown_ms = sens["push_cooldown_ms"]
    shm._shm.push_enabled = 1
    shm._shm.cfg_crc32 = shm._config_crc32()

    people = max((e.get("person_id", 1) for e in events), default=1)
    fps = 16.6
    start = time.time()
    next_status = start
    seq = 0
    idx = 0
    last_fall = 0.0
    meta = data.get("meta", {})
    print("[mock] 启动: %s mode=%s interval=%dms sensitivity=%s events=%d"
          % (meta.get("name", "?"), args.mode, interval_ms,
             data.get("sensitivity", "medium"), len(events)), flush=True)
    try:
        while True:
            now = time.time()
            t_ms = int((now - start) * 1000 * scale)   # 场景时间 = 墙钟 × scale
            # 定时：只刷状态（v3 起 shm 里没有直播帧区域，画面不进共享内存）
            if now >= next_status:
                next_status += interval_ms / 1000.0
                falling = 1 if (idx > 0 and now - last_fall < 2.0) else 0
                shm.update_status(falling, people, fps)
                if not args.quiet:
                    print("[mock] status falling=%d people=%d fps=%.1f frame_seq=%d"
                          % (falling, people, fps, shm._frame_seq), flush=True)
            # 事件
            while idx < len(events) and t_ms >= events[idx]["t_ms"]:
                ev = events[idx]
                seq += 1
                snap = None
                if snap_enabled and ev.get("snapshot", True):
                    snap = _make_jpeg(seq, ev.get("class", ""))
                d = {
                    "seq": seq,
                    "person_id": ev["person_id"],
                    "class_id": ev["class_id"],
                    "confidence": float(ev.get("confidence", 0.9)),
                    "bbox": ev.get("bbox", [0, 0, 320, 240]),
                    "hybrid_score": float(ev["hybrid_score"]),
                    "reason": ev.get("reason", "mock"),
                }
                shm.push_event(d, snap)
                print("[mock] EVENT seq=%d %s(%d) hybrid=%.2f reason=%s snap=%s"
                      % (seq, ev.get("class", "?"), ev["class_id"], ev["hybrid_score"],
                         ev.get("reason", "mock"), "yes" if snap else "no"), flush=True)
                if ev["class_id"] == 5:
                    last_fall = time.time()
                idx += 1
            # 场景播完
            if idx >= len(events):
                if args.mode == "once":
                    print("[mock] 场景播完，退出", flush=True)
                    break
                print("[mock] 循环重播", flush=True)
                idx = 0
                start = time.time()
                next_status = start
            time.sleep(0.01)
    finally:
        if args.no_clean:
            print("[mock] --no-clean 保留 shm 供 dump", flush=True)
        else:
            shm.cleanup()
            print("[mock] 已清理 shm/sem", flush=True)


def main():
    ap = argparse.ArgumentParser(description="FallGuard 模拟事件源")
    ap.add_argument("--scenario", help="scenario JSON 路径")
    ap.add_argument("--mode", choices=["once", "loop"], default="once")
    ap.add_argument("--interval", type=int, default=None, help="状态刷新周期 ms（覆盖 scenario）")
    ap.add_argument("--scale", type=float, default=1.0, help="时间加速倍率（scenario t_ms 除以该值，验收回放用）")
    ap.add_argument("--snapshot", dest="snapshot", action="store_true", default=None)
    ap.add_argument("--no-snapshot", dest="snapshot", action="store_false")
    ap.add_argument("--no-clean", action="store_true", help="退出时保留 shm 供 dump")
    ap.add_argument("--fresh", action="store_true", help="启动前清理已有 shm/sem")
    ap.add_argument("--quiet", action="store_true", help="不打印逐条 status（验收脚本用）")
    ap.add_argument("--self-test", action="store_true", help="写读自检后退出")
    args = ap.parse_args()
    if args.self_test:
        sys.exit(self_test())
    if not args.scenario:
        ap.error("需要 --scenario 或 --self-test")
    run(args)


if __name__ == "__main__":
    main()
