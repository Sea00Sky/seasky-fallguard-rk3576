#!/usr/bin/env python3
"""p8/verify_batch_b.py — 任务 2/3 批次 B 验收"""
import json
import re
import subprocess
import sys
import time

UNIT = "fallguard-gateway"


def sh(cmd: str, timeout: float = 30) -> tuple[int, str]:
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                       timeout=timeout)
    return p.returncode, (p.stdout + p.stderr).strip()


def show_props() -> dict:
    """读 systemd 实际生效的属性（不是读 unit 文件 —— 文件对≠加载了）。"""
    props = ("ActiveState", "SubState", "NRestarts", "WatchdogUSec",
             "MemoryMax", "LimitNOFILE", "StartLimitBurst",
             "StartLimitIntervalUSec", "Type")
    _, out = sh("systemctl show " + " ".join(f"-p {p}" for p in props)
                + f" {UNIT}")
    d = {}
    for line in out.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            d[k] = v
    return d


def case_notify() -> bool:
    """B1：服务 active，且跨过一个 WatchdogSec(30s) 周期后仍未被重启"""
    p0 = show_props()
    print(f"  ActiveState={p0.get('ActiveState')} SubState={p0.get('SubState')} "
          f"WatchdogUSec={p0.get('WatchdogUSec')}")
    if p0.get("ActiveState") != "active":
        print("  [FAIL] 服务不是 active")
        return False
    # `systemctl show -p WatchdogUSec` 返回的是带单位的字符串（`30s`），
    # 不是纯微秒数 —— 第一版直接 int 就炸了。
    wd = (p0.get("WatchdogUSec") or "").strip()
    if wd in ("", "infinity", "0"):
        print(f"  [FAIL] WatchdogUSec={wd!r} ⇒ WatchdogSec 没生效"
              f"（Type 必须是 notify）")
        return False
    mult = {"s": 1, "ms": 1e-3, "us": 1e-6, "min": 60}
    num = "".join(c for c in wd if c.isdigit() or c == ".")
    unit = wd[len(num):] or "s"
    wd_s = float(num) * mult.get(unit, 1)
    print(f"  看门狗周期 {wd_s:.0f}s")
    wait = wd_s * 2.5                 # 跨 2.5 个看门狗周期
    print(f"  等 {wait:.0f}s 跨过 2.5 个看门狗周期…")
    time.sleep(wait)
    p1 = show_props()
    nr0, nr1 = int(p0.get("NRestarts") or 0), int(p1.get("NRestarts") or 0)
    print(f"  NRestarts {nr0} → {nr1}；ActiveState={p1.get('ActiveState')}")
    ok = (p1.get("ActiveState") == "active" and nr1 == nr0)
    print(f"  未被看门狗杀掉: {ok}")
    return ok


def case_metrics() -> bool:
    """B2：journal 里每 10 s 一行 METRICS，JSON 可解析且字段齐全"""
    # 清空游标，只看本轮
    t0 = time.time()
    time.sleep(25)                     # 至少覆盖 2 个 10s 周期
    _, out = sh(f"journalctl -u {UNIT} --since '@{int(t0)}' --no-pager -o cat")
    lines = [l for l in out.splitlines() if "METRICS " in l]
    print(f"  {len(lines)} 行 METRICS")
    if len(lines) < 2:
        print("  [FAIL] 少于 2 行 ⇒ 不是每 10 s 一行")
        return False

    need = {"t", "evt_seen", "push_ok", "push_fail", "push_gated",
            "queue_depth", "queue_dropped", "conn_live", "conn_rejected",
            "ring_count", "sensor_alive"}
    ok = True
    for ln in lines[-3:]:
        body = ln.split("METRICS ", 1)[1]
        try:
            d = json.loads(body)
        except Exception as e:
            print(f"  [FAIL] JSON 解析失败: {e}  body={body[:80]}")
            ok = False
            continue
        miss = need - set(d)
        if miss:
            print(f"  [FAIL] 缺字段 {sorted(miss)}")
            ok = False
        else:
            print(f"  样例 {body}")

    # 间隔检查
    ts = []
    for ln in lines:
        try:
            ts.append(json.loads(ln.split("METRICS ", 1)[1])["t"])
        except Exception:
            pass
    if len(ts) >= 2:
        gaps = [b - a for a, b in zip(ts, ts[1:])]
        print(f"  间隔 {gaps} 秒（期望约 10）")
        ok &= all(8 <= g <= 13 for g in gaps)
    return ok


def case_unit() -> bool:
    """B3：unit 护栏真的加载了（读 systemd 属性，不读文件）"""
    p = show_props()
    print(f"  MemoryMax={p.get('MemoryMax')} LimitNOFILE={p.get('LimitNOFILE')}")
    print(f"  StartLimitBurst={p.get('StartLimitBurst')} "
          f"StartLimitIntervalUSec={p.get('StartLimitIntervalUSec')}")
    # MemoryMax 在 systemd 里以字节返回
    mm = p.get("MemoryMax", "")
    ok_mem = mm.isdigit() and int(mm) > 0
    ok_nofile = (p.get("LimitNOFILE") or "") not in ("", "infinity")
    ok_burst = (p.get("StartLimitBurst") or "") not in ("", "0")
    print(f"  MemoryMax 已设: {ok_mem}  LimitNOFILE 已设: {ok_nofile}  "
          f"StartLimitBurst 已设: {ok_burst}")
    return ok_mem and ok_nofile and ok_burst


CASES = {"notify": case_notify, "metrics": case_metrics, "unit": case_unit}


def main() -> int:
    results = {}
    for name, fn in CASES.items():
        print(f"=== {name} ===")
        try:
            results[name] = fn()
        except Exception as e:
            print(f"  [EXC] {e}")
            results[name] = False
        print()

    print("=== 汇总 ===")
    for k, v in results.items():
        print(f"  {k:8s} {'PASS' if v else 'FAIL'}")
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
