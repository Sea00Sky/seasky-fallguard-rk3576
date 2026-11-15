# -*- coding: utf-8 -*-
"""核心服务与整机资源采样（信息科首页「系统健康」面板， 四改）。"""
from __future__ import annotations

import os
import subprocess
import threading
import time

CGROUP_ROOT = "/sys/fs/cgroup"


def _ncpu() -> int:
    try:
        return len(os.sched_getaffinity(0)) or 1
    except (AttributeError, OSError):
        return os.cpu_count() or 1


NCPU = _ncpu()
UNITS = [
    ("fg-web.service", "fg-web"),
    ("postgresql@16-main.service", "PostgreSQL"),
    ("nginx.service", "nginx"),
    ("mediamtx.service", "mediamtx"),
]
SAMPLE_S = 5.0
FIRST_SAMPLE_S = 1.0
CGROUP_REFRESH_S = 300.0

_lock = threading.Lock()
_state: dict = {
    "system": {"cpu_text": "—", "mem_text": "—", "load_text": "—"},
    "services": [],
}
_started = False


def _read(path: str | None) -> str | None:
    if not path:
        return None
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            return fh.read()
    except OSError:
        return None


def _cgroup_path(unit: str) -> str | None:
    """systemd 单元 → cgroup 目录。解析失败返回 None（服务没装/没跑）。"""
    try:
        out = subprocess.run(
            ["systemctl", "show", unit, "-p", "ControlGroup", "--value"],
            capture_output=True, text=True, timeout=3, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    cg = (out.stdout or "").strip()
    if not cg:
        return None
    return CGROUP_ROOT + cg


def _usage_usec(base: str | None) -> int | None:
    txt = _read(base + "/cpu.stat" if base else None)
    for line in (txt or "").splitlines():
        if line.startswith("usage_usec "):
            try:
                return int(line.split()[1])
            except (IndexError, ValueError):
                return None
    return None


def _mem_bytes(base: str | None) -> int | None:
    """anon + shmem（≈ RSS）。anonym 是进程私有内存，shmem 里装着 PostgreSQL 的
    shared_buffers；file/page cache 不算 —— 可回收，且会与「整机内存」重复。"""
    txt = _read(base + "/memory.stat" if base else None)
    if txt:
        vals: dict[str, int] = {}
        for line in txt.splitlines():
            key, _, rest = line.partition(" ")
            if key in ("anon", "shmem"):
                try:
                    vals[key] = int(rest.strip())
                except ValueError:
                    continue
        if "anon" in vals:
            return vals["anon"] + vals.get("shmem", 0)
    # 老内核没有 memory.stat 对应键时的兜底（宁可显示偏大，也别显示 —）
    txt = _read(base + "/memory.current" if base else None)
    if txt:
        try:
            return int(txt.strip())
        except ValueError:
            return None
    return None


def _proc_count(base: str | None) -> int | None:
    txt = _read(base + "/cgroup.procs" if base else None)
    if txt is None:
        return None
    return len([x for x in txt.splitlines() if x.strip()])


def _proc_cpu() -> tuple[int, int] | None:
    txt = _read("/proc/stat")
    if not txt:
        return None
    parts = txt.splitlines()[0].split()
    try:
        vals = [int(x) for x in parts[1:9]]  # user…steal；guest 已含在 user 里，不重复加
    except ValueError:
        return None
    if len(vals) < 4:
        return None
    idle = vals[3] + (vals[4] if len(vals) > 4 else 0)
    return sum(vals), idle


def _meminfo() -> tuple[int | None, int | None]:
    """/proc/meminfo → (已用 MB, 总 MB)。已用 = MemTotal - MemAvailable。"""
    txt = _read("/proc/meminfo")
    if not txt:
        return None, None
    info: dict[str, int] = {}
    for line in txt.splitlines():
        key, _, rest = line.partition(":")
        try:
            info[key] = int(rest.split()[0])  # kB
        except (IndexError, ValueError):
            continue
    total_kb = info.get("MemTotal")
    avail_kb = info.get("MemAvailable") or info.get("MemFree")
    if not total_kb or avail_kb is None:
        return None, None
    return round((total_kb - avail_kb) / 1024), round(total_kb / 1024)


def _loadavg() -> str:
    txt = _read("/proc/loadavg")
    if not txt:
        return "—"
    parts = txt.split()
    if len(parts) < 3:
        return "—"
    return " / ".join(parts[:3])


def _cpu_text(pct: float | None) -> str:
    return "—" if pct is None else f"{pct:.1f}%"


def _mem_text(mb: int | None) -> str:
    if mb is None:
        return "—"
    if mb >= 1024:
        return f"{mb / 1024:.1f} GB"
    return f"{mb} MB"


def _sys_mem_text(used: int | None, total: int | None) -> str:
    if used is None or total is None:
        return "—"
    if total >= 1024:
        return f"{used / 1024:.1f} / {total / 1024:.1f} GB"
    return f"{used} / {total} MB"


def _loop() -> None:
    prev: dict[str, tuple[int, float]] = {}
    prev_sys = _proc_cpu()
    cgroups: dict[str, str | None] = {}
    cgroups_at = 0.0
    delay = FIRST_SAMPLE_S
    while True:
        time.sleep(delay)
        delay = SAMPLE_S
        if (not cgroups) or (time.monotonic() - cgroups_at > CGROUP_REFRESH_S):
            cgroups = {unit: _cgroup_path(unit) for unit, _ in UNITS}
            cgroups_at = time.monotonic()
        # 时间戳必须在解析 cgroup 之后取：systemctl 最坏要几秒，
        # 用解析前的时间算 dt 会把 CPU% 算大（差多少倍就虚高多少倍）。
        now = time.monotonic()

        services = []
        for unit, label in UNITS:
            base = cgroups.get(unit)
            usec = _usage_usec(base)
            mem = _mem_bytes(base)
            procs = _proc_count(base)
            cpu = None
            if usec is None:
                prev.pop(unit, None)
            else:
                old = prev.get(unit)
                if old is not None:
                    dt = now - old[1]
                    if dt > 0:
                        # /NCPU：折算成「占全机」，与上面 /proc/stat 的整机 CPU 同口径
                        cpu = (usec - old[0]) / (dt * 1e6) * 100.0 / NCPU
                prev[unit] = (usec, now)
            ok = bool(procs)
            services.append({
                "label": label,
                "cpu_text": _cpu_text(cpu) if ok else "—",
                "mem_text": _mem_text(round(mem / 1048576) if mem is not None else None) if ok else "—",
                "procs": procs,
                "ok": ok,
                "title": f"{unit} · " + (f"{procs} 个进程" if ok else "未运行"),
            })

        cur = _proc_cpu()
        sys_cpu = None
        if cur and prev_sys:
            total_d = cur[0] - prev_sys[0]
            idle_d = cur[1] - prev_sys[1]
            if total_d > 0:
                sys_cpu = (total_d - idle_d) * 100.0 / total_d
        if cur:
            prev_sys = cur
        used_mb, total_mb = _meminfo()

        with _lock:
            _state["system"] = {
                "cpu_text": _cpu_text(sys_cpu),
                "mem_text": _sys_mem_text(used_mb, total_mb),
                "load_text": _loadavg(),
            }
            _state["services"] = services


def start() -> None:
    """启动后台采样（幂等）。应用启动时调用一次；没人访问也不会占多少 CPU。"""
    global _started
    if _started:
        return
    with _lock:
        if _started:
            return
        _started = True
    threading.Thread(target=_loop, name="fg-sysmon", daemon=True).start()


def snapshot() -> dict:
    """当前快照（深拷贝一层，模板随便读）。"""
    start()
    with _lock:
        return {
            "system": dict(_state["system"]),
            "services": [dict(s) for s in _state["services"]],
        }
