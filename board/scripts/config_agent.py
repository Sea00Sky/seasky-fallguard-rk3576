#!/usr/bin/env python3
"""config_agent.py — 板端配置代理。"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "runtime"
ENV_PATH = RUNTIME / "multi.env"
ENV_BAK = RUNTIME / "multi.env.bak"
STATUS_PATH = RUNTIME / "status.json"
CONFIG_JSON = Path("/etc/fallguard/config.json")

POLL_WAIT = 25
POLL_TIMEOUT = POLL_WAIT + 15
HEARTBEAT_S = 60
APPLY_TIMEOUT_S = 40
SERVICE = "fallguard-sensor-multi"

# 必须与板端 MultiConfig.as_dict 逐键一致，否则 config_hash 对不上
PARAM_KEYS = [
    ("confirm_windows", int, 1, 5),
    ("retract_windows", int, 1, 20),
    ("cooldown_s", float, 0, 300),
    ("max_persons_stgcn", int, 1, 3),
    ("stgcn_queue_max", int, 1, 8),
    ("geo_veto_ar_max", float, 0, 5),
    ("geo_veto_bottom_shift", float, 0, 200),
    ("geo_veto_r_end", float, 0, 1),
    ("geo_veto_min_box", int, 0, 60),
    ("strong_confirm", float, 0, 1),
]


def log(msg: str) -> None:
    print(f"[agent] {time.strftime('%m-%d %H:%M:%S')} {msg}", flush=True)


def load_target() -> tuple[str, str, str]:
    """从 /etc/fallguard/config.json 的 push_url 拆出 base / device_code / key。"""
    cfg = json.loads(CONFIG_JSON.read_text(encoding="utf-8"))
    url = (cfg.get("push_url") or "").strip()
    parts = urllib.parse.urlsplit(url)
    seg = [p for p in parts.path.split("/") if p]
    if len(seg) < 4 or seg[0] != "api" or seg[1] != "v1" or seg[2] != "ingest":
        raise SystemExit(f"[agent] push_url 解析失败：{url}")
    return f"{parts.scheme}://{parts.netloc}", seg[-2], seg[-1]


def http_json(url: str, payload=None, timeout: float = 20):
    """返回 (status, json|None)。HTTPError 也当普通状态码返回。"""
    data = None
    headers = {"Accept": "application/json", "User-Agent": "fallguard-agent/1"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(
        url, data=data, headers=headers,
        method="POST" if data is not None else "GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read()
            return resp.status, (json.loads(body) if body else None)
    except urllib.error.HTTPError as exc:
        return exc.code, None


def read_status() -> dict:
    try:
        return json.loads(STATUS_PATH.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}


def validate(params: dict) -> dict:
    out = {}
    for name, cast, lo, hi in PARAM_KEYS:
        if name not in params:
            raise ValueError(f"缺少参数 {name}")
        v = cast(params[name])
        if not (lo <= v <= hi):
            raise ValueError(f"{name}={v} 超出 [{lo},{hi}]")
        out[name] = v
    return out


def write_env(params: dict, rev: int) -> None:
    lines = ["# 由 config_agent 从云端写入；手改会被下次下发覆盖"]
    for name, _cast, _lo, _hi in PARAM_KEYS:
        lines.append(f"FG_MULTI_{name.upper()}={params[name]}")
    lines += [f"FG_POLICY_REV={rev}", "FG_CONFIG_SOURCE=cloud"]
    tmp = ENV_PATH.with_name(ENV_PATH.name + ".tmp")
    tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if ENV_PATH.exists():
        shutil.copyfile(ENV_PATH, ENV_BAK)
    os.replace(tmp, ENV_PATH)


def systemctl(action: str) -> tuple[int, str]:
    r = subprocess.run(["sudo", "-n", "systemctl", action, SERVICE],
                       capture_output=True, text=True, timeout=60)
    return r.returncode, (r.stdout + r.stderr).strip()


def apply_config(rev: int, params: dict) -> dict:
    write_env(params, rev)
    rc, out = systemctl("restart")
    if rc != 0:
        raise RuntimeError(f"restart 失败 rc={rc}: {out}")
    deadline = time.monotonic() + APPLY_TIMEOUT_S
    while time.monotonic() < deadline:
        st = read_status()
        if st.get("policy_rev") == rev and st.get("ready"):
            return st
        time.sleep(1.0)
    raise TimeoutError(f"{APPLY_TIMEOUT_S}s 内没等到 policy_rev={rev} ready=true")


def hardware_id() -> str:
    for path in ("/proc/device-tree/serial-number", "/etc/machine-id"):
        try:
            v = Path(path).read_text(encoding="utf-8").strip().strip("\x00")
            if v:
                return v[:64]
        except Exception:  # noqa: BLE001
            continue
    return ""


_prev_cpu_sample: tuple[int, int] | None = None


def _read_text(path: str) -> str | None:
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def _cpu_sample() -> tuple[int, int] | None:
    txt = _read_text("/proc/stat")
    if not txt:
        return None
    parts = txt.splitlines()[0].split()
    try:
        vals = [int(x) for x in parts[1:9]]  # user…steal（guest 已含在 user 里）
    except (IndexError, ValueError):
        return None
    if len(vals) < 4:
        return None
    idle = vals[3] + (vals[4] if len(vals) > 4 else 0)
    return sum(vals), idle


def _cpu_pct() -> float | None:
    global _prev_cpu_sample
    cur = _cpu_sample()
    old, _prev_cpu_sample = _prev_cpu_sample, cur
    if not cur or not old:
        return None
    total_d = cur[0] - old[0]
    idle_d = cur[1] - old[1]
    if total_d <= 0:
        return None
    return round(max(0.0, min(100.0, (total_d - idle_d) * 100.0 / total_d)), 1)


def _mem_info() -> tuple[int | None, int | None, float | None]:
    """/proc/meminfo → (已用 MB, 总 MB, 使用率 %)。已用 = MemTotal - MemAvailable。"""
    txt = _read_text("/proc/meminfo")
    if not txt:
        return None, None, None
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
        return None, None, None
    used_kb = total_kb - avail_kb
    return (round(used_kb / 1024), round(total_kb / 1024),
            round(used_kb * 100.0 / total_kb, 1))


def _temp_c() -> float | None:
    """所有 thermal zone 取最高温（RK3576 有 soc/bigcore/littlecore/npu 等分区）。"""
    vals: list[float] = []
    for path in Path("/sys/class/thermal").glob("thermal_zone*/temp"):
        txt = _read_text(str(path))
        try:
            t = int((txt or "").strip())
        except ValueError:
            continue
        if t > 0:
            vals.append(t / 1000.0)
    return round(max(vals), 1) if vals else None


def _npu_load() -> float | None:
    """debugfs 的 NPU 占用（如 `NPU load: Core0: 12%`）；读不到就 None。"""
    txt = _read_text("/sys/kernel/debug/rknpu/load")
    if not txt:
        return None
    pcts = [int(x) for x in re.findall(r"(\d+)%", txt)]
    return float(max(pcts)) if pcts else None


def heartbeat_payload() -> dict:
    st = read_status()
    started = st.get("started_monotonic")
    uptime = int(time.monotonic() - started) if isinstance(started, (int, float)) else None
    fresh = False
    ts = st.get("ts")
    if isinstance(ts, (int, float)):
        fresh = (time.time() - ts) < 15
    used_mb, total_mb, mem_pct = _mem_info()
    return {
        "fps": st.get("fps"),
        "frame_age_s": st.get("frame_age_s"),
        "policy_rev": st.get("policy_rev"),
        "config_hash": st.get("config_hash"),
        "service_state": "active" if fresh and st.get("ready") else "unknown",
        "model_ver": st.get("model_ver"),
        "uptime_s": uptime,
        "hardware_id": hardware_id(),
        # 板端遥测（四改）：老云端忽略多余键也不会坏 —— 接口是 JSONB 原样存
        "cpu_pct": _cpu_pct(),
        "mem_used_mb": used_mb,
        "mem_total_mb": total_mb,
        "mem_pct": mem_pct,
        "temp_c": _temp_c(),
        "npu_load": _npu_load(),
    }


def main() -> int:
    base, code, key = load_target()
    cfg_url = f"{base}/api/v1/ingest/{code}/{key}/config"
    ack_url = cfg_url + "/ack"
    hb_url = f"{base}/api/v1/ingest/{code}/{key}/heartbeat"

    applied = int(read_status().get("policy_rev") or 0)
    _cpu_pct()
    last_hb = 0.0
    backoff = 1.0
    log(f"启动 device={code} applied_rev={applied} 云端={base}")

    while True:
        try:
            status, data = http_json(
                f"{cfg_url}?applied_rev={applied}&wait={POLL_WAIT}",
                timeout=POLL_TIMEOUT)
            if status == 200 and data:
                rev = int(data["rev"])
                params = validate(data.get("params") or {})
                log(f"收到配置 rev={rev}，开始应用")
                try:
                    st = apply_config(rev, params)
                    applied = rev
                    http_json(ack_url, {
                        "rev": rev, "status": "APPLIED", "reported": st,
                        "config_hash": st.get("config_hash"),
                    })
                    log(f"rev={rev} 已生效（fps={st.get('fps')}）")
                except Exception as exc:  # noqa: BLE001
                    log(f"应用失败：{exc}；恢复 multi.env.bak")
                    if ENV_BAK.exists():
                        shutil.copyfile(ENV_BAK, ENV_PATH)
                        systemctl("restart")
                    http_json(ack_url, {
                        "rev": rev, "status": "FAILED", "error": str(exc),
                    })
            backoff = 1.0
        except Exception as exc:  # noqa: BLE001
            log(f"轮询失败：{type(exc).__name__}: {exc}；{backoff:.0f}s 后重试")
            time.sleep(backoff)
            backoff = min(backoff * 2, 60)
            continue

        cur = read_status().get("policy_rev")
        if isinstance(cur, int) and cur > applied:
            applied = cur

        now = time.monotonic()
        if now - last_hb >= HEARTBEAT_S:
            last_hb = now
            try:
                http_json(hb_url, heartbeat_payload(), timeout=15)
            except Exception as exc:  # noqa: BLE001
                log(f"心跳失败：{type(exc).__name__}: {exc}")


if __name__ == "__main__":
    sys.exit(main())
