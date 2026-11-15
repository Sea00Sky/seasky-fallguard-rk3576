#!/usr/bin/env python3
"""板端快照上传器 —— 把网关存下来的告警快照推到云端。"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

CONFIG_PATH = "/etc/fallguard/config.json"
EVENTS_DB = "/var/lib/fallguard/events.db"
DATA_DIR = "/var/lib/fallguard"
# 状态文件不能放 DATA_DIR：那目录是 root 建的（网关以 root 跑），fallguard 只读。
# 放用户自己的 XDG 状态目录，无论以 fallguard 还是 root 跑都写得进去。
STATE_DIR = os.path.join(os.path.expanduser("~"), ".local", "state", "fallguard")
STATE_NAME = "snap_upload.json"

# 云端窗口 120s；留 20s 余量，超过就别传了（会被 400 拒掉，白跑）
MAX_AGE_S = 100
HTTP_TIMEOUT = 15

# 404 = 事件还没上云，属于正常竞争；记个次数，别无限重试
MAX_MISS_RETRY = 20


def log(msg: str) -> None:
    print(f"[snap] {msg}", flush=True)


def jpeg_size(data: bytes):
    """从 JPEG 的 SOF 段读真实宽高，读不到返回 None。"""
    if len(data) < 4 or data[0] != 0xFF or data[1] != 0xD8:
        return None
    i, n = 2, len(data)
    while i + 9 < n:
        if data[i] != 0xFF:
            i += 1
            continue
        marker = data[i + 1]
        if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
            i += 2
            continue
        if marker == 0xD9:
            break
        seg_len = (data[i + 2] << 8) | data[i + 3]
        if seg_len < 2:
            break
        if marker in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7,
                      0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
            h = (data[i + 5] << 8) | data[i + 6]
            w = (data[i + 7] << 8) | data[i + 8]
            return w, h
        i += 2 + seg_len
    return None


def parse_push_url(url: str) -> tuple[str, str, str]:
    """``http://host/api/v1/ingest/CODE/KEY`` → (base, code, key)。"""
    parts = urllib.parse.urlsplit(url)
    seg = [s for s in parts.path.strip("/").split("/") if s]
    if len(seg) < 5 or seg[0] != "api" or seg[2] != "ingest":
        raise ValueError(f"push_url 不符合 /api/v1/ingest/CODE/KEY 形式: {url[:40]}…")
    return f"{parts.scheme}://{parts.netloc}", seg[-2], seg[-1]


def load_target() -> tuple[str, str, str]:
    with open(CONFIG_PATH, encoding="utf-8") as fh:
        cfg = json.load(fh)
    if not cfg.get("push_enabled", True):
        raise SystemExit("push_enabled = false，板端未开启推送，上传器退出")
    url = (cfg.get("push_url") or "").strip()
    if not url:
        raise SystemExit("push_url 为空，无法推导云端地址")
    return parse_push_url(url)


def load_state(path: str) -> dict:
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, dict) and isinstance(data.get("done"), dict):
            return data
    except (OSError, ValueError):
        pass
    return {"done": {}, "miss": {}}


def save_state(path: str, state: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(state, fh)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def scan(db_path: str) -> list:
    """只读打开网关 SQLite；网关在写，用 ro + timeout 避免锁冲突。"""
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=5.0)
    try:
        con.row_factory = sqlite3.Row
        return con.execute(
            "SELECT seq, snapshot_seq, snap_file, ts_sec, pushed "
            "  FROM events "
            " WHERE snap_file IS NOT NULL AND snap_file <> '' "
            "   AND pushed = 1 "
            " ORDER BY seq"
        ).fetchall()
    finally:
        con.close()


def put_snapshot(base: str, code: str, key: str, board_seq: int,
                 snapshot_seq, data: bytes, digest: str,
                 width: int = 320, height: int = 240) -> tuple[int, str]:
    query = urllib.parse.urlencode({
        "board_seq": board_seq,
        "snapshot_seq": snapshot_seq if snapshot_seq is not None else 0,
        "sha256": digest,
        "width": width,
        "height": height,
    })
    url = f"{base}/api/v1/ingest/{code}/{key}/snapshot?{query}"
    req = urllib.request.Request(
        url, data=data, method="PUT",
        headers={"Content-Type": "image/jpeg", "X-Content-SHA256": digest},
    )
    try:
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as resp:
            return resp.status, resp.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", "replace")
    except (urllib.error.URLError, OSError) as exc:
        return 0, f"{type(exc).__name__}: {exc}"


def run_once(verbose: bool = False) -> int:
    base, code, key = load_target()
    state_path = os.path.join(STATE_DIR, STATE_NAME)
    state = load_state(state_path)

    try:
        rows = scan(EVENTS_DB)
    except sqlite3.Error as exc:
        log(f"读事件库失败：{exc}")
        return 1

    now = int(time.time())
    sent = skipped = waiting = 0

    for row in rows:
        seq = int(row["seq"])
        snap_seq = row["snapshot_seq"]
        stamp = f"{seq}:{snap_seq}"
        if state["done"].get(str(seq)) == str(snap_seq):
            continue

        age = now - int(row["ts_sec"] or 0)
        if age > MAX_AGE_S:
            # 超过云端时间窗，传了也会被 400 拒；记成已完成，不再反复试
            if verbose:
                log(f"seq={seq} 快照已过期（{age}s > {MAX_AGE_S}s），跳过")
            state["done"][str(seq)] = str(snap_seq)
            skipped += 1
            continue

        rel = row["snap_file"]
        path = rel if os.path.isabs(rel) else os.path.join(DATA_DIR, rel)
        try:
            with open(path, "rb") as fh:
                data = fh.read()
        except OSError as exc:
            if verbose:
                log(f"seq={seq} 读图失败 {path}: {exc}")
            continue

        digest = hashlib.sha256(data).hexdigest()
        wh = jpeg_size(data) or (320, 240)
        status, body = put_snapshot(base, code, key, seq, snap_seq, data, digest,
                                    width=wh[0], height=wh[1])

        if status == 200:
            state["done"][str(seq)] = str(snap_seq)
            state["miss"].pop(str(seq), None)
            sent += 1
            if verbose:
                log(f"seq={seq} snapshot_seq={snap_seq} 上传成功 "
                    f"{wh[0]}×{wh[1]} {len(data)}B")
        elif status == 404:
            n = int(state["miss"].get(str(seq), 0)) + 1
            state["miss"][str(seq)] = n
            waiting += 1
            if n >= MAX_MISS_RETRY:
                state["done"][str(seq)] = str(snap_seq)
                log(f"seq={seq} 重试 {n} 次仍找不到事件，放弃")
            elif verbose:
                log(f"seq={seq} 事件尚未上云（404），第 {n} 次重试")
        elif status == 400 and "SNAPSHOT_EXPIRED" in body:
            state["done"][str(seq)] = str(snap_seq)
            if verbose:
                log(f"seq={seq} 云端判定过期，跳过")
        else:
            log(f"seq={seq} 上传失败 HTTP {status}: {body[:160]}")

    save_state(state_path, state)
    if sent or skipped or verbose:
        log(f"本轮 成功={sent} 过期跳过={skipped} 等待={waiting}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="板端快照上云")
    ap.add_argument("--once", action="store_true", help="只跑一轮（排障用）")
    ap.add_argument("--interval", type=float, default=2.0, help="常驻模式轮询间隔（秒）")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    if args.once:
        return run_once(args.verbose)

    log(f"常驻启动，间隔 {args.interval}s")
    while True:
        try:
            run_once(args.verbose)
        except SystemExit:
            raise
        except Exception as exc:  # 守护进程不能因为一次异常就死
            log(f"本轮异常（已忽略）：{type(exc).__name__}: {exc}")
        time.sleep(args.interval)


if __name__ == "__main__":
    sys.exit(main())
