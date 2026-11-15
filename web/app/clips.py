"""告警视频片段—— 云端录制切片。"""

from __future__ import annotations

import asyncio
import datetime as dt
import hashlib
import os
import re
import subprocess
from pathlib import Path
from typing import Any

from app.config import settings
from app.db import execute, query

# 段文件名模板 %Y-%m-%d_%H-%M-%S-%f（mediamtx 的 recordPath）
_SEG_RE = re.compile(r"^(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}-\d{6})\.(ts|mp4)$")
# mediamtx 的 recordDeleteAfter 是 10 分钟；超过这个岁数的 PENDING 再切也切不到
_RECORD_KEEP_S = 600
# 一条证据最多试几次。不要设成 3 —— 3 × 5 s = 15 s，而录制在盘上留 10 分钟，
# mediamtx 重启一次就可能持续几十秒，三次全落在故障窗口里就把本可成功的片段判死了。
# 24 × 5 s ≈ 2 分钟，覆盖瞬时故障；真正等不到画面的由 recording_expired 判死。
_MAX_ATTEMPTS = 24
# 扫描周期（秒）
_TICK_S = 5.0
# 一轮最多处理几条，避免积压时长时间占住连接
_BATCH = 5


# ════════════════════════════════════════════════════════════════════════════
# 入队：ingest 确认落库时调用
# ════════════════════════════════════════════════════════════════════════════
def create_evidence(conn: Any, event_uid: Any, received_at: dt.datetime) -> bool:
    """给一个已确认的事件排一条片段任务。"""
    if not settings.clip_enabled:
        return False
    # lag 是 加不是减。码流从板端到落盘有 0.2 s SRT 缓冲 + 编码/封装延迟，
    # 所以「板端墙钟 T 发生的事」在录制的时间轴上出现在 T + lag。窗口要整体往后挪，
    # 不是往前。写成 `-(lead + lag)` 的话窗口提前了 2×lag，跌倒时刻会被推到片段更靠后，
    # 后段变短；clip_lag_s 调大时甚至会整段落在事件之前，而状态照样是 READY。
    start = (
        received_at
        - dt.timedelta(seconds=settings.clip_lead_s)
        + dt.timedelta(seconds=settings.clip_lag_s)
    )
    cur = conn.execute(
        """
        INSERT INTO fg.event_evidence (event_uid, kind, status, clip_start_at)
        VALUES (%s, 'clip', 'PENDING', %s)
        ON CONFLICT (event_uid, kind) DO NOTHING
        """,
        (event_uid, start),
    )
    return cur.rowcount > 0


# ════════════════════════════════════════════════════════════════════════════
# 切片
# ════════════════════════════════════════════════════════════════════════════
def _local_tz() -> dt.tzinfo:
    try:
        from zoneinfo import ZoneInfo

        return ZoneInfo(settings.tz)
    except Exception:  # noqa: BLE001 —— 时区库缺失时退回本机时区，别把 worker 打死
        return dt.datetime.now().astimezone().tzinfo or dt.timezone.utc


def _list_segments(record_dir: Path) -> list[tuple[dt.datetime, Path]]:
    """返回 [(段起点, 路径)]，按起点升序。认不出的文件名直接跳过。"""
    tz = _local_tz()
    out: list[tuple[dt.datetime, Path]] = []
    try:
        entries = list(record_dir.iterdir())
    except OSError:
        return out
    for entry in entries:
        match = _SEG_RE.match(entry.name)
        if not match:
            continue
        try:
            start = dt.datetime.strptime(match.group(1), "%Y-%m-%d_%H-%M-%S-%f")
        except ValueError:
            continue
        out.append((start.replace(tzinfo=tz), entry))
    out.sort(key=lambda item: item[0])
    return out


def _segment_length(segments: list[tuple[dt.datetime, Path]]) -> dt.timedelta:
    """从相邻段起点的间隔反推段长，而不是写死 10 s。"""
    if len(segments) < 2:
        return dt.timedelta(seconds=10)
    gaps = sorted(
        (segments[i + 1][0] - segments[i][0]).total_seconds()
        for i in range(len(segments) - 1)
    )
    # 断流会在列表里留下很大的空档，中位数不受影响
    return dt.timedelta(seconds=gaps[len(gaps) // 2])


def _pick_segments(
    segments: list[tuple[dt.datetime, Path]], start: dt.datetime, end: dt.datetime
) -> list[tuple[dt.datetime, Path]]:
    """挑出与 [start, end] 有交集的段。"""
    seg_len = _segment_length(segments)
    return [
        item for item in segments
        if item[0] + seg_len > start and item[0] < end
    ]


def _seek_offset(
    picked: list[tuple[dt.datetime, Path]],
    clip_start_at: dt.datetime,
    seg_len: dt.timedelta,
) -> float:
    """算出 `clip_start_at` 在 concat 之后的时间轴上对应的秒数。"""
    offset = 0.0
    for seg_start, seg_path in picked:
        dur = _probe_duration(seg_path)
        span_s = dur if dur is not None else seg_len.total_seconds()
        if clip_start_at < seg_start + dt.timedelta(seconds=span_s):
            return offset + max(0.0, (clip_start_at - seg_start).total_seconds())
        offset += span_s
    # clip_start_at 比所有段都早：从头开始（前段必然不完整，属于录制没覆盖到）
    return 0.0


def _probe_duration(path: Path) -> float | None:
    rc, out = _run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        timeout=15.0,
    )
    if rc != 0:
        return None
    try:
        value = float(out.strip().splitlines()[0])
    except (ValueError, IndexError):
        return None
    return value if value > 0 else None


def _run(cmd: list[str], timeout: float = 60.0) -> tuple[int, str]:
    """跑子进程，返回 (rc, 合并输出)。超时按失败处理，不抛异常。"""
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, check=False
        )
        return proc.returncode, (proc.stdout or "") + (proc.stderr or "")
    except (OSError, subprocess.TimeoutExpired) as exc:
        return -1, f"{type(exc).__name__}: {exc}"


def _probe(path: Path) -> dict | None:
    """读成品片段的宽高与时长。失败返回 None。"""
    rc, out = _run(
        [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1",
            str(path),
        ],
        timeout=20.0,
    )
    if rc != 0:
        return None
    info: dict[str, Any] = {}
    for line in out.splitlines():
        key, _, value = line.partition("=")
        info[key.strip()] = value.strip()
    try:
        return {
            "width": int(info["width"]),
            "height": int(info["height"]),
            "duration_ms": int(float(info["duration"]) * 1000),
        }
    except (KeyError, ValueError):
        return None


def cut_clip(
    clip_start_at: dt.datetime, stem: str
) -> tuple[Path | None, str | None, dict | None]:
    """从环形录制里切一段，落成 `<clip_dir>/<stem>.mp4`。"""
    lead = settings.clip_lead_s
    tail = settings.clip_tail_s
    end = clip_start_at + dt.timedelta(seconds=lead + tail)

    segments = _list_segments(Path(settings.clip_record_dir))
    if not segments:
        return None, "no_recording", None

    picked = _pick_segments(segments, clip_start_at, end)
    if not picked:
        oldest, newest = segments[0][0], segments[-1][0]
        return (
            None,
            f"no_segment_in_window(oldest={oldest:%H:%M:%S},newest={newest:%H:%M:%S})",
            None,
        )

    clip_dir = Path(settings.clip_dir)
    clip_dir.mkdir(parents=True, exist_ok=True)
    list_path = clip_dir / f".concat_{stem}.txt"
    tmp_path = clip_dir / f".tmp_{stem}.mp4"
    full_path = clip_dir / f".full_{stem}.mp4"
    # 从这一行起，任何 return 都要经过下面的 finally 清理（含 list_path 写失败）

    seek = _seek_offset(picked, clip_start_at, _segment_length(segments))
    duration = (end - clip_start_at).total_seconds()

    # 必须两段式，不能一步到位。 实测（4 段 11 s 的 TS）：
    # `-ss 3` 放在 `-f concat -i` 之前 + `-t 10` ⇒ 12.03 s（多出 2 s）
    # 同样参数放在输出侧 ⇒ 9.03 s（少 1 s）
    # 先 concat 成整段、再在单文件上 `-ss 3 -t 10` ⇒ 10.03 s（准）
    # concat demuxer 的输入侧 seek 与 `-t` 的计时基准对不上，是 ffmpeg 的已知行为。
    try:
        # ⓪ 写 concat 列表。放在 try 里面 —— 否则磁盘写满时 `.concat_*.txt`
        # 会绕过 finally 永远留在目录里（cleanup 只认 .mp4，看不见它）。
        # ffmpeg concat demuxer 的路径里单引号要转义（文件名一般没有，防一手）
        try:
            with open(list_path, "w", encoding="utf-8") as handle:
                for _start, path in picked:
                    safe = str(path).replace("'", "'\\''")
                    handle.write(f"file '{safe}'\n")
        except OSError as exc:
            return None, f"concat_list_write_failed: {exc}", None

        # ① concat：不重编码，只是把 TS 段拼成一个连续时间轴的文件
        rc, out = _run(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-f", "concat", "-safe", "0", "-i", str(list_path),
                "-c", "copy", "-avoid_negative_ts", "make_zero",
                "-f", "mp4", str(full_path),
            ],
            timeout=90.0,
        )
        if rc != 0 or not full_path.exists() or full_path.stat().st_size == 0:
            return None, f"concat_failed(rc={rc}): {out.strip()[:300]}", None

        # ② 在单文件上精确切：`-ss` 落 I 帧（板端 gop=fps ⇒ 1 s 一个），误差 ≤1 s
        rc, out = _run(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-ss", f"{seek:.3f}", "-i", str(full_path),
                "-t", f"{duration:.3f}",
                "-c", "copy",                      # 不重编码：保画质、零 CPU
                "-avoid_negative_ts", "make_zero",
                "-movflags", "+faststart",         # 浏览器拖动进度条要 moov 在前
                "-f", "mp4", str(tmp_path),
            ],
            timeout=90.0,
        )
        if rc != 0 or not tmp_path.exists() or tmp_path.stat().st_size == 0:
            return None, f"ffmpeg_failed(rc={rc}): {out.strip()[:300]}", None

        info = _probe(tmp_path)
        if info is None:
            return None, "probe_failed", None
        # 短于 3 s 基本等于没切到东西（比如整段都在关键帧之外），判失败
        if info["duration_ms"] < 3000:
            return None, f"clip_too_short({info['duration_ms']}ms)", None

        final_path = clip_dir / f"{stem}.mp4"
        os.replace(tmp_path, final_path)
        return final_path, None, info
    finally:
        for leftover in (list_path, tmp_path, full_path):
            try:
                leftover.unlink()
            except OSError:
                pass


# ════════════════════════════════════════════════════════════════════════════
# worker 主循环
# ════════════════════════════════════════════════════════════════════════════
def _due_seconds() -> float:
    """一条 PENDING 行要等多久才算「到期可切」。"""
    return settings.clip_lead_s + settings.clip_tail_s + settings.clip_lag_s + settings.clip_settle_s


def _mark_failed(evidence_id: Any, attempts: int, reason: str, *, final: bool) -> None:
    """回写一次失败。`final=False` 表示留在 PENDING 等下一轮。"""
    execute(
        """
        UPDATE fg.event_evidence
           SET status = %s, attempts = %s, fail_reason = %s
         WHERE evidence_id = %s
        """,
        ("FAILED" if final else "PENDING", attempts, (reason or "")[:400], evidence_id),
    )


def _process_one(row: dict) -> None:
    """切一条并回写结果。"""
    evidence_id = row["evidence_id"]
    start_at: dt.datetime = row["clip_start_at"]
    attempts = int(row["attempts"]) + 1
    stem = str(row["event_uid"])

    age = (dt.datetime.now(_local_tz()) - start_at).total_seconds()
    if age > _RECORD_KEEP_S:
        _mark_failed(evidence_id, attempts, "recording_expired", final=True)
        return

    try:
        path, reason, info = cut_clip(start_at, stem)
    except Exception as exc:  # noqa: BLE001 —— 环境异常（只读挂载、ENOSPC、EIO）
        reason = f"{type(exc).__name__}: {exc}"
        path, info = None, None
        print(f"[clip] {evidence_id} 切片段抛异常: {reason}", flush=True)

    if path is None:
        # 录制环还在（age < _RECORD_KEEP_S）就继续等：mediamtx 重启、磁盘抖动都是
        # 十几秒级的瞬时故障，重试能救回来。次数上限只是防止无脑空转。
        final = attempts >= _MAX_ATTEMPTS
        _mark_failed(evidence_id, attempts, reason or "unknown", final=final)
        print(
            f"[clip] {evidence_id} 失败（第 {attempts}/{_MAX_ATTEMPTS} 次"
            f"{'，判死' if final else '，下轮重试'}）: {reason}",
            flush=True,
        )
        return

    try:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:  # 文件刚写就被别的东西动过 / 磁盘错误
        _mark_failed(evidence_id, attempts, f"readback_failed: {exc}",
                     final=attempts >= _MAX_ATTEMPTS)
        return

    execute(
        """
        UPDATE fg.event_evidence
           SET status = 'READY', attempts = %s, fail_reason = NULL,
               storage_path = %s, mime = 'video/mp4',
               width = %s, height = %s, duration_ms = %s,
               bytes = %s, sha256 = %s, ready_at = now()
         WHERE evidence_id = %s
        """,
        (
            attempts,
            str(path),
            info["width"],
            info["height"],
            info["duration_ms"],
            path.stat().st_size,
            digest,
            evidence_id,
        ),
    )
    print(
        f"[clip] {evidence_id} READY {info['width']}x{info['height']} "
        f"{info['duration_ms'] / 1000:.1f}s {path.stat().st_size} B",
        flush=True,
    )
    # 通知已打开的详情页：占位换 <video>。延迟导入是为了避开
    # clips ← events 的模块环（events.py 在模块层 import 本模块）。
    try:
        from app.routes.events import publish_event

        publish_event(
            row.get("ward_id") or "",
            "event.updated",
            {"event_uid": str(row["event_uid"]), "has_clip": True},
            zone_id=row.get("zone_id"),
        )
    except Exception as exc:  # noqa: BLE001 —— 广播失败不影响片段本身
        print(f"[clip] 广播失败: {type(exc).__name__}: {exc}", flush=True)


def sweep_once() -> int:
    """扫一轮到期任务，返回处理条数。"""
    if not settings.clip_enabled:
        return 0
    rows = query(
        """
        SELECT ev.evidence_id, ev.event_uid, ev.clip_start_at, ev.attempts,
               e.ward_id, e.zone_id
          FROM fg.event_evidence ev
          JOIN fg.event e ON e.event_uid = ev.event_uid
         WHERE ev.status = 'PENDING'
           AND ev.clip_start_at <= now() - make_interval(secs => %s)
         ORDER BY ev.clip_start_at
         LIMIT %s
        """,
        (_due_seconds(), _BATCH),
    )
    for row in rows:
        try:
            _process_one(row)
        except Exception as exc:  # noqa: BLE001 —— 单条炸了不能拖垮整轮
            print(f"[clip] {row.get('evidence_id')} 异常: {type(exc).__name__}: {exc}", flush=True)
    return len(rows)


def cleanup_once() -> int:
    """删过期成品 + 回收崩溃残留的临时文件。录制环形由 mediamtx 自己管。"""
    days = settings.clip_retention_days
    if days <= 0:
        return 0
    now = dt.datetime.now(_local_tz())
    cutoff = now - dt.timedelta(days=days)
    stale = now - dt.timedelta(hours=1)
    clip_dir = Path(settings.clip_dir)
    removed = 0
    try:
        entries = list(clip_dir.iterdir())
    except OSError:
        return 0
    for entry in entries:
        try:
            mtime = dt.datetime.fromtimestamp(entry.stat().st_mtime, _local_tz())
        except OSError:
            continue
        if entry.name.startswith("."):
            # 临时文件：只要够旧就删，不看后缀（.txt / .mp4 都有）
            try:
                if mtime < stale:
                    entry.unlink()
                    removed += 1
            except OSError:
                continue
            continue
        if entry.suffix != ".mp4":
            continue
        try:
            if mtime < cutoff:
                entry.unlink()
                removed += 1
                # 文件删了，READY 行还指着它就是一条坏链接：详情页会渲染一个
                # 永远 404 的 <video>。必须同步改状态 —— 用 FAILED + 专用 reason，
                # 页面据此显示「已过保留期被清理」而不是「生成失败」。
                try:
                    execute(
                        """
                        UPDATE fg.event_evidence
                           SET status = 'FAILED', fail_reason = 'retention_expired'
                         WHERE storage_path = %s AND status = 'READY'
                        """,
                        (str(entry),),
                    )
                except Exception as exc:  # noqa: BLE001 —— 库抖一下不该中断清理
                    print(f"[clip] 回写 retention_expired 失败 {entry.name}: {exc}", flush=True)
        except OSError:
            continue
    return removed


async def worker_loop() -> None:
    """常驻任务：由 main.lifespan 起停。任何异常都不允许退出循环。"""
    print(
        f"[clip] worker 启动 record={settings.clip_record_dir} out={settings.clip_dir} "
        f"窗口={settings.clip_lead_s:.0f}s+{settings.clip_tail_s:.0f}s",
        flush=True,
    )
    last_cleanup = float("-inf")
    while True:
        try:
            done = await asyncio.to_thread(sweep_once)
            now = asyncio.get_running_loop().time()
            if now - last_cleanup > 3600:
                removed = await asyncio.to_thread(cleanup_once)
                last_cleanup = now
                if removed:
                    print(f"[clip] 清理过期片段 {removed} 个", flush=True)
            await asyncio.sleep(_TICK_S if done else max(_TICK_S, 5.0))
        except asyncio.CancelledError:
            print("[clip] worker 退出", flush=True)
            raise
        except Exception as exc:  # noqa: BLE001
            print(f"[clip] 轮询异常: {type(exc).__name__}: {exc}", flush=True)
            await asyncio.sleep(_TICK_S)


# ════════════════════════════════════════════════════════════════════════════
# 读侧：给详情页 / API 用
# ════════════════════════════════════════════════════════════════════════════
def get_clip(conn: Any, event_uid: Any) -> dict | None:
    cur = conn.execute(
        """
        SELECT evidence_id, status, storage_path, mime, width, height,
               duration_ms, bytes, fail_reason, ready_at
          FROM fg.event_evidence
         WHERE event_uid = %s AND kind = 'clip'
        """,
        (event_uid,),
    )
    return cur.fetchone()


def clip_status_map(event_uids: list[Any]) -> dict[str, str]:
    """批量取片段状态，给列表页/详情页拼 `has_clip`。"""
    if not event_uids:
        return {}
    rows = query(
        """
        SELECT event_uid, status
          FROM fg.event_evidence
         WHERE kind = 'clip' AND event_uid = ANY(%s)
        """,
        (list(event_uids),),
    )
    return {str(row["event_uid"]): row["status"] for row in rows}
