#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""衡护 HENGUARD —— 每日对账脚本（只读，不改任何数据）。"""

from __future__ import annotations

import argparse
import os
import socket
import sqlite3
import sys
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import psycopg

DEFAULT_ENV_FILE = "/etc/fallguard/web.env"
DEFAULT_LOG_DIR = "/var/log/fallguard"

# ── SQL（全部使用 %s 占位符）─────────────────────────────────────────
SQL_TOTAL = """
    SELECT count(*) FROM fg.event
     WHERE received_at >= %s AND received_at < %s
"""
SQL_BY_DETECTION = """
    SELECT detection_state, count(*) FROM fg.event
     WHERE received_at >= %s AND received_at < %s
     GROUP BY detection_state ORDER BY detection_state
"""
SQL_BY_DISPOSITION = """
    SELECT disposition_state, count(*) FROM fg.event
     WHERE received_at >= %s AND received_at < %s
     GROUP BY disposition_state ORDER BY disposition_state
"""
SQL_BY_SEVERITY = """
    SELECT severity, count(*) FROM fg.event
     WHERE received_at >= %s AND received_at < %s
     GROUP BY severity ORDER BY severity
"""
SQL_DEVICES = """
    SELECT d.device_code, d.location, count(e.event_uid) AS event_count,
           max(e.received_at) AS last_event_at
      FROM fg.device d
      LEFT JOIN fg.event e
             ON e.device_id = d.device_id
            AND e.received_at >= %s AND e.received_at < %s
     GROUP BY d.device_code, d.location
     ORDER BY d.device_code
"""
SQL_ORPHAN_TOTAL = "SELECT count(*) FROM fg.event WHERE orphan_retract = TRUE"
SQL_ORPHANS = """
    SELECT e.event_uid, d.device_code, e.board_seq, e.person_id,
           e.window_start, e.model_ver, e.detection_state, e.received_at
      FROM fg.event e
      LEFT JOIN fg.device d ON d.device_id = e.device_id
     WHERE e.orphan_retract = TRUE
     ORDER BY e.received_at DESC
     LIMIT %s
"""
SQL_CLOUD_SEQS = """
    SELECT board_seq FROM fg.event
     WHERE received_at >= %s AND received_at < %s AND board_seq IS NOT NULL
"""

BOARD_SQL_CANDIDATES = (
    "SELECT seq, ts FROM events",
    "SELECT row_seq, ts FROM events",
    "SELECT seq, created_at FROM events",
    "SELECT row_seq, created_at FROM events",
    "SELECT seq, event_ts FROM events",
    "SELECT row_seq, event_ts FROM events",
)

_APP_TZ = None


# ── 基础工具 ─────────────────────────────────────────────────────────
def _utf8_stdout() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


def _app_tz():
    global _APP_TZ
    if _APP_TZ is None:
        try:
            from zoneinfo import ZoneInfo

            _APP_TZ = ZoneInfo("Asia/Shanghai")
        except Exception:  # noqa: BLE001 - 缺 tzdata 时退回固定 +8
            _APP_TZ = timezone(timedelta(hours=8))
    return _APP_TZ


def mask_dsn(dsn: str) -> str:
    if "://" not in dsn:
        return "（key=value 形式，已隐藏）"
    parts = urlsplit(dsn)
    if parts.password is None:
        return dsn
    netloc = parts.netloc.replace(f":{parts.password}@", ":***@", 1)
    return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))


def load_env_file(path: str) -> dict[str, str]:
    values: dict[str, str] = {}
    file_path = Path(path)
    if not file_path.is_file():
        return values
    try:
        text = file_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return values
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        if key:
            values[key] = value
    return values


def resolve_database_url(args: argparse.Namespace) -> str:
    dsn = args.database_url or os.environ.get("DATABASE_URL")
    if dsn:
        return dsn
    env_file = args.env_file or DEFAULT_ENV_FILE
    dsn = load_env_file(env_file).get("DATABASE_URL")
    if dsn:
        return dsn
    print(
        "[错误] 找不到 DATABASE_URL：请设置环境变量 DATABASE_URL、用 --database-url 指定，"
        f"或用 --env-file 指向 web.env（当前尝试：{env_file}）",
        file=sys.stderr,
    )
    raise SystemExit(1)


def resolve_day(text: str, tz) -> date:
    value = (text or "today").strip().lower()
    today = datetime.now(tz).date()
    if value == "today":
        return today
    if value == "yesterday":
        return today - timedelta(days=1)
    return date.fromisoformat(value)


def day_bounds(day: date, tz) -> tuple[datetime, datetime]:
    start = datetime.combine(day, time.min, tzinfo=tz)
    return start, start + timedelta(days=1)


def _fmt_ts(value, tz) -> str:
    if value is None:
        return "—"
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(tz).isoformat(timespec="seconds")
    return str(value)


def _num(value, width: int) -> str:
    text = "—" if value is None else str(value)
    return f"{text:>{width}}"


# ── 输出（终端 + 日志文件）───────────────────────────────────────────
class Reporter:
    def __init__(self, log_path: Path | None) -> None:
        self._fh = None
        if log_path is not None:
            try:
                log_path.parent.mkdir(parents=True, exist_ok=True)
                self._fh = log_path.open("a", encoding="utf-8")
            except OSError as exc:
                print(
                    f"[警告] 无法写日志文件 {log_path}：{exc}；本次只输出到终端",
                    file=sys.stderr,
                )

    def __call__(self, message: str = "") -> None:
        print(message)
        if self._fh is not None:
            self._fh.write(message + "\n")
            self._fh.flush()

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None


# ── 查询 ─────────────────────────────────────────────────────────────
def _scalar(conn: psycopg.Connection, sql: str, params: tuple = ()):
    row = conn.execute(sql, params).fetchone()
    return row[0] if row else None


def _rows(conn: psycopg.Connection, sql: str, params: tuple = ()) -> list[tuple]:
    return conn.execute(sql, params).fetchall()


def _report_cloud(conn: psycopg.Connection, start: datetime, end: datetime,
                  tz, rep: Reporter) -> None:
    total = _scalar(conn, SQL_TOTAL, (start, end))
    rep(f"一、云端事件数（received_at ∈ [ {_fmt_ts(start, tz)} , {_fmt_ts(end, tz)} )）")
    rep(f"   总数：{total}")
    rep()

    rep("   按模型态 detection_state：")
    rows = _rows(conn, SQL_BY_DETECTION, (start, end))
    if rows:
        for state, count in rows:
            rep(f"     {state or '—':<12} {count}")
    else:
        rep("     —")

    rep("   按处置态 disposition_state：")
    rows = _rows(conn, SQL_BY_DISPOSITION, (start, end))
    if rows:
        for state, count in rows:
            rep(f"     {state or '—':<12} {count}")
    else:
        rep("     —")

    rep("   按严重度 severity：")
    rows = _rows(conn, SQL_BY_SEVERITY, (start, end))
    if rows:
        for severity, count in rows:
            rep(f"     {severity or '—':<12} {count}")
    else:
        rep("     —")
    rep()

    rep("二、按设备（未出事件的设备也列出）")
    rep(f"   {'device_code':<16} {'location':<20} {'事件数':>6}  最后事件")
    for device_code, location, count, last_event_at in _rows(conn, SQL_DEVICES, (start, end)):
        rep(
            f"   {device_code or '—':<16} {location or '—':<20} {count:>6}  "
            f"{_fmt_ts(last_event_at, tz)}"
        )
    rep()


def _report_orphans(conn: psycopg.Connection, limit: int, tz, rep: Reporter) -> int:
    total = _scalar(conn, SQL_ORPHAN_TOTAL) or 0
    rep(f"三、孤儿撤回（orphan_retract = TRUE）：共 {total} 行（全部历史，不限当天）")
    if total:
        rows = _rows(conn, SQL_ORPHANS, (limit,))
        rep(f"   最近 {len(rows)} 行：")
        rep(
            f"   {'event_uid':<38} {'device':<14} {'board_seq':>10} {'person':>7} "
            f"{'window_start':>13} {'model_ver':>12} {'模型态':<10} 收到时间"
        )
        for (event_uid, device_code, board_seq, person_id,
             window_start, model_ver, detection_state, received_at) in rows:
            rep(
                f"   {str(event_uid):<38} {device_code or '—':<14} {_num(board_seq, 10)} "
                f"{_num(person_id, 7)} {_num(window_start, 13)} {_num(model_ver, 12)} "
                f"{detection_state or '—':<10} {_fmt_ts(received_at, tz)}"
            )
        rep()
        rep("   处置建议（设计 §9.4，本脚本只读，不会自动修改）：")
        rep("   - 确有对应事件但匹配失败 → 人工合并（UPDATE 目标行 + DELETE 孤儿行 + 写审计）；")
        rep("   - 无对应事件 → 保留孤儿行，detection_state=RETRACTED，不进统计。")
    rep()
    return 1 if total else 0


# ── 板端 SQLite 比对（可选）──────────────────────────────────────────
def _parse_board_ts(value):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        if number > 1e12:  # 毫秒时间戳
            number /= 1000.0
        try:
            return datetime.fromtimestamp(number, tz=_app_tz())
        except (OverflowError, OSError, ValueError):
            return None
    text = str(value).strip()
    try:
        number = float(text)
    except ValueError:
        number = None
    if number is not None:
        if number > 1e12:
            number /= 1000.0
        try:
            return datetime.fromtimestamp(number, tz=_app_tz())
        except (OverflowError, OSError, ValueError):
            return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=_app_tz())


def _load_board_seqs(board_db: str, start: datetime, end: datetime, rep: Reporter):
    path = Path(board_db)
    if not path.is_file():
        rep(f"   [警告] 板端库不存在：{board_db}（跳过比对）")
        return None, ""

    uri = path.resolve().as_uri() + "?mode=ro"
    try:
        connection = sqlite3.connect(uri, uri=True)
    except sqlite3.Error as exc:
        rep(f"   [警告] 打不开板端库：{exc}（跳过比对）")
        return None, ""

    try:
        for sql in BOARD_SQL_CANDIDATES:
            try:
                rows = connection.execute(sql).fetchall()
            except sqlite3.Error:
                continue
            parsed = [(int(row[0]), _parse_board_ts(row[1])) for row in rows if row[0] is not None]
            if not parsed:
                return set(), f"{sql}（空表）"
            if all(ts is not None for _seq, ts in parsed):
                # 能解析时间就按对账日过滤，避免拿全量历史去比一天
                seqs = {seq for seq, ts in parsed if start <= ts < end}
                return seqs, f"{sql}（按时间过滤）"
            seqs = {seq for seq, _ts in parsed}
            return seqs, f"{sql}（时间列无法解析，按全量比对）"
    finally:
        connection.close()
    rep("   [警告] 板端库没有可识别的 events.seq/row_seq 列，跳过比对（表结构以板端实际为准）")
    return None, ""


def _report_board(conn: psycopg.Connection, args: argparse.Namespace,
                  start: datetime, end: datetime, rep: Reporter) -> int:
    rep("四、板端对账（可选）")
    if not args.board_db:
        rep("   未提供 --board-db，跳过。板端 SQLite 先拷到云端再跑：")
        rep("   scp <板端用户>@<板端IP>:<板端SQLite路径> /tmp/board.db")
        rep("   然后：--board-db /tmp/board.db")
        rep()
        return 0

    board_seqs, scope = _load_board_seqs(args.board_db, start, end, rep)
    if board_seqs is None:
        rep()
        return 0

    cloud_seqs = {
        int(row[0]) for row in _rows(conn, SQL_CLOUD_SEQS, (start, end)) if row[0] is not None
    }
    missing = sorted(board_seqs - cloud_seqs)
    extra = sorted(cloud_seqs - board_seqs)

    rep(f"   板端库：{args.board_db}    取数方式：{scope}")
    rep(f"   板端序号数：{len(board_seqs)}    云端序号数：{len(cloud_seqs)}")
    rep(f"   云端缺失（板端有、云端无）：{len(missing)}")
    if missing:
        rep("     " + "、".join(str(seq) for seq in missing[:50]) + (" …" if len(missing) > 50 else ""))
    rep(f"   云端多出（云端有、板端无）：{len(extra)}")
    if extra:
        rep("     " + "、".join(str(seq) for seq in extra[:50]) + (" …" if len(extra) > 50 else ""))
    rep()
    return 1 if (missing or extra) else 0


# ── 参数与主流程 ─────────────────────────────────────────────────────
def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="reconcile.py",
        description="衡护 HENGUARD 每日对账（只读）：云端事件数、孤儿撤回、板端序号比对",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--date",
        default="today",
        help="对账日期：today / yesterday / YYYY-MM-DD（默认 today）",
    )
    parser.add_argument("--database-url", default=None, help="PostgreSQL DSN；默认读 DATABASE_URL")
    parser.add_argument(
        "--env-file",
        default=os.environ.get("FG_WEB_ENV", DEFAULT_ENV_FILE),
        help="systemd EnvironmentFile 路径（默认 %(default)s）",
    )
    parser.add_argument(
        "--log-dir",
        default=os.environ.get("FG_LOG_DIR", DEFAULT_LOG_DIR),
        help="日志目录（默认 %(default)s）",
    )
    parser.add_argument("--no-log-file", action="store_true", help="只输出到终端，不写日志文件")
    parser.add_argument("--board-db", default=None, help="板端 SQLite 文件路径（可选）")
    parser.add_argument(
        "--orphan-limit",
        type=int,
        default=200,
        help="孤儿撤回最多列出多少行（默认 200）",
    )
    parser.add_argument(
        "--fail-on-issues",
        action="store_true",
        help="发现孤儿撤回或板端差异时返回退出码 2（cron 告警用）",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    _utf8_stdout()
    args = _parse_args(argv)
    tz = _app_tz()

    try:
        target_day = resolve_day(args.date, tz)
    except ValueError:
        print(
            f"[错误] 无法识别的日期：{args.date}（支持 today / yesterday / YYYY-MM-DD）",
            file=sys.stderr,
        )
        return 1
    start, end = day_bounds(target_day, tz)

    run_day = datetime.now(tz).date()
    log_path = None
    if not args.no_log_file:
        log_path = Path(args.log_dir) / f"reconcile-{run_day:%Y%m%d}.log"

    rep = Reporter(log_path)
    issues = 0
    try:
        rep("=" * 72)
        rep(
            "衡护 HENGUARD 每日对账    运行时间："
            f"{datetime.now(tz).isoformat(timespec='seconds')}"
        )
        rep(f"对账日期：{target_day.isoformat()}（参数 {args.date}）    时区：Asia/Shanghai")
        rep(f"主机：{socket.gethostname()}    日志：{log_path if log_path else '（不写文件）'}")
        rep("=" * 72)

        dsn = resolve_database_url(args)
        rep(f"数据库：{mask_dsn(dsn)}")
        rep()

        try:
            conn = psycopg.connect(dsn)
        except psycopg.Error as exc:
            rep(f"[错误] 无法连接数据库：{exc}")
            return 1

        try:
            with conn:
                _report_cloud(conn, start, end, tz, rep)
                issues += _report_orphans(conn, args.orphan_limit, tz, rep)
                issues += _report_board(conn, args, start, end, rep)

                if issues:
                    rep(f"结论：发现 {issues} 类需要人工跟进的问题（见上）。")
                else:
                    rep("结论：未发现异常（孤儿撤回 0、板端比对无差异）。")
                rep("=" * 72)
        except psycopg.Error as exc:
            rep(f"[错误] 查询失败：{exc}")
            return 1
        finally:
            conn.close()
    finally:
        rep.close()

    return 2 if (issues and args.fail_on_issues) else 0


if __name__ == "__main__":
    raise SystemExit(main())
