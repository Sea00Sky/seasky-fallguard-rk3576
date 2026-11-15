#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""衡护 HENGUARD —— 账号运维命令行工具（建号 / 改密 / 重置 / 启停 / 列表）。"""

from __future__ import annotations

import argparse
import getpass
import os
import re
import secrets
import sys
from pathlib import Path
from typing import NoReturn
from urllib.parse import urlsplit, urlunsplit

import psycopg
from argon2 import PasswordHasher
from argon2.low_level import Type
from psycopg.types.json import Jsonb

# ── 常量 ─────────────────────────────────────────────────────────────
DEFAULT_ENV_FILE = "/etc/fallguard/web.env"
ROLES = ("nurse", "head_nurse", "admin")
LOGIN_NAME_RE = re.compile(r"^[a-z][a-z0-9_.-]{2,31}$")
MIN_PASSWORD_LENGTH = 12
# 防御性上限，与 app/security.py 的 MAX_PASSWORD_LENGTH 保持一致
MAX_PASSWORD_LENGTH = 256
# 去掉容易看错的字符：0/O、1/l/I
PASSWORD_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz23456789"

_ARGON2 = PasswordHasher(
    time_cost=3,
    memory_cost=65536,  # 64 MiB
    parallelism=4,
    hash_len=32,
    salt_len=16,
    type=Type.ID,  # 强制 argon2id，不是 argon2i / argon2d
)


# ── 基础工具 ─────────────────────────────────────────────────────────
def _utf8_stdout() -> None:
    """Windows 控制台默认 GBK，打印中文/一次性密码时避免编码异常。"""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


def die(message: str) -> NoReturn:
    print(f"[错误] {message}", file=sys.stderr)
    raise SystemExit(1)


def mask_dsn(dsn: str) -> str:
    """打印连接串前把密码换成 *，避免口令进终端记录。"""
    if "://" not in dsn:
        return "（key=value 形式，已隐藏）"
    parts = urlsplit(dsn)
    if parts.password is None:
        return dsn
    netloc = parts.netloc.replace(f":{parts.password}@", ":***@", 1)
    return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))


def load_env_file(path: str) -> dict[str, str]:
    """极简 EnvironmentFile 解析：KEY=VALUE，支持 # 注释与成对引号。"""
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
    die(
        "找不到 DATABASE_URL：请设置环境变量 DATABASE_URL、用 --database-url 指定，"
        f"或用 --env-file 指向 web.env（当前尝试：{env_file}）"
    )


def generate_password(length: int = 16) -> str:
    length = min(max(length, MIN_PASSWORD_LENGTH), MAX_PASSWORD_LENGTH)
    return "".join(secrets.choice(PASSWORD_ALPHABET) for _ in range(length))


def check_password_policy(password: str) -> None:
    if len(password) < MIN_PASSWORD_LENGTH:
        die(f"密码长度必须 ≥ {MIN_PASSWORD_LENGTH} 位（设计 §10.2 定版，不许放宽）")
    if len(password) > MAX_PASSWORD_LENGTH:
        die(f"密码长度不能超过 {MAX_PASSWORD_LENGTH} 个字符")


def hash_password(password: str) -> str:
    return _ARGON2.hash(password)


def _os_user() -> str:
    try:
        return getpass.getuser()
    except Exception:  # noqa: BLE001 - 拿不到就写 unknown，不能因此失败
        return "unknown"


def _cli_detail(extra: dict | None = None) -> dict:
    detail: dict = {"via": "scripts/create_user.py", "os_user": _os_user()}
    if extra:
        detail.update(extra)
    return detail


def write_audit(conn: psycopg.Connection, *, action: str, ward_id: str | None,
                object_id: str, detail: dict) -> None:
    """账号变更一律写审计（设计 §10.4 第 5 条）。actor_id 为空 = 运维命令行操作。"""
    conn.execute(
        """
        INSERT INTO fg.audit_log
            (actor_id, actor_role, ward_id, action, object_type, object_id,
             result, client_ip, detail)
        VALUES (NULL, %s, %s, %s, 'user', %s, 'OK', NULL, %s)
        """,
        ("system", ward_id, action, str(object_id), Jsonb(detail)),
    )


def _print_one_time_password(login_name: str, password: str) -> None:
    line = "=" * 60
    print()
    print(line)
    print(f"  一次性密码（只显示这一次）：{password}")
    print(f"  账号：{login_name}    首次登录必须修改密码")
    print(line)
    print("  注意：不要粘贴到聊天工具、工单、日志或截图里。")
    print()


def _fetch_user(conn: psycopg.Connection, login_name: str):
    return conn.execute(
        "SELECT user_id, ward_id, is_active FROM fg.app_user WHERE login_name = %s",
        (login_name,),
    ).fetchone()


# ── 子命令实现 ───────────────────────────────────────────────────────
def cmd_create(conn: psycopg.Connection, args: argparse.Namespace) -> None:
    login_name = args.login_name.strip()
    if not LOGIN_NAME_RE.match(login_name):
        die("登录名只允许小写字母开头，后续为小写字母/数字/下划线/点/短横线，长度 3–32")

    display_name = args.display_name.strip()
    if not display_name:
        die("显示名不能为空")
    if args.role not in ROLES:
        die(f"角色只能是 {' / '.join(ROLES)}")

    # admin 是全院角色：不绑病区，DB 有 CHECK 强制 (role='admin') = (ward_id IS NULL)
    ward_id = None
    if args.role == "admin":
        if args.ward:
            die("admin 是全院角色，不要给 --ward（DB 约束要求病区为空）")
    else:
        if not args.ward:
            die(f"角色 {args.role} 必须给 --ward")
        ward = conn.execute(
            "SELECT ward_name FROM fg.ward WHERE ward_id = %s", (args.ward,)
        ).fetchone()
        if ward is None:
            die(f"病区 {args.ward} 不存在：先执行 db/schema.sql 与 db/seed.sql")
        ward_id = args.ward

    exists = conn.execute(
        "SELECT user_id FROM fg.app_user WHERE login_name = %s", (login_name,)
    ).fetchone()
    if exists is not None:
        die(f"登录名 {login_name} 已存在（user_id={exists[0]}）；改密请用 passwd / reset-password")

    created_by = None
    if args.created_by:
        creator = conn.execute(
            "SELECT user_id FROM fg.app_user WHERE login_name = %s", (args.created_by,)
        ).fetchone()
        if creator is None:
            die(f"--created-by 指定的账号 {args.created_by} 不存在")
        created_by = creator[0]

    if args.password:
        password = args.password
        check_password_policy(password)
    else:
        password = generate_password(16)

    row = conn.execute(
        """
        INSERT INTO fg.app_user
            (login_name, display_name, role, ward_id, password_hash,
             must_change_password, is_active, failed_count, locked_until,
             created_by, created_at)
        VALUES (%s, %s, %s, %s, %s, TRUE, TRUE, 0, NULL, %s, now())
        RETURNING user_id
        """,
        (login_name, display_name, args.role, ward_id, hash_password(password), created_by),
    ).fetchone()
    user_id = row[0]

    write_audit(
        conn,
        action="USER_CREATE",
        ward_id=ward_id,
        object_id=login_name,
        detail=_cli_detail(
            {"user_id": user_id, "role": args.role, "display_name": display_name}
        ),
    )
    conn.commit()

    print(
        f"[完成] 已建号 user_id={user_id} login_name={login_name} "
        f"role={args.role} ward={ward_id or '-'}"
    )
    if args.password:
        print("[提示] 已使用你指定的初始密码；该账号首次登录仍会被要求改密。")
    else:
        _print_one_time_password(login_name, password)


def cmd_passwd(conn: psycopg.Connection, args: argparse.Namespace) -> None:
    login_name = args.login_name.strip()
    row = _fetch_user(conn, login_name)
    if row is None:
        die(f"账号 {login_name} 不存在")
    user_id, ward_id, is_active = row
    if not is_active:
        print(f"[警告] 账号 {login_name} 当前是停用状态，改密不会自动启用。")

    if args.password:
        password = args.password
    else:
        password = getpass.getpass("新密码（≥12 位，输入不回显）：")
        confirm = getpass.getpass("再输一次：")
        if password != confirm:
            die("两次输入的密码不一致")
    check_password_policy(password)

    conn.execute(
        """
        UPDATE fg.app_user
           SET password_hash = %s,
               must_change_password = %s,
               failed_count = 0,
               locked_until = NULL
         WHERE login_name = %s
        """,
        (hash_password(password), not args.no_force_change, login_name),
    )
    write_audit(
        conn,
        action="PASSWORD_CHANGE",
        ward_id=ward_id,
        object_id=login_name,
        detail=_cli_detail({"user_id": user_id, "by": "cli"}),
    )
    conn.commit()

    suffix = "" if args.no_force_change else "；该账号下次登录必须再改一次"
    print(f"[完成] 已修改 {login_name} 的密码{suffix}")


def cmd_reset_password(conn: psycopg.Connection, args: argparse.Namespace) -> None:
    login_name = args.login_name.strip()
    row = _fetch_user(conn, login_name)
    if row is None:
        die(f"账号 {login_name} 不存在")
    user_id, ward_id, _is_active = row

    password = generate_password(args.length)
    conn.execute(
        """
        UPDATE fg.app_user
           SET password_hash = %s,
               must_change_password = TRUE,
               failed_count = 0,
               locked_until = NULL
         WHERE login_name = %s
        """,
        (hash_password(password), login_name),
    )
    write_audit(
        conn,
        action="USER_RESET",
        ward_id=ward_id,
        object_id=login_name,
        detail=_cli_detail({"user_id": user_id}),
    )
    conn.commit()

    print(f"[完成] 已重置 {login_name} 的密码（原密码立即失效，锁定计数清零）")
    _print_one_time_password(login_name, password)


def _set_active(conn: psycopg.Connection, args: argparse.Namespace, active: bool) -> None:
    login_name = args.login_name.strip()
    row = _fetch_user(conn, login_name)
    if row is None:
        die(f"账号 {login_name} 不存在")
    user_id, ward_id, old_active = row
    if old_active == active:
        print(f"[跳过] {login_name} 已经是{'启用' if active else '停用'}状态")
        return

    conn.execute(
        "UPDATE fg.app_user SET is_active = %s WHERE login_name = %s",
        (active, login_name),
    )
    write_audit(
        conn,
        action="USER_ENABLE" if active else "USER_DISABLE",
        ward_id=ward_id,
        object_id=login_name,
        detail=_cli_detail({"user_id": user_id}),
    )
    conn.commit()
    print(f"[完成] 已{'启用' if active else '停用'} {login_name}")


def cmd_disable(conn: psycopg.Connection, args: argparse.Namespace) -> None:
    _set_active(conn, args, False)


def cmd_enable(conn: psycopg.Connection, args: argparse.Namespace) -> None:
    _set_active(conn, args, True)


def cmd_list(conn: psycopg.Connection, args: argparse.Namespace) -> None:
    if args.ward:
        rows = conn.execute(
            """
            SELECT user_id, login_name, display_name, role, ward_id,
                   is_active, must_change_password, last_login_at
              FROM fg.app_user
             WHERE ward_id = %s
             ORDER BY user_id
            """,
            (args.ward,),
        ).fetchall()
    else:
        rows = conn.execute(
            """
            SELECT user_id, login_name, display_name, role, ward_id,
                   is_active, must_change_password, last_login_at
              FROM fg.app_user
             ORDER BY user_id
            """
        ).fetchall()

    if not rows:
        print("（无账号）")
        return

    print(f"{'ID':>4}  {'登录名':<12} {'显示名':<8} {'角色':<11} {'病区':<7} {'状态':<5} {'首登改密':<7} 最后登录")
    for user_id, login_name, display_name, role, ward_id, is_active, must_change, last_login in rows:
        print(
            f"{user_id:>4}  {login_name:<12} {display_name:<8} {role:<11} {ward_id:<7} "
            f"{'启用' if is_active else '停用':<5} {'是' if must_change else '否':<7} "
            f"{last_login.isoformat() if last_login else '—'}"
        )


# ── 参数解析 ─────────────────────────────────────────────────────────
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="create_user.py",
        description="衡护 HENGUARD 账号运维工具（建号 / 改密 / 重置 / 启停 / 列表）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--database-url",
        default=None,
        help="PostgreSQL DSN；默认读环境变量 DATABASE_URL",
    )
    common.add_argument(
        "--env-file",
        default=os.environ.get("FG_WEB_ENV", DEFAULT_ENV_FILE),
        help="systemd EnvironmentFile 路径（默认 %(default)s）",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    p_create = sub.add_parser("create", parents=[common], help="建号（默认生成一次性密码）")
    p_create.add_argument("--login-name", required=True, help="登录名（小写）")
    p_create.add_argument("--display-name", required=True, help="显示名（如：李敏）")
    p_create.add_argument("--role", required=True, choices=ROLES, help="角色")
    p_create.add_argument("--ward", default=None,
                          help="病区 ID（如 4F-A）；admin 不要给（全院角色不绑病区）")
    p_create.add_argument("--password", default=None, help="指定初始密码（≥12 位）；不给则随机生成")
    p_create.add_argument("--created-by", default=None, help="创建人登录名（写入 created_by）")
    p_create.set_defaults(func=cmd_create)

    p_passwd = sub.add_parser("passwd", parents=[common], help="改密（交互输入或 --password）")
    p_passwd.add_argument("--login-name", required=True)
    p_passwd.add_argument("--password", default=None, help="新密码（≥12 位）；不给则交互输入两次")
    p_passwd.add_argument(
        "--no-force-change",
        action="store_true",
        help="不强制该账号下次登录改密（默认强制）",
    )
    p_passwd.set_defaults(func=cmd_passwd)

    p_reset = sub.add_parser(
        "reset-password", parents=[common], help="重置为随机一次性密码并打印"
    )
    p_reset.add_argument("--login-name", required=True)
    p_reset.add_argument("--length", type=int, default=16, help="一次性密码长度（默认 16，最小 12）")
    p_reset.set_defaults(func=cmd_reset_password)

    p_disable = sub.add_parser("disable", parents=[common], help="停用账号")
    p_disable.add_argument("--login-name", required=True)
    p_disable.set_defaults(func=cmd_disable)

    p_enable = sub.add_parser("enable", parents=[common], help="启用账号")
    p_enable.add_argument("--login-name", required=True)
    p_enable.set_defaults(func=cmd_enable)

    p_list = sub.add_parser("list", parents=[common], help="列出账号")
    p_list.add_argument("--ward", default=None, help="只看某个病区")
    p_list.set_defaults(func=cmd_list)

    return parser


def main(argv: list[str] | None = None) -> int:
    _utf8_stdout()
    args = build_parser().parse_args(argv)
    dsn = resolve_database_url(args)

    try:
        conn = psycopg.connect(dsn)
    except psycopg.OperationalError as exc:
        die(f"无法连接数据库（{mask_dsn(dsn)}）：{exc}")

    try:
        args.func(conn, args)
        return 0
    except psycopg.Error as exc:
        conn.rollback()
        die(f"数据库操作失败（已回滚）：{exc}")
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
