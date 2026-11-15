"""app/routes/admin.py —— 护士长的账号管理、审计日志、设备只读列表。"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
import secrets
import uuid
from typing import Any, Literal

import psycopg
from fastapi import APIRouter, Depends, Query, Request
from pydantic import BaseModel, Field, field_validator

from app.attribution import LIVE_POINT_JOIN
from app.db import get_conn
from app.deps import CurrentUser, audit, current_user, require_role, scoped_ward
from app.metrics import fetch_all, fetch_one, iso, parse_time_param
from app.routes.events import (
    _count_cache,
    client_ip,
    cursor_timestamp,
    decode_cursor,
    encode_cursor,
    error_json,
)
from app.routes.live import invalidate_stream_cache
from app.security import MAX_PASSWORD_LENGTH, MIN_PASSWORD_LENGTH, hash_password

router = APIRouter(
    prefix="/api/v1/admin",
    tags=["admin"],
    dependencies=[Depends(require_role("head_nurse", "admin"))],
)

# 登录名：小写字母开头，3–32 位，允许小写字母/数字/下划线/点/连字符
LOGIN_NAME_RE = re.compile(r"^[a-z][a-z0-9_.-]{2,31}$")
ROLES = ("nurse", "head_nurse", "admin")
# 一次性密码字符集：去掉 0/O/1/l/I 这些抄写时容易看错的字符
PASSWORD_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz23456789"


def _gen_password(length: int = 12) -> str:
    """生成一次性密码（只在响应里出现一次，不入库明文）。"""
    return "".join(secrets.choice(PASSWORD_ALPHABET) for _ in range(length))


def _is_admin(user: CurrentUser) -> bool:
    return user.role == "admin"


def _actor_ward(user: CurrentUser) -> str | None:
    """护士长 → 本病区；信息科管理员 → ``None``（不限病区）。"""
    return None if _is_admin(user) else scoped_ward(user)


def _admin_only(user: CurrentUser):
    """设备/点位/区域/策略的写操作只给信息科管理员。"""
    if not _is_admin(user):
        return error_json(403, "ADMIN_ONLY", "该操作仅信息科管理员可执行")
    return None


def _create_ward(conn: Any, user: CurrentUser, requested: str | None):
    """新建资源时确定病区，返回 ``(ward, error)``。"""
    if _is_admin(user):
        ward = (requested or "").strip()
        if not ward:
            return None, error_json(422, "WARD_REQUIRED", "请指定病区")
        if fetch_one(conn, "SELECT 1 AS ok FROM fg.ward WHERE ward_id = %s",
                     (ward,)) is None:
            return None, error_json(422, "WARD_NOT_FOUND", "病区不存在")
        return ward, None
    return scoped_ward(user), None


def _target_user(conn: Any, user_id: int, user: CurrentUser):
    """按操作者权限取目标账号，返回 ``(row, error)``。"""
    row = fetch_one(
        conn,
        """
        SELECT user_id, login_name, display_name, role, ward_id, is_active,
               must_change_password, failed_count, locked_until,
               last_login_at, created_at
          FROM fg.app_user WHERE user_id = %s
        """,
        (user_id,),
    )
    if row is None:
        return None, error_json(404, "USER_NOT_FOUND", "账号不存在")
    if not _is_admin(user):
        if row.get("ward_id") != user.ward_id or row.get("role") != "nurse":
            return None, error_json(404, "USER_NOT_FOUND", "账号不存在")
    return row, None


def _user_item(row: dict) -> dict:
    """统一的账号视图。键名冻结 —— `main.py` 的账号页直接复用这套键。"""
    return {
        "user_id": row["user_id"],
        "login_name": row["login_name"],
        "display_name": row["display_name"],
        "role": row["role"],
        "ward_id": row.get("ward_id"),
        "is_active": bool(row["is_active"]),
        "must_change_password": bool(row["must_change_password"]),
        "failed_count": row.get("failed_count", 0),
        "locked_until": iso(row.get("locked_until")),
        "last_login_at": iso(row.get("last_login_at")),
        "created_at": iso(row.get("created_at")),
    }


def _model_ver_hex(value: Any) -> str | None:
    if value is None:
        return None
    return f"{int(value) & 0xFFFFFFFF:08X}"


# ════════════════════════════════════════════════════════════════════════════
# 账号：列表 / 新建 / 禁用启用 / 重置密码
# ════════════════════════════════════════════════════════════════════════════
@router.get("/users")
def list_users(
    user: CurrentUser = Depends(current_user),
    conn: Any = Depends(get_conn),
) -> Any:
    """账号列表：护士长看本病区；信息科管理员看全院。"""
    cols = """
        SELECT user_id, login_name, display_name, role, ward_id, is_active,
               must_change_password, failed_count, locked_until,
               last_login_at, created_at
          FROM fg.app_user
    """
    if _is_admin(user):
        rows = fetch_all(conn, cols + " ORDER BY ward_id NULLS FIRST, user_id")
    else:
        rows = fetch_all(conn, cols + " WHERE ward_id = %s ORDER BY user_id",
                         (scoped_ward(user),))
    return {"items": [_user_item(row) for row in rows], "total": len(rows)}


class UserCreateIn(BaseModel):
    """新建账号请求体。password 可选；不传就由服务端生成一次性密码。"""

    login_name: str
    display_name: str
    role: str
    # 起建号只给管理员：建非 admin 角色必须显式给病区
    ward_id: str | None = None
    password: str | None = None


@router.post("/users")
def create_user(
    body: UserCreateIn,
    request: Request,
    user: CurrentUser = Depends(current_user),
    conn: Any = Depends(get_conn),
) -> Any:
    """建号 → 返回一次性密码（只显示这一次，§5.7）。"""
    login_name = (body.login_name or "").strip().lower()
    display_name = (body.display_name or "").strip()

    if not _is_admin(user):
        return error_json(403, "ADMIN_ONLY", "新建账号仅信息科管理员可执行")

    if not LOGIN_NAME_RE.match(login_name):
        return error_json(422, "LOGIN_NAME_INVALID", "登录名需以小写字母开头，3–32 位，可含数字、下划线、点、连字符")
    if not display_name or len(display_name) > 32:
        return error_json(422, "DISPLAY_NAME_INVALID", "姓名不能为空且不超过 32 字")

    role = (body.role or "").strip()
    if role not in ROLES:
        return error_json(422, "ROLE_INVALID",
                          "角色只能是 nurse / head_nurse / admin")
    if role == "admin":
        if (body.ward_id or "").strip():
            return error_json(422, "WARD_INVALID", "管理员是全院角色，不能绑定病区")
        ward = None
    else:
        ward = (body.ward_id or "").strip()
        if not ward:
            return error_json(422, "WARD_REQUIRED", "请指定病区")
        if fetch_one(conn, "SELECT 1 AS ok FROM fg.ward WHERE ward_id = %s",
                     (ward,)) is None:
            return error_json(422, "WARD_NOT_FOUND", "病区不存在")
    if body.password is not None:
        if not (MIN_PASSWORD_LENGTH <= len(body.password) <= MAX_PASSWORD_LENGTH):
            return error_json(
                422,
                "PASSWORD_INVALID",
                f"密码长度需在 {MIN_PASSWORD_LENGTH}–{MAX_PASSWORD_LENGTH} 位之间",
            )
        one_time_password = body.password
    else:
        one_time_password = _gen_password()

    password_hash = hash_password(one_time_password)

    try:
        with conn.transaction():
            row = fetch_one(
                conn,
                """
                INSERT INTO fg.app_user
                    (login_name, display_name, role, ward_id, password_hash,
                     must_change_password, created_by)
                VALUES (%s, %s, %s, %s, %s, TRUE, %s)
                RETURNING user_id, login_name, display_name, role, ward_id, is_active,
                          must_change_password, failed_count, locked_until,
                          last_login_at, created_at
                """,
                (login_name, display_name, role, ward, password_hash, user.user_id),
            )
            audit(
                conn,
                user,
                "USER_CREATE",
                "user",
                str(row["user_id"]),
                "OK",
                detail={"login_name": login_name, "display_name": display_name, "role": role, "ward_id": ward},
                client_ip=client_ip(request),
            )
    except psycopg.errors.UniqueViolation:
        return error_json(409, "LOGIN_NAME_TAKEN", "登录名已存在")

    return {
        "user": _user_item(row),
        "one_time_password": one_time_password,
        "notice": "此密码只显示一次，请立即抄写并交给本人；首次登录必须修改密码。",
    }


class UserPatchIn(BaseModel):
    """账号变更：启用/禁用、改姓名、改角色、改病区（ 起）。"""

    model_config = {"extra": "forbid"}

    is_active: bool | None = None
    display_name: str | None = None
    role: str | None = None
    ward_id: str | None = None


@router.patch("/users/{user_id}")
def patch_user(
    user_id: int,
    body: UserPatchIn,
    request: Request,
    user: CurrentUser = Depends(current_user),
    conn: Any = Depends(get_conn),
) -> Any:
    """改账号：启用/禁用、姓名、角色、病区。不能删除，也不能改自己的角色/病区。"""
    fields = body.model_dump(exclude_unset=True)

    # ① 字段级权限。先于目标查询：越权看的是「你想改什么」，
    # 与 patch_point / patch_device 同序（护士长在那两条路径上也是先吃 403）。
    if not _is_admin(user):
        forbidden = sorted(k for k in fields if k != "is_active")
        if forbidden:
            return error_json(
                403, "ADMIN_ONLY",
                "护士长只能启用/停用本病区护士，不能修改：" + "、".join(forbidden),
            )
    if not fields:
        return error_json(422, "NO_FIELDS", "没有可修改的字段")

    # ② 不许动自己。禁用自己的码沿用 400 CANNOT_DISABLE_SELF（口径不变），
    # 改自己的角色/病区换一个码 —— 改错了没人能把自己改回来。
    if int(user_id) == int(user.user_id):
        if fields.get("is_active") is False:
            return error_json(400, "CANNOT_DISABLE_SELF", "不能禁用自己的账号")
        if "role" in fields or "ward_id" in fields:
            return error_json(
                400, "CANNOT_CHANGE_SELF",
                "不能修改自己的角色或病区（会把自己锁在门外）",
            )

    changed: dict[str, Any] = {}
    dropped: list[dict] = []

    with conn.transaction():
        target, err = _target_user(conn, user_id, user)
        if err:
            return err

        sets: list[str] = []
        params: list[Any] = []

        # ── 姓名 ──────────────────────────────────────────────────────────
        if "display_name" in fields:
            name = (fields["display_name"] or "").strip()
            if not name or len(name) > 32:
                return error_json(422, "DISPLAY_NAME_INVALID", "姓名不能为空且不超过 32 字")
            if name != target["display_name"]:
                changed["display_name"] = [target["display_name"], name]
                sets.append("display_name = %s")
                params.append(name)

        # ── 角色 + 病区：合并之后再判不变量 ──────────────────────────
        role = target["role"]
        if "role" in fields:
            role = (fields["role"] or "").strip()
            if role not in ROLES:
                return error_json(422, "ROLE_INVALID", "角色只能是 nurse / head_nurse / admin")

        ward = target["ward_id"]
        if role == "admin":
            if (fields.get("ward_id") or "").strip():
                return error_json(422, "WARD_INVALID", "管理员是全院角色，不能绑定病区")
            ward = None
        else:
            if "ward_id" in fields:
                # 前端 <select> 的空值是 ""，必须归一成 None：存成 '' 时
                # `'' IS NULL` 为假，CHECK 会判定「非 admin 却没病区」→ 500。
                ward = (fields["ward_id"] or "").strip() or None
            if not ward:
                return error_json(422, "WARD_REQUIRED", "护士 / 护士长必须绑定病区")
            if (
                ward != target["ward_id"]
                and fetch_one(conn, "SELECT 1 AS ok FROM fg.ward WHERE ward_id = %s", (ward,)) is None
            ):
                return error_json(422, "WARD_NOT_FOUND", "病区不存在")

        if role != target["role"]:
            changed["role"] = [target["role"], role]
        if ward != target["ward_id"]:
            changed["ward_id"] = [target["ward_id"], ward]
        if "role" in fields or "ward_id" in fields:
            # 同一条 SET 里写掉这两个字段，不留中间态
            sets += ["role = %s", "ward_id = %s"]
            params += [role, ward]

        # ── 启用 / 禁用 ───────────────────────────────────────────────────
        if "is_active" in fields:
            active = bool(fields["is_active"])
            if active:
                # 启用时顺手解锁（保留旧行为）。失败计数还在或仍在锁定期都算
                # 真变化，否则会出现「点启用没反应」；两个都干净才算无变化、
                # 不写库也不进审计（否则每次点启用都留一条空审计）。
                if not target["is_active"] or target["failed_count"] or target["locked_until"]:
                    changed["is_active"] = [bool(target["is_active"]), True]
                    sets += ["is_active = TRUE", "failed_count = 0", "locked_until = NULL"]
            elif target["is_active"]:
                changed["is_active"] = [True, False]
                sets.append("is_active = FALSE")

        # ── 幂等短路：一个字段都没真变 ⇒ 不写库、不写审计、不清缓存 ──────
        if not changed:
            return {
                "user": _user_item(target),
                "changed": {},
                "removed_zones": [],
                "removed_zone_names": [],
                "notice": "没有需要变更的内容。",
            }

        # ── 区域绑定：换病区，或新角色不再是护士 ⇒ 同一事务里全清 ─────────
        # 「新角色不是护士也清」必须做：对护士长/管理员本来就没效果
        # （scoped_zones 恒返回 None），但留着就是脏状态，日后改回护士
        # 还会凭空继承旧病区的绑定。
        if role != "nurse" or ward != target["ward_id"]:
            dropped = fetch_all(
                conn,
                """
                SELECT z.zone_id, z.name
                  FROM fg.app_user_zone uz
                  JOIN fg.zone z ON z.zone_id = uz.zone_id
                 WHERE uz.user_id = %s
                 ORDER BY z.name
                """,
                (user_id,),
            )
            if dropped:
                conn.execute("DELETE FROM fg.app_user_zone WHERE user_id = %s", (user_id,))

        params.append(user_id)
        row = fetch_one(
            conn,
            "UPDATE fg.app_user SET "
            + ", ".join(sets)
            + " WHERE user_id = %s"
            " RETURNING user_id, login_name, display_name, role, ward_id,"
            "           is_active, must_change_password, failed_count,"
            "           locked_until, last_login_at, created_at",
            tuple(params),
        )
        if row is None:
            return error_json(404, "USER_NOT_FOUND", "账号不存在")

        # 审计：角色/病区/姓名变了记 USER_UPDATE（from→to 差集 + 被撤的区域）；
        # 只动 is_active 时沿用 USER_ENABLE / USER_DISABLE，别搅乱既有审计字典。
        if {"role", "ward_id", "display_name"} & set(changed):
            action = "USER_UPDATE"
        else:
            action = "USER_ENABLE" if changed["is_active"][1] else "USER_DISABLE"
        audit(
            conn,
            user,
            action,
            "user",
            str(user_id),
            "OK",
            detail={
                "login_name": row["login_name"],
                "changed": changed,
                "removed_zones": [str(z["zone_id"]) for z in dropped],
            },
            client_ip=client_ip(request),
        )

    # 必须在事务提交之后清缓存（同 put_user_zones 的约定）：改角色会把
    # scoped_zones 从 tuple 变 None，改病区会换一个范围，撤掉区域绑定同样
    # 改变范围 —— 三者都会让 60s 的 _count_cache 与流授权缓存指向旧范围。
    # 只动 is_active 不影响数据范围，不清。
    if ({"role", "ward_id"} & set(changed)) or dropped:
        _count_cache.clear()
        invalidate_stream_cache()

    return {
        "user": _user_item(row),
        "changed": changed,
        "removed_zones": [str(z["zone_id"]) for z in dropped],
        "removed_zone_names": [z["name"] for z in dropped],
        "notice": (
            "身份已保存；该账号负责的区域已一并撤掉（角色或病区变了），需要的话请重新分配。"
            if dropped
            else "已保存，下一个请求即生效，本人无需重新登录。"
        ),
    }


@router.post("/users/{user_id}/reset-password")
def reset_user_password(
    user_id: int,
    request: Request,
    user: CurrentUser = Depends(current_user),
    conn: Any = Depends(get_conn),
) -> Any:
    """重置密码 → 返回新的一次性密码；同时清失败计数与锁定。"""
    one_time_password = _gen_password()
    password_hash = hash_password(one_time_password)

    with conn.transaction():
        _target, err = _target_user(conn, user_id, user)
        if err:
            return err
        row = fetch_one(
            conn,
            """
            UPDATE fg.app_user
               SET password_hash = %s, must_change_password = TRUE,
                   failed_count = 0, locked_until = NULL
             WHERE user_id = %s
             RETURNING user_id, login_name, display_name, role, ward_id,
                       is_active, must_change_password, failed_count,
                       locked_until, last_login_at, created_at
            """,
            (password_hash, user_id),
        )
        if row is None:
            return error_json(404, "USER_NOT_FOUND", "账号不存在")
        audit(
            conn,
            user,
            "USER_RESET",
            "user",
            str(user_id),
            "OK",
            detail={"login_name": row["login_name"]},
            client_ip=client_ip(request),
        )

    return {
        "user": _user_item(row),
        "one_time_password": one_time_password,
        "notice": "此密码只显示一次，请立即抄写并交给本人；首次登录必须修改密码。",
    }


class UserZonesIn(BaseModel):
    """全量替换：「这个护士负责哪几个区」以本次提交为准。"""

    model_config = {"extra": "forbid"}

    zone_ids: list[str] = Field(default_factory=list)


def _user_zone_rows(conn: Any, user_id: int) -> list[str]:
    return [
        str(r["zone_id"])
        for r in fetch_all(
            conn, "SELECT zone_id FROM fg.app_user_zone WHERE user_id = %s ORDER BY zone_id",
            (user_id,),
        )
    ]


@router.get("/users/{user_id}/zones")
def get_user_zones(
    user_id: int,
    user: CurrentUser = Depends(current_user),
    conn: Any = Depends(get_conn),
) -> Any:
    """某账号负责的区域 + 本病区可分配的全部区域（给界面用）。"""
    target, err = _target_user(conn, user_id, user)
    if err:
        return err
    ward = target["ward_id"]
    zones = fetch_all(
        conn,
        """
        SELECT zone_id, ward_id, name, zone_type, floor, is_active, created_at, updated_at
          FROM fg.zone WHERE ward_id = %s ORDER BY name
        """,
        (ward,),
    )
    return {
        "user": _user_item(target),
        # 护士长/管理员恒为空：它们的数据范围不受区域限制（见 scoped_zones）
        "applicable": target["role"] == "nurse",
        "zone_ids": _user_zone_rows(conn, user_id),
        "zones": [_zone_item(z) for z in zones],
    }


@router.put("/users/{user_id}/zones")
def put_user_zones(
    user_id: int,
    body: UserZonesIn,
    request: Request,
    user: CurrentUser = Depends(current_user),
    conn: Any = Depends(get_conn),
) -> Any:
    """全量替换某护士负责的区域。空列表 = 撤掉全部 = 默认拒绝。"""
    with conn.transaction():
        target, err = _target_user(conn, user_id, user)
        if err:
            return err
        if target["role"] != "nurse":
            return error_json(
                422, "NOT_A_NURSE",
                "只有值班护士按区域限定数据范围；护士长与管理员本来就管全病区/全院，"
                "分配区域不会生效。",
            )

        ward = target["ward_id"]
        wanted = sorted({str(z).strip() for z in body.zone_ids if z and str(z).strip()})

        if wanted:
            rows = fetch_all(
                conn,
                "SELECT zone_id, ward_id, name FROM fg.zone WHERE zone_id = ANY(%s)",
                (wanted,),
            )
            by_id = {str(r["zone_id"]): r for r in rows}
            missing = [z for z in wanted if z not in by_id]
            if missing:
                return error_json(404, "ZONE_NOT_FOUND", "区域不存在")
            crossed = [z for z in wanted if by_id[z]["ward_id"] != ward]
            if crossed:
                return error_json(
                    409, "WARD_MISMATCH",
                    "只能分配该护士所在病区的区域：" + "、".join(
                        by_id[z]["name"] for z in crossed),
                )

        old = set(_user_zone_rows(conn, user_id))
        added = sorted(set(wanted) - old)
        removed = sorted(old - set(wanted))
        if removed:
            conn.execute(
                "DELETE FROM fg.app_user_zone WHERE user_id = %s AND zone_id = ANY(%s)",
                (user_id, removed),
            )
        for z in added:
            conn.execute(
                "INSERT INTO fg.app_user_zone (user_id, zone_id, created_by) "
                "VALUES (%s, %s, %s) ON CONFLICT DO NOTHING",
                (user_id, z, user.user_id),
            )
        # 审计记差集而不是最终集合：出了事要能回答「谁把谁从哪挪到哪」，
        # 只记最终状态的话，两次操作之间发生了什么就查不出来了。
        audit(
            conn, user, "USER_ZONE_SET", "user", str(user_id), "OK",
            detail={
                "login_name": target["login_name"],
                "added": added,
                "removed": removed,
                "zone_ids": wanted,
            },
            client_ip=client_ip(request),
        )

    _count_cache.clear()
    invalidate_stream_cache()

    return {
        "user": _user_item(target),
        "applicable": True,
        "zone_ids": _user_zone_rows(conn, user_id),
        "added": added,
        "removed": removed,
        "notice": (
            "该护士负责的区域已更新，下一个请求即生效，无需重新登录。"
            if wanted else
            "已撤掉全部区域：该护士登录后将看不到任何数据（默认拒绝），请确认这是有意的。"
        ),
    }


# ════════════════════════════════════════════════════════════════════════════
# 审计日志（只读）
# ════════════════════════════════════════════════════════════════════════════
AUDIT_RESULTS = ("OK", "DENIED", "FAIL")


def _audit_item(row: dict) -> dict:
    return {
        "audit_id": row["audit_id"],
        "at": iso(row["at"]),
        "actor": (
            {"user_id": row["actor_id"], "display_name": row.get("actor_name")}
            if row["actor_id"] is not None
            else None
        ),
        "actor_role": row.get("actor_role"),
        "action": row["action"],
        "object_type": row["object_type"],
        "object_id": row["object_id"],
        "result": row["result"],
        "client_ip": row.get("client_ip"),
        "detail": row.get("detail"),
    }


def _split_values(raw: str | None, allowed: tuple[str, ...] | None, name: str) -> list[str] | None:
    if raw is None or not raw.strip():
        return None
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if allowed is not None:
        bad = [item for item in values if item not in allowed]
        if bad:
            raise ValueError(f"{name} 参数非法：{','.join(bad)}")
    return values or None


@router.get("/audit")
def list_audit(
    actor: str | None = Query(None, description="操作人 user_id 或 me"),
    action: str | None = Query(None, description="动作，逗号分隔多值"),
    result: str | None = Query(None, description="OK / DENIED / FAIL"),
    object_type: str | None = Query(None),
    object_id: str | None = Query(None),
    frm: str | None = Query(None, alias="from"),
    to: str | None = Query(None),
    q: str | None = Query(None, description="对象号或动作模糊搜索"),
    page: int = Query(1, ge=1),
    size: int = Query(50, ge=1, le=100),
    cursor: str | None = Query(None),
    user: CurrentUser = Depends(current_user),
    conn: Any = Depends(get_conn),
) -> Any:
    """审计日志（只读，不可删改；表上有触发器禁 UPDATE/DELETE）。"""
    if not _is_admin(user):
        return error_json(403, "ADMIN_ONLY", "审计日志仅信息科管理员可见")

    try:
        clauses = ["TRUE"]
        params: list[Any] = []

        if actor and actor.strip():
            if actor.strip() == "me":
                params_actor = int(user.user_id)
            else:
                params_actor = int(actor.strip())
            clauses.append("a.actor_id = %s")
            params.append(params_actor)

        actions = _split_values(action, None, "action")
        if actions:
            clauses.append("a.action = ANY(%s)")
            params.append(actions)

        results = _split_values(result, AUDIT_RESULTS, "result")
        if results:
            clauses.append("a.result = ANY(%s)")
            params.append(results)

        if object_type:
            clauses.append("a.object_type = %s")
            params.append(object_type)
        if object_id:
            clauses.append("a.object_id = %s")
            params.append(object_id)

        period_from = parse_time_param(frm) if frm else None
        period_to = parse_time_param(to, end_of_day=True) if to else None
        if period_from and period_to and period_from >= period_to:
            raise ValueError("from 必须早于 to")
        if period_from:
            clauses.append("a.at >= %s")
            params.append(period_from)
        if period_to:
            clauses.append("a.at < %s")
            params.append(period_to)

        if q and q.strip():
            needle = q.strip()
            clauses.append("(a.object_id ILIKE %s OR a.action ILIKE %s)")
            params.append("%" + needle + "%")
            params.append("%" + needle + "%")

        cursor_data = decode_cursor(cursor) if cursor else None
    except ValueError as exc:
        return error_json(400, "QUERY_INVALID", str(exc))

    where_sql = " AND ".join(clauses)
    keyset_sql = ""
    keyset_params: list[Any] = []
    if cursor_data is not None:
        try:
            keyset_sql = " AND (a.at, a.audit_id) < (%s, %s)"
            keyset_params = [parse_time_param(str(cursor_data["t"])), int(cursor_data["i"])]
        except (KeyError, TypeError, ValueError):
            return error_json(400, "CURSOR_INVALID", "cursor 不是合法游标")

    offset = 0
    if cursor_data is None and page > 1:
        offset = (page - 1) * size

    sql = (
        """
        SELECT a.audit_id, a.at, a.actor_id, u.display_name AS actor_name,
               a.actor_role, a.action, a.object_type, a.object_id, a.result,
               host(a.client_ip) AS client_ip, a.detail  -- INET 别 ::text（会带 /32），见 main.py::_audit_page
          FROM fg.audit_log a
          LEFT JOIN fg.app_user u ON u.user_id = a.actor_id
         WHERE """
        + where_sql
        + keyset_sql
        + " ORDER BY a.at DESC, a.audit_id DESC LIMIT %s"
    )
    query_params = list(params) + list(keyset_params) + [size + 1]
    if offset:
        sql = sql.replace(" LIMIT %s", " LIMIT %s OFFSET %s")
        query_params = list(params) + list(keyset_params) + [size + 1, offset]

    rows = fetch_all(conn, sql, query_params)
    has_more = len(rows) > size
    rows = rows[:size]
    next_cursor = None
    if has_more and rows:
        last = rows[-1]
        # 游标时间戳必须带微秒（cursor_timestamp）——同一秒内多条审计会被
        # 截断成同值，keyset 翻页会整段跳过。见 events.cursor_timestamp 的说明。
        next_cursor = encode_cursor({"t": cursor_timestamp(last["at"]),
                                     "i": last["audit_id"]})

    total_row = fetch_one(
        conn,
        "SELECT count(*) AS n FROM fg.audit_log a WHERE " + where_sql,
        params,
    )
    return {
        "items": [_audit_item(row) for row in rows],
        "next_cursor": next_cursor,
        "total_estimate": int(total_row["n"]) if total_row else 0,
        "size": size,
    }


DEVICE_CODE_RE = re.compile(r"^[A-Z0-9][A-Z0-9_-]{2,31}$")
MOUNT_TYPES = ("ceiling", "wall", "pole", "bedside", "other")
ZONE_TYPES = ("corridor", "ward_room", "bathroom", "dayroom",
              "stairwell", "nurse_station", "other")

DEVICE_SELECT_SQL = """
SELECT d.device_id, d.device_code, d.device_name, d.ward_id,
       w.ward_name AS ward_name,
       d.location_detail, d.mount_type,
       d.mount_height_cm, d.view_direction, d.lens_fov_deg, d.device_model,
       d.hardware_id, d.firmware_ver, d.camera_model, d.camera_serial,
       d.stream_path,
       d.notes, d.commissioned_at, d.is_active, d.model_ver, d.last_event_at,
       d.last_seen_at, d.last_seen_ip, d.last_policy_rev, d.last_config_hash,
       d.last_status, d.created_at, d.profile_updated_at,
       lp.point_id AS live_point_id, lp.name AS live_point_name,
       lp.zone_id AS live_zone_id, lz.name AS live_zone_name,
       lw.ward_name AS live_ward_name
  FROM fg.device d
  LEFT JOIN fg.ward w ON w.ward_id = d.ward_id
  """ + LIVE_POINT_JOIN + """
  LEFT JOIN fg.ward lw ON lw.ward_id = lp.ward_id
  LEFT JOIN fg.zone lz ON lz.zone_id = lp.zone_id
 WHERE d.ward_id = COALESCE(%s, d.ward_id)
"""


def _device_item(row: dict) -> dict:
    return {
        "device_id": str(row["device_id"]),
        "device_code": row["device_code"],
        # 七改：设备行上不再有位置副本，名字也不从点位带出 —— 没填就用设备码
        "device_name": row.get("device_name") or row["device_code"],
        "ward_id": row["ward_id"],
        "ward_name": row.get("ward_name"),
        # 区域 = 绑定了这台设备的点位所在的区域（不再有 installation zone）
        "zone_id": str(row["live_zone_id"]) if row.get("live_zone_id") else None,
        "zone_name": row.get("live_zone_name"),
        "location_detail": row.get("location_detail"),
        "mount_type": row.get("mount_type"),
        "mount_height_cm": row.get("mount_height_cm"),
        "view_direction": row.get("view_direction"),
        "lens_fov_deg": row.get("lens_fov_deg"),
        "device_model": row.get("device_model"),
        "hardware_id": row.get("hardware_id"),
        "firmware_ver": row.get("firmware_ver"),
        "camera_model": row.get("camera_model"),
        "camera_serial": row.get("camera_serial"),
        # 这台板子推的流（≠ 点位显示的流，见 patch_2026_11_05_device_stream.sql）
        "stream_path": row.get("stream_path"),
        # 绑定了这台设备的点位（七改后唯一的方向：点位选设备），见 app/attribution.py
        "live_point_id": str(row["live_point_id"]) if row.get("live_point_id") else None,
        "live_point_name": row.get("live_point_name"),
        "live_ward_name": row.get("live_ward_name"),
        "live_zone_name": row.get("live_zone_name"),
        "notes": row.get("notes"),
        "commissioned_at": iso(row.get("commissioned_at")),
        "is_active": bool(row["is_active"]),
        "model_ver": row.get("model_ver"),
        "model_ver_hex": _model_ver_hex(row.get("model_ver")),
        "last_event_at": iso(row.get("last_event_at")),
        "last_seen_at": iso(row.get("last_seen_at")),
        "last_seen_ip": str(row["last_seen_ip"]) if row.get("last_seen_ip") else None,
        "last_policy_rev": row.get("last_policy_rev"),
        "last_config_hash": row.get("last_config_hash"),
        "last_status": row.get("last_status"),
        "created_at": iso(row.get("created_at")),
        "profile_updated_at": iso(row.get("profile_updated_at")),
    }


def _fetch_device(conn, device_id: str, ward: str | None) -> dict | None:
    return fetch_one(
        conn,
        DEVICE_SELECT_SQL + " AND d.device_id = %s",
        (ward, device_id),
    )


def _zone_item(row: dict) -> dict:
    return {
        "zone_id": str(row["zone_id"]),
        "ward_id": row["ward_id"],
        "name": row["name"],
        "zone_type": row["zone_type"],
        "floor": row.get("floor"),
        "is_active": bool(row["is_active"]),
        "created_at": iso(row.get("created_at")),
        "updated_at": iso(row.get("updated_at")),
    }


def _check_zone(conn, zone_id: str | None, ward: str | None) -> str | None:
    """返回错误响应或 None。区域必须存在；``ward=None`` 表示不限病区（管理员）。"""
    if zone_id is None:
        return None
    row = fetch_one(
        conn,
        "SELECT zone_id FROM fg.zone "
        "WHERE zone_id = %s AND ward_id = COALESCE(%s, ward_id)",
        (zone_id, ward),
    )
    if row is None:
        return error_json(404, "ZONE_NOT_FOUND", "区域不存在")
    return None


_OPTIONAL_DEVICE_FIELDS = (
    "device_name", "location_detail", "mount_type", "mount_height_cm",
    "view_direction", "lens_fov_deg", "device_model", "hardware_id",
    "firmware_ver", "camera_model", "camera_serial", "stream_path", "notes",
    "commissioned_at",
)


class _BlankToNone:
    """把空字符串转成 None（只用于可选字段）。"""

    @field_validator(*_OPTIONAL_DEVICE_FIELDS, mode="before")
    @classmethod
    def _blank_to_none(cls, v):
        if isinstance(v, str) and v.strip() == "":
            return None
        return v


POINT_SELECT_SQL = """
SELECT p.point_id, p.ward_id, p.zone_id, z.name AS zone_name, p.name,
       p.light_key, p.source_type,
       p.is_active, p.created_at, p.updated_at,
       -- 这个点位显示哪台设备（可选）。流本身在设备行上，这里只是带出来给页面用
       p.device_id, d.device_code, d.stream_path AS device_stream_path,
       d.is_active AS device_active, d.last_seen_at AS device_last_seen_at
  FROM fg.point p
  LEFT JOIN fg.zone z ON z.zone_id = p.zone_id
  LEFT JOIN fg.device d ON d.device_id = p.device_id
 WHERE p.ward_id = COALESCE(%s, p.ward_id)
"""


def _point_item(row: dict) -> dict:
    return {
        "point_id": str(row["point_id"]),
        "ward_id": row["ward_id"],
        "zone_id": str(row["zone_id"]) if row.get("zone_id") else None,
        "zone_name": row.get("zone_name"),
        "name": row["name"],
        "light_key": row["light_key"],
        "device_id": str(row["device_id"]) if row.get("device_id") else None,
        "device_code": row.get("device_code"),
        "device_stream_path": row.get("device_stream_path"),
        "device_active": (bool(row["device_active"])
                          if row.get("device_active") is not None else None),
        "device_last_seen_at": iso(row.get("device_last_seen_at")),
        "source_type": row.get("source_type") or "camera",
        "is_active": bool(row["is_active"]),
        "created_at": iso(row.get("created_at")),
        "updated_at": iso(row.get("updated_at")),
    }


def _fetch_point(conn, point_id: str, ward: str | None) -> dict | None:
    return fetch_one(conn, POINT_SELECT_SQL + " AND p.point_id = %s", (ward, point_id))


# 摄像头流路径名。与 live.py 的 _STREAM_RE 必须保持一致：
# 服务端两处都会校验，前端那层只是提示，不能当防线。
_STREAM_PATH_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,31}$")


def _norm_stream_path(v):
    """空串按 NULL 处理；其余必须匹配 mediamtx 路径白名单。"""
    if v is None or not isinstance(v, str):
        return v
    v = v.strip()
    if not v:
        return None
    if not _STREAM_PATH_RE.match(v):
        raise ValueError("摄像头流只能用字母、数字、下划线、连字符，且不超过 32 位")
    return v


def _norm_device_id(v):
    """点位绑定设备：空串 = 解绑（NULL），其余必须是合法 UUID 形态。"""
    if v is None or not isinstance(v, str):
        return v
    v = v.strip()
    if not v:
        return None
    try:
        uuid.UUID(v)
    except (ValueError, AttributeError, TypeError):
        raise ValueError("设备 id 不合法")
    return v


class PointCreate(BaseModel):
    model_config = {"extra": "forbid"}

    # 只有管理员需要；护士长一律落到自己病区
    ward_id: str | None = None
    name: str = Field(min_length=1, max_length=64)
    zone_id: str | None = None
    # 门灯键留空 = 用点位名（单摄像头场景两者一样）
    light_key: str | None = Field(default=None, max_length=64)
    # 这个点位显示哪台设备。留空 = 还没接设备（可随时再绑）
    device_id: str | None = None
    source_type: str = Field(default="camera", pattern="^(camera|video_file|none)$")
    is_active: bool = True

    @field_validator("device_id", mode="before")
    @classmethod
    def _norm_device(cls, v):
        return _norm_device_id(v)

    @field_validator("zone_id", mode="before")
    @classmethod
    def _blank_zone(cls, v):
        # 空串要变成 None：直接拿去和 uuid 列比较会抛
        # invalid input syntax for type uuid —— 表现为「没选区域就 500」
        return None if isinstance(v, str) and not v.strip() else v


class PointPatch(BaseModel):
    model_config = {"extra": "forbid"}

    name: str | None = Field(default=None, min_length=1, max_length=64)
    zone_id: str | None = None
    light_key: str | None = Field(default=None, max_length=64)
    device_id: str | None = None
    source_type: str | None = Field(default=None, pattern="^(camera|video_file|none)$")
    is_active: bool | None = None

    @field_validator("device_id", mode="before")
    @classmethod
    def _norm_device(cls, v):
        return _norm_device_id(v)

    @field_validator("zone_id", mode="before")
    @classmethod
    def _blank_zone(cls, v):
        return None if isinstance(v, str) and not v.strip() else v


class DeviceCreate(_BlankToNone, BaseModel):
    model_config = {"extra": "forbid"}

    device_code: str = Field(min_length=3, max_length=32)
    # 只有管理员需要：护士长一律落到自己病区
    ward_id: str | None = None
    # 设备名选填：没填就用设备码（七改：不再从点位名带出）
    device_name: str | None = Field(default=None, min_length=1, max_length=64)
    location_detail: str | None = Field(default=None, max_length=100)
    mount_type: Literal["ceiling", "wall", "pole", "bedside", "other"] | None = None
    mount_height_cm: int | None = Field(default=None, ge=30, le=600)
    view_direction: str | None = Field(default=None, max_length=100)
    lens_fov_deg: int | None = Field(default=None, ge=10, le=180)
    device_model: str | None = Field(default=None, max_length=64)
    hardware_id: str | None = Field(default=None, max_length=64)
    firmware_ver: str | None = Field(default=None, max_length=64)
    camera_model: str | None = Field(default=None, max_length=64)
    camera_serial: str | None = Field(default=None, max_length=64)
    # 这台板子推的是哪条流（mediamtx path）。留空 = 未知，
    # 骨架退回按设备的病区+区域分发（老行为）。板端上报值优先于本字段。
    stream_path: str | None = None
    notes: str | None = Field(default=None, max_length=500)
    commissioned_at: dt.date | None = None

    @field_validator("stream_path", mode="before")
    @classmethod
    def _norm_stream(cls, v):
        return _norm_stream_path(v)


class DevicePatch(_BlankToNone, BaseModel):
    model_config = {"extra": "forbid"}

    device_name: str | None = Field(default=None, min_length=1, max_length=64)
    location_detail: str | None = Field(default=None, max_length=100)
    mount_type: Literal["ceiling", "wall", "pole", "bedside", "other"] | None = None
    mount_height_cm: int | None = Field(default=None, ge=30, le=600)
    view_direction: str | None = Field(default=None, max_length=100)
    lens_fov_deg: int | None = Field(default=None, ge=10, le=180)
    device_model: str | None = Field(default=None, max_length=64)
    hardware_id: str | None = Field(default=None, max_length=64)
    firmware_ver: str | None = Field(default=None, max_length=64)
    camera_model: str | None = Field(default=None, max_length=64)
    camera_serial: str | None = Field(default=None, max_length=64)
    stream_path: str | None = None
    notes: str | None = Field(default=None, max_length=500)
    commissioned_at: dt.date | None = None

    @field_validator("stream_path", mode="before")
    @classmethod
    def _norm_stream(cls, v):
        return _norm_stream_path(v)


class DeviceToggle(BaseModel):
    reason: str = Field(min_length=1, max_length=200)


class ZoneCreate(BaseModel):
    model_config = {"extra": "forbid"}

    # 只有管理员需要；护士长一律落到自己病区
    ward_id: str | None = None
    name: str = Field(min_length=1, max_length=40)
    zone_type: Literal["corridor", "ward_room", "bathroom", "dayroom",
                       "stairwell", "nurse_station", "other"] = "other"
    floor: str | None = Field(default=None, max_length=16)
    is_active: bool = True


class ZonePatch(BaseModel):
    model_config = {"extra": "forbid"}

    name: str | None = Field(default=None, min_length=1, max_length=40)
    zone_type: Literal["corridor", "ward_room", "bathroom", "dayroom",
                       "stairwell", "nurse_station", "other"] | None = None
    floor: str | None = Field(default=None, max_length=16)
    is_active: bool | None = None


WARD_ID_RE = re.compile(r"^[A-Z0-9][A-Z0-9_-]{1,31}$")


class WardCreate(BaseModel):
    model_config = {"extra": "forbid"}

    # ward_id 是 TEXT 主键且 创建后不可改（与 device_code 同规矩）：
    # 改它等于断开 zone/point/app_user/device 四张表的既有归属。
    ward_id: str = Field(min_length=2, max_length=32)
    ward_name: str = Field(min_length=1, max_length=40)
    floor: str | None = Field(default=None, max_length=16)


class WardPatch(BaseModel):
    """只允许改名 / 改楼层 —— `ward_id` 不在 allowed 里，PATCH 不动它。"""

    model_config = {"extra": "forbid"}

    ward_name: str | None = Field(default=None, min_length=1, max_length=40)
    floor: str | None = Field(default=None, max_length=16)


def _ward_item(row: Any) -> dict:
    return {
        "ward_id": row["ward_id"],
        "ward_name": row["ward_name"],
        "floor": row.get("floor"),
        "created_at": iso(row.get("created_at")),
    }


@router.get("/wards")
def list_wards(
    user: CurrentUser = Depends(current_user),
    conn: Any = Depends(get_conn),
) -> Any:
    """全部病区（管理员）。护士长拿不到 —— 她只关心自己那一个。"""
    err = _admin_only(user)
    if err:
        return err
    rows = fetch_all(
        conn,
        "SELECT ward_id, ward_name, floor, created_at FROM fg.ward ORDER BY ward_id",
    )
    return {"items": [_ward_item(r) for r in rows], "total": len(rows)}


@router.post("/wards")
def create_ward(
    payload: WardCreate,
    request: Request,
    user: CurrentUser = Depends(current_user),
    conn: Any = Depends(get_conn),
) -> Any:
    err = _admin_only(user)
    if err:
        return err
    # 病区编码统一大写：库里现存的都是 `4F-A` / `DEMO-OFF` 这种，
    # 不归一化的话 `4f-c` 和 `4F-C` 会变成两个病区。
    ward_id = payload.ward_id.strip().upper()
    if not WARD_ID_RE.match(ward_id):
        return error_json(422, "WARD_ID_INVALID",
                          "病区编码只能是大写字母/数字/下划线/短横线，"
                          "2–32 位且以字母或数字开头")
    try:
        row = fetch_one(
            conn,
            """
            INSERT INTO fg.ward (ward_id, ward_name, floor)
            VALUES (%s, %s, %s)
            RETURNING ward_id, ward_name, floor, created_at
            """,
            (ward_id, payload.ward_name.strip(), (payload.floor or "").strip() or None),
        )
    except psycopg.errors.UniqueViolation:
        return error_json(409, "WARD_ID_TAKEN", "这个病区编码已经存在了")
    audit(conn, user, "WARD_CREATE", "ward", ward_id,
          detail={"ward_name": payload.ward_name, "floor": payload.floor},
          client_ip=client_ip(request))
    return {"ward": _ward_item(row)}


@router.patch("/wards/{ward_id}")
def patch_ward(
    ward_id: str,
    payload: WardPatch,
    request: Request,
    user: CurrentUser = Depends(current_user),
    conn: Any = Depends(get_conn),
) -> Any:
    err = _admin_only(user)
    if err:
        return err
    fields = payload.model_dump(exclude_unset=True)
    allowed = {"ward_name", "floor"}
    # 与 `create_ward` 同口径：空串归一成 NULL。不归一的话「清空楼层」会在库里
    # 留下 `''` 而不是 NULL —— 页面上看着一样（`{% if wg.floor %}` 两者都假），
    # 但同一个「没填」有两种存储形态，后面写查询的人一定会被绊一次。
    if "floor" in fields:
        fields["floor"] = (fields["floor"] or "").strip() or None
    if "ward_name" in fields:
        fields["ward_name"] = fields["ward_name"].strip()
    sets = [f"{k} = %s" for k in fields if k in allowed]
    params = [fields[k] for k in fields if k in allowed]
    if not sets:
        return error_json(422, "NO_FIELDS", "没有可修改的字段")
    params.append(ward_id)
    row = fetch_one(
        conn,
        f"UPDATE fg.ward SET {', '.join(sets)} WHERE ward_id = %s "
        "RETURNING ward_id, ward_name, floor, created_at",
        tuple(params),
    )
    if row is None:
        return error_json(404, "WARD_NOT_FOUND", "病区不存在")
    audit(conn, user, "WARD_UPDATE", "ward", ward_id,
          detail={"fields": sorted(k for k in fields if k in allowed)},
          client_ip=client_ip(request))
    return {"ward": _ward_item(row)}


@router.get("/devices")
def list_devices(
    user: CurrentUser = Depends(current_user),
    conn: Any = Depends(get_conn),
) -> Any:
    """设备台账。不判「在线/离线」——只有板端心跳时间（§9.6）。"""
    ward = _actor_ward(user)
    rows = fetch_all(conn, DEVICE_SELECT_SQL + " ORDER BY lz.name NULLS LAST, d.device_code",
                     (ward,))
    zones = fetch_all(
        conn,
        """
        SELECT zone_id, ward_id, name, zone_type, floor, is_active, created_at, updated_at
          FROM fg.zone WHERE ward_id = COALESCE(%s, ward_id) ORDER BY name
        """,
        (ward,),
    )
    points = fetch_all(conn, POINT_SELECT_SQL + " ORDER BY p.name", (ward,))
    items = [_device_item(row) for row in rows]
    return {
        "items": items,
        "zones": [_zone_item(z) for z in zones],
        "points": [_point_item(p) for p in points],
        "total": len(items),
    }


@router.post("/devices")
def create_device(
    payload: DeviceCreate,
    request: Request,
    user: CurrentUser = Depends(current_user),
    conn: Any = Depends(get_conn),
) -> Any:
    """新建设备。ingest key 明文只在本次响应里出现一次，库里只存 sha256。"""
    err = _admin_only(user)
    if err:
        return err
    ward, ward_err = _create_ward(conn, user, payload.ward_id)
    if ward_err:
        return ward_err
    code = payload.device_code.strip()
    name = (payload.device_name or "").strip() or code

    ingest_key = f"fg-{code}-{secrets.token_hex(16)}"
    key_hash = hashlib.sha256(ingest_key.encode("utf-8")).hexdigest()
    try:
        row = fetch_one(
            conn,
            """
            INSERT INTO fg.device (
                device_code, device_name, ward_id,
                location_detail, mount_type, mount_height_cm, view_direction,
                lens_fov_deg, device_model, hardware_id, firmware_ver,
                camera_model, camera_serial, stream_path, notes, commissioned_at,
                ingest_key_hash, profile_updated_at, profile_updated_by
            ) VALUES (
                %s, %s, %s,
                %s, %s, %s, %s,
                %s, %s, %s, %s,
                %s, %s, %s, %s, %s,
                %s, now(), %s
            )
            RETURNING device_id
            """,
            (
                code, name, ward,
                payload.location_detail,
                payload.mount_type, payload.mount_height_cm,
                payload.view_direction, payload.lens_fov_deg,
                payload.device_model, payload.hardware_id, payload.firmware_ver,
                payload.camera_model, payload.camera_serial, payload.stream_path,
                payload.notes, payload.commissioned_at, key_hash, user.user_id,
            ),
        )
    except psycopg.errors.UniqueViolation:
        return error_json(409, "DEVICE_CODE_TAKEN", "设备码已存在")

    audit(conn, user, "DEVICE_CREATE", "device", str(row["device_id"]),
          detail={"device_code": code, "device_name": name},
          client_ip=client_ip(request))
    device = _device_item(_fetch_device(conn, str(row["device_id"]), ward))
    return {"device": device, "ingest_key": ingest_key}


@router.patch("/devices/{device_id}")
def patch_device(
    device_id: str,
    payload: DevicePatch,
    request: Request,
    user: CurrentUser = Depends(current_user),
    conn: Any = Depends(get_conn),
) -> Any:
    """改设备资料。device_code / ward_id 不在模型里 ⇒ 传了直接 422。"""
    err = _admin_only(user)
    if err:
        return err
    ward = _actor_ward(user)
    current = _fetch_device(conn, device_id, ward)
    if current is None:
        return error_json(404, "DEVICE_NOT_FOUND", "设备不存在")

    fields = payload.model_dump(exclude_unset=True)
    # 设备名留空 = 用设备码（与新建同一套规则）；改名不能改成空串。
    # 位置（ward/zone/location/point）不在这个模型里 —— 七改起它们只属于点位。
    if "device_name" in fields and fields["device_name"] is None:
        fields["device_name"] = current["device_code"]

    allowed = {
        "device_name", "location_detail", "mount_type", "mount_height_cm",
        "view_direction", "lens_fov_deg", "device_model", "hardware_id",
        "firmware_ver", "camera_model", "camera_serial", "stream_path",
        "notes", "commissioned_at",
    }
    sets = [f"{k} = %s" for k in fields if k in allowed]
    if not sets:
        return error_json(422, "NO_FIELDS", "没有可修改的字段")
    params = [fields[k] for k in fields if k in allowed]
    sets.append("profile_updated_at = now()")
    sets.append("profile_updated_by = %s")
    params.append(user.user_id)
    params.append(device_id)
    conn.execute(
        f"UPDATE fg.device SET {', '.join(sets)} WHERE device_id = %s AND ward_id = COALESCE(%s, ward_id)",
        (*params, ward),
    )
    audit(conn, user, "DEVICE_UPDATE", "device", device_id,
          detail={"fields": sorted(k for k in fields if k in allowed)},
          client_ip=client_ip(request))
    return {"device": _device_item(_fetch_device(conn, device_id, ward))}


def _toggle_device(device_id: str, active: bool, reason: str, request: Request,
                   user: CurrentUser, conn: Any) -> Any:
    err = _admin_only(user)
    if err:
        return err
    ward = _actor_ward(user)
    current = _fetch_device(conn, device_id, ward)
    if current is None:
        return error_json(404, "DEVICE_NOT_FOUND", "设备不存在")
    conn.execute(
        "UPDATE fg.device SET is_active = %s, profile_updated_at = now(), "
        "profile_updated_by = %s WHERE device_id = %s",
        (active, user.user_id, device_id),
    )
    audit(conn, user, "DEVICE_ENABLE" if active else "DEVICE_DISABLE",
          "device", device_id, detail={"reason": reason},
          client_ip=client_ip(request))
    return {"device": _device_item(_fetch_device(conn, device_id, ward))}


@router.post("/devices/{device_id}/disable")
def disable_device(device_id: str, payload: DeviceToggle, request: Request,
                   user: CurrentUser = Depends(current_user),
                   conn: Any = Depends(get_conn)) -> Any:
    return _toggle_device(device_id, False, payload.reason, request, user, conn)


@router.post("/devices/{device_id}/enable")
def enable_device(device_id: str, payload: DeviceToggle, request: Request,
                  user: CurrentUser = Depends(current_user),
                  conn: Any = Depends(get_conn)) -> Any:
    return _toggle_device(device_id, True, payload.reason, request, user, conn)


# ── 区域 ───────────────────────────────────────────────────────────────────
@router.get("/zones")
def list_zones(
    user: CurrentUser = Depends(current_user),
    conn: Any = Depends(get_conn),
) -> Any:
    ward = _actor_ward(user)
    rows = fetch_all(
        conn,
        """
        SELECT zone_id, ward_id, name, zone_type, floor, is_active, created_at, updated_at
          FROM fg.zone WHERE ward_id = COALESCE(%s, ward_id) ORDER BY name
        """,
        (ward,),
    )
    return {"items": [_zone_item(r) for r in rows], "total": len(rows)}


@router.post("/zones")
def create_zone(
    payload: ZoneCreate,
    request: Request,
    user: CurrentUser = Depends(current_user),
    conn: Any = Depends(get_conn),
) -> Any:
    err = _admin_only(user)
    if err:
        return err
    ward, ward_err = _create_ward(conn, user, payload.ward_id)
    if ward_err:
        return ward_err
    try:
        row = fetch_one(
            conn,
            """
            INSERT INTO fg.zone (ward_id, name, zone_type, floor, is_active)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING zone_id, ward_id, name, zone_type, floor, is_active,
                      created_at, updated_at
            """,
            (ward, payload.name.strip(), payload.zone_type, payload.floor,
             payload.is_active),
        )
    except psycopg.errors.UniqueViolation:
        return error_json(409, "ZONE_NAME_TAKEN", "同一病区内区域名不能重复")
    audit(conn, user, "ZONE_CREATE", "zone", str(row["zone_id"]),
          detail={"name": payload.name}, client_ip=client_ip(request))
    return {"zone": _zone_item(row)}


@router.patch("/zones/{zone_id}")
def patch_zone(
    zone_id: str,
    payload: ZonePatch,
    request: Request,
    user: CurrentUser = Depends(current_user),
    conn: Any = Depends(get_conn),
) -> Any:
    err = _admin_only(user)
    if err:
        return err
    ward = _actor_ward(user)
    fields = payload.model_dump(exclude_unset=True)
    if not fields:
        return error_json(422, "NO_FIELDS", "没有可修改的字段")
    allowed = {"name", "zone_type", "floor", "is_active"}
    sets = [f"{k} = %s" for k in fields if k in allowed]
    params = [fields[k] for k in fields if k in allowed]
    if not sets:
        return error_json(422, "NO_FIELDS", "没有可修改的字段")
    sets.append("updated_at = now()")
    params.extend([zone_id, ward])
    try:
        row = fetch_one(
            conn,
            f"UPDATE fg.zone SET {', '.join(sets)} "
            "WHERE zone_id = %s AND ward_id = COALESCE(%s, ward_id) "
            "RETURNING zone_id, ward_id, name, zone_type, floor, is_active, "
            "created_at, updated_at",
            tuple(params),
        )
    except psycopg.errors.UniqueViolation:
        return error_json(409, "ZONE_NAME_TAKEN", "同一病区内区域名不能重复")
    if row is None:
        return error_json(404, "ZONE_NOT_FOUND", "区域不存在")
    audit(conn, user, "ZONE_UPDATE", "zone", zone_id,
          detail={"fields": sorted(k for k in fields if k in allowed)},
          client_ip=client_ip(request))
    return {"zone": _zone_item(row)}


# ── 点位 ───────────────────────────────────────────────────────────────────
@router.get("/points")
def list_points(
    user: CurrentUser = Depends(current_user),
    conn: Any = Depends(get_conn),
) -> Any:
    ward = _actor_ward(user)
    rows = fetch_all(conn, POINT_SELECT_SQL + " ORDER BY p.name", (ward,))
    return {"items": [_point_item(r) for r in rows], "total": len(rows)}


def _device_owner(conn, device_id: str, exclude_point_id: str | None = None) -> dict | None:
    """这台设备现在被哪个点位绑着。"""
    if not device_id:
        return None
    return fetch_one(
        conn,
        "SELECT name, ward_id FROM fg.point "
        " WHERE device_id = %s::uuid AND (%s::uuid IS NULL OR point_id <> %s::uuid) "
        " LIMIT 1",
        (device_id, exclude_point_id, exclude_point_id),
    )


def _fetch_device_brief(conn, device_id: str) -> dict | None:
    """绑定点位时确认设备存在（不限病区：4F-A 的机器可以显示在 4F-B 的点位上，
    这正是「无论怎么调都行」要的能力）。"""
    if not device_id:
        return None
    return fetch_one(
        conn,
        "SELECT device_id, device_code, ward_id, stream_path, is_active "
        "  FROM fg.device WHERE device_id = %s::uuid",
        (device_id,),
    )


def _taken_response(owner: dict, device_code: str | None):
    return error_json(
        409,
        "DEVICE_TAKEN",
        f"这台设备现在挂在「{owner['name']}（{owner['ward_id']}）」上；"
        "先去那个点位解绑，再绑到这里",
        detail={"owner_point": owner["name"], "owner_ward": owner["ward_id"],
                "device_code": device_code},
    )


def _point_conflict(exc: psycopg.errors.UniqueViolation):
    """点位唯一约束冲突：区分「点位重名」和「设备被别的点位绑着」。"""
    constraint = getattr(getattr(exc, "diag", None), "constraint_name", "") or ""
    if "device_id" in constraint:
        return error_json(
            409,
            "DEVICE_TAKEN",
            "这台设备刚被别的点位绑走了，刷新页面看看是谁，再重试",
        )
    return error_json(409, "POINT_NAME_TAKEN", "同一病区内点位名不能重复")


@router.post("/points")
def create_point(
    payload: PointCreate,
    request: Request,
    user: CurrentUser = Depends(current_user),
    conn: Any = Depends(get_conn),
) -> Any:
    err = _admin_only(user)
    if err:
        return err
    ward, ward_err = _create_ward(conn, user, payload.ward_id)
    if ward_err:
        return ward_err
    bad_zone = _check_zone(conn, payload.zone_id, ward)
    if bad_zone:
        return bad_zone
    light_key = (payload.light_key or "").strip() or payload.name.strip()
    dev = None
    if payload.device_id:
        dev = _fetch_device_brief(conn, payload.device_id)
        if dev is None:
            return error_json(404, "DEVICE_NOT_FOUND", "设备不存在")
        owner = _device_owner(conn, payload.device_id)
        if owner:
            return _taken_response(owner, dev["device_code"])
    try:
        row = fetch_one(
            conn,
            """
            INSERT INTO fg.point (ward_id, zone_id, name, light_key,
                                  device_id, source_type, is_active)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            RETURNING point_id
            """,
            (ward, payload.zone_id, payload.name.strip(), light_key,
             payload.device_id, payload.source_type, payload.is_active),
        )
    except psycopg.errors.UniqueViolation as exc:
        return _point_conflict(exc)
    audit(conn, user, "POINT_CREATE", "point", str(row["point_id"]),
          detail={"name": payload.name, "light_key": light_key,
                  "device_id": payload.device_id,
                  "device_code": dev["device_code"] if dev else None},
          client_ip=client_ip(request))
    # 新建点位可能带来一条新的「谁在显示这台设备」→ 画面授权随之变化。
    # 授权缓存键里没有点位身份，不清的话那 10 秒内护士仍按旧归属放行。
    # 先提交再清（理由见 patch_point 那段）。
    fresh = _fetch_point(conn, str(row["point_id"]), ward)
    item = _point_item(fresh)
    conn.commit()
    invalidate_stream_cache()
    return {"point": item}


@router.patch("/points/{point_id}")
def patch_point(
    point_id: str,
    payload: PointPatch,
    request: Request,
    user: CurrentUser = Depends(current_user),
    conn: Any = Depends(get_conn),
) -> Any:
    fields = payload.model_dump(exclude_unset=True)
    err = _admin_only(user)
    if err:
        if user.role != "head_nurse":
            return err
        extra_fields = sorted(k for k in fields if k != "zone_id")
        if extra_fields:
            return error_json(
                403, "FORBIDDEN",
                "护士长只能修改点位的「所属区域」：" + "、".join(extra_fields),
            )
        if not fields:
            return error_json(422, "NO_FIELDS", "没有可修改的字段")
    ward = _actor_ward(user)
    current = _fetch_point(conn, point_id, ward)
    if current is None:
        return error_json(404, "POINT_NOT_FOUND", "点位不存在")
    # 这几列是 NOT NULL：显式传 null 会被数据库拒绝（NotNullViolation → 500）。
    # 「传 null」在语义上就是「不改这一项」，剔掉即可。
    # device_id 不在此列——它的 null 是有意义的「解绑设备」。
    for _k in ("name", "light_key", "source_type", "is_active"):
        if _k in fields and fields[_k] is None:
            fields.pop(_k)
    # 抢绑检查放在 UPDATE 之前（见 _device_owner 的说明：约束炸了以后查不了库）
    if fields.get("device_id"):
        bound = _fetch_device_brief(conn, fields["device_id"])
        if bound is None:
            return error_json(404, "DEVICE_NOT_FOUND", "设备不存在")
        owner = _device_owner(conn, fields["device_id"], point_id)
        if owner:
            return _taken_response(owner, bound["device_code"])
    if "zone_id" in fields:
        if user.role != "admin" and fields["zone_id"] is None:
            return error_json(
                403, "FORBIDDEN",
                "护士长不能把点位从区域里摘出来（会让所有护士立刻看不到它）。"
                "如需改归，请直接选另一个区域。",
            )
        bad_zone = _check_zone(conn, fields["zone_id"], ward)
        if bad_zone:
            return bad_zone
    if "light_key" in fields:
        fields["light_key"] = (fields["light_key"] or "").strip() or fields.get(
            "name", current["name"])
    allowed = {"name", "zone_id", "light_key", "device_id", "source_type", "is_active"}
    sets = [f"{k} = %s" for k in fields if k in allowed]
    if not sets:
        return error_json(422, "NO_FIELDS", "没有可修改的字段")
    params = [fields[k] for k in fields if k in allowed]
    sets.append("updated_at = now()")
    params.extend([point_id, ward])
    try:
        row = fetch_one(
            conn,
            f"UPDATE fg.point SET {', '.join(sets)} "
            "WHERE point_id = %s AND ward_id = COALESCE(%s, ward_id) RETURNING point_id",
            tuple(params),
        )
    except psycopg.errors.UniqueViolation as exc:
        return _point_conflict(exc)
    if row is None:
        return error_json(404, "POINT_NOT_FOUND", "点位不存在")

    updated = _fetch_point(conn, point_id, ward)
    # 记「改前 → 改后」，不只是字段名。设备绑定是可跨病区挪动的配置，
    # 只留字段名的话，出事以后查不出这个点位原来显示的是哪台机器。
    changed = {
        k: [None if current.get(k) is None else str(current.get(k)),
            None if fields[k] is None else str(fields[k])]
        for k in allowed
        if k in fields and str(current.get(k)) != str(fields[k])
    }
    audit(conn, user, "POINT_UPDATE", "point", point_id,
          detail={"fields": sorted(k for k in fields if k in allowed),
                  "changed": changed},
          client_ip=client_ip(request))
    # 点位一改，画面授权缓存里的 (stream, ward, zones) 就可能指向别的点位了
    # ——缓存键里没有点位身份，不清的话那 10 秒里护士看的还是旧归属的画面。
    #
    # 必须先提交再清缓存，顺序反了等于没修。get_conn 是生成器依赖，
    # COMMIT 只在本函数 return 之后的 teardown 里做；如果先清、后提交，那么
    # 「已清缓存、尚未提交」这段窗口里，护士的 HLS 分片请求会在另一条连接上
    # 以 READ COMMITTED 读到提交前的旧状态（点位还属旧区域、还 is_active）⇒
    # 放行并写回缓存，这条记录随后照样存活满 10 秒。清了个寂寞。
    conn.commit()
    invalidate_stream_cache()
    return {"point": _point_item(updated)}


class ConfigParams(BaseModel):
    model_config = {"extra": "forbid"}

    confirm_windows: int = Field(ge=1, le=5)
    retract_windows: int = Field(ge=1, le=20)
    cooldown_s: float = Field(ge=0, le=300)
    max_persons_stgcn: int = Field(default=3, ge=1, le=3)
    stgcn_queue_max: int = Field(default=4, ge=1, le=8)
    geo_veto_ar_max: float = Field(ge=0, le=5)
    geo_veto_bottom_shift: float = Field(ge=0, le=200)
    geo_veto_r_end: float = Field(ge=0, le=1)
    geo_veto_min_box: int = Field(ge=0, le=60)
    strong_confirm: float = Field(ge=0, le=1)


class ConfigPut(BaseModel):
    model_config = {"extra": "forbid"}

    base_rev: int = Field(ge=0)
    params: ConfigParams
    note: str = Field(min_length=1, max_length=200)


class ConfigRollback(BaseModel):
    model_config = {"extra": "forbid"}

    to_rev: int = Field(ge=1)
    note: str = Field(min_length=1, max_length=200)


def _params_hash(params: dict) -> str:
    """与板端 `_config_hash()` 同一算法：排序 JSON 的 sha256。"""
    blob = json.dumps(params, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _config_state(dev: dict, row: dict | None) -> str:
    if row is None:
        return "unconfigured"
    rev = int(row["rev"])
    if dev.get("last_policy_rev") == rev:
        return "applied"
    st = dev.get("last_status") or {}
    if isinstance(st, dict) and st.get("ack") == "FAILED" and st.get("rev") == rev:
        return "failed"
    return "pending"


@router.get("/devices/{device_id}/config")
def get_device_config(
    device_id: str,
    user: CurrentUser = Depends(current_user),
    conn: Any = Depends(get_conn),
) -> Any:
    ward = _actor_ward(user)
    dev = _fetch_device(conn, device_id, ward)
    if dev is None:
        return error_json(404, "DEVICE_NOT_FOUND", "设备不存在")
    row = fetch_one(
        conn,
        "SELECT rev, params, note, updated_at FROM fg.device_config WHERE device_id = %s",
        (device_id,),
    )
    revs = fetch_all(
        conn,
        "SELECT rev, params, note, created_at FROM fg.device_config_rev "
        "WHERE device_id = %s ORDER BY rev DESC LIMIT 20",
        (device_id,),
    )
    return {
        "device_id": device_id,
        "desired": None if row is None else {
            "rev": int(row["rev"]),
            "params": row["params"],
            "note": row.get("note"),
            "updated_at": iso(row.get("updated_at")),
            "config_hash": _params_hash(row["params"]),
        },
        "applied": {
            "rev": dev.get("last_policy_rev"),
            "config_hash": dev.get("last_config_hash"),
            "last_seen_at": iso(dev.get("last_seen_at")),
            "last_status": dev.get("last_status"),
        },
        "state": _config_state(dev, row),
        "revisions": [
            {"rev": int(r["rev"]), "params": r["params"], "note": r.get("note"),
             "created_at": iso(r.get("created_at"))}
            for r in revs
        ],
    }


def _save_config(conn, device_id: str, params: dict, note: str, user,
                 request: Request, action: str) -> dict:
    """写一条新 rev（当前 + 1）并同步 device_config。调用方负责事务。"""
    cur = fetch_one(conn, "SELECT rev FROM fg.device_config WHERE device_id = %s",
                    (device_id,))
    new_rev = (int(cur["rev"]) if cur else 0) + 1
    conn.execute(
        "INSERT INTO fg.device_config_rev (device_id, rev, params, note, created_by) "
        "VALUES (%s, %s, %s, %s, %s)",
        (device_id, new_rev, json.dumps(params), note, user.user_id),
    )
    conn.execute(
        "INSERT INTO fg.device_config (device_id, rev, params, note, updated_by) "
        "VALUES (%s, %s, %s, %s, %s) "
        "ON CONFLICT (device_id) DO UPDATE SET rev = EXCLUDED.rev, "
        "params = EXCLUDED.params, note = EXCLUDED.note, "
        "updated_by = EXCLUDED.updated_by, updated_at = now()",
        (device_id, new_rev, json.dumps(params), note, user.user_id),
    )
    audit(conn, user, action, "device", device_id,
          detail={"rev": new_rev, "note": note}, client_ip=client_ip(request))
    return {"rev": new_rev, "params": params, "config_hash": _params_hash(params)}


@router.put("/devices/{device_id}/config")
def put_device_config(
    device_id: str,
    payload: ConfigPut,
    request: Request,
    user: CurrentUser = Depends(current_user),
    conn: Any = Depends(get_conn),
) -> Any:
    err = _admin_only(user)
    if err:
        return err
    ward = _actor_ward(user)
    if _fetch_device(conn, device_id, ward) is None:
        return error_json(404, "DEVICE_NOT_FOUND", "设备不存在")
    cur = fetch_one(conn, "SELECT rev FROM fg.device_config WHERE device_id = %s",
                    (device_id,))
    cur_rev = int(cur["rev"]) if cur else 0
    if payload.base_rev != cur_rev:
        return error_json(409, "REV_CONFLICT",
                          f"配置已被修改（当前 rev {cur_rev}），请刷新后重试")
    with conn.transaction():
        saved = _save_config(conn, device_id, payload.params.model_dump(),
                             payload.note.strip(), user, request, "CONFIG_SAVE")
    return saved


@router.post("/devices/{device_id}/config/rollback")
def rollback_device_config(
    device_id: str,
    payload: ConfigRollback,
    request: Request,
    user: CurrentUser = Depends(current_user),
    conn: Any = Depends(get_conn),
) -> Any:
    err = _admin_only(user)
    if err:
        return err
    ward = _actor_ward(user)
    if _fetch_device(conn, device_id, ward) is None:
        return error_json(404, "DEVICE_NOT_FOUND", "设备不存在")
    old = fetch_one(
        conn,
        "SELECT params FROM fg.device_config_rev WHERE device_id = %s AND rev = %s",
        (device_id, payload.to_rev),
    )
    if old is None:
        return error_json(404, "REV_NOT_FOUND", "目标版本不存在")
    params = old["params"] if isinstance(old["params"], dict) else json.loads(old["params"])
    with conn.transaction():
        saved = _save_config(conn, device_id, params,
                             f"回滚到 rev {payload.to_rev}：{payload.note.strip()}",
                             user, request, "CONFIG_ROLLBACK")
    return saved
