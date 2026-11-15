"""认证 API：登录、登出、当前用户、修改密码。"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ..db import get_conn
from ..deps import CurrentUser, audit, current_user, get_client_ip
from ..security import (
    MIN_PASSWORD_LENGTH,
    SESSION_COOKIE_NAME,
    SESSION_MAX_AGE,
    hash_password,
    sign_session,
    unsign_session,
    verify_password,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/auth", tags=["认证"])

# 登录失败锁定策略（设计文档 2.3：5 次 / 15 分钟 / 账号）
LOCK_THRESHOLD = 5
LOCK_MINUTES = 15


# ── 请求体 ────────────────────────────────────────────────────────────


class LoginBody(BaseModel):
    """登录请求体。字段名与前端表单 / 设计文档一致。"""

    login_name: str = Field(min_length=1, max_length=64, description="登录名")
    password: str = Field(min_length=1, max_length=256, description="密码")


class PasswordBody(BaseModel):
    """修改密码请求体。新密码长度在业务层再校验一次（≥ 12）。"""

    old_password: str = Field(min_length=1, max_length=256, description="原密码")
    new_password: str = Field(min_length=1, max_length=256, description="新密码")


# ── 内部工具 ──────────────────────────────────────────────────────────


def _error(
    status_code: int,
    code: str,
    message: str,
    detail: dict | None = None,
) -> JSONResponse:
    """按设计文档 8.1 的错误体形状返回：``{"error": {code, message, detail}}``。"""
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "code": code,
                "message": message,
                "detail": detail or {},
            }
        },
    )


def _login_error() -> JSONResponse:
    """登录失败的唯一出口：文案不区分账号不存在 / 密码错 / 已锁定。"""
    return _error(401, "INVALID_CREDENTIALS", "账号或密码不正确")


def _row_dict(cur: Any) -> dict | None:
    """把游标结果转成 dict（兼容 db 层是否配置了 dict_row）。"""
    row = cur.fetchone()
    if row is None:
        return None
    if isinstance(row, dict):
        return row
    names = [column.name for column in (cur.description or [])]
    return dict(zip(names, row))


def _load_user_by_login(conn: Any, login_name: str) -> dict | None:
    """按登录名查账号（含登录失败计数与锁定时间）。"""
    cur = conn.execute(
        """
        SELECT user_id,
               login_name,
               display_name,
               role,
               ward_id,
               password_hash,
               must_change_password,
               is_active,
               failed_count,
               locked_until
          FROM fg.app_user
         WHERE login_name = %s
        """,
        (login_name,),
    )
    return _row_dict(cur)


def _load_user_by_id(conn: Any, user_id: int) -> dict | None:
    """按主键查账号（登出 / 改密用）。"""
    cur = conn.execute(
        """
        SELECT user_id,
               login_name,
               display_name,
               role,
               ward_id,
               password_hash,
               must_change_password,
               is_active
          FROM fg.app_user
         WHERE user_id = %s
        """,
        (user_id,),
    )
    return _row_dict(cur)


def _to_current(row: dict) -> CurrentUser:
    """把数据库行转成 CurrentUser（只取白名单字段）。"""
    return CurrentUser(
        user_id=int(row["user_id"]),
        display_name=str(row.get("display_name") or ""),
        role=str(row.get("role") or ""),
        ward_id=str(row.get("ward_id") or ""),
        must_change_password=bool(row.get("must_change_password", False)),
    )


def _is_locked(locked_until: Any, now: datetime) -> bool:
    """``locked_until`` 是否仍在未来。兼容驱动返回 naive datetime 的情况。"""
    if locked_until is None:
        return False
    if getattr(locked_until, "tzinfo", None) is None:
        locked_until = locked_until.replace(tzinfo=timezone.utc)
    return locked_until > now


def _home_for(role: str) -> str:
    """按角色返回登录后的首页（设计文档 5.1：角色由服务端决定）。"""
    if role == "admin":
        return "/admin/overview"
    return "/head/overview" if role == "head_nurse" else "/nurse/today"


def _set_session_cookie(response: Response, token: str) -> None:
    """写会话 Cookie。 不加 ``Secure``：HTTP 环境下浏览器会直接丢弃。"""
    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=token,
        max_age=SESSION_MAX_AGE,
        path="/",
        httponly=True,
        samesite="lax",
        secure=False,
    )


def _record_login_fail(
    conn: Any,
    user: CurrentUser | None,
    login_name: str,
    reason: str,
    client_ip: str | None,
    **extra: Any,
) -> None:
    """写 ``LOGIN_FAIL`` 审计。``reason`` 只进审计，不进响应。"""
    detail: dict[str, Any] = {"login_name": login_name, "reason": reason}
    detail.update(extra)
    audit(
        conn,
        user,
        "LOGIN_FAIL",
        "session",
        login_name,
        result="FAIL",
        detail=detail,
        client_ip=client_ip,
    )


# ── 登录 ──────────────────────────────────────────────────────────────


@router.post("/login")
def login(
    body: LoginBody,
    request: Request,
    response: Response,
    conn: Any = Depends(get_conn),
):
    """账号密码登录。"""
    login_name = body.login_name.strip()
    now = datetime.now(timezone.utc)
    client_ip = get_client_ip(request)

    row = _load_user_by_login(conn, login_name)

    # ① 账号不存在：仍做一次假哈希校验，避免用响应时间枚举账号
    if row is None:
        verify_password(body.password, None)
        _record_login_fail(conn, None, login_name, "unknown_user", client_ip)
        conn.commit()
        return _login_error()

    user = _to_current(row)

    # ② 锁定期内：即使密码正确也拒绝；文案与密码错误完全一致
    if _is_locked(row.get("locked_until"), now):
        verify_password(body.password, row.get("password_hash"))
        _record_login_fail(
            conn,
            user,
            login_name,
            "locked",
            client_ip,
            locked_until=str(row.get("locked_until")),
        )
        conn.commit()
        return _login_error()

    # ③ 已禁用账号：同样不泄露状态
    if not row.get("is_active"):
        verify_password(body.password, row.get("password_hash"))
        _record_login_fail(conn, user, login_name, "inactive", client_ip)
        conn.commit()
        return _login_error()

    # ④ 密码错误：失败计数 +1；达到 5 次时锁定 15 分钟
    if not verify_password(body.password, row.get("password_hash")):
        cur = conn.execute(
            """
            UPDATE fg.app_user
               SET failed_count = failed_count + 1,
                   locked_until = CASE
                       WHEN failed_count + 1 >= %s
                       THEN now() + (%s * interval '1 minute')
                       ELSE locked_until
                   END
             WHERE user_id = %s
            RETURNING failed_count, locked_until
            """,
            (LOCK_THRESHOLD, LOCK_MINUTES, user.user_id),
        )
        stat = _row_dict(cur) or {}
        _record_login_fail(
            conn,
            user,
            login_name,
            "bad_password",
            client_ip,
            failed_count=stat.get("failed_count"),
            locked_until=(
                str(stat.get("locked_until"))
                if stat.get("locked_until") is not None
                else None
            ),
        )
        conn.commit()
        return _login_error()

    # ⑤ 登录成功：先签发令牌（密钥缺失时在写库前失败），再清零计数并写审计
    token = sign_session(user.user_id, user.role, user.ward_id)

    conn.execute(
        """
        UPDATE fg.app_user
           SET failed_count = 0,
               locked_until = NULL,
               last_login_at = now()
         WHERE user_id = %s
        """,
        (user.user_id,),
    )
    audit(
        conn,
        user,
        "LOGIN",
        "session",
        str(user.user_id),
        result="OK",
        detail={"login_name": login_name},
        client_ip=client_ip,
    )
    conn.commit()

    _set_session_cookie(response, token)

    redirect_to = _home_for(user.role)

    return {
        "user": {
            "user_id": user.user_id,
            "display_name": user.display_name,
            "role": user.role,
            "ward_id": user.ward_id,
            "must_change_password": user.must_change_password,
        },
        "redirect_to": redirect_to,
    }


# ── 登出 ──────────────────────────────────────────────────────────────


@router.post("/logout")
def logout(
    request: Request,
    response: Response,
    conn: Any = Depends(get_conn),
):
    """登出：写 ``LOGOUT`` 审计并清 Cookie。"""
    token = request.cookies.get(SESSION_COOKIE_NAME)
    payload = unsign_session(token) if token else None

    if payload is not None:
        try:
            row = _load_user_by_id(conn, payload["uid"])
            if row is not None:
                actor = _to_current(row)
            else:
                # 账号已被删（理论上不会），用令牌里的信息尽量保留归因
                actor = CurrentUser(
                    user_id=payload["uid"],
                    display_name="",
                    role=payload["role"],
                    ward_id=payload["ward"],
                    must_change_password=False,
                )
            audit(
                conn,
                actor,
                "LOGOUT",
                "session",
                str(payload["uid"]),
                result="OK",
                client_ip=get_client_ip(request),
            )
            conn.commit()
        except Exception:
            try:
                conn.rollback()
            except Exception:
                pass
            logger.warning("登出审计写入失败，已跳过", exc_info=True)

    response.delete_cookie(
        key=SESSION_COOKIE_NAME,
        path="/",
        httponly=True,
        samesite="lax",
        secure=False,
    )
    return {"ok": True}


# ── 当前用户 ──────────────────────────────────────────────────────────


@router.get("/me")
def me(user: CurrentUser = Depends(current_user)):
    """返回当前登录用户（页面壳与前端权限判断用）。"""
    return {
        "user_id": user.user_id,
        "display_name": user.display_name,
        "role": user.role,
        "ward_id": user.ward_id,
        "must_change_password": user.must_change_password,
    }


# ── 修改密码 ──────────────────────────────────────────────────────────


@router.post("/password")
def change_password(
    body: PasswordBody,
    request: Request,
    user: CurrentUser = Depends(current_user),
    conn: Any = Depends(get_conn),
):
    """修改自己的密码：需旧密码，新密码 ≥ 12 位。"""
    if len(body.new_password) < MIN_PASSWORD_LENGTH or not body.new_password.strip():
        return _error(
            422,
            "PASSWORD_TOO_SHORT",
            f"新密码至少需要 {MIN_PASSWORD_LENGTH} 位",
        )

    row = _load_user_by_id(conn, user.user_id)
    if row is None:
        return _error(401, "UNAUTHENTICATED", "登录已失效，请重新登录")

    if not verify_password(body.old_password, row.get("password_hash")):
        return _error(400, "OLD_PASSWORD_INCORRECT", "原密码不正确")

    new_hash = hash_password(body.new_password)
    conn.execute(
        """
        UPDATE fg.app_user
           SET password_hash = %s,
               must_change_password = FALSE,
               failed_count = 0,
               locked_until = NULL
         WHERE user_id = %s
        """,
        (new_hash, user.user_id),
    )
    audit(
        conn,
        user,
        "PASSWORD_CHANGE",
        "user",
        str(user.user_id),
        result="OK",
        client_ip=get_client_ip(request),
    )
    conn.commit()

    return {"ok": True, "redirect_to": _home_for(user.role)}
