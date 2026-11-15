"""请求级依赖：当前用户、角色守卫、病区隔离、审计写入。"""

from __future__ import annotations

import ipaddress
import json
from dataclasses import dataclass
from typing import Any, Callable

from fastapi import Depends, HTTPException, Request

from .db import query_one
from .security import SESSION_COOKIE_NAME, unsign_session


@dataclass(frozen=True, slots=True)
class CurrentUser:
    """已通过会话校验的当前用户（只带业务需要的字段，不包含密码哈希）。"""

    user_id: int
    display_name: str
    role: str
    ward_id: str
    must_change_password: bool = False
    zone_ids: tuple[str, ...] = ()


def _unauthorized(message: str = "请先登录") -> HTTPException:
    """统一的 401 构造器。错误体形状由 main.py 的异常处理器统一包装。"""
    return HTTPException(status_code=401, detail=message)


def current_user(request: Request) -> CurrentUser:
    """FastAPI 依赖：解析 Cookie → 校验签名 → 查库确认账号仍然有效。"""
    token = request.cookies.get(SESSION_COOKIE_NAME)
    if not token:
        raise _unauthorized("请先登录")

    payload = unsign_session(token)
    if payload is None:
        raise _unauthorized("登录已过期，请重新登录")

    row = query_one(
        """
        SELECT u.user_id,
               u.display_name,
               u.role,
               u.ward_id,
               u.must_change_password,
               u.is_active,
               COALESCE(array_agg(z.zone_id) FILTER (WHERE z.zone_id IS NOT NULL), '{}')
                 AS zone_ids
          FROM fg.app_user u
          LEFT JOIN fg.app_user_zone uz ON uz.user_id = u.user_id
          LEFT JOIN fg.zone z ON z.zone_id = uz.zone_id
                             AND z.is_active
                             AND z.ward_id = u.ward_id
         WHERE u.user_id = %s
         GROUP BY u.user_id
        """,
        (payload["uid"],),
    )
    if row is None or not row.get("is_active", False):
        # 账号被删除（V1 不物理删除）或被护士长停用：旧会话立即作废
        raise _unauthorized("登录已失效，请重新登录")

    role = row.get("role")
    if role not in ("nurse", "head_nurse", "admin"):
        raise _unauthorized("账号角色无效，请联系管理员")

    return CurrentUser(
        user_id=int(row["user_id"]),
        display_name=str(row.get("display_name") or ""),
        role=str(role),
        ward_id=str(row.get("ward_id") or ""),
        must_change_password=bool(row.get("must_change_password", False)),
        # psycopg 把 UUID 数组取回成 uuid.UUID 对象，判据要跟 zone_id 列比，
        # SQL 侧比的是 UUID 类型 —— 但 scope_clauses 是把它们当参数传回去，
        # 字符串也能被 psycopg 正确适配（它认 UUID 的字符串形式）。
        # 统一转 str 是为了 dataclass 可哈希/可比较，且日志好读。
        zone_ids=tuple(str(z) for z in (row.get("zone_ids") or ())),
    )


def get_client_ip(request: Request) -> str | None:
    """取客户端 IP（写入 ``audit_log.client_ip``，类型是 INET）。"""
    for header_name in ("x-real-ip", "x-forwarded-for"):
        raw = request.headers.get(header_name)
        if not raw:
            continue
        candidate = raw.split(",")[0].strip()
        try:
            return str(ipaddress.ip_address(candidate))
        except ValueError:
            continue

    if request.client is not None:
        try:
            return str(ipaddress.ip_address(request.client.host))
        except ValueError:
            return None
    return None


def audit(
    conn: Any,
    user: CurrentUser | None,
    action: str,
    object_type: str,
    object_id: str | int | None,
    result: str = "OK",
    detail: dict | None = None,
    client_ip: str | None = None,
) -> None:
    """写一条审计记录（``fg.audit_log`` 的唯一写入口）。"""
    if result not in ("OK", "DENIED", "FAIL"):
        raise ValueError(f"非法的审计结果: {result!r}")
    if not action or not object_type:
        raise ValueError("审计 action / object_type 不能为空")

    # INET 列不接受空字符串或非法地址，先清洗
    if client_ip:
        try:
            client_ip = str(ipaddress.ip_address(str(client_ip).strip()))
        except ValueError:
            client_ip = None
    else:
        client_ip = None

    detail_json = (
        json.dumps(detail, ensure_ascii=False, default=str)
        if detail is not None
        else None
    )

    conn.execute(
        """
        INSERT INTO fg.audit_log
            (actor_id, actor_role, ward_id, action, object_type,
             object_id, result, client_ip, detail)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s::inet, %s::jsonb)
        """,
        (
            user.user_id if user is not None else None,
            user.role if user is not None else None,
            user.ward_id if user is not None else None,
            action,
            object_type,
            str(object_id) if object_id is not None else None,
            result,
            client_ip,
            detail_json,
        ),
    )


def _write_denied_audit(
    user: CurrentUser,
    request: Request,
    required_roles: frozenset[str],
) -> None:
    """越权时补写 ``DENIED`` 审计。"""
    from .db import get_pool  # 局部导入，避免模块级循环依赖

    pool = get_pool()
    with pool.connection() as conn:
        audit(
            conn,
            user,
            "DENIED",
            "endpoint",
            request.url.path,
            result="DENIED",
            detail={
                "required_roles": sorted(required_roles),
                "actual_role": user.role,
                "method": request.method,
            },
            client_ip=get_client_ip(request),
        )
        conn.commit()


def require_role(*roles: str) -> Callable[..., CurrentUser]:
    """角色守卫工厂：``Depends(require_role("head_nurse"))``。"""
    allowed = frozenset(roles)
    if not allowed:
        raise ValueError("require_role 至少需要一个角色")

    def _guard(
        request: Request,
        user: CurrentUser = Depends(current_user),
    ) -> CurrentUser:
        if user.role not in allowed:
            try:
                _write_denied_audit(user, request, allowed)
            except Exception:
                # 审计写失败不能把"无权限"变成 500；403 的结论不变
                pass
            raise HTTPException(status_code=403, detail="当前账号无权执行此操作")
        return user

    return _guard


# 现成依赖：护士长专用 / 护士（含护士长）通用
require_head = require_role("head_nurse")
require_nurse = require_role("nurse", "head_nurse")
require_admin = require_role("admin")


def home_for(role: str) -> str:
    """登录后的落地页：管理员去系统管理，护士长去总览，护士去值班台。"""
    if role == "admin":
        return "/admin/overview"
    if role == "head_nurse":
        return "/head/overview"
    return "/nurse/today"


def scoped_ward(user: CurrentUser) -> str:
    """返回当前用户的数据范围（病区）。"""
    if not user.ward_id:
        raise HTTPException(status_code=403, detail="账号未绑定病区")
    return user.ward_id




def scoped_zones(user: CurrentUser) -> tuple[str, ...] | None:
    """返回当前用户的区域范围。"""
    if user.role != "nurse":
        return None
    return user.zone_ids


def zone_clauses(alias: str, zones: tuple[str, ...] | None) -> tuple[list[str], list[Any]]:
    """产生区域权限 SQL 的唯一一处。 返回 ``(clauses, params)`` 供 ``WHERE`` 追加。"""
    if zones is None:
        return [], []
    # alias 传空串 ⇒ 拼出裸列名（给没有别名、只有一张表的 SQL 用）
    prefix = f"{alias}." if alias else ""
    return [f"{prefix}zone_id = ANY(%s)"], [list(zones)]


def scope_clauses(alias: str, user: CurrentUser) -> tuple[list[str], list[Any]]:
    """按当前用户产生区域权限条件。绝大多数调用方用这个。"""
    return zone_clauses(alias, scoped_zones(user))




def zone_key(zone: Any) -> str | None:
    """区域 id 在内存结构里的统一表示（``str`` / ``None``）。"""
    return None if zone is None else str(zone)


def zone_keys(zones: tuple[Any, ...] | None) -> tuple[str, ...] | None:
    """``zone_key`` 的元组版。``None`` 保持 ``None``（= 不收窄），空元组保持空。"""
    return None if zones is None else tuple(str(z) for z in zones)
