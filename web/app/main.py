"""衡护 HENGUARD · 应用入口与页面路由。"""

from __future__ import annotations

import asyncio
import datetime as dt
import os
import shutil
import uuid as _uuid
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import quote

from fastapi import FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from . import clips
from . import sysmon
from .attribution import LIVE_POINT_JOIN
from .config import settings
from .db import close_pool, get_pool, query, query_one
from .deps import CurrentUser, current_user, home_for, scope_clauses, scoped_zones
from .grouping import device_ward_groups, ward_groups
from .telemetry import fmt_uptime, telemetry_chips
from .metrics import _devices as metrics_devices # noqa: F401 （保留导出，便于其他模块复用）
from .metrics import hourly as metrics_hourly
from .metrics import day_start, iso, now_sh, parse_time_param
from .metrics import disposition_counts, normalize_segment, segment_disposition, worklist
from .metrics import kpi as metrics_kpi
from .metrics import overview as metrics_overview
from .metrics import UNHANDLED_STATES, pending_by_device, pending_by_location
from .routes import admin as admin_routes
from .routes import auth as auth_routes
from .routes import events as events_routes
from .routes import ingest as ingest_routes
from .routes import live as live_routes
from .routes import stats as stats_routes
# 策略是否生效的判定只有一份（routes/admin.py::_config_state）：设备抽屉里的
# 「已生效 / 待下发 / 失败」和首页运维台说的必须是同一句话，否则同屏两个答案。
from .routes.admin import _config_state as _policy_state
from .routes.admin import _user_item
from .routes.events import (
    EVENT_COLUMNS,
    EVENT_FROM,
    _count_estimate,
    _event_item,
    build_event_filters,
)
from .security import SESSION_COOKIE_NAME

BASE_DIR = Path(__file__).resolve().parent           # .../web/app
ROOT_DIR = BASE_DIR.parent                           # .../web
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = ROOT_DIR / "static"


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """启动探一次库（失败只打日志，不阻断启动；/healthz 会暴露真实状态）。"""
    try:
        get_pool()
    except Exception as exc:  # noqa: BLE001
        print(f"[startup] 数据库连接池初始化失败：{exc}")

    clip_task: asyncio.Task | None = None
    if settings.clip_enabled:
        clip_task = asyncio.create_task(clips.worker_loop(), name="fg-clip-worker")

    # 核心服务资源采样（信息科首页「系统健康」）：后台 5 秒一次，页面只读缓存。
    # 放在 lifespan 而不是模块导入：uvicorn 的 --reload/测试导入都不会多起线程。
    sysmon.start()

    yield

    if clip_task is not None:
        clip_task.cancel()
        try:
            await clip_task
        except (asyncio.CancelledError, Exception):  # noqa: BLE001
            pass
    await live_routes.aclose()   # 释放 HLS 反代的 httpx 连接池
    close_pool()


app = FastAPI(
    title="衡护 HENGUARD",
    description="病区跌倒监护 Web 系统（V1）",
    version="1.0.0",
    lifespan=lifespan,
    docs_url=None,        # 生产不暴露 Swagger
    redoc_url=None,
    openapi_url=None,
)

if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


def _asset(path: str) -> str:
    """静态资源 URL 带 mtime 版本号。"""
    try:
        stamp = int((STATIC_DIR / path).stat().st_mtime)
    except OSError:
        return f"/static/{path}"
    return f"/static/{path}?v={stamp}"


templates.env.globals["asset"] = _asset

ROLE_LABELS = {
    "nurse": "值班护士",
    "head_nurse": "护士长",
    "admin": "信息科管理员",
}


def _role_label(role: str | None) -> str:
    return ROLE_LABELS.get(str(role or ""), "未知角色")


ZONE_TYPE_LABELS = {
    "corridor": "走廊",
    "ward_room": "病房",
    "bathroom": "卫生间",
    "dayroom": "活动室",
    "stairwell": "楼梯间",
    "nurse_station": "护士站",
    "other": "其他",
}


def _zone_label(zone_type: str | None) -> str:
    return ZONE_TYPE_LABELS.get(str(zone_type or ""), "其他")


templates.env.globals["role_label"] = _role_label
templates.env.globals["zone_label"] = _zone_label
templates.env.globals["home_for"] = home_for
# 遥测条页脚「运行 18 分钟」：格式化与判级同在 telemetry.py，模板只调用
templates.env.globals["fmt_uptime"] = fmt_uptime

# ── API 路由：5 个 router，前缀互不冲突 ────────────────────────────────────
app.include_router(auth_routes.router)      # /api/v1/auth
app.include_router(events_routes.router)    # /api/v1/events ...
app.include_router(ingest_routes.router)    # /api/v1/ingest
app.include_router(live_routes.router)      # /api/v1/live（HLS 反代，会话保护）
app.include_router(stats_routes.router)     # /api/v1/stats /export
app.include_router(admin_routes.router)     # /api/v1/admin


# ══════════════════════════════════════════════════════════════════════════
# 小工具
# ══════════════════════════════════════════════════════════════════════════

@contextmanager
def _db() -> Iterator[Any]:
    """借一条连接给 metrics.* 用（它们要的是 conn，不是自动借还的 query）。"""
    with get_pool().connection() as conn:
        yield conn


def _to_dt(value: Any) -> Any:
    """ISO 字符串 → datetime（模板要 strftime）。无法解析的原样返回。"""
    if isinstance(value, str) and value:
        try:
            return dt.datetime.fromisoformat(value)
        except ValueError:
            return value
    return value


# 十七改：「在线」的唯一判据 —— 3 分钟内有心跳（与 /admin/overview 同口径）。
# 以前这条 180 秒散在三个路由里各写一遍；设备页的「状态」筛选把它变成一个字段，
# 就收成一处，别再抄第四遍。
ONLINE_WINDOW_S = 180


def _is_online(seen: Any) -> bool:
    """``last_seen_at`` → 是否在线。None / 解析不了 = 离线（不是在线）。"""
    seen = _to_dt(seen) if isinstance(seen, str) else seen
    if not isinstance(seen, dt.datetime):
        return False
    if seen.tzinfo is None:
        seen = seen.replace(tzinfo=dt.timezone.utc)
    return (dt.datetime.now(dt.timezone.utc) - seen).total_seconds() < ONLINE_WINDOW_S


_TIME_KEYS = (
    "received_at", "board_ts", "acked_at", "arrived_at", "resolved_at",
    "last_event_at", "last_login_at", "created_at", "at", "locked_until",
)


def _fix(item: dict) -> dict:
    """把一条记录里的时间字段还原成 datetime，就地修改并返回。"""
    for key in _TIME_KEYS:
        if key in item:
            item[key] = _to_dt(item[key])
    return item


def _day_window(now: dt.datetime | None = None) -> tuple[dt.datetime, dt.datetime]:
    """「今日」的 [起, 止)：北京时间当天 00:00 → 次日 00:00。"""
    now = now or now_sh()
    start = dt.datetime.combine(now.date(), dt.time.min, tzinfo=now.tzinfo)
    return start, start + dt.timedelta(days=1)


def _fmt_duration(seconds: float | None) -> str:
    """秒 → 人能读的短时长。None 一律给「—」——0 会被读成「真的没人管」。"""
    if seconds is None:
        return "—"
    total = int(round(seconds))
    if total < 60:
        return f"{total}s"
    mins, sec = divmod(total, 60)
    if mins < 60:
        return f"{mins}′{sec:02d}″" if sec else f"{mins}′"
    hours, mins = divmod(mins, 60)
    return f"{hours}h{mins:02d}m"


def _me_stats(user: CurrentUser) -> dict[str, Any]:
    """「我的」页三个数（今日口径，见 `_day_window()`）。"""
    start, end = _day_window()

    row = query_one(
        """
        SELECT count(*) FILTER (
                 WHERE action IN ('RESOLVE', 'FALSE_ALARM', 'CLOSE')
               ) AS n,
               count(*) AS n_all,
               count(*) FILTER (WHERE action = 'ACK') AS acks
          FROM fg.event_action
         WHERE actor_id = %s AND created_at >= %s AND created_at < %s
        """,
        (user.user_id, start, end),
    )
    if user.role == "head_nurse":
        action_label, action_count = "今日动作", int(row["n_all"] or 0) if row else 0
    else:
        action_label, action_count = "今日处置", int(row["n"] or 0) if row else 0
    ack_count = int(row["acks"] or 0) if row else 0

    ack = query_one(
        """
        SELECT percentile_cont(0.5) WITHIN GROUP (
                 ORDER BY EXTRACT(EPOCH FROM (e.acked_at - e.received_at))
               ) AS median_s
          FROM fg.event e
          JOIN fg.event_action a
            ON a.event_uid = e.event_uid AND a.action = 'ACK'
         WHERE a.actor_id = %s
           AND e.acked_at IS NOT NULL
           AND a.created_at >= %s
           AND a.created_at <  %s
        """,
        (user.user_id, start, end),
    )
    median_s = ack["median_s"] if ack else None

    return {
        "action_label": action_label,
        "action_count": action_count,
        "median_disp": _fmt_duration(float(median_s) if median_s is not None else None),
        "ack_count": ack_count,
    }


def _ward_card_stats(user: CurrentUser) -> list[dict[str, Any]]:
    """护士长卡片上的三个数：病区的，不是她个人的。"""
    start, end = _day_window()
    zsql, zparams = _zone_and("", user)
    row = query_one(
        """
        SELECT count(*) FILTER (WHERE ever_confirmed
                                  AND received_at >= %s AND received_at < %s) AS alerts,
               count(*) FILTER (WHERE disposition_state = ANY(%s))              AS unhandled,
               count(*) FILTER (WHERE is_false_alarm
                                  AND received_at >= %s AND received_at < %s)  AS false_alarms
          FROM fg.v_event_fact
         WHERE ward_id = %s
        """ + zsql,
        (start, end, list(UNHANDLED_STATES), start, end, user.ward_id, *zparams),
    ) or {}
    return [
        {"label": "今日告警", "value": int(row.get("alerts") or 0)},
        {"label": "待处置", "value": int(row.get("unhandled") or 0)},
        {"label": "今日误报", "value": int(row.get("false_alarms") or 0)},
    ]


def _card_stats(user: CurrentUser) -> list[dict[str, Any]] | None:
    """顶栏个人卡片的三个数。按角色取口径，模板只负责画。"""
    if user.role == "nurse":
        s = _me_stats(user)
        return [
            {"label": s["action_label"], "value": s["action_count"]},
            {"label": "中位响应", "value": s["median_disp"]},
            {"label": "今日接警", "value": s["ack_count"]},
        ]
    if user.role == "head_nurse":
        return _ward_card_stats(user)
    return None


# `_device_summary()` 删除：它只喂顶栏个人卡片页脚那行
# 「N/M 台设备启用 · 模型 vXXXX」，该页脚已按用户要求整条去掉。
# 设备数与模型版本仍可在设备管理页看到（`_device_card.html` 的 model_ver）。


def _ward(ward_id: str | None) -> dict | None:
    if not ward_id:
        return None
    return query_one(
        "SELECT ward_id, ward_name, floor FROM fg.ward WHERE ward_id = %s", (ward_id,)
    )


def _zone_and(alias: str, user: CurrentUser | None) -> tuple[str, list]:
    """区域权限条件，返回可直接拼进 ``WHERE`` 的 ``(" AND ...", params)``。"""
    if user is None:
        return "", []
    clauses, params = scope_clauses(alias, user)
    if not clauses:
        return "", []
    return " AND " + " AND ".join(clauses), params


def _point_list(ward_id: str | None, user: CurrentUser | None = None) -> list[dict]:
    """点位下拉（告警中心 / 护士长台账的「点位」筛选）。"""
    if not ward_id:
        return []
    zsql, zparams = _zone_and("p", user)
    rows = query(
        """
        SELECT p.point_id, p.name, p.light_key, p.is_active,
               p.source_type, z.name AS zone_name,
               d.device_code, d.stream_path
          FROM fg.point p
          LEFT JOIN fg.zone z ON z.zone_id = p.zone_id
          LEFT JOIN fg.device d ON d.device_id = p.device_id
         WHERE p.ward_id = %s
        """ + zsql + """
         ORDER BY p.name
        """,
        (ward_id, *zparams),
    )
    return [
        {
            "point_id": str(r["point_id"]),
            "name": r["name"],
            "light_key": r["light_key"],
            "is_active": bool(r["is_active"]),
            "stream_path": r.get("stream_path"),
            "source_type": r.get("source_type") or "camera",
            "zone_name": r.get("zone_name"),
            "device_code": r.get("device_code"),
        }
        for r in rows
    ]


def _assignee_list(ward_id: str | None) -> list[dict]:
    """「处置人」下拉的候选：本病区实际处置过事件的人（ ）。"""
    if not ward_id:
        return []
    rows = query(
        """
        SELECT u.user_id, u.display_name
          FROM fg.event e JOIN fg.app_user u ON u.user_id = e.assignee_id
         WHERE e.ward_id = %s AND u.display_name IS NOT NULL
         GROUP BY u.user_id, u.display_name
         ORDER BY u.display_name
        """,
        (ward_id,),
    )
    return [
        {"user_id": int(r["user_id"]), "display_name": r["display_name"]}
        for r in rows
    ]


def _pending_count(user: CurrentUser | None) -> int | None:
    """侧边栏 / 顶栏铃铛 / 手机底栏那个红色角标：还没人接的条数。"""
    ward_id = user.ward_id if user is not None else None
    if not ward_id:
        return None
    zsql, zparams = _zone_and("", user)
    row = query_one(
        """
        SELECT count(*) AS n FROM fg.v_event_fact
         WHERE ward_id = %s AND disposition_state = 'PENDING'
        """ + zsql,
        (ward_id, *zparams),
    )
    return int(row["n"]) if row else None


def _devices(ward_id: str | None, user: CurrentUser | None = None) -> list[dict]:
    """设备条（含停用行）。``ward_id=None`` = 全院（信息科管理员）。"""
    zsql, zparams = _zone_and("lp", user)
    rows = query(
        """
        SELECT d.device_id, d.device_code, d.device_name,
               d.ward_id, w.ward_name,
               d.location_detail, d.mount_type,
               d.mount_height_cm, d.camera_model,
               -- 编辑表单要回填这 8 列；漏查会表现为「打开编辑全是空，一保存就清空」
               d.view_direction, d.lens_fov_deg, d.device_model, d.hardware_id,
               d.firmware_ver, d.camera_serial, d.stream_path, d.commissioned_at, d.notes,
               d.last_seen_at,
               d.last_policy_rev, d.last_config_hash, d.last_status,
               d.last_event_at, d.model_ver, d.is_active, d.created_at,
               -- 设备的位置 = **绑定了它的点位**（七改：设备行上不再有位置副本）
               lp.point_id AS live_point_id, lp.name AS live_point_name,
               lp.light_key AS live_light_key, lp.zone_id AS live_zone_id,
               -- 2026-11-06 十七改：设备页要画「病区 → 区域 → 卡片」三层，病区得从**点位**拿
               lp.ward_id AS live_ward_id,
               lw.ward_name AS live_ward_name, lz.name AS live_zone_name
          FROM fg.device d
          LEFT JOIN fg.ward w ON w.ward_id = d.ward_id
          """ + LIVE_POINT_JOIN + """
          LEFT JOIN fg.ward lw ON lw.ward_id = lp.ward_id
          LEFT JOIN fg.zone lz ON lz.zone_id = lp.zone_id
         WHERE COALESCE(lp.ward_id, d.ward_id)
               = COALESCE(%s, COALESCE(lp.ward_id, d.ward_id))
        """ + zsql + """
         ORDER BY w.ward_name NULLS LAST, lz.name NULLS LAST, d.device_code
        """,
        (ward_id, *zparams),
    )
    return [
        {
            "device_id": str(r["device_id"]),
            "device_code": r["device_code"],
            "device_name": r.get("device_name") or r["device_code"],
            "ward_id": r.get("ward_id"),
            "ward_name": r.get("ward_name"),
            # 区域来自绑定点位：卡片分组、门灯、权限都按它
            "zone_id": str(r["live_zone_id"]) if r.get("live_zone_id") else None,
            "zone_name": r.get("live_zone_name"),
            "location_detail": r.get("location_detail"),
            "mount_type": r.get("mount_type"),
            "mount_height_cm": r.get("mount_height_cm"),
            "camera_model": r.get("camera_model"),
            "view_direction": r.get("view_direction"),
            "lens_fov_deg": r.get("lens_fov_deg"),
            "device_model": r.get("device_model"),
            "hardware_id": r.get("hardware_id"),
            "firmware_ver": r.get("firmware_ver"),
            "camera_serial": r.get("camera_serial"),
            # 这台板子推的流（≠ point_stream_path 点位显示的流）。同样必须回填：
            # 漏查时编辑表单渲染成空串，点一次保存就把登记值清成 NULL。
            "stream_path": r.get("stream_path"),
            # 绑定了这台设备的点位 —— 位置、画面、骨架、告警全按它
            "live_point_id": (str(r["live_point_id"]) if r.get("live_point_id") else None),
            "live_point_name": r.get("live_point_name"),
            "live_light_key": r.get("live_light_key"),
            "live_ward_id": r.get("live_ward_id"),
            "live_ward_name": r.get("live_ward_name"),
            "live_zone_name": r.get("live_zone_name"),
            # 模板里用 [:10] 切日期，必须是 'YYYY-MM-DD' 形态的字符串
            "commissioned_at": str(r["commissioned_at"]) if r.get("commissioned_at") else None,
            "notes": r.get("notes"),
            "last_seen_at": _to_dt(iso(r["last_seen_at"])),
            "last_policy_rev": r.get("last_policy_rev"),
            "last_config_hash": r.get("last_config_hash"),
            "last_status": r.get("last_status"),
            "last_event_at": _to_dt(iso(r["last_event_at"])),
            "model_ver": r["model_ver"],
            "is_active": bool(r["is_active"]),
            "online": _is_online(r.get("last_seen_at")),
            # 十八改：遥测小格（CPU / NPU / 温度 / 内存 / FPS）。判级在
            # `app/telemetry.py`，模板只画 —— 阈值只许有那一份。
            "telemetry": telemetry_chips(r.get("last_status")),
            "created_at": _to_dt(iso(r["created_at"])),
        }
        for r in rows
    ]


def _ctx(request: Request, user: CurrentUser, **extra: Any) -> dict[str, Any]:
    """页面公共上下文。base.html 依赖 user / ward / pending_count / system_status。"""
    ctx: dict[str, Any] = {
        "request": request,
        "user": user,
        "ward": _ward(user.ward_id),
        "pending_count": _pending_count(user),
        "system_status": "online",
        "active_nav": request.url.path,
        "no_zone": user.role == "nurse" and not user.zone_ids,
        # 顶栏个人卡片（ 取代「我的」页）：今日三个数。
        # 只给临床角色 —— 管理员没有处置动作，查了也全是 0（还白跑两条 SQL）。
        # （原来的 `device_sum` 设备页脚 已删，见 `_device_summary` 处注释。）
    }
    ctx["card_stats"] = _card_stats(user)
    ctx.update(extra)
    return ctx


def _render(name: str, ctx: dict[str, Any], status_code: int = 200) -> HTMLResponse:
    return templates.TemplateResponse(
        request=ctx["request"], name=name, context=ctx, status_code=status_code
    )


def _login_redirect(request: Request) -> RedirectResponse:
    nxt = request.url.path
    if request.url.query:
        nxt = f"{nxt}?{request.url.query}"
    return RedirectResponse(f"/login?next={quote(nxt, safe='/')}", status_code=302)


def _guard(request: Request, *roles: str) -> tuple[CurrentUser | None, Any | None]:
    """页面守卫：未登录 → 登录页；角色不符 → 403 页。"""
    try:
        user = current_user(request)
    except HTTPException:
        return None, _login_redirect(request)

    if roles and user.role not in roles:
        return None, _render(
            "errors/403.html",
            {
                "request": request,
                "user": user,
                "error_message": "当前账号无权访问该页面",
            },
            status_code=403,
        )
    return user, None


def _error_page(request: Request, status_code: int, message: str) -> HTMLResponse:
    name = {403: "errors/403.html", 404: "errors/404.html"}.get(status_code, "errors/500.html")
    user = None
    try:
        user = current_user(request)
    except HTTPException:
        pass
    return _render(
        name,
        {
            "request": request,
            "user": user,
            "error_message": message,
            "request_id": getattr(request.state, "request_id", ""),
        },
        status_code=status_code,
    )


def _parse_uid(raw: str) -> _uuid.UUID | None:
    try:
        return _uuid.UUID(str(raw))
    except (ValueError, AttributeError, TypeError):
        return None


def _blamed_filter(exc: ValueError, args: dict[str, Any]) -> str | None:
    """从 ``build_event_filters`` 的报错文案里认出是哪一项非法。"""
    msg = str(exc)
    for key in args:
        if msg.startswith(key + " "):
            return key
    return None


DISPOSITION_RANK = (
    "CASE e.disposition_state "
    "WHEN 'PENDING' THEN 0 WHEN 'ACKED' THEN 1 WHEN 'ON_SCENE' THEN 2 ELSE 3 END"
)
SEVERITY_RANK = "CASE e.severity WHEN 'S1' THEN 1 WHEN 'S2' THEN 2 ELSE 3 END"
ORDER_DISPOSITION = (
    f"ORDER BY {DISPOSITION_RANK}, {SEVERITY_RANK}, e.received_at DESC, e.event_uid DESC"
)
ORDER_TIME_DESC = "ORDER BY e.received_at DESC, e.event_uid DESC"


def _page_events(
    request: Request,
    user: CurrentUser,
    *,
    sort: str,
    default_size: int = 20,
    assignee_id: int | None = None,
    default_today: bool = False,
) -> dict[str, Any]:
    """按 URL 查询参数取一页事件（列表页共用）。"""
    qp = request.query_params
    today_str = now_sh().date().isoformat()

    def _bound(key: str, *, end_of_day: bool = False) -> dt.datetime | None:
        """页面时间参数。``today`` / ``yesterday`` 统一交给 parse_time_param，
        避免页面与导出使用两套解析逻辑。"""
        raw = qp.get(key)
        if raw is None:
            if not default_today:
                return None
            raw = "today"
        raw = raw.strip()
        if not raw:
            return None
        try:
            return parse_time_param(raw, end_of_day=end_of_day)
        except ValueError:
            return None

    def _echo_date(key: str) -> str:
        """回填给 <input type="date"> 的值：必须是 YYYY-MM-DD。"""
        raw = qp.get(key)
        if raw is None:
            return today_str if default_today else ""
        raw = raw.strip()
        if not raw:
            return ""
        keyword = raw.lower()
        if keyword in ("today", "yesterday"):
            day = now_sh().date() - dt.timedelta(days=1 if keyword == "yesterday" else 0)
            return day.isoformat()
        return raw

    # 手改 URL 传了非法枚举值时逐项剔掉那一项，而不是整批丢弃：
    # 「静默放宽」比 500 更危险 —— 页面照常显示，却是另一批数据。
    filter_args: dict[str, Any] = {
        "disposition": qp.get("disposition"),
        "detection": qp.get("detection"),
        "severity": qp.get("severity"),
        "device": qp.get("device"),
        # 点位筛选走位置快照（见 build_event_filters 的注释）。
        # 老的 ?device=<device_code> 仍然有效，只是界面上不再拿它当「点位」用。
        "point": qp.get("point"),
        "q": qp.get("q"),
        "is_test": qp.get("test"),
        "outcome": qp.get("outcome"),
    }

    # 处置人筛选（ ）：URL 里 ?assignee=<user_id>，与 /api/v1/events、
    # /api/v1/export/events.csv 的同名参数同义（"me" 也认）。
    # 只有代码层没钉死处置人时才读 URL：/nurse/history 是
    # `assignee_id=user.user_id` 传进来的，URL 不许把它改成别人
    # （否则「只读自己」的边界被一个查询参数推翻）。
    # 非法值当没筛，与其它筛选「逐项剔掉」同一条规矩。
    if assignee_id is None:
        raw_assignee = (qp.get("assignee") or "").strip()
        if raw_assignee == "me":
            assignee_id = int(user.user_id)
        elif raw_assignee:
            try:
                assignee_id = int(raw_assignee)
            except ValueError:
                assignee_id = None

    # 时间窗只解析一次：计数查询要复用同一对边界，各算各的必然对不上
    frm = _bound("from")
    to_bound = _bound("to", end_of_day=True)

    while True:
        try:
            where, params = build_event_filters(
                ward_id=user.ward_id,
                zone_ids=scoped_zones(user),
                frm=frm,
                to=to_bound,
                assignee_id=assignee_id,
                **filter_args,
            )
            break
        except ValueError as exc:
            bad = _blamed_filter(exc, filter_args)
            if bad is None:                       # 认不出是谁的错，别再猜了
                raise
            filter_args[bad] = None               # 只剔掉出问题的那一项

    try:
        page = max(1, int(qp.get("page") or 1))
    except ValueError:
        page = 1
    try:
        size = min(100, max(1, int(qp.get("size") or default_size)))
    except ValueError:
        size = default_size

    # 两种排序的来龙去脉见模块级的 ORDER_DISPOSITION 那段注释。
    order = ORDER_DISPOSITION if sort == "disposition" else ORDER_TIME_DESC

    rows = query(
        f"SELECT {EVENT_COLUMNS} {EVENT_FROM} WHERE {where} {order} LIMIT %s OFFSET %s",
        (*params, size, (page - 1) * size),
    )

    with _db() as conn:
        total = _count_estimate(conn, where, list(params))

    return {
        "events": [_fix(_event_item(r)) for r in rows],
        "total": total,
        "page": page,
        "size": size,
        # 实际生效的时间窗（已解析成 datetime），给同页的计数查询复用
        "range": (frm, to_bound),
        # 实际生效的 (where, params)。同页要做「和列表逐条对得上」的统计时必须用它：
        # 拿 URL 原值重算一遍会把「非法筛选值逐项剔除」的逻辑再走一次，
        # 两处只要差一项，就会出现「表里 37 条、统计说 35 条」。
        "where": where,
        "params": list(params),
        # 用实际生效的值回填筛选框，而不是 URL 里的原值：
        # 非法值被剔掉后，界面必须显示"没筛这一项"，不能假装筛了。
        "filters": {
            "disposition": filter_args["disposition"] or "",
            "detection": filter_args["detection"] or "",
            "severity": filter_args["severity"] or "",
            "device": filter_args["device"] or "",
            "point": filter_args["point"] or "",
            "from": _echo_date("from"),
            "to": _echo_date("to"),
            "q": filter_args["q"] or "",
            "test": filter_args["is_test"] or "",
            "outcome": filter_args["outcome"] or "",
            # 回填的是实际生效的值：非法 user_id 被剔掉后，下拉必须显示「全部」
            "assignee": str(assignee_id) if assignee_id is not None else "",
            "sort": sort,
        },
    }


def _with_response_s(events: list[dict]) -> list[dict]:
    """给每行补「响应用时（秒）」= acked_at - received_at，没接警的给 None。"""
    for ev in events:
        acked, received = ev.get("acked_at"), ev.get("received_at")
        ev["response_s"] = None
        if not acked or not received:
            continue
        try:
            ev["response_s"] = int(
                (dt.datetime.fromisoformat(acked) - dt.datetime.fromisoformat(received))
                .total_seconds()
            )
        except (TypeError, ValueError):
            pass
    return events


def _my_record_summary(where: str, params: list) -> dict:
    """「我的处置记录」顶部那三个数：共几条 / 已闭环几条 / 中位响应几秒。"""
    try:
        row = query_one(
            f"""
            SELECT count(*) AS n,
                   count(*) FILTER (
                     WHERE e.disposition_state IN ('RESOLVED', 'FALSE_ALARM')
                   ) AS closed,
                   ROUND(percentile_cont(0.5) WITHIN GROUP (
                     ORDER BY CASE WHEN e.acked_at IS NOT NULL
                                   THEN EXTRACT(EPOCH FROM (e.acked_at - e.received_at))
                              END
                   ))::int AS median_s
            {EVENT_FROM}
            WHERE {where}
            """,
            tuple(params),
        )
    except Exception:  # noqa: BLE001 - 统计挂了不该让整个台账白屏
        return {"n": None, "closed": None, "median_s": None}
    if not row:
        return {"n": 0, "closed": 0, "median_s": None}
    return {
        "n": int(row["n"] or 0),
        "closed": int(row["closed"] or 0),
        "median_s": int(row["median_s"]) if row.get("median_s") is not None else None,
    }


def _event_detail(uid: _uuid.UUID | None, ward_id: str,
                  user: CurrentUser | None = None) -> dict | None:
    """事件详情（含 actions / assignee），抽屉与整页共用。"""
    if uid is None or not ward_id:
        return None
    zsql, zparams = _zone_and("e", user)
    row = query_one(
        f"SELECT {EVENT_COLUMNS} {EVENT_FROM} WHERE e.event_uid = %s AND e.ward_id = %s" + zsql,
        (uid, ward_id, *zparams),
    )
    if row is None:
        return None
    item = _fix(_event_item(row))
    actions = query(
        """
        SELECT a.action_id, a.action, a.from_state, a.to_state, a.outcome,
               a.reason, a.created_at, a.actor_id,
               u.display_name AS actor_name
          FROM fg.event_action a LEFT JOIN fg.app_user u ON u.user_id = a.actor_id
         WHERE a.event_uid = %s ORDER BY a.created_at
        """,
        (uid,),
    )
    for act in actions:
        act["created_at"] = _to_dt(iso(act["created_at"]))
        act["actor"] = (
            {"user_id": act["actor_id"], "display_name": act["actor_name"]}
            if act["actor_id"] is not None
            else None
        )
    item["actions"] = actions
    return item


def _point_detail(point_id: str, ward_id: str,
                  user: CurrentUser | None = None) -> dict | None:
    """点位详情 + 绑定在它上面的那台设备（可能没有）。"""
    try:
        pid = _uuid.UUID(point_id)
    except (ValueError, AttributeError, TypeError):
        return None

    zsql, zparams = _zone_and("p", user)
    row = query_one(
        """
        SELECT p.point_id, p.ward_id, p.name, p.light_key, p.zone_id,
               p.is_active, p.source_type,
               z.name AS zone_name,
               d.device_id, d.device_code,
               -- 设备名可为 NULL（2026-11-05 清空了演示期从点位名拼出的名字）⇒ 回落设备码
               COALESCE(d.device_name, d.device_code) AS device_name,
               d.stream_path, d.last_seen_at, d.is_active AS device_active
          FROM fg.point p
          LEFT JOIN fg.zone z ON z.zone_id = p.zone_id
          LEFT JOIN fg.device d ON d.device_id = p.device_id
         WHERE p.point_id = %s AND p.ward_id = %s
        """ + zsql + """
         LIMIT 1
        """,
        (pid, ward_id, *zparams),
    )
    if row is None:
        return None
    return {
        "point_id": str(row["point_id"]),
        "ward_id": row["ward_id"],
        "name": row["name"],
        "light_key": row["light_key"],
        "zone_id": str(row["zone_id"]) if row.get("zone_id") else None,
        "zone_name": row.get("zone_name"),
        "is_active": bool(row["is_active"]),
        # 画面地址 = 绑定设备推的那条流；没有绑定、或设备没填推流 ⇒ None = 未配置
        "stream_path": row.get("stream_path"),
        "source_type": row.get("source_type") or "camera",
        "device_id": str(row["device_id"]) if row.get("device_id") else None,
        "device_code": row.get("device_code"),
        "device_name": row.get("device_name"),
        "location": row.get("light_key"),
        "last_seen_at": _to_dt(iso(row["last_seen_at"])),
        "device_active": bool(row["device_active"]) if row.get("device_active") is not None else None,
    }


def _point_events(ward_id: str, light_key: str, limit: int = 20) -> list[dict]:
    """该点位的历史事件。"""
    # COALESCE 与 pending_by_location / build_event_filters(point=) 逐字一致：
    # 三处只要有一处口径不同，就会出现「灯上 19 条、点进去 18 条」。
    rows = query(
        f"SELECT {EVENT_COLUMNS} {EVENT_FROM} "
        "WHERE e.ward_id = %s AND COALESCE(e.location_text, '') = %s "
        "ORDER BY e.received_at DESC, e.event_uid DESC LIMIT %s",
        (ward_id, light_key, limit),
    )
    return [_fix(_event_item(r)) for r in rows]


# ══════════════════════════════════════════════════════════════════════════
# 登录 / 登出 / 根路径
# ══════════════════════════════════════════════════════════════════════════

DEMO_HINT_ENABLED = os.environ.get("FG_DEMO_HINT", "1").strip() != "0"
# 演示口令不写死在代码里：部署时用 FG_DEMO_PASSWORD 指定；不设置的话
# 登录页只提示账号名、不显示口令，chip 点一下也就只填账号。
DEMO_PASSWORD = os.environ.get("FG_DEMO_PASSWORD", "")
DEMO_ACCOUNTS = (
    {"login": "wangf", "name": "王芳", "role": "护士长",
     "desc": "病区总览 · 事件台账 · 设备与模型"},
    {"login": "limin", "name": "李敏", "role": "值班护士",
     "desc": "值班工作台 · 告警处置 · 历史事件"},
    # 信息科管理员：与上面两个账号共用同一个演示密码，
    # must_change_password 已置 FALSE，否则一点「填入」就被拽去改密页。
    {"login": "admin", "name": "信息科管理员", "role": "管理员",
     "desc": "设备台账 · 判定策略 · 账号与审计"},
    # ── 4F-B 神经内科（ 演示数据一起建的）──────────────────────────
    # 这三个账号随演示数据存在（部署脚本清掉演示数据时会一并删掉）。
    # 所以登录页渲染前会按库里的 is_active 过滤一遍，删了就不会再提示，
    # 免得有人照着登录页去点一个不存在的账号（连错 5 次锁 15 分钟）。
    {"login": "qianh", "name": "钱虹", "role": "护士长",
     "desc": "4F-B 病区总览 · 事件台账 · 设备与模型"},
    {"login": "zhaol", "name": "赵磊", "role": "值班护士",
     "desc": "4F-B 北区病房 · 值班工作台 · 告警处置"},
    {"login": "sunq", "name": "孙倩", "role": "值班护士",
     "desc": "4F-B 北走廊 · 值班工作台 · 告警处置"},
)


def _alive_demo_accounts() -> tuple[dict[str, str], ...]:
    """登录页的演示账号提示，按库里的启用账号过滤一遍。"""
    try:
        alive = {r["login_name"] for r in query("SELECT login_name FROM fg.app_user WHERE is_active")}
    except Exception:  # noqa: BLE001
        return DEMO_ACCOUNTS
    return tuple(a for a in DEMO_ACCOUNTS if a["login"] in alive)


@app.get("/login", response_class=HTMLResponse)
def page_login(request: Request):
    """登录页。"""
    try:
        user = current_user(request)
    except HTTPException:
        user = None
    if user is not None:
        return RedirectResponse(
            home_for(user.role), 302
        )

    try:
        row = query_one("SELECT count(*) AS n FROM fg.device WHERE is_active")
        cam_total = int(row["n"]) if row else 0
    except Exception:  # noqa: BLE001 - 库没起来时登录页仍要能打开
        cam_total = 0

    return _render(
        "login.html",
        {
            "request": request,
            "next_url": request.query_params.get("next") or "",
            "error": "登录已过期，请重新登录"
            if request.query_params.get("reason") == "expired"
            else "",
            "reason": request.query_params.get("reason") or "",
            "gateway_online": True,     # 能渲染出这一页就说明 Web 层活着
            "camera_ok": cam_total,
            "camera_total": cam_total,
            "last_login_at": None,
            "demo_accounts": _alive_demo_accounts() if DEMO_HINT_ENABLED else (),
            "demo_password": DEMO_PASSWORD,
        },
    )


@app.post("/logout")
def page_logout():
    """base.html 的登出按钮是普通 POST，清 Cookie 后回登录页。"""
    resp = RedirectResponse("/login", status_code=302)
    resp.delete_cookie(SESSION_COOKIE_NAME, path="/")
    return resp


@app.get("/")
def page_root(request: Request):
    user, denied = _guard(request)
    if denied:
        return denied
    return RedirectResponse(
        home_for(user.role), 302
    )


# ══════════════════════════════════════════════════════════════════════════
# 护士端
# ══════════════════════════════════════════════════════════════════════════

@app.get("/nurse/today", response_class=HTMLResponse)
def page_nurse_today(request: Request, focus: str | None = None):
    user, denied = _guard(request, "nurse", "head_nurse")
    if denied:
        return denied

    ward = user.ward_id
    # 改版：主区从「告警流（按时间倒序的流水）」换成「按处置态分段的处置队列」。
    # 流水回答不了「我接警之后下一步在哪」——那条事件会掉出前 5、也会被新事件顶走；
    # 分段队列把「待接警 / 已接警 / 已到场 / 已闭环」各自成段，行内只留一个下一步。
    seg = normalize_segment(request.query_params.get("disposition"))

    # 点位筛选（ 用户要的「灯角小按钮」）：
    # 点一下 = 队列只看这个点位，再点一下 = 取消；可以多选（点几个就是几个）。
    # 只筛队列（四段 Tab + 列表 + 工作区落点），白板那三个数仍是病区级 ——
    # 白板回答「病区现在有没有事」，跟着点位变会让人误读成「告警变少了」。
    # 值就是 light_key（事件 location_text 快照里存的东西），不是 point_id：
    # 与门灯、点位页历史、告警中心同一口径，换 id 会对不上历史事件。
    #
    # 用 `getlist` 而不是 `get`：`?point=A&point=B` 重复参数才是多值，
    # 逗号拼串会被「点位名里本来就有逗号」破功（light_key 是自由文本）。
    # 去重保序：灯上点两下同一个点位不该产生两个 `point=`。
    points: list[str] = []
    for raw in request.query_params.getlist("point"):
        k = (raw or "").strip()
        if k and k not in points:
            points.append(k)

    as_of = now_sh()
    day_from, _ = _day_window(as_of)
    zones = scoped_zones(user)
    with _db() as conn:
        data = metrics_overview(conn, ward, day_from, as_of, received_since=day_from,
                                zone_ids=zones)
        # 闭环只增不减，不截的话跑几个月「已闭环」这格就上万了。
        # 计数和列表必须传同一个日界，否则标签数字和列表对不上。
        queue_counts = disposition_counts(conn, ward, closed_since=day_from,
                                          points=points, received_since=day_from,
                                          zone_ids=zones)
        # 白板是病区级、不跟点位筛选（见上面的注释）。有筛选时单独取一份不筛的，
        # 没有筛选就复用同一份，不多查一次。
        day_counts = (disposition_counts(conn, ward, closed_since=day_from,
                                         received_since=day_from, zone_ids=zones)
                      if points else queue_counts)
        rows = worklist(conn, ward, seg, closed_since=day_from, points=points,
                        received_since=day_from, zone_ids=zones)

    raw_kpi = data.get("kpi") or {}
    devices = _devices(ward, user)

    # 状态带按点位渲染，不按设备（ 用户拍板）：
    # 点位可以没有设备（实验点位 / 还没绑机）也必须点得开；
    # 混着两套样式（门灯 + 小 chip）在同一行里，排版会散。
    # 六改：并进来的是绑定在这个点位上的设备（point.device_id），
    # 不是「装在这儿」的设备 —— 装在哪和给谁显示是两个事实，别混。
    _dev_by_binding: dict[str, dict] = {}
    for _d in devices:
        if _d.get("live_point_id"):
            _dev_by_binding.setdefault(_d["live_point_id"], _d)
    _pending = data.get("pending_by_location") or {}
    point_rows = []
    _p_scope, _p_params = scope_clauses("", user)
    for r in query(
        "SELECT point_id, name, light_key, is_active FROM fg.point "
        "WHERE ward_id = %s"
        + "".join(" AND " + c for c in _p_scope)
        + " ORDER BY name",
        (ward, *_p_params),
    ):
        pid = str(r["point_id"])
        dev = _dev_by_binding.pop(pid, None)
        lk = r.get("light_key") or r["name"]
        point_rows.append(
            {
                "point_id": pid,
                "name": r["name"],
                "light_key": lk,
                # 画面地址 = 绑定设备推的那条流
                "stream_path": dev["stream_path"] if dev else None,
                "point_active": bool(r["is_active"]),
                "pending": _pending.get(lk, 0),
                "device_code": dev["device_code"] if dev else None,
                "device_active": dev["is_active"] if dev else None,
                "last_event_at": dev["last_event_at"] if dev else None,
            }
        )
    # 没有被任何点位显示的设备不能因为改版就消失 —— 剩下的按旧门灯渲染。
    orphan_devices = list(_dev_by_binding.values())

    # 筛选条上要写点位名。查不到（链接过期 / 点位被删）就原样显示 key ——
    # 不静默退回「全部」：URL 说在看某点位、列表却是整个病区，比空列表更难发现。
    # 多选时逗号顿号拼起来（「4F电梯口、409病房」），模板只负责渲染。
    _name_by_key = {r["light_key"]: r["name"] for r in point_rows}
    cur_point_labels = [_name_by_key.get(k, k) for k in points]
    cur_point_label = "、".join(cur_point_labels)

    ok = sum(1 for d in devices if d["is_active"])
    median = raw_kpi.get("median_response_s") or {}

    pending_today = sum(day_counts.get(s, 0) for s in ("PENDING", "ACKED", "ON_SCENE"))
    # 往期积压 = 不限日期的未处置总数 − 今天这 15。负数不该出现，兜一下。
    backlog = max(0, int(raw_kpi.get("unhandled") or 0) - pending_today)

    kpi = {
        "pending": pending_today,
        # 白板「今日告警」：窗口就是上面的 day_from…as_of（`overview` 的 KPI 口径），
        # 和护士长总览同一套；不再是班次窗口。 前这个键叫 shift_alerts。
        "day_alerts": raw_kpi.get("alerts", 0),
        "median_response_s": median.get("value"),
        "median_n": median.get("n", 0),
        "devices_ok": ok,
        "devices_total": len(devices),
    }

    # 工作区默认选中当前段的第一条：进页面右栏就有内容，护士不用先点一下才知道这儿能干活。
    # 显式给了 focus 就以它为准 —— 接完警那条会离开「待接警」，但工作区不该跟着跑掉，
    # 它要一直停在同一条事件上，把「接警 → 到场 → 完成处置」一路推完。
    # focus 来自 URL ⇒ 必须过区域判据，否则改一下 ?focus=<别区域UUID> 就能打开
    focus_event = _event_detail(_parse_uid(focus), ward, user) if focus else None
    if focus_event is None and rows:
        focus_event = _event_detail(_parse_uid(rows[0]["event_uid"]), ward, user)

    return _render(
        "nurse_today.html",
        _ctx(
            request,
            user,
            kpi=kpi,
            # 主区：当前分段的行 + 四段计数 + 当前分段键
            worklist=rows,
            queue_counts=queue_counts,
            cur_seg=seg,
            # 下钻链接的 disposition：把「段」翻译成「状态集合」（见 metrics.segment_disposition）
            go_disposition=segment_disposition(seg),
            # 门灯用的全量聚合（不受队列 LIMIT 截断）
            pending_by_location=data.get("pending_by_location") or {},
            devices=devices,
            # 状态带按点位渲染（见上面的注释）；orphan_devices 是没挂点位的老设备
            point_rows=point_rows,
            orphan_devices=orphan_devices,
            # 往期没处置的条数（首页只看今天，但不能假装它们不存在）
            backlog=backlog,
            now=as_of,
            focus_event=focus_event,
            # 点位筛选（灯角小按钮）：cur_points 空列表 = 没筛
            cur_points=points,
            cur_point_label=cur_point_label,
        ),
    )


@app.get("/nurse/camera/{point_id}", response_class=HTMLResponse)
def page_nurse_camera(point_id: str, request: Request):
    """点位画面页：整页该点位摄像头的大画面。"""
    user, denied = _guard(request, "nurse", "head_nurse")
    if denied:
        return denied

    ward = user.ward_id or ""
    point = _point_detail(point_id, ward, user)
    if point is None:
        # 非法 UUID / 不属于本病区 / 不存在，一律 404：
        # 403 会暴露「这个点位存在但你看不到」。
        return _error_page(request, 404, "没有找到这个点位")

    return _render(
        "nurse_camera.html",
        _ctx(
            request,
            user,
            point=point,
            # 左导航仍高亮「值班工作台」：点位页是它的子页
            active_nav="/nurse/today",
        ),
    )


def _head_camera_points(user: CurrentUser) -> list[dict]:
    """本病区的点位清单 + 每点位的未闭环条数，供画面页左侧竖列渲染。"""
    ward = user.ward_id or ""
    with _db() as conn:
        pending = pending_by_location(conn, ward, zone_ids=scoped_zones(user))

    zsql, zparams = _zone_and("p", user)
    rows = query(
        "SELECT p.point_id, p.name, p.light_key, p.is_active, "
        "       d.stream_path, d.device_code, d.is_active AS device_active "
        "FROM fg.point p LEFT JOIN fg.device d ON d.device_id = p.device_id "
        "WHERE p.ward_id = %s"
        + zsql
        + " ORDER BY p.name",
        (ward, *zparams),
    )
    out = []
    for r in rows:
        lk = r.get("light_key") or r["name"]
        out.append(
            {
                "point_id": str(r["point_id"]),
                "name": r["name"],
                "light_key": lk,
                # 六改：画面地址来自绑定设备；没绑 / 设备没填推流 ⇒ 空串 = 未配置
                "stream_path": r.get("stream_path") or "",
                "device_code": r.get("device_code"),
                "device_active": (bool(r["device_active"])
                                  if r.get("device_active") is not None else None),
                "is_active": bool(r["is_active"]),
                "pending": int(pending.get(lk, 0)),
            }
        )
    return out


def _safe_point_ids(raw: str | None) -> list[str]:
    """把 `?points=` 里的一串 id 洗成可直接拼回 URL 的安全集合。"""
    out: list[str] = []
    for tok in (raw or "").split(","):
        tok = tok.strip()
        if 0 < len(tok) <= 64 and all(c.isalnum() or c in "-_" for c in tok):
            out.append(tok)
        if len(out) >= 9:
            break
    return out


def _head_camera_page(request: Request, point_id: str, points: str | None = None):
    """单点位大画面（宫格里「单独查看 ↗」的去处，保留骨架叠加）。"""
    user, denied = _guard(request, "head_nurse")
    if denied:
        return denied

    ward = user.ward_id or ""
    cam = _point_detail(point_id, ward, user)
    if cam is None:
        # 找不到就 404，不回落到"第一个摄像头"：
        # 那会让人以为点的是别的点位，比错误更难发现。
        return _error_page(request, 404, "没有找到这个点位")

    back = "/head/camera"
    keep = _safe_point_ids(points)
    if keep:
        back += "?points=" + ",".join(keep)

    return _render(
        "nurse_camera.html",
        _ctx(
            request,
            user,
            point=cam,
            active_nav="/head/camera",
            camera_back_url=back,
            camera_back_label="返回画面墙",
            camera_alerts_base="/head/events",
            # 护士长这一页整天开着：只探状态，媒体流等人点「播放」再建。
            camera_autoplay=False,
        ),
    )


@app.get("/head/camera", response_class=HTMLResponse)
def page_head_camera(request: Request):
    """病区画面墙：左列点位（可多选），右侧最多 3×3 宫格。"""
    user, denied = _guard(request, "head_nurse")
    if denied:
        return denied
    return _render(
        "head_camera.html",
        _ctx(
            request,
            user,
            points=_head_camera_points(user),
            active_nav="/head/camera",
        ),
    )


@app.get("/head/camera/{point_id}", response_class=HTMLResponse)
def page_head_camera_point(point_id: str, request: Request, points: str | None = None):
    return _head_camera_page(request, point_id, points)


@app.get("/alerts", response_class=HTMLResponse)
def page_alerts(request: Request):
    user, denied = _guard(request, "nurse", "head_nurse")
    if denied:
        return denied

    listing = _page_events(request, user, sort="disposition", default_size=20,
                           default_today=True)

    # Tab 上的数字必须描述当前视图，不能是另一个口径。
    # 旧实现是「全病区、全时间、不含测试数据」的裸计数，和表格差三处：
    # ① 不看时间窗 → 表格默认今天时，Tab 还是历史累计；
    # ② 排除 is_test → 表格把测试数据算进来，Tab 不算；
    # ③ 固定用 v_event_fact → 和列表的 fg.event 不是同一套 join。
    # 改成直接复用列表那对 (where, params)，数字与点进去看到的行数逐条相等。
    frm, to_bound = listing["range"]
    counts: dict[str, Any] = {}
    try:
        where, cparams = build_event_filters(
            ward_id=user.ward_id,
            zone_ids=scoped_zones(user),
            frm=frm,
            to=to_bound,
        )
        rows = query(
            f"SELECT e.disposition_state, count(*) AS n {EVENT_FROM} "
            f"WHERE {where} GROUP BY e.disposition_state",
            cparams,
        )
        for r in rows:
            counts[r["disposition_state"]] = int(r["n"])
        counts["all"] = sum(counts.values())
    except Exception:  # noqa: BLE001 - 计数失败不该让列表页白屏
        counts = {}

    return _render(
        "alerts.html",
        _ctx(
            request,
            user,
            events=listing["events"],
            total=listing["total"],
            page=listing["page"],
            size=listing["size"],
            filters=listing["filters"],
            devices=_devices(user.ward_id, user),
            points=_point_list(user.ward_id, user),
            counts=counts,
            drawer_event=None,
        ),
    )


@app.get("/alerts/{event_uid}", response_class=HTMLResponse)
def page_alert_detail(request: Request, event_uid: str):
    user, denied = _guard(request, "nurse", "head_nurse")
    if denied:
        return denied

    event = _event_detail(_parse_uid(event_uid), user.ward_id, user)
    if event is None:
        return _error_page(request, 404, "事件不存在")

    # 抽屉是覆盖式的：关掉后要回到原来的列表，所以背景列表照样渲染。
    # 默认时间窗必须和 /alerts 一致 —— 否则关掉抽屉会突然多出一批历史事件。
    listing = _page_events(request, user, sort="disposition", default_size=20,
                           default_today=True)

    return _render(
        "alerts.html",
        _ctx(
            request,
            user,
            events=listing["events"],
            total=listing["total"],
            page=listing["page"],
            size=listing["size"],
            filters=listing["filters"],
            devices=_devices(user.ward_id, user),
            points=_point_list(user.ward_id, user),
            counts={},
            drawer_event=event,
        ),
    )


@app.get("/nurse/history", response_class=HTMLResponse)
def page_nurse_history(request: Request):
    user, denied = _guard(request, "nurse", "head_nurse")
    if denied:
        return denied

    # 用户拍板：「处置记录」的排列顺序和告警中心统一 ——
    # 处置态 → 级别 → 时间倒序。改前是纯时间倒序。
    # 护士长台账 /head/events 仍是纯 time_desc：用户只点名了告警中心和处置记录，
    # 那边是全院台账，没有「先做哪一条」的诉求，别顺手改。
    listing = _page_events(
        request, user, sort="disposition", default_size=20, assignee_id=user.user_id
    )
    return _render(
        "nurse_history.html",
        _ctx(
            request,
            user,
            # 每行补「响应用时」：这页的「处置人」永远是当前用户（SQL 已经按 assignee 过滤），
            # 那一列留着是噪音；换成只有个人台账才看得到的东西。
            events=_with_response_s(listing["events"]),
            total=listing["total"],
            page=listing["page"],
            size=listing["size"],
            filters=listing["filters"],
            # 方案 A：这页的定位是「我的个人台账」，顶部把口径和三个数写在明面上
            summary=_my_record_summary(listing["where"], listing["params"]),
        ),
    )


# ══════════════════════════════════════════════════════════════════════════
# 护士长端
# ══════════════════════════════════════════════════════════════════════════

@app.get("/head/overview", response_class=HTMLResponse)
def page_head_overview(request: Request):
    user, denied = _guard(request, "head_nurse")
    if denied:
        return denied

    ward = user.ward_id
    as_of = now_sh()
    day_from, _ = _day_window(as_of)

    with _db() as conn:
        data = metrics_overview(conn, ward, day_from, as_of)
        hours = metrics_hourly(conn, ward, as_of)
        # 「较昨日」：与 stats.kpi_block 同一条规则 —— 昨日一条告警都没有就不给差值
        yesterday = metrics_kpi(
            conn, ward, day_from - dt.timedelta(days=1), as_of - dt.timedelta(days=1)
        )

    _k = data.get("kpi") or {}
    yday_alerts = int(yesterday.get("alerts") or 0)
    kpi_ctx = {
        "alerts": {
            "value": _k.get("alerts"),
            "delta_vs_yesterday": (
                int(_k.get("alerts") or 0) - yday_alerts if yday_alerts > 0 else None
            ),
        },
        "confirmed_falls": {
            "value": _k.get("confirmed_falls"),
            "delta_vs_yesterday": (
                int(_k.get("confirmed_falls") or 0) - int(yesterday.get("confirmed_falls") or 0)
                if yday_alerts > 0
                else None
            ),
        },
        "false_alarm_rate": _k.get("false_alarm_rate") or {},
        "unhandled": {"value": _k.get("unhandled")},
        "median_response_s": _k.get("median_response_s") or {},
    }

    # 「今日撤回」= 确认之后又被模型否定的条数。
    # 排除孤立撤回（orphan_retract）：那种告警从没响过，不是"自我否定"，
    # 也正好让这张卡与 KPI-1（ever_confirmed）同一口径。
    retracted = query_one(
        """
        SELECT count(*) AS n FROM fg.v_event_fact
         WHERE ward_id = %s AND detection_state = 'RETRACTED'
           AND ever_confirmed AND received_at >= %s
        """,
        (ward, day_from),
    )
    handlers = query(
        """
        SELECT u.display_name, count(*) AS n
          FROM fg.v_event_fact e JOIN fg.app_user u ON u.user_id = e.assignee_id
         WHERE e.ward_id = %s AND e.received_at >= %s
         GROUP BY u.display_name ORDER BY n DESC LIMIT 6
        """,
        (ward, day_from),
    )

    return _render(
        "head_overview.html",
        _ctx(
            request,
            user,
            kpi=kpi_ctx,
            hourly=hours,
            pending_events=[_fix(e) for e in (data.get("pending_events") or [])],
            recent_handlers=[
                {"display_name": h["display_name"], "count": int(h["n"])} for h in handlers
            ],
            retracted_today=int(retracted["n"]) if retracted else None,
            as_of=as_of,
            period="today",
        ),
    )


@app.get("/head/events", response_class=HTMLResponse)
def page_head_events(request: Request):
    user, denied = _guard(request, "head_nurse")
    if denied:
        return denied

    listing = _page_events(request, user, sort="disposition", default_size=50)
    return _render(
        "head_events.html",
        _ctx(
            request,
            user,
            events=listing["events"],
            total=listing["total"],
            page=listing["page"],
            size=listing["size"],
            filters=listing["filters"],
            devices=_devices(user.ward_id, user),
            points=_point_list(user.ward_id, user),
            assignees=_assignee_list(user.ward_id),
        ),
    )


@app.get("/head/events/{event_uid}", response_class=HTMLResponse)
def page_head_event_detail(request: Request, event_uid: str):
    user, denied = _guard(request, "head_nurse")
    if denied:
        return denied

    event = _event_detail(_parse_uid(event_uid), user.ward_id, user)
    if event is None:
        return _error_page(request, 404, "事件不存在")

    # 证据链来自 audit_log（只追加）。列名必须与 head_event_detail.html 读的字段一致：
    # 模板读 row.at / row.result / row.detail / row.object_type / row.client_ip。
    # audit_log 没有 from_state / to_state —— 那是 event_action 的列，两张表别混。
    trail = [
        _fix(t)
        for t in query(
            """
            SELECT a.at, a.action, a.object_type, a.object_id, a.result,
                   host(a.client_ip) AS client_ip, a.detail,  -- INET 别 ::text，见 _audit_page
                   u.display_name AS actor_name, u.role AS actor_role
              FROM fg.audit_log a LEFT JOIN fg.app_user u ON u.user_id = a.actor_id
             WHERE a.object_type = 'event' AND a.object_id = %s
             ORDER BY a.at
            """,
            (str(event["event_uid"]),),
        )
    ]

    return _render(
        "head_event_detail.html",
        _ctx(
            request,
            user,
            event=event,
            audit_trail=trail,
            actions=event.get("actions") or [],
            back_url="/head/events",
        ),
    )


@app.get("/head/staff", response_class=HTMLResponse)
def page_head_staff(
    request: Request,
    u: int | None = None,
    notice: str | None = None,
    error: str | None = None,
):
    """人员与账号（ 起是一张表，不再是人员/账号两个 Tab）。"""
    user, denied = _guard(request, "head_nurse")
    if denied:
        return denied

    # 用户：「这个里面不能显示护士长」。
    # 只列值班护士 —— 这正好是本页每个按钮的权限边界，不是审美取舍：
    # routes/admin.py 的 _target_user 规定护士长只能操作「本病区的 nurse」，
    # 碰到 head_nurse / admin 一律 404。列出来只会给出点一下就报错的按钮。
    # （信息科管理员 ward_id 为 NULL，本来也进不了这条按病区的查询。）
    raw_rows = query(
        """
        SELECT au.user_id, au.login_name, au.display_name, au.role, au.is_active,
               au.must_change_password, au.failed_count, au.locked_until,
               au.last_login_at, au.created_at,
               c.display_name AS created_by_name
          FROM fg.app_user au
          LEFT JOIN fg.app_user c ON c.user_id = au.created_by
         WHERE au.ward_id = %s AND au.role = 'nurse'
         ORDER BY au.display_name
        """,
        (user.ward_id,),
    )
    # _user_item 只挑固定字段，created_by_name 得单独挂回去（档案抽屉的「创建人」）。
    rows = [_fix(_user_item(r)) for r in raw_rows]
    for row, raw in zip(rows, raw_rows):
        row["created_by_name"] = raw.get("created_by_name")

    zones = [
        {"zone_id": str(z["zone_id"]), "name": z["name"], "is_active": bool(z["is_active"])}
        for z in query(
            "SELECT zone_id, name, is_active FROM fg.zone WHERE ward_id = %s ORDER BY name",
            (user.ward_id,),
        )
    ]
    _zone_name = {z["zone_id"]: z["name"] for z in zones}
    _by_user: dict[int, list[str]] = {}
    for r in query(
        """
        SELECT uz.user_id, uz.zone_id
          FROM fg.app_user_zone uz
          JOIN fg.app_user u ON u.user_id = uz.user_id
         WHERE u.ward_id = %s
         ORDER BY uz.zone_id
        """,
        (user.ward_id,),
    ):
        _by_user.setdefault(int(r["user_id"]), []).append(str(r["zone_id"]))
    for row in rows:
        _ids = _by_user.get(int(row["user_id"]), [])
        row["zone_ids"] = _ids
        row["zone_names"] = [_zone_name.get(z, z) for z in _ids]

    day_from, _ = _day_window()
    week_from = day_from - dt.timedelta(days=6)
    stats = {
        int(r["user_id"]): r
        for r in query(
            """
            SELECT assignee_id AS user_id,
                   count(*) FILTER (WHERE acked_at >= %s)                  AS acked_today,
                   count(*) FILTER (WHERE is_closed AND resolved_at >= %s) AS closed_today,
                   count(*) FILTER (WHERE is_false_alarm AND resolved_at >= %s)
                                                                           AS false_today,
                   count(*) FILTER (WHERE acked_at >= %s)                  AS acked_week,
                   count(*) FILTER (WHERE is_closed AND resolved_at >= %s) AS closed_week,
                   count(*) FILTER (WHERE is_false_alarm AND resolved_at >= %s)
                                                                           AS false_week,
                   count(*) FILTER (WHERE disposition_state IN ('ACKED','ON_SCENE'))
                                                                           AS open_now,
                   percentile_cont(0.5) WITHIN GROUP (ORDER BY response_s)
                     FILTER (WHERE acked_at >= %s)                         AS median_today,
                   percentile_cont(0.5) WITHIN GROUP (ORDER BY response_s)
                     FILTER (WHERE acked_at >= %s)                         AS median_week
              FROM fg.v_event_fact
             WHERE ward_id = %s AND assignee_id IS NOT NULL
             GROUP BY assignee_id
            """,
            (day_from, day_from, day_from,
             week_from, week_from, week_from,
             day_from, week_from, user.ward_id),
        )
    }
    _now = now_sh()
    for row in rows:
        st = stats.get(int(row["user_id"])) or {}
        for key in ("acked_today", "closed_today", "false_today",
                    "acked_week", "closed_week", "false_week", "open_now"):
            row[key] = int(st.get(key) or 0)
        for key in ("median_today", "median_week"):
            v = st.get(key)
            row[key] = int(round(float(v))) if v is not None else None
        # 「已锁定」看的是此刻还在锁定期内：locked_until 过了就自动解锁，
        # 账号上的 failed_count 却不会自己清零，别按它判。
        row["locked"] = bool(row["locked_until"] and row["locked_until"] > _now)

    summary = {
        "total": len(rows),
        "active": sum(1 for r in rows if r["is_active"]),
        "acked_today": sum(r["acked_today"] for r in rows),
        "closed_today": sum(r["closed_today"] for r in rows),
        "open_now": sum(r["open_now"] for r in rows),
        # 「未配区域」只数启用的：已停用的账号本来就登录不了，
        # 把它们算进告警数会让这个数字永远降不到 0。
        "no_zone": sum(1 for r in rows if r["is_active"] and not r["zone_ids"]),
    }

    # ── 人员档案抽屉（?u=）───────────────────────────────────────────────────
    card = None
    if u is not None:
        card = next((r for r in rows if int(r["user_id"]) == int(u)), None)
    card_events: list[dict] = []
    ward_median: int | None = None
    if card:
        # 最近处置的 6 条。v_event_fact 没有位置列，回到 fg.event 取快照，
        # 点位名为空时用设备位置兜底（与台账、点位页同一套 COALESCE）。
        card_events = [
            _fix(r)
            for r in query(
                """
                SELECT e.event_uid, e.received_at, e.severity, e.detection_state,
                       e.disposition_state, e.location_text AS location
                  FROM fg.event e
                 WHERE e.ward_id = %s AND e.is_test = FALSE AND e.assignee_id = %s
                 ORDER BY e.received_at DESC
                 LIMIT 6
                """,
                (user.ward_id, card["user_id"]),
            )
        ]
        # 全病区今日中位响应：给档案里的「中位响应」当参照物。
        # 「48 秒」孤零零一个数字说明不了快慢，和同事比才有意义。
        _wm = query_one(
            """
            SELECT percentile_cont(0.5) WITHIN GROUP (ORDER BY response_s) AS median
              FROM fg.v_event_fact
             WHERE ward_id = %s AND assignee_id IS NOT NULL AND acked_at >= %s
            """,
            (user.ward_id, day_from),
        )
        if _wm and _wm["median"] is not None:
            ward_median = int(round(float(_wm["median"])))

    return _render(
        "head_staff.html",
        _ctx(
            request,
            user,
            accounts=rows,
            zones=zones,
            summary=summary,
            card_user=card,
            card_events=card_events,
            ward_median=ward_median,
            notice=notice,
            error=error,
            temp_password=None,
            new_user=None,
        ),
    )


@app.get("/head/zones", response_class=HTMLResponse)
def page_head_zones(request: Request):
    """区域值守：每个区域当前有谁负责，还有哪些点位没归区。"""
    user, denied = _guard(request, "head_nurse")
    if denied:
        return denied

    zone_rows = query(
        """
        SELECT z.zone_id, z.name, z.zone_type, z.floor, z.is_active,
               (SELECT count(*) FROM fg.point p WHERE p.zone_id = z.zone_id) AS point_n,
               -- 七改：设备不再直接归区域；区域里的设备 = 绑定了本区点位的设备
               (SELECT count(*) FROM fg.point lp
                 WHERE lp.zone_id = z.zone_id AND lp.device_id IS NOT NULL) AS device_n
          FROM fg.zone z
         WHERE z.ward_id = %s
         ORDER BY z.name
        """,
        (user.ward_id,),
    )
    # 负责护士反查（`ix_auz_zone` 就是为这一句建的）
    assignees: dict[str, list[dict]] = {}
    for r in query(
        """
        SELECT uz.zone_id, u.user_id, u.display_name, u.login_name, u.is_active
          FROM fg.app_user_zone uz
          JOIN fg.app_user u ON u.user_id = uz.user_id
         WHERE u.ward_id = %s AND u.role = 'nurse'
         ORDER BY u.display_name
        """,
        (user.ward_id,),
    ):
        assignees.setdefault(str(r["zone_id"]), []).append(
            {
                "user_id": int(r["user_id"]),
                "name": r["display_name"] or r["login_name"],
                "is_active": bool(r["is_active"]),
            }
        )
    zones = []
    for z in zone_rows:
        zid = str(z["zone_id"])
        zones.append(
            {
                "zone_id": zid,
                "name": z["name"],
                "zone_type": z["zone_type"],
                "floor": z.get("floor"),
                "is_active": bool(z["is_active"]),
                "point_n": int(z["point_n"] or 0),
                "device_n": int(z["device_n"] or 0),
                "assignees": assignees.get(zid, []),
            }
        )

    # 未归区点位：决策 4 的「补归」。护士在补归之前看不到它们（NULL = ANY 为 NULL），
    # 所以这一块不是提醒，是必须做掉的事。
    unzoned = [
        {
            "point_id": str(r["point_id"]),
            "name": r["name"],
            "light_key": r["light_key"],
            # 六改：画面地址 = 绑定设备推的流（没有绑定/没填推流 ⇒ 未配置）
            "stream_path": r.get("stream_path"),
            "is_active": bool(r["is_active"]),
        }
        for r in query(
            "SELECT p.point_id, p.name, p.light_key, p.is_active, d.stream_path "
            "FROM fg.point p LEFT JOIN fg.device d ON d.device_id = p.device_id "
            "WHERE p.ward_id = %s AND p.zone_id IS NULL ORDER BY p.name",
            (user.ward_id,),
        )
    ]

    # 没有任何区域的护士：默认拒绝，登录后什么都看不到 —— 比未归区点位更急。
    unassigned = [
        {
            "user_id": int(r["user_id"]),
            "name": r["display_name"] or r["login_name"],
            "login_name": r["login_name"],
            "is_active": bool(r["is_active"]),
        }
        for r in query(
            """
            SELECT u.user_id, u.display_name, u.login_name, u.is_active
              FROM fg.app_user u
             WHERE u.ward_id = %s AND u.role = 'nurse' AND u.is_active
               AND NOT EXISTS (SELECT 1 FROM fg.app_user_zone uz WHERE uz.user_id = u.user_id)
             ORDER BY u.display_name
            """,
            (user.ward_id,),
        )
    ]

    return _render(
        "head_zones.html",
        _ctx(request, user, zones=zones, unzoned=unzoned, unassigned=unassigned),
    )


def _point_rows(ward_id: str | None, user: CurrentUser | None = None) -> list[dict]:
    """点位台账 + 绑定设备的只读派生列（ 六改）。"""
    zsql, zparams = _zone_and("p", user)
    rows = query(
        """
        SELECT p.point_id, p.ward_id, p.zone_id, p.name, p.light_key,
               p.is_active, p.source_type, z.name AS zone_name,
               d.device_id, d.device_code, d.stream_path AS device_stream_path,
               d.is_active AS device_active, d.last_seen_at AS device_last_seen_at
          FROM fg.point p
          LEFT JOIN fg.zone z ON z.zone_id = p.zone_id
          LEFT JOIN fg.device d ON d.device_id = p.device_id
         WHERE p.ward_id = COALESCE(%s, p.ward_id)
        """ + zsql + """
         ORDER BY p.name
        """,
        (ward_id, *zparams),
    )
    return [
        {
            "point_id": str(r["point_id"]),
            "ward_id": r["ward_id"],
            "zone_id": str(r["zone_id"]) if r.get("zone_id") else None,
            "zone_name": r.get("zone_name"),
            "name": r["name"],
            "light_key": r["light_key"],
            "is_active": bool(r["is_active"]),
            "source_type": r.get("source_type") or "camera",
            # 这个点位显示哪台设备 + 它的流（没绑 / 没填推流 ⇒ None = 未配置）
            "device_id": str(r["device_id"]) if r.get("device_id") else None,
            "device_code": r.get("device_code"),
            "stream_path": r.get("device_stream_path"),
            "device_active": (bool(r["device_active"])
                              if r.get("device_active") is not None else None),
            "device_last_seen_at": _to_dt(iso(r["device_last_seen_at"])),
            # 点位卡用：3 分钟内有心跳 = 在线（与 /admin/devices 同口径）
            "online": bool(
                r.get("device_last_seen_at")
                and (dt.datetime.now(dt.timezone.utc)
                     - r["device_last_seen_at"]).total_seconds() < 180),
        }
        for r in rows
    ]


def _ward_groups(points: list[dict], zones: list[dict],
                 wards: list[dict]) -> list[dict]:
    """病区 → 区域 → 卡片 分桶。"""
    return ward_groups(points, zones, wards)


@app.get("/head/devices", response_class=HTMLResponse)
def page_head_devices(request: Request):
    user, denied = _guard(request, "head_nurse")
    if denied:
        return denied
    # 门灯要点亮就得知道每台设备积压多少条：全量聚合，不能拿台账前 N 页去凑。
    with _db() as conn:
        pending_by_location_map = pending_by_location(conn, user.ward_id)
        pending_by_device_map = pending_by_device(conn, user.ward_id)
        zones = [
            {
                "zone_id": str(r["zone_id"]),
                "name": r["name"],
                "zone_type": r["zone_type"],
                "floor": r.get("floor"),
                "is_active": bool(r["is_active"]),
            }
            for r in query(
                "SELECT zone_id, name, zone_type, floor, is_active FROM fg.zone "
                "WHERE ward_id = %s ORDER BY name",
                (user.ward_id,),
            )
        ]
        points = _point_rows(user.ward_id, user)
        devices = _devices(user.ward_id, user)
        # ?config=<device_id>：判定策略面板
        config_device = None
        config = None
        cfg_id = request.query_params.get("config")
        if cfg_id:
            config_device = next((d for d in devices if d["device_id"] == cfg_id), None)
            if config_device is not None:
                config = {
                    "desired": query_one(
                        "SELECT rev, params, note, updated_at FROM fg.device_config "
                        "WHERE device_id = %s", (cfg_id,)),
                    "revisions": query(
                        "SELECT rev, params, note, created_at FROM fg.device_config_rev "
                        "WHERE device_id = %s ORDER BY rev DESC LIMIT 10", (cfg_id,)),
                }
    return _render(
        "head_devices.html",
        _ctx(
            request,
            user,
            devices=devices,
            zones=zones,
            points=points,
            config_device=config_device,
            config=config,
            pending_by_location=pending_by_location_map,
            pending_by_device=pending_by_device_map,
        ),
    )


@app.get("/head/points", response_class=HTMLResponse)
def page_head_points(request: Request):
    """点位页（护士长）：点位 ↔ 绑定哪台设备。不碰设备台账（六改）。"""
    user, denied = _guard(request, "head_nurse")
    if denied:
        return denied
    with _db() as conn:
        pending_by_location_map = pending_by_location(conn, user.ward_id)
        # 十改：区域补上 `ward_id`，病区只传她自己那一个 —— 这样两条入口走的是
        # 同一个 `_ward_groups()`，模板里只有一份分组渲染逻辑。
        # 模板那边靠 `ward_layer`（= admin_mode）决定画不画病区标题：护士长只有
        # 一个病区，给她加个大标题是废话，用户明确不要。
        zones = [
            {
                "zone_id": str(r["zone_id"]),
                "ward_id": r["ward_id"],
                "name": r["name"],
                "zone_type": r["zone_type"],
                "floor": r.get("floor"),
                "is_active": bool(r["is_active"]),
            }
            for r in query(
                "SELECT zone_id, ward_id, name, zone_type, floor, is_active "
                "FROM fg.zone WHERE ward_id = %s ORDER BY name",
                (user.ward_id,),
            )
        ]
        points = _point_rows(user.ward_id, user)
        devices = _devices(user.ward_id, user)
        her_ward = _ward(user.ward_id)
    wards = ([{"ward_id": her_ward["ward_id"], "ward_name": her_ward["ward_name"],
               "floor": her_ward.get("floor")}] if her_ward else [])
    return _render(
        "head_points.html",
        _ctx(
            request,
            user,
            points=points,
            zones=zones,
            devices=devices,
            wards=wards,
            ward_groups=_ward_groups(points, zones, wards),
            pending_by_location=pending_by_location_map,
        ),
    )



@app.get("/admin/overview", response_class=HTMLResponse)
def page_admin_overview(request: Request):
    """信息科首页 = 运维台。"""
    user, denied = _guard(request, "admin")
    if denied:
        return denied

    now = now_sh()
    today = day_start(now)
    tomorrow = today + dt.timedelta(days=1)
    now_utc = dt.datetime.now(dt.timezone.utc)

    dev = query_one(
        """
        SELECT count(*) AS total,
               count(*) FILTER (WHERE is_active) AS active,
               -- is_active 必须一起判：板端不管云端有没有停用它都继续推心跳，
               -- 少这个条件，一台停用的板子会同时出现在 KPI「在线」里和健康墙的
               -- 灰灯上 —— 同屏两个答案，而且错的那个还更醒目。
               count(*) FILTER (WHERE is_active
                                  AND last_seen_at > now() - interval '3 minutes') AS online
          FROM fg.device
        """
    ) or {}
    usr = query_one(
        """
        SELECT count(*) AS total,
               count(*) FILTER (WHERE is_active) AS active,
               count(*) FILTER (WHERE is_active AND must_change_password) AS must_change,
               count(*) FILTER (WHERE is_active AND locked_until IS NOT NULL
                                  AND locked_until > now()) AS locked
          FROM fg.app_user
        """
    ) or {}
    devices = query(
        """
        SELECT d.device_id, d.device_code,
               -- 设备名可为 NULL ⇒ 首页设备表回落设备码（否则模板会印出 "None"）
               COALESCE(d.device_name, d.device_code) AS device_name,
               -- 八改：病区**只认绑定的点位**，不再读 d.ward_id。
               -- 七改后「装在哪」的唯一真源是 point.device_id；d.ward_id 是登记/权限
               -- 字段。拿它当病区印出来，就会出现「没绑点位却有病区」这种行
               -- （现网 P12-DRILL-01：ward_id=4F-A，点位 NULL）。
               -- 没绑点位 = 没有位置，显示「—」，不回落任何副本。
               lp.ward_id AS point_ward_id,
               lw.ward_name AS point_ward_name,
               lp.name AS point_name,
               d.is_active, d.last_seen_at, d.last_seen_ip::text AS last_seen_ip,
               d.last_policy_rev, d.last_status,
               c.rev AS desired_rev
          FROM fg.device d
          """ + LIVE_POINT_JOIN + """
          LEFT JOIN fg.ward lw ON lw.ward_id = lp.ward_id
          LEFT JOIN fg.device_config c ON c.device_id = d.device_id
         ORDER BY lw.ward_name NULLS LAST, d.device_code
        """
    )
    hourly_rows = query(
        """
        SELECT EXTRACT(HOUR FROM received_at AT TIME ZONE 'Asia/Shanghai')::int AS hour,
               count(*) AS count
          FROM fg.v_event_fact
         WHERE received_at >= %s AND received_at < %s
         GROUP BY 1
        """,
        (today, tomorrow),
    )
    today_row = query_one(
        "SELECT count(*) AS n FROM fg.v_event_fact "
        "WHERE received_at >= %s AND received_at < %s",
        (today, tomorrow),
    ) or {}
    health = _health_snapshot()

    # 八改：数据库实况整块下线（用户：「这部分功能整体去掉」）⇒ 五条统计查询一并删：
    # count(*) / pg_total_relation_size / pg_database_size / max(received_at) / 今日审计。
    # 只留「累计事件」KPI 用的 fact_total —— 那是告警口径，不是数据库面板的。
    fact_total = query_one("SELECT count(*) AS n FROM fg.v_event_fact") or {}

    # ── 设备行（在线判定 3 分钟，与 /admin/devices 同口径） ────────────────
    device_rows = []
    for r in devices:
        seen = _to_dt(iso(r.get("last_seen_at")))
        st = r.get("last_status") if isinstance(r.get("last_status"), dict) else {}
        online = bool(seen and (now_utc - seen).total_seconds() < 180)
        ip = r.get("last_seen_ip")
        if ip:
            ip = ip.split("/")[0]  # inet 文本可能带 /32
        # 板端遥测（ 四改）：心跳里新加的 CPU/内存/温度，有就显示、没有就不显示。
        # 老固件的心跳没有这些键 ⇒ tele_text 为空，设备名下面不会多出一行空占位。
        tele = []
        tele_title = []
        for key, label, unit in (("cpu_pct", "CPU", "%"), ("mem_pct", "内存", "%"),
                                 ("temp_c", "温度", "°C"), ("npu_load", "NPU", "%")):
            val = st.get(key)
            if val is None or isinstance(val, bool):
                continue
            try:
                num = float(val)
            except (TypeError, ValueError):
                continue
            tele_title.append(f"{label} {num:.0f}{unit}")
            if key != "npu_load":  # NPU 只在 tooltip 里，避免把设备名那格撑爆
                tele.append(f"{label} {num:.0f}{unit}")
        device_rows.append({
            **r,
            "device_id": str(r["device_id"]),
            "last_seen_at": seen,
            "last_seen_ip": ip,
            "online": online,
            "fps": st.get("fps"),
            "uptime_s": st.get("uptime_s"),
            "tele_text": " · ".join(tele),
            # 二十改：原来写「板端最近一次心跳上报」（首页运维台悬停提示）—— 用户
            # 「不要反复提云端和板端，正常一点」⇒ 换「设备」。口径与 `_is_online()` 同源。
            "tele_title": ("设备最近一次心跳上报：" + " · ".join(tele_title)) if tele_title else "",
            "policy_state": _policy_state(
                {"last_policy_rev": r.get("last_policy_rev"), "last_status": st},
                None if r.get("desired_rev") is None else {"rev": r["desired_rev"]},
            ),
        })

    # 设备实况：不活（离线/停用）的排前面 —— 这张表的用途是扫一眼看出哪台不对。
    # 八改：同组内按点位的病区 + 点位名排（病区列印的就是点位的病区，两列同一口径）；
    # 没绑点位的设备没有位置可归组 ⇒ 排在同组末尾，不按登记病区硬塞进某个病区。
    device_list = sorted(
        device_rows,
        key=lambda d: (d["is_active"] and d["online"],
                       d["point_ward_name"] is None,
                       d["point_ward_name"] or "",
                       d["point_name"] or "",
                       d["device_code"]))

    locked = int(usr.get("locked") or 0)
    must_change = int(usr.get("must_change") or 0)

    # ── KPI ───────────────────────────────────────────────────────────────
    configured = [d for d in device_rows if d["policy_state"] != "unconfigured"]
    applied = [d for d in configured if d["policy_state"] == "applied"]
    buckets = {int(r["hour"]): int(r["count"]) for r in hourly_rows if r["hour"] is not None}
    hourly = [{"hour": h, "count": buckets.get(h, 0)} for h in range(24)]

    return _render(
        "admin_overview.html",
        _ctx(
            request,
            user,
            dev_total=int(dev.get("total") or 0),
            dev_active=int(dev.get("active") or 0),
            dev_online=int(dev.get("online") or 0),
            user_total=int(usr.get("total") or 0),
            user_locked=locked,
            user_must_change=must_change,
            # 设备实况表用排过序的 device_list（八改后本页只剩三段）。
            device_list=device_list,
            fact_total=int(fact_total.get("n") or 0),
            hourly=hourly,
            today_events=int(today_row.get("n") or 0),
            health=health,
            sysmon=sysmon.snapshot(),
            strategy_applied=len(applied),
            strategy_total=len(configured),
            as_of=now,
        ),
    )


@app.get("/admin/devices", response_class=HTMLResponse)
def page_admin_devices(request: Request):
    """设备 / 点位 / 区域 / 判定策略（信息科管理员，全院）。"""
    user, denied = _guard(request, "admin")
    if denied:
        return denied
    with _db() as conn:
        devices = _devices(None, user)
        zones = [
            {
                "zone_id": str(r["zone_id"]), "ward_id": r["ward_id"],
                "name": r["name"], "zone_type": r["zone_type"],
                "floor": r.get("floor"), "is_active": bool(r["is_active"]),
            }
            for r in query(
                "SELECT zone_id, ward_id, name, zone_type, floor, is_active "
                "FROM fg.zone ORDER BY ward_id, name")
        ]
        wards = [
            {"ward_id": r["ward_id"], "ward_name": r["ward_name"]}
            for r in query("SELECT ward_id, ward_name FROM fg.ward ORDER BY ward_id")
        ]
        # 二改（用户拍板「方案 A」）：本页 = 设备登记簿。
    # ?edit= 对所有设备都有效（不再按有没有绑定点位分裂）；
    # ?config= 判定策略属于点位详情，一律 302 过去（未绑定设备没有策略入口）。
    cfg_id = request.query_params.get("config")
    if cfg_id:
        _target = next((d for d in devices if d["device_id"] == cfg_id), None)
        if _target and _target.get("live_point_id"):
            return RedirectResponse(
                "/admin/points/%s?config=%s" % (_target["live_point_id"], cfg_id),
                status_code=302)
        return RedirectResponse("/admin/devices", status_code=302)
    return _render(
        "head_devices.html",
        _ctx(
            request,
            user,
            admin_mode=True,
            devices=devices,
            zones=zones,
            wards=wards,
            device_groups=device_ward_groups(devices, zones, wards),
            pending_by_location={},
            pending_by_device={},
        ),
    )


@app.get("/admin/points", response_class=HTMLResponse)
def page_admin_points(request: Request):
    """点位页（信息科）：点位 ↔ 绑定哪台设备，全院。不碰设备台账（六改）。"""
    user, denied = _guard(request, "admin")
    if denied:
        return denied
    return _render("head_points.html", _ctx(request, user, admin_mode=True, **_points_page_ctx()))


def _points_page_ctx() -> dict[str, Any]:
    """`/admin/points` 与 `/admin/points/<id>` 共用的一屏数据。"""
    points = _point_rows(None, None)
    devices = _devices(None, None)
    zones = [
        {
            "zone_id": str(r["zone_id"]), "ward_id": r["ward_id"],
            "name": r["name"], "zone_type": r["zone_type"],
            "floor": r.get("floor"), "is_active": bool(r["is_active"]),
        }
        for r in query(
            "SELECT zone_id, ward_id, name, zone_type, floor, is_active "
            "FROM fg.zone ORDER BY ward_id, name")
    ]
    wards = [
        {"ward_id": r["ward_id"], "ward_name": r["ward_name"],
         "floor": r.get("floor")}
        for r in query(
            "SELECT ward_id, ward_name, floor FROM fg.ward ORDER BY ward_id")
    ]
    return {
        "points": points,
        "devices": devices,
        "zones": zones,
        "wards": wards,
        # 十改：病区 → 区域 → 卡片 三层分组，在 Python 里分好再交给模板。
        "ward_groups": _ward_groups(points, zones, wards),
        "pending_by_location": {},
    }


@app.get("/admin/points/{point_id}", response_class=HTMLResponse)
def page_admin_point_detail(request: Request, point_id: str):
    """点位详情（信息科）：和列表同一张模板，抽屉展开。"""
    user, denied = _guard(request, "admin")
    if denied:
        return denied
    # 旧链接兼容：设备档案编辑在 二改里移回设备台账（方案 A）
    _edit_id = request.query_params.get("edit")
    if _edit_id:
        return RedirectResponse("/admin/devices?edit=%s" % _edit_id, status_code=302)
    page_ctx = _points_page_ctx()
    point = next((p for p in page_ctx["points"] if p["point_id"] == point_id), None)
    if point is None:
        raise HTTPException(status_code=404, detail="点位不存在")
    device = next((d for d in page_ctx["devices"]
                   if d["device_id"] == point.get("device_id")), None)
    config_device = None
    config = None
    if device is not None:
        # 在线判定：3 分钟内有心跳（与 /admin/devices 同口径）
        now_utc = dt.datetime.now(dt.timezone.utc)
        seen = device.get("last_seen_at")
        device["online"] = bool(seen and (now_utc - seen).total_seconds() < 180)
        # 期望版本/历史常查（面板顶部直接显示 rev 状态，不能只在 ?config 时才查）
        config = {
            "desired": query_one(
                "SELECT rev, params, note, updated_at FROM fg.device_config "
                "WHERE device_id = %s", (device["device_id"],)),
            "revisions": query(
                "SELECT rev, params, note, created_at FROM fg.device_config_rev "
                "WHERE device_id = %s ORDER BY rev DESC LIMIT 10", (device["device_id"],)),
        }
        if request.query_params.get("config") == device["device_id"]:
            config_device = device
    return _render(
        "head_points.html",
        _ctx(
            request,
            user,
            admin_mode=True,
            manage_point=point,
            config_device=config_device,
            config=config,
            **page_ctx,
        ),
    )


AUDIT_SCOPE_ACTIONS: dict[str, tuple[str, ...]] = {
    "manage": (
        "USER_CREATE", "USER_UPDATE", "USER_ENABLE", "USER_DISABLE", "USER_RESET",
        "USER_ZONE_SET", "PASSWORD_RESET", "PASSWORD_CHANGE",
        "DEVICE_CREATE", "DEVICE_UPDATE", "DEVICE_ENABLE", "DEVICE_DISABLE",
        "DEVICE_FOLLOW_STREAM", "POINT_CREATE", "POINT_UPDATE",
        "ZONE_CREATE", "ZONE_UPDATE", "WARD_CREATE", "WARD_UPDATE",
        "CONFIG_SAVE", "CONFIG_ROLLBACK",
    ),
    "disposition": ("ACK", "ARRIVE", "RESOLVE", "FALSE_ALARM", "REOPEN", "CLOSE"),
    "denied": ("DENIED",),
    "system": ("LOGIN", "LOGIN_FAIL", "LOGOUT", "SNAPSHOT_VIEW", "CLIP_VIEW",
               "EXPORT", "INGEST", "CONFIG_APPLY", "CONFIG_APPLY_FAIL"),
}
# 顺序 = 从「人干的事」到「机器留的痕」；all 永远排第一。
AUDIT_SCOPES: tuple[tuple[str, str, tuple[str, ...] | None], ...] = (
    ("all", "全部", None),
    ("manage", "管理动作", AUDIT_SCOPE_ACTIONS["manage"]),
    ("disposition", "处置记录", AUDIT_SCOPE_ACTIONS["disposition"]),
    ("denied", "越权拦截", AUDIT_SCOPE_ACTIONS["denied"]),
    ("system", "系统留痕", AUDIT_SCOPE_ACTIONS["system"]),
)

# 时间范围快捷项（值 = 含今天在内往前数几天）。日期优先：两个日期都空时它才生效。
AUDIT_RANGES: dict[str, int] = {"today": 1, "7d": 7, "30d": 30}
AUDIT_RANGE_OPTIONS = (("today", "今天"), ("7d", "近 7 天"), ("30d", "近 30 天"))

_AUDIT_DETAIL_NAMES = {
    "session": ("login_name",),
    "device": ("device_name", "device_code"),
    "point": ("name",),
    "zone": ("name",),
    "ward": ("ward_name",),
}


def _audit_object_labels(rows: list[dict]) -> dict[tuple[str, str], dict]:
    """把审计行的「对象」翻成人话：``{(object_type, object_id): {"text", "missing"}}``。"""
    ids: dict[str, set[str]] = {}
    for r in rows:
        t, oid = r["object_type"], r["object_id"]
        if not oid or t in ("session", "endpoint"):
            continue
        ids.setdefault(t, set()).add(str(oid))

    live: dict[tuple[str, str], str] = {}
    lookups = {
        "user": ("SELECT user_id::text AS k, display_name AS v FROM fg.app_user "
                 "WHERE user_id::text = ANY(%s)", "display_name"),
        "device": ("SELECT device_id::text AS k, device_name AS v, device_code FROM fg.device "
                   "WHERE device_id::text = ANY(%s)", "device"),
        "point": ("SELECT point_id::text AS k, name AS v FROM fg.point "
                  "WHERE point_id::text = ANY(%s)", None),
        "zone": ("SELECT zone_id::text AS k, name AS v FROM fg.zone "
                 "WHERE zone_id::text = ANY(%s)", None),
        "ward": ("SELECT ward_id AS k, ward_name AS v FROM fg.ward "
                 "WHERE ward_id = ANY(%s)", None),
        "event": ("SELECT event_uid::text AS k, location_text AS v FROM fg.event "
                  "WHERE event_uid::text = ANY(%s)", None),
    }
    for t, wanted in ids.items():
        # snapshot / clip 的 object_id 也是事件号，走同一张 event 表。
        sql_t = "event" if t in ("snapshot", "clip") else t
        if sql_t not in lookups:
            continue
        sql, _ = lookups[sql_t]
        for row in query(sql, (sorted(wanted),)):
            live[(t, row["k"])] = row.get("v")

    out: dict[tuple[str, str], dict] = {}
    for r in rows:
        t = r["object_type"]
        oid = "" if r["object_id"] is None else str(r["object_id"])
        key = (t, oid)
        if key in out:
            continue
        detail = r.get("detail") if isinstance(r.get("detail"), dict) else {}

        if t == "endpoint":
            out[key] = {"text": oid or "—", "ref": None, "missing": False}
            continue
        if t == "user":
            name, login = live.get((t, oid)), detail.get("login_name")
            text = f"{name}（{login}）" if name and login else (name or login)
        elif t in ("event", "snapshot", "clip"):
            # 事件类：库里只有 location_text（如「409病房」），一行一个地点根本分不出是哪一条
            # ⇒ 把 uid 也带上（模板用 `event_ref` 渲成「8 位前缀-末 4 位」，与事件台账同款）。
            # 「查不到」只能按这一行在不在 event 表里判，不能按 location_text 是否为空：
            # 地点为空的事件是存在的，按 text 判会把它误标成「已不存在」。
            present = (t, oid) in live
            loc = live.get((t, oid))
            text = (loc or "—") if present else None
            out[key] = {"text": text or (oid[:8] + "…" if oid else "—"),
                        "ref": oid if present and oid else None,
                        "missing": not present}
            continue
        else:
            text = next((detail[k] for k in _AUDIT_DETAIL_NAMES.get(t, ()) if detail.get(k)), None)
            text = text or live.get((t, oid))
        out[key] = {"text": text or (oid[:8] + "…" if oid else "—"),
                    "ref": None, "missing": text is None}
    return out


def _audit_page(request: Request, user: CurrentUser):
    """审计页（全院； 起只有信息科管理员能进）。"""
    qp = request.query_params
    # 非 action 层：时间 / 操作人 / 结果 / 搜索。快捷片的计数要用这一层（去掉 action），
    # 这样片上的数字才会跟着「今天 / 近 7 天」变 ——「今天 12 条越权拦截」才有意义。
    base_clauses = ["TRUE"]
    base_params: list[Any] = []

    if qp.get("actor"):
        try:
            base_clauses.append("a.actor_id = %s")
            base_params.append(int(qp["actor"]))
        except ValueError:
            pass
    if qp.get("result"):
        base_clauses.append("a.result = %s")
        base_params.append(qp["result"])
    # 时间：具体日期优先。两个日期都空时「今天 / 近 7 天 / 近 30 天」才生效，
    # 一旦给了日期就把 range 清掉（否则「选了今天又填了日期」两套条件互相打架）。
    raw_from = (qp.get("from") or "").strip()
    raw_to = (qp.get("to") or "").strip()
    range_sel = (qp.get("range") or "").strip()
    if not raw_from and not raw_to and range_sel in AUDIT_RANGES:
        _today = now_sh().date()
        raw_from = (_today - dt.timedelta(days=AUDIT_RANGES[range_sel] - 1)).isoformat()
        raw_to = _today.isoformat()
    elif raw_from or raw_to:
        range_sel = ""

    if raw_from:
        try:
            base_clauses.append("a.at >= %s")
            base_params.append(parse_time_param(raw_from))
        except ValueError:
            pass
    if raw_to:
        try:
            base_clauses.append("a.at < %s")
            base_params.append(parse_time_param(raw_to, end_of_day=True))
        except ValueError:
            pass
    if qp.get("q"):
        base_clauses.append("(a.object_id ILIKE %s OR a.action ILIKE %s)")
        like = f"%{qp['q']}%"
        base_params.extend([like, like])

    # 动作 / 范围：两者互斥。手选了具体动作就不再叠加范围（两层过滤叠加没人算得清），
    # 此时快捷片全部不高亮 —— scope 传空串给模板。
    action_sel = (qp.get("action") or "").strip()
    scope_sel = (qp.get("scope") or "all").strip()
    act_clause = ""
    act_params: list[Any] = []
    if action_sel:
        actions = [a for a in action_sel.split(",") if a]
        if actions:
            act_clause = "a.action = ANY(%s)"
            act_params = [actions]
            scope_sel = ""
    elif scope_sel in AUDIT_SCOPE_ACTIONS:
        act_clause = "a.action = ANY(%s)"
        # 必须 `list(...)`：`AUDIT_SCOPE_ACTIONS` 里存的是 tuple，而 psycopg 把 tuple
        # 当复合类型适配（不是数组）⇒ `= ANY(%s)` 直接 500。上面 action 那条路是
        # `[... for ... in split()]` 生成的 list 所以才没炸 —— 线上验收抓到：
        # 五个快捷片全都链到 `?scope=xxx`，点哪个都 500（本地纯 Jinja 夹具看不见 SQL）。
        act_params = [list(AUDIT_SCOPE_ACTIONS[scope_sel])]
    else:
        # 未知/写错的 scope 一律当「全部」—— 只读页不该因为一个参数拼错就 400 挡人。
        scope_sel = "all"

    clauses = base_clauses + ([act_clause] if act_clause else [])
    where = " AND ".join(clauses)
    params = base_params + act_params

    try:
        page = max(1, int(qp.get("page") or 1))
    except ValueError:
        page = 1
    size = 50

    logs = query(
        f"""
        SELECT a.audit_id, a.at, a.action, a.object_type, a.object_id, a.result,
               -- ⚠️ 必须 `host()` 不能 `::text`：client_ip 是 INET，::text 会把掩码一起带出来，
               --    页面 IP 列就印成「127.0.0.1/32」（真数据截图抓到的）。
               host(a.client_ip) AS client_ip, a.detail,
               a.actor_id, u.display_name AS actor_name, u.role AS actor_role
          FROM fg.audit_log a LEFT JOIN fg.app_user u ON u.user_id = a.actor_id
         WHERE {where}
         ORDER BY a.at DESC LIMIT %s OFFSET %s
        """,
        (*params, size, (page - 1) * size),
    )
    total_row = query_one(
        f"SELECT count(*) AS n FROM fg.audit_log a WHERE {where}", tuple(params)
    )

    # 快捷片计数：一条 SQL 里 count(*) FILTER，只看 base 层（不叠 action/scope）。
    # psycopg 按 SQL 文本顺序绑参 ⇒ FILTER 的参数排在 WHERE 的参数前面。
    scope_keys = list(AUDIT_SCOPE_ACTIONS)
    scope_sql = (
        "SELECT count(*) AS n_all, "
        + ", ".join(f"count(*) FILTER (WHERE a.action = ANY(%s)) AS n_{k}" for k in scope_keys)
        + f" FROM fg.audit_log a WHERE {' AND '.join(base_clauses)}"
    )
    scope_row = query_one(
        scope_sql, tuple([list(AUDIT_SCOPE_ACTIONS[k]) for k in scope_keys] + base_params)
    ) or {}
    scope_counts = {"all": int(scope_row.get("n_all") or 0)}
    scope_counts.update({k: int(scope_row.get(f"n_{k}") or 0) for k in scope_keys})

    logs = [_fix(r) for r in logs]
    labels = _audit_object_labels(logs) if logs else {}
    for r in logs:
        r["obj"] = labels.get((r["object_type"], "" if r["object_id"] is None else str(r["object_id"])))

    # 「操作人」下拉：全院（管理员看两个病区 + 信息科自己）
    actors = query(
        "SELECT user_id, display_name FROM fg.app_user ORDER BY display_name")

    # 抽屉：只有真的要看详情时才去查那两张映射表和这条记录本身。
    card_row = None
    zone_names: dict[str, str] = {}
    ward_names: dict[str, str] = {}
    log_id = qp.get("log")
    if log_id:
        try:
            lid = int(log_id)
        except ValueError:
            lid = None
        if lid is not None:
            card_row = query_one(
                """
                SELECT a.audit_id, a.at, a.action, a.object_type, a.object_id, a.result,
                       host(a.client_ip) AS client_ip, a.detail,  -- INET 别 ::text，见列表查询那条
                       a.actor_id, u.display_name AS actor_name, u.role AS actor_role
                  FROM fg.audit_log a LEFT JOIN fg.app_user u ON u.user_id = a.actor_id
                 WHERE a.audit_id = %s
                """,
                (lid,),
            )
        if card_row:
            card_row = _fix(card_row)
            card_row["obj"] = _audit_object_labels([card_row]).get(
                (card_row["object_type"],
                 "" if card_row["object_id"] is None else str(card_row["object_id"]))
            )
            zone_names = {r["k"]: r["v"] for r in query(
                "SELECT zone_id::text AS k, name AS v FROM fg.zone")}
            ward_names = {r["ward_id"]: r["ward_name"] for r in query(
                "SELECT ward_id, ward_name FROM fg.ward")}

    return _render(
        "admin_audit.html",
        _ctx(
            request,
            user,
            logs=logs,
            total=int(total_row["n"]) if total_row else None,
            page=page,
            size=size,
            scopes=AUDIT_SCOPES,
            scope=scope_sel,
            scope_counts=scope_counts,
            range_options=AUDIT_RANGE_OPTIONS,
            filters={
                # 用解析后的日期回填输入框：选了「近 7 天」时用户能看见到底查的是哪一段。
                "from": raw_from,
                "to": raw_to,
                "range": range_sel,
                "actor": qp.get("actor") or "",
                "action": action_sel,
                "result": qp.get("result") or "",
                "q": qp.get("q") or "",
            },
            actors=actors,
            card_row=card_row,
            zone_names=zone_names,
            ward_names=ward_names,
        ),
    )


@app.get("/admin/accounts", response_class=HTMLResponse)
def page_admin_accounts(
    request: Request,
    u: int | None = None,
    acc_new: str | None = None,
):
    """账号管理（信息科管理员，全院）。"""
    user, denied = _guard(request, "admin")
    if denied:
        return denied

    raw_rows = query(
        """
        SELECT u.user_id, u.login_name, u.display_name, u.role, u.ward_id,
               u.is_active, u.must_change_password, u.failed_count, u.locked_until,
               u.last_login_at, u.created_at,
               w.ward_name, w.floor AS ward_floor,
               c.display_name AS created_by_name
          FROM fg.app_user u
          LEFT JOIN fg.ward w ON w.ward_id = u.ward_id
          LEFT JOIN fg.app_user c ON c.user_id = u.created_by
         ORDER BY u.role, u.ward_id NULLS FIRST, u.user_id
        """
    )
    # _user_item 只挑固定字段，病区名/创建人也得单独挂回去（抽屉要用）。
    rows = [_fix(_user_item(r)) for r in raw_rows]
    _now = now_sh()
    for row, raw in zip(rows, raw_rows):
        row["ward_name"] = raw.get("ward_name")
        row["ward_floor"] = raw.get("ward_floor")
        row["created_by_name"] = raw.get("created_by_name")
        # 「已锁定」看的是此刻还在锁定期内：locked_until 过了就自动解锁，
        # 而 failed_count 不会自己清零，别按它判（与 /head/staff 同口径）。
        row["locked"] = bool(row["locked_until"] and row["locked_until"] > _now)

    _zone_name = {str(z["zone_id"]): z["name"] for z in query("SELECT zone_id, name FROM fg.zone")}
    _by_user: dict[int, list[str]] = {}
    for r in query("SELECT user_id, zone_id FROM fg.app_user_zone ORDER BY zone_id"):
        _by_user.setdefault(int(r["user_id"]), []).append(str(r["zone_id"]))
    for row in rows:
        _ids = _by_user.get(int(row["user_id"]), [])
        row["zone_ids"] = _ids
        row["zone_names"] = [_zone_name.get(z, z) for z in _ids]

    zones = [
        {
            "zone_id": str(z["zone_id"]),
            "ward_id": z["ward_id"],
            "name": z["name"],
            "is_active": bool(z["is_active"]),
        }
        for z in query("SELECT zone_id, ward_id, name, is_active FROM fg.zone ORDER BY ward_id, name")
    ]
    wards = query("SELECT ward_id, ward_name, floor FROM fg.ward ORDER BY ward_id")

    card = next((r for r in rows if r["user_id"] == int(u)), None) if u is not None else None

    return _render(
        "admin_accounts.html",
        _ctx(
            request,
            user,
            users=rows,
            wards=wards,
            zones=zones,
            # card_user 优先于 acc_new：两个参数同时给时以抽屉内容为准
            card_user=card,
            new_account=bool(acc_new) and card is None,
        ),
    )


@app.get("/admin/audit", response_class=HTMLResponse)
def page_admin_audit(request: Request):
    user, denied = _guard(request, "admin")
    if denied:
        return denied
    return _audit_page(request, user)




@app.get("/settings", response_class=HTMLResponse)
def page_settings(request: Request, error: str | None = None):
    """改密页（无 JS 兜底 + 头像菜单那行的 href 落点）。"""
    user, denied = _guard(request)
    if denied:
        return denied
    return _render("settings.html", _ctx(request, user, error=error))


# ══════════════════════════════════════════════════════════════════════════
# 健康检查
# ══════════════════════════════════════════════════════════════════════════

def _health_snapshot() -> dict:
    """系统健康快照。"""
    db_ok = "fail"
    events_today = None
    last_event_at = None
    try:
        query_one("SELECT 1 AS ok")
        db_ok = "ok"
        today = dt.datetime.combine(now_sh().date(), dt.time.min, tzinfo=now_sh().tzinfo)
        row = query_one(
            """
            SELECT count(*) AS n, max(received_at) AS t
              FROM fg.v_event_fact WHERE received_at >= %s
            """,
            (today,),
        )
        if row:
            events_today = int(row["n"])
            last_event_at = iso(row["t"])
    except Exception as exc:  # noqa: BLE001
        db_ok = f"fail: {type(exc).__name__}"

    schema_ok = None
    try:
        query_one("SELECT ever_confirmed FROM fg.v_event_fact LIMIT 0")
        query_one("SELECT device_id, device_name, stream_path FROM fg.device LIMIT 0")
        query_one("SELECT location_text, zone_id FROM fg.event LIMIT 0")
        query_one("SELECT zone_id FROM fg.zone LIMIT 0")
        # 点位摄像头 → 六改：点位改成绑定设备（point.device_id），
        # 漏跑 patch_2026_11_05b_point_device.sql 时 /nurse/camera/*、/head/points、
        # /admin/points 全 500 —— 让它在 /healthz 先可见。
        query_one("SELECT device_id, source_type FROM fg.point LIMIT 0")
        query_one("SELECT user_id, zone_id FROM fg.app_user_zone LIMIT 0")
        query_one("SELECT zone_id FROM fg.v_event_fact LIMIT 0")
        query_one("SELECT zone_id FROM fg.point LIMIT 0")
        schema_ok = "ok"
    except Exception as exc:  # noqa: BLE001
        schema_ok = f"fail: {type(exc).__name__}"

    try:
        free_gb = round(shutil.disk_usage("/").free / 1024**3, 1)
    except Exception:  # noqa: BLE001 - Windows 开发机没有 /
        free_gb = None

    # SSE 在线连接数 = 订阅队列数（events.py 用 dict[Queue, ward_id] 记账）
    queues = getattr(events_routes, "_sse_queues", None)
    return {
        "status": "ok" if db_ok == "ok" and schema_ok in (None, "ok") else "degraded",
        "db": db_ok,
        "schema": schema_ok,
        "sse_conns": len(queues) if queues is not None else None,
        "last_event_at": last_event_at,
        "events_today": events_today,
        "disk_free_gb": free_gb,
    }


@app.get("/healthz")
def healthz():
    """原样返回健康快照。"""
    return _health_snapshot()


# ══════════════════════════════════════════════════════════════════════════
# 异常处理
# ══════════════════════════════════════════════════════════════════════════

def _api_error(status: int, code: str, message: str, **extra: Any) -> JSONResponse:
    return JSONResponse(
        {"error": {"code": code, "message": message, **extra}}, status_code=status
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """/api/* 一律 JSON（契约 §8.1）；页面路由给错误页或跳登录。"""
    if request.url.path.startswith("/api/"):
        detail = exc.detail
        if isinstance(detail, dict):
            return JSONResponse({"error": detail}, status_code=exc.status_code)
        return _api_error(exc.status_code, f"HTTP_{exc.status_code}", str(detail))

    if exc.status_code == 401:
        return _login_redirect(request)
    if exc.status_code in (403, 404):
        return _error_page(request, exc.status_code, str(exc.detail or "出错了"))
    return _error_page(request, 500, "服务暂时不可用")


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    if request.url.path.startswith("/api/"):
        # 不能直接塞 exc.errors()：Pydantic v2 会在 ctx 里放原始异常对象
        # （{'error': ValueError('…')}），JSONResponse 序列化它时抛 TypeError，
        # 于是被下面的 500 兜底接走 —— 表现是「填了个非法值 → 服务内部错误」。
        # ctx 剔掉：可读文案本来就在 msg 里，ctx 只是那个异常对象的壳。
        return _api_error(422, "VALIDATION_ERROR", "请求参数不合法",
                          fields=jsonable_encoder([
                              {k: v for k, v in err.items() if k != "ctx"}
                              for err in exc.errors()
                          ]))
    return _error_page(request, 404, "请求参数不合法")


@app.exception_handler(404)
async def not_found_handler(request: Request, _exc):
    if request.url.path.startswith("/api/"):
        return _api_error(404, "NOT_FOUND", "资源不存在")
    return _error_page(request, 404, "页面或事件不存在")


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    print(f"[error] {request.method} {request.url.path}: {type(exc).__name__}: {exc}")
    if request.url.path.startswith("/api/"):
        return _api_error(500, "INTERNAL_ERROR", "服务内部错误")
    return _error_page(request, 500, "服务暂时不可用，请稍后重试")
