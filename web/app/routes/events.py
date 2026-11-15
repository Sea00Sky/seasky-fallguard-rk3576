"""app/routes/events.py —— 事件列表 / 详情 / 处置动作 / 快照 / SSE 实时通道。"""

from __future__ import annotations

import asyncio
import base64
import binascii
import datetime as dt
import json
import time
import uuid
from pathlib import Path
from typing import Any

import psycopg
from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from app import clips
from app.config import settings
from app.db import get_conn
from app.deps import (
    CurrentUser,
    audit,
    current_user,
    scope_clauses,
    scoped_ward,
    scoped_zones,
    zone_clauses,
    zone_key,
    zone_keys,
)
from app.metrics import (
    fetch_all,
    fetch_one,
    iso,
    now_sh,
    parse_time_param,
)

router = APIRouter(prefix="/api/v1", tags=["events"])


# ── 枚举白名单（与 DDL 的 CHECK 逐字一致） ─────────────────────────────────
VALID_DISPOSITION = ("PENDING", "ACKED", "ON_SCENE", "RESOLVED", "FALSE_ALARM", "CLOSED")
VALID_DETECTION = ("CONFIRMED", "RETRACTED")
DETECTION_PSEUDO = ("EVER",)
VALID_SEVERITY = ("S1", "S2", "S3")
VALID_OUTCOME = ("ASSISTED", "NO_ABNORMALITY", "MEDICAL", "HOSPITALIZED", "OTHER")
SEVERITY_RANK = {"S1": 1, "S2": 2, "S3": 3}

EVER_CONFIRMED_SQL = (
    "(e.detection_state = 'CONFIRMED'"
    " OR (e.detection_state = 'RETRACTED' AND NOT e.orphan_retract))"
)

# §6.2 处置状态机白名单：不在表里的转换一律 409
ALLOWED_TRANSITIONS: dict[str, set[tuple[str, str]]] = {
    "ACK": {("PENDING", "ACKED")},
    "ARRIVE": {("ACKED", "ON_SCENE")},
    "RESOLVE": {("ACKED", "RESOLVED"), ("ON_SCENE", "RESOLVED")},
    "FALSE_ALARM": {
        ("PENDING", "FALSE_ALARM"),
        ("ACKED", "FALSE_ALARM"),
        ("ON_SCENE", "FALSE_ALARM"),
    },
    "CLOSE": {("RESOLVED", "CLOSED"), ("FALSE_ALARM", "CLOSED")},
    "REOPEN": {("RESOLVED", "PENDING"), ("FALSE_ALARM", "PENDING"), ("CLOSED", "PENDING")},
}
TO_STATE = {
    "ACK": "ACKED",
    "ARRIVE": "ON_SCENE",
    "RESOLVE": "RESOLVED",
    "FALSE_ALARM": "FALSE_ALARM",
    "CLOSE": "CLOSED",
    "REOPEN": "PENDING",
}
ALL_OUTCOMES = ("ASSISTED", "NO_ABNORMALITY", "MEDICAL", "HOSPITALIZED", "OTHER")
FALSE_ALARM_REASONS = ("MODEL_MISJUDGE", "OCCLUSION", "NOT_PATIENT", "DUPLICATE", "OTHER")

STATE_CN = {
    "PENDING": "待处置",
    "ACKED": "已接警",
    "ON_SCENE": "已到场",
    "RESOLVED": "已处置",
    "FALSE_ALARM": "误报",
    "CLOSED": "已归档",
}
ACTION_CN = {
    "ACK": "接警",
    "ARRIVE": "到场",
    "RESOLVE": "处置",
    "FALSE_ALARM": "标记误报",
    "CLOSE": "归档",
    "REOPEN": "重开",
}

# 列表/详情共用的列（别处拼 SQL 时不要再抄一遍）
EVENT_COLUMNS = f"""
        e.event_uid, e.device_id, d.device_code,
        e.location_text AS location,
        e.board_seq, e.board_ts, e.received_at,
        e.person_id, e.window_start, e.window_end, e.model_ver,
        e.prob_fall, e.coverage, e.hybrid_score, e.bbox,
        e.severity, e.detection_state, e.disposition_state,
        e.outcome, e.false_alarm_reason, e.assignee_id,
        u.display_name AS assignee_name,
        e.acked_at, e.arrived_at, e.resolved_at, e.note,
        e.orphan_retract, e.is_test, e.revision,
        {EVER_CONFIRMED_SQL} AS ever_confirmed,
        (s.snapshot_id IS NOT NULL) AS has_snapshot,
        ce.status AS clip_status,
        ce.fail_reason AS clip_fail_reason
"""
EVENT_FROM = """
          FROM fg.event e
          JOIN fg.device d ON d.device_id = e.device_id
          LEFT JOIN fg.app_user u ON u.user_id = e.assignee_id
          LEFT JOIN fg.snapshot s ON s.event_uid = e.event_uid
          LEFT JOIN fg.event_evidence ce
                 ON ce.event_uid = e.event_uid AND ce.kind = 'clip'
"""

# 总数缓存：§5.6「计数用缓存（60 秒 TTL），不每次 count(*)」
_COUNT_TTL_S = 60.0
_count_cache: dict[str, tuple[float, int]] = {}

# 快照查看审计去重：同一用户同一事件 60 秒内只写一条（§8.7 规则 2）
_SNAPSHOT_AUDIT_TTL_S = 60.0
_snapshot_audit_cache: dict[tuple[int, str], float] = {}


def error_json(
    status_code: int, code: str, message: str, detail: Any = None
) -> JSONResponse:
    """统一错误体（§8.1）：{"error": {"code", "message", "detail"}}。"""
    return JSONResponse(
        status_code=status_code,
        content={"error": {"code": code, "message": message, "detail": detail}},
    )


def encode_cursor(payload: dict) -> str:
    """把 keyset 游标编码成 URL 安全的 base64（不透明串，前端不要解析）。"""
    raw = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def decode_cursor(cursor: str) -> dict:
    """解码游标；格式不对抛 ValueError（路由转 400）。"""
    padded = cursor + "=" * (-len(cursor) % 4)
    try:
        raw = base64.urlsafe_b64decode(padded.encode("ascii"))
        data = json.loads(raw.decode("utf-8"))
    except (ValueError, binascii.Error, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("cursor 不是合法游标") from exc
    if not isinstance(data, dict):
        raise ValueError("cursor 不是合法游标")
    return data


def client_ip(request: Request | None) -> str | None:
    """取客户端 IP。nginx 反代后真实 IP 在 X-Forwarded-For 第一段。"""
    if request is None:
        return None
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        first = forwarded.split(",")[0].strip()
        if first:
            return first
    return request.client.host if request.client else None


def _split_enum(raw: str | None, allowed: tuple[str, ...], name: str) -> list[str] | None:
    """解析逗号分隔的多值枚举；出现白名单外的值抛 ValueError。"""
    if raw is None or not raw.strip():
        return None
    values = [item.strip() for item in raw.split(",") if item.strip()]
    bad = [item for item in values if item not in allowed]
    if bad:
        raise ValueError(f"{name} 参数非法：{','.join(bad)}")
    return values or None


def _like_escape(text: str) -> str:
    """转义 LIKE 通配符（默认转义符是反斜杠）。"""
    return text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def build_event_filters(
    *,
    ward_id: str,
    zone_ids: tuple[str, ...] | None = None,
    disposition: str | None = None,
    detection: str | None = None,
    severity: str | None = None,
    device: str | None = None,
    point: str | None = None,
    frm: dt.datetime | None = None,
    to: dt.datetime | None = None,
    q: str | None = None,
    assignee_id: int | None = None,
    is_test: str | None = None,
    outcome: str | None = None,
) -> tuple[str, list[Any]]:
    """拼事件筛选条件，返回 ``(where_sql, params)``。"""
    clauses = ["e.ward_id = %s"]
    params: list[Any] = [ward_id]

    # 区域条件一律来自 zone_clauses()，别处不许手写 zone_id = ...
    zone_sql, zone_params = zone_clauses("e", zone_ids)
    clauses.extend(zone_sql)
    params.extend(zone_params)

    disp = _split_enum(disposition, VALID_DISPOSITION, "disposition")
    if disp:
        clauses.append("e.disposition_state = ANY(%s)")
        params.append(disp)

    if detection:
        if detection in DETECTION_PSEUDO:
            clauses.append(EVER_CONFIRMED_SQL)
        elif detection not in VALID_DETECTION:
            raise ValueError(f"detection 参数非法：{detection}")
        else:
            clauses.append("e.detection_state = %s")
            params.append(detection)

    sev = _split_enum(severity, VALID_SEVERITY, "severity")
    if sev:
        clauses.append("e.severity = ANY(%s)")
        params.append(sev)

    if point:
        device = None

    if device:
        clauses.append("d.device_code = %s")
        params.append(device)

    if point:
        clauses.append("COALESCE(e.location_text, '') = %s")
        params.append(point)

    if frm is not None:
        clauses.append("e.received_at >= %s")
        params.append(frm)
    if to is not None:
        clauses.append("e.received_at < %s")
        params.append(to)

    if q and q.strip():
        needle = q.strip()
        # 事件号前 8 位（UUID 文本小写）或点位模糊匹配
        clauses.append(
            "(e.event_uid::text LIKE %s OR e.location_text ILIKE %s)")
        params.append(_like_escape(needle.lower()) + "%")
        params.append("%" + _like_escape(needle) + "%")

    if assignee_id is not None:
        clauses.append("e.assignee_id = %s")
        params.append(int(assignee_id))

    # 处置结果（KPI-2「确认跌倒」下钻用）：逗号分隔多值
    out = _split_enum(outcome, VALID_OUTCOME, "outcome")
    if out:
        clauses.append("e.outcome = ANY(%s)")
        params.append(out)

    # test=0 只看生产数据 / test=1 只看测试数据 / 空 = 全都要（默认，列表靠标签区分）
    # KPI 卡下钻一律带 test=0，否则「数字」和「点进去看到的行数」会对不上
    # —— v_event_fact 只含 is_test = FALSE，列表页却默认全都要。
    if is_test in ("0", "1"):
        clauses.append("e.is_test = %s")
        params.append(is_test == "1")

    return " AND ".join(clauses), params


def _obj_scope(user: CurrentUser, alias: str = "e") -> tuple[str, list[Any]]:
    """对象级鉴权的附加 SQL 片段。"""
    clauses, params = scope_clauses(alias, user)
    if not clauses:
        return "", []
    return " AND " + " AND ".join(clauses), params


# ════════════════════════════════════════════════════════════════════════════
# 序列化
# ════════════════════════════════════════════════════════════════════════════
def _model_ver_hex(value: Any) -> str | None:
    """模型版本按「低 32 位 16 进制大写」展示（UI 显示 v7F3A21C9）。"""
    if value is None:
        return None
    return f"{int(value) & 0xFFFFFFFF:08X}"


def _event_item(row: dict) -> dict:
    """单条事件的标准输出结构（列表 / 详情 / 动作响应共用字段名）。"""
    assignee = None
    if row.get("assignee_id") is not None:
        assignee = {
            "user_id": row["assignee_id"],
            "display_name": row.get("assignee_name"),
        }
    return {
        "event_uid": str(row["event_uid"]),
        "device_id": str(row["device_id"]),
        "device_code": row["device_code"],
        "location": row["location"],
        "board_seq": row["board_seq"],
        "board_ts": iso(row["board_ts"]),
        "received_at": iso(row["received_at"]),
        "person_id": row["person_id"],
        "window_start": row["window_start"],
        "window_end": row["window_end"],
        "model_ver": row["model_ver"],
        "model_ver_hex": _model_ver_hex(row["model_ver"]),
        "prob_fall": row["prob_fall"],
        "coverage": row["coverage"],
        "hybrid_score": row["hybrid_score"],
        "bbox": row["bbox"],
        "severity": row["severity"],
        "detection_state": row["detection_state"],
        "disposition_state": row["disposition_state"],
        "outcome": row["outcome"],
        "false_alarm_reason": row["false_alarm_reason"],
        "assignee": assignee,
        "acked_at": iso(row["acked_at"]),
        "arrived_at": iso(row["arrived_at"]),
        "resolved_at": iso(row["resolved_at"]),
        "note": row["note"],
        "orphan_retract": bool(row["orphan_retract"]),
        "ever_confirmed": bool(row.get("ever_confirmed")),
        "is_test": bool(row["is_test"]),
        "revision": row["revision"],
        "has_snapshot": bool(row.get("has_snapshot")),
        "clip_status": row.get("clip_status"),
        "has_clip": row.get("clip_status") == "READY",
        # 「过了保留期被清理」与「生成失败」在页面上要分开说，
        # 但 fail_reason 原文不外泄（含 ffmpeg stderr 与服务器绝对路径）
        "clip_expired": row.get("clip_fail_reason") == "retention_expired",
    }


def _action_item(row: dict) -> dict:
    """事件时间线的一条动作流水。"""
    actor = None
    if row.get("actor_id") is not None:
        actor = {"user_id": row["actor_id"], "display_name": row.get("actor_name")}
    return {
        "action_id": row["action_id"],
        "action": row["action"],
        "actor": actor,
        "from_state": row["from_state"],
        "to_state": row["to_state"],
        "outcome": row["outcome"],
        "reason": row["reason"],
        "created_at": iso(row["created_at"]),
    }


def _parse_uuid_or_none(raw: str) -> uuid.UUID | None:
    try:
        return uuid.UUID(str(raw))
    except (ValueError, AttributeError, TypeError):
        return None


# ════════════════════════════════════════════════════════════════════════════
# GET /api/v1/events —— 列表（keyset 分页）
# ════════════════════════════════════════════════════════════════════════════
def _parse_assignee(raw: str | None, user: CurrentUser) -> int | None:
    """``assignee=me``（护士历史页）或 ``assignee=<user_id>``。"""
    if raw is None or not raw.strip():
        return None
    if raw.strip() == "me":
        return int(user.user_id)
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError("assignee 参数必须是 me 或 user_id") from exc


def _order_by(sort: str) -> str:
    if sort == "time_asc":
        return "ORDER BY e.received_at ASC, e.event_uid ASC"
    if sort == "workbench":
        return (
            "ORDER BY CASE e.severity WHEN 'S1' THEN 1 WHEN 'S2' THEN 2 ELSE 3 END ASC, "
            "CASE e.disposition_state WHEN 'PENDING' THEN 0 ELSE 1 END ASC, "
            "e.received_at ASC, e.event_uid ASC"
        )
    return "ORDER BY e.received_at DESC, e.event_uid DESC"


def _keyset_clause(sort: str, cursor_data: dict) -> tuple[str, list[Any]]:
    """按排序方式拼 keyset 条件。游标里带着上一页最后一行的排序键。"""
    try:
        last_uid = uuid.UUID(str(cursor_data["u"]))
        last_ts = parse_time_param(str(cursor_data["t"]))
        if sort == "workbench":
            return (
                "AND (CASE e.severity WHEN 'S1' THEN 1 WHEN 'S2' THEN 2 ELSE 3 END, "
                "CASE e.disposition_state WHEN 'PENDING' THEN 0 ELSE 1 END, "
                "e.received_at, e.event_uid) > (%s, %s, %s, %s::uuid)",
                [int(cursor_data["s"]), int(cursor_data["p"]), last_ts, str(last_uid)],
            )
        operator = ">" if sort == "time_asc" else "<"
        return (
            f"AND (e.received_at, e.event_uid) {operator} (%s, %s::uuid)",
            [last_ts, str(last_uid)],
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("cursor 与当前排序方式不匹配") from exc


def cursor_timestamp(value: Any) -> str:
    """游标里的时间戳必须带微秒。"""
    if isinstance(value, dt.datetime):
        return value.isoformat()
    return str(value)


def _cursor_of(row: dict, sort: str) -> dict:
    payload = {"t": cursor_timestamp(row["received_at"]), "u": str(row["event_uid"])}
    if sort == "workbench":
        payload["s"] = SEVERITY_RANK.get(row["severity"], 3)
        payload["p"] = 0 if row["disposition_state"] == "PENDING" else 1
    return payload


def _count_estimate(conn, where_sql: str, params: list[Any]) -> int:
    """带 60 秒 TTL 的总数估算（§5.6）。键含筛选参数，避免串味。"""
    key = where_sql + "|" + repr(params)
    now = time.monotonic()
    cached = _count_cache.get(key)
    if cached is not None and now - cached[0] < _COUNT_TTL_S:
        return cached[1]
    row = fetch_one(conn, "SELECT count(*) AS n" + EVENT_FROM + " WHERE " + where_sql, params)
    total = int(row["n"]) if row else 0
    if len(_count_cache) > 512:  # 防止长期运行无限增长
        _count_cache.clear()
    _count_cache[key] = (now, total)
    return total


@router.get("/events")
def list_events(
    disposition: str | None = Query(None, description="处置态，逗号分隔多值"),
    detection: str | None = Query(
        None, description="CONFIRMED / RETRACTED / EVER（EVER=曾经确认过，含已撤回）"
    ),
    severity: str | None = Query(None, description="S1,S2,S3 逗号分隔"),
    device: str | None = Query(None, description="device_code"),
    point: str | None = Query(None, description="点位名（按事件位置快照匹配）"),
    outcome: str | None = Query(None, description="处置结果，逗号分隔"),
    test: str | None = Query(None, description="0 只看生产数据 / 1 只看测试数据"),
    frm: str | None = Query(None, alias="from", description="起始时间（含）"),
    to: str | None = Query(None, description="结束时间（不含；纯日期含当天全天）"),
    q: str | None = Query(None, description="事件号前缀或点位模糊搜索"),
    page: int = Query(1, ge=1),
    size: int = Query(50, ge=1, le=100),
    sort: str | None = Query(None, description="time_desc / time_asc / workbench；默认按角色"),
    cursor: str | None = Query(None, description="上一页返回的 next_cursor"),
    assignee: str | None = Query(None, description="me 或 user_id（护士历史页）"),
    user: CurrentUser = Depends(current_user),
    conn: Any = Depends(get_conn),
) -> Any:
    """事件列表。病区由登录身份决定，URL 里传 ward_id 一律忽略（§8.8 用例 2）。"""
    ward = scoped_ward(user)

    if sort is None or sort == "":
        sort = "workbench" if user.role == "nurse" else "time_desc"
    if sort not in ("time_desc", "time_asc", "workbench"):
        return error_json(400, "SORT_INVALID", "sort 只支持 time_desc / time_asc / workbench")

    try:
        period_from = parse_time_param(frm) if frm else None
        period_to = parse_time_param(to, end_of_day=True) if to else None
        if period_from and period_to and period_from >= period_to:
            raise ValueError("from 必须早于 to")
        assignee_id = _parse_assignee(assignee, user)
        where_sql, params = build_event_filters(
            ward_id=ward,
            zone_ids=scoped_zones(user),
            disposition=disposition,
            detection=detection,
            severity=severity,
            device=device,
            point=point,
            frm=period_from,
            to=period_to,
            q=q,
            assignee_id=assignee_id,
            outcome=outcome,
            is_test=test,
        )
        cursor_data = decode_cursor(cursor) if cursor else None
        keyset_sql, keyset_params = (
            _keyset_clause(sort, cursor_data) if cursor_data else ("", [])
        )
    except ValueError as exc:
        return error_json(400, "QUERY_INVALID", str(exc))

    # 说明：正常翻页走 keyset（next_cursor）。?page=N 仅为兼容设计方案里的页码 UI
    # 保留；page>1 且没有 cursor 时退化为 OFFSET，仅用于浅翻页（≤ 数页）。
    offset = 0
    if cursor_data is None and page > 1:
        offset = (page - 1) * size

    sql = (
        "SELECT" + EVENT_COLUMNS + EVENT_FROM + " WHERE " + where_sql + " " + keyset_sql + " " + _order_by(sort)
    )
    query_params = list(params) + list(keyset_params)
    if offset:
        sql += " LIMIT %s OFFSET %s"
        query_params.extend([size + 1, offset])
    else:
        sql += " LIMIT %s"
        query_params.append(size + 1)

    rows = fetch_all(conn, sql, query_params)
    has_more = len(rows) > size
    rows = rows[:size]
    next_cursor = encode_cursor(_cursor_of(rows[-1], sort)) if has_more and rows else None
    total_estimate = _count_estimate(conn, where_sql, params)

    return {
        "items": [_event_item(row) for row in rows],
        "next_cursor": next_cursor,
        "total_estimate": total_estimate,
        # total 是给前端角标轮询用的同义字段（app.js 的 extractTotal 先读 total）
        "total": total_estimate,
        "size": size,
        "sort": sort,
    }


# ════════════════════════════════════════════════════════════════════════════
# GET /api/v1/events/{event_uid} —— 详情（含动作时间线与快照元信息）
# ════════════════════════════════════════════════════════════════════════════
@router.get("/events/{event_uid}")
def get_event(
    event_uid: str,
    user: CurrentUser = Depends(current_user),
    conn: Any = Depends(get_conn),
) -> Any:
    ward = scoped_ward(user)
    uid = _parse_uuid_or_none(event_uid)
    if uid is None:
        # 跨病区与不存在一律 404（§2.5），连格式错也不额外暴露信息
        return error_json(404, "EVENT_NOT_FOUND", "事件不存在")

    scope_sql, scope_params = _obj_scope(user)
    row = fetch_one(
        conn,
        "SELECT" + EVENT_COLUMNS + EVENT_FROM
        + " WHERE e.event_uid = %s AND e.ward_id = %s" + scope_sql,
        (uid, ward, *scope_params),
    )
    if row is None:
        return error_json(404, "EVENT_NOT_FOUND", "事件不存在")

    actions = fetch_all(
        conn,
        """
        SELECT a.action_id, a.action, a.actor_id, u.display_name AS actor_name,
               a.from_state, a.to_state, a.outcome, a.reason, a.created_at
          FROM fg.event_action a
          LEFT JOIN fg.app_user u ON u.user_id = a.actor_id
         WHERE a.event_uid = %s
         ORDER BY a.created_at ASC, a.action_id ASC
        """,
        (uid,),
    )

    snapshot = fetch_one(
        conn,
        """
        SELECT snapshot_id, width, height, bytes, sha256, source, uploaded_at
          FROM fg.snapshot
         WHERE event_uid = %s
        """,
        (uid,),
    )

    item = _event_item(row)
    item["actions"] = [_action_item(a) for a in actions]
    item["snapshot"] = (
        {
            "snapshot_id": str(snapshot["snapshot_id"]),
            "width": snapshot["width"],
            "height": snapshot["height"],
            "bytes": snapshot["bytes"],
            "sha256": (snapshot["sha256"] or "").strip(),
            "source": snapshot["source"],
            "uploaded_at": iso(snapshot["uploaded_at"]),
        }
        if snapshot
        else None
    )
    clip = clips.get_clip(conn, uid)
    item["clip"] = (
        {
            "status": clip["status"],
            "duration_ms": clip["duration_ms"],
            "width": clip["width"],
            "height": clip["height"],
            "bytes": clip["bytes"],
            "ready_at": iso(clip["ready_at"]) if clip["ready_at"] else None,
            # fail_reason 原文不回显（含 ffmpeg stderr 与服务器路径），只给这一个布尔
            "expired": clip["fail_reason"] == "retention_expired",
        }
        if clip
        else None
    )
    # 技术详情用：板端墙钟与云端收件时间的偏差（秒），正数表示板端慢
    if row["board_ts"] is not None and row["received_at"] is not None:
        item["board_ts_drift_s"] = round(
            (row["received_at"] - row["board_ts"]).total_seconds(), 1
        )
    else:
        item["board_ts_drift_s"] = None
    return item


# ════════════════════════════════════════════════════════════════════════════
# POST /api/v1/events/{event_uid}/actions —— 处置动作
# ════════════════════════════════════════════════════════════════════════════
class ActionIn(BaseModel):
    """动作请求体（§8.3）。"""

    action: str
    outcome: str | None = None
    reason: str | None = None
    note: str | None = None
    client_request_id: str | None = None
    expected_revision: int | None = None


def _assignee_name(conn, assignee_id: Any) -> str | None:
    if assignee_id is None:
        return None
    row = fetch_one(conn, "SELECT display_name FROM fg.app_user WHERE user_id = %s", (assignee_id,))
    return row["display_name"] if row else None


def _conflict(conn, ev: dict, code: str, message: str) -> JSONResponse:
    """409 冲突体（§8.3）：带当前状态、责任人、接警时间、revision。"""
    return error_json(
        409,
        code,
        message,
        detail={
            "disposition_state": ev["disposition_state"],
            "detection_state": ev["detection_state"],
            "assignee": _assignee_name(conn, ev.get("assignee_id")),
            "acked_at": iso(ev.get("acked_at")),
            "revision": ev["revision"],
        },
    )


def _action_result(
    conn, uid: uuid.UUID, ward: str, duplicate: bool, user: CurrentUser
) -> Any:
    """动作成功响应（§8.3）与幂等重放响应共用。"""
    scope_sql, scope_params = _obj_scope(user)
    row = fetch_one(
        conn,
        """
        SELECT e.event_uid, e.detection_state, e.disposition_state, e.outcome,
               e.assignee_id, u.display_name AS assignee_name,
               e.acked_at, e.arrived_at, e.resolved_at, e.revision, e.zone_id
          FROM fg.event e
          LEFT JOIN fg.app_user u ON u.user_id = e.assignee_id
         WHERE e.event_uid = %s AND e.ward_id = %s
        """ + scope_sql,
        (uid, ward, *scope_params),
    )
    if row is None:
        return error_json(404, "EVENT_NOT_FOUND", "事件不存在")
    return {
        "event_uid": str(row["event_uid"]),
        "detection_state": row["detection_state"],
        "disposition_state": row["disposition_state"],
        "outcome": row["outcome"],
        "assignee": (
            {"user_id": row["assignee_id"], "display_name": row["assignee_name"]}
            if row["assignee_id"] is not None
            else None
        ),
        "acked_at": iso(row["acked_at"]),
        "arrived_at": iso(row["arrived_at"]),
        "resolved_at": iso(row["resolved_at"]),
        "revision": row["revision"],
        "duplicate": duplicate,
        "zone_id": zone_key(row["zone_id"]),
    }


def _apply_update(
    conn,
    action: str,
    uid: uuid.UUID,
    ward: str,
    from_state: str,
    revision: int,
    user: CurrentUser,
    body: ActionIn,
) -> int:
    """条件更新（§6.5）：WHERE 里同时带状态与 revision，影响行数 0 即冲突。"""
    note = (body.note or "").strip() or None
    scope_sql, scope_params = _obj_scope(user, alias="")
    where = (
        " WHERE event_uid = %s AND ward_id = %s"
        " AND disposition_state = %s AND revision = %s" + scope_sql +
        " RETURNING event_uid"
    )
    tail = (uid, ward, from_state, revision, *scope_params)

    if action == "ACK":
        # 认领：assignee 与 acked_at 一起落库
        cur = conn.execute(
            "UPDATE fg.event SET disposition_state = 'ACKED', assignee_id = %s,"
            " acked_at = now(), note = COALESCE(%s, note), revision = revision + 1" + where,
            (user.user_id, note, *tail),
        )
    elif action == "ARRIVE":
        cur = conn.execute(
            "UPDATE fg.event SET disposition_state = 'ON_SCENE',"
            " arrived_at = now(), note = COALESCE(%s, note), revision = revision + 1" + where,
            (note, *tail),
        )
    elif action == "RESOLVE":
        # ck_event_outcome：RESOLVED 必须有 outcome；顺手把误报原因清空
        cur = conn.execute(
            "UPDATE fg.event SET disposition_state = 'RESOLVED', outcome = %s,"
            " false_alarm_reason = NULL, resolved_at = now(),"
            " note = COALESCE(%s, note), revision = revision + 1" + where,
            (body.outcome, note, *tail),
        )
    elif action == "FALSE_ALARM":
        # ck_event_fa：FALSE_ALARM 必须有 reason；同时清空 outcome
        cur = conn.execute(
            "UPDATE fg.event SET disposition_state = 'FALSE_ALARM', false_alarm_reason = %s,"
            " outcome = NULL, resolved_at = now(),"
            " note = COALESCE(%s, note), revision = revision + 1" + where,
            (body.reason, note, *tail),
        )
    elif action == "CLOSE":
        cur = conn.execute(
            "UPDATE fg.event SET disposition_state = 'CLOSED',"
            " note = COALESCE(%s, note), revision = revision + 1" + where,
            (note, *tail),
        )
    else:  # REOPEN：回到 PENDING，清空责任人；acked_at/arrived_at/resolved_at 保留
        cur = conn.execute(
            "UPDATE fg.event SET disposition_state = 'PENDING', assignee_id = NULL,"
            " outcome = NULL, false_alarm_reason = NULL,"
            " note = COALESCE(%s, note), revision = revision + 1" + where,
            (note, *tail),
        )
    return cur.rowcount


@router.post("/events/{event_uid}/actions")
def post_event_action(
    event_uid: str,
    body: ActionIn,
    request: Request,
    user: CurrentUser = Depends(current_user),
    conn: Any = Depends(get_conn),
) -> Any:
    """处置动作：白名单状态机 + client_request_id 幂等 + expected_revision 乐观锁。"""
    ward = scoped_ward(user)
    uid = _parse_uuid_or_none(event_uid)
    if uid is None:
        return error_json(404, "EVENT_NOT_FOUND", "事件不存在")

    action = (body.action or "").strip().upper()
    if action not in ALLOWED_TRANSITIONS:
        return error_json(
            400, "ACTION_UNKNOWN", f"未知动作：{body.action}", detail={"allowed": sorted(ALLOWED_TRANSITIONS)}
        )

    # 角色边界（§3.5 / §8.8 用例 5）：护士长只能 REOPEN，护士不能 REOPEN
    role_denied = (action == "REOPEN" and user.role != "head_nurse") or (
        action != "REOPEN" and user.role != "nurse"
    )
    if role_denied:
        with conn.transaction():
            audit(
                conn,
                user,
                "DENIED",
                "event",
                str(uid),
                "DENIED",
                detail={"action": action, "role": user.role},
                client_ip=client_ip(request),
            )
        return error_json(403, "FORBIDDEN", "当前账号无权执行该动作")

    # 参数校验（写错就是脏数据，宁可 422 也不要进库）
    if action == "RESOLVE" and body.outcome not in ALL_OUTCOMES:
        return error_json(422, "OUTCOME_REQUIRED", "处置必须选择处置结果", detail={"allowed": list(ALL_OUTCOMES)})
    if action == "FALSE_ALARM" and body.reason not in FALSE_ALARM_REASONS:
        return error_json(
            422, "REASON_REQUIRED", "标记误报必须选择原因", detail={"allowed": list(FALSE_ALARM_REASONS)}
        )
    if action == "REOPEN" and not (body.reason or "").strip():
        return error_json(422, "REASON_REQUIRED", "重开必须填写原因")
    if body.note and len(body.note) > 200:
        return error_json(422, "NOTE_TOO_LONG", "备注不得超过 200 字")
    request_id = (body.client_request_id or "").strip() or None
    if request_id and len(request_id) > 128:
        return error_json(400, "CLIENT_REQUEST_ID_INVALID", "client_request_id 过长")

    # ① 幂等：同一个 client_request_id 直接返回上次结果，不新增 event_action
    if request_id:
        prev = fetch_one(
            conn,
            "SELECT event_uid, action FROM fg.event_action WHERE client_request_id = %s",
            (request_id,),
        )
        if prev is not None:
            if str(prev["event_uid"]) != str(uid):
                return error_json(
                    409,
                    "CLIENT_REQUEST_ID_REUSED",
                    "该请求号已用于其他事件",
                    detail={"event_uid": str(prev["event_uid"])},
                )
            return _action_result(conn, uid, ward, duplicate=True, user=user)

    scope_sql, scope_params = _obj_scope(user, alias="")

    try:
        with conn.transaction():
            # ② 锁行：同一事件的动作串行化
            ev = fetch_one(
                conn,
                """
                SELECT event_uid, detection_state, disposition_state, assignee_id,
                       acked_at, revision
                  FROM fg.event
                 WHERE event_uid = %s AND ward_id = %s
                """ + scope_sql + """
                 FOR UPDATE
                """,
                (uid, ward, *scope_params),
            )
            if ev is None:
                return error_json(404, "EVENT_NOT_FOUND", "事件不存在")

            # ③ 乐观锁：客户端拿到的 revision 过期就 409
            if body.expected_revision is not None and int(ev["revision"]) != int(body.expected_revision):
                return _conflict(conn, ev, "EVENT_REVISION_CONFLICT", "事件已被他人更新，请刷新后重试")

            # ④ 状态机白名单
            from_state = ev["disposition_state"]
            to_state = TO_STATE[action]
            if (from_state, to_state) not in ALLOWED_TRANSITIONS[action]:
                if action == "ACK":
                    who = _assignee_name(conn, ev.get("assignee_id")) or "他人"
                    message = f"该事件已被 {who} 接警"
                else:
                    message = (
                        f"事件当前状态为「{STATE_CN.get(from_state, from_state)}」，"
                        f"不能执行「{ACTION_CN.get(action, action)}」"
                    )
                return _conflict(conn, ev, "EVENT_STATE_CONFLICT", message)

            # ⑤ 条件更新（状态 + revision 都进 WHERE）
            affected = _apply_update(
                conn, action, uid, ward, from_state, int(ev["revision"]), user, body
            )
            if affected == 0:
                latest = fetch_one(
                    conn,
                    """
                    SELECT event_uid, detection_state, disposition_state, assignee_id,
                           acked_at, revision
                      FROM fg.event
                     WHERE event_uid = %s AND ward_id = %s
                    """ + scope_sql + """
                    """,
                    (uid, ward, *scope_params),
                )
                return _conflict(
                    conn, latest or ev, "EVENT_STATE_CONFLICT", "事件状态已变化，请刷新后重试"
                )

            # ⑥ 写动作流水（证据）；reason 只对 FALSE_ALARM / REOPEN 有业务含义
            reason = body.reason if action in ("FALSE_ALARM", "REOPEN") else None
            conn.execute(
                """
                INSERT INTO fg.event_action
                    (event_uid, actor_id, action, from_state, to_state,
                     outcome, reason, client_request_id)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    uid,
                    user.user_id,
                    action,
                    from_state,
                    to_state,
                    body.outcome if action == "RESOLVE" else None,
                    reason,
                    request_id,
                ),
            )

            # ⑦ 审计与业务同事务（§7.5：分开写会出现「改了状态但没审计」）
            audit(
                conn,
                user,
                action,
                "event",
                str(uid),
                "OK",
                detail={
                    "from_state": from_state,
                    "to_state": to_state,
                    "outcome": body.outcome,
                    "reason": reason,
                    "note": body.note,
                    "client_request_id": request_id,
                },
                client_ip=client_ip(request),
            )

            result = _action_result(conn, uid, ward, duplicate=False, user=user)
    except psycopg.errors.UniqueViolation:
        # 并发下同一 client_request_id 撞唯一键：按幂等重放处理
        return _action_result(conn, uid, ward, duplicate=True, user=user)

    _count_cache.clear()

    # 事务提交后再广播，保证前端收到 SSE 立刻能查到最新行。
    # 过滤用的是事件自己的 zone_id，不是当前用户的 —— 广播面向的是
    # 同病区的一群人，每个人按自己的区域范围收。
    if isinstance(result, dict):
        publish_event(
            ward,
            "event.updated",
            {
                "event_uid": result["event_uid"],
                "detection_state": result["detection_state"],
                "disposition_state": result["disposition_state"],
                "revision": result["revision"],
                "assignee": result["assignee"],
            },
            zone_id=result.get("zone_id"),
        )
    return result


# ════════════════════════════════════════════════════════════════════════════
# 快照（原图写审计，缩略图不写审计）
# ════════════════════════════════════════════════════════════════════════════
def _snapshot_root() -> Path:
    # settings.snapshot_dir 必须在 webroot 之外（§8.7 规则 1）
    return Path(getattr(settings, "snapshot_dir", "/var/lib/fallguard/snapshots")).resolve()


def _resolve_snapshot_file(storage_path: str | None) -> Path | None:
    """把库里存的路径解析成真实文件；越出快照根目录一律拒绝（防目录穿越）。"""
    if not storage_path:
        return None
    root = _snapshot_root()
    path = Path(storage_path)
    if not path.is_absolute():
        path = root / path
    try:
        path = path.resolve()
        path.relative_to(root)
    except (OSError, ValueError):
        return None
    return path if path.is_file() else None


def _thumb_file(path: Path) -> Path:
    """优先用 ``<uid>_thumb.jpg``；V1 没有缩略图生成器时回落到原图（仍不写审计）。"""
    candidate = path.with_name(f"{path.stem}_thumb{path.suffix}")
    return candidate if candidate.is_file() else path


def _image_media_type(path: Path) -> str:
    """按扩展名给 media_type（板端快照默认 JPEG，允许 PNG）。"""
    return "image/png" if path.suffix.lower() == ".png" else "image/jpeg"


def _load_snapshot(conn, uid: uuid.UUID, ward: str, user: CurrentUser):
    """返回 (event_uid 或 None, snapshot 行 或 None)。跨病区 / 跨区域都按不存在处理。"""
    # 这条 SQL 的 fg.event 没有别名，alias 必须传空串，
    # 否则拼出 "e.zone_id" 会直接 UndefinedTable 500（验收脚本当场抓到过一次）。
    scope_sql, scope_params = _obj_scope(user, alias="")
    ev = fetch_one(
        conn,
        "SELECT event_uid FROM fg.event WHERE event_uid = %s AND ward_id = %s" + scope_sql,
        (uid, ward, *scope_params),
    )
    if ev is None:
        return None, None
    snap = fetch_one(
        conn,
        "SELECT snapshot_id, storage_path FROM fg.snapshot WHERE event_uid = %s",
        (uid,),
    )
    return ev, snap


def _audit_media_view(
    conn,
    user: CurrentUser,
    uid: uuid.UUID,
    action: str,
    object_type: str,
    request: Request,
) -> None:
    """写媒体查看审计；同一用户同一事件同一类媒体 60 秒内去重（§8.7 规则 2）。"""
    key = (int(user.user_id), str(uid), action)
    now = time.monotonic()
    last = _snapshot_audit_cache.get(key)
    if last is not None and now - last < _SNAPSHOT_AUDIT_TTL_S:
        return
    with conn.transaction():
        audit(
            conn,
            user,
            action,
            object_type,
            str(uid),
            "OK",
            detail=None,
            client_ip=client_ip(request),
        )
    _snapshot_audit_cache[key] = now
    if len(_snapshot_audit_cache) > 4096:  # 顺手清理过期键，避免无限增长
        for item_key, at in list(_snapshot_audit_cache.items()):
            if now - at >= _SNAPSHOT_AUDIT_TTL_S:
                _snapshot_audit_cache.pop(item_key, None)


def _audit_snapshot_view(conn, user: CurrentUser, uid: uuid.UUID, request: Request) -> None:
    """写 SNAPSHOT_VIEW 审计（原图才写，缩略图不写）。"""
    _audit_media_view(conn, user, uid, "SNAPSHOT_VIEW", "snapshot", request)


@router.get("/events/{event_uid}/snapshot")
def get_event_snapshot(
    event_uid: str,
    request: Request,
    user: CurrentUser = Depends(current_user),
    conn: Any = Depends(get_conn),
) -> Any:
    """原图（写 SNAPSHOT_VIEW 审计）。文件不在 webroot，只能经本接口读。"""
    ward = scoped_ward(user)
    uid = _parse_uuid_or_none(event_uid)
    if uid is None:
        return error_json(404, "EVENT_NOT_FOUND", "事件不存在")

    ev, snap = _load_snapshot(conn, uid, ward, user)
    if ev is None:
        return error_json(404, "EVENT_NOT_FOUND", "事件不存在")
    if snap is None:
        return error_json(404, "SNAPSHOT_NOT_FOUND", "该事件没有快照")
    path = _resolve_snapshot_file(snap.get("storage_path"))
    if path is None:
        return error_json(404, "SNAPSHOT_FILE_MISSING", "快照文件不存在")

    _audit_snapshot_view(conn, user, uid, request)
    return FileResponse(
        path,
        media_type=_image_media_type(path),
        headers={
            "Cache-Control": "private, no-store",
            "Content-Disposition": f'inline; filename="{uid}.jpg"',
        },
    )


@router.get("/events/{event_uid}/snapshot/thumb")
def get_event_snapshot_thumb(
    event_uid: str,
    user: CurrentUser = Depends(current_user),
    conn: Any = Depends(get_conn),
) -> Any:
    """列表用缩略图：不写审计（否则一页 20 行 = 20 条审计，见 §8.7 规则 3）。"""
    ward = scoped_ward(user)
    uid = _parse_uuid_or_none(event_uid)
    if uid is None:
        return error_json(404, "EVENT_NOT_FOUND", "事件不存在")

    ev, snap = _load_snapshot(conn, uid, ward, user)
    if ev is None:
        return error_json(404, "EVENT_NOT_FOUND", "事件不存在")
    if snap is None:
        return error_json(404, "SNAPSHOT_NOT_FOUND", "该事件没有快照")
    path = _resolve_snapshot_file(snap.get("storage_path"))
    if path is None:
        return error_json(404, "SNAPSHOT_FILE_MISSING", "快照文件不存在")

    thumb = _thumb_file(path)
    return FileResponse(
        thumb,
        media_type=_image_media_type(thumb),
        headers={
            "Cache-Control": "private, max-age=60",
            "Content-Disposition": f'inline; filename="{uid}_thumb.jpg"',
        },
    )


def _clip_root() -> Path:
    return Path(getattr(settings, "clip_dir", "/var/lib/fallguard/clips")).resolve()


def _resolve_clip_file(storage_path: str | None) -> Path | None:
    """同 `_resolve_snapshot_file`：越出片段根目录一律拒绝（防目录穿越）。"""
    if not storage_path:
        return None
    root = _clip_root()
    path = Path(storage_path)
    if not path.is_absolute():
        path = root / path
    try:
        path = path.resolve()
        path.relative_to(root)
    except (OSError, ValueError):
        return None
    return path if path.is_file() else None


@router.get("/events/{event_uid}/clip")
def get_event_clip(
    event_uid: str,
    request: Request,
    user: CurrentUser = Depends(current_user),
    conn: Any = Depends(get_conn),
) -> Any:
    """告警视频片段（前 5 s + 后 5 s）。"""
    ward = scoped_ward(user)
    uid = _parse_uuid_or_none(event_uid)
    if uid is None:
        return error_json(404, "EVENT_NOT_FOUND", "事件不存在")

    scope_sql, scope_params = _obj_scope(user, alias="")
    ev = fetch_one(
        conn,
        "SELECT event_uid FROM fg.event WHERE event_uid = %s AND ward_id = %s" + scope_sql,
        (uid, ward, *scope_params),
    )
    if ev is None:
        return error_json(404, "EVENT_NOT_FOUND", "事件不存在")

    clip = clips.get_clip(conn, uid)
    if clip is None:
        return error_json(404, "CLIP_NOT_FOUND", "该事件没有视频片段")
    if clip["status"] == "PENDING":
        return error_json(409, "CLIP_NOT_READY", "片段仍在生成中，请稍后重试")
    if clip["status"] == "FAILED":
        # 不要把 fail_reason 回显给前端：里面是 ffmpeg 的 stderr、Python 异常文本
        # 和服务器绝对路径，同病区任何护士都能看到，属于信息泄露。详情只给一句人话，
        # 真正的原因留在库和 journal 里给运维看。
        print(f"[clip] {uid} 片段失败原因: {clip['fail_reason']}", flush=True)
        return error_json(404, "CLIP_FAILED", "片段生成失败")

    path = _resolve_clip_file(clip.get("storage_path"))
    if path is None:
        return error_json(404, "CLIP_FILE_MISSING", "片段文件不存在")

    _audit_media_view(conn, user, uid, "CLIP_VIEW", "clip", request)
    return FileResponse(
        path,
        media_type=clip.get("mime") or "video/mp4",
        headers={
            "Cache-Control": "private, no-store",
            "Content-Disposition": f'inline; filename="{uid}.mp4"',
        },
    )


# ════════════════════════════════════════════════════════════════════════════
# GET /api/v1/stream —— SSE（单独定义，不挂在 /events 下）
# ════════════════════════════════════════════════════════════════════════════
# 单 worker 前提下的进程内广播（§8.6）：多 worker 会各广播各的，必须 --workers 1。
# 值 = (ward_id, zones)：zones 为 None 表示不受区域限制（护士长），
# 为元组表示只收 zone_id 属于它的推送（空元组 = 一条都收不到）。
_sse_queues: dict[asyncio.Queue, tuple[str, tuple[str, ...] | None]] = {}
_sse_loop: asyncio.AbstractEventLoop | None = None
_PING_INTERVAL_S = 15


def _put_nowait(queue: asyncio.Queue, item: dict) -> None:
    try:
        queue.put_nowait(item)
    except asyncio.QueueFull:
        # 慢客户端丢帧：前端还有 5 秒轮询兜底（§8.6），不阻塞广播
        pass


def publish_event(ward_id: str, event_name: str, data: dict, *, zone_id: Any) -> None:
    """向「同病区 且 区域范围内」的 SSE 连接广播。可在任意线程调用。"""
    loop = _sse_loop
    if loop is None or loop.is_closed():
        return
    zk = zone_key(zone_id)
    item = {"event": event_name, "data": data}
    for queue, (subscriber_ward, subscriber_zones) in list(_sse_queues.items()):
        if subscriber_ward != ward_id:
            continue
        # None = 护士长/管理员，不收窄；元组 = 护士，只收本区域的
        if subscriber_zones is not None and zk not in subscriber_zones:
            continue
        try:
            loop.call_soon_threadsafe(_put_nowait, queue, item)
        except RuntimeError:
            # 事件循环已关闭（进程正在退出）
            return


def _sse_frame(event_name: str, data: dict) -> str:
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    return f"event: {event_name}\ndata: {payload}\n\n"


@router.get("/stream")
async def stream_events(
    request: Request,
    user: CurrentUser = Depends(current_user),
) -> Any:
    """SSE 实时通道（§8.6）。"""
    global _sse_loop
    _sse_loop = asyncio.get_running_loop()
    ward = scoped_ward(user)
    zones = zone_keys(scoped_zones(user))

    async def recheck(queue: asyncio.Queue) -> bool:
        """借心跳复核一次权限：账号被停用 / 病区或区域被改，都要当场断线。"""
        try:
            fresh = await run_in_threadpool(current_user, request)
            ward2 = scoped_ward(fresh)
        except Exception:  # noqa: BLE001 —— 401/403/库抖动都按「该重连」处理
            return False
        _sse_queues[queue] = (ward2, zone_keys(scoped_zones(fresh)))
        return True

    async def event_generator():
        # 订阅登记在生成器体内：登记与退订必须在同一段生命周期里。
        # 放外面的话，响应对象一旦没被真正 dispatch，`finally` 就永远跑不到，
        # `_sse_queues` 里留一条僵尸 —— 而 /healthz 的 sse_conns 就是它的长度。
        # 两处 SSE（本函数与 live.live_meta_stream）用同一个形状，别只改一处。
        queue: asyncio.Queue = asyncio.Queue(maxsize=200)
        # 订阅时钉死区域范围；投递时 publish_event 按事件自己的 zone_id 过滤。
        # 只在 REST/HTML 层收窄是不够的：SSE 仍会把别区域的告警弹窗推过来，
        # 点进去还是 404 —— 用户能直接看见这个漏洞。
        _sse_queues[queue] = (ward, zones)
        last_check = time.monotonic()
        try:
            yield _sse_frame("hello", {"server_time": iso(now_sh()), "ward_id": ward})
            while True:
                try:
                    item = await asyncio.wait_for(queue.get(), timeout=_PING_INTERVAL_S)
                except asyncio.TimeoutError:
                    item = None
                    yield _sse_frame("ping", {"t": int(time.time())})
                # 复核必须按时间触发，不能只挂在「空闲超时」上：
                # 回放/连发时队列一直有货，超时分支永远走不到 ⇒ 被撤权的
                # 连接会一直收推送。这与缓存 10 秒、心跳 15 秒都无关，
                # 是「收得越勤越要记得查权限」。
                if time.monotonic() - last_check >= _PING_INTERVAL_S:
                    last_check = time.monotonic()
                    if not await recheck(queue):
                        break
                if item is not None:
                    yield _sse_frame(item["event"], item["data"])
        finally:
            _sse_queues.pop(queue, None)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # nginx 反代必须关缓冲
        },
    )
