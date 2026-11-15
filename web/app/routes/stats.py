"""app/routes/stats.py —— 护士长统计、24 小时分布、CSV 导出。"""

from __future__ import annotations

import csv
import datetime as dt
import io
from typing import Any
from urllib.parse import quote

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import Response

from app.db import get_conn
from app.deps import (
    CurrentUser,
    audit,
    current_user,
    require_role,
    scoped_ward,
    scoped_zones,
)
from app.metrics import (
    fetch_all,
    fetch_one,
    hourly,
    iso,
    kpi as kpi_block,
    now_sh,
    overview as metrics_overview,
    parse_time_param,
)
from app.routes.events import build_event_filters, client_ip, error_json

# 整个模块只给护士长；护士访问由 require_role 统一 403（并写 DENIED 审计）
router = APIRouter(
    prefix="/api/v1",
    tags=["stats"],
    dependencies=[Depends(require_role("head_nurse"))],
)

# CSV 导出分批拉取，避免一次性 fetchall 把 1 万行读进内存（§12.1）
EXPORT_BATCH = 500

# V1 固定 14 列（§11.5「为什么固定列」：自选列会引入口径不一致）
EXPORT_HEADERS = (
    "事件号",
    "点位",
    "事件时间",
    "入库时间",
    "级别",
    "检测态",
    "处置态",
    "处置结果",
    "误报原因",
    "处置人",
    "接警时间",
    "到场时间",
    "处置时间",
    "响应时长(秒)",
)


# ════════════════════════════════════════════════════════════════════════════
# GET /api/v1/stats/overview
# ════════════════════════════════════════════════════════════════════════════
@router.get("/stats/overview")
def stats_overview(
    frm: str | None = Query(None, alias="from", description="起始时间（含），默认今天 00:00"),
    to: str | None = Query(None, description="结束时间（不含），默认现在"),
    user: CurrentUser = Depends(current_user),
    conn: Any = Depends(get_conn),
) -> Any:
    """5 KPI + 待跟进 + 设备条（§8.4 响应结构）。"""
    ward = scoped_ward(user)
    now = now_sh()
    try:
        period_from = (
            parse_time_param(frm)
            if frm
            else now.replace(hour=0, minute=0, second=0, microsecond=0)
        )
        period_to = parse_time_param(to, end_of_day=True) if to else now
        if period_from >= period_to:
            raise ValueError("from 必须早于 to")
    except ValueError as exc:
        return error_json(400, "TIME_INVALID", str(exc))

    data = metrics_overview(conn, ward, period_from, period_to)

    # 「较昨日」：只有昨日有数据时才给差值，否则 None（§11.5 展示规则）
    yesterday_from = period_from - dt.timedelta(days=1)
    yesterday_to = period_to - dt.timedelta(days=1)
    yesterday = kpi_block(conn, ward, yesterday_from, yesterday_to)
    yesterday_alerts = yesterday["alerts"]
    delta_alerts = data["kpi"]["alerts"] - yesterday_alerts if yesterday_alerts > 0 else None
    delta_falls = (
        data["kpi"]["confirmed_falls"] - yesterday["confirmed_falls"]
        if yesterday_alerts > 0
        else None
    )

    rate = data["kpi"]["false_alarm_rate"]
    median = data["kpi"]["median_response_s"]
    return {
        "ward_id": ward,
        "as_of": iso(now),
        "period": {"from": iso(period_from), "to": iso(period_to)},
        "kpi": {
            "alerts": {"value": data["kpi"]["alerts"], "delta_vs_yesterday": delta_alerts},
            "confirmed_falls": {
                "value": data["kpi"]["confirmed_falls"],
                "delta_vs_yesterday": delta_falls,
            },
            "false_alarm_rate": {
                "value": rate["value"],
                "numerator": rate["numerator"],
                "denominator": rate["denominator"],
                # 分母 < 5 显示「样本不足」，不显示百分比（§11.3 KPI-3）
                "sample_sufficient": rate["denominator"] >= 5,
            },
            "unhandled": {"value": data["kpi"]["unhandled"]},
            "median_response_s": {
                "value": median["value"],
                "n": median["n"],
                # n < 5 显示「样本不足」（§11.3 KPI-5）
                "sample_sufficient": median["n"] >= 5,
            },
        },
        "pending_events": data["pending_events"],
        "devices": data["devices"],
    }


# ════════════════════════════════════════════════════════════════════════════
# GET /api/v1/stats/hourly
# ════════════════════════════════════════════════════════════════════════════
@router.get("/stats/hourly")
def stats_hourly(
    day: str | None = Query(None, description="YYYY-MM-DD，默认今天"),
    user: CurrentUser = Depends(current_user),
    conn: Any = Depends(get_conn),
) -> Any:
    """24 小时告警分布（柱状图）。固定 24 个桶，空桶补 0。"""
    ward = scoped_ward(user)
    try:
        start = parse_time_param(day) if day else now_sh().replace(
            hour=0, minute=0, second=0, microsecond=0
        )
    except ValueError as exc:
        return error_json(400, "TIME_INVALID", str(exc))
    return {
        "ward_id": ward,
        "day": start.date().isoformat(),
        "buckets": hourly(conn, ward, start),
    }


# ════════════════════════════════════════════════════════════════════════════
# GET /api/v1/export/events.csv
# ════════════════════════════════════════════════════════════════════════════
def csv_safe(value: Any) -> str:
    """CSV 单元格防公式注入（§11.5）。"""
    text = "" if value is None else str(value)
    if text and text[0] in ("=", "+", "-", "@", "\t", "\r"):
        text = "'" + text
    return text


def _csv_line(values: list[Any]) -> str:
    """一行 CSV：统一 \\r\\n 换行（Excel 友好），字段按需加引号。"""
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\r\n")
    writer.writerow([csv_safe(v) for v in values])
    return buffer.getvalue()


EXPORT_SELECT = """
SELECT e.event_uid, e.location_text AS location, e.board_ts, e.received_at, e.severity,
       e.detection_state, e.disposition_state, e.outcome, e.false_alarm_reason,
       u.display_name AS assignee_name,
       e.acked_at, e.arrived_at, e.resolved_at,
       EXTRACT(EPOCH FROM (e.acked_at - e.received_at))::int AS response_s
  FROM fg.event e
  JOIN fg.device d ON d.device_id = e.device_id
  LEFT JOIN fg.app_user u ON u.user_id = e.assignee_id
 WHERE {where_sql}
"""


def _export_row(row: dict) -> list[Any]:
    return [
        row["event_uid"],
        row["location"],
        iso(row["board_ts"]),
        iso(row["received_at"]),
        row["severity"],
        row["detection_state"],
        row["disposition_state"],
        row["outcome"],
        row["false_alarm_reason"],
        row["assignee_name"],
        iso(row["acked_at"]),
        iso(row["arrived_at"]),
        iso(row["resolved_at"]),
        row["response_s"],
    ]


@router.get("/export/events.csv")
def export_events_csv(
    request: Request,
    disposition: str | None = Query(None),
    detection: str | None = Query(None),
    severity: str | None = Query(None),
    device: str | None = Query(None),
    point: str | None = Query(None),
    frm: str | None = Query(None, alias="from"),
    to: str | None = Query(None),
    q: str | None = Query(None),
    outcome: str | None = Query(None),
    test: str | None = Query(None),
    assignee: str | None = Query(None),
    user: CurrentUser = Depends(current_user),
    conn: Any = Depends(get_conn),
) -> Any:
    """CSV 导出：与列表同一套筛选条件，utf-8-sig + \\r\\n + 公式注入转义 + EXPORT 审计。"""
    ward = scoped_ward(user)
    try:
        period_from = parse_time_param(frm) if frm else None
        period_to = parse_time_param(to, end_of_day=True) if to else None
        if period_from and period_to and period_from >= period_to:
            raise ValueError("from 必须早于 to")
        assignee_id = None
        if assignee and assignee.strip():
            assignee_id = (
                int(user.user_id) if assignee.strip() == "me" else int(assignee.strip())
            )
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
    except ValueError as exc:
        return error_json(400, "QUERY_INVALID", str(exc))

    # 台账口径与 fg.v_event_fact 一致：测试告警默认不进导出。
    # 但显式传了 test=0 / test=1 时以调用方为准 —— 无条件再追加一条
    # `is_test = FALSE` 会和 `is_test = TRUE` 互斥，让 test=1 的导出恒为空。
    if test not in ("0", "1"):
        where_sql = where_sql + " AND e.is_test = FALSE"
    select_sql = EXPORT_SELECT.format(where_sql=where_sql)

    # 先数一遍，写进 EXPORT 审计（含行数与筛选条件，§11.5 第 ④ 条）
    total_row = fetch_one(
        conn,
        "SELECT count(*) AS n FROM fg.event e"
        " JOIN fg.device d ON d.device_id = e.device_id"
        " WHERE " + where_sql,
        params,
    )
    total = int(total_row["n"]) if total_row else 0
    with conn.transaction():
        audit(
            conn,
            user,
            "EXPORT",
            "event",
            None,
            "OK",
            detail={
                "rows": total,
                "filters": {
                    "disposition": disposition,
                    "detection": detection,
                    "severity": severity,
                    "device": device,
                    "from": iso(period_from),
                    "to": iso(period_to),
                    "q": q,
                    "outcome": outcome,
                    "test": test,
                    "assignee": assignee,
                },
            },
            client_ip=client_ip(request),
        )

    # 分批按 keyset 读取（不一次性 fetchall），在内存里拼装。
    # V1 一年约 1 万条 ≈ 2 MB，内存拼装足够；同时避免「流式响应 + 依赖连接生命周期」
    # 的耦合（连接在响应发送完毕后由依赖回收）。
    chunks = [chr(0xFEFF), _csv_line(list(EXPORT_HEADERS))]  # BOM：Excel 打开 utf-8 不乱码
    last_ts: dt.datetime | None = None
    last_uid: str | None = None
    while True:
        if last_ts is None:
            batch = fetch_all(
                conn,
                select_sql + " ORDER BY e.received_at DESC, e.event_uid DESC LIMIT %s",
                params + [EXPORT_BATCH],
            )
        else:
            batch = fetch_all(
                conn,
                select_sql
                + " AND (e.received_at, e.event_uid) < (%s, %s::uuid)"
                + " ORDER BY e.received_at DESC, e.event_uid DESC LIMIT %s",
                params + [last_ts, last_uid, EXPORT_BATCH],
            )
        if not batch:
            break
        for row in batch:
            chunks.append(_csv_line(_export_row(row)))
        if len(batch) < EXPORT_BATCH:
            break
        last_ts = batch[-1]["received_at"]
        last_uid = str(batch[-1]["event_uid"])

    filename = f"事件台账_{now_sh().strftime('%Y%m%d_%H%M')}.csv"
    disposition_header = (
        'attachment; filename="events.csv"; '
        f"filename*=UTF-8''{quote(filename, safe='')}"
    )
    return Response(
        content="".join(chunks).encode("utf-8"),
        media_type="text/csv; charset=utf-8",
        headers={
            "Content-Disposition": disposition_header,
            "Cache-Control": "no-store",
        },
    )
