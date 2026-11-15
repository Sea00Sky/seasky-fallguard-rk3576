"""app/metrics.py —— 统计口径唯一真源（对应设计文档 §11）。"""

from __future__ import annotations

import datetime as dt
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from .deps import zone_clauses

# ── 时区 ────────────────────────────────────────────────────────────────────
# Windows 缺 tzdata 时 ZoneInfo 会抛 ZoneInfoNotFoundError，兜底用固定 +08:00
# （上海无夏令时，两者等价）。
try:
    TZ = ZoneInfo("Asia/Shanghai")
except ZoneInfoNotFoundError:  # pragma: no cover - 仅 Windows 未装 tzdata 时走到
    TZ = dt.timezone(dt.timedelta(hours=8), "Asia/Shanghai")


# ── 口径常量（§11.1，唯一真源，别处不许再定义） ────────────────────────────
# 「确认跌倒」只看这三个 outcome；NO_ABNORMALITY（到场无异常）不算确认跌倒。
CONFIRMED_FALL_OUTCOMES = ("ASSISTED", "MEDICAL", "HOSPITALIZED")
# 「已闭环」：误报率的分母只含这三态，未闭环不进分母。
CLOSED_STATES = ("RESOLVED", "FALSE_ALARM", "CLOSED")
# 「未处置」：不限时间区间，昨天没处置的今天还得处置。
UNHANDLED_STATES = ("PENDING", "ACKED", "ON_SCENE")


# ── 时间工具 ────────────────────────────────────────────────────────────────
def now_sh() -> dt.datetime:
    """当前时间（Asia/Shanghai，带时区）。"""
    return dt.datetime.now(TZ)


def iso(value: Any) -> str | None:
    """把 datetime 统一序列化为 `` T22:40:12+08:00``。"""
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, dt.datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=dt.timezone.utc)
        return value.astimezone(TZ).isoformat(timespec="seconds")
    return str(value)


def parse_time_param(value: str, *, end_of_day: bool = False) -> dt.datetime:
    """解析接口时间参数。"""
    text = (value or "").strip()
    if not text:
        raise ValueError("时间参数为空")
    keyword = text.lower()
    if keyword in ("today", "yesterday"):
        day = now_sh().date() - dt.timedelta(days=1 if keyword == "yesterday" else 0)
        base = dt.datetime(day.year, day.month, day.day, tzinfo=TZ)
        return base + dt.timedelta(days=1) if end_of_day else base
    # 先按纯日期尝试（date.fromisoformat 不接受带 T 的串，不会误判）
    try:
        day = dt.date.fromisoformat(text)
        base = dt.datetime(day.year, day.month, day.day, tzinfo=TZ)
        return base + dt.timedelta(days=1) if end_of_day else base
    except ValueError:
        pass
    normalized = text[:-1] + "+00:00" if text.endswith("Z") else text
    parsed = dt.datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=TZ)
    return parsed


def day_start(day: Any) -> dt.datetime:
    """把 date / datetime / 'YYYY-MM-DD' 归一成当天 00:00（Asia/Shanghai）。"""
    if isinstance(day, dt.datetime):
        d = day.astimezone(TZ).date() if day.tzinfo else day.date()
    elif isinstance(day, dt.date):
        d = day
    else:
        d = dt.date.fromisoformat(str(day).strip())
    return dt.datetime(d.year, d.month, d.day, tzinfo=TZ)


# ── 入库口径：严重度（§6.6） ────────────────────────────────────────────────
def severity_of(prob_fall: float, coverage: float) -> str:
    """入库时算 severity。只用板端真实存在的两个量，规则可解释、可复现。"""
    prob = float(prob_fall or 0.0)
    cov = float(coverage or 0.0)
    if prob >= 0.85 and cov >= 0.80:
        return "S1"
    if prob >= 0.65 and cov >= 0.70:
        return "S2"
    return "S3"


# ── 行工厂兼容层 ────────────────────────────────────────────────────────────
def fetch_all(conn, sql: str, params: Any = None) -> list[dict]:
    """执行查询并返回 ``list[dict]``。"""
    with conn.cursor() as cur:
        cur.execute(sql, params)
        names = [d.name for d in (cur.description or [])]
        rows = cur.fetchall()
    return [r if isinstance(r, dict) else dict(zip(names, r)) for r in rows]


def fetch_one(conn, sql: str, params: Any = None) -> dict | None:
    """执行查询并返回第一行（``dict``）或 None。INSERT ... RETURNING 也走这里。"""
    rows = fetch_all(conn, sql, params)
    return rows[0] if rows else None


# ── KPI（§11.3，SQL 与文档逐条对应） ───────────────────────────────────────
def _zone_filter(alias: str, zone_ids: tuple[str, ...] | None) -> tuple[str, list]:
    """区域权限条件，返回可直接拼进 ``WHERE`` 的 ``" AND ..."`` 片段与参数。"""
    clauses, params = zone_clauses(alias, zone_ids)
    if not clauses:
        return "", []
    return " AND " + " AND ".join(clauses), params


def kpi(conn, ward_id: str, frm: dt.datetime, to: dt.datetime,
        zone_ids: tuple[str, ...] | None = None) -> dict:
    """五个 KPI 的原始值。区间一律左闭右开 ``[frm, to)``，时间基准 received_at。"""

    # 视图没有别名，alias 传空串
    zsql, zparams = _zone_filter("", zone_ids)

    alerts_row = fetch_one(
        conn,
        """
        SELECT count(*) AS n
          FROM fg.v_event_fact
         WHERE ward_id = %s AND ever_confirmed
           AND received_at >= %s AND received_at < %s
        """ + zsql,
        (ward_id, frm, to, *zparams),
    )

    # KPI-2 确认跌倒：已闭环且 outcome ∈ {ASSISTED, MEDICAL, HOSPITALIZED}
    falls_row = fetch_one(
        conn,
        """
        SELECT count(*) AS n
          FROM fg.v_event_fact
         WHERE ward_id = %s AND is_confirmed_fall
           AND received_at >= %s AND received_at < %s
        """ + zsql,
        (ward_id, frm, to, *zparams),
    )

    # KPI-3 误报率：FALSE_ALARM 数 ÷ 已闭环数（未闭环不进分母）
    rate_row = fetch_one(
        conn,
        """
        SELECT
          count(*) FILTER (WHERE is_false_alarm)                     AS numerator,
          count(*) FILTER (WHERE is_closed)                          AS denominator,
          ROUND(count(*) FILTER (WHERE is_false_alarm)::numeric
                / NULLIF(count(*) FILTER (WHERE is_closed), 0), 4)   AS rate
          FROM fg.v_event_fact
         WHERE ward_id = %s AND received_at >= %s AND received_at < %s
        """ + zsql,
        (ward_id, frm, to, *zparams),
    )

    # KPI-4 未处置：不限区间（积压也要算），状态白名单来自常量
    unhandled_row = fetch_one(
        conn,
        """
        SELECT count(*) AS n
          FROM fg.v_event_fact
         WHERE ward_id = %s AND disposition_state = ANY(%s)
        """ + zsql,
        (ward_id, list(UNHANDLED_STATES), *zparams),
    )

    # KPI-5 中位响应：median(acked_at - received_at)，只算有 acked_at 的
    median_row = fetch_one(
        conn,
        """
        SELECT
          ROUND(percentile_cont(0.5) WITHIN GROUP (ORDER BY response_s))::int AS median_s,
          count(*) AS n
          FROM fg.v_event_fact
         WHERE ward_id = %s AND received_at >= %s AND received_at < %s
           AND acked_at IS NOT NULL
        """ + zsql,
        (ward_id, frm, to, *zparams),
    )

    def _int(row: dict | None, key: str) -> int:
        return int(row[key]) if row and row.get(key) is not None else 0

    numerator = _int(rate_row, "numerator")
    denominator = _int(rate_row, "denominator")
    rate = rate_row.get("rate") if rate_row else None
    median_s = median_row.get("median_s") if median_row else None

    return {
        "alerts": _int(alerts_row, "n"),
        "confirmed_falls": _int(falls_row, "n"),
        "false_alarm_rate": {
            # numeric 在 psycopg 里是 Decimal，JSON 不能直接序列化，转 float
            "value": float(rate) if rate is not None else None,
            "numerator": numerator,
            "denominator": denominator,
        },
        "unhandled": _int(unhandled_row, "n"),
        "median_response_s": {
            "value": int(median_s) if median_s is not None else None,
            "n": _int(median_row, "n"),
        },
    }


def pending_by_location(conn, ward_id: str, received_since=None,
                        zone_ids: tuple[str, ...] | None = None) -> dict[str, int]:
    """按点位聚合的未闭环条数：{location: n}。"""
    sql = """
        SELECT COALESCE(e.location_text, '') AS loc, count(*) AS n
          FROM fg.event e
         WHERE e.ward_id = %s AND e.is_test = FALSE
           AND e.disposition_state = ANY(%s)
    """
    params: list = [ward_id, list(UNHANDLED_STATES)]
    zsql, zparams = _zone_filter("e", zone_ids)
    sql += zsql
    params.extend(zparams)
    if received_since is not None:
        # 灯上的「N 条待处置」必须和队列同一天界，否则灯亮着、点进去是空的。
        sql += " AND e.received_at >= %s"
        params.append(received_since)
    # 七改：device.location 已删，位置只认入库快照（e.location_text）。
    # 没绑定点位的设备在 ingest 里用设备码当快照，口径与未绑定设备那盏灯一致。
    rows = fetch_all(conn, sql + " GROUP BY COALESCE(e.location_text, '')", tuple(params))
    return {(r["loc"] or ""): int(r["n"]) for r in rows}


def pending_by_device(conn, ward_id: str,
                      zone_ids: tuple[str, ...] | None = None) -> dict[str, int]:
    """按设备聚合的未闭环条数：{device_code: n}。"""
    zsql, zparams = _zone_filter("e", zone_ids)
    rows = fetch_all(
        conn,
        """
        SELECT d.device_code, count(*) AS n
          FROM fg.event e
          JOIN fg.device d ON d.device_id = e.device_id
         WHERE e.ward_id = %s AND e.is_test = FALSE
           AND e.disposition_state = ANY(%s)
        """ + zsql + """
         GROUP BY d.device_code
        """,
        (ward_id, list(UNHANDLED_STATES), *zparams),
    )
    return {r["device_code"]: int(r["n"]) for r in rows}


def _pending_events(conn, ward_id: str, limit: int = 5,
                    zone_ids: tuple[str, ...] | None = None) -> list[dict]:
    """病区总览右栏「待跟进」：未闭环事件。"""
    zsql, zparams = _zone_filter("e", zone_ids)
    rows = fetch_all(
        conn,
        """
        SELECT e.event_uid, e.severity, e.detection_state, e.disposition_state,
               e.received_at, e.person_id, e.prob_fall, e.coverage,
               (e.detection_state = 'CONFIRMED'
                OR (e.detection_state = 'RETRACTED' AND NOT e.orphan_retract)) AS ever_confirmed,
               d.device_code, e.location_text AS location,
               (s.snapshot_id IS NOT NULL) AS has_snapshot
          FROM fg.event e
          JOIN fg.device d ON d.device_id = e.device_id
          LEFT JOIN fg.snapshot s ON s.event_uid = e.event_uid
         WHERE e.ward_id = %s AND e.is_test = FALSE
           AND e.disposition_state = ANY(%s)
        """ + zsql + """
         ORDER BY CASE e.detection_state WHEN 'CONFIRMED' THEN 0 ELSE 1 END,
                  CASE e.severity WHEN 'S1' THEN 1 WHEN 'S2' THEN 2 ELSE 3 END,
                  CASE e.disposition_state WHEN 'PENDING' THEN 0 ELSE 1 END,
                  e.received_at ASC
         LIMIT %s
        """,
        (ward_id, list(UNHANDLED_STATES), *zparams, limit),
    )
    return [
        {
            "event_uid": str(r["event_uid"]),
            "severity": r["severity"],
            "detection_state": r["detection_state"],
            # 带上历史口径：将来谁把这个 dict 喂给 alert_card，徽章才不会错
            "ever_confirmed": bool(r["ever_confirmed"]),
            "disposition_state": r["disposition_state"],
            "received_at": iso(r["received_at"]),
            "person_id": r["person_id"],
            "prob_fall": r["prob_fall"],
            "coverage": r["coverage"],
            "device_code": r["device_code"],
            "location": r["location"],
            "has_snapshot": bool(r["has_snapshot"]),
        }
        for r in rows
    ]


# ── 工作台处置队列（ 改版） ──────────────────────────────────────
# 四段：前三段是未闭环，最后一段是已闭环。
# 键名同时是 URL 参数 `?disposition=` 的取值，也是模板里 tab 的顺序。
WORKLIST_SEGMENTS = ("PENDING", "ACKED", "ON_SCENE", "CLOSED")

_SEGMENT_STATES = {
    "PENDING": ("PENDING",),
    "ACKED": ("ACKED",),
    "ON_SCENE": ("ON_SCENE",),
    "CLOSED": CLOSED_STATES,
}

# 每段的排序：回答的是「先做哪一条」，不是「哪条新」。
# PENDING：模型仍确认的在前（撤回的先下沉，但还在队列里），再按严重度、等得久
# ACKED / ON_SCENE：已经认领了，按严重度、等得久（谁最早被晾着）
# CLOSED：最近闭环的排前面（刚处理完的就在顶上，方便回看）
_SEGMENT_ORDER = {
    "PENDING": (
        "CASE e.detection_state WHEN 'CONFIRMED' THEN 0 ELSE 1 END, "
        "CASE e.severity WHEN 'S1' THEN 1 WHEN 'S2' THEN 2 ELSE 3 END, "
        "e.received_at ASC"
    ),
    "ACKED": (
        "CASE e.severity WHEN 'S1' THEN 1 WHEN 'S2' THEN 2 ELSE 3 END, "
        "e.received_at ASC"
    ),
    "ON_SCENE": (
        "CASE e.severity WHEN 'S1' THEN 1 WHEN 'S2' THEN 2 ELSE 3 END, "
        "e.received_at ASC"
    ),
    "CLOSED": "COALESCE(e.resolved_at, e.received_at) DESC",
}


def normalize_segment(value: str | None) -> str:
    """URL 里的分段参数 → 合法分段。非法值一律回到 PENDING。"""
    seg = (value or "").strip().upper()
    return seg if seg in WORKLIST_SEGMENTS else "PENDING"


def segment_disposition(segment: str | None) -> str:
    """分段 → 告警中心 ``?disposition=`` 的取值（逗号分隔）。"""
    return ",".join(_SEGMENT_STATES[normalize_segment(segment)])


def point_filter(points: str | list[str] | None) -> tuple[str, list]:
    """点位筛选的 SQL 片段与参数。``("", [])`` = 没筛。"""
    vals = [points] if isinstance(points, str) else list(points or [])
    vals = [v for v in vals if v]
    if not vals:
        return "", []
    if len(vals) == 1:
        return " AND COALESCE(e.location_text, '') = %s", [vals[0]]
    return " AND COALESCE(e.location_text, '') = ANY(%s)", [vals]


def disposition_counts(conn, ward_id: str, closed_since=None,
                       points: str | list[str] | None = None,
                       received_since=None,
                       zone_ids: tuple[str, ...] | None = None) -> dict[str, int]:
    """工作台四段计数：{PENDING, ACKED, ON_SCENE, CLOSED}。"""
    sql = """
        SELECT e.disposition_state, count(*) AS n
          FROM fg.event e
          JOIN fg.device d ON d.device_id = e.device_id
         WHERE e.ward_id = %s AND e.is_test = FALSE
           AND e.disposition_state = ANY(%s)
    """
    params: list = [ward_id, list(UNHANDLED_STATES + CLOSED_STATES)]
    zsql, zparams = _zone_filter("e", zone_ids)
    sql += zsql
    params += zparams
    if closed_since is not None:
        sql += " AND (e.disposition_state <> ALL(%s) OR COALESCE(e.resolved_at, e.received_at) >= %s)"
        params += [list(CLOSED_STATES), closed_since]
    if received_since is not None:
        # 只截未闭环三段（见 docstring）；CLOSED 由上面的 closed_since 管。
        sql += " AND (e.disposition_state = ANY(%s) OR e.received_at >= %s)"
        params += [list(CLOSED_STATES), received_since]
    frag, extra = point_filter(points)
    sql += frag
    params += extra
    rows = fetch_all(conn, sql + " GROUP BY e.disposition_state", tuple(params))
    out = {seg: 0 for seg in WORKLIST_SEGMENTS}
    for r in rows:
        state = r["disposition_state"]
        seg = "CLOSED" if state in CLOSED_STATES else state
        if seg in out:
            out[seg] += int(r["n"])
    return out


def worklist(conn, ward_id: str, segment: str = "PENDING", limit: int = 30,
             closed_since=None, points: str | list[str] | None = None,
             received_since=None,
             zone_ids: tuple[str, ...] | None = None) -> list[dict]:
    """处置队列的一个分段（工作台主区）。"""
    seg = normalize_segment(segment)
    order = _SEGMENT_ORDER[seg]
    where = "e.ward_id = %s AND e.is_test = FALSE AND e.disposition_state = ANY(%s)"
    params: list = [ward_id, list(_SEGMENT_STATES[seg])]
    zsql, zparams = _zone_filter("e", zone_ids)
    where += zsql
    params += zparams
    if seg == "CLOSED" and closed_since is not None:
        # 与 disposition_counts 同一口径：按闭环时间截，不按收到时间（理由见那个函数）。
        where += " AND COALESCE(e.resolved_at, e.received_at) >= %s"
        params.append(closed_since)
    elif received_since is not None:
        # 「首页只看今天」：未闭环三段按收到时间截到自然日 0 点（与告警中心默认同一个界）。
        where += " AND e.received_at >= %s"
        params.append(received_since)
    if points:
        frag, extra = point_filter(points)
        where += frag
        params += extra
    params.append(limit)
    rows = fetch_all(
        conn,
        f"""
        SELECT e.event_uid, e.severity, e.detection_state, e.disposition_state,
               e.received_at, e.acked_at, e.resolved_at, e.person_id, e.revision,
               (e.detection_state = 'CONFIRMED'
                OR (e.detection_state = 'RETRACTED' AND NOT e.orphan_retract)) AS ever_confirmed,
               d.device_code,
               e.location_text AS location,
               u.display_name AS assignee_name,
               (s.snapshot_id IS NOT NULL) AS has_snapshot
          FROM fg.event e
          JOIN fg.device d ON d.device_id = e.device_id
          LEFT JOIN fg.app_user u ON u.user_id = e.assignee_id
          LEFT JOIN fg.snapshot s ON s.event_uid = e.event_uid
         WHERE {where}
         ORDER BY {order}
         LIMIT %s
        """,
        tuple(params),
    )
    return [
        {
            "event_uid": str(r["event_uid"]),
            "severity": r["severity"],
            "detection_state": r["detection_state"],
            "disposition_state": r["disposition_state"],
            "ever_confirmed": bool(r["ever_confirmed"]),
            "received_at": r["received_at"],
            "acked_at": r["acked_at"],
            "resolved_at": r["resolved_at"],
            "person_id": r["person_id"],
            "revision": r["revision"],
            "device_code": r["device_code"],
            "location": r["location"],
            "assignee_name": r["assignee_name"],
            "has_snapshot": bool(r["has_snapshot"]),
        }
        for r in rows
    ]


def _devices(conn, ward_id: str,
             zone_ids: tuple[str, ...] | None = None) -> list[dict]:
    """设备条：只读。V1 不判在线，只给「最后事件时间」（§9.6）。"""
    zsql, zparams = _zone_filter("lp", zone_ids)
    rows = fetch_all(
        conn,
        """
        SELECT d.device_code, d.device_name, d.last_event_at,
               d.last_seen_at, d.model_ver, d.is_active,
               lp.point_id AS live_point_id, lp.name AS live_point_name,
               lp.zone_id AS live_zone_id, lp.light_key AS live_light_key
          FROM fg.device d
          LEFT JOIN fg.point lp
                 ON lp.device_id = d.device_id AND lp.is_active
         WHERE COALESCE(lp.ward_id, d.ward_id) = %s
        """ + zsql + """
         ORDER BY d.device_code
        """,
        (ward_id, *zparams),
    )
    return [
        {
            "device_code": r["device_code"],
            "device_name": r.get("device_name") or r["device_code"],
            # 位置 = 绑定了这台设备的点位（七改：设备行上已无位置副本）
            "location": (r.get("live_light_key") or r.get("live_point_name")
                         or r["device_code"]),
            "zone_id": str(r["live_zone_id"]) if r.get("live_zone_id") else None,
            "last_seen_at": iso(r.get("last_seen_at")),
            "last_event_at": iso(r["last_event_at"]),
            "model_ver": r["model_ver"],
            # UI 显示 v7F3A21C9：模型 hash 的低 32 位，16 进制大写
            "model_ver_hex": (
                f"{int(r['model_ver']) & 0xFFFFFFFF:08X}"
                if r["model_ver"] is not None
                else None
            ),
            "is_active": bool(r["is_active"]),
        }
        for r in rows
    ]


def overview(conn, ward_id: str, frm: dt.datetime, to: dt.datetime,
             received_since=None,
             zone_ids: tuple[str, ...] | None = None) -> dict:
    """病区总览数据（不含 ward_id / as_of / period，那三个由路由层拼）。"""
    return {
        "kpi": kpi(conn, ward_id, frm, to, zone_ids=zone_ids),
        "pending_events": _pending_events(conn, ward_id, zone_ids=zone_ids),
        # received_since 只给工作台传（「首页只看今天」）；护士长总览不传 = 不限，看全量积压。
        "pending_by_location": pending_by_location(
            conn, ward_id, received_since=received_since, zone_ids=zone_ids),
        "devices": _devices(conn, ward_id, zone_ids=zone_ids),
    }


def hourly(conn, ward_id: str, day: Any) -> list[dict]:
    """24 小时告警分布（柱状图）：固定返回 24 个桶，没有数据的小时补 0。"""
    start = day_start(day)
    end = start + dt.timedelta(days=1)
    rows = fetch_all(
        conn,
        """
        SELECT EXTRACT(HOUR FROM received_at AT TIME ZONE 'Asia/Shanghai')::int AS hour,
               count(*) AS count
          FROM fg.v_event_fact
         WHERE ward_id = %s AND ever_confirmed
           AND received_at >= %s AND received_at < %s
         GROUP BY 1
        """,
        (ward_id, start, end),
    )
    buckets = {int(r["hour"]): int(r["count"]) for r in rows if r["hour"] is not None}
    return [{"hour": h, "count": buckets.get(h, 0)} for h in range(24)]
