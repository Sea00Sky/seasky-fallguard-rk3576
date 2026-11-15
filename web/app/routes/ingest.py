"""app/routes/ingest.py —— 板端接入（push 事件 + 快照上传）。"""

from __future__ import annotations

import asyncio
import datetime as dt
import hashlib
import hmac
import json
import os
import re
import secrets
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import JSONResponse, Response
from psycopg.types.json import Jsonb

from app import clips, live_meta
from app.attribution import (
    LIVE_POINT_FIELDS,
    LIVE_POINT_JOIN,
    apply_stream_attribution,
)
from app.config import settings
from app.db import get_conn, get_pool
from app.deps import audit
from app.metrics import TZ, fetch_one, iso, now_sh, severity_of
from app.routes.events import client_ip, error_json, publish_event

router = APIRouter(prefix="/api/v1/ingest", tags=["ingest"])

RETRACT_WINDOW_SQL = "now() - interval '10 minutes'"
# 兜底级专用窗口：2 分钟，比上面紧得多。
# 理由：兜底存在的唯一正当场景是「网关没找到原行、给撤回重新分配了一个 seq」
# （db.c 的 db_update_existing 落空 → db_insert_new）。那种情况撤回紧跟着确认到达，
# 秒级。而生产上真出过的串行事故是同逻辑键相隔 165 s（1287/1288）——
# 2 分钟窗口正好把 165 s 排除在外。宁可插孤儿行（可见、不计 KPI），
# 也不要静默把上一轮的告警撤掉。
RETRACT_FUZZY_WINDOW_SQL = "now() - interval '2 minutes'"
# 快照与事件时间差上限（§8.5）
SNAPSHOT_MAX_LAG_S = 120
# 快照大小上限：板端 320×240 JPEG q70 约 15–25 KB，2 MB 已是极宽松的上限
SNAPSHOT_MAX_BYTES = 2 * 1024 * 1024


# ════════════════════════════════════════════════════════════════════════════
# 基础工具
# ════════════════════════════════════════════════════════════════════════════
def uuid7() -> uuid.UUID:
    """生成 UUIDv7（时间有序，索引友好；Python 3.12 标准库还没有 uuid7）。"""
    millis = int(time.time() * 1000) & 0xFFFFFFFFFFFF
    rand = secrets.randbits(74)
    rand_a = (rand >> 62) & 0x0FFF
    rand_b = rand & 0x3FFFFFFFFFFFFFFF
    value = (millis << 80) | (0x7 << 76) | (rand_a << 64) | (0b10 << 62) | rand_b
    return uuid.UUID(int=value)


def _parse_board_ts(raw: Any) -> dt.datetime:
    """解析板端 ``ts``。"""
    if isinstance(raw, (int, float)):
        return dt.datetime.fromtimestamp(float(raw), tz=dt.timezone.utc)
    text = str(raw).strip()
    if not text:
        raise _BadBody("event.ts 为空")
    if "." in text:
        seconds_part, frac_part = text.split(".", 1)
        frac_part = (frac_part + "000000")[:6]  # 毫秒 → 微秒，超长截断
    else:
        seconds_part, frac_part = text, "000000"
    try:
        seconds = int(seconds_part)
        micros = int(frac_part.ljust(6, "0")[:6])
    except ValueError as exc:
        raise _BadBody("event.ts 格式不是 <sec>.<ms>") from exc
    return dt.datetime.fromtimestamp(seconds, tz=dt.timezone.utc) + dt.timedelta(
        microseconds=micros
    )


class _BadBody(ValueError):
    """请求体字段缺失或类型不对（统一转 422）。"""


def _req_int(ev: dict, key: str) -> int:
    try:
        return int(ev[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise _BadBody(f"event.{key} 缺失或不是整数") from exc


def _opt_int(ev: dict, key: str) -> int | None:
    value = ev.get(key)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise _BadBody(f"event.{key} 不是整数") from exc


def _req_float(ev: dict, key: str) -> float:
    try:
        return float(ev[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise _BadBody(f"event.{key} 缺失或不是数值") from exc


def _opt_float(ev: dict, key: str) -> float | None:
    value = ev.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise _BadBody(f"event.{key} 不是数值") from exc


def _opt_bbox(ev: dict) -> list[float] | None:
    bbox = ev.get("bbox")
    if bbox is None:
        return None
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        raise _BadBody("event.bbox 必须是 4 个数的数组")
    try:
        return [float(v) for v in bbox]
    except (TypeError, ValueError) as exc:
        raise _BadBody("event.bbox 必须是 4 个数的数组") from exc


def _parse_event(ev: dict) -> dict:
    """把 push body 的 ``event`` 对象解析成入库需要的字段（缺字段一律 422）。"""
    state = _req_int(ev, "state")
    parsed = {
        "state": state,
        "board_seq": _req_int(ev, "seq"),  # seq = 网关 SQLite row_seq，不是 evt->seq
        "board_ts": _parse_board_ts(ev.get("ts")),
        "person_id": _req_int(ev, "person_id"),
        "window_start": _req_int(ev, "window_start"),
        "window_end": _opt_int(ev, "window_end"),
        "model_ver": _req_int(ev, "model_ver"),
        "prob_fall": _req_float(ev, "prob_fall"),
        "coverage": _req_float(ev, "coverage"),
        "hybrid_score": _opt_float(ev, "hybrid_score"),
        "bbox": _opt_bbox(ev),
        "reason": (str(ev.get("reason")).strip() or None) if ev.get("reason") is not None else None,
        "snapshot_seq": _opt_int(ev, "snapshot_seq"),
    }
    return parsed


def _auth_device(conn, device_code: str, ingest_key: str):
    """设备校验（§8.5 第 ①② 步）：存在且启用 → 404；sha256 不符 → 401。"""
    dev = fetch_one(
        conn,
        """
        SELECT d.device_id, d.ward_id, d.device_code,
               d.device_name, d.ingest_key_hash, d.is_active, d.stream_path,
               """ + LIVE_POINT_FIELDS + """
          FROM fg.device d
               """ + LIVE_POINT_JOIN + """
         WHERE d.device_code = %s
        """,
        (device_code,),
    )
    if dev is None or not dev["is_active"]:
        return None, error_json(404, "DEVICE_NOT_FOUND", "设备不存在或已停用")
    expected = (dev["ingest_key_hash"] or "").strip().lower()
    actual = hashlib.sha256(ingest_key.encode("utf-8")).hexdigest()
    # compare_digest 防时序侧信道
    if not hmac.compare_digest(expected, actual):
        return None, error_json(401, "INGEST_KEY_INVALID", "接入密钥不正确")
    return apply_stream_attribution(dev), None


class _SystemActor:
    """系统动作的伪用户。"""

    def __init__(self, ward_id: str) -> None:
        self.user_id = None
        self.display_name = "系统"
        self.role = "system"
        self.ward_id = ward_id
        self.must_change_password = False


def _touch_device(conn, device: dict, model_ver: int) -> None:
    """更新设备最后事件时间与最近上报的模型版本（§8.5 第 ⑥ 步）。"""
    conn.execute(
        "UPDATE fg.device SET last_event_at = now(), model_ver = %s WHERE device_id = %s",
        (model_ver, device["device_id"]),
    )


def _system_actor_id(conn) -> int | None:
    """找系统账号（login_name='system'）。"""
    row = fetch_one(conn, "SELECT user_id FROM fg.app_user WHERE login_name = %s LIMIT 1", ("system",))
    return int(row["user_id"]) if row else None


def _record_retract_action(conn, event_uid: Any, reason: str | None) -> None:
    """撤回写 event_action(RETRACT)。没有系统账号时跳过（见 _system_actor_id 说明）。"""
    actor_id = _system_actor_id(conn)
    if actor_id is None:
        return
    conn.execute(
        """
        INSERT INTO fg.event_action
            (event_uid, actor_id, action, from_state, to_state, reason)
        VALUES (%s, %s, 'RETRACT', 'CONFIRMED', 'RETRACTED', %s)
        """,
        (event_uid, actor_id, reason),
    )


def _ingest_audit(
    conn, dev: dict, parsed: dict, event_uid: Any, extra: dict, request: Request | None = None
) -> None:
    """写 INGEST 审计（actor_id=NULL，ward_id 落到设备所属病区，IP 记板端地址）。"""
    detail = {
        "device_code": dev["device_code"],
        "board_seq": parsed["board_seq"],
        "state": parsed["state"],
        "person_id": parsed["person_id"],
        "window_start": parsed["window_start"],
        "model_ver": parsed["model_ver"],
        "prob_fall": parsed["prob_fall"],
        "coverage": parsed["coverage"],
        "reason": parsed["reason"],
    }
    detail.update(extra)
    audit(
        conn,
        _SystemActor(dev["ward_id"]),
        "INGEST",
        "event",
        str(event_uid),
        "OK",
        detail=detail,
        client_ip=client_ip(request),
    )


# ════════════════════════════════════════════════════════════════════════════
# POST /api/v1/ingest/{device_code}/{ingest_key}
# ════════════════════════════════════════════════════════════════════════════
INSERT_CONFIRMED_SQL = """
INSERT INTO fg.event (
    event_uid, device_id, ward_id, board_seq, board_ts,
    person_id, window_start, window_end, model_ver,
    prob_fall, coverage, hybrid_score, bbox,
    severity, detection_state, disposition_state,
    location_text, zone_id
) VALUES (
    %s, %s, %s, %s, %s,
    %s, %s, %s, %s,
    %s, %s, %s, %s,
    %s, 'CONFIRMED', 'PENDING',
    %s, %s
)
ON CONFLICT (device_id, board_seq, board_ts) DO NOTHING
RETURNING event_uid, received_at
"""

INSERT_ORPHAN_SQL = """
INSERT INTO fg.event (
    event_uid, device_id, ward_id, board_seq, board_ts,
    person_id, window_start, window_end, model_ver,
    prob_fall, coverage, hybrid_score, bbox,
    severity, detection_state, disposition_state, orphan_retract,
    location_text, zone_id
) VALUES (
    %s, %s, %s, %s, %s,
    %s, %s, %s, %s,
    %s, %s, %s, %s,
    %s, 'RETRACTED', 'PENDING', TRUE,
    %s, %s
)
ON CONFLICT (device_id, board_seq, board_ts) DO NOTHING
RETURNING event_uid, received_at
"""

EXACT_EVENT_SQL = """
SELECT event_uid, received_at, detection_state, disposition_state, zone_id
  FROM fg.event
 WHERE device_id = %s AND board_seq = %s AND board_ts = %s
"""

MATCH_RETRACT_BY_SEQ_SQL = f"""
SELECT event_uid, received_at, detection_state, disposition_state, zone_id
  FROM fg.event
 WHERE device_id = %s
   AND board_seq = %s
   AND orphan_retract = FALSE
   AND received_at > {RETRACT_WINDOW_SQL}
 ORDER BY received_at DESC
 LIMIT 1
 FOR UPDATE
"""

MATCH_ANY_RETRACT_SQL = f"""
SELECT event_uid FROM fg.event
 WHERE device_id = %s
   AND person_id = %s
   AND window_start = %s
   AND model_ver = %s
   AND detection_state = 'RETRACTED'
   AND received_at > {RETRACT_FUZZY_WINDOW_SQL}
 ORDER BY received_at DESC
 LIMIT 1
"""

# ② 兜底级：只在精确级落空时才用，且护栏比原来紧得多 ——
# 只匹配 CONFIRMED 非孤儿行。原来的写法允许命中 RETRACTED，配合
# 「取 received_at 最新的一条」，在生产上真出过事：1287/1288 同逻辑键相隔
# 165 s，若 1288 的 confirm 丢包，1288 的撤回会命中已经撤回过的 1287 ⇒
# 被当成重复吞掉，新告警卡永远红着且没有台账。宁可插孤儿行（可见、不计入
# KPI）也不要静默错配。
MATCH_RETRACT_SQL = f"""
SELECT event_uid, received_at, detection_state, disposition_state, zone_id
  FROM fg.event
 WHERE device_id = %s
   AND person_id = %s
   AND window_start = %s
   AND model_ver = %s
   AND detection_state = 'CONFIRMED'
   AND orphan_retract = FALSE
   AND received_at > {RETRACT_FUZZY_WINDOW_SQL}
 ORDER BY received_at DESC
 LIMIT 1
 FOR UPDATE
"""


@router.post("/{device_code}/{ingest_key}")
async def ingest_push(
    device_code: str,
    ingest_key: str,
    request: Request,
    conn: Any = Depends(get_conn),
) -> Any:
    """接收板端 push body（原样兼容），按 §7.4 规则幂等入库。"""
    dev, denied = _auth_device(conn, device_code, ingest_key)
    if denied is not None:
        return denied

    # 位置快照：绑定点位时用点位的门灯键；没绑定的设备没有位置（七改），
    # 退回设备码 —— 与护士端未绑定设备那盏灯用的键逐字一致（doorlight 的 fallback）。
    # 快照在入库时写死，之后改绑定不会影响历史事件。
    loc_text = dev["location"] or dev["device_code"]

    raw = await request.body()
    if not raw:
        return error_json(422, "BODY_INVALID", "请求体为空")
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return error_json(422, "BODY_INVALID", "请求体不是合法 JSON")

    # ③ type 必须是 fall_alert
    if not isinstance(payload, dict) or payload.get("type") != "fall_alert":
        return error_json(422, "BODY_INVALID", 'type 必须是 "fall_alert"')
    ev = payload.get("event")
    if not isinstance(ev, dict):
        return error_json(422, "BODY_INVALID", "缺少 event 对象")

    try:
        parsed = _parse_event(ev)
    except _BadBody as exc:
        return error_json(422, "BODY_INVALID", str(exc))

    # ④ state 只处理 1（confirmed）/ 2（retracted）；0（suspected）网关本不该推。
    # 真收到也返回 200，免得板端 g_push_fail++（它没有补偿队列）。
    if parsed["state"] not in (1, 2):
        return JSONResponse(
            status_code=200,
            content={
                "ok": True,
                "event_uid": None,
                "duplicate": False,
                "ignored": True,
                "reason": "state_not_supported",
            },
        )

    duplicate = False
    orphan = False
    event_uid: Any = None
    received_at: Any = None
    publish_name: str | None = None
    publish_data: dict | None = None
    publish_zone: Any = None

    with conn.transaction():
        if parsed["state"] == 1:
            # ① CONFIRMED：唯一键 (device_id, board_seq, board_ts) + ON CONFLICT DO NOTHING
            candidate_uid = uuid7()
            severity = severity_of(parsed["prob_fall"], parsed["coverage"])
            row = fetch_one(
                conn,
                INSERT_CONFIRMED_SQL,
                (
                    candidate_uid,
                    dev["device_id"],
                    dev["ward_id"],
                    parsed["board_seq"],
                    parsed["board_ts"],
                    parsed["person_id"],
                    parsed["window_start"],
                    parsed["window_end"],
                    parsed["model_ver"],
                    parsed["prob_fall"],
                    parsed["coverage"],
                    parsed["hybrid_score"],
                    parsed["bbox"],
                    severity,
                    loc_text,
                    dev["zone_id"],
                ),
            )
            if row is None:
                # 重推：返回已有事件，不再写库（不是 409——板端不重试，重放脚本会重复发）
                duplicate = True
                existing = fetch_one(
                    conn,
                    EXACT_EVENT_SQL,
                    (dev["device_id"], parsed["board_seq"], parsed["board_ts"]),
                )
                if existing is not None:
                    event_uid = existing["event_uid"]
                    received_at = existing["received_at"]
            else:
                event_uid = row["event_uid"]
                received_at = row["received_at"]
                _touch_device(conn, dev, parsed["model_ver"])
                _ingest_audit(
                    conn, dev, parsed, event_uid, {"severity": severity, "duplicate": False}, request
                )
                try:
                    with conn.transaction():
                        clips.create_evidence(conn, event_uid, received_at)
                except Exception as exc:  # noqa: BLE001
                    print(f"[clip] 排队失败 {event_uid}: {type(exc).__name__}: {exc}", flush=True)
                publish_name = "event.created"
                publish_zone = dev["zone_id"]
                publish_data = {
                    "event_uid": str(event_uid),
                    "severity": severity,
                    "detection_state": "CONFIRMED",
                    "disposition_state": "PENDING",
                    "received_at": iso(received_at),
                    "location": loc_text,
                    "device_code": dev["device_code"],
                    "person_id": parsed["person_id"],
                    "prob_fall": parsed["prob_fall"],
                    "coverage": parsed["coverage"],
                    "has_snapshot": False,
                }
        else:
            # ② RETRACTED：不走唯一键（撤回不是新事件，是对已有事件的更正）
            exact = fetch_one(
                conn,
                EXACT_EVENT_SQL,
                (dev["device_id"], parsed["board_seq"], parsed["board_ts"]),
            )
            if exact is not None:
                # 同一条撤回 body 被重放：库里已经有这条（或孤儿行），直接判重复
                duplicate = True
                event_uid = exact["event_uid"]
                received_at = exact["received_at"]
            else:
                # ① 精确级：board_seq 是确认与撤回共用的键，正常情况下必中
                matched = fetch_one(
                    conn,
                    MATCH_RETRACT_BY_SEQ_SQL,
                    (dev["device_id"], parsed["board_seq"]),
                )
                matched_by = "board_seq"
                if matched is None:
                    matched = fetch_one(
                        conn,
                        MATCH_RETRACT_SQL,
                        (
                            dev["device_id"],
                            parsed["person_id"],
                            parsed["window_start"],
                            parsed["model_ver"],
                        ),
                    )
                    matched_by = "fuzzy"
                if matched is not None:
                    event_uid = matched["event_uid"]
                    received_at = matched["received_at"]
                    if matched["detection_state"] == "CONFIRMED":
                        # 只改 detection_state；disposition_state 绝不因模型改主意而变
                        conn.execute(
                            "UPDATE fg.event SET detection_state = 'RETRACTED' WHERE event_uid = %s",
                            (event_uid,),
                        )
                        _record_retract_action(conn, event_uid, parsed["reason"])
                        _touch_device(conn, dev, parsed["model_ver"])
                        _ingest_audit(
                            conn,
                            dev,
                            parsed,
                            event_uid,
                            {
                                "duplicate": False,
                                "orphan_retract": False,
                                "matched": True,
                                "matched_by": matched_by,
                            },
                            request,
                        )
                        publish_name = "event.updated"
                        publish_zone = matched["zone_id"]
                        publish_data = {
                            "event_uid": str(event_uid),
                            "detection_state": "RETRACTED",
                            "disposition_state": matched["disposition_state"],
                        }
                    else:
                        # 已经撤回过：幂等，不重复写
                        duplicate = True
                else:
                    # 先查一次重放：同逻辑键近期已经撤回过 ⇒ 这是同一条撤回被重放，
                    # 不是新的孤儿。不做这一步的话，每次重放都会多一张灰色幽灵卡。
                    seen = fetch_one(
                        conn,
                        MATCH_ANY_RETRACT_SQL,
                        (
                            dev["device_id"],
                            parsed["person_id"],
                            parsed["window_start"],
                            parsed["model_ver"],
                        ),
                    )
                    if seen is not None:
                        duplicate = True
                        event_uid = seen["event_uid"]
                    else:
                        # 找不到对应事件：插孤儿行，等每日对账人工回填（§9.4）
                        orphan = True
                        candidate_uid = uuid7()
                        severity = severity_of(parsed["prob_fall"], parsed["coverage"])
                        row = fetch_one(
                            conn,
                            INSERT_ORPHAN_SQL,
                            (
                                candidate_uid,
                                dev["device_id"],
                                dev["ward_id"],
                                parsed["board_seq"],
                                parsed["board_ts"],
                                parsed["person_id"],
                                parsed["window_start"],
                                parsed["window_end"],
                                parsed["model_ver"],
                                parsed["prob_fall"],
                                parsed["coverage"],
                                parsed["hybrid_score"],
                                parsed["bbox"],
                                severity,
                                loc_text,
                                dev["zone_id"],
                            ),
                        )
                        if row is None:
                            duplicate = True
                            existing = fetch_one(
                                conn,
                                EXACT_EVENT_SQL,
                                (dev["device_id"], parsed["board_seq"], parsed["board_ts"]),
                            )
                            if existing is not None:
                                event_uid = existing["event_uid"]
                                received_at = existing["received_at"]
                        else:
                            event_uid = row["event_uid"]
                            received_at = row["received_at"]
                            _touch_device(conn, dev, parsed["model_ver"])
                            _ingest_audit(
                                conn,
                                dev,
                                parsed,
                                event_uid,
                                {"severity": severity, "duplicate": False, "orphan_retract": True},
                                request,
                            )
                            publish_name = "event.created"
                            publish_zone = dev["zone_id"]
                            publish_data = {
                                "event_uid": str(event_uid),
                                "severity": severity,
                                "detection_state": "RETRACTED",
                                "disposition_state": "PENDING",
                                "received_at": iso(received_at),
                                "location": loc_text,
                                "device_code": dev["device_code"],
                                "person_id": parsed["person_id"],
                                "prob_fall": parsed["prob_fall"],
                                "coverage": parsed["coverage"],
                                "has_snapshot": False,
                            }

    # 事务提交后再广播（前端收到 SSE 立刻拉 REST 时能查到新行）
    if publish_name and publish_data is not None:
        publish_event(dev["ward_id"], publish_name, publish_data, zone_id=publish_zone)

    return JSONResponse(
        status_code=200,
        content={
            "ok": True,
            "event_uid": str(event_uid) if event_uid is not None else None,
            "duplicate": duplicate,
            "orphan_retract": orphan,
        },
    )


# ════════════════════════════════════════════════════════════════════════════
# POST /api/v1/ingest/{device_code}/{ingest_key}/frames —— 帧元数据
# ════════════════════════════════════════════════════════════════════════════
# 一帧 4 人约 1.5 KB，10 Hz。一批最多 8 帧 ⇒ 12 KB；给到 256 KB 是极宽松的上限，
# 只为挡住畸形/恶意请求，不是正常路径。
FRAMES_MAX_BYTES = 256 * 1024
FRAMES_MAX_BATCH = 32


@router.post("/{device_code}/{ingest_key}/frames")
async def ingest_frames(
    device_code: str,
    ingest_key: str,
    request: Request,
    conn: Any = Depends(get_conn),
) -> Any:
    """接收帧元数据并只转发不落库（见 app/live_meta.py 顶部说明）。"""
    dev, denied = _auth_device(conn, device_code, ingest_key)
    if denied is not None:
        return denied

    raw = await request.body()
    if not raw:
        return JSONResponse(status_code=204, content=None)
    if len(raw) > FRAMES_MAX_BYTES:
        return error_json(413, "FRAMES_TOO_LARGE", "元数据批次超过 256 KB")

    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return error_json(422, "BODY_INVALID", "请求体不是合法 JSON")
    frames = payload.get("frames") if isinstance(payload, dict) else None
    if not isinstance(frames, list):
        return error_json(422, "BODY_INVALID", '缺少 frames 数组')
    if len(frames) > FRAMES_MAX_BATCH:
        frames = frames[-FRAMES_MAX_BATCH:]     # 只留最新的，旧的没有价值

    # 板端推的流名（ 五改）：骨架按流分发，与视频跟点位走的口径对齐。
    # 老固件没有这个字段 ⇒ None ⇒ 退回原来的「病区+区域」广播，不会因为升级顺序不同而瞎。
    # 非法值静默忽略、不报 422：元数据是尽力而为的实时数据，一批帧不该因为
    # 一个多余字段被整批丢掉。名字只用来路由，真正的权限在 SSE 订阅那边校验。
    # 上报值优先；没有上报就用设备台账上登记的（fg.device.stream_path）。
    # 老固件永远不带这个字段，靠台账这一条就能把口径对齐 —— 不用等板子升级。
    stream = None
    raw_stream = payload.get("stream")
    if isinstance(raw_stream, str):
        candidate = raw_stream.strip()
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,31}", candidate):
            stream = candidate
    if stream is None:
        stream = dev.get("stream_path") or None

    published = live_meta.ingest_frames(
        dev["device_code"], dev["ward_id"], dev["zone_id"], frames, stream=stream
    )
    return JSONResponse(status_code=200, content={"ok": True, "published": published})


# ════════════════════════════════════════════════════════════════════════════
# PUT /api/v1/ingest/{device_code}/{ingest_key}/snapshot
# ════════════════════════════════════════════════════════════════════════════
def _snapshot_dir_for(received_at: dt.datetime) -> Path:
    """按事件时间分目录：<snapshot_dir>/YYYY/MM/（§8.7 规则 1）。"""
    local = received_at.astimezone(TZ)
    root = Path(getattr(settings, "snapshot_dir", "/var/lib/fallguard/snapshots"))
    return root / f"{local.year:04d}" / f"{local.month:02d}"


def _find_event_for_snapshot(conn, device_id: Any, board_seq: int):
    """按 (device_id, board_seq) 找事件；优先非孤儿行。"""
    base = """
        SELECT event_uid, received_at, detection_state, zone_id
          FROM fg.event
         WHERE device_id = %s AND board_seq = %s
    """
    row = fetch_one(
        conn,
        base + " AND orphan_retract = FALSE ORDER BY received_at DESC LIMIT 1",
        (device_id, board_seq),
    )
    if row is None:
        row = fetch_one(
            conn,
            base + " ORDER BY received_at DESC LIMIT 1",
            (device_id, board_seq),
        )
    return row


@router.put("/{device_code}/{ingest_key}/snapshot")
async def upload_snapshot(
    device_code: str,
    ingest_key: str,
    request: Request,
    board_seq: int = Query(..., description="板端 row_seq"),
    snapshot_seq: int | None = Query(None, description="板端快照帧号（V1 只记录，不入库）"),
    sha256: str | None = Query(None, description="快照内容的 sha256（小写十六进制）"),
    width: int = Query(320, ge=1, le=8192, description="快照实际宽度（板端如实上报）"),
    height: int = Query(240, ge=1, le=8192, description="快照实际高度（板端如实上报）"),
    conn: Any = Depends(get_conn),
) -> Any:
    """板端手动脚本上传快照（§9.3）。找不到事件不落盘，sha256 / 时间差都要校验。"""
    dev, denied = _auth_device(conn, device_code, ingest_key)
    if denied is not None:
        return denied

    # ① 按 (device_id, board_seq) 找事件 —— 找不到 404 且不落盘
    ev = _find_event_for_snapshot(conn, dev["device_id"], board_seq)
    if ev is None:
        return error_json(404, "EVENT_NOT_FOUND", "找不到对应事件，快照未落盘")

    # ①b 首帧优先：一个事件只留第一张快照。
    # 撤回时板端会换一张「站姿图」（网关 db.c 的 snap_file 被 COALESCE 更新），
    # 旧实现用 `ON CONFLICT (event_uid) DO UPDATE` + 固定落盘名 `<event_uid>.jpg`
    # 把它连行带文件一起覆盖 —— 告警时刻那张图从此不可恢复（ 实测
    # 1288/1289 的卡片显示的是撤回后的画面）。证据链要的是响铃那一刻，
    # 所以改成只认第一张；后续上传一律 200 + ignored（板端上传器按 200 标记
    # 完成，不会无限重试）。
    existing = fetch_one(
        conn,
        """
        SELECT snapshot_id, storage_path, width, height, bytes, sha256, uploaded_at
          FROM fg.snapshot
         WHERE event_uid = %s
        """,
        (ev["event_uid"],),
    )
    repair = False
    if existing is not None:
        kept = Path(existing.get("storage_path") or "")
        if kept.is_file():
            return JSONResponse(
                status_code=200,
                content={
                    "ok": True,
                    "ignored": True,
                    "reason": "snapshot_exists",
                    "event_uid": str(ev["event_uid"]),
                    "snapshot_id": str(existing["snapshot_id"]),
                    "board_seq": board_seq,
                    "snapshot_seq": snapshot_seq,
                    "kept_sha256": existing["sha256"],
                },
            )
        repair = True

    # ② 时间差校验：快照必须是这次事件的，不能是双缓冲里被覆盖后的错图
    received_at = ev["received_at"]
    lag_s = abs((now_sh() - received_at.astimezone(TZ)).total_seconds())
    if lag_s > SNAPSHOT_MAX_LAG_S:
        return error_json(
            400,
            "SNAPSHOT_EXPIRED",
            f"快照与事件时间差 {int(lag_s)} 秒，超过 {SNAPSHOT_MAX_LAG_S} 秒，拒绝入库",
        )

    # ③ 读二进制 + sha256 校验
    data = await request.body()
    if not data:
        return error_json(422, "SNAPSHOT_EMPTY", "快照内容为空")
    if len(data) > SNAPSHOT_MAX_BYTES:
        return error_json(413, "SNAPSHOT_TOO_LARGE", "快照超过 2 MB")
    digest = hashlib.sha256(data).hexdigest()
    claimed = (sha256 or request.headers.get("x-content-sha256") or "").strip().lower()
    if not claimed:
        return error_json(400, "SHA256_REQUIRED", "缺少 sha256 校验值")
    if not hmac.compare_digest(digest, claimed):
        return error_json(400, "SHA256_MISMATCH", "sha256 校验失败，快照未落盘")

    # ④ 落盘：<snapshot_dir>/YYYY/MM/<event_uid>.<ext>
    # 顺序：写临时文件 → rename 到正式名 → 再写库。
    # 「先 rename 再写库」是刻意的：写库失败只会留下一个没行指向的文件（下次上传
    # 会覆盖它，无害）；反过来（先写库再 rename）一旦 rename 失败，就会留下一条
    # 指向空气的行，而 fg_app 没有 DELETE 权限，删都删不掉。
    content_type = (request.headers.get("content-type") or "").split(";")[0].strip().lower()
    ext = {"image/png": ".png", "image/jpeg": ".jpg", "image/jpg": ".jpg"}.get(content_type, ".jpg")
    subdir = _snapshot_dir_for(received_at)
    final_path = subdir / f"{ev['event_uid']}{ext}"
    tmp_path = subdir / f".{ev['event_uid']}.{os.getpid()}.tmp"
    try:
        subdir.mkdir(parents=True, exist_ok=True)
        with open(tmp_path, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, final_path)
    except OSError as exc:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        return error_json(500, "SNAPSHOT_WRITE_FAILED", f"快照写盘失败：{exc}")

    # ⑤ 写 snapshot 行
    with conn.transaction():
        if repair:
            # 行已存在、文件缺失：用 UPDATE 把行指回刚补上的文件
            try:
                conn.execute(
                    """
                    UPDATE fg.snapshot
                       SET storage_path = %s, width = %s, height = %s, bytes = %s,
                           sha256 = %s, source = 'board_upload', uploaded_at = now()
                     WHERE event_uid = %s
                    """,
                    (str(final_path), width, height, len(data), digest, ev["event_uid"]),
                )
            except Exception:
                # 写库失败就把刚补上的文件撤掉，让状态回到「行在、文件缺」——
                # 那是可重试的干净状态。留着文件的话，行里的 sha256/bytes 与盘上
                # 内容对不上，而且下次上传会被「文件还在」判成 ignored，永久不一致。
                try:
                    os.unlink(final_path)
                except OSError:
                    pass
                raise
            snapshot_id = existing["snapshot_id"]
        else:
            # DO NOTHING：并发重传只有第一个赢
            row = fetch_one(
                conn,
                """
                INSERT INTO fg.snapshot
                    (event_uid, storage_path, width, height, bytes, sha256, source)
                VALUES (%s, %s, %s, %s, %s, %s, 'board_upload')
                ON CONFLICT (event_uid) DO NOTHING
                RETURNING snapshot_id
                """,
                (ev["event_uid"], str(final_path), width, height, len(data), digest),
            )
            if row is None:
                # 并发里输了：盘上文件是我们刚 rename 过去的，和赢家的内容可能不同。
                # 但两边都是同一次告警的图，且下一轮板端不会再传 —— 接受这点偏差，
                # 比「删行」更安全（没有 DELETE 权限，也不该删证据）。
                return JSONResponse(
                    status_code=200,
                    content={
                        "ok": True,
                        "ignored": True,
                        "reason": "snapshot_exists",
                        "event_uid": str(ev["event_uid"]),
                        "board_seq": board_seq,
                        "snapshot_seq": snapshot_seq,
                    },
                )
            snapshot_id = row["snapshot_id"]
    # 快照到位后广播一次，列表里的「无快照」占位能立刻换成图。
    # 区域用事件自己的（ev["zone_id"]），不用设备的 —— 设备可能已经改绑。
    publish_event(
        dev["ward_id"],
        "event.updated",
        {"event_uid": str(ev["event_uid"]), "has_snapshot": True},
        zone_id=ev["zone_id"],
    )

    return JSONResponse(
        status_code=200,
        content={
            "ok": True,
            "event_uid": str(ev["event_uid"]),
            "snapshot_id": str(snapshot_id) if snapshot_id is not None else None,
            "board_seq": board_seq,
            "snapshot_seq": snapshot_seq,
            "bytes": len(data),
            "sha256": digest,
            "storage_path": str(final_path),
        },
    )


CONFIG_POLL_MAX_S = 25


def _poll_config(device_code: str, ingest_key: str, applied_rev: int):
    """返回 ("denied", resp) / ("ok", payload) / ("none", None)。"""
    with get_pool().connection() as conn:
        dev, denied = _auth_device(conn, device_code, ingest_key)
        if denied is not None:
            return "denied", denied
        row = conn.execute(
            "SELECT rev, params, updated_at FROM fg.device_config WHERE device_id = %s",
            (dev["device_id"],),
        ).fetchone()
        # 轮询本身就是心跳：25 秒一次足够判在线
        conn.execute(
            "UPDATE fg.device SET last_seen_at = now() WHERE device_id = %s",
            (dev["device_id"],),
        )
    if row is not None and int(row["rev"]) > int(applied_rev):
        return "ok", {
            "rev": int(row["rev"]),
            "params": row["params"],
            "updated_at": iso(row["updated_at"]),
        }
    return "none", None


@router.get("/{device_code}/{ingest_key}/config")
async def get_config(
    device_code: str,
    ingest_key: str,
    applied_rev: int = Query(0, ge=0),
    wait: int = Query(CONFIG_POLL_MAX_S, ge=0, le=CONFIG_POLL_MAX_S),
) -> Any:
    """长轮询：有比 `applied_rev` 新的配置立刻 200；等满 `wait` 秒返回 204。"""
    deadline = time.monotonic() + wait
    while True:
        kind, payload = await asyncio.to_thread(
            _poll_config, device_code, ingest_key, applied_rev)
        if kind == "denied":
            return payload
        if kind == "ok":
            return JSONResponse(status_code=200, content=payload)
        if time.monotonic() >= deadline:
            return Response(status_code=204)
        await asyncio.sleep(1.0)


@router.post("/{device_code}/{ingest_key}/config/ack")
async def ack_config(device_code: str, ingest_key: str, request: Request) -> Any:
    """板端确认配置是否生效。只有 APPLIED 才推进 last_policy_rev。"""
    raw = await request.body()
    try:
        payload = json.loads(raw.decode("utf-8")) if raw else {}
    except (UnicodeDecodeError, json.JSONDecodeError):
        return error_json(422, "BODY_INVALID", "请求体不是合法 JSON")
    try:
        rev = int(payload["rev"])
        status = str(payload.get("status") or "").upper()
    except (KeyError, TypeError, ValueError):
        return error_json(422, "BODY_INVALID", "缺少 rev / status")
    if status not in ("APPLIED", "FAILED"):
        return error_json(422, "BODY_INVALID", "status 只能是 APPLIED / FAILED")

    with get_pool().connection() as conn:
        dev, denied = _auth_device(conn, device_code, ingest_key)
        if denied is not None:
            return denied
        reported = payload.get("reported") or {}
        if status == "APPLIED":
            conn.execute(
                "UPDATE fg.device SET last_policy_rev = %s, last_config_hash = %s, "
                "last_seen_at = now(), last_status = %s WHERE device_id = %s",
                (rev, payload.get("config_hash") or None, Jsonb(reported),
                 dev["device_id"]),
            )
        else:
            conn.execute(
                "UPDATE fg.device SET last_seen_at = now(), last_status = %s "
                "WHERE device_id = %s",
                (Jsonb({"ack": "FAILED", "rev": rev,
                        "error": payload.get("error"), "reported": reported}),
                 dev["device_id"]),
            )
        audit(
            conn, _SystemActor(dev["ward_id"]),
            "CONFIG_APPLY" if status == "APPLIED" else "CONFIG_APPLY_FAIL",
            "device", str(dev["device_id"]), "OK",
            detail={"rev": rev, "status": status, "error": payload.get("error")},
            client_ip=client_ip(request),
        )
    return {"ok": True, "rev": rev, "status": status}


@router.post("/{device_code}/{ingest_key}/heartbeat")
async def heartbeat(device_code: str, ingest_key: str, request: Request) -> Any:
    """板端心跳：在线状态 + 实际生效版本 + 运行指标。不写审计（每 60 秒一条太吵）。"""
    raw = await request.body()
    try:
        payload = json.loads(raw.decode("utf-8")) if raw else {}
    except (UnicodeDecodeError, json.JSONDecodeError):
        return error_json(422, "BODY_INVALID", "请求体不是合法 JSON")
    if not isinstance(payload, dict):
        return error_json(422, "BODY_INVALID", "心跳必须是 JSON 对象")

    with get_pool().connection() as conn:
        dev, denied = _auth_device(conn, device_code, ingest_key)
        if denied is not None:
            return denied
        conn.execute(
            "UPDATE fg.device SET last_seen_at = now(), "
            "last_seen_ip = NULLIF(%s, '')::inet, last_status = %s, "
            "last_policy_rev = COALESCE(%s, last_policy_rev), "
            "model_ver = COALESCE(%s, model_ver) WHERE device_id = %s",
            (client_ip(request) or "", Jsonb(payload),
             payload.get("policy_rev"), payload.get("model_ver"),
             dev["device_id"]),
        )
    return {"ok": True}
