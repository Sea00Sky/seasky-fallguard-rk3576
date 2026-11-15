"""实时画面：把 mediamtx 的两条出口都藏在会话鉴权之后。"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import threading
import time
from typing import Any

import httpx
import psycopg
from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from starlette.concurrency import run_in_threadpool

from app import live_meta
from app.config import settings
from app.db import query_one
from app.deps import (
    CurrentUser,
    current_user,
    require_nurse,
    scoped_ward,
    scoped_zones,
    zone_clauses,
    zone_keys,
)

log = logging.getLogger(__name__)

# 点位绑定的流路径名。比 mediamtx 的规则更严：
# 单段、字母数字开头、最长 32、只允许字母数字下划线连字符。
# 这条正则会同时用在「URL 里的 stream 段」和「上游拼 URL 之前」，
# 保证用户可控的字符串永远不会改写上游 URL 的 path/query/fragment。
_STREAM_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,31}$")

_STREAM_CHECK_TTL_S = 10.0
# 值 = (写入时刻 monotonic, 结果, 写入时的代际号)
#
# 代际号是干什么的：光有「改绑后清缓存」挡不住这一种交错 ——
# 护士的查询在管理员提交之前开始（READ COMMITTED 读到旧状态），
# 而写回缓存落在 invalidate_stream_cache() 之后。清空被这条回填抵消，
# 旧放行照样活满 10 秒，而那 10 秒里画面已经换人了。
# 所以每条缓存都记下自己是哪一代写的；代际号一变，旧条目读的时候自然作废。
# 这样不需要锁、也不需要重试：陈旧条目只是躺在字典里没人认，下次查询直接覆盖。
_stream_allowed_cache: dict[
    tuple[str, str, tuple[str, ...] | None], tuple[float, bool, int]
] = {}
_stream_cache_gen = 0
_stream_cache_lock = threading.Lock()

# 发布者探针缓存。页面打开时会先问一次状态，多开几个标签就多打几次 mediamtx API。
_PROBE_TTL_S = 5.0
_probe_cache: dict[str, tuple[float, dict]] = {}

router = APIRouter(prefix="/api/v1/live", tags=["live"])

# 上游只在本机，且观众数上限很低（1–2），一个共享客户端足够。
_client: httpx.AsyncClient | None = None

# 只把必要的响应头透给浏览器；Set-Cookie 一定要丢掉，否则 mediamtx 的
# 会话 cookie 会覆盖到站点域上。
_PASS_HEADERS = ("content-type", "cache-control", "content-length")


def _http() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(
            timeout=httpx.Timeout(20.0, connect=5.0),
            # mediamtx 的 HLS 会 302 到 ?cookieCheck=1 做一次探测，
            # 让 httpx 自己跟掉，浏览器就不用看见这个跳转。
            follow_redirects=True,
            limits=httpx.Limits(max_connections=8, max_keepalive_connections=4),
        )
    return _client


async def aclose() -> None:
    """应用退出时释放连接池（由 main.py 的 lifespan 调用）。"""
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None


def _err(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(
        {"error": {"code": code, "message": message}}, status_code=status
    )


@router.get("/meta")
async def live_meta_stream(
    request: Request,
    user: CurrentUser = Depends(require_nurse),
) -> Any:
    """把板端 10 Hz 的帧元数据推给浏览器，用来在 `<video>` 上叠骨架。"""
    ward = scoped_ward(user)
    zones = zone_keys(scoped_zones(user))

    # 五改：页面把「自己正在看的那条流」带上来，骨架就按流分发。
    # 视频跟点位的 stream_path 走、骨架原先却跟设备的病区/区域走，两者一旦
    # 不一致（把流绑到隔壁病区的点位、或设备换了病区）就会出现「有画面、没骨架」。
    # 权限判据与 HLS/WHEP 完全共用 `_stream_allowed`（病区 + 区域、默认拒绝、
    # 10 秒缓存 + 改绑失效）：能看这条流的画面，才能收这条流的骨架。
    # 不带 ?stream= 的老页面/老缓存照旧走病区+区域，两端可以各升各的。
    stream = (request.query_params.get("stream") or "").strip() or None
    if stream is not None:
        if not _STREAM_RE.match(stream):
            return _err(403, "STREAM_INVALID", "流名不合法")
        if not await _stream_allowed(user, stream):
            return _err(403, "STREAM_FORBIDDEN", "这个账号看不到这条流")

    async def recheck(queue: asyncio.Queue) -> bool:
        """借心跳复核一次权限：账号停用 / 区域被撤，都要当场断线。"""
        try:
            fresh = await run_in_threadpool(current_user, request)
            ward2 = scoped_ward(fresh)
        except Exception:  # noqa: BLE001 —— 401/403/库抖动都按「该重连」处理
            return False
        # 流订阅还要复核一次「这条流是不是还归你能看」：管理员把点位解绑/改绑后，
        # 这条长连接不能继续收老流的骨架（与 HLS 侧 invalidate_stream_cache 同源）。
        if stream is not None and not await _stream_allowed(fresh, stream):
            return False
        live_meta.refresh_scope(queue, ward2, zone_keys(scoped_zones(fresh)))
        return True

    async def gen():
        # 订阅登记在生成器体内，不在外面。订阅是「当前这条连接」的资源，
        # 登记与退订必须在同一段生命周期里：放在外面的话，只要响应对象没被
        # 真正 dispatch（首帧前就被取消 / 进程内直接调端点不迭代），
        # `finally` 就跑不到，`_queues` 里留一条僵尸订阅 —— 而 `ingest_frames`
        # 是 10 Hz 常流，每帧都会遍历 `_queues` 并往每条队列 put。宁可少登记。
        # 订阅时就钉死区域范围：帧的 payload 里没有设备身份，浏览器无法自救，
        # 所以「别区域的骨架绝不出现」只能在服务端这一层拦。
        queue = live_meta.subscribe(ward, zones, stream)
        last_check = time.monotonic()
        try:
            # 首帧就把云端时钟发下去，浏览器立刻能算 bias
            yield _sse("hello", {"server_ts": live_meta.server_now_ms()})
            while True:
                if await request.is_disconnected():
                    break
                try:
                    item = await asyncio.wait_for(queue.get(), timeout=_PING_S)
                except asyncio.TimeoutError:
                    item = None
                    yield ": ping\n\n"          # 注释帧：保活，浏览器会忽略
                if time.monotonic() - last_check >= _PING_S:
                    last_check = time.monotonic()
                    if not await recheck(queue):
                        break
                if item is not None:
                    yield _sse(item["event"], item["data"])
        finally:
            live_meta.unsubscribe(queue)

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-store",
            "X-Accel-Buffering": "no",      # nginx 不要缓冲 SSE
            "Connection": "keep-alive",
        },
    )


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False, separators=(',', ':'))}\n\n"


_PING_S = 15.0


# ════════════════════════════════════════════════════════════════════════════
# 按点位取流（ ）
#
# 背景：点位（fg.point）可选绑定一台设备（point.device_id），
# 画面地址 = 那台设备的 stream_path（ 六改）。摄像头自己推到云端，
# 管理员在设备上填推流、在点位上选设备——绑到哪个点位，哪个点位就出画面。
#
# 为什么 stream 必须进路径段而不是查询参数：
# HLS 播放列表里的分片是相对 URI，浏览器解析时会丢掉 query，
# 表现是「index.m3u8 拿到了，但所有 .ts 都回到默认流」——极难排查。
#
# 注册顺序：本段的 GET 路由必须全部位于下方 `/{path:path}` catch-all 之前。
# 写在后面会被当成"HLS 文件名"，被 BAD_PATH 直接判 400。
# ════════════════════════════════════════════════════════════════════════════

_STREAM_LOOKUP_BASE = (
    "SELECT 1 AS ok FROM fg.point p "
    "  JOIN fg.device d ON d.device_id = p.device_id "
    " WHERE d.stream_path = %s AND p.ward_id = %s AND p.is_active"
)


async def _stream_allowed(user: CurrentUser, stream: str) -> bool:
    """这条流是不是属于当前账号的病区和区域。"""
    ward = scoped_ward(user)
    zones = scoped_zones(user)
    key = (stream, ward, zone_keys(zones))
    now = time.monotonic()
    hit = _stream_allowed_cache.get(key)
    # 代际号必须当场读：写回时要比的是「我这个结果属于哪一代」，
    # 而不是「现在最新是哪一代」。
    gen0 = _stream_cache_gen
    # 代际号对不上 = 我这条缓存写于某次 invalidate 之前 ⇒ 直接作废。
    # 见 _stream_allowed_cache 上面那段：这是「清空被回填抵消」的唯一解。
    if hit is not None and hit[2] == gen0 and now - hit[0] < _STREAM_CHECK_TTL_S:
        return hit[1]

    # 区域条件仍然只由 deps.zone_clauses() 产生（全系统唯一一处）。
    # 六改后这条 SQL 里有 p / d 两个别名，必须传 "p"：
    # 裸 `zone_id` 会变成 "column reference is ambiguous" ⇒ 护士看画面全 403。
    # 取点位（p）的区域是对的：授权按「点位归属哪个区域」，不是按设备装在哪。
    clauses, extra = zone_clauses("p", zones)
    sql = _STREAM_LOOKUP_BASE + "".join(" AND " + c for c in clauses)

    try:
        row = await run_in_threadpool(query_one, sql, (stream, ward, *extra))
    except psycopg.errors.UndefinedColumn as exc:
        # zone_id 列不存在（补丁没跑）＝ schema 漂移。降级重查：去掉区域条件、
        # 保留病区条件。绝不整条放行 —— 这条 SQL 报错时病区条件也没被求值。
        log.warning(
            "live.stream_check_schema_drift stream=%s ward=%s zones=%s err=%s",
            stream, ward, "all" if zones is None else len(zones), exc,
        )
        try:
            row = await run_in_threadpool(query_one, _STREAM_LOOKUP_BASE, (stream, ward))
        except Exception as exc2:  # noqa: BLE001
            # 连 stream_path / is_active 都没有 ⇒ 判据彻底失效，拒绝。
            log.error(
                "live.stream_check_degraded_failed stream=%s ward=%s err=%s",
                stream, ward, exc2,
            )
            return False
    except psycopg.errors.UndefinedTable as exc:
        log.error(
            "live.stream_check_no_table stream=%s ward=%s err=%s", stream, ward, exc
        )
        return False
    except Exception as exc:  # noqa: BLE001
        log.error(
            "live.stream_check_failed stream=%s ward=%s zones=%s err=%s",
            stream, ward, "all" if zones is None else len(zones), exc,
        )
        return False

    allowed = row is not None

    # 只缓存放行。拒绝必须每次重新问库：管理员刚把流改绑到某点位，
    # 演示时页面是立刻刷的，一个 10 秒的拒绝缓存会让"刚配好却打不开"重现。
    # 拒绝本身很便宜（一次索引查询，之后前端 404 直接停手，不会再有分片请求）。
    if allowed:
        with _stream_cache_lock:
            if len(_stream_allowed_cache) > 1024:
                _stream_allowed_cache.clear()
            _stream_allowed_cache[key] = (now, True, gen0)
    return allowed


def invalidate_stream_cache() -> None:
    """把 10 秒的放行缓存作废。点位/设备的流归属一变就要调。"""
    global _stream_cache_gen
    with _stream_cache_lock:
        _stream_cache_gen += 1
        _stream_allowed_cache.clear()


async def _probe_stream(stream: str) -> dict:
    """这条路径上现在有没有发布者。"""
    now = time.monotonic()
    hit = _probe_cache.get(stream)
    if hit is not None and now - hit[0] < _PROBE_TTL_S:
        return hit[1]

    url = f"{settings.mediamtx_api_url.rstrip('/')}/v3/paths/get/{stream}"
    try:
        # 不带 Basic Auth：API 只监听回环，authInternalUsers 给本机 any 开了 api 权限。
        # 带上 viewer 凭据反而可能被拒（viewer 只有 read/api 之外的权限）。
        r = await _http().get(url, timeout=3.0)
    except httpx.HTTPError as exc:
        result = {"state": "unknown", "reason": type(exc).__name__}
    else:
        if r.status_code == 404:
            # 路径没配发布者（mediamtx 的 all_others 会收下这个名字，但里面是空的）
            result = {"state": "not_connected"}
        elif r.status_code != 200:
            result = {"state": "unknown", "reason": f"api {r.status_code}"}
        else:
            try:
                ready = bool(r.json().get("ready"))
            except ValueError:
                ready = False
            result = {"state": "online" if ready else "not_connected", "ready": ready}

    # 上限兜底：key 来自 URL（虽然已过 _STREAM_RE 和归属校验），
    # 不设上限等于给"能登录的人"一个缓慢涨内存的入口。
    if len(_probe_cache) > 1024:
        _probe_cache.clear()
    _probe_cache[stream] = (now, result)
    return result


def _hls_bad_path(path: str) -> JSONResponse | None:
    """HLS 文件名白名单 + 目录穿越防护。返回 None 表示通过。"""
    if "/" in path or ".." in path:
        return _err(400, "BAD_PATH", "非法的画面路径")
    if not (path.endswith(".m3u8") or path.endswith(".ts") or path.endswith(".mp4")):
        return _err(400, "BAD_PATH", "只允许 m3u8 / ts / mp4")
    return None


async def _proxy_hls(stream: str, path: str, request: Request) -> Response:
    """把 ``/<stream>/<file>`` 转发给 mediamtx 的 HLS 出口。"""
    if not settings.mediamtx_pass:
        return _err(503, "LIVE_DISABLED", "实时画面未配置（MEDIAMTX_PASS 为空）")

    bad = _hls_bad_path(path)
    if bad is not None:
        return bad

    url = f"{settings.mediamtx_url.rstrip('/')}/{stream}/{path}"
    try:
        upstream = await _http().get(
            url,
            params=dict(request.query_params),
            auth=(settings.mediamtx_user, settings.mediamtx_pass),
        )
    except httpx.HTTPError as exc:
        return _err(502, "LIVE_UPSTREAM", f"取流失败：{type(exc).__name__}")

    headers = {
        k: v for k, v in upstream.headers.items() if k.lower() in _PASS_HEADERS
    }
    headers["Cache-Control"] = "no-store"
    return Response(
        content=upstream.content,
        status_code=upstream.status_code,
        headers=headers,
        media_type=upstream.headers.get("content-type"),
    )


@router.get("/hls/{stream}/{path:path}")
async def hls_proxy_stream(
    stream: str,
    path: str,
    request: Request,
    user: CurrentUser = Depends(require_nurse),
) -> Response:
    """按点位取 HLS。``stream`` / ``path`` 两段，必须注册在 catch-all 之前。"""
    if not _STREAM_RE.match(stream):
        return _err(400, "BAD_STREAM", "非法的画面路径名")
    if not await _stream_allowed(user, stream):
        # 404 而不是 403：不暴露"这条流存在但不属于你"
        return _err(404, "LIVE_NOT_FOUND", "该点位没有可用的画面")
    return await _proxy_hls(stream, path, request)


@router.get("/status/{stream}")
async def stream_status(
    stream: str,
    user: CurrentUser = Depends(require_nurse),
) -> Response:
    """这条流现在能不能看。前端据此决定「自动播放」还是「显示未连接」。"""
    if not _STREAM_RE.match(stream):
        return _err(400, "BAD_STREAM", "非法的画面路径名")
    if not await _stream_allowed(user, stream):
        return _err(404, "LIVE_NOT_FOUND", "该点位没有可用的画面")
    if not settings.mediamtx_pass:
        return _err(503, "LIVE_DISABLED", "实时画面未配置（MEDIAMTX_PASS 为空）")
    payload = await _probe_stream(stream)
    payload["stream"] = stream
    return JSONResponse(payload)


@router.get("/{path:path}")
async def hls_proxy(
    path: str,
    request: Request,
    user: CurrentUser = Depends(require_nurse),
) -> Response:
    """【兼容】旧版单路地址，等价于 ``stream='live'``。"""
    if not await _stream_allowed(user, "live"):
        return _err(404, "LIVE_NOT_FOUND", "该点位没有可用的画面")
    return await _proxy_hls("live", path, request)


# ══════════════════════════════════════════════════════════════════════════
# WebRTC / WHEP（低延迟播放）
#
# 这一层只能代理信令，代理不了媒体。
# WHEP 是纯 HTTP（POST 送 SDP offer、拿 answer、DELETE 收尾），可以完整转发；
# 但真正承载视频的是浏览器 ↔ mediamtx 之间的 UDP/ICE（8189）直连——
# 那是 UDP，不经过 FastAPI。所以：
# 1. 腾讯云安全组必须放行 UDP 8189，否则 WHEP 返回 201 却永远黑屏
# （这是最常见的一种"看起来成功了其实没通"）；
# 2. 媒体面绕过了会话校验，靠的是 WHEP 会话 ID 的随机性。
# 前端因此做成"WebRTC 优先、失败回落 HLS"（见 _live_panel_js.html）。
# ══════════════════════════════════════════════════════════════════════════

_SESSION_RE = re.compile(r"^[A-Za-z0-9_-]{8,64}$")


def _webrtc_base() -> str:
    return settings.mediamtx_webrtc_url.rstrip("/")


async def _whep_create(stream: str, request: Request) -> Response:
    """建 WHEP 会话的共用实现。``stream`` 已由调用方校验过。"""
    if not settings.mediamtx_pass:
        return _err(503, "LIVE_DISABLED", "实时画面未配置（MEDIAMTX_PASS 为空）")

    body = await request.body()
    if not body or b"v=0" not in body[:64]:
        return _err(400, "BAD_SDP", "请求体不是 SDP offer")

    try:
        up = await _http().post(
            f"{_webrtc_base()}/{stream}/whep",
            content=body,
            headers={"Content-Type": "application/sdp"},
            auth=(settings.mediamtx_user, settings.mediamtx_pass),
        )
    except httpx.HTTPError as exc:
        return _err(502, "LIVE_UPSTREAM", f"WHEP 失败：{type(exc).__name__}")

    if up.status_code != 201:
        # 404 = 这条 path 上没有发布者。必须和"上游挂了"分开：
        # 原来一律改写成 502，前端只能看到「WHEP 502」，永远说不出
        # 「这个点位没接摄像头」。分开之后前端才能显示「摄像头未连接」而不是黑屏。
        if up.status_code == 404:
            return _err(404, "LIVE_NOT_CONNECTED", "该点位的摄像头没有推流")
        return _err(502, "WHEP_REJECTED", f"上游返回 {up.status_code}")

    # Location: /<path>/whep/<uuid> —— 只取最后一段，换成我们自己的地址，
    # 浏览器后面就不会知道（也不需要知道）mediamtx 的存在。
    # 先按 "?" 切掉 query：带着 ?xxx 拼进 session 会让后续 PATCH/DELETE 502。
    session = (up.headers.get("location") or "").split("?")[0].rstrip("/").rsplit("/", 1)[-1]
    if not _SESSION_RE.match(session):
        return _err(502, "WHEP_BAD_SESSION", "上游没有返回合法会话标识")

    return Response(
        content=up.content,
        status_code=201,
        media_type="application/sdp",
        headers={
            "Location": f"/api/v1/live/whep/{stream}/{session}",
            "Cache-Control": "no-store",
        },
    )


@router.post("/whep/{stream}")
async def whep_create_stream(
    stream: str,
    request: Request,
    user: CurrentUser = Depends(require_nurse),
) -> Response:
    """按点位创建 WHEP 会话。"""
    if not _STREAM_RE.match(stream):
        return _err(400, "BAD_STREAM", "非法的画面路径名")
    if not await _stream_allowed(user, stream):
        return _err(404, "LIVE_NOT_FOUND", "该点位没有可用的画面")
    return await _whep_create(stream, request)


@router.post("/whep")
async def whep_create(
    request: Request,
    user: CurrentUser = Depends(require_nurse),
) -> Response:
    """【兼容】旧版单路，等价于 ``stream='live'``。"""
    if not await _stream_allowed(user, "live"):
        return _err(404, "LIVE_NOT_FOUND", "该点位没有可用的画面")
    return await _whep_create("live", request)


async def _whep_session(
    method: str, stream: str, session: str, request: Request | None
) -> Response:
    if not _SESSION_RE.match(session):
        return _err(400, "BAD_SESSION", "非法的会话标识")
    # 没配凭据时别拿空密码去撞上游（会得到一个和"未配置"无关的 401/502），
    # 直接说清楚：这一步是收尾，收不掉也要让调用方知道原因。
    if not settings.mediamtx_pass:
        return _err(503, "LIVE_DISABLED", "实时画面未配置（MEDIAMTX_PASS 为空）")

    kwargs: dict = {"auth": (settings.mediamtx_user, settings.mediamtx_pass)}
    if method == "PATCH" and request is not None:
        kwargs["content"] = await request.body()
        kwargs["headers"] = {
            "Content-Type": request.headers.get(
                "content-type", "application/trickle-ice-sdpfrag"
            )
        }
    try:
        up = await _http().request(
            method, f"{_webrtc_base()}/{stream}/whep/{session}", **kwargs
        )
    except httpx.HTTPError as exc:
        return _err(502, "LIVE_UPSTREAM", f"WHEP {method} 失败：{type(exc).__name__}")
    return Response(status_code=up.status_code, content=up.content)


@router.patch("/whep/{stream}/{session}")
async def whep_trickle_stream(
    stream: str,
    session: str,
    request: Request,
    user: CurrentUser = Depends(require_nurse),
) -> Response:
    """按点位转发 trickle-ICE 候选（前端目前用 non-trickle，留着备用）。"""
    if not _STREAM_RE.match(stream):
        return _err(400, "BAD_STREAM", "非法的画面路径名")
    return await _whep_session("PATCH", stream, session, request)


@router.delete("/whep/{stream}/{session}")
async def whep_close_stream(
    stream: str,
    session: str,
    user: CurrentUser = Depends(require_nurse),
) -> Response:
    """按点位结束会话。必须调：不调上游会一直按"有观众"往外送流。"""
    if not _STREAM_RE.match(stream):
        return _err(400, "BAD_STREAM", "非法的画面路径名")
    return await _whep_session("DELETE", stream, session, None)


@router.patch("/whep/{session}")
async def whep_trickle(
    session: str,
    request: Request,
    user: CurrentUser = Depends(require_nurse),
) -> Response:
    """【兼容】旧版单路 trickle。"""
    return await _whep_session("PATCH", "live", session, request)


@router.delete("/whep/{session}")
async def whep_close(
    session: str,
    user: CurrentUser = Depends(require_nurse),
) -> Response:
    """【兼容】旧版单路会话结束。"""
    return await _whep_session("DELETE", "live", session, None)


@router.get("")
async def live_status(user: CurrentUser = Depends(require_nurse)) -> dict:
    """画面可用性探针：前端用它决定播放按钮是否可点。"""
    ward = scoped_ward(user)
    zones = zone_keys(scoped_zones(user))
    # 骨架元数据的诊断计数（frames_in 不涨 = 板端没在发 / 鉴权没过）。
    # offsets 是「设备 → 时钟偏移」，必须按病区+区域过滤 —— 它是设备清单本身。
    meta_stats = live_meta.stats(ward, zones)
    if not settings.mediamtx_pass:
        return {"enabled": False, "reason": "not_configured", "online": False,
                "meta": meta_stats}
    if not await _stream_allowed(user, "live"):
        return {"enabled": False, "reason": "not_configured", "online": False,
                "meta": meta_stats}

    try:
        r = await _http().get(
            f"{settings.mediamtx_url.rstrip('/')}/live/index.m3u8",
            auth=(settings.mediamtx_user, settings.mediamtx_pass),
        )
    except httpx.HTTPError as exc:
        return {"enabled": True, "online": False, "reason": type(exc).__name__,
                "meta": meta_stats}

    body = r.text if r.status_code == 200 else ""
    return {
        "enabled": True,
        "online": "EXTM3U" in body,
        "status": r.status_code,
        "playlist": "/api/v1/live/index.m3u8",
        "meta": meta_stats,
    }
