"""帧元数据广播—— 网页 canvas 叠骨架的数据源。"""
from __future__ import annotations

import asyncio
import logging
import math
import time
from typing import Any

log = logging.getLogger("fallguard.live_meta")

# 订阅者队列：queue -> (ward_id, zones, stream)（单 worker 进程内广播，与 events.py 的 SSE 同一套前提）
# zones 为 None ⇒ 不受区域限制（护士长）；为元组 ⇒ 只收 zone_id 属于它的帧。
# 元组里是 str（见 deps.zone_key），空元组 = 什么都收不到（默认拒绝）。
# stream 为 None ⇒ 老口径「按病区+区域」；非 None ⇒ 按流名（ 五改）：
# 视频跟点位的 stream_path 走，骨架也得跟着那条流走，否则「流绑到别的病区的
# 点位」就会出现「有画面、没骨架」。订阅时的权限已由 _stream_allowed 校验过。
_queues: dict[asyncio.Queue, tuple[str, tuple[str, ...] | None, str | None]] = {}
# 设备 → (病区, 区域)：给 stats() 过滤设备清单用（只进不出，供诊断）
_device_scope: dict[str, tuple[str, str | None]] = {}
# 板端时钟偏移 EMA：device_code -> (cloud − board) 秒
_offsets: dict[str, float] = {}
# device_code -> 连续被「离谱」判定拒绝的批次数（见 _OFFSET_JUMP_MAX_S 的说明）
_offset_rejects: dict[str, int] = {}
# 诊断计数
_frames_in = 0
_frames_out = 0
_dropped = 0

# 设备台账上限：_device_scope 按 device_code 累积，正常只有几台；
# 兜底是为了挡住"有人拿同一个 ingest_key 换着 device_code 刷"的涨内存路径。
_DEVICE_SCOPE_MAX = 1024

# 每个订阅者的积压上限。10 Hz 下 30 帧 = 3 秒，超过就该丢了。
QUEUE_MAX = 30
# EMA 平滑系数：0.2 ≈ 5 帧的时间常数（0.5 s），跟得上漂移又不被单帧抖动带偏
_EMA_ALPHA = 0.2
# 单次采样相对既有偏移的离谱阈值（秒）。逻辑时钟差、甚至小步 NTP 校时都远小于它，
# 而「把 epoch 毫秒当秒用」这类错误是 1e9 量级。超过就作废这一次估计，不动 EMA。
_OFFSET_JUMP_MAX_S = 3600.0
_OFFSET_REJECT_LIMIT = 50


def ingest_frames(
    device_code: str,
    ward_id: str,
    zone_id: Any,
    frames: list[dict],
    stream: str | None = None,
) -> int:
    """接收一批板端帧，换算到云端时间轴后广播。返回广播的帧数。"""
    global _frames_in, _frames_out, _dropped

    if not frames:
        return 0
    zk = None if zone_id is None else str(zone_id)

    # 记设备台账（stats() 要按病区/区域过滤设备清单，光有 device_code 分不出是谁的）。
    # 淘汰要逐台做，不能整体 clear：`_offsets` 是按 device_code 存的另一个
    # 字典，_device_scope 一清、它就成了一批查不到归属的孤儿 —— stats() 对查不到
    # 归属的 code 直接跳过，于是 /api/v1/live 会出现「frames_in 在涨、offsets 空」
    # 这种自相矛盾的诊断，而且在看的设备偏移会整段消失。两边同步淘汰最旧的一台，
    # 顺手把 `_offsets` 也真正限了上界（原来它无界增长，这个兜底只兜了一半）。
    if len(_device_scope) >= _DEVICE_SCOPE_MAX:
        oldest = next(iter(_device_scope))   # dict 保序：取最早写入的那台
        _device_scope.pop(oldest, None)
        _offsets.pop(oldest, None)
        _offset_rejects.pop(oldest, None)
    _device_scope[device_code] = (ward_id, zk)

    clean: list[tuple[float, int, int, Any]] = []
    newest = 0.0
    for frame in frames:
        _frames_in += 1
        if not isinstance(frame, dict):
            continue
        ts_raw = frame.get("ts")
        if isinstance(ts_raw, bool):
            continue
        try:
            board_ts = float(ts_raw or 0.0)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(board_ts) or board_ts <= 0:
            continue
        w_raw, h_raw = frame.get("w"), frame.get("h")
        if isinstance(w_raw, bool) or isinstance(h_raw, bool):
            continue
        try:
            w = int(w_raw or 0)
            h = int(h_raw or 0)
        except (TypeError, ValueError, OverflowError):
            continue
        if w <= 0 or h <= 0:
            # 前端要拿 w/h 把归一化坐标还原成像素；0 会直接除零，
            # 负数是坏输入。这种帧画不出来，丢掉。
            continue
        persons = frame.get("persons")
        clean.append((board_ts, w, h, persons if isinstance(persons, list) else []))
        if board_ts > newest:
            newest = board_ts

    if not clean:
        return 0

    now_cloud = time.time()
    offset = _offsets.get(device_code)
    sample = now_cloud - newest
    if math.isfinite(sample):
        if offset is None:
            offset = sample
            _offset_rejects.pop(device_code, None)
        elif abs(sample - offset) <= _OFFSET_JUMP_MAX_S:
            offset = offset * (1.0 - _EMA_ALPHA) + sample * _EMA_ALPHA
            _offset_rejects.pop(device_code, None)
        else:
            # 离谱的采样（epoch 差 / 野生 ts）——这一批的估计作废，沿用旧偏移。
            # 但连续 N 次都离谱就不是噪声了，见 _OFFSET_REJECT_LIMIT 的说明。
            n = _offset_rejects.get(device_code, 0) + 1
            if n >= _OFFSET_REJECT_LIMIT:
                log.warning(
                    "live_meta.offset_reacquire device=%s old=%.3f new_sample=%.3f "
                    "rejected=%d —— 板端时钟疑似大跳，重新对齐",
                    device_code, offset, sample, n,
                )
                offset = sample
                _offset_rejects.pop(device_code, None)
            else:
                _offset_rejects[device_code] = n
        if math.isfinite(offset):
            _offsets[device_code] = offset
        else:
            # 已经有脏值在 _offsets 里（老进程/极端输入）：丢掉这台设备的偏移重来，
            # 否则它永远算不出一帧能用的时间戳。
            _offsets.pop(device_code, None)
            offset = None
    if offset is None:
        # 这台设备还没有可用的偏移：宁可不画，也不要画在错误的时间轴上
        # （骨架画在错的时刻 = 用户看到"人跌倒了但骨架没动"，比没有更糟）。
        return 0

    # ── 第二遍：换算 + 广播 ──────────────────────────────────────────────
    # 广播的是板端采样时刻换算过来的云端时刻，不是"现在"：
    # 板端到云端有 ~100 ms 延迟，用"现在"会让骨架比视频超前。
    published = 0
    for board_ts, w, h, persons in clean:
        item = {
            "event": "meta",
            "data": {
                "t": round((board_ts + offset) * 1000.0, 1),   # 毫秒，前端直接用
                "w": w,
                "h": h,
                "persons": persons,
            },
        }
        for queue, (sub_ward, sub_zones, sub_stream) in list(_queues.items()):
            if sub_stream is not None and stream is not None:
                if stream != sub_stream:
                    continue
            else:
                if sub_ward != ward_id:
                    continue
                # None = 护士长（不收窄）；元组 = 护士，只收本区域的。
                # zk 为 None（设备没归区）时 `None in ('…',)` 为假 ⇒ 护士收不到，正确。
                if sub_zones is not None and zk not in sub_zones:
                    continue
            try:
                queue.put_nowait(item)
                _frames_out += 1
            except asyncio.QueueFull:
                # 丢最旧的再塞：实时数据积压 = 画错的东西
                try:
                    queue.get_nowait()
                    queue.put_nowait(item)
                except (asyncio.QueueEmpty, asyncio.QueueFull):
                    pass
                _dropped += 1
        published += 1

    return published


def subscribe(
    ward_id: str, zones: tuple[str, ...] | None, stream: str | None = None
) -> asyncio.Queue:
    """订阅帧广播。``zones`` 是 ``deps.zone_keys(scoped_zones(user))`` 的结果：
    ``None`` = 不收窄（护士长），元组 = 只收这些区域（空元组 = 一帧都收不到）。"""
    queue: asyncio.Queue = asyncio.Queue(maxsize=QUEUE_MAX)
    _queues[queue] = (ward_id, zones, stream)
    return queue


def unsubscribe(queue: asyncio.Queue) -> None:
    _queues.pop(queue, None)


_KEEP = object()


def refresh_scope(
    queue: asyncio.Queue,
    ward_id: str,
    zones: tuple[str, ...] | None,
    stream: Any = _KEEP,
) -> None:
    """心跳时复核订阅范围。"""
    cur = _queues.get(queue)
    if cur is None:
        return
    if stream is _KEEP:
        stream = cur[2]
    _queues[queue] = (ward_id, zones, stream)  # type: ignore[assignment]


def server_now_ms() -> int:
    """给 `hello` 事件用：浏览器拿它校正自己的时钟偏差。"""
    return int(time.time() * 1000)


def stats(ward_id: str, zones: tuple[str, ...] | None) -> dict[str, Any]:
    """诊断计数 + 本账号可见的设备时钟偏移。"""
    offsets = {}
    for code, off in _offsets.items():
        # 非有限值一律不出现在响应里：Starlette 的 JSONResponse 用
        # allow_nan=False，一个 NaN 就能把 /api/v1/live 打成 500。
        if not math.isfinite(off):
            continue
        scope = _device_scope.get(code)
        if scope is None:
            continue
        dev_ward, dev_zone = scope
        if dev_ward != ward_id:
            continue
        if zones is not None and dev_zone not in zones:
            continue
        offsets[code] = round(off, 3)

    return {
        "subscribers": len(_queues),
        "frames_in": _frames_in,
        "frames_out": _frames_out,
        "dropped": _dropped,
        "offsets": offsets,
    }
