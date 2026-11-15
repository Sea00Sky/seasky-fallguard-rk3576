# -*- coding: utf-8 -*-
"""「这台设备的画面/告警算在哪个点位」—— 归属判据的唯一真源。"""
from __future__ import annotations

# 用法：``FROM fg.device d`` + LIVE_POINT_JOIN + ``WHERE ...``
# ``uq_point_device_id`` 保证一台设备最多被一个点位绑 ⇒ 这个 LEFT JOIN 不会放大行数。
LIVE_POINT_JOIN = """
  LEFT JOIN fg.point lp
         ON lp.device_id = d.device_id
        AND lp.is_active
"""

LIVE_POINT_FIELDS = (
    "lp.point_id AS live_point_id, lp.name AS live_point_name, "
    "lp.ward_id AS live_ward_id, lp.zone_id AS live_zone_id, "
    "lp.light_key AS live_light_key"
)


def apply_stream_attribution(dev: dict) -> dict:
    """把设备行的 ward / zone / location 换成「绑定了它的那个点位」的。"""
    if dev.get("live_point_id"):
        dev["ward_id"] = dev["live_ward_id"]
        dev["zone_id"] = dev["live_zone_id"]
        # 门灯键可能就是点位名，两者取其一（与点位页、告警筛选同一口径）
        dev["location"] = dev["live_light_key"] or dev["live_point_name"]
    else:
        dev["zone_id"] = None
        dev["location"] = None
    return dev