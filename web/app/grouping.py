"""纯分桶函数：点位 → 病区 → 区域（ 「十改」）。"""
from __future__ import annotations


def ward_groups(points: list[dict], zones: list[dict],
                wards: list[dict]) -> list[dict]:
    """把点位分好 病区 → 区域 → 卡片 的桶。"""
    by_zone: dict[str | None, list[dict]] = {}
    for p in points:
        by_zone.setdefault(p.get("zone_id"), []).append(p)

    groups: list[dict] = []
    for w in wards:
        wid = w["ward_id"]
        zone_groups: list[dict] = []
        w_points: list[dict] = []
        for z in zones:
            if z.get("ward_id") != wid:
                continue
            members = by_zone.get(z["zone_id"]) or []
            if not members:
                continue
            zone_groups.append({"zone": z, "points": members})
            w_points.extend(members)
        # 没归区域的点位，按它自己的 ward_id 落在这个病区下的「未分组」段
        loose = [p for p in points
                 if p.get("ward_id") == wid and not p.get("zone_id")]
        w_points.extend(loose)
        groups.append({
            "ward_id": wid,
            "ward_name": w.get("ward_name") or wid,
            "floor": w.get("floor"),
            "zone_groups": zone_groups,
            "loose_points": loose,
            "zone_total": sum(1 for z in zones if z.get("ward_id") == wid),
            "point_total": len(w_points),
            "bound_total": sum(1 for p in w_points if p.get("device_id")),
        })
    return groups


def device_ward_groups(devices: list[dict], zones: list[dict],
                       wards: list[dict]) -> list[dict]:
    """把设备分好 病区 → 区域 → 卡片 的桶（ 十七改）。"""
    zone_by_id = {z["zone_id"]: z for z in zones}
    groups: list[dict] = []
    for w in wards:
        wid = w["ward_id"]
        w_devs = [d for d in devices
                  if d.get("live_point_id") and d.get("live_ward_id") == wid]
        zone_groups: list[dict] = []
        for z in zones:
            if z.get("ward_id") != wid:
                continue
            # 再按 live_ward_id 过一遍：点位病区与区域病区不一致时不许同台设备落两处
            members = [d for d in w_devs if d.get("zone_id") == z["zone_id"]]
            if not members:
                continue
            zone_groups.append({"zone": z, "devices": members})
        loose = [d for d in w_devs
                 if (zone_by_id.get(d.get("zone_id")) is None
                     or zone_by_id[d["zone_id"]].get("ward_id") != wid)]
        groups.append({
            "ward_id": wid,
            "ward_name": w.get("ward_name") or wid,
            "floor": w.get("floor"),
            "zone_groups": zone_groups,
            "loose_devices": loose,
            "device_total": len(w_devs),
        })
    return groups
