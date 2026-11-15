"""板端遥测 → 设备卡上的小格（ 「十八改」）。"""
from __future__ import annotations

# key, 标签, 单位, 偏高阈值, 危险阈值, 小数位, 危险时的说法
# 顺序即页面顺序（ 遥测条重排）：FPS（管线）→ CPU/NPU/内存（同单位 %）
# → 温度（°C）。同单位的读数挨着放，扫读时不用来回换单位。
TELEMETRY_LIMITS = (
    # fps 是业务指标，不判级 —— 但 0 本身就该被看见，所以照样占一格
    ("fps", "FPS", "", 0, 0, 1, ""),
    ("cpu_pct", "CPU", "%", 80, 95, 0, "满载"),
    ("npu_load", "NPU", "%", 85, 95, 0, "满载"),
    ("mem_pct", "内存", "%", 85, 95, 0, "将满"),
    ("temp_c", "温度", "°C", 60, 75, 1, "过热"),
)


def _num(value) -> float | None:
    """JSONB 里的数可能是 int / float / 字符串 / None / bool —— 只要真数字。"""
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def fmt_uptime(value) -> str | None:
    """运行时长 → 人话（ 遥测条重排）。"""
    num = _num(value)
    if num is None or num < 0:
        return None
    if num < 60:
        return "不到 1 分钟"
    if num < 3600:
        return "%d 分钟" % int(num // 60)
    if num < 86400:
        return "%.1f 小时" % (num / 3600)
    return "%.1f 天" % (num / 86400)


def telemetry_chips(status) -> list[dict]:
    """``last_status`` → ``[{key, label, text, level, note, title}]``。"""
    st = status if isinstance(status, dict) else {}
    out: list[dict] = []
    for key, label, unit, warn, bad, digits, bad_word in TELEMETRY_LIMITS:
        num = _num(st.get(key))
        if num is None:
            continue
        text = "%.*f%s" % (digits, num, unit)
        level, note = "ok", ""
        title = None
        if warn:                      # fps（阈值 0）不判级
            if num >= bad:
                level, note = "bad", bad_word
            elif num >= warn:
                level, note = "warn", "偏高"
            title = "%s 正常 < %s%s，≥ %s%s %s" % (label, warn, unit, bad, unit, bad_word)
        out.append({
            "key": key, "label": label, "text": text,
            "level": level, "note": note, "title": title,
        })
    return out
