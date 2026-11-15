#!/usr/bin/env python3
"""③ 强窗破冷却 · 单元验收（`cooldown_break_prob`，默认关）。"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.multi.config import MultiConfig # noqa: E402
from src.multi.events import EventBook # noqa: E402

FAILS = []

FPS = 20.0


def check(name, got, want):
    ok = got == want
    print(f"  {'PASS' if ok else 'FAIL'}  {name}  got={got!r} want={want!r}")
    if not ok:
        FAILS.append(name)


def cfg(**kw):
    base = dict(confirm_windows=2, retract_windows=4, cooldown_s=30.0,
                strong_confirm=0.65)
    base.update(kw)
    return MultiConfig(**base)


def replay(c, rows):
    """rows = [(pid, prob, start, end), ...]，时钟 = end/fps。"""
    book = EventBook(c, clock=lambda: 0.0)
    t = 0.0
    evs = []
    for pid, p, s, e in rows:
        t = e / FPS
        book._clock = (lambda tt: (lambda: tt))(t)
        evs.extend(book.observe(pid, p, s, e, 1.0))
    return evs, book


def basic_seq():
    """先确认一次非强事件（p=0.6），4 低窗撤回，再进入冷却期。"""
    return [
        (1, 0.60, 0, 60),
        (1, 0.60, 10, 70), # confirm @10-70, t=3.5s（非强）
        (1, 0.10, 20, 80),
        (1, 0.10, 30, 90),
        (1, 0.10, 40, 100),
        (1, 0.10, 50, 110), # retract @50-110
    ]


def labels(evs):
    return [(e.state.label(), e.window_start, e.window_end)
            for e in evs]


def main() -> int:
    print("[1] 默认关：冷却期强窗仍被压")
    seq = basic_seq() + [(1, 0.999, 60, 120), (1, 0.999, 70, 130)]
    c0 = cfg()
    ev0, book0 = replay(c0, seq)
    check("默认 prob", c0.cooldown_break_prob, 0.0)
    check("默认 windows", c0.cooldown_break_windows, 2)
    check("事件", labels(ev0), [
        ("confirmed", 10, 70), ("retracted", 10, 70)])
    check("cooldown_suppressed", book0.counters.cooldown_suppressed, 1)
    check("cooldown_broken", book0.counters.cooldown_broken, 0)

    print("\n[2] 连续 2 窗 >=0.9：放行")
    c1 = cfg(cooldown_break_prob=0.9)
    ev1, book1 = replay(c1, seq)
    check("事件", labels(ev1), [
        ("confirmed", 10, 70), ("retracted", 10, 70),
        ("confirmed", 70, 130)])
    check("cooldown_broken", book1.counters.cooldown_broken, 1)
    check("cooldown_suppressed", book1.counters.cooldown_suppressed, 0)
    check("事件概率", round(ev1[-1].prob_fall, 4), 0.999)
    check("by_person", book1.counters.cooldown_broken_by_person, {1: 1})

    print("\n[3] 只有 1 个强窗：不放行")
    ev2, book2 = replay(c1, basic_seq() + [
        (1, 0.999, 60, 120), (1, 0.600, 70, 130)])
    check("事件", labels(ev2), [
        ("confirmed", 10, 70), ("retracted", 10, 70)])
    check("cooldown_broken", book2.counters.cooldown_broken, 0)
    check("cooldown_suppressed", book2.counters.cooldown_suppressed, 1)

    print("\n[4] 中间夹一个不够强的命中窗：连续清零，再攒两窗才放行")
    ev3, book3 = replay(c1, basic_seq() + [
        (1, 0.999, 60, 120), # hit1 / streak1
        (1, 0.600, 70, 130), # hit2（确认被压）/ streak0
        (1, 0.999, 80, 140), # hit3 / streak1
        (1, 0.999, 90, 150), # hit4 / streak2 ⇒ 放行
    ])
    check("事件", labels(ev3), [
        ("confirmed", 10, 70), ("retracted", 10, 70),
        ("confirmed", 90, 150)])
    check("cooldown_broken", book3.counters.cooldown_broken, 1)
    check("cooldown_suppressed", book3.counters.cooldown_suppressed, 1)

    print("\n[5] windows=3：两窗不够、三窗才放行")
    c3 = cfg(cooldown_break_prob=0.9, cooldown_break_windows=3)
    ev4, book4 = replay(c3, basic_seq() + [
        (1, 0.999, 60, 120), (1, 0.999, 70, 130)]) # 两窗：仍压
    check("两窗事件", labels(ev4), [
        ("confirmed", 10, 70), ("retracted", 10, 70)])
    check("两窗 broken", book4.counters.cooldown_broken, 0)
    ev5, book5 = replay(c3, basic_seq() + [
        (1, 0.999, 60, 120), (1, 0.999, 70, 130), (1, 0.999, 80, 140)])
    check("三窗事件", labels(ev5)[-1], ("confirmed", 80, 140))
    check("三窗 broken", book5.counters.cooldown_broken, 1)

    print("\n[6] 冷却已过：正常确认，不算破冷却")
    seq6 = basic_seq()
 # 从 t=5.5s 起每 0.5s 一个低窗，推到 t>33.5s（冷却 30s 后）
    t = 5.5
    s = 110
    while t <= 34.0:
        seq6.append((1, 0.0, s, s + 60))
        t += 0.5
        s += 10
    seq6 += [(1, 0.999, s, s + 60), (1, 0.999, s + 10, s + 70)]
    ev6, book6 = replay(c1, seq6)
    check("事件尾", labels(ev6)[-1][0], "confirmed")
    check("cooldown_broken", book6.counters.cooldown_broken, 0)

    print("\n[7] MultiConfig 环境变量")
    mc = MultiConfig.from_env({"FG_MULTI_COOLDOWN_BREAK_PROB": "0.9",
                              "FG_MULTI_COOLDOWN_BREAK_WINDOWS": "2"})
    check("from_env prob", mc.cooldown_break_prob, 0.9)
    check("from_env windows", mc.cooldown_break_windows, 2)
    check("describe", "cooldown_break_prob=0.9" in mc.describe(), True)
    check("as_dict 不含（云端契约不变）",
          "cooldown_break_prob" in mc.as_dict(), False)
    check("默认 prob", MultiConfig().cooldown_break_prob, 0.0)
    for env, name in (({"FG_MULTI_COOLDOWN_BREAK_PROB": "-0.1"}, "负 prob"),
                      ({"FG_MULTI_COOLDOWN_BREAK_WINDOWS": "0"}, "windows=0")):
        try:
            MultiConfig.from_env(env)
            check(name, "no error", "ValueError")
        except ValueError:
            check(name, "ValueError", "ValueError")

    print("\n[8] 默认关与显式 0 逐事件相同")
    a, _ = replay(cfg(), seq)
    b, _ = replay(cfg(cooldown_break_prob=0.0), seq)
    check("逐事件", labels(a), labels(b))

    print()
    print("COOLDOWN-BREAK", "PASS" if not FAILS else f"FAIL ({len(FAILS)})")
    return 0 if not FAILS else 1


if __name__ == "__main__":
    raise SystemExit(main())
