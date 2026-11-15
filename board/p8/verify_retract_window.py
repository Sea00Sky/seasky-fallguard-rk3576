"""守住「每日重启」唯一的安全依据：云端撤回匹配的两道时间窗。"""
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INGEST = ROOT / "web" / "app" / "routes" / "ingest.py"
TIMER = ROOT / "p8" / "fallguard-daily-restart.timer"

MAX_EXACT_MIN = 10          # RETRACT_WINDOW_SQL 的上限
MAX_FUZZY_MIN = 2           # RETRACT_FUZZY_WINDOW_SQL 的上限（关键的那道）
_IVL = re.compile(r"^now\(\)\s*-\s*interval\s*'(\d+)\s+minutes?'$")
_DAILY = re.compile(r"^\*-\*-\*\s+\d{2}:\d{2}:\d{2}$")

fails: list[str] = []
oks: list[str] = []
skips: list[str] = []


def literal_assignments(path: Path) -> dict[str, object]:
    """取源码里顶层 `NAME = <字面量>` 的值（用 ast，不受排版影响）。"""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    out: dict[str, object] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        tgt = node.targets[0]
        if not isinstance(tgt, ast.Name):
            continue
        try:
            out[tgt.id] = ast.literal_eval(node.value)
        except ValueError:
            continue
    return out


def check_window(name: str, consts: dict[str, object], limit: int) -> None:
    if name not in consts:
        fails.append(f"{name} 不见了（改名？删了？）—— 撤回匹配的护栏没了")
        return
    raw = consts[name]
    if not isinstance(raw, str):
        fails.append(f"{name} 不是字符串字面量：{raw!r}")
        return
    m = _IVL.match(raw.strip())
    if not m:
        fails.append(f"{name} 形状不认识（期望 `now() - interval 'N minutes'`）：{raw!r}")
        return
    n = int(m.group(1))
    if n <= 0:
        fails.append(f"{name} = {n} 分钟，窗口为零 ⇒ 撤回永远匹配不上，全部变孤儿")
    elif n > limit:
        fails.append(f"{name} = {n} 分钟 > 上限 {limit} —— 每日重启会开始静默错配历史告警")
    else:
        oks.append(f"{name} = {n} 分钟（<= {limit}）")


def check_timer() -> None:
    if not TIMER.exists():
        fails.append(f"定时器不见了：{TIMER.name} —— 若已弃用请一并删掉本检查")
        return
    text = TIMER.read_text(encoding="utf-8")
    cal = None
    persistent = None
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("#") or "=" not in s:
            continue
        k, _, v = s.partition("=")
        k, v = k.strip(), v.strip()
        if k == "OnCalendar" and cal is None:
            cal = v
        elif k == "Persistent" and persistent is None:
            persistent = v

    if cal is None:
        fails.append("定时器里找不到 OnCalendar")
    elif not _DAILY.match(cal):
        fails.append(
            f"OnCalendar = {cal!r} 不是「每日固定时刻」形式 —— "
            f"间隔可能已经小于撤回窗的 2 分钟")
    else:
        oks.append(f"OnCalendar = {cal}（每日一次，远大于 2 分钟窗）")

    if persistent is None:
        fails.append("定时器里找不到 Persistent")
    elif persistent.lower() != "false":
        fails.append(
            f"Persistent = {persistent} —— 应为 false：开机本身就是一次完整重置，"
            f"补跑只会多一段无谓的盲区")
    else:
        oks.append("Persistent = false")


def check_windows() -> None:
    """查两道窗。板端没有 web/ 源码 —— 那是显式跳过，不是通过。"""
    if not INGEST.exists():
        if (ROOT / "web").is_dir():
            fails.append(f"web/ 在，但找不到 {INGEST.relative_to(ROOT)} —— 被挪走或改名了")
        else:
            skips.append(
                "本机没有 web/ 源码（板端就是这样）⇒ **跳过**两道窗的检查；"
                "请在仓库/云端侧再跑一次本脚本")
        return
    consts = literal_assignments(INGEST)
    check_window("RETRACT_WINDOW_SQL", consts, MAX_EXACT_MIN)
    check_window("RETRACT_FUZZY_WINDOW_SQL", consts, MAX_FUZZY_MIN)


def main() -> int:
    for lst in (fails, oks, skips):     # 允许同进程内重复调用（负向测试会这么做）
        lst.clear()
    print("=" * 74)
    print("撤回匹配时间窗守卫（保住「每日重启」的合法前提）")
    print("=" * 74)

    check_windows()
    check_timer()

    for s in oks:
        print(f"  PASS  {s}")
    for s in skips:
        print(f"  SKIP  {s}")
    for s in fails:
        print(f"  FAIL  {s}")

    print("-" * 74)
    if fails:
        print(f" {len(fails)} 项不通过。改这两道窗 / 这个定时器之前，"
              f"先读 web/app/routes/ingest.py 里它们上方的注释。")
        return 1
    if skips:
        print(" 定时器没问题，但有检查被跳过（exit=2）。"
              "别把这个返回码当成「全通过」——去有 web/ 源码的机器上再跑一次。")
        return 2
    print(" 全部通过：每日重启与云端撤回匹配互不冲突。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
