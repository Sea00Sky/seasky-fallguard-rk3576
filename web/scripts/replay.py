#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""衡护 HENGUARD —— 演示 / 回归数据回放脚本。"""

from __future__ import annotations

import argparse
import getpass
import http.cookiejar
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

DEFAULT_CLOUD = os.environ.get("FG_CLOUD_URL", "http://127.0.0.1")
DEFAULT_DEVICE = "K7C-4FA-01"

ACTIONS = {"ACK", "ARRIVE", "RESOLVE", "FALSE_ALARM", "CLOSE", "REOPEN", "NOTE", "RETRACT"}
OUTCOMES = {"ASSISTED", "NO_ABNORMALITY", "MEDICAL", "HOSPITALIZED", "OTHER"}
FALSE_ALARM_REASONS = {"MODEL_MISJUDGE", "OCCLUSION", "NOT_PATIENT", "DUPLICATE", "OTHER"}
# 与 app/routes/ingest.py 的 _parse_event 对齐：这些是服务端必填项；
# window_end / hybrid_score / bbox / snapshot_seq 都是可选项。
REQUIRED_EVENT_FIELDS = (
    "seq", "ts", "person_id", "state",
    "window_start", "model_ver", "prob_fall", "coverage",
)

_APP_TZ = None


# ── 基础工具 ─────────────────────────────────────────────────────────
def _utf8_stdout() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


def _app_tz():
    global _APP_TZ
    if _APP_TZ is None:
        try:
            from zoneinfo import ZoneInfo

            _APP_TZ = ZoneInfo("Asia/Shanghai")
        except Exception:  # noqa: BLE001 - 缺 tzdata 时退回固定 +8
            _APP_TZ = timezone(timedelta(hours=8))
    return _APP_TZ


def _parse_at(text: str) -> datetime:
    dt = datetime.fromisoformat(str(text).strip().replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=_app_tz())
    return dt


def _mask(secret: str | None) -> str:
    if not secret:
        return "***"
    text = str(secret)
    return text[:4] + "***" if len(text) > 4 else "***"


def _decode_body(raw: bytes) -> Any:
    if not raw:
        return {}
    text = raw.decode("utf-8", errors="replace")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"_raw": text[:500]}


def _err_text(data: Any) -> str:
    if isinstance(data, dict):
        error = data.get("error")
        if isinstance(error, dict):
            code = error.get("code") or "?"
            message = error.get("message") or ""
            return f"{code}: {message}".strip()
        if "_raw" in data:
            return str(data["_raw"])[:200]
    return str(data)[:200]


# ── HTTP 客户端（标准库，带 Cookie 会话）─────────────────────────────
class ApiClient:
    def __init__(self, base: str, timeout: float = 10.0, retries: int = 2) -> None:
        self.base = base.rstrip("/")
        self.timeout = timeout
        self.retries = max(0, retries)
        self._jar = http.cookiejar.CookieJar()
        self._opener = urllib.request.build_opener(
            urllib.request.HTTPCookieProcessor(self._jar)
        )

    def request(self, method: str, path: str, body: Any = None) -> tuple[int, Any]:
        url = self.base + path
        headers = {
            "Accept": "application/json",
            # 设计 §10.2：所有写操作都要求这个头，缺失会被服务端拒绝
            "X-Requested-With": "XMLHttpRequest",
            "User-Agent": "hengard-replay/1.0",
        }
        payload = None
        if body is not None:
            payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
            headers["Content-Type"] = "application/json; charset=utf-8"

        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            request = urllib.request.Request(url, data=payload, headers=headers, method=method)
            try:
                with self._opener.open(request, timeout=self.timeout) as response:
                    return response.status, _decode_body(response.read())
            except urllib.error.HTTPError as exc:
                status = exc.code
                data = _decode_body(exc.read())
                if 500 <= status < 600 and attempt < self.retries:
                    time.sleep(min(2 ** attempt, 8))
                    continue
                return status, data
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                last_error = exc
                if attempt < self.retries:
                    time.sleep(min(2 ** attempt, 8))
                    continue
                raise RuntimeError(f"{method} {url} 网络失败：{exc}") from exc
        raise RuntimeError(f"{method} {url} 重试耗尽：{last_error}")

    def login(self, login_name: str, password: str) -> dict:
        status, data = self.request(
            "POST", "/api/v1/auth/login", {"login_name": login_name, "password": password}
        )
        if status != 200:
            raise RuntimeError(f"登录失败（HTTP {status}）：{_err_text(data)}")
        return data if isinstance(data, dict) else {}


# ── 事件文件读取与校验 ───────────────────────────────────────────────
def _load_items(path_text: str) -> tuple[list[Any], str | None]:
    raw_text = Path(path_text).read_text(encoding="utf-8")
    data = json.loads(raw_text)
    if isinstance(data, dict) and "events" in data:
        items = data["events"]
        device = data.get("device_code")
    elif isinstance(data, list):
        items = data
        device = None
    elif isinstance(data, dict):
        items = [data]
        device = None
    else:
        raise ValueError('顶层必须是数组，或 {"device_code": ..., "events": [...]}')
    if not isinstance(items, list):
        raise ValueError('"events" 必须是数组')
    return items, device


def _maybe_sort(items: list[Any], keep_order: bool) -> list[Any]:
    if keep_order or not items:
        return items
    if all(isinstance(item, dict) and item.get("at") for item in items):
        try:
            return sorted(items, key=lambda item: _parse_at(item["at"]))
        except ValueError:
            print("[警告] 有条目的 at 不是合法 ISO 时间，保持文件顺序", file=sys.stderr)
    return items


def _split_item(item: Any) -> tuple[Any, list[Any], str | None]:
    if not isinstance(item, dict):
        raise ValueError("条目必须是 JSON 对象")
    if "push" in item:
        then = item.get("then") or []
        if not isinstance(then, list):
            raise ValueError('"then" 必须是数组')
        return item["push"], then, item.get("at")
    return item, [], item.get("at")


def _item_actions(item: Any) -> list[Any]:
    if isinstance(item, dict) and "push" in item:
        then = item.get("then") or []
        return then if isinstance(then, list) else []
    return []


def _validate_push(push: Any) -> str | None:
    if not isinstance(push, dict):
        return "push 不是 JSON 对象"
    if push.get("type") != "fall_alert":
        return f"type 必须是 fall_alert（实际 {push.get('type')!r}）"
    event = push.get("event")
    if not isinstance(event, dict):
        return "缺少 event 对象"
    missing = [key for key in REQUIRED_EVENT_FIELDS if key not in event]
    if missing:
        return "event 缺少字段：" + "、".join(missing)
    if event.get("state") not in (1, 2):
        return "event.state 必须是 1（确认）或 2（撤回），0（疑似）网关不推"
    return None


def _validate_action(action: Any) -> str | None:
    if not isinstance(action, dict):
        return "动作必须是 JSON 对象"
    name = action.get("action")
    if name not in ACTIONS:
        return f"未知动作 {name!r}（可用：{' / '.join(sorted(ACTIONS))}）"
    if name == "RESOLVE" and action.get("outcome") not in OUTCOMES:
        return "RESOLVE 需要 outcome：" + " / ".join(sorted(OUTCOMES))
    if name == "FALSE_ALARM" and action.get("reason") not in FALSE_ALARM_REASONS:
        return "FALSE_ALARM 需要 reason：" + " / ".join(sorted(FALSE_ALARM_REASONS))
    if name == "REOPEN" and not (action.get("reason") or action.get("note")):
        return "REOPEN 必须写原因（reason 或 note）"
    return None


def _action_body(action: dict) -> dict:
    body: dict[str, Any] = {"action": action["action"]}
    for key in ("outcome", "reason", "note"):
        if action.get(key) is not None:
            body[key] = action[key]
    body["client_request_id"] = action.get("client_request_id") or str(uuid.uuid4())
    if "expected_revision" in action:
        body["expected_revision"] = action["expected_revision"]
    return body


def _action_suffix(action: dict) -> str:
    parts = [f"{key}={action[key]}" for key in ("outcome", "reason") if action.get(key)]
    return " ".join(parts)


def _ingest_result(status: int, data: Any) -> tuple[bool, str | None, bool, str | None]:
    if status != 200:
        return False, None, False, f"HTTP {status}：{_err_text(data)}"
    if not isinstance(data, dict) or not data.get("ok"):
        return False, None, False, f"响应异常：{_err_text(data)}"
    return True, data.get("event_uid"), bool(data.get("duplicate")), None


def _fetch_revision(client: ApiClient, event_uid: str) -> int | None:
    path = "/api/v1/events/" + urllib.parse.quote(str(event_uid), safe="")
    status, data = client.request("GET", path)
    if status != 200 or not isinstance(data, dict):
        return None
    revision = data.get("revision")
    if revision is None and isinstance(data.get("event"), dict):
        revision = data["event"].get("revision")
    return revision if isinstance(revision, int) else None


def _run_actions(client: ApiClient, event_uid: str, actions: list[Any],
                 counters: Counter) -> None:
    revision: int | None = None
    revision_known = False
    warned = False

    for action in actions:
        problem = _validate_action(action)
        if problem:
            print(f"    [跳过动作] {problem}")
            counters["action_fail"] += 1
            continue

        body = _action_body(action)
        if "expected_revision" not in body:
            if not revision_known:
                revision = _fetch_revision(client, event_uid)
                revision_known = revision is not None
                if revision is None and not warned:
                    print("    [警告] 读不到 revision，后续动作不发送 expected_revision")
                    warned = True
            if revision is not None:
                body["expected_revision"] = revision

        path = "/api/v1/events/" + urllib.parse.quote(str(event_uid), safe="") + "/actions"
        try:
            status, data = client.request("POST", path, body)
        except RuntimeError as exc:
            print(f"    [动作失败] {body['action']}：{exc}", file=sys.stderr)
            counters["action_fail"] += 1
            continue

        if status == 200:
            counters["action_ok"] += 1
            new_state = data.get("disposition_state") if isinstance(data, dict) else None
            new_revision = data.get("revision") if isinstance(data, dict) else None
            print(f"    [动作] {body['action']} → 200 {new_state or ''}".rstrip())
            if isinstance(new_revision, int):
                revision = new_revision
                revision_known = True
            else:
                revision_known = False
        elif status == 409:
            counters["action_conflict"] += 1
            print(f"    [动作冲突] {body['action']} → 409 {_err_text(data)}")
        else:
            counters["action_fail"] += 1
            print(f"    [动作失败] {body['action']} → {status} {_err_text(data)}", file=sys.stderr)


# ── 参数解析与主流程 ─────────────────────────────────────────────────
def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="replay.py",
        description="衡护 HENGUARD 演示数据回放（POST 到 /api/v1/ingest/{device}/{key}）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--file", required=True, help="事件 JSON 文件路径")
    parser.add_argument(
        "--cloud",
        default=os.environ.get("FG_CLOUD_BASE", DEFAULT_CLOUD),
        help="云端根地址（默认 %(default)s）",
    )
    parser.add_argument(
        "--device",
        default=None,
        help=f"设备编号；默认取文件顶层 device_code，再默认 {DEFAULT_DEVICE}",
    )
    parser.add_argument(
        "--key",
        default=None,
        help="ingest_key；也可用环境变量 FG_INGEST_KEY（不要写进事件文件）",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=0.0,
        help="时间倍率：2=两倍速，0=完全不等待（默认）",
    )
    parser.add_argument(
        "--max-wait",
        type=float,
        default=5.0,
        help="单次等待上限秒数（默认 5，防止演示文件跨天时卡住）",
    )
    parser.add_argument("--dry-run", action="store_true", help="只打印计划，不发任何网络请求")
    parser.add_argument("--login", default=None, help="执行处置动作的护士登录名")
    parser.add_argument(
        "--password",
        default=None,
        help="密码；建议用环境变量 FG_REPLAY_PASSWORD 或交互输入，避免进 shell 历史",
    )
    parser.add_argument("--no-actions", action="store_true", help="只灌事件，不执行 then 里的动作")
    parser.add_argument("--keep-order", action="store_true", help="严格按文件顺序，不按 at 排序")
    parser.add_argument("--timeout", type=float, default=10.0, help="单请求超时秒数（默认 10）")
    parser.add_argument("--retry", type=int, default=2, help="网络错误 / 5xx 重试次数（默认 2）")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    _utf8_stdout()
    args = _parse_args(argv)

    try:
        raw_items, file_device = _load_items(args.file)
    except (OSError, ValueError) as exc:
        print(f"[错误] 读取事件文件失败：{exc}", file=sys.stderr)
        return 1
    if not raw_items:
        print("[错误] 事件文件里没有条目", file=sys.stderr)
        return 1

    device = args.device or file_device or DEFAULT_DEVICE
    key = args.key or os.environ.get("FG_INGEST_KEY")
    if not key and not args.dry_run:
        print(
            "[错误] 未提供 ingest_key：用 --key 或环境变量 FG_INGEST_KEY"
            "（密钥不要写进演示文件）",
            file=sys.stderr,
        )
        return 1

    items = _maybe_sort(raw_items, args.keep_order)
    total = len(items)
    counters: Counter = Counter()
    client = ApiClient(args.cloud, timeout=args.timeout, retries=args.retry)

    need_actions = any(_item_actions(item) for item in items) and not args.no_actions
    if need_actions and not args.dry_run:
        login_name = args.login or os.environ.get("FG_REPLAY_LOGIN")
        if not login_name:
            print(
                "[错误] 文件里含处置动作，需要登录：用 --login 指定护士账号；"
                "只想灌事件请加 --no-actions",
                file=sys.stderr,
            )
            return 1
        password = args.password or os.environ.get("FG_REPLAY_PASSWORD")
        if password is None:
            password = getpass.getpass(f"{login_name} 的密码：")
        try:
            login_data = client.login(login_name, password)
        except RuntimeError as exc:
            print(f"[错误] {exc}", file=sys.stderr)
            return 1
        user = login_data.get("user") or {}
        print(
            f"[登录] {user.get('display_name', login_name)}"
            f"（{user.get('role', '?')} / {user.get('ward_id', '?')}）"
        )
        if user.get("must_change_password"):
            print("[警告] 该账号 must_change_password=true；服务端可能拒绝处置动作，先用 passwd 改密。")

    print(
        f"[开始] 共 {total} 条，设备 {device}，密钥 {_mask(key)}，"
        f"云端 {args.cloud}{'（dry-run）' if args.dry_run else ''}"
    )

    prev_at: datetime | None = None
    try:
        for index, item in enumerate(items, start=1):
            try:
                push, then_actions, at_text = _split_item(item)
            except ValueError as exc:
                print(f"[{index}/{total}] [跳过] {exc}")
                counters["bad"] += 1
                continue

            at = None
            if at_text:
                try:
                    at = _parse_at(at_text)
                except ValueError:
                    print(f"[{index}/{total}] [警告] at 不是合法 ISO 时间，本条不做节流：{at_text}")
            if at is not None and prev_at is not None and args.speed > 0:
                wait = min(max((at - prev_at).total_seconds() / args.speed, 0.0), args.max_wait)
                if wait > 0:
                    time.sleep(wait)
            if at is not None:
                prev_at = at

            problem = _validate_push(push)
            if problem:
                print(f"[{index}/{total}] [跳过] {problem}")
                counters["bad"] += 1
                continue

            event = push["event"]
            seq = event.get("seq")
            state = event.get("state")

            if args.dry_run:
                print(
                    f"[dry-run {index}/{total}] POST /api/v1/ingest/{device}/{_mask(key)}"
                    f"  seq={seq} state={state}"
                )
                counters["dry"] += 1
                if not args.no_actions:
                    for action in then_actions:
                        action_problem = _validate_action(action)
                        if action_problem:
                            print(f"    [跳过动作] {action_problem}")
                            counters["bad_action"] += 1
                            continue
                        suffix = _action_suffix(action)
                        print(f"    -> {action.get('action')}{(' ' + suffix) if suffix else ''}")
                continue

            path = (
                "/api/v1/ingest/"
                + urllib.parse.quote(str(device), safe="")
                + "/"
                + urllib.parse.quote(str(key), safe="")
            )
            try:
                status, data = client.request("POST", path, push)
            except RuntimeError as exc:
                print(f"[{index}/{total}] [失败] seq={seq}：{exc}", file=sys.stderr)
                counters["ingest_fail"] += 1
                continue

            ok, event_uid, duplicate, error = _ingest_result(status, data)
            if not ok:
                print(f"[{index}/{total}] [失败] seq={seq} state={state} {error}", file=sys.stderr)
                counters["ingest_fail"] += 1
                continue
            if duplicate:
                counters["ingest_dup"] += 1
                print(f"[{index}/{total}] [重复] seq={seq} state={state}（已存在，跳过动作）")
                continue

            counters["ingest_ok"] += 1
            print(f"[{index}/{total}] [入库] seq={seq} state={state} event_uid={event_uid}")
            if isinstance(data, dict) and data.get("orphan_retract"):
                counters["ingest_orphan"] += 1
                print(
                    "    [警告] 该撤回没有匹配到原事件，已记为孤儿撤回（orphan_retract=TRUE），"
                    "需要每日对账人工处理"
                )

            if args.no_actions or not then_actions:
                continue
            if not event_uid:
                print("    [跳过动作] 响应里没有 event_uid")
                counters["action_fail"] += 1
                continue
            _run_actions(client, str(event_uid), then_actions, counters)
    except KeyboardInterrupt:
        print("\n[中断] 用户按下 Ctrl+C", file=sys.stderr)
        return 130

    print()
    print("==== 回放完成 ====")
    if args.dry_run:
        print(f"干跑：计划发送 {counters['dry']} 条事件（未发出任何网络请求）")
    else:
        print(
            f"事件：共 {total} 条 / 入库 {counters['ingest_ok']} / "
            f"重复 {counters['ingest_dup']} / 孤儿撤回 {counters['ingest_orphan']} / "
            f"失败 {counters['ingest_fail']}"
        )
        print(
            f"动作：成功 {counters['action_ok']} / "
            f"冲突 {counters['action_conflict']} / 失败 {counters['action_fail']}"
        )
    if counters["bad"] or counters["bad_action"]:
        print(
            f"文件校验：跳过 {counters['bad']} 条事件、{counters['bad_action']} 个动作（见上方提示）"
        )

    failures = (
        counters["ingest_fail"]
        + counters["action_fail"]
        + counters["bad"]
        + counters["bad_action"]
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
