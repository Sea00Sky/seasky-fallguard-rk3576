"""帧元数据上行—— 网页 canvas 叠加骨架的数据源。"""
from __future__ import annotations

import http.client
import json
import queue
import threading
import time
import urllib.parse
from typing import Any


def parse_push_url(url: str) -> tuple[str, str, str]:
    """把 `.../api/v1/ingest/CODE/KEY` 拆成 (base, code, key)。"""
    parts = urllib.parse.urlsplit(url.strip())
    seg = [s for s in parts.path.split("/") if s]
    if len(seg) < 2 or "ingest" not in seg:
        raise ValueError(f"push_url 不符合 /api/v1/ingest/CODE/KEY 形式: {url[:40]}…")
    return f"{parts.scheme}://{parts.netloc}", seg[-2], seg[-1]


class MetaUploader:
    """后台线程把帧元数据 POST 到云端。线程安全，可随时 stop。"""

    def __init__(
        self,
        base: str,
        code: str,
        key: str,
        interval_s: float = 0.1,
        queue_max: int = 8,
        timeout_s: float = 3.0,
        stream: str | None = None,
    ) -> None:
        self.base = base.rstrip("/")
        self.code = code
        self.key = key
        # 五改：本机推的是哪条流（mediamtx 的 path 名，如 live）。
        # 云端拿它把骨架路由到正在看这条流的页面 —— 视频跟点位的 stream_path
        # 走，骨架跟设备走的话，流一改绑就会出现「有画面、没骨架」。
        # 留空 = 云端退回老口径（按设备病区/区域），不会因为没配就完全不显示。
        self.stream = (stream or "").strip()[:32] or None
        self.interval_s = float(interval_s)
        self.timeout_s = float(timeout_s)
        self._q: queue.Queue = queue.Queue(maxsize=int(queue_max))
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._conn: http.client.HTTPConnection | None = None
        self._last_submit = 0.0
        self.sent = 0
        self.dropped = 0
        self.failed = 0
        self.last_error = ""

    # ------------------------------------------------------------------ 生命周期

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="meta-uploader", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None
        self._close()

    # ------------------------------------------------------------------ 主循环侧

    def submit(self, persons: list[dict], width: int, height: int,
               ts: float | None = None) -> bool:
        """把一帧的人员快照塞进队列。非阻塞，队列满返回 False。"""
        now = time.monotonic()
        if now - self._last_submit < self.interval_s:
            return False
        self._last_submit = now

        frame = {
            "ts": float(ts if ts is not None else time.time()),
            "w": int(width),
            "h": int(height),
            "persons": [_pack_person(p) for p in persons],
        }
        try:
            self._q.put_nowait(frame)
            return True
        except queue.Full:
            # 丢最旧的，再塞新的：实时数据，旧的没有价值
            try:
                self._q.get_nowait()
                self._q.put_nowait(frame)
            except (queue.Empty, queue.Full):
                pass
            self.dropped += 1
            return False

    # ------------------------------------------------------------------ 后台线程

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                frame = self._q.get(timeout=0.5)
            except queue.Empty:
                continue
            # 同一批把已经排队的也带走（通常 1 帧，网络抖动时可能 2–3 帧）
            frames = [frame]
            while len(frames) < 8:
                try:
                    frames.append(self._q.get_nowait())
                except queue.Empty:
                    break
            self._post(frames)

    def _connect(self) -> http.client.HTTPConnection:
        if self._conn is not None:
            return self._conn
        parts = urllib.parse.urlsplit(self.base)
        cls = http.client.HTTPSConnection if parts.scheme == "https" else http.client.HTTPConnection
        self._conn = cls(parts.hostname, parts.port, timeout=self.timeout_s)
        return self._conn

    def _close(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:  # noqa: BLE001
                pass
            self._conn = None

    def _post(self, frames: list[dict]) -> None:
        path = f"/api/v1/ingest/{self.code}/{self.key}/frames"
        payload: dict[str, Any] = {"frames": frames}
        if self.stream:
            payload["stream"] = self.stream
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        try:
            conn = self._connect()
            conn.request(
                "POST", path, body=body,
                headers={"Content-Type": "application/json",
                         "Content-Length": str(len(body))},
            )
            resp = conn.getresponse()
            resp.read()
            if resp.status >= 400:
                self.failed += 1
                self.last_error = f"HTTP {resp.status}"
                self._close()          # 4xx/5xx 后连接状态不可靠，重建
                return
            self.sent += len(frames)
        except Exception as exc:       # noqa: BLE001 —— 网络问题一律丢帧，不重试
            self.failed += 1
            self.last_error = f"{type(exc).__name__}: {exc}"
            self._close()

    # ------------------------------------------------------------------ 信息

    def stats(self) -> dict[str, Any]:
        return {
            "sent": self.sent,
            "dropped": self.dropped,
            "failed": self.failed,
            "queued": self._q.qsize(),
            "last_error": self.last_error,
            "interval_s": self.interval_s,
        }


def _pack_person(p: dict) -> dict:
    """把 `display_persons` 的一条压成上行 JSON。"""
    kp = p.get("keypoints")
    flat: list[float] = []
    if kp is not None:
        try:
            for row in kp:
                flat.extend((round(float(row[0]), 1),
                             round(float(row[1]), 1),
                             round(float(row[2]), 3)))
        except (TypeError, IndexError, ValueError):
            flat = []
    bbox = p.get("bbox") or [0, 0, 0, 0]
    out = {
        "track_id": int(p.get("track_id", 0)),
        "bbox": [round(float(v), 1) for v in bbox[:4]],
        "kp": flat,
    }
    if p.get("person_id") is not None:
        out["person_id"] = int(p["person_id"])
    if p.get("prob") is not None:
        out["prob"] = round(float(p["prob"]), 3)
    if p.get("coverage") is not None:
        out["coverage"] = round(float(p["coverage"]), 3)
    if p.get("status") is not None:
        out["status"] = str(p["status"])
    return out
