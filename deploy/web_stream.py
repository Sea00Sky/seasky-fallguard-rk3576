#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""web_stream.py — 感知进程内嵌 MJPEG Web 直播（ 新增）
==============================================================
给带屏 demo（class_display_demo.py）提供浏览器实时画面，与告警闭环同进程并联：
    摄像头 → 感知 → 同一个进程里既推 shm 告警、又当 web 服务器。"""

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

import cv2
import numpy as np


# ════════════════════════════════════════════════════════════════
# 通道：单槽 + 序号 + 订阅计数（单生产者 / 多消费者）
# ════════════════════════════════════════════════════════════════

class FrameChannel:
    """一路画面的"最新帧"槽位。"""

    def __init__(self, name, min_interval=0.0):
        self.name = name
        self.min_interval = float(min_interval or 0.0)   # 发布节流（秒）
        self._cond = threading.Condition()
        self._slot = None        # np.ndarray，预分配后复用（避免每帧 malloc）
        self._seq = 0
        self._wall = 0.0         # 该帧的发布时刻（板钟 epoch 秒）
        self._viewers = 0
        self._last_pub_t = 0.0
        # ── 统计（自检与诊断用）──
        self.pub_count = 0            # 真正发布的帧数
        self.dropped_no_viewer = 0    # 因无人观看而跳过
        self.dropped_throttle = 0     # 因节流而跳过
        self.encode_count = 0         # 消费者侧实际编码次数
        self.bytes_sent = 0

    # ── 生产者侧 ────────────────────────────────────────────────
    def publish(self, bgr):
        """把最新一帧放进槽位。返回 True 表示真的发布了。"""
        now = time.time()
        with self._cond:
            if self._viewers <= 0:
                self.dropped_no_viewer += 1
                return False
            if self.min_interval and (now - self._last_pub_t) < self.min_interval:
                self.dropped_throttle += 1
                return False
            if (self._slot is None or self._slot.shape != bgr.shape
                    or self._slot.dtype != bgr.dtype):
                self._slot = np.empty_like(bgr)
            np.copyto(self._slot, bgr)
            self._seq += 1
            self._wall = now
            self._last_pub_t = now
            self.pub_count += 1
            seq, viewers, cnt = self._seq, self._viewers, self.pub_count
            self._cond.notify_all()
        # 板端留痕（时延对账的原始凭据；每 30 帧一条，别改格式）
        if cnt % 30 == 0:
            print(f"[web] pub ch={self.name} seq={seq} wall_t={now:.3f} "
                  f"viewers={viewers}", flush=True)
        return True

    # ── 消费者侧 ────────────────────────────────────────────────
    def add_viewer(self):
        with self._cond:
            self._viewers += 1
            return self._viewers

    def remove_viewer(self):
        with self._cond:
            self._viewers = max(0, self._viewers - 1)
            return self._viewers

    def wait_frame(self, last_seq, timeout=1.0, stop_event=None):
        """等到一帧比 last_seq 新的画面。"""
        with self._cond:
            if self._seq == last_seq and not (stop_event and stop_event.is_set()):
                self._cond.wait(timeout)
            if stop_event is not None and stop_event.is_set():
                return None, None, None
            if self._seq == last_seq or self._slot is None:
                return None, None, None
            return self._seq, self._slot.copy(), self._wall

    def snapshot(self):
        with self._cond:
            return dict(seq=self._seq, viewers=self._viewers,
                        pub=self.pub_count, enc=self.encode_count,
                        drop_no_viewer=self.dropped_no_viewer,
                        drop_throttle=self.dropped_throttle,
                        kbytes=self.bytes_sent / 1024.0)


# ════════════════════════════════════════════════════════════════
# Hub：一进程一组通道
# ════════════════════════════════════════════════════════════════

class FrameHub:
    """主循环与 HTTP 线程之间的唯一接口。"""

    def __init__(self, channels=('ann', 'raw'), web_fps=10.0):
        mi = (1.0 / web_fps) if web_fps and web_fps > 0 else 0.0
        self.web_fps = web_fps
        self._ch = {n: FrameChannel(n, mi) for n in channels}
        self._meta = {}
        self._meta_lock = threading.Lock()
        self._stop = threading.Event()
        self._t0 = time.time()

    def has(self, name):
        return name in self._ch

    def publish(self, name, bgr):
        ch = self._ch.get(name)
        if ch is None or bgr is None:
            return False
        return ch.publish(bgr)

    def set_meta(self, **kw):
        with self._meta_lock:
            self._meta.update(kw)

    def viewers(self, name):
        ch = self._ch.get(name)
        return 0 if ch is None else ch.snapshot()['viewers']

    def any_viewers(self):
        return any(c.snapshot()['viewers'] > 0 for c in self._ch.values())

    def stats(self):
        with self._meta_lock:
            meta = dict(self._meta)
        return {
            'uptime_s': round(time.time() - self._t0, 1),
            'web_fps': self.web_fps,
            'channels': {n: c.snapshot() for n, c in self._ch.items()},
            **meta,
        }

    def close(self):
        """叫醒所有等帧的 HTTP 线程（服务停止时用）。"""
        self._stop.set()
        for c in self._ch.values():
            with c._cond:
                c._cond.notify_all()


# ════════════════════════════════════════════════════════════════
# HTTP 服务
# ════════════════════════════════════════════════════════════════

BOUNDARY = 'frame'

_PAGE = """<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>{title}</title>
<style>
 body{{margin:0;background:#111;color:#ddd;font-family:system-ui,"Microsoft YaHei",sans-serif}}
 header{{padding:10px 16px;background:#1b1b1b;border-bottom:1px solid #333}}
 h1{{margin:0;font-size:16px;font-weight:600}}
 #st{{margin-top:6px;font-size:13px;color:#9c9;font-family:ui-monospace,Consolas,monospace}}
 .wrap{{display:flex;flex-wrap:wrap;gap:16px;padding:16px}}
 .box{{background:#181818;border:1px solid #2a2a2a;border-radius:6px;padding:8px}}
 .box h2{{margin:0 0 6px 0;font-size:13px;font-weight:500;color:#8ab}}
 img{{display:block;width:100%;max-width:{w}px;height:auto;background:#000}}
</style></head><body>
<header><h1>{title}</h1><div id="st">连接中…</div></header>
<div class="wrap">
 <div class="box" style="flex:1 1 640px">
  <h2>标注画面 /stream（骨架 + 六类面板 + 跌倒横幅）</h2>
  <img src="/stream" alt="annotated">
 </div>
 <div class="box" style="flex:0 1 480px">
  <h2>原始画面 /raw（未标注）</h2>
  <img src="/raw" alt="raw">
 </div>
</div>
<script>
async function tick(){{
  try{{
    const r = await fetch('/api/status', {{cache:'no-store'}});
    const s = await r.json();
    const c = s.channels || {{}};
    document.getElementById('st').textContent =
      `fps=${{s.fps ?? '-'}}  frames=${{s.frames ?? '-'}}  people=${{s.people ?? '-'}}  `
      + `falling=${{s.falling ?? '-'}}  falls=${{s.falls_total ?? '-'}}  `
      + `ann[seq=${{c.ann?.seq ?? 0}} viewers=${{c.ann?.viewers ?? 0}}]  `
      + `raw[seq=${{c.raw?.seq ?? 0}} viewers=${{c.raw?.viewers ?? 0}}]  `
      + `uptime=${{s.uptime_s}}s`;
  }}catch(e){{ document.getElementById('st').textContent = '状态获取失败: ' + e; }}
}}
tick(); setInterval(tick, 1000);
</script></body></html>
"""


class MjpegServer:
    """纯标准库 MJPEG 服务（板上零依赖）。"""

    def __init__(self, hub, port=8888, bind='0.0.0.0', quality=70, scale=1.0,
                 title='FallGuard 实时监控'):
        self.hub = hub
        self.port = int(port)
        self.bind = bind
        self.quality = int(quality)
        self.scale = float(scale)
        self.title = title
        self.httpd = None
        self._th = None

    # ── 生命周期 ────────────────────────────────────────────────
    def start(self):
        handler = self._make_handler()
        self.httpd = ThreadingHTTPServer((self.bind, self.port), handler)
        self.httpd.daemon_threads = True          # 客户端线程不阻塞退出
        self._th = threading.Thread(target=self.httpd.serve_forever,
                                    kwargs={'poll_interval': 0.5},
                                    daemon=True, name='web-http')
        self._th.start()
        print(f"[web] MJPEG 服务已启动  http://{self.bind}:{self.port}/  "
              f"(fps<={self.hub.web_fps:g} q={self.quality} scale={self.scale:g})",
              flush=True)
        return self

    def stop(self):
        self.hub.close()
        if self.httpd is not None:
            try:
                self.httpd.shutdown()
                self.httpd.server_close()
            except Exception:
                pass
        print("[web] MJPEG 服务已停止", flush=True)

    # ── 编码 ────────────────────────────────────────────────────
    def _encode(self, bgr):
        if self.scale != 1.0:
            bgr = cv2.resize(bgr, None, fx=self.scale, fy=self.scale,
                             interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode('.jpg', bgr,
                               [int(cv2.IMWRITE_JPEG_QUALITY), self.quality])
        return buf.tobytes() if ok else None

    # ── 路由 ────────────────────────────────────────────────────
    def _make_handler(self):
        srv = self

        class _Handler(BaseHTTPRequestHandler):
            server_version = 'FallGuardWeb/1.0'
            protocol_version = 'HTTP/1.0'   # 每请求一连接；MJPEG 是无限长响应

            def log_message(self, *a):      # 静默默认访问日志
                pass

            def do_GET(self):
                path = urlsplit(self.path).path
                if path in ('/', '/index.html'):
                    self._page()
                elif path == '/stream':
                    self._mjpeg('ann')
                elif path == '/raw':
                    self._mjpeg('raw')
                elif path == '/api/status':
                    self._json()
                elif path == '/favicon.ico':
                    self.send_response(204)
                    self.end_headers()
                else:
                    self._err(404, 'not found')

            # ── 各路由实现 ──
            def _page(self):
                w = 940 if srv.scale == 1.0 else int(940 * srv.scale)
                body = _PAGE.format(title=srv.title, w=w).encode('utf-8')
                self.send_response(200)
                self.send_header('Content-Type', 'text/html; charset=utf-8')
                self.send_header('Cache-Control', 'no-store')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _json(self):
                body = json.dumps(srv.hub.stats(), ensure_ascii=False).encode('utf-8')
                self.send_response(200)
                self.send_header('Content-Type', 'application/json; charset=utf-8')
                self.send_header('Cache-Control', 'no-store')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _err(self, code, msg):
                body = msg.encode()
                self.send_response(code)
                self.send_header('Content-Type', 'text/plain; charset=utf-8')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _mjpeg(self, ch_name):
                ch = srv.hub._ch.get(ch_name)
                if ch is None:
                    return self._err(404, f'no channel {ch_name}')
                self.send_response(200)
                self.send_header('Content-Type',
                                 f'multipart/x-mixed-replace; boundary={BOUNDARY}')
                self.send_header('Cache-Control', 'no-cache, private')
                self.send_header('Pragma', 'no-cache')
                self.end_headers()
                ch.add_viewer()
                last = 0
                try:
                    while not srv.hub._stop.is_set():
                        seq, frame, wall = ch.wait_frame(last, timeout=1.0,
                                                         stop_event=srv.hub._stop)
                        if seq is None:
                            continue
                        jpg = srv._encode(frame)
                        if jpg is None:
                            continue
                        head = (f'--{BOUNDARY}\r\n'
                                f'Content-Type: image/jpeg\r\n'
                                f'Content-Length: {len(jpg)}\r\n'
                                f'X-Seq: {seq}\r\n'
                                f'X-Wall: {wall:.3f}\r\n\r\n').encode()
                        self.wfile.write(head)
                        self.wfile.write(jpg)
                        self.wfile.write(b'\r\n')
                        self.wfile.flush()
                        last = seq
                        ch.encode_count += 1
                        ch.bytes_sent += len(jpg)
                except (BrokenPipeError, ConnectionResetError, OSError):
                    pass          # 浏览器关页面/切流 → 正常路径，不是错误
                finally:
                    ch.remove_viewer()

        return _Handler


# ════════════════════════════════════════════════════════════════
# 便捷入口
# ════════════════════════════════════════════════════════════════

def create(port=8888, web_fps=10.0, quality=70, scale=1.0, bind='0.0.0.0',
           channels=('ann', 'raw'), title='FallGuard 实时监控'):
    """建 Hub + 起服务，返回 (hub, server)。port<=0 返回 (None, None)。"""
    if not port or port <= 0:
        return None, None
    hub = FrameHub(channels=channels, web_fps=web_fps)
    server = MjpegServer(hub, port=port, bind=bind, quality=quality,
                         scale=scale, title=title).start()
    return hub, server
