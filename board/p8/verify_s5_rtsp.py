#!/usr/bin/env python3
"""p8/verify_s5_rtsp.py — 验收（RTSP 推流，分支 γ）"""
from __future__ import annotations

import argparse
import os
import re
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import gi  # noqa: E402

gi.require_version("Gst", "1.0")
gi.require_version("GstRtspServer", "1.0")
from gi.repository import GLib, Gst, GstRtspServer  # noqa: E402

Gst.init(None)

BY_PATH_MAIN = "/dev/v4l/by-path/platform-rkisp-vir1-video-index0"
BY_PATH_SELF = "/dev/v4l/by-path/platform-rkisp-vir1-video-index1"
RUN_STREAM = str(ROOT / "scripts" / "run_stream.py")

RESULTS: list[tuple[str, str, str]] = []       # (case, verdict, detail)
PROCS: list[subprocess.Popen] = []


def record(case: str, ok: bool, detail: str) -> None:
    RESULTS.append((case, "PASS" if ok else "FAIL", detail))
    print(f"  → {' PASS' if ok else ' FAIL'}  {detail}", flush=True)


def report(case: str, detail: str) -> None:
    RESULTS.append((case, "REPORT", detail))
    print(f"  →  {detail}", flush=True)


# ------------------------------------------------------------------ 前置检查


def preflight() -> None:
    if os.geteuid() == 0:
        raise SystemExit(
            "不要用 sudo 跑：sudo 把 HOME 换成 /root，会丢掉 ~/.local 里的 numpy；"
            "而且会在 /tmp 留下 root 所有的测试文件。")
    for dev, what in ((BY_PATH_MAIN, "mainpath（推流）"),
                      (BY_PATH_SELF, "selfpath（检测）")):
        if not Path(dev).exists():
            raise SystemExit(
                f"找不到 {dev}（{what}）。"
                f"别用 /dev/video-camera0 —— 它指向不能出帧的 video51")
    for port in PORTS.values():
        if port_in_use(port):
            raise SystemExit(f"端口 {port} 已被占用，先停掉旧进程")
    check_launch_strings()


def check_launch_strings() -> None:
    """先 parse 一遍两条 launch 串。"""
    from src.stream import StreamConfig, build_launch

    for source in ("v4l2", "test"):
        desc = build_launch(StreamConfig(source=source))
        try:
            bin_ = Gst.parse_launch(desc)
        except Exception as exc:
            raise SystemExit(f"launch 串解析失败（source={source}）：{exc}\n{desc}")
        bin_.set_state(Gst.State.NULL)
        print(f"  [launch] source={source} 解析 OK", flush=True)


PORTS = {"V0": 8560, "V1": 8561, "V2": 8561, "V3": 8562, "V4": 8563,
         "V5": 8564, "V6": 8565}
SECONDS = 8.0


def port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.3)
        return s.connect_ex(("127.0.0.1", port)) == 0


def wait_port(port: int, timeout: float = 15.0) -> bool:
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if port_in_use(port):
            return True
        time.sleep(0.1)
    return False


# ------------------------------------------------------------------ 推流进程


def wait_port_free(port: int, timeout: float = 5.0) -> bool:
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if not port_in_use(port):
            return True
        time.sleep(0.05)
    return False


def start_streamer(port: int, res: str, extra: list[str] | None = None
                   ) -> subprocess.Popen:
    if not wait_port_free(port):
        raise SystemExit(f"端口 {port} 还没释放，上一轮的推流进程没退干净")
    log_path = f"/tmp/fg_s5_{port}.log"
    log = open(log_path, "w")
    cmd = [sys.executable, RUN_STREAM, "--res", res, "--port", str(port)]
    cmd += (extra or [])
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
    PROCS.append(proc)
    if not wait_port(port):
        print(read_log(log_path), flush=True)
        raise SystemExit(f"推流端没起来（端口 {port} 未监听），日志见 {log_path}")
    return proc


def stop_streamer(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


def read_log(path: str) -> str:
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace")[-3000:]
    except OSError:
        return "(无日志)"


# ------------------------------------------------------------------ 拉流客户端


class Pull:
    """一个 RTSP 客户端：解码计数 + 码率 + 分辨率 + 首帧时间。"""

    def __init__(self, url: str, tag: str = "c"):
        self.tag = tag
        self.frames = 0
        self.nbytes = 0
        self.first_t: float | None = None
        self.last_t: float | None = None
        self.t_start: float | None = None
        self.width = self.height = None
        self.error: str | None = None
        self.pipe = Gst.parse_launch(
            f"rtspsrc location={url} latency=0 protocols=tcp "
            f"! rtph264depay name=depay ! h264parse ! avdec_h264 "
            f"! appsink name=sink emit-signals=true max-buffers=5 sync=false")
        bus = self.pipe.get_bus()
        bus.add_signal_watch()
        bus.connect("message::error", self._on_error)
        self.pipe.get_by_name("sink").connect("new-sample", self._on_sample)
        self.pipe.get_by_name("depay").get_static_pad("src").add_probe(
            Gst.PadProbeType.BUFFER, self._on_buffer)

    # ---- 回调（异常必须吞掉：PyGObject 会把信号回调里的异常直接丢弃）----
    def _on_error(self, _bus, msg):
        try:
            err, _dbg = msg.parse_error()
            self.error = err.message
        except Exception:
            self.error = "unknown"

    def _on_buffer(self, _pad, info):
        try:
            self.nbytes += info.get_buffer().get_size()
        except Exception:
            pass
        return Gst.PadProbeReturn.OK

    def _on_sample(self, sink):
        try:
            smp = sink.emit("pull-sample")
            now = time.monotonic()
            if self.first_t is None:
                self.first_t = now
                caps = smp.get_caps()
                if caps and caps.get_size() > 0:
                    st = caps.get_structure(0)
                    self.width = st.get_value("width")
                    self.height = st.get_value("height")
            self.last_t = now
            self.frames += 1
        except Exception:
            pass
        return Gst.FlowReturn.OK

    # ---- 控制 ----
    def start(self):
        self.t_start = time.monotonic()
        self.pipe.set_state(Gst.State.PLAYING)

    def stop(self):
        self.pipe.set_state(Gst.State.NULL)

    def stats(self) -> dict:
        now = time.monotonic()
        wall = (now - self.t_start) if self.t_start else 0.0
        steady = ((self.last_t - self.first_t)
                  if (self.first_t and self.last_t and self.frames > 1) else 0.0)
        return {
            "tag": self.tag,
            "frames": self.frames,
            "fps": (self.frames - 1) / steady if steady > 0 else 0.0,
            "fps_wall": self.frames / wall if wall > 0 else 0.0,
            "mbps": self.nbytes * 8 / wall / 1e6 if wall > 0 else 0.0,
            "first_s": (self.first_t - self.t_start) if self.first_t else None,
            "w": self.width, "h": self.height,
            "error": self.error,
        }


def pull_once(url: str, seconds: float, tag: str = "c") -> dict:
    c = Pull(url, tag)
    c.start()
    time.sleep(seconds)
    st = c.stats()
    c.stop()
    return st


def fmt(st: dict) -> str:
    return (f"{st['tag']}: {st['frames']} 帧 {st['fps']:.1f} fps "
            f"{st['mbps']:.2f} Mbps {st['w']}x{st['h']}"
            + (f" 首帧 {st['first_s'] * 1000:.0f} ms" if st['first_s'] else "")
            + (f" ERR={st['error']}" if st["error"] else ""))


def url_for(port: int, mount: str = "/live", auth: bool = True) -> str:
    cred = f"{os.environ.get('FG_STREAM_USER','')}:{os.environ.get('FG_STREAM_PASSWORD','')}@" if auth else ""
    return f"rtsp://{cred}127.0.0.1:{port}{mount}"


# ------------------------------------------------------------------ V0 鉴权


def case_v0(seconds: float) -> None:
    print("\n=== V0 鉴权三态 ===", flush=True)
    port = PORTS["V0"]
    proc = start_streamer(port, "sd", ["--source", "test"])
    try:
        good = pull_once(url_for(port), seconds, "正确凭据")
        print("   " + fmt(good), flush=True)

        noauth = pull_once(f"rtsp://127.0.0.1:{port}/live", seconds, "无凭据")
        print("   " + fmt(noauth), flush=True)

        wrong = pull_once(f"rtsp://admin:wrongpass@127.0.0.1:{port}/live",
                          seconds, "错凭据")
        print("   " + fmt(wrong), flush=True)

        record("V0", good["frames"] > 0,
               f"正确凭据 {good['frames']} 帧 / {good['fps']:.1f} fps")
        record("V0", noauth["frames"] == 0,
               f"无凭据 {noauth['frames']} 帧（必须 0）"
               + (f" [{noauth['error']}]" if noauth["error"] else ""))
        record("V0", wrong["frames"] == 0,
               f"错凭据 {wrong['frames']} 帧（必须 0）"
               + (f" [{wrong['error']}]" if wrong["error"] else ""))
    finally:
        stop_streamer(proc)


# ------------------------------------------------------------------ V1 + V2


def case_v1_v2(seconds: float) -> None:
    print("\n=== V1 单客户端 1080p / V2 三客户端并发 ===", flush=True)
    port = PORTS["V1"]
    proc = start_streamer(port, "hd")
    try:
        url = url_for(port)
        st = pull_once(url, seconds, "单客户端")
        print("   " + fmt(st), flush=True)
        ok_res = (st["w"], st["h"]) == (1920, 1080)
        record("V1", ok_res, f"分辨率 {st['w']}x{st['h']}（应为 1920x1080）")
        record("V1", st["fps"] >= 25.0, f"fps {st['fps']:.2f} ≥ 25")
        ok_first = st["first_s"] is not None and st["first_s"] < 3.0
        record("V1", ok_first,
               f"首帧 {st['first_s'] * 1000:.0f} ms < 3000" if st["first_s"]
               else "首帧：没收到帧")

        # ---- V2：三个客户端同时拉同一条 shared media
        clients = [Pull(url, f"c{i + 1}") for i in range(3)]
        for c in clients:
            c.start()
        time.sleep(seconds)
        stats = [c.stats() for c in clients]
        for c in clients:
            c.stop()
        for s in stats:
            print("   " + fmt(s), flush=True)
        record("V2", all(s["frames"] > 0 for s in stats),
               f"3/3 客户端都收到帧（{', '.join(str(s['frames']) for s in stats)}）")
        worst = min(s["fps"] for s in stats)
        record("V2", worst >= 20.0, f"最差客户端 {worst:.2f} fps ≥ 20")
        total = sum(s["mbps"] for s in stats)
        report("V2", f"三客户端总下行 {total:.2f} Mbps（单编码器，"
                     f"每客户端 {total / 3:.2f} Mbps）")
    finally:
        stop_streamer(proc)


# ------------------------------------------------------------------ V3 延迟


def case_v3(seconds: float, width: int = 1920, height: int = 1080,
            fps: int = 30) -> None:
    """appsrc 注入序号 → 编码 → RTP → 解码 → 读回序号。"""
    print(f"\n=== V3 端到端延迟（appsrc 序号法，{width}x{height}）===", flush=True)
    from src.stream import RtspServer, StreamConfig

    port = PORTS["V3"]
    block = 32
    b0, c0 = 8, 8

    launch = (
        f"( appsrc name=src is-live=true format=time do-timestamp=false "
        f"caps=video/x-raw,format=NV12,width={width},height={height},"
        f"framerate={fps}/1 "
        f"! mpph264enc rc-mode=cbr bps=4000000 gop={fps} header-mode=each-idr "
        f"! h264parse config-interval=-1 ! rtph264pay name=pay0 pt=96 )")
    cfg = StreamConfig(port=port, mount="/live", width=width, height=height,
                       fps=fps, auth=True, source="test")
    srv = RtspServer(cfg, launch=launch)
    srv.start()
    holder: dict = {}
    srv.factory.connect(
        "media-configure",
        lambda _f, media: holder.setdefault(
            "appsrc", media.get_element().get_by_name("src")))

    seq_tab: dict[int, list] = {}
    lock = threading.Lock()
    lat: list[float] = []
    nxt = [0]
    pipe = None
    try:
        pipe = Gst.parse_launch(
            f"rtspsrc location={url_for(port)} latency=0 protocols=tcp "
            f"! rtph264depay ! h264parse ! avdec_h264 "
            # 不要加 `video/x-raw,format=NV12`：板上 avdec_h264 只出 I420，
            # capsfilter 会 not-negotiated 直接把整条客户端管线打死（0 帧）。
            # I420/NV12 的 Y 平面都在最前面，按 stride 读就行，不需要转换。
            f"! appsink name=sink emit-signals=true max-buffers=5 sync=false")
        sink = pipe.get_by_name("sink")
        stride_box = [width, False]   # [stride, 是否已按 caps 修正]
        dbg = {"n": 0, "vals": [], "caps": None, "err": None}
        pipe.get_bus().add_signal_watch()
        pipe.get_bus().connect(
            "message::error",
            lambda _b, m: dbg.__setitem__("err", m.parse_error()[0].message))

        def on_sample(s):
            try:
                smp = s.emit("pull-sample")
                dbg["n"] += 1
                if not stride_box[1]:
                    stride_box[1] = True
                    try:
                        from gi.repository import GstVideo
                        vi = GstVideo.VideoInfo.new_from_caps(smp.get_caps())
                        stride_box[0] = vi.stride[0]
                        dbg["caps"] = (f"{vi.fourcc} {vi.width}x{vi.height} "
                                       f"stride={vi.stride[0]}")
                    except Exception as exc:
                        dbg["caps"] = f"caps 解析失败: {exc}"
                buf = smp.get_buffer()
                ok, info = buf.map(Gst.MapFlags.READ)
                if not ok:
                    return Gst.FlowReturn.OK
                try:
                    arr = np.frombuffer(info.data, dtype=np.uint8)
                    # avdec 的 NV12 stride 可能 > width；首帧按 caps 取一次
                    stride = stride_box[0]
                    y = arr[:stride * height].reshape(height, stride)[:, :width]
                    v = int(np.median(y[b0:b0 + block, c0:c0 + block]))
                    if len(dbg["vals"]) < 12:
                        dbg["vals"].append(v)
                    now = time.monotonic()
                    with lock:
                        rec = seq_tab.get(v)
                        if rec and rec[1] is None and now - rec[0] < 2.0:
                            rec[1] = now
                            lat.append((now - rec[0]) * 1000.0)
                finally:
                    buf.unmap(info)
            except Exception:
                pass
            return Gst.FlowReturn.OK

        sink.connect("new-sample", on_sample)
        pipe.set_state(Gst.State.PLAYING)

        t0 = time.monotonic()
        while "appsrc" not in holder and time.monotonic() - t0 < 10:
            time.sleep(0.05)
        if "appsrc" not in holder:
            record("V3", False, "media 未构建（客户端没连上）")
            return

        feeder = holder["appsrc"]
        # 只分配一次：序号块位置固定，每帧原地覆盖即可（1080p 少一次 3 MB 拷贝）
        frame = np.full((height * 3 // 2, width), 60, dtype=np.uint8)
        frame[height:, :] = 128                    # UV 平面：中性灰
        nbytes = frame.nbytes

        interval = 1.0 / fps
        n_pushed = 0
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            with lock:
                seq = nxt[0]
                nxt[0] += 1
                seq_tab[seq & 0xFF] = [time.monotonic(), None]
            frame[b0:b0 + block, c0:c0 + block] = seq & 0xFF
            buf = Gst.Buffer.new_allocate(None, nbytes, None)
            buf.fill(0, frame.tobytes())
            feeder.emit("push-buffer", buf)
            n_pushed += 1
            time.sleep(interval)
        push_fps = n_pushed / seconds
        print(f"   推入 {n_pushed} 帧 = {push_fps:.1f} fps", flush=True)
    finally:
        if pipe is not None:
            pipe.set_state(Gst.State.NULL)
        srv.stop()

    if len(lat) < 20:
        print(f"   [诊断] 收到 {dbg['n']} 帧，前几个读数 {dbg['vals']}", flush=True)
        print(f"   [诊断] caps = {dbg['caps']}", flush=True)
        print(f"   [诊断] bus  = {dbg['err']}", flush=True)
        record("V3", False, f"只测到 {len(lat)} 个样本（<20，收到 {dbg['n']} 帧）")
        return
    a = np.array(lat)
    p50, p95 = float(np.median(a)), float(np.percentile(a, 95))
    record("V3", p50 < 250.0,
           f"p50 {p50:.1f} ms < 250（p95 {p95:.1f} max {a.max():.1f}，"
           f"n={len(a)}）")
    report("V3", "口径：appsrc→编码→RTP→解码→客户端，**不含相机曝光**")


# ------------------------------------------------------------------ V4 标清


def case_v4(seconds: float) -> None:
    print("\n=== V4 标清 640×480（改配置，不缩放）===", flush=True)
    port = PORTS["V4"]
    proc = start_streamer(port, "sd")
    try:
        st = pull_once(url_for(port), seconds, "sd")
        print("   " + fmt(st), flush=True)
        record("V4", (st["w"], st["h"]) == (640, 480),
               f"分辨率 {st['w']}x{st['h']}（应为 640x480）")
        record("V4", st["fps"] >= 25.0, f"fps {st['fps']:.2f} ≥ 25")
        report("V4", f"码率 {st['mbps']:.2f} Mbps（bps 目标 4.0）")
    finally:
        stop_streamer(proc)


# ------------------------------------------------------------------ V7 检测循环


def run_cam_fps(seconds: float) -> tuple[float, str]:
    """跑一次 `scripts/run_cam.py`，从汇总里抠出采集 fps。"""
    cmd = [sys.executable, str(ROOT / "scripts" / "run_cam.py"),
           "--seconds", str(seconds), "--quiet"]
    p = subprocess.run(cmd, capture_output=True, text=True,
                       timeout=seconds + 180)
    for line in (p.stdout or "").splitlines():
        if line.startswith("采集"):
            m = re.search(r"([\d.]+)\s*fps", line)
            if m:
                return float(m.group(1)), line.strip()
    tail = ((p.stdout or "")[-400:] + (p.stderr or "")[-400:]).strip()
    return 0.0, tail


def case_v7(seconds: float) -> None:
    """完整检测循环（含推理）在推流开/关下的 fps。"""
    print("\n=== V7 检测循环 fps（只报不判）===", flush=True)
    off, line_off = run_cam_fps(seconds)
    print(f"   无推流    : {off:.2f} fps   [{line_off}]", flush=True)

    port = PORTS["V6"]
    proc = start_streamer(port, "hd")
    client = Pull(url_for(port), "拉流-hd")
    client.start()
    time.sleep(1.5)
    on, line_on = run_cam_fps(seconds)
    st = client.stats()
    client.stop()
    stop_streamer(proc)
    print(f"   1080p 推流: {on:.2f} fps   [{line_on}]", flush=True)
    print(f"   （推流侧 {st['fps']:.1f} fps {st['mbps']:.2f} Mbps）", flush=True)
    if off > 0 and on > 0:
        report("V7", f"检测循环 无推流 {off:.2f} → 1080p 推流 {on:.2f} fps "
                     f"（Δ {on - off:+.2f}，NPU 未锁频，只报不判）")
    else:
        report("V7", "检测循环没跑起来（模型/相机问题），本项无数据")


# ------------------------------------------------------------------ V6 逃生口


def case_v6(seconds: float) -> None:
    """`--no-auth` 必须真能用 —— 逃生口坏了比没有逃生口更糟。"""
    print("\n=== V6 --no-auth 逃生口 ===", flush=True)
    port = PORTS["V6"]
    proc = start_streamer(port, "sd", ["--source", "test", "--no-auth"])
    try:
        st = pull_once(f"rtsp://127.0.0.1:{port}/live", seconds, "无凭据")
        print("   " + fmt(st), flush=True)
        record("V6", st["frames"] > 0,
               f"--no-auth 下无凭据能拉流（{st['frames']} 帧）")
        if st["frames"] == 0:
            print(read_log(f"/tmp/fg_s5_{port}.log"), flush=True)
    finally:
        stop_streamer(proc)


# ------------------------------------------------------------------ V5 γ 争用


def capture_fps(seconds: float) -> float:
    """只采集、不推理 —— 干净地量 selfpath 的取帧能力。"""
    from src.capture import CaptureConfig, open_source

    src = open_source(CaptureConfig(backend="gst", width=640, height=480,
                                    fps=30))
    src.start()
    n = 0
    t0 = time.monotonic()
    try:
        while time.monotonic() - t0 < seconds:
            if src.read(timeout_s=1.0) is not None:
                n += 1
    finally:
        src.stop()
    return n / (time.monotonic() - t0)


def case_v5(seconds: float) -> None:
    print("\n=== V5 γ 争用：推流开着时检测侧还能取多少帧 ===", flush=True)
    base = capture_fps(seconds)
    print(f"   无推流            : {base:.2f} fps", flush=True)

    port = PORTS["V5"]
    measured = {}
    for res in ("hd", "sd"):
        proc = start_streamer(port, res)
        client = Pull(url_for(port), f"拉流-{res}")
        client.start()
        time.sleep(1.5)                       # 等编码器真跑起来
        fps = capture_fps(seconds)
        st = client.stats()
        client.stop()
        stop_streamer(proc)
        measured[res] = fps
        print(f"   推流 {res:2s} + 客户端拉流: {fps:.2f} fps   "
              f"（客户端 {st['fps']:.1f} fps {st['mbps']:.2f} Mbps）",
              flush=True)
        time.sleep(0.5)

    for res, fps in measured.items():
        record("V5", fps >= 25.0, f"推流 {res} 时采集 {fps:.2f} fps ≥ 25")
    report("V5", f"Δ(1080p) {base - measured['hd']:+.2f} fps，"
                 f"Δ(640) {base - measured['sd']:+.2f} fps（只报不判）")


# ------------------------------------------------------------------ main


def main() -> int:
    global SECONDS
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="V0,V1,V2,V3,V4,V5,V6")
    ap.add_argument("--seconds", type=float, default=SECONDS,
                    help="每个用例的稳态测量时长")
    ap.add_argument("--with-detect", action="store_true",
                    help="额外跑 V7（完整检测循环，多花 ~1 分钟，只报不判）")
    args = ap.parse_args()
    SECONDS = args.seconds
    wanted = {c.strip().upper() for c in args.cases.split(",") if c.strip()}

    preflight()
    print(f"[rtsp] 端口 {PORTS}", flush=True)
    print(f"[rtsp] 每个用例稳态 {SECONDS:.0f} s", flush=True)

    loop = GLib.MainLoop()
    threading.Thread(target=loop.run, daemon=True).start()

    try:
        if "V0" in wanted:
            case_v0(SECONDS)
        if "V1" in wanted or "V2" in wanted:
            case_v1_v2(SECONDS)
        if "V3" in wanted:
            case_v3(SECONDS)
        if "V4" in wanted:
            case_v4(SECONDS)
        if "V5" in wanted:
            case_v5(SECONDS)
        if "V6" in wanted:
            case_v6(SECONDS)
        if args.with_detect or "V7" in wanted:
            case_v7(SECONDS)
    finally:
        for p in PROCS:
            stop_streamer(p)
        loop.quit()

    print("\n================ 验收汇总 ================")
    for name, verdict, detail in RESULTS:
        print(f"  {name:3s} {verdict:6s} {detail}")
    hard = [r for r in RESULTS if r[1] != "REPORT"]
    bad = [r for r in hard if r[1] != "PASS"]
    print(f"\n硬门 {len(hard) - len(bad)}/{len(hard)} PASS"
          + ("" if not bad else f"    {len(bad)} 项失败"))
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
