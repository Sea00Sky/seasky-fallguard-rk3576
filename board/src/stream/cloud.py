"""云出口：本地 RTSP → SRT → 云端 mediamtx。"""
from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass

ENV_PREFIX = "FG_CLOUD_"

# mediamtx 的 SRT 发布约定：streamid=publish:<path>
DEFAULT_STREAMID = "publish:live"
DEFAULT_LATENCY_MS = 200
DEFAULT_LOCAL_RTSP = "rtsp://127.0.0.1:8554/live"

_PASS_RE = re.compile(r"^[A-Za-z0-9_.\-]{10,79}$")
# streamid 要原样进 launch 串 ⇒ 不允许引号/反斜杠/空白
_STREAMID_RE = re.compile(r"^[A-Za-z0-9_:.\-]+$")

# ---- 日志脱敏 ----------------------------------------------------------
# 本模块的两处配置都含明文口令，而 systemd 会把 stdout 收进 journald。
# 打日志前一律过一遍下面的函数，否则凭据就长期躺在日志里。
_PASS_PROP_RE = re.compile(r'(passphrase=")[^"]*(")')
_STREAMID_PROP_RE = re.compile(r'(streamid=")([^"]*)(")')
_URL_CRED_RE = re.compile(r"(//)[^/@\s]+@")


def redact_streamid(streamid: str) -> str:
    """把 `publish:<path>:<user>:<口令>` 的最后一段打码。"""
    parts = streamid.split(":")
    if len(parts) >= 4:
        parts[-1] = "***"
    return ":".join(parts)


def redact_launch(launch: str) -> str:
    """打印 launch 串前调用：三处口令全打码。"""
    out = _PASS_PROP_RE.sub(r'\1***\2', launch)
    out = _STREAMID_PROP_RE.sub(
        lambda m: m.group(1) + redact_streamid(m.group(2)) + m.group(3), out)
    return redact_url_credentials(out)


def redact_url_credentials(url: str) -> str:
    """把 `rtsp://user:pass@host` 的 userinfo 换成 `*@`。"""
    return _URL_CRED_RE.sub(r"\1***@", url)


@dataclass
class CloudConfig:
    """云出口参数。"""

    uri: str = ""
    streamid: str = DEFAULT_STREAMID
    passphrase: str = ""
    latency_ms: int = DEFAULT_LATENCY_MS
    local_rtsp: str = DEFAULT_LOCAL_RTSP

    def __post_init__(self) -> None:
        self.uri = str(self.uri).strip()
        self.streamid = str(self.streamid).strip()
        self.passphrase = str(self.passphrase).strip()
        self.local_rtsp = str(self.local_rtsp).strip()
        self.latency_ms = int(self.latency_ms)

        if self.uri and not self.uri.startswith("srt://"):
            raise ValueError(f"uri 必须以 srt:// 开头，收到 {self.uri!r}")
        if self.streamid and not _STREAMID_RE.match(self.streamid):
            raise ValueError(
                f"streamid 只允许 [A-Za-z0-9_:.-]，收到 {self.streamid!r}；"
                "它要原样进 URI query，不做百分号转义")
        if self.passphrase and not _PASS_RE.match(self.passphrase):
            raise ValueError(
                "passphrase 必须是 10–79 个 [A-Za-z0-9_.-]（SRT 规范 + "
                "不能含引号/反斜杠，否则会撑破 launch 串）")
        if self.latency_ms < 0:
            raise ValueError(f"latency_ms 不能为负，收到 {self.latency_ms}")

    @property
    def enabled(self) -> bool:
        """uri 非空才算启用（V1.2 的预留语义）。"""
        return bool(self.uri)

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None,
                 **overrides) -> "CloudConfig":
        """读 `FG_CLOUD_*`，再叠显式 overrides。配错就报错，不静默回退。"""
        src = os.environ if env is None else env
        kw: dict = {}
        spec = {
            "srt_uri": ("uri", str),
            "streamid": ("streamid", str),
            "passphrase": ("passphrase", str),
            "latency_ms": ("latency_ms", int),
            "local_rtsp": ("local_rtsp", str),
        }
        for key, (name, cast) in spec.items():
            raw = src.get(ENV_PREFIX + key.upper())
            if raw is None or raw.strip() == "":
                continue
            try:
                kw[name] = cast(raw.strip())
            except ValueError as exc:
                raise ValueError(
                    f"环境变量 {ENV_PREFIX}{key.upper()}={raw!r} 不是合法 "
                    f"{cast.__name__}") from exc
        kw.update(overrides)
        return cls(**kw)

    def describe(self) -> str:
        """一行生效值。口令只报开关，不报值（streamid / 本地 RTSP 一并脱敏）。"""
        if not self.enabled:
            return "cloud=OFF（FG_CLOUD_SRT_URI 为空）"
        enc = "passphrase=ON" if self.passphrase else "passphrase=OFF(明文)"
        return (f"{self.uri} streamid={redact_streamid(self.streamid)} "
                f"latency={self.latency_ms}ms {enc} ← "
                f"{redact_url_credentials(self.local_rtsp)}")


def local_rtsp_from_stream(stream_cfg) -> str:
    """由推流配置拼出本地拉流地址。"""
    from urllib.parse import quote

    cred = ""
    if getattr(stream_cfg, "auth", False):
        user = quote(str(stream_cfg.user), safe="")
        password = quote(str(stream_cfg.password), safe="")
        cred = f"{user}:{password}@"
    return (f"rtsp://{cred}127.0.0.1:"
            f"{int(stream_cfg.port)}{stream_cfg.mount}")


def build_launch(cfg: CloudConfig) -> str:
    """返回 `Gst.parse_launch` 用的管线描述串。"""
    if not cfg.enabled:
        raise ValueError("uri 为空：云出口未启用，不该构造管线")

    props = [f'uri="{cfg.uri.rstrip("/")}"', "mode=caller",
             f'streamid="{cfg.streamid}"', f"latency={cfg.latency_ms}"]
    if cfg.passphrase:
        props.append(f'passphrase="{cfg.passphrase}"')

    return (
        f'rtspsrc location="{cfg.local_rtsp}" latency=0 protocols=tcp '
        f"! rtph264depay "
        f"! h264parse name=hp "
        f"! video/x-h264,stream-format=byte-stream,alignment=au "
        f"! mpegtsmux "
        f"! srtsink {' '.join(props)}"
    )


def _on_h264_buffer(_pad, info):
    """清掉 `h264parse` 打在关键帧 AU 上的 HEADER 标志。"""
    from gi.repository import Gst

    buf = info.get_buffer()
    if buf is not None:
        buf.unset_flags(Gst.BufferFlags.HEADER)
    return Gst.PadProbeReturn.OK


def _install_header_flag_fix(pipeline) -> None:
    """挂探针，让关键帧不被 `srtsink` 静默吞掉。"""
    from gi.repository import Gst

    hp = pipeline.get_by_name("hp")
    if hp is None:
        raise RuntimeError("launch 串里找不到 name=hp 的 h264parse")
    pad = hp.get_static_pad("src")
    if pad is None:
        raise RuntimeError("h264parse 没有 src pad")
    pad.add_probe(Gst.PadProbeType.BUFFER, _on_h264_buffer)


class CloudSender:
    """SRT 云出口的自愈运行器。"""

    def __init__(self, cfg: CloudConfig, backoff_min: float = 1.0,
                 backoff_max: float = 30.0, healthy_after: float = 20.0,
                 launch: str | None = None):
        self.cfg = cfg
        self.backoff_min = float(backoff_min)
        self.backoff_max = float(backoff_max)
        self.healthy_after = float(healthy_after)
        self.launch = launch if launch is not None else build_launch(cfg)
        self._stop = False

    # ------------------------------------------------------------------ 预检

    def preflight(self) -> None:
        """静态解析 launch 串。"""
        import gi

        gi.require_version("Gst", "1.0")
        from gi.repository import Gst

        if not Gst.is_initialized():
            Gst.init(None)
        try:
            bin_ = Gst.parse_launch(self.launch)
        except Exception as exc:
            raise RuntimeError(
                f"launch 串解析失败：{exc}\n{self.launch}") from exc
        # 顺带验 `name=hp` 还在：探针装不上就说明 launch 串被改坏了
        _install_header_flag_fix(bin_)
        bin_.set_state(Gst.State.NULL)

    # ------------------------------------------------------------------ 运行

    def stop(self) -> None:
        """请求退出（信号处理里调用）。最迟一个轮询周期生效。"""
        self._stop = True

    def run_forever(self) -> int:
        """跑管线，出错就重建。只有 stop 或进程被杀才返回。"""
        import gi

        gi.require_version("Gst", "1.0")
        from gi.repository import Gst

        if not Gst.is_initialized():
            Gst.init(None)

        print(f"[cloud] {self.cfg.describe()}", flush=True)
        print(f"[cloud] launch = {redact_launch(self.launch)}", flush=True)

        attempt = 0
        backoff = self.backoff_min
        while not self._stop:
            attempt += 1
            started = time.monotonic()
            print(f"[cloud] 连接 #{attempt} …", flush=True)

            try:
                pipeline = Gst.parse_launch(self.launch)
                # 必须紧跟 parse_launch：不清 HEADER 标志，关键帧会被 srtsink 吞光
                _install_header_flag_fix(pipeline)
            except Exception as exc:
                print(f"[cloud] 管线构造失败：{exc}", flush=True)
                pipeline = None

            if pipeline is not None:
                bus = pipeline.get_bus()
                pipeline.set_state(Gst.State.PLAYING)
                while not self._stop:
                    msg = bus.timed_pop_filtered(
                        500 * Gst.MSECOND,
                        Gst.MessageType.ERROR | Gst.MessageType.EOS)
                    if msg is None:
                        continue
                    if msg.type == Gst.MessageType.ERROR:
                        err, dbg = msg.parse_error()
                        print(f"[cloud] ERROR: {err.message}"
                              f"{' | ' + dbg if dbg else ''}", flush=True)
                    else:
                        print("[cloud] EOS（对端关闭）", flush=True)
                    break
                pipeline.set_state(Gst.State.NULL)

            if self._stop:
                break

            ran = time.monotonic() - started
            if ran >= self.healthy_after:
                backoff = self.backoff_min       # 稳定过 ⇒ 下次从最短退避起
            print(f"[cloud] 本次存活 {ran:.1f}s，{backoff:.0f}s 后重试", flush=True)
            # 用可被 stop 打断的等待，别用 time.sleep(backoff)
            deadline = time.monotonic() + backoff
            while not self._stop and time.monotonic() < deadline:
                time.sleep(0.1)
            backoff = min(backoff * 2, self.backoff_max)

        print("[cloud] 已停止", flush=True)
        return 0

    def describe(self) -> str:
        return f"{self.cfg.describe()}\nlaunch = {redact_launch(self.launch)}"
