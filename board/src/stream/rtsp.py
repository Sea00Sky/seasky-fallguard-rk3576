"""RTSP 推流服务端。"""
from __future__ import annotations

import threading

from .config import StreamConfig


def build_launch(cfg: StreamConfig) -> str:
    """返回 `RTSPMediaFactory.set_launch` 用的管线描述串。"""
    caps = (f"video/x-raw,format=NV12,width={cfg.width},"
            f"height={cfg.height},framerate={cfg.fps}/1")

    if cfg.source == "v4l2":
        src = (f"v4l2src device={cfg.device} io-mode={cfg.io_mode} "
               f"! {caps} ")
    elif cfg.source == "test":
        # 只有测试源需要 videoconvert：videotestsrc 出的是 RGB
        src = (f"videotestsrc is-live=true pattern=ball "
               f"! videoconvert ! {caps} ")
    else:                                   # __post_init__ 已挡过
        raise ValueError(f"未知 source: {cfg.source!r}")

    enc = (f"! mpph264enc rc-mode=cbr bps={cfg.bps} gop={cfg.gop} "
           f"! h264parse config-interval=-1 "
           f"! rtph264pay name=pay0 pt=96")
    return f"( {src}{enc} )"


def _make_token(role: str):
    """造一个带 role + 授权位的 RTSPToken（见模块 docstring）。"""
    from gi.repository import GstRtspServer as R

    token = R.RTSPToken()
    token.set_string("media.factory.role", role)
    # 播放只需要这四项；transport.client-settings（SET_PARAMETER）不给，最小权限
    for check in (R.RTSP_AUTH_CHECK_CONNECT,
                  R.RTSP_AUTH_CHECK_URL,
                  R.RTSP_AUTH_CHECK_MEDIA_FACTORY_ACCESS,
                  R.RTSP_AUTH_CHECK_MEDIA_FACTORY_CONSTRUCT):
        token.set_bool(check, True)
    return token


def _make_permissions(role: str):
    """媒体工厂的权限表（设了 auth 之后必须显式给，否则一律拒）。"""
    from gi.repository import GstRtspServer as R

    perms = R.RTSPPermissions()
    perms.add_permission_for_role(role, "media.factory.access", True)
    perms.add_permission_for_role(role, "media.factory.construct", True)
    return perms


class RtspServer:
    """gst-rtsp-server 的薄封装。"""

    def __init__(self, cfg: StreamConfig, launch: str | None = None):
        self.cfg = cfg
        self.launch = launch if launch is not None else build_launch(cfg)
        self._server = None
        self._factory = None
        self._loop = None
        self._source_id = None
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------------ 构建

    @property
    def server(self):
        """`RTSPServer`（`start` 之后可用；connect `client-connected` 用）。"""
        return self._server

    @property
    def factory(self):
        """`RTSPMediaFactory`（测试里 connect `media-configure` 用）。"""
        return self._factory

    @property
    def url(self) -> str:
        """本机可用的拉流 URL（带凭据，方便直接喂给 gst-launch）。"""
        c = self.cfg
        cred = f"{c.user}:{c.password}@" if c.auth else ""
        return f"rtsp://{cred}127.0.0.1:{c.port}{c.mount}"

    def _build(self) -> None:
        import gi

        gi.require_version("Gst", "1.0")
        gi.require_version("GstRtspServer", "1.0")
        from gi.repository import Gst, GstRtspServer

        if not Gst.is_initialized():
            Gst.init(None)

        c = self.cfg
        server = GstRtspServer.RTSPServer()
        server.set_service(str(c.port))
        server.set_address("0.0.0.0")

        factory = GstRtspServer.RTSPMediaFactory()
        factory.set_launch(self.launch)
        factory.set_shared(True)        # N 个客户端共用一条管线，只编码一次
        factory.set_latency(0)

        if c.auth:
            auth = GstRtspServer.RTSPAuth()
            # key 必须是 base64("user:pass")，不是明文（见模块 docstring ③）
            auth.add_basic(GstRtspServer.RTSPAuth.make_basic(c.user, c.password),
                           _make_token("user"))
            server.set_auth(auth)
            factory.set_permissions(_make_permissions("user"))

        server.get_mount_points().add_factory(c.mount, factory)

        self._server = server
        self._factory = factory
        self._loop = None

    # ------------------------------------------------------------------ 运行

    def start(self) -> None:
        """attach 并在后台线程跑 GLib 主循环（测试/嵌入用）。"""
        import gi

        gi.require_version("GLib", "2.0")
        from gi.repository import GLib

        if self._server is not None:
            raise RuntimeError("RtspServer 已经启动过了")
        self._build()
        # attach 之后才真正 bind 端口；失败会抛（端口占用等）
        self._source_id = self._server.attach(None)
        self._loop = GLib.MainLoop()
        self._thread = threading.Thread(target=self._loop.run, daemon=True)
        self._thread.start()

    def run_forever(self) -> None:
        """attach 并在当前线程跑主循环（生产入口用，便于收信号）。"""
        import gi

        gi.require_version("GLib", "2.0")
        from gi.repository import GLib

        if self._server is not None:
            raise RuntimeError("RtspServer 已经启动过了")
        self._build()
        self._source_id = self._server.attach(None)
        self._loop = GLib.MainLoop()
        self._loop.run()

    def stop(self) -> None:
        """停止监听并退出主循环。可重复调用。"""
        if self._source_id is not None:
            import gi

            gi.require_version("GLib", "2.0")
            from gi.repository import GLib

            GLib.source_remove(self._source_id)   # 真正解绑端口
            self._source_id = None
        if self._loop is not None:
            self._loop.quit()
            self._loop = None
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        self._server = None
        self._factory = None

    # ------------------------------------------------------------------ 信息

    def describe(self) -> str:
        return (f"{self.cfg.describe()}\n"
                f"launch = {self.launch}")
