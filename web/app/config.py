"""应用配置（pydantic-settings）。"""

import secrets

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """全局配置。字段名即环境变量名（不区分大小写）。"""

    model_config = SettingsConfigDict(
        env_file=".env",          # 可选；不存在时静默跳过
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",           # 环境里有多余变量时不报错
    )

    # 数据库连接串（psycopg3 格式）
    # 生产环境务必通过 DATABASE_URL 覆盖，不要在代码里写真实密码。
    database_url: str = (
        "postgresql://fg_app:CHANGE_ME_BEFORE_DEPLOY@127.0.0.1:5432/fallguard"
    )

    # 签名 Cookie 的密钥（itsdangerous）。
    # 未配置 SECRET_KEY 时生成一次性随机密钥：进程重启后所有会话失效
    # （开发期可接受）。生产必须在 web.env 里设置固定的随机长字符串，
    # 否则重启即全员掉线；也绝不能使用可猜的默认值（设计方案 §10.4）。
    secret_key: str = Field(default_factory=lambda: secrets.token_urlsafe(48))

    # 会话 Cookie 名称与有效期（秒）。设计方案 §8.2：Max-Age=86400。
    session_cookie: str = "fg_session"
    session_max_age: int = 86400

    # 空闲超时（秒）：超过该时长无操作则要求重新登录。
    idle_timeout: int = 1800

    # 快照根目录。必须不在 webroot 下（设计方案 §8.7）：
    # 文件只能经 GET /api/v1/events/{uid}/snapshot 读取并写审计。
    snapshot_dir: str = "/var/lib/fallguard/snapshots"

    # 单病区部署时的默认病区号（seed.sql 里的 4F-A）。
    ward_default: str = "4F-A"

    # 展示与入库使用的时区。数据库一律存 TIMESTAMPTZ（UTC 时刻），
    # 只在展示/按天分桶时用这个时区换算。
    tz: str = "Asia/Shanghai"

    # ── 实时画面（V1.1）──────────────────────────────────────────────────
    # mediamtx 的 HLS 出口。绝不直接暴露给浏览器：mediamtx 有 authInternalUsers
    # 鉴权，但浏览器拿不到也不需要拿到那个凭据。所有 /api/v1/live/* 请求先过会话
    # 校验（见 app/routes/live.py），再由服务端带 Basic Auth 转发给 127.0.0.1。
    mediamtx_url: str = "http://127.0.0.1:8888"
    # mediamtx 的 WebRTC(WHEP) 信令出口。只走信令——媒体本身由浏览器与
    # mediamtx 的 UDP/ICE 端口（8189）直连，反代无法承载（见 live.py 顶部说明）。
    mediamtx_webrtc_url: str = "http://127.0.0.1:8889"
    # mediamtx 的管理 API。只读探针用它判断"这条 path 上有没有发布者"：
    # GET /v3/paths/get/<path> → {"ready": true/false, ...}
    # 比"先建一个 WHEP 会话再看它成不成"干净得多——不占会话、不触发前端回落逻辑、
    # 也不会因为一次探测就让 mediamtx 以为有人在看而开始转码/切片。
    # 只监听回环，不对外；authInternalUsers 里本机 any 有 api 权限，所以不带凭据。
    mediamtx_api_url: str = "http://127.0.0.1:9997"
    mediamtx_user: str = "viewer"
    # 为空 = 关闭实时画面（页面上播放按钮置灰）。部署时由 d2_deploy 从
    # /etc/mediamtx/mediamtx.yml 读出并写进 /etc/fallguard/web.env（640）。
    mediamtx_pass: str = ""

    clip_enabled: bool = True
    # mediamtx 的环形录制目录（`paths.live.recordPath` 的父目录，按 path 名分一层）
    clip_record_dir: str = "/var/lib/mediamtx/record/live"
    # 切片成品目录。同样不在 webroot 下，只能经
    # GET /api/v1/events/{uid}/clip 读取并写审计。
    clip_dir: str = "/var/lib/fallguard/clips"
    clip_lead_s: float = 5.0
    clip_tail_s: float = 5.0
    # 码流从板端到落盘有 0.2 s SRT 缓冲 + 编码/封装延迟。切早了会把「人刚倒」
    # 切到窗口外，所以整段往后挪一点。取 0.5 s 是实测观感值，可按需调。
    clip_lag_s: float = 0.5
    # 事件到达后至少等这么久再切，保证 [t+tail] 那段已经落盘
    clip_settle_s: float = 3.0
    # 成品保留天数（环形录制由 mediamtx 自己按 recordDeleteAfter 清理）
    clip_retention_days: int = 30


# 全局单例：其他模块统一 `from app.config import settings`
settings = Settings()
