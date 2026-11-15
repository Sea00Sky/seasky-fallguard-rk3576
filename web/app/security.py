"""认证基础设施：密码哈希与会话签名。"""

from __future__ import annotations

import os
import secrets
from functools import lru_cache

from argon2 import PasswordHasher, Type
from argon2.exceptions import Argon2Error
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer

# ── 会话 Cookie 的固定参数（与设计文档 8.2 逐字一致）────────────────
SESSION_COOKIE_NAME = "fg_session"
SESSION_MAX_AGE = 24 * 60 * 60  # 24 小时，单位秒
SESSION_SALT = "fg-session-v1"  # itsdangerous 的命名空间盐，改动会让旧 Cookie 全部失效

# ── 密码策略（全项目唯一口径：≥ 12 位，不许一处写 10 一处写 12）──────
MIN_PASSWORD_LENGTH = 12
MAX_PASSWORD_LENGTH = 256  # 防御性上限，避免超长输入拖慢 argon2

# argon2id 参数：memory_cost 单位是 KiB，64 MB = 65536 KiB
_hasher = PasswordHasher(
    time_cost=3,
    memory_cost=64 * 1024,
    parallelism=4,
    hash_len=32,
    salt_len=16,
    type=Type.ID,
)


def _plain_secret(value: object) -> str:
    """把 pydantic 的 SecretStr / 普通字符串统一成明文。"""
    if value is None:
        return ""
    getter = getattr(value, "get_secret_value", None)
    if callable(getter):
        value = getter()
    return str(value).strip()


@lru_cache(maxsize=1)
def _secret_key() -> str:
    """读取会话签名密钥。"""
    # ① 显式覆盖：方便在测试/多实例部署时单独指定会话密钥
    secret = _plain_secret(os.environ.get("FG_SESSION_SECRET"))
    if secret:
        return secret

    # ② 走 Settings（延迟导入：避免 config -> security -> config 的循环依赖，
    # 也方便本模块在纯脚本环境（没有 Settings）里单独使用）
    try:
        from .config import settings  # noqa: WPS433（局部导入是有意为之）
    except Exception:
        settings = None

    if settings is not None:
        for field_name in (
            "secret_key",
            "session_secret",
            "session_secret_key",
            "app_secret",
            "cookie_secret",
        ):
            secret = _plain_secret(getattr(settings, field_name, None))
            if secret:
                return secret

    # ③ config 不可用时的环境变量兜底
    for env_name in (
        "SECRET_KEY",
        "FG_SECRET_KEY",
        "SESSION_SECRET",
    ):
        secret = _plain_secret(os.environ.get(env_name))
        if secret:
            return secret

    raise RuntimeError(
        "未配置会话签名密钥：请在 web.env 中设置 SECRET_KEY"
        "（或 FG_SESSION_SECRET），或在 app.config.Settings 中提供 "
        "secret_key / session_secret 字段。"
    )


@lru_cache(maxsize=1)
def _serializer() -> URLSafeTimedSerializer:
    """构造签名器（进程内复用；密钥变化需重启，V1 可接受）。"""
    return URLSafeTimedSerializer(_secret_key(), salt=SESSION_SALT)


@lru_cache(maxsize=1)
def _dummy_hash() -> str:
    """生成一次性的"假哈希"，用于账号不存在时的等时校验。"""
    return _hasher.hash(secrets.token_urlsafe(32))


def hash_password(password: str) -> str:
    """把明文密码哈希成 argon2id 字符串（含随机盐，可直接入库）。"""
    if not isinstance(password, str):
        raise TypeError("password 必须是 str")
    if len(password) > MAX_PASSWORD_LENGTH:
        raise ValueError(f"密码长度不能超过 {MAX_PASSWORD_LENGTH} 个字符")
    return _hasher.hash(password)


def verify_password(password: str, password_hash: str | None) -> bool:
    """校验明文密码与 argon2id 哈希是否匹配。"""
    if not isinstance(password, str):
        return False
    if not password_hash:
        password_hash = _dummy_hash()
    try:
        return bool(_hasher.verify(password_hash, password))
    except (Argon2Error, ValueError):
        # argon2-cffi 21.x 的 InvalidHash 与 23.x 的 InvalidHashError
        # 都继承自 ValueError，这样写可同时兼容两个版本。
        return False


ROLES = ("nurse", "head_nurse", "admin")


def sign_session(user_id: int, role: str, ward_id: str | None = None) -> str:
    """签发会话令牌：``{uid, role, ward}`` + 当前时间戳 + HMAC 签名。"""
    try:
        uid = int(user_id)
    except (TypeError, ValueError) as exc:
        raise ValueError("user_id 必须是整数") from exc
    if role not in ROLES:
        raise ValueError(f"非法角色: {role!r}")
    ward = (ward_id or "").strip()
    if role != "admin" and not ward:
        raise ValueError("非管理员账号必须绑定病区")
    payload = {"uid": uid, "role": role, "ward": ward}
    return _serializer().dumps(payload)


def unsign_session(token: str) -> dict | None:
    """校验并解出会话 payload。"""
    if not token or not isinstance(token, str):
        return None
    try:
        data = _serializer().loads(token, max_age=SESSION_MAX_AGE)
    except (BadSignature, SignatureExpired):
        return None

    if not isinstance(data, dict):
        return None

    try:
        uid = int(data.get("uid"))
    except (TypeError, ValueError):
        return None

    role = data.get("role")
    ward = data.get("ward")
    if role not in ROLES:
        return None
    if not isinstance(ward, str):
        return None
    ward = ward.strip()
    # admin 的 ward 允许为空；其他角色必须有病区
    if role != "admin" and not ward:
        return None

    # 只返回白名单字段，避免把签名器里的其它内容带进业务层
    return {"uid": uid, "role": role, "ward": ward}
