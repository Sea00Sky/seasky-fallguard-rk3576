"""数据库访问层（psycopg3 + psycopg_pool）。"""

from __future__ import annotations

import threading
from collections.abc import Iterator, Mapping, Sequence
from typing import Any

import psycopg
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from app.config import settings

# 连接池单例。惰性创建：模块导入时数据库可能还没启动，不能让 import 失败。
_pool: ConnectionPool | None = None
_pool_lock = threading.Lock()


def get_pool() -> ConnectionPool:
    """返回全局连接池；首次调用时创建并打开（min 1 / max 10）。"""
    global _pool
    if _pool is None:
        with _pool_lock:
            if _pool is None:
                _pool = ConnectionPool(
                    conninfo=settings.database_url,
                    min_size=1,
                    max_size=10,
                    # 连接级参数：
                    # row_factory -> 查询结果直接是 dict
                    # options -> 每条连接都固定 search_path，避免依赖调用方
                    kwargs={
                        "row_factory": dict_row,
                        "options": "-c search_path=fg,public",
                    },
                    name="fg_pool",
                    # 借连接最多等 5 秒；数据库挂了要快速失败，不能把请求挂死
                    timeout=5.0,
                    open=False,  # 先构造，再显式 open，避免导入期连库
                )
                _pool.open()
    return _pool


def close_pool() -> None:
    """关闭连接池（应用退出时调用；可重复调用）。"""
    global _pool
    with _pool_lock:
        if _pool is not None:
            _pool.close()
            _pool = None


def get_conn() -> Iterator[psycopg.Connection]:
    """FastAPI 依赖：从连接池借一条连接，请求结束后自动归还。"""
    with get_pool().connection() as conn:
        yield conn


def query(
    sql: str,
    params: Sequence[Any] | Mapping[str, Any] | None = None,
) -> list[dict]:
    """执行查询，返回全部行（list[dict]）。"""
    with get_pool().connection() as conn:
        cur = conn.execute(sql) if params is None else conn.execute(sql, params)
        return list(cur.fetchall())


def query_one(
    sql: str,
    params: Sequence[Any] | Mapping[str, Any] | None = None,
) -> dict | None:
    """执行查询，返回第一行（dict）；没有结果返回 None。"""
    with get_pool().connection() as conn:
        cur = conn.execute(sql) if params is None else conn.execute(sql, params)
        return cur.fetchone()


def execute(
    sql: str,
    params: Sequence[Any] | Mapping[str, Any] | None = None,
) -> int:
    """执行 INSERT / UPDATE / DELETE，返回受影响行数。"""
    with get_pool().connection() as conn:
        cur = conn.execute(sql) if params is None else conn.execute(sql, params)
        return cur.rowcount
