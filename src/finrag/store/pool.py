"""进程级数据库连接池：把"借连接 / 还连接"集中到一处。

**为什么需要它**：FastAPI 的同步路由跑在线程池里，而 psycopg 的 ``Connection`` **非线程安全**
⇒ 不能跨请求共享同一连接，只能"一借一还"。每请求新建虽然省心，但每次都要走一遍 TCP + 认证；
高频探针与并发一上来就不划算。更关键的是**流式接口**：若把连接 Hold 到整个 SSE 生成过程结束，
"正在生成"的请求会把连接池占满（见 ``api/server._run_stream`` 的两阶段改造）。

本模块保证两件事：

1. **一借一还**：``pooled_connection()`` 退出即归还（异常路径也还）。归还前会把未提交事务
   **rollback**，故 ``SET LOCAL hnsw.ef_search`` 之类的会话设置不会漏给下一个借用者。
2. **与 :func:`finrag.store.writer.connect` 同口径**：建连参数（``CONNECT_KWARGS``）与 pgvector
   注册（``configure_connection``）都复用 ``store.writer`` 的定义，**不在这里重写一份**
   （重写等于多出两处需要同步的"不可省"配置）。

优先用 ``psycopg_pool.ConnectionPool``（成熟的健康检查与生命周期管理）；**未安装时回落到本模块
自带的极简池**。两者对外都只暴露 ``.connection()`` 一个上下文管理器，调用方无须知道用的是哪个。
"""

from __future__ import annotations

import logging
import threading
import time
from contextlib import contextmanager
from typing import Any, Iterator

from finrag.config import AppConfig, get_config
from finrag.store.writer import CONNECT_KWARGS, configure_connection, connect

__all__ = [
    "close_pool",
    "get_pool",
    "pooled_connection",
    "pool_backend",
]

logger = logging.getLogger(__name__)

# 建池锁：get_pool 是进程级单例，并发首调会各建一个池
_LOCK = threading.Lock()
_POOL: Any = None
_BACKEND: str = ""


def get_pool(cfg: AppConfig | None = None) -> Any:
    """进程内单例池；线程安全，**首次调用才建池**（不在导入期连库）。"""
    global _POOL, _BACKEND
    with _LOCK:
        if _POOL is None:
            cfg = cfg or get_config()
            _POOL, _BACKEND = _build_pool(cfg)
            logger.info(
                "连接池就绪：backend=%s min=%d max=%d timeout=%.0fs",
                _BACKEND, cfg.pool.min_size, cfg.pool.max_size, cfg.pool.timeout,
            )
        return _POOL


def pool_backend() -> str:
    """当前生效的后端名（``psycopg_pool`` / ``builtin``）；未建池时为空串。供探针与报告。"""
    return _BACKEND


def close_pool() -> None:
    """关闭并清空池（进程退出 / 测试用）。幂等。"""
    global _POOL, _BACKEND
    with _LOCK:
        pool, _POOL, _BACKEND = _POOL, None, ""
    if pool is not None:
        try:
            pool.close()
        except Exception:  # noqa: BLE001
            logger.debug("关闭连接池时异常", exc_info=True)


@contextmanager
def pooled_connection(cfg: AppConfig | None = None) -> Iterator[Any]:
    """借一条连接；退出时自动归还（含异常路径）。"""
    with get_pool(cfg).connection() as conn:
        yield conn


def _build_pool(cfg: AppConfig) -> tuple[Any, str]:
    """建池：优先 ``psycopg_pool``，缺席时回落内置极简池。返回 ``(池对象, 后端名)``。"""
    pcfg = cfg.pool
    try:
        from psycopg_pool import ConnectionPool
    except ImportError:
        logger.warning(
            "未安装 psycopg_pool，回落到内置极简连接池"
            "（建议：python -m pip install --user psycopg-pool）"
        )
        return _SimplePool(cfg), "builtin"

    pool = ConnectionPool(
        conninfo=cfg.pg_dsn.get_secret_value(),
        min_size=pcfg.min_size,
        max_size=pcfg.max_size,
        timeout=pcfg.timeout,
        max_idle=pcfg.max_idle,
        # 建连参数与 pgvector 注册都复用 writer 的定义，不重写。
        kwargs=dict(CONNECT_KWARGS),
        configure=configure_connection,
        name="finrag",
        open=True,
    )
    return pool, "psycopg_pool"


class _SimplePool:
    """极简连接池 —— ``psycopg_pool`` 缺席时的回落实现。

    只做"借出 / 归还 + 上限 + 等待超时"，不做后台回收与健康检查（够单机 API 用）。
    归还前 **rollback** 未提交事务，保证会话级设置（如 ``SET LOCAL``）不残留；
    连接已关闭 / rollback 失败时直接丢弃并释放名额。
    """

    def __init__(self, cfg: AppConfig) -> None:
        self._cfg = cfg
        self._max = max(1, cfg.pool.max_size)
        self._timeout = cfg.pool.timeout
        self._idle: list[Any] = []
        self._n = 0
        self._cond = threading.Condition(threading.Lock())
        for _ in range(max(0, min(cfg.pool.min_size, self._max))):
            self._idle.append(connect(cfg))
            self._n += 1

    def _acquire(self) -> Any:
        deadline = time.monotonic() + self._timeout
        while True:
            create = False
            with self._cond:
                while self._idle:
                    conn = self._idle.pop()
                    if not getattr(conn, "closed", False):
                        return conn
                    self._n -= 1  # 丢弃坏连接，释放名额
                if self._n < self._max:
                    # 先占名额，再在锁外建连 —— 建连可能阻塞（网络），不该占着条件变量。
                    self._n += 1
                    create = True
                else:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError(
                            f"连接池已满且等待超时（max_size={self._max}）"
                        )
                    self._cond.wait(remaining)
            if create:
                try:
                    return connect(self._cfg)
                except Exception:
                    with self._cond:
                        self._n -= 1
                        self._cond.notify()
                    raise

    def _release(self, conn: Any) -> None:
        with self._cond:
            if getattr(conn, "closed", False):
                self._n -= 1
            else:
                try:
                    conn.rollback()
                except Exception:  # noqa: BLE001
                    self._n -= 1
                    try:
                        conn.close()
                    except Exception:  # noqa: BLE001
                        pass
                else:
                    self._idle.append(conn)
            self._cond.notify()

    @contextmanager
    def connection(self) -> Iterator[Any]:
        conn = self._acquire()
        try:
            yield conn
        finally:
            self._release(conn)

    def close(self) -> None:
        with self._cond:
            idle, self._idle = self._idle, []
            self._n -= len(idle)
        for conn in idle:
            try:
                conn.close()
            except Exception:  # noqa: BLE001
                pass
