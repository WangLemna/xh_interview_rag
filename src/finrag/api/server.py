"""HTTP 层：把 :mod:`finrag.service` 的事件流转成 SSE。业务判断都在 ``service``。

连接走**进程级连接池**（``store/pool.py``）：借出 → 用完即还。psycopg 的 ``Connection``
非线程安全，而 FastAPI 同步路由跑在线程池里，故"一借一还、不跨线程共享"是硬约束。

**流式接口是两阶段的**：检索（``prepare``）借连接、算完**立刻归还**；随后的 SSE 生成
**不再持有任何 DB 连接** —— 否则"正在生成"的请求会把连接池占满（见 :func:`_run_stream`）。

错误统一走 SSE 事件而非 HTTP 状态码（``/api/health`` 例外）；``index.html`` 由本服务同源吐出，无需 CORS。
"""

from __future__ import annotations

import json
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Iterator

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from finrag.config import AppConfig, get_config
from finrag.gen.client import LLMError
from finrag.service import (
    SCOPE_AUTO,
    answer,
    prepare,
    prepare_failed_event,
    scope_modes,
    stream_answer,
)
from finrag.store.pool import close_pool, get_pool, pool_backend, pooled_connection

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent / "static"
INDEX_HTML = STATIC_DIR / "index.html"

#: SSE 响应头。``X-Accel-Buffering: no`` 阻止 nginx 缓冲上游响应而破坏流式。
_SSE_HEADERS = {
    "Cache-Control": "no-cache, no-transform",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


# ------------------------------------------------------------------
# 请求模型
# ------------------------------------------------------------------
class HistoryTurnReq(BaseModel):
    """一轮历史对话（多轮记忆用）。由前端持有并随请求回传。

    **刻意不收 ``doc_ids``** —— 客户端不可信，检索范围一律由服务端用这些字段重新解析
    （``store.resolve``），而不是直接采信客户端给的范围。
    """

    query: str = Field(..., min_length=1, max_length=500)
    #: 该轮改写出的自足问句（未改写 / 改写失败时为空）：下一次改写优先用它，避免指代层层回叠
    standalone: str | None = Field(None, max_length=500)
    #: 该轮解析出的公司短名（``documents.meta->>'company'``，即别名表的键）
    company: str | None = Field(None, max_length=80)
    year: int | None = Field(None, ge=2000, le=2099)
    #: 该轮问句提到的公司名（含库外），供范围继承兜底判断"该不该往回找"
    entities: list[str] = Field(default_factory=list, max_length=20)


class AskRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    #: 口径，默认 ``auto``；可选值由 ``service.scope_modes()`` 给出，不在此另写字面量。
    mode: str = SCOPE_AUTO
    #: 覆盖最终注入 prompt 的条数。``None`` = 用配置值：单实体取
    #: ``retrieve.final_top_k``，多实体按 ``min(top_k_cap, final_top_k × 组数)`` 放大。
    #: **显式给了就不再放大**，以请求为准。
    #: ⚠ ``le=20`` 与 ``retrieve.top_k_cap``（默认也是 20）是**两处独立的 20**，改配置
    #: 不会带动这里。
    top_k: int | None = Field(None, ge=1, le=20)
    #: 是否额外取整节父块，默认关。
    backtrack: bool = False
    #: 是否把模型思考过程（``reasoning_content``）也流给前端，默认开。
    #: 关掉**只影响转发**，请求参数与答案不变（思考模式是服务端默认，见 ``gen/client.py``）。
    show_reasoning: bool = True
    #: 历史（多轮记忆）。前端持有、按时间升序回传；服务端**保留**最近 ``multiturn.max_turns`` 轮
    #: （默认 100），但只有最近 ``multiturn.rewrite_turns`` 轮会真正进改写 prompt。
    #: **上一轮答案正文不进这里** —— 只回传用户问句与解析出的公司/年份。
    history: list[HistoryTurnReq] = Field(default_factory=list, max_length=100)


# ------------------------------------------------------------------
# 应用
# ------------------------------------------------------------------
def create_app(cfg: AppConfig | None = None) -> FastAPI:
    """构造 app。``cfg`` 可注入 ⇒ 验收脚本能在不起服务的前提下拿到路由。"""
    cfg = cfg or get_config()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """预热连接池与模型，避免首个请求等建连 / 加载模型；``FINRAG_SKIP_PRELOAD=1`` 只跳过模型预热（调前端样式时不需要模型）。"""
        import os

        skip = os.environ.get("FINRAG_SKIP_PRELOAD") == "1"
        # 连接池与模型无关，独立预热；失败不让服务起不来，交给 /api/health 报错。
        try:
            get_pool(cfg)
        except Exception:  # noqa: BLE001
            logger.exception("连接池预热失败 —— 服务仍会启动，由 /api/health 报错")
        try:
            if skip:
                logger.warning("已按 FINRAG_SKIP_PRELOAD 跳过模型预热 —— 首个请求会很慢")
            else:
                t0 = time.perf_counter()
                try:
                    from finrag.embed.bge_m3 import get_encoder
                    from finrag.rerank import get_reranker

                    logger.info("预热编码器：%s", cfg.embed.model_path)
                    get_encoder(cfg)
                    logger.info("预热重排器：%s", cfg.rerank.model_path)
                    get_reranker(cfg)
                    logger.info("预热完成，用时 %.1fs", time.perf_counter() - t0)
                except Exception:  # noqa: BLE001
                    # 不让服务启动失败：预热失败（多为显存不够）时仍可起，交给 /api/health 报错。
                    logger.exception("预热失败 —— 服务仍会启动，但首个请求会很慢或直接失败")
            yield
        finally:
            close_pool()

    app = FastAPI(
        title="FinRAG",
        description="A 股年报检索增强问答（双路召回 + RRF + 重排 + DeepSeek）",
        version="0.1.0",
        lifespan=lifespan,
    )

    # ---------------- 页面与健康检查 ----------------
    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        if not INDEX_HTML.exists():
            raise HTTPException(500, f"缺少前端文件：{INDEX_HTML}")
        # 不缓存：改前端后刷新即可见。
        return FileResponse(INDEX_HTML, headers={"Cache-Control": "no-store"})

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        """给编排/自己看的探针，用状态码（异常的例外）。真的连一次库，而非只回 "ok"。"""
        out: dict[str, Any] = {
            "ok": True,
            "model": cfg.llm.model,
            "has_key": cfg.llm.api_key is not None,
            "scope_modes": list(scope_modes()),
            "final_top_k": cfg.retrieve.final_top_k,
            "rerank_enabled": cfg.rerank.enabled,
            #: 连接池后端（psycopg_pool / builtin）—— 排障时先看这一项
            "pool_backend": pool_backend(),
        }
        try:
            with pooled_connection(cfg) as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT count(*) FROM chunks WHERE role = 'child'")
                    out["n_child_chunks"] = cur.fetchone()[0]
                out["n_documents"] = _count_documents(conn)
        except Exception as exc:  # noqa: BLE001
            out["ok"] = False
            out["db_error"] = f"{type(exc).__name__}: {exc}"
        return JSONResponse(out, status_code=200 if out["ok"] else 503)

    # ---------------- 问答 ----------------
    @app.post("/api/chat")
    def api_chat(req: AskRequest) -> StreamingResponse:
        """SSE 流式问答。事件序列：``meta`` → ``reasoning``* → ``delta``* → ``audit`` → ``done``。
        ``reasoning`` 只在请求了思考过程且模型确实产出思维链时出现（开关见 ``AskRequest``）。
        """
        _validate(req)
        return StreamingResponse(
            _run_stream(req, cfg), media_type="text/event-stream", headers=_SSE_HEADERS
        )

    @app.post("/api/answer")
    def api_answer(req: AskRequest) -> dict[str, Any]:
        """非流式问答，给脚本与人工核对用。与 ``/api/chat`` 走同一个 ``service`` 入口。"""
        _validate(req)
        # 连接在 ``with`` 退出时即归还池；生成（``answer``）发生在它之外。
        with pooled_connection(cfg) as conn:
            prepared = prepare(
                req.query, conn, cfg, mode=req.mode,
                top_k=req.top_k, backtrack=req.backtrack, history=_history(req),
            )
        try:
            res = answer(prepared, cfg)
        except LLMError as exc:
            # 非流式可以用状态码：502 = 上游（模型服务）问题，422 = 我们的请求问题。
            code = 502 if exc.kind in ("timeout", "connection", "rate_limit") else 422
            return JSONResponse(
                {"error": exc.to_dict(), "meta": prepared.meta()}, status_code=code
            )
        return res.to_dict()

    @app.get("/api/corpus")
    def corpus() -> dict[str, Any]:
        """库内文档清单，前端用它拼示例问题。必须回 ``company``（``documents.meta`` 的清单短名、
        即 ``resolve`` 别名表的键）—— 前端从全称 ``title`` 截出的短名未必命中别名表，会导致示例解析失败。
        """
        with pooled_connection(cfg) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT doc_id, title, meta->>'code', meta->>'year', "
                    "       meta->>'company', source_path "
                    "FROM documents ORDER BY meta->>'code', meta->>'year'"
                )
                rows = cur.fetchall()
        return {
            "n": len(rows),
            "documents": [
                {"doc_id": r[0], "title": r[1], "code": r[2], "year": r[3],
                 "company": r[4], "source_path": r[5]}
                for r in rows
            ],
        }

    return app


def _count_documents(conn: Any) -> int:
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM documents")
        return int(cur.fetchone()[0])


def _validate(req: AskRequest) -> None:
    """口径校验：路由给用户 400，``service`` 再校一次是纵深防御（拦调用方脚本的编程错误）。"""
    if req.mode not in scope_modes():
        raise HTTPException(
            400, f"mode 只支持 {list(scope_modes())}；open 是评测的作弊口径，线上无对应物"
        )


def _history(req: AskRequest) -> list[dict[str, Any]]:
    """请求里的历史 → ``service`` 要的裸字典列表（只含白名单字段）。"""
    return [t.model_dump() for t in req.history]


def _run_stream(req: AskRequest, cfg: AppConfig) -> Iterator[str]:
    """SSE 生成器，**两阶段**：

    ① **检索阶段** —— 借池连接调 ``prepare``，``with`` 退出即**归还连接**（检索是唯一需要 DB 的部分）；
    ② **生成阶段** —— 只做"组装好的 messages → LLM 流"，**全程不持有 DB 连接**。

    为什么要拆开：若把连接 Hold 到整个生成过程结束，"正在生成"的请求会把连接池占满
    （生成耗时以秒计，而检索只有几十毫秒）。拆开后，连接占用时间与"用户提问频率"成正比，
    与"模型吐字快慢"无关。

    生成器在**路由返回后**才被迭代，故连接必须在这里借、这里还 —— 不能在路由函数里
    ``with``（那时 ``with`` 会随路由返回立即退出，连接在生成开始前就被关了）。
    """
    # ① 检索：借连接 → prepare → 立刻归还
    t0 = time.perf_counter()
    try:
        with pooled_connection(cfg) as conn:
            prepared = prepare(
                req.query, conn, cfg, mode=req.mode,
                top_k=req.top_k, backtrack=req.backtrack, history=_history(req),
            )
    except Exception as exc:  # noqa: BLE001
        logger.exception("prepare 阶段失败")
        yield _sse("error", prepare_failed_event(exc))
        return
    # 逐请求延迟读数落日志：线上才能看 P50/P95 与退化趋势（B 档观测面）
    logger.info(
        "检索完成 用时%.0fms timings_ms=%s",
        (time.perf_counter() - t0) * 1000,
        {k: round(v, 1) for k, v in prepared.timings_ms.items()},
    )

    # ② 生成：不持有 DB 连接
    for ev in stream_answer(prepared, cfg, show_reasoning=req.show_reasoning):
        yield _sse(ev["event"], ev["data"])


def _sse(event: str, data: Any) -> str:
    """事件 → SSE 文本；``ensure_ascii=False`` 避免中文被转义，且每条事件必须以两个换行（``\\n\\n``）结尾，
    少一个浏览器就不认为事件结束。
    """
    payload = json.dumps(data, ensure_ascii=False, default=str)
    return f"event: {event}\ndata: {payload}\n\n"


#: 供 ``scripts/serve.py`` 与 ``uvicorn --factory`` 使用
app = create_app()
