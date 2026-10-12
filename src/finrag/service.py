"""一次问答的组合根：检索 → 重排 → 组装 → 生成。

唯一允许横向依赖各层（``store`` / ``rerank`` / ``gen``）的模块；路由与验收脚本
都经此走同一条调用链，不各自拼接链路。

**多轮记忆**（可选，``multiturn`` 配置）在这里串起来：历史 → 问句改写 → 范围继承 → 检索。
上一轮**答案正文永不进 prompt**，只传用户问句与解析出的实体（见 ``gen/rewrite.py``）。
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator

from finrag.config import AppConfig, get_config
from finrag.gen.client import ChatReply, LLMError, chat, stream_chat
from finrag.gen.prompt import (
    Citation,
    NumberAudit,
    audit_numbers,
    build_citations,
    build_messages,
    is_refusal,
)
from finrag.gen.rewrite import HistoryTurn, RewriteResult, rewrite_query
from finrag.rerank import (
    RerankResult,
    retrieve_and_rerank,
)
from finrag.store.resolve import QueryScope, resolve
from finrag.store.search import RecallResult, fetch_backtrack

__all__ = [
    "SCOPE_AUTO",
    "SCOPE_CLOSED",
    "Answer",
    "Prepared",
    "answer",
    "is_refusal",
    "prepare",
    "prepare_failed_event",
    "scope_modes",
    "stream_answer",
]

logger = logging.getLogger(__name__)

#: 从问句里认公司名/年份再限定文档 —— **线上真实形态**（``eval_set.yaml``）
SCOPE_AUTO = "auto"
#: 全库不过滤 —— 含"专名消歧"的难度
SCOPE_CLOSED = "closed"


def scope_modes() -> tuple[str, ...]:
    """服务支持的口径：``auto`` / ``closed``，不含 ``open``（评测作弊口径）；做成函数而非常量，便于路由与文档共用同一处定义。"""
    return (SCOPE_AUTO, SCOPE_CLOSED)


def _route_breakdown(recall: RecallResult | None) -> dict[str, int]:
    """两路**召回射程内**的命中按"命中哪几路"计数 —— 路线级检测的入口。

    回答的是"稀疏路到底有没有用"：``sparse_only`` 大，说明它带回了稠密路够不着的片；
    全落在 ``both`` 则它只是重复劳动。**按射程算、不看融合后的池子** —— 后者已被
    RRF + 截断洗过，看不出某一路的原始贡献。
    """
    if recall is None:
        return {"dense": 0, "sparse": 0, "both": 0, "dense_only": 0, "sparse_only": 0}
    dense = {h.id for h in recall.dense}
    sparse = {h.id for h in recall.sparse}
    return {
        "dense": len(dense),
        "sparse": len(sparse),
        "both": len(dense & sparse),
        "dense_only": len(dense - sparse),
        "sparse_only": len(sparse - dense),
    }


# ------------------------------------------------------------------
# 组合结果
# ------------------------------------------------------------------
@dataclass(slots=True)
class Prepared:
    """一次问答的检索侧全部产物，保留 ``recall`` / ``rerank`` / ``messages`` 三段中间态。"""

    #: 用户**原**问句（前端展示、审计留痕）
    query: str
    mode: str
    scope: QueryScope | None = None
    recall: RecallResult | None = None
    rerank: RerankResult | None = None
    citations: list[Citation] = field(default_factory=list)
    messages: list[dict[str, str]] = field(default_factory=list)
    #: 命中 ``doc_ids`` 之外的父块 / 表头片（``backtrack=True`` 时才有值）
    backtrack: list[Any] = field(default_factory=list)
    #: 检索与 prompt 实际使用的问句（多轮改写后；未改写时为空串，回落到 ``query``）
    effective_query: str = ""
    #: 多轮改写结果；无历史 / 未启用时为 ``None``
    rewrite: RewriteResult | None = None
    #: 本次**保留**的历史轮数（受 ``multiturn.max_turns`` 限制）
    history_kept: int = 0
    #: 其中真正**送进改写 prompt** 的轮数（受 ``multiturn.rewrite_turns`` 限制）——
    #: "保留多、喂得少"就体现在这两个数的差上。
    history_for_rewrite: int = 0

    #: 非空 = 不要调模型，直接把它作为答复
    blocked: str = ""
    timings_ms: dict[str, float] = field(default_factory=dict)

    @property
    def items(self) -> list[Any]:
        return list(self.rerank.items) if self.rerank else []

    @property
    def resolved_query(self) -> str:
        """下游（检索 / prompt）实际使用的问句：改写过就是改写结果，否则是原问句。"""
        return self.effective_query or self.query

    def meta(self) -> dict[str, Any]:
        """给前端的检索侧元信息。**字段名即前端契约**，改动要同步 ``index.html``。"""
        rr = self.rerank
        items = rr.items if rr else []

        def _rrf_rank(chunk: Any) -> int | None:
            """该片在**融合池**里的位次（重排前，1-based）。与 ``score`` 一起读，就知道重排把它挪了多少。"""
            return next((x.rrf_rank for x in items if x.chunk is chunk), None)

        cites = [
            {
                "index": c.index,
                "label": c.label,
                "chunk_id": getattr(c.chunk, "id", None),
                "score": next(
                    (x.score for x in items if x.chunk is c.chunk),
                    None,
                ),
                #: 融合池位次（重排前）
                "rrf_rank": _rrf_rank(c.chunk),
                "page_from": getattr(c.chunk, "page_from", None),
                "page_to": getattr(c.chunk, "page_to", None),
                "token_count": getattr(c.chunk, "token_count", None),
                #: **路线级溯源**：这条引用被哪几路召回、各路排第几、两路的原始分、是否两路都中
                "routes": {
                    "dense_rank": getattr(c.chunk, "dense_rank", None),
                    "sparse_rank": getattr(c.chunk, "sparse_rank", None),
                    "dense_dist": getattr(c.chunk, "dense_dist", None),
                    "sparse_negip": getattr(c.chunk, "sparse_negip", None),
                    "n_hits": getattr(c.chunk, "n_hits", None),
                },
                "preview": (c.text or "")[:220],
            }
            for c in self.citations
        ]
        return {
            "query": self.query,
            #: 实际用于检索与 prompt 的问句（多轮改写后可能与 query 不同）
            "effective_query": self.resolved_query,
            "mode": self.mode,
            "blocked": self.blocked,
            "rewrite": (
                self.rewrite.to_dict() if self.rewrite is not None
                else {"original": self.query, "standalone": self.query, "used": False, "error": "off"}
            ),
            #: 多轮窗口：保留了几轮、真正喂给改写模型几轮（"保留多、喂得少"）
            "history": {
                "kept": self.history_kept,
                "for_rewrite": self.history_for_rewrite,
            },
            "scope": {
                "company": self.scope.company if self.scope else None,
                "year": self.scope.year if self.scope else None,
                "code": self.scope.code if self.scope else None,
                "matched_alias": self.scope.matched_alias if self.scope else None,
                "n_doc_ids": len(self.scope.doc_ids) if self.scope else 0,
                "reason": self.scope.reason if self.scope else "",
                #: True = 本轮没认出公司，范围沿用自上一轮（多轮记忆）
                "inherited": bool(self.scope.inherited) if self.scope else False,
                #: 问句命中的全部公司（多实体：对比类问句会有多个）
                "companies": list(self.scope.companies) if self.scope else [],
                #: 提到了但本次覆盖不到的公司（库外 / 无对应年份），非空时答复里须说明
                "missing": list(self.scope.missing) if self.scope else [],
                #: 按公司分组的文档集合（多实体配额配平的单位）
                "groups": (
                    [{"label": g.label, "n_docs": len(g.doc_ids)} for g in self.scope.groups]
                    if self.scope else []
                ),
            },
            "recall": {
                "n_dense": self.recall.n_dense if self.recall else 0,
                "n_sparse": self.recall.n_sparse if self.recall else 0,
                "used_routes": list(self.recall.used_routes) if self.recall else [],
                "skipped": dict(self.recall.skipped) if self.recall else {},
                "ef_search": self.recall.ef_search if self.recall else 0,
                #: 路线级检测：两路射程内命中的"共同 / 各自独占"分解（看稀疏路有没有真贡献）
                "routes": _route_breakdown(self.recall),
                #: 逐路耗时（ms）：encode / set_ef_search / dense / sparse / fuse。
                #: 这是判断"稀疏路要不要上索引""哪一路是瓶颈"的前提 —— 线上延迟画像的入口。
                "timings_ms": (
                    {k: round(v, 1) for k, v in self.recall.timings_ms.items()}
                    if self.recall else {}
                ),
            },
            "rerank": {
                "scored": bool(rr and rr.scored),
                "n_candidates": rr.n_candidates if rr else 0,
                "top_k": rr.top_k if rr else 0,
                "elapsed_ms": round(rr.elapsed_ms, 1) if rr else 0.0,
                "model": rr.model if rr else "",
                "device": rr.device if rr else "",
                #: True = 多实体，已按组配额配平（items 不再是 candidates 的前缀）
                "grouped": bool(rr and rr.grouped),
                #: 生效的配对长度上限；与候选池规模一起解释重排耗时
                "max_length": rr.max_length if rr else 0,
                #: 进了候选池但没进 items 的条数（供报告核对，不丢弃、仍在 candidates 里）
                "n_dropped": len(rr.dropped) if rr else 0,
            },
            "citations": cites,
            "timings_ms": {k: round(v, 1) for k, v in self.timings_ms.items()},
        }


@dataclass(slots=True)
class Answer:
    """一次非流式问答的完整结果。``reasoning`` 是思考模式的思维链，与 ``text`` 分开存放 ——
    审计与拒答判断只看 ``text``，不要把它算进去。
    """

    prepared: Prepared
    text: str = ""
    reasoning: str = ""
    refusal: bool = False
    audit: NumberAudit = field(default_factory=NumberAudit)
    usage: dict[str, int] = field(default_factory=dict)
    model: str = ""
    elapsed_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "meta": self.prepared.meta(),
            "answer": self.text,
            "reasoning": self.reasoning,
            "refusal": self.refusal,
            "audit": self.audit.to_dict(),
            "usage": self.usage,
            "model": self.model,
            "elapsed_ms": round(self.elapsed_ms, 1),
        }


# ------------------------------------------------------------------
# 阶段一：检索侧
# ------------------------------------------------------------------
def prepare(
    query: str,
    conn: Any,
    cfg: AppConfig | None = None,
    *,
    mode: str = SCOPE_AUTO,
    top_k: int | None = None,
    backtrack: bool = False,
    history: Iterable[dict[str, Any] | HistoryTurn] | None = None,
) -> Prepared:
    """范围解析 → 双路召回 → 重排 → 组装 prompt，不调模型；``conn`` 必传、由调用方持有
    （服务按请求复用连接，评测脚本用一个连接跑几十条）。

    ``history`` 是最近若干轮的**结构化**历史（每轮：``query`` + ``company`` / ``year``），
    由调用方（前端）持有并回传。服务端把历史当作**不可信输入**：只取字段、不认客户端给的
    ``doc_ids``，检索范围一律在服务端重新解析。

    命中多家公司时走**公司多实体**的分组召回（同句 query、各组换 ``doc_ids``），重排后按组保底。
    """
    cfg = cfg or get_config()
    if mode not in scope_modes():
        raise ValueError(f"未知口径 {mode!r}，只支持 {scope_modes()}")

    t0 = time.perf_counter()
    out = Prepared(query=query, mode=mode)

    # ---- ⓪ 多轮：归一化历史 → 问句改写（含话题判定；无历史时= 只归一化 + 抽实体）----
    # **保留**（``max_turns``，服务端接受多少）与**送进 prompt**（``rewrite_turns``，模型看多少）
    # 是两件事：越远的轮次越不相关，把整段长历史塞给改写模型只会让它选错指代对象
    # （代词几乎只指上一轮），代价是每次提问都多付一份 token。
    # ⚠️ **首轮也会调改写**：没有 history 时不做指代补全，但**要抽实体**——`missing` 的唯一来源，
    #缺了它首轮问句里的库外公司名检测不到，会静默退回全库检索拿别家数字作答（见 `rewrite.py`）。
    hist = _norm_history(history, cfg)
    hist_rw = (
        hist[-cfg.multiturn.rewrite_turns:] if cfg.multiturn.rewrite_turns > 0 else []
    )
    out.history_kept = len(hist)
    out.history_for_rewrite = len(hist_rw)
    t = time.perf_counter()
    # 无条件调用（首轮亦然）：函数内部只自会处理开关关闭（`disabled`）的情形。
    rewrite = rewrite_query(query, hist_rw, cfg)
    out.rewrite = rewrite
    out.effective_query = rewrite.query
    out.timings_ms["rewrite"] = (time.perf_counter() - t) * 1000
    # 下游一律用自足问句：改写过就是改写结果，否则等于原问句。
    q = rewrite.query

    # ---- ① 范围（在自足问句上解析；认不出则按配置沿用上一轮）----
    prior = _prior_scope(hist, cfg)
    t = time.perf_counter()
    if mode == SCOPE_AUTO:
        # cfg 透传给 resolve：Q2 的模糊匹配阈值从这里来（fuzzy 段）。
        # 不传则 resolve 拿不到配置 ⇒ 模糊自动关闭、退回纯精确匹配（fail-closed）。
        scope = resolve(q, conn, prior=prior, entities=rewrite.entities, cfg=cfg)
    else:
        scope = QueryScope(query=q)
    out.scope = scope
    out.timings_ms["resolve"] = (time.perf_counter() - t) * 1000

    # ---- ② 认出了公司名但库里没有 ⇒ 拦住，不问模型 ----
    # 多实体下"认出了公司"包括两类：库内命中的（companies）、以及点了名但覆盖不到的（missing）。
    # 两者都存在而 doc_ids 为空，说明这次答案会缺主体 —— 拦住比给半个答案更诚实。
    if mode == SCOPE_AUTO and (scope.companies or scope.missing) and not scope.doc_ids:
        out.blocked = (
            f"{scope.reason}。"
            f"本次**没有退回全库检索** —— 那会给出别家公司的数字却看着像答案。"
            f"请换一个库内已有的公司/年份，或改用 closed 口径跨全库检索。"
        )
        # 拦截是负样本（库外 / 缺年份）的关键观测点：记一条，便于统计闸门是否真的在生效
        logger.info(
            "范围拦截（未退回全库）：companies=%s missing=%s reason=%s",
            list(scope.companies), list(scope.missing), scope.reason,
        )
        out.timings_ms["total"] = (time.perf_counter() - t0) * 1000
        return out

    # ---- ③ 召回 + 重排 ----
    # 命中多家公司时按"公司分组"各召回再配平（同一句 query、各组换 doc_ids）。
    t = time.perf_counter()
    recall, rr = retrieve_and_rerank(
        q, conn, cfg,
        doc_ids=scope.doc_ids or None,
        top_k=top_k,
        groups=[(g.label, g.doc_ids) for g in scope.groups] or None,
    )
    out.recall, out.rerank = recall, rr
    out.timings_ms["retrieve_rerank"] = (time.perf_counter() - t) * 1000

    # ---- ④ 组装 ----
    t = time.perf_counter()
    out.citations = build_citations(rr.items, cfg)
    out.messages = build_messages(q, rr.items, cfg, missing=scope.missing)
    out.timings_ms["build_prompt"] = (time.perf_counter() - t) * 1000

    # ---- ⑤ 可选：回溯取整节父块 / 表头片（默认关）----
    if backtrack:
        t = time.perf_counter()
        ids = [c.chunk.id for c in rr.items]
        target = fetch_backtrack(conn, ids)
        # 表头片（``role='child'``，参与排名、给列名）与节父块（``role='parent'``，
        # 仅上下文）两类都收下，不做注入优先级的判断。
        out.backtrack = sorted(target.values(), key=lambda x: x.id)
        out.timings_ms["backtrack"] = (time.perf_counter() - t) * 1000

    out.timings_ms["total"] = (time.perf_counter() - t0) * 1000
    return out


def _norm_history(
    history: Iterable[dict[str, Any] | HistoryTurn] | None,
    cfg: AppConfig,
) -> list[HistoryTurn]:
    """原始历史 → :class:`HistoryTurn` 列表。只认白名单字段，**丢弃 ``doc_ids`` 等其余键**
    （客户端不可信）；裁剪到 ``multiturn.max_turns``，只留最近的若干轮。

    ``standalone``（该轮改写出的自足问句）与 ``entities``（该轮提到的公司名，含库外）都由前端回传，
    同样按不可信输入处理：只做长度 / 条数截断，不采信任何"范围"。
    """
    if not history:
        return []
    n = cfg.multiturn.max_turns
    if n <= 0:
        return []

    out: list[HistoryTurn] = []
    for raw in history:
        if isinstance(raw, HistoryTurn):
            out.append(raw)
            continue
        if not isinstance(raw, dict):
            continue
        q = str(raw.get("query") or "").strip()
        if not q:
            continue
        comp = raw.get("company")
        yr = raw.get("year")
        sa = str(raw.get("standalone") or "").strip()
        ents: list[str] = []
        raw_ents = raw.get("entities")
        if isinstance(raw_ents, (list, tuple)):
            for e in raw_ents[:20]:
                s = str(e).strip()
                if s:
                    ents.append(s[:40])
        out.append(
            HistoryTurn(
                query=q[:500],
                company=(str(comp).strip()[:80] or None) if comp else None,
                year=int(yr) if isinstance(yr, int) and 2000 <= yr <= 2099 else None,
                standalone=sa[:500],
                entities=tuple(dict.fromkeys(ents)),
            )
        )
    return out[-n:]


def _prior_scope(hist: list[HistoryTurn], cfg: AppConfig) -> QueryScope | None:
    """供 :func:`resolve` 的 ``prior`` 用的"上一轮"范围 —— 多轮继承的**兜底**。

    改写是主路径：它把公司名写进自足问句，``resolve`` 直接从问句解析，此时本函数不参与。
    仅当本轮（改写后）问句解析不出公司时，才回头用这里的 ``prior``。

    现在**每一轮都进历史**（含被拒答的，"全量保存"），最近一轮未必带公司（纯代词、打错名字、
    点了库外的名），故不能再死盯 ``hist[-1]`` —— 改为**从最近往前找第一个建立了公司的轮次**。
    若先遇到的是"带了实体却没有库内公司"的轮次（多为点了库外的名），返回 ``None``：
    此时该继承的语义是"沿着那家库外公司"，而它没有文档可定位，不能退回更早的公司。
    ``inherit_scope`` 关闭时一律返回 ``None``。
    """
    if not cfg.multiturn.inherit_scope or not hist:
        return None
    for turn in reversed(hist):
        if turn.company:
            return QueryScope(query=turn.query, company=turn.company, year=turn.year)
        if turn.entities:
            return None
    return None


# ------------------------------------------------------------------
# 阶段二：生成侧
# ------------------------------------------------------------------
def answer(
    prepared: Prepared,
    cfg: AppConfig | None = None,
) -> Answer:
    """非流式完整问答。评测与验收走这条。"""
    cfg = cfg or get_config()
    result = Answer(prepared=prepared)
    t0 = time.perf_counter()

    if prepared.blocked:
        result.text = prepared.blocked
        result.elapsed_ms = (time.perf_counter() - t0) * 1000
        return result

    # ``LLMError`` 不在这里吞掉：``kind`` / ``hint`` 要原样交到路由与前端。
    reply: ChatReply = chat(prepared.messages, cfg)
    result.text = reply.text
    result.reasoning = reply.reasoning
    result.usage = reply.usage
    result.model = reply.model
    # 审计用与 prompt 同一批 items
    result.audit = audit_numbers(reply.text, prepared.items, cfg)
    result.refusal = is_refusal(reply.text)
    result.elapsed_ms = (time.perf_counter() - t0) * 1000
    return result


def stream_answer(
    prepared: Prepared,
    cfg: AppConfig | None = None,
    *,
    show_reasoning: bool = True,
) -> Iterator[dict[str, Any]]:
    """流式问答，产出事件字典（``{"event": ..., "data": ...}``）；编码成 SSE 由 ``api/server.py`` 负责。
    阻塞与失败都走事件而非 HTTP 状态码 —— 流式一旦开始响应头已是 200，``kind`` 须进事件体。

    ``show_reasoning`` 只决定思维链是否**转发**给前端：请求参数与答案一字不变，
    关掉纯粹是省掉这几千 token 的传输（服务端该算的 reasoning token 照样算）。
    """
    cfg = cfg or get_config()
    yield {"event": "meta", "data": prepared.meta()}

    if prepared.blocked:
        yield {"event": "delta", "data": {"text": prepared.blocked}}
        yield {"event": "done", "data": {"finish_reason": "blocked", "usage": {}}}
        return

    buf: list[str] = []
    think: list[str] = []
    finish_reason: str | None = None
    t0 = time.perf_counter()
    try:
        for d in stream_chat(prepared.messages, cfg):
            # 思维链与答案分两条流，顺序上 reasoning 一定在前；不拼进 buf，审计只认答案。
            if d.reasoning and show_reasoning:
                think.append(d.reasoning)
                yield {"event": "reasoning", "data": {"text": d.reasoning}}
            if d.text:
                buf.append(d.text)
                yield {"event": "delta", "data": {"text": d.text}}
            if d.finish_reason:
                finish_reason = d.finish_reason
    except LLMError as exc:
        yield {"event": "error", "data": exc.to_dict()}
        # 已吐出的半截答案照样交付并算审计，供前端显示"失败但以下是已生成部分"。
        text = "".join(buf)
        yield {
            "event": "aborted",
            "data": {
                "text": text,
                "audit": audit_numbers(text, prepared.items, cfg).to_dict(),
                "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1),
            },
        }
        return

    text = "".join(buf)
    yield {
        "event": "audit",
        "data": audit_numbers(text, prepared.items, cfg).to_dict(),
    }
    yield {
        "event": "done",
        "data": {
            "finish_reason": finish_reason,
            "refusal": is_refusal(text),
            "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1),
            # 流式拿不到 usage（未传 ``stream_options``），故空字典而非 0。
            "usage": {},
            # 思维链字符数：流式下唯一能观测"这次思考有多长"的量（没有 reasoning_tokens）。
            # ``show_reasoning=False`` 时恒为 0 —— 它反映的是"有没有转发"，不是"模型想没想"。
            "n_reasoning_chars": sum(len(x) for x in think),
        },
    }


def prepare_failed_event(exc: BaseException) -> dict[str, Any]:
    """``prepare`` 抛出异常时给前端的 SSE ``error`` 事件体。

    **单点定义**：路由（``api/server._run_stream``）与任何直接调 ``prepare`` 的调用方共用，
    避免错误体在两处各写一份、日后漂移。检索侧失败时**不调模型**，事件即终态。
    """
    return {
        "kind": "prepare_failed",
        "message": f"{type(exc).__name__}: {exc}",
        "hint": "检索侧在生成之前就失败了。先跑 scripts/check_env.py 看库与模型是否就绪",
        "status": None,
    }
