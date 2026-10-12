"""重排层 —— 把召回候选按 Cross-Encoder 分数重排，截到 ``final_top_k``。

本层另开 :class:`RerankResult`、不改 ``RecallResult`` 以保留 M5 消融所需的原始命中序列；
无分组时 ``items == candidates[:top_k]``。**给了分组就按组保底选取**，``items`` 不再是
``candidates`` 的前缀（见 :func:`_select_with_quota`）。分组只有一种来源：**公司多实体**
（对比类问句）—— 同一句 query、各组换 ``doc_ids``；合并后每片归它自己的组，故用
``group_of_chunk``（按片 id）传标签。

``enabled=false`` 时退回"召回 + RRF"形态。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Final, Iterable, Mapping, Sequence

from finrag.config import AppConfig, get_config
from finrag.rerank.bge_reranker import get_reranker
from finrag.rerank.bge_reranker import score_pairs as _score_pairs
from finrag.store.search import (
    ROUTE_DENSE,
    ROUTE_SPARSE,
    RecallResult,
    RetrievedChunk,
    retrieve,
)

__all__ = [
    "RerankResult",
    "RerankedChunk",
    "rerank",
    "retrieve_and_rerank",
]

#: 两条召回路的固定次序；合并分组召回结果时按它遍历。
_ROUTES: Final[tuple[str, ...]] = (ROUTE_DENSE, ROUTE_SPARSE)

#: 每组候选池的绝对下限：太小会让某一组在重排池里缺席，配额也就无从谈起。
_MIN_PER_GROUP: Final[int] = 8

#: 每组候选池相对"该组保底名额"的倍数。保底名额 = ``top_k // 组数``；池子只比名额多一点点时，
#: 重排几乎没有挑选余地（把不相关的也塞进 prompt）。取 3 是为了留出被重排否掉的余量。
_POOL_PER_QUOTA: Final[int] = 3


@dataclass(frozen=True, slots=True)
class RerankedChunk:
    """一条重排后的候选，包着原 ``RetrievedChunk`` 不替换，以保留 dense_rank 等溯源头。"""

    chunk: RetrievedChunk

    #: 它在输入候选池里的名次（1-based，即 RRF 融合后的次序）。
    rrf_rank: int

    #: Cross-Encoder 原始 logit（未过 sigmoid，量级约 -10~+10）；
    #: ``None`` 表示走了 ``enabled=false`` 的回退路径。
    score: float | None

    #: 重排后的名次（1-based）。``None`` 同 ``score``。
    rerank_rank: int | None


@dataclass(slots=True)
class RerankResult:
    """一次重排的完整结果。

    ``items`` 给 prompt、``candidates`` 给报告与排查。**无分组时** ``items == candidates[:top_k]``；
    多实体问句走了配额配平时 ``items`` 是 ``candidates`` 的一个子集而非前缀（见 ``grouped``）。
    """

    query: str

    #: 最终 top_k 条，**顺序即引用顺序**。
    items: list[RerankedChunk] = field(default_factory=list)
    #: 全部候选（已按重排分排好），``items`` 是它的子集。
    candidates: list[RerankedChunk] = field(default_factory=list)
    #: 进了候选池但没进 ``items`` 的，仅供观测，未被丢弃（仍在 ``candidates`` 中带分数）。
    dropped: list[RerankedChunk] = field(default_factory=list)

    #: 是否真的跑了模型；``False`` = 走了 ``enabled=false`` 的回退路径，
    #: 是区分"关掉重排"与"重排未改变顺序"的唯一出口。
    scored: bool = False

    #: 是否按组做了配额配平（多实体问句）。``True`` 时 ``items`` 不再等于 ``candidates[:top_k]``。
    grouped: bool = False

    top_k: int = 0
    n_candidates: int = 0
    elapsed_ms: float = 0.0

    #: 生效口径，落进报告；``device`` 须取对象上的真实值而非配置值。
    model: str = ""
    device: str = ""
    max_length: int = 0


def rerank(
    query: str,
    chunks: Sequence[RetrievedChunk],
    cfg: AppConfig | None = None,
    *,
    top_k: int | None = None,
    reranker: Any = None,
    group_of_doc: Mapping[str, str] | None = None,
    group_of_chunk: Mapping[int, str] | None = None,
) -> RerankResult:
    """把候选片按 Cross-Encoder 分数重排，取前 ``top_k``；``chunks`` 须已按 RRF 排好
    （本函数用传入次序作 ``rrf_rank`` 与同分 tie-break）。

    ``group_of_doc`` / ``group_of_chunk`` 二选一，给一个"片 → 组标签"的映射：
    前者按 ``doc_id``，后者按片 ``id``（公司多实体走分组召回时用后者）。给了且不止一组时，
    选取改走**按组保底**（:func:`_select_with_quota`），保证每组都有片进 ``items``。
    """
    cfg = cfg or get_config()
    rcfg = cfg.rerank
    k = cfg.retrieve.final_top_k if top_k is None else int(top_k)
    cands = list(chunks)
    n = len(cands)
    t0 = time.perf_counter()
    label_of = _label_resolver(group_of_doc, group_of_chunk)
    grouped = _multi_labels(label_of(c) for c in cands)

    base = dict(
        query=query, top_k=k, n_candidates=n,
        model=cfg.rerank.model, device=cfg.rerank.device,
        max_length=rcfg.max_length, grouped=grouped,
    )

    if not cands:
        return RerankResult(**base, elapsed_ms=(time.perf_counter() - t0) * 1000)

    if not rcfg.enabled:
        # ---- 回退网：不跑模型，原序返回 ----
        # 不重新排序，只按传入次序编号：传入的即 RRF 序，再排一次会引入无报错的偏差。
        ordered = [
            RerankedChunk(chunk=c, rrf_rank=i, score=None, rerank_rank=None)
            for i, c in enumerate(cands, 1)
        ]
        items = _select_with_quota(ordered, label_of, k)
        return RerankResult(
            **base, candidates=ordered, items=items, dropped=_dropped(ordered, items),
            scored=False, elapsed_ms=(time.perf_counter() - t0) * 1000,
        )

    # ---- 真重排 ----
    scores = _score_pairs(query, [c.content for c in cands], cfg, reranker=reranker)
    # 同分 tie-break：先比 rrf_rank（= 下标 i+1），再比片 id，
    # 两级都确定 ⇒ 输出与集合迭代顺序无关。
    order = sorted(
        range(n),
        key=lambda i: (-scores[i], i, cands[i].id),
    )
    ranked = [
        RerankedChunk(chunk=cands[i], rrf_rank=i + 1, score=scores[i], rerank_rank=r)
        for r, i in enumerate(order, 1)
    ]
    items = _select_with_quota(ranked, label_of, k)
    return RerankResult(
        **base, candidates=ranked, items=items, dropped=_dropped(ranked, items),
        scored=True, elapsed_ms=(time.perf_counter() - t0) * 1000,
    )


def _label_resolver(
    group_of_doc: Mapping[str, str] | None,
    group_of_chunk: Mapping[int, str] | None,
) -> Callable[[RetrievedChunk], str | None]:
    """构造"片 → 组标签"的取标签函数；两种映射给任一即可（``chunk`` 优先），都给或都不给都安全。"""
    if group_of_chunk is not None:
        return lambda c: group_of_chunk.get(c.id)
    if group_of_doc is not None:
        return lambda c: group_of_doc.get(c.doc_id)
    return lambda c: None


def _multi_labels(labels: Iterable[str | None]) -> bool:
    """标签流里是否有 ≥2 个**非空**标签 —— 是"要不要按组保底"的开关。"""
    return len({l for l in labels if l}) > 1


def _dropped(ranked: Sequence[RerankedChunk], items: Sequence[RerankedChunk]) -> list[RerankedChunk]:
    """``candidates`` 里没进 ``items`` 的那些；按片 id 比对，不依赖顺序。"""
    keep = {x.chunk.id for x in items}
    return [x for x in ranked if x.chunk.id not in keep]


def _select_with_quota(
    ranked: Sequence[RerankedChunk],
    label_of: Callable[[RetrievedChunk], str | None],
    k: int,
) -> list[RerankedChunk]:
    """从已按分降序的 ``ranked`` 里选 ``k`` 条；有 ≥2 组时先给每组保底 ``k // 组数``（有则取），
    再按全局分数补齐剩余名额，最后按分数序输出（引用顺序体现相关性，而非组序）。

    ``label_of(chunk)`` 给出一个片属于哪一组（``None`` = 不属任何组）。标签通常是"公司"
    —— 本函数只认标签，不关心标签从哪来。

    保底是"有则取"：某组候选不够时不补，把名额让给别组 —— 宁可配额不填满，也不重复同一家。
    """
    labels = list(dict.fromkeys(l for l in (label_of(x.chunk) for x in ranked) if l))
    if len(labels) <= 1 or k <= 0:
        return list(ranked[:k])

    floor_q = max(1, k // len(labels))

    picked: list[RerankedChunk] = []
    taken: set[int] = set()
    for label in labels:
        got = 0
        for x in ranked:
            if got >= floor_q:
                break
            if label_of(x.chunk) == label and x.chunk.id not in taken:
                picked.append(x)
                taken.add(x.chunk.id)
                got += 1
    for x in ranked:
        if len(picked) >= k:
            break
        if x.chunk.id not in taken:
            picked.append(x)
            taken.add(x.chunk.id)

    # 恢复分数序：``ranked`` 本身有序，用它的下标当序键。
    pos = {x.chunk.id: i for i, x in enumerate(ranked)}
    picked.sort(key=lambda x: pos.get(x.chunk.id, len(ranked)))
    return picked[:k]


def retrieve_and_rerank(
    query: str,
    conn: Any | None = None,
    cfg: AppConfig | None = None,
    *,
    doc_ids: Sequence[str] | None = None,
    top_k: int | None = None,
    reranker: Any = None,
    groups: Sequence[tuple[str, Sequence[str]]] | None = None,
) -> tuple[RecallResult, RerankResult]:
    """召回 + 重排，返回 ``(RecallResult, RerankResult)`` 两个都给；不合成一个对象以免半新半旧，
    取候选用默认 limit 而非 ``top_k``。

    ``groups`` 是 ``[(组标签, 该组 doc_ids), ...]``（多实体问句，由调用方从 ``QueryScope.groups``
    转成裸元组传入）。有 2 组以上时走**分组召回 + 配额配平**；否则沿用"并集一次召回"。
    """
    cfg = cfg or get_config()
    usable = [(label, tuple(ids)) for label, ids in (groups or []) if ids]
    if len(usable) >= 2:
        k = _effective_top_k(cfg, len(usable), top_k)
        # 公司多实体：各组**同一句 query**、只换 doc_ids
        plans = [(label, query, ids) for label, ids in usable]
        recall, label_of_chunk = _retrieve_plans(plans, conn, cfg, top_k=k, query=query)
        rr = rerank(query, recall.items, cfg, top_k=k, reranker=reranker,
                    group_of_chunk=label_of_chunk)
        return recall, rr

    recall = retrieve(query, conn, cfg, doc_ids=doc_ids)
    rr = rerank(query, recall.items, cfg, top_k=top_k, reranker=reranker)
    return recall, rr


def _effective_top_k(cfg: AppConfig, n_groups: int, top_k: int | None = None) -> int:
    """多实体问句的最终条数。

    调用方显式给了 ``top_k`` 就以它为准（尊重请求的显式意图，不擅自放大）；否则按
    **每家 ``retrieve.final_top_k`` 条**放大：``final_top_k × 组数``。
    放大**永远开启**（2026-10-11 起不再有独立开关）—— 每家的条数就是单实体问句的条数，
    目的是"多家各自都要有与单家同等深度的证据"，而不是把同一份预算摊薄。

    例（``final_top_k=15`` / ``top_k_cap=20``）：2 组 → ``min(20, 30)`` = **20**（每家保底 10）、
    3 组 → 20、4 组 → 20。**放大会撞 ``top_k_cap``**，故组数越多每家实际拿到的越少
    （保底是"有则取"，见 :func:`_select_with_quota`）。
    """
    if top_k is not None:
        return int(top_k)
    base = cfg.retrieve.final_top_k
    return min(cfg.retrieve.top_k_cap, base * max(1, n_groups))


def _per_group_pool(cfg: AppConfig, n_groups: int, top_k: int | None = None) -> int:
    """每组送进重排的候选数（RRF 融合后的截断值）。

    ``retrieve.group_pool_full`` 开且组数 ≤ ``full_pool_max_groups`` 时，**每组各取满**
    ``rerank_candidates``（当前默认 100：两家 → 200，三家 → 300）—— 重排耗时随公司数线性增长，
    这是多家对比的固有代价。否则按组数均分（各家共享一个总池）。两种情况都受两条下限兜底：

    1. ``(top_k // 组数) × _POOL_PER_QUOTA`` —— 保证每组池子比它的保底名额宽裕，重排有挑选余地；
    2. ``_MIN_PER_GROUP`` —— 绝对下限。

    当前默认（``group_pool_full=true`` / ``rerank_candidates=100`` / 2 组）下结果是 100，即每家
    各 100 条、合计 200 条进重排（≈5.8s）。``rerank_candidates=40`` 时代是每家 40、合计 80（≈2.3s）。
    """
    n = max(1, n_groups)
    k = cfg.retrieve.final_top_k if top_k is None else int(top_k)
    floor_q = max(1, k // n)
    full = cfg.retrieve.group_pool_full and n <= cfg.retrieve.full_pool_max_groups
    base = cfg.retrieve.rerank_candidates if full else cfg.retrieve.rerank_candidates // n
    return max(base, floor_q * _POOL_PER_QUOTA, _MIN_PER_GROUP)


def _retrieve_plans(
    plans: Sequence[tuple[str, str, Sequence[str]]],
    conn: Any | None,
    cfg: AppConfig,
    top_k: int | None = None,
    query: str = "",
) -> tuple[RecallResult, dict[int, str]]:
    """``[(组标签, 问句, doc_ids)]`` → 各组**按自己的问句**分别召回，再合并成一份
    ``RecallResult``，并附上 ``片 id → 组标签`` 的映射（重排后按组保底要用）。

    调用方是公司多实体（各组问句相同、只换 doc_ids）。

    为什么必须分组召回：每条路的 SQL 都带 ``LIMIT``（``dense_top_k`` / ``sparse_top_k``），
    在"文档并集"上取 top-N 时，片段多的一组可以把别组**整个挤出候选池** —— 到了重排那一步
    再想配平已经没有候选可用。

    注意**路由 LIMIT 是按组各自生效的**：每组各跑一次完整召回，每组的 30 是各自的，不是从
    一个公共 30 里分 —— 被均分掉的只是这里 RRF 融合后的候选池（``_per_group_pool``）。

    一个片被**多个**组命中时，归给 **rrf 分最高**的那个组。
    合并后的 ``dense`` / ``sparse`` 是各组命中序列的拼接，仅供观测；真正的次序
    由 :func:`rerank` 用 Cross-Encoder 重做，不依赖这里。
    """
    per = _per_group_pool(cfg, len(plans), top_k)
    pairs = [
        (label, q, retrieve(q, conn, cfg, doc_ids=list(ids), limit=per))
        for label, q, ids in plans
    ]

    merged = RecallResult(query=query, ef_search=cfg.retrieve.ef_search)
    best: dict[int, tuple[str, float, RetrievedChunk]] = {}
    for label, _q, r in pairs:
        for name in _ROUTES:
            merged.hits_of(name).extend(r.hits_of(name))
        for k, v in r.skipped.items():
            merged.skipped.setdefault(k, v)
        for k, v in r.timings_ms.items():
            merged.timings_ms[k] = merged.timings_ms.get(k, 0.0) + v
        for c in r.items:
            cur = best.get(c.id)
            if cur is None or c.rrf_score > cur[1]:
                best[c.id] = (label, c.rrf_score, c)

    merged.used_routes = tuple(n for n in _ROUTES if merged.hits_of(n))
    # 跨组的 rrf 分同量纲（都是 1/(k+rank) 之和），可作合并后的粗略序；最终排序交给重排。
    merged.items = sorted((v[2] for v in best.values()), key=lambda c: (-c.rrf_score, c.id))
    label_of_chunk = {cid: lab for cid, (lab, _s, _c) in best.items()}
    merged.skipped["_multi_group"] = (
        f"按 {len(plans)} 组分别召回（每组候选 {per} 条）后在应用层合并"
    )
    return merged, label_of_chunk
