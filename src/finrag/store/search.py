"""检索：稠密 + 稀疏两路召回，在应用层做 RRF 融合。

位于链路中段，输入一条查询、输出带排名溯源的候选片，不做重排/拒答/prompt 组装；
每条结果显示各路 rank 与原始命中序列供消融分析。
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Final, Sequence

from psycopg import sql

from finrag.config import AppConfig, get_config
from finrag.embed.bge_m3 import encode_query

# 连接从 writer 借用：``connect`` 已负责 ``client_encoding="utf8"`` 与
# ``register_vector`` 两件事，重写一份等于多出一处需要同步的定义。
from finrag.store.writer import connect

__all__ = [
    "ROUTE_DENSE",
    "ROUTE_SPARSE",
    "RecallResult",
    "RetrievedChunk",
    "RouteHit",
    "connect",
    "fetch_backtrack",
    "retrieve",
]

logger = logging.getLogger(__name__)

#: 两条召回路的机器名，用于 :attr:`RecallResult.used_routes` 与观测报告分组。
ROUTE_DENSE: Final[str] = "dense"
ROUTE_SPARSE: Final[str] = "sparse"

# 每条路的 SQL 各带 ``role = 'child'`` 与向量列的 ``IS NOT NULL``，不靠外层统一
# 过滤：漏掉会混入父块或把 NULL 向量排到末尾，且都不报错。
#
# ``doc_ids`` 过滤也必须写在每一路里，且与 LIMIT 同层 —— 若等到取完 top-k 再筛，
# 目标文档的片早已被挤掉，返回空集与"确实没有"从结果上看不出区别。
# 用 ``(%s::text[] IS NULL OR doc_id = ANY(...))`` 让过滤与不过滤共用一条路径。
#
# ``rank`` 由 SQL 的 ``ROW_NUMBER()`` 给出，不在 Python 里重排。
_CTE_DENSE_BODY: Final[str] = """
    SELECT id,
           ROW_NUMBER() OVER (ORDER BY d) AS rank,
           d AS score
    FROM (
        SELECT id, dense_vec <=> %s AS d
        FROM chunks
        WHERE role = 'child' AND dense_vec IS NOT NULL
          AND (%s::text[] IS NULL OR doc_id = ANY(%s::text[]))
        ORDER BY d
        LIMIT %s
    ) t
"""

# ``<#>`` 返回负内积，升序即内积降序，与 ``<=>`` 同向，无需给稀疏路取负。
_CTE_SPARSE_BODY: Final[str] = """
    SELECT id,
           ROW_NUMBER() OVER (ORDER BY s) AS rank,
           s AS score
    FROM (
        SELECT id, sparse_vec <#> %s AS s
        FROM chunks
        WHERE role = 'child' AND sparse_vec IS NOT NULL
          AND (%s::text[] IS NULL OR doc_id = ANY(%s::text[]))
        ORDER BY s
        LIMIT %s
    ) t
"""

_CTE_BODY_BY_ROUTE: Final[dict[str, str]] = {
    ROUTE_DENSE: _CTE_DENSE_BODY,
    ROUTE_SPARSE: _CTE_SPARSE_BODY,
}


# ------------------------------------------------------------------
# 结果类型
# ------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class RouteHit:
    """单路的一条命中；``score`` 是该路自己的距离/负分，与其余路量纲不可比。"""

    id: int
    rank: int
    score: float


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    """一条候选片。字段与 ``chunks`` 表的列对应，另加召回溯源。"""

    id: int
    doc_id: str
    content: str
    heading_path: str | None
    page_from: int | None
    page_to: int | None
    token_count: int | None
    role: str
    parent_id: int | None
    meta: dict[str, Any]
    title: str | None
    source_path: str
    doc_meta: dict[str, Any]

    #: RRF 融合分；唯一可以跨路相加的量。
    rrf_score: float
    #: 被几路召回（1~3）。``== 1`` 意味着它只靠某一路挤进候选集。
    n_hits: int

    dense_rank: int | None = None
    sparse_rank: int | None = None
    #: 余弦距离（0 = 同向，2 = 反向）。**不参与排序**，只供观测。
    dense_dist: float | None = None
    #: 负内积。**不参与排序**，只供观测。
    sparse_negip: float | None = None

    def citation(self) -> str:
        """人类可读的引用标签。前端展示与人工抽查都用它。"""
        bits = [self.title or self.source_path]
        if self.heading_path:
            bits.append(" > ".join(p for p in self.heading_path.split(" > ") if p))
        if self.page_from:
            pages = (
                f"p{self.page_from}"
                if self.page_from == self.page_to
                else f"p{self.page_from}-{self.page_to}"
            )
            bits.append(pages)
        return " ｜ ".join(bits)


@dataclass(slots=True)
class RecallResult:
    """一次召回的完整结果，保留 ``dense`` / ``sparse`` 两路原始命中序列。"""

    query: str
    items: list[RetrievedChunk] = field(default_factory=list)
    dense: list[RouteHit] = field(default_factory=list)
    sparse: list[RouteHit] = field(default_factory=list)
    #: 实际参与融合的路；未列出的路在 :attr:`skipped` 里给出原因。
    used_routes: tuple[str, ...] = ()
    #: 路名 → 跳过原因（恒有值，可能为空字典）。
    skipped: dict[str, str] = field(default_factory=dict)
    timings_ms: dict[str, float] = field(default_factory=dict)
    ef_search: int = 0

    def hits_of(self, route: str) -> list[RouteHit]:
        return {
            ROUTE_DENSE: self.dense,
            ROUTE_SPARSE: self.sparse,
        }[route]

    @property
    def n_dense(self) -> int:
        return len(self.dense)

    @property
    def n_sparse(self) -> int:
        return len(self.sparse)


# ------------------------------------------------------------------
# 单路召回
# ------------------------------------------------------------------
def _run_route(cur: Any, name: str, params: Sequence[Any]) -> list[RouteHit]:
    """跑一路召回，返回 ``[(id, rank, score), ...]``；CTE 名走 :func:`psycopg.sql.Identifier` 拼接。"""
    cte = sql.SQL("WITH {name} AS ({body}) SELECT id, rank, score FROM {name}").format(
        name=sql.Identifier(name), body=sql.SQL(_CTE_BODY_BY_ROUTE[name])
    )
    cur.execute(cte, tuple(params))
    return [RouteHit(id=r[0], rank=r[1], score=float(r[2])) for r in cur.fetchall()]


# ------------------------------------------------------------------
# 主入口
# ------------------------------------------------------------------
def retrieve(
    query: str,
    conn: Any | None = None,
    cfg: AppConfig | None = None,
    *,
    limit: int | None = None,
    doc_ids: Sequence[str] | None = None,
) -> RecallResult:
    """召回 + RRF 融合，不含重排与拒答。``limit`` 为返回条数（默认 ``cfg.retrieve.rerank_candidates``）；
    ``doc_ids`` 为前置过滤，会显著抬高召回率，评测用到时须注明是开卷口径。
    """
    cfg = cfg or get_config()
    rcfg = cfg.retrieve
    own_conn = conn is None
    conn = conn or connect(cfg)

    res = RecallResult(query=query, ef_search=rcfg.ef_search)

    try:
        with conn.transaction():
            # ---- 查询编码：与入库共用同一函数 ----
            t0 = time.perf_counter()
            vecs = encode_query(query, cfg)
            res.timings_ms["encode"] = (time.perf_counter() - t0) * 1000

            # ---- ef_search 设置 ----
            # SET 不接受占位符，只能拼字面量；值来自 pydantic 校验过的 int。
            t0 = time.perf_counter()
            with conn.cursor() as cur:
                cur.execute(
                    sql.SQL("SET LOCAL hnsw.ef_search = {}").format(
                        sql.Literal(int(rcfg.ef_search))
                    )
                )
            res.timings_ms["set_ef_search"] = (time.perf_counter() - t0) * 1000

            with conn.cursor() as cur:
                # ---- 稠密路 ----
                t0 = time.perf_counter()
                res.dense = _run_route(
                    cur, ROUTE_DENSE,
                    [_dense_param(vecs.dense[0]), _doc_param(doc_ids), _doc_param(doc_ids),
                     rcfg.dense_top_k],
                )
                res.timings_ms[ROUTE_DENSE] = (time.perf_counter() - t0) * 1000

                # ---- 稀疏路（IS NOT NULL 已写在 CTE 里）----
                # 稀疏路无结果有两种成因，须分开报：use_sparse=false 是刻意关掉，
                # 稀疏向量为空则是查询本身无词元命中词表（语料/编码异常）。
                t0 = time.perf_counter()
                if not rcfg.use_sparse:
                    res.skipped[ROUTE_SPARSE] = "use_sparse=false（M5 消融①：刻意关掉稀疏路）"
                elif _is_empty_sparse(vecs.sparse[0]):
                    # ``{}`` 内积恒为 0，跑出来是任意顺序的噪声行，整路标记跳过。
                    res.skipped[ROUTE_SPARSE] = "查询的稀疏向量为空（无任何词元命中词表）"
                else:
                    res.sparse = _run_route(
                        cur, ROUTE_SPARSE,
                        [_sparse_param(vecs.sparse[0], vecs.sparse_dim),
                         _doc_param(doc_ids), _doc_param(doc_ids), rcfg.sparse_top_k],
                    )
                res.timings_ms[ROUTE_SPARSE] = (time.perf_counter() - t0) * 1000

            res.used_routes = tuple(
                name
                for name in (ROUTE_DENSE, ROUTE_SPARSE)
                if res.hits_of(name)
            )

            # ---- RRF 融合 ----
            t0 = time.perf_counter()
            res.items = _fuse(conn, res, cfg, limit, doc_ids)
            res.timings_ms["fuse"] = (time.perf_counter() - t0) * 1000
    finally:
        if own_conn:
            conn.close()

    return res


def _fuse(
    conn: Any,
    res: RecallResult,
    cfg: AppConfig,
    limit: int | None,
    doc_ids: Sequence[str] | None,
) -> list[RetrievedChunk]:
    """RRF 融合并回库取正文（在应用层做）。"""
    rcfg = cfg.retrieve
    n = rcfg.rerank_candidates if limit is None else int(limit)
    if not res.used_routes or n <= 0:
        return []

    rrf: dict[int, float] = {}
    hits: dict[int, int] = {}
    rank_of: dict[int, dict[str, int]] = {}
    score_of: dict[int, dict[str, float]] = {}
    for name in res.used_routes:
        for h in res.hits_of(name):
            rrf[h.id] = rrf.get(h.id, 0.0) + 1.0 / (rcfg.rrf_k + h.rank)
            hits[h.id] = hits.get(h.id, 0) + 1
            rank_of.setdefault(h.id, {})[name] = h.rank
            score_of.setdefault(h.id, {})[name] = h.score

    # 取 top-n 回库取正文；用 ``= ANY(%s)`` 而非拼接 IN，参数均为外部值。
    ids = sorted(rrf, key=lambda i: (-rrf[i], i))[:n]
    if not ids:
        return []
    where = ["c.id = ANY(%s)"]
    params: list[Any] = [ids]
    if doc_ids:
        where.append("c.doc_id = ANY(%s)")
        params.append(list(doc_ids))

    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT c.id, c.doc_id, c.content, c.heading_path, c.page_from, c.page_to,
                   c.token_count, c.role, c.parent_id, c.meta,
                   d.title, d.source_path, d.meta
            FROM chunks c
            JOIN documents d ON d.doc_id = c.doc_id
            WHERE {' AND '.join(where)}
            """,
            tuple(params),
        )
        rows = cur.fetchall()

    by_id = {r[0]: r for r in rows}
    dropped = [i for i in ids if i not in by_id]
    if dropped:
        # 正常情况下是被 doc_ids 过滤掉的；记日志而非静默。
        logger.debug("融合后有 %d 条未取回（多为 doc_ids 过滤）：%s",
                     len(dropped), dropped[:5])

    # ``= ANY`` 不保证返回顺序，而结果顺序是本模块的产物之一，故按 rrf 分在
    # Python 里排回，并以 id 做稳定 tie-break。
    out: list[RetrievedChunk] = []
    for cid in ids:
        row = by_id.get(cid)
        if row is None:
            continue
        out.append(
            RetrievedChunk(
                id=row[0], doc_id=row[1], content=row[2], heading_path=row[3],
                page_from=row[4], page_to=row[5], token_count=row[6],
                role=row[7], parent_id=row[8], meta=row[9] or {},
                title=row[10], source_path=row[11], doc_meta=row[12] or {},
                rrf_score=rrf[cid], n_hits=hits[cid],
                dense_rank=rank_of[cid].get(ROUTE_DENSE),
                sparse_rank=rank_of[cid].get(ROUTE_SPARSE),
                dense_dist=score_of[cid].get(ROUTE_DENSE),
                sparse_negip=score_of[cid].get(ROUTE_SPARSE),
            )
        )
    return out


# ------------------------------------------------------------------
# 父块回溯（只取数，不决策）
# ------------------------------------------------------------------
def fetch_backtrack(conn: Any, chunk_ids: Sequence[int]) -> dict[int, RetrievedChunk]:
    """按 ``chunks.parent_id`` 取回溯目标，返回 ``{目标 id: 目标}``。目标含 ``role='parent'`` 节父块
    （供整节上下文、不参与排名）与 ``role='child'`` 表头片（供列名、本身可召回）；只取数，不决策不截断。
    """
    if not chunk_ids:
        return {}
    # 先按 id 取子块并读出其 parent_id（不是按 parent_id 反查，反查会返回空集），
    # 再据此取目标内容；两条 SQL 需成对理解。
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id, parent_id FROM chunks WHERE id = ANY(%s)",
            (list(chunk_ids),),
        )
        pairs = [(r[0], r[1]) for r in cur.fetchall()]

    target_ids = sorted({pid for _cid, pid in pairs if pid is not None})
    if not target_ids:
        return {}

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT c.id, c.doc_id, c.content, c.heading_path, c.page_from, c.page_to,
                   c.token_count, c.role, c.parent_id, c.meta,
                   d.title, d.source_path, d.meta
            FROM chunks c
            JOIN documents d ON d.doc_id = c.doc_id
            WHERE c.id = ANY(%s)
            """,
            (target_ids,),
        )
        rows = cur.fetchall()

    # 把挂载关系写进 meta['backtrack_of']，调用方无需再查库。
    # 该 meta 运行时拼成、不落库，与库内那行的 meta 字段集不同。
    kids_of: dict[int, list[int]] = {}
    for cid, pid in pairs:
        kids_of.setdefault(pid, []).append(cid)

    return {
        row[0]: RetrievedChunk(
            id=row[0], doc_id=row[1], content=row[2], heading_path=row[3],
            page_from=row[4], page_to=row[5], token_count=row[6],
            role=row[7], parent_id=row[8],
            meta={**(row[9] or {}), "backtrack_of": sorted(kids_of.get(row[0], []))},
            title=row[10], source_path=row[11], doc_meta=row[12] or {},
            rrf_score=0.0, n_hits=0,
        )
        for row in rows
    }


# ------------------------------------------------------------------
# 向量参数
# ------------------------------------------------------------------
def _dense_param(vec: Any) -> Any:
    """稠密查询向量 → ``<=>`` 的参数，传 :class:`pgvector.Vector` 对象而非文本字面量。"""
    from pgvector import Vector

    return Vector(vec)


def _sparse_param(weights: dict[int, float], dim: int) -> Any:
    """稀疏查询权重 → ``<#>`` 的参数；维度必须显式给，否则按最大索引推断会与列定义不符。"""
    from pgvector import SparseVector

    return SparseVector(weights, int(dim))


def _doc_param(doc_ids: Sequence[str] | None) -> list[str] | None:
    """``doc_ids`` → 召回 SQL 的过滤参数；``None`` 表示不过滤（空列表会过滤掉全部结果）。"""
    return list(doc_ids) if doc_ids else None


def _is_empty_sparse(weights: dict[int, float]) -> bool:
    return not weights