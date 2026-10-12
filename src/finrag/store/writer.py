"""入库：把解析、分块、编码的产物写入 ``documents`` / ``chunks`` 两张表。

位于链路末端，不做解析/分块/检索决策，只按 ``schema.sql`` 的契约落库，
并保证同一份文档可重复入库而不留下残留。
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Final, Sequence

from finrag.alias import alias_variants
from finrag.chunk.chunker import HEADING_SEP, ROLE_CHILD, Chunk, ChunkResult
from finrag.config import AppConfig, PROJECT_ROOT, get_config
from finrag.embed.bge_m3 import encode_texts
from finrag.manifest import (
    Manifest,
    ManifestError,
    ManifestRecord,
    load_manifest,
    missing_from_manifest,
)
from finrag.parse.ir import ParsedDoc

__all__ = [
    "CONNECT_KWARGS",
    "IngestStats",
    "configure_connection",
    "connect",
    "content_fingerprint",
    "ingest_document",
    "ingest_many",
    "status_label",
    "write_chunks",
]

logger = logging.getLogger(__name__)

#: 建连参数：``client_encoding`` 不可省 —— 缺了中文会整批乱码且**不报错**。
#: 单点定义：``connect()`` 与连接池（``store/pool.py``）都取这一份，别各写一遍。
CONNECT_KWARGS: Final[dict[str, Any]] = {"connect_timeout": 10, "client_encoding": "utf8"}

# 文件名含其中任一标记才认定为年报；仅在清单**未**给 ``category`` 时作兜底。
# 两种命名都要认：`000333_2024_年报.pdf` 与 `000061_农产品_2023年年度报告.pdf`。
_ANNUAL_REPORT_MARKS: Final[tuple[str, ...]] = ("年报", "年度报告")

#: :func:`content_fingerprint` 取 sha256 前多少位；与 ``manifest.DOC_ID_LEN`` 无关（两者语义不同）。
FINGERPRINT_LEN: Final[int] = 16


# ------------------------------------------------------------------
# 内容指纹
# ------------------------------------------------------------------
def _normalized_rel_path(path: str | Path) -> str:
    """规范化相对路径：相对 ``PROJECT_ROOT``，越界则退回绝对路径；统一用正斜杠。"""
    p = Path(path).resolve()
    try:
        return p.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return p.as_posix()


def content_fingerprint(chunks: Sequence[Chunk]) -> str:
    """分块产物的指纹；覆盖 ``content`` / ``chunk_index`` / ``role`` / ``parent_index`` /
    ``heading_path`` / ``page_from`` / ``page_to``，不含 ``meta``，用 ``\\x1f`` / ``\\x1e`` 分隔以保证拼接无歧义。
    """
    h = hashlib.sha256()
    for c in chunks:
        h.update(
            "\x1f".join(
                (
                    str(c.chunk_index),
                    c.role,
                    c.content,
                    HEADING_SEP.join(c.heading_path),
                    str(c.page_from),
                    str(c.page_to),
                    "" if c.parent_index is None else str(c.parent_index),
                )
            ).encode("utf-8")
        )
        h.update(b"\x1e")
    return h.hexdigest()[:FINGERPRINT_LEN]


# ------------------------------------------------------------------
# 连接
# ------------------------------------------------------------------
def configure_connection(conn: Any) -> None:
    """连接**就绪后**的统一配置：注册 pgvector 适配器（``vector`` / ``sparsevec``）。

    ``client_encoding="utf8"`` 是**建连参数**，在建连时给（见 :data:`CONNECT_KWARGS`），不在这里设 ——
    两者都属于"不可省"的一类，排查中文乱码 / 认不出向量 OID 时要一起看。
    单点定义：``connect()`` 与连接池的 ``configure=`` 回调都调它。
    """
    from pgvector.psycopg import register_vector

    register_vector(conn)


def connect(cfg: AppConfig | None = None, *, autocommit: bool = False) -> Any:
    """连库：显式 ``client_encoding="utf8"`` 并注册 pgvector 适配器 —— 两者都不可省，
    缺 ``client_encoding`` 中文会整批乱码且不报错，缺 ``register_vector`` 认不出 vector/sparsevec OID。
    """
    import psycopg

    cfg = cfg or get_config()
    conn = psycopg.connect(
        cfg.pg_dsn.get_secret_value(),
        autocommit=autocommit,
        **CONNECT_KWARGS,
    )
    configure_connection(conn)
    return conn


# ------------------------------------------------------------------
# 统计
# ------------------------------------------------------------------
@dataclass(slots=True)
class IngestStats:
    """一份文档入库后的产出统计。"""

    source_path: Path
    doc_id: str
    #: ``written`` 已写入 / ``skipped`` 指纹一致未动 / ``rejected`` 准入未过 /
    #: ``failed`` 解析或入库时抛异常（单份失败，不中断整批）。
    status: str = "written"
    n_rows: int = 0             # 写进 chunks 的行数（含父块）
    n_children: int = 0         # role='child'，参与排名
    n_parents: int = 0          # role='parent'，只服务回溯
    n_table_headers: int = 0    # 表头片，是 children 的子集
    n_vectors: int = 0          # 实际编码的条数（父块恒不计入）
    n_cached: int = 0           # 上面这些里命中向量缓存的条数
    reason: str = ""            # skipped / rejected 时的原因
    #: 分块警告的**文本**（供报告列明细）。只影响观测，不参与任何产物或指纹。
    warnings: list[str] = field(default_factory=list)

    @property
    def skipped(self) -> bool:
        return self.status == "skipped"


# ------------------------------------------------------------------
# 文档层
# ------------------------------------------------------------------
def _document_row(
    doc: ParsedDoc, record: ManifestRecord, aliases: Sequence[str]
) -> dict[str, Any]:
    """``documents`` 一行的取值；身份与 ``code`` / ``company`` / ``year`` 一律来自清单，不从文件名猜。"""
    return {
        "doc_id": record.doc_id,
        "source_path": _normalized_rel_path(doc.source_path),
        "title": _title_of(record, doc.source_path),
        "doc_type": doc.meta.doc_type,
        "category": record.category or _category_of(doc.source_path),
        "meta": json.dumps(
            {
                # ---- 清单维度：身份与检索期前置过滤（"只查某公司"）----
                "doc_key": record.doc_key,
                "code": record.code,
                "company": record.company,
                "year": record.year,
                # 入库时预计算的公司全称候选：查询期建别名表直接用它，不必再逐文档扫 chunks。
                "aliases": list(aliases),
                # ---- 解析期事实：审计追踪哪些文档走了 TOC 路径 ----
                "doc_type": doc.meta.doc_type,
                "has_toc": doc.meta.has_toc,
                "page_count": doc.meta.page_count,
                "multicol_ratio": doc.meta.multicol_ratio,
                # 字号兜底路径可信度的量化指标，供评测按它分组
                "body_share": doc.meta.body_share,
                # ---- 版本三元组：用于事后分辨"这行数据是哪套判据产的" ----
                "ir_schema": doc.meta.ir_schema,
                "parser_version": doc.meta.parser_version,
                "probe_version": doc.meta.probe_version,
            },
            ensure_ascii=False,
        ),
    }


def _title_of(record: ManifestRecord, path: Path) -> str:
    """展示用标题：清单给了就用；否则"公司 + 年份 + 年度报告"，公司也缺则退回文件名（不含扩展名）。"""
    if record.title:
        return record.title
    if not record.company:
        return path.stem
    return f"{record.company} {record.year}年年度报告" if record.year else f"{record.company}年度报告"


def _category_of(path: Path) -> str | None:
    """``documents.category`` 的兜底：文件名含年报标记才标"财报"，否则留 NULL（清单给了就以清单为准）。"""
    return "财报" if any(m in path.name for m in _ANNUAL_REPORT_MARKS) else None


def _merge_aliases(record: ManifestRecord) -> list[str]:
    """清单给的额外别名（原始名 / 曾用名），经归一后去重保序。

    清单的原始名（如带 ``ST``/``XD`` 标签的 ``ST沈化``）必须留一份：``company`` 已剥标签，
    用户若照原样写"ST沈化"也得能命中。

    每个写法再经 :func:`finrag.alias.alias_variants` 派生归一形态：``*ST`` 在文件名里非法、
    落盘成前导 ``_``（清单里是 ``_ST高斯``），故顺手补出 ``ST高斯`` —— 否则用户写"ST高斯"
    命中不了。注意查询期（``store.resolve`` 建别名表时）也调同一个归一函数，
    所以**历史数据无需重灌**即已生效。

    ⚠ **2026-10-11 移除了"从正文抽公司全称"那一路**（原 `full_name_candidates`）。别名**一律
    来自清单、已人工核对**，不再自动抽取：自动抽取会捞进「就比亚迪」这类句法碎片是不可靠来源，
    而清单侧既然已核对，就不需要它在旁自动化。改动对既有文档**零影响**（公司表正常装载时
    查询期走的是人工剪枝的 ``companies`` / ``company_aliases``，压根不读 ``meta.aliases``），
    **不需要重灌库** —— 只是不再有新数据从这条路进来。
    """
    out: list[str] = []
    seen: set[str] = set()
    for name in record.aliases:
        for n in alias_variants(name):
            if n not in seen:
                seen.add(n)
                out.append(n)
    return out


# ------------------------------------------------------------------
# 分块层
# ------------------------------------------------------------------
# 写入用的列清单；``tsv`` 生成列已随 lexical 路移除，列集不再有例外。
_CHUNK_COLUMNS: Final[tuple[str, ...]] = (
    "doc_id",
    "chunk_index",
    "content",
    "token_count",
    "heading_path",
    "page_from",
    "page_to",
    "parent_id",
    "role",
    "dense_vec",
    "sparse_vec",
    "meta",
)

# 所有占位符都是裸 ``%s``，不加任何类型转换：``%s::vector`` / ``CAST(%s AS vector)``
# 会被 psycopg 的占位符扫描器拆错，列名上更不能加 ``::``。向量的类型信息由
# :func:`connect` 注册的适配器提供，传入 ``Vector`` / ``SparseVector`` 对象。
def _placeholder(name: str) -> str:
    """一列的占位符，恒为 ``%s``（集中在此以便将来引入自定义类型列）。"""
    del name  # 恒为 %s，参数只为让调用点的意图显式（"这一列也要占位符"）
    return "%s"


def _chunk_row(
    doc_id: str,
    c: Chunk,
    vectors: tuple[Any, Any] | None,
    parent_id: int | None,
) -> tuple[Any, ...]:
    dense, sparse = vectors if vectors else (None, None)
    return (
        doc_id,
        c.chunk_index,
        c.content,
        c.token_count,
        c.heading_text,
        c.page_from,
        c.page_to,
        parent_id,
        c.role,
        dense,
        sparse,
        json.dumps(c.meta, ensure_ascii=False),
    )


# 单条 INSERT 攒多少片：一条多行 INSERT + RETURNING 远快于逐条往返，但每行带
# 两个向量（稠密 1024 维 + 稀疏），攒太多会触发驱动的单语句体积上限。64 为实测值。
_INSERT_BATCH: Final[int] = 64


def _insert_round(cur: Any, rows: Sequence[tuple[Any, ...]]) -> dict[int, int]:
    """插入一批片，返回 ``chunk_index -> id``。用多行 INSERT + RETURNING（``executemany`` 拿不到 id）；
    占位符按行数重复、``one`` 已含括号，且 ``fetchall()`` 须紧跟每次 execute，否则只剩最后一批的 id。
    """
    cols = ", ".join(_CHUNK_COLUMNS)
    one = "(" + ", ".join([_placeholder(c) for c in _CHUNK_COLUMNS]) + ")"
    id_of: dict[int, int] = {}
    for start in range(0, len(rows), _INSERT_BATCH):
        batch = rows[start : start + _INSERT_BATCH]
        flat: list[Any] = []
        for r in batch:
            flat.extend(r)
        # VALUES 后面是 one 本身，不要再包一层。
        values = ", ".join([one] * len(batch))
        cur.execute(
            f"INSERT INTO chunks ({cols}) VALUES {values} RETURNING chunk_index, id",
            flat,
        )
        id_of.update({row[0]: row[1] for row in cur.fetchall()})
    return id_of


def write_chunks(
    cur: Any,
    doc_id: str,
    chunks: Sequence[Chunk],
    *,
    vectors: dict[int, tuple[Any, Any]] | None = None,
) -> dict[int, int]:
    """把一份文档的片全部写入，返回 ``chunk_index -> chunks.id``。分轮插入，``parent_index``
    已解析出 id 的片先写（不依赖列表顺序）；走完仍悬空则抛 ``ValueError``，绝不写 NULL。
    """
    vectors = vectors or {}
    pending = list(chunks)
    id_of: dict[int, int] = {}

    while pending:
        ready = [
            c for c in pending
            if c.parent_index is None or c.parent_index in id_of
        ]
        if not ready:
            unresolved = sorted(
                {c.parent_index for c in pending if c.parent_index is not None} - set(id_of)
            )
            raise ValueError(
                f"父块回溯无法解析：{len(pending)} 片悬空，指向 chunk_index "
                f"{unresolved[:5]}。这是分块层的产物不自洽 —— 宁可整份入库失败，"
                f"也不要写出一批 parent_id 为 NULL 的行（那会让回溯静默失效、"
                f"而系统不报任何异常）。"
            )
        id_of.update(
            _insert_round(
                cur,
                [
                    _chunk_row(
                        doc_id,
                        c,
                        vectors.get(c.chunk_index),
                        id_of.get(c.parent_index) if c.parent_index is not None else None,
                    )
                    for c in ready
                ],
            )
        )
        written = {c.chunk_index for c in ready}
        pending = [c for c in pending if c.chunk_index not in written]

    return id_of


# ------------------------------------------------------------------
# 编码
# ------------------------------------------------------------------
def _encode_for(
    chunks: Sequence[Chunk], cfg: AppConfig, *, use_cache: bool = True
) -> tuple[dict[int, tuple[Any, Any]], int]:
    """编码参与排名的片，返回 ``(chunk_index -> (Vector, SparseVector)), 缓存命中数``。
    只编码 ``role == 'child'``；命中数由 :func:`encode_texts` 顺带回报（``EmbeddingSet.n_cached``），
    不再单独预扫一遍缓存；稀疏向量须显式传 ``sparse_dim``（否则推断值偏小）。
    """
    from pgvector import SparseVector, Vector

    children = [c for c in chunks if c.role == ROLE_CHILD]
    if not children:
        return {}, 0

    texts = [c.content for c in children]
    vecs = encode_texts(texts, cfg, use_cache=use_cache)
    n_cached = vecs.n_cached
    dense = vecs.dense
    sparse = vecs.sparse
    out: dict[int, tuple[Any, Any]] = {}
    for c, d, s in zip(children, dense, sparse, strict=True):
        out[c.chunk_index] = (Vector(d), SparseVector(s, vecs.sparse_dim))
    return out, n_cached


# ------------------------------------------------------------------
# 一份文档
# ------------------------------------------------------------------
def ingest_document(
    conn: Any,
    doc: ParsedDoc,
    chunks: ChunkResult | Sequence[Chunk],
    record: ManifestRecord,
    cfg: AppConfig | None = None,
    *,
    force: bool = False,
    use_cache: bool = True,
) -> IngestStats:
    """把一份文档入库（一个事务）。``record`` 是清单里的一行，供身份（``doc_id``）与元数据。

    幂等判据是 :func:`content_fingerprint`：``content_hash`` 一致且行数吻合才整份跳过；
    ``force=True`` 或行数不符时先 DELETE 旧片再插；换 embed 模型需 ``force=True``。
    """
    cfg = cfg or get_config()
    items = list(chunks.chunks if isinstance(chunks, ChunkResult) else chunks)
    children = [c for c in items if c.role == ROLE_CHILD]
    stats = IngestStats(
        source_path=doc.source_path,
        doc_id=record.doc_id,
        n_rows=len(items),
        n_children=len(children),
        n_parents=len(items) - len(children),
        n_table_headers=sum(1 for c in items if c.meta.get("table_header")),
    )

    fingerprint = content_fingerprint(items)
    # 别名在入库时归一好（清单给的写法 → 去前导 _/* 的变体），存进 meta.aliases ——
    # 于是查询期建别名表只需一次 documents 查询，不必再逐文档扫 chunks。
    aliases = _merge_aliases(record)
    row = _document_row(doc, record, aliases)

    with conn.transaction():
        with conn.cursor() as cur:
            # 跳过判定须同时核对 content_hash 与实际行数：只比 hash 无法发现残缺的库。
            cur.execute(
                "SELECT content_hash, n_chunks FROM documents WHERE doc_id = %s",
                (stats.doc_id,),
            )
            found = cur.fetchone()
            stored_rows = 0
            if found and not force:
                cur.execute(
                    "SELECT count(*) FROM chunks WHERE doc_id = %s", (stats.doc_id,)
                )
                stored_rows = cur.fetchone()[0]

    if found and not force:
        same_hash = found[0] == fingerprint
        if same_hash and stored_rows == len(items):
            stats.status = "skipped"
            stats.reason = "content_hash 一致且行数吻合"
            stats.n_rows = 0
            logger.info(
                "跳过 %s：content_hash 一致且行数吻合（%d 片未重写）",
                stats.doc_id, len(items),
            )
            return stats
        # 走到这里说明库内那份是残的，必须重写。
        logger.warning(
            "%s 需要重写：content_hash %s，但库内%s%d 片 / 应有 %d 片",
            stats.doc_id,
            "一致" if same_hash else "已变",
            "" if same_hash else "",
            stored_rows,
            len(items),
        )
        stats.reason = (
            f"库内行数 {stored_rows} 与应有 {len(items)} 不符"
            if same_hash else "content_hash 已变"
        )

    # 编码在事务之外：它可能耗时数秒（GPU），放进事务会让数据库长时间 idle-in-transaction。
    # 判定提到编码前，命中跳过时连编码都省掉。
    vectors, n_cached = _encode_for(items, cfg, use_cache=use_cache)

    with conn.transaction():
        cur = conn.cursor()
        # 先删旧片；documents 行保留，由下面的 upsert 更新。
        cur.execute("DELETE FROM chunks WHERE doc_id = %s", (stats.doc_id,))

        cur.execute(
            """
            INSERT INTO documents
                (doc_id, source_path, title, doc_type, category,
                 content_hash, meta, n_chunks, char_count, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s, now())
            ON CONFLICT (doc_id) DO UPDATE SET
                source_path  = EXCLUDED.source_path,
                title        = EXCLUDED.title,
                doc_type     = EXCLUDED.doc_type,
                category     = EXCLUDED.category,
                content_hash = EXCLUDED.content_hash,
                meta         = EXCLUDED.meta,
                n_chunks     = EXCLUDED.n_chunks,
                char_count   = EXCLUDED.char_count,
                updated_at   = now()
            """,
            (
                row["doc_id"],
                row["source_path"],
                row["title"],
                row["doc_type"],
                row["category"],
                fingerprint,
                row["meta"],
                stats.n_children,
                # char_count 只数子块：父块是子块的拼接，计入即双重计数。
                sum(len(c.content) for c in children),
            ),
        )

        write_chunks(cur, stats.doc_id, items, vectors=vectors)

    stats.n_vectors = len(vectors)
    stats.n_cached = n_cached
    logger.info(
        "%s %s：%d 片（子 %d / 父 %d / 表头 %d），编码 %d 条其中缓存命中 %d",
        status_label(stats.status), stats.doc_id, stats.n_rows,
        stats.n_children, stats.n_parents, stats.n_table_headers,
        stats.n_vectors, stats.n_cached,
    )
    return stats


# ------------------------------------------------------------------
# 全量
# ------------------------------------------------------------------
def _assert_manifest_covers(mf: Manifest, pdfs: Sequence[Path], raw_dir: Path) -> None:
    """磁盘上的 PDF 必须都在清单里；缺一即抛（fail-closed）。"""
    missing = missing_from_manifest(mf, pdfs, raw_dir)
    if missing:
        shown = "、".join(missing[:10])
        more = f" 等 {len(missing)} 份" if len(missing) > 10 else ""
        raise ManifestError(
            f"以下文件不在语料清单里：{shown}{more}。清单是身份与元数据的唯一来源，"
            f"缺一即拒绝入库（请补清单，或把这些文件移出语料目录）。清单：{mf.source}"
        )


def ingest_many(
    pdfs: Sequence[str | Path],
    cfg: AppConfig | None = None,
    *,
    conn: Any | None = None,
    cache_dir: str | Path | None = None,
    jobs: int = 1,
    force: bool = False,
    use_cache: bool = True,
    manifest: Manifest | None = None,
    raw_dir: str | Path | None = None,
    progress: Callable[[int, int, IngestStats], None] | None = None,
) -> list[IngestStats]:
    """端到端入库：解析 → 分块 → 编码 → 落库。

    语料元数据与文档身份一律由 ``manifest``（清单）裁决，**缺清单即失败**（fail-closed）：
    磁盘上有、清单里没有的文件会让整批在开跑前就报错，绝不"按文件名猜"。被准入拒绝的文档
    跳过并记原因、不建 ``documents`` 行；单份失败（解析异常 / 分块或写库抛错）只记一条
    ``failed``、**不中断整批**；``jobs > 1`` 只作用于解析阶段（编码批量在 ``embed.batch_size``）。

    解析与入库**逐份交错**（边收边写），不把全部文档一次收进内存；这也让解析（CPU 进程池）与
    编码（主进程 GPU）在时间上重叠。
    """
    from finrag.chunk.chunker import chunk_document
    from finrag.parse.cache import iter_parsed

    cfg = cfg or get_config()
    raw = Path(raw_dir) if raw_dir is not None else cfg.paths.resolve(cfg.paths.raw_dir)
    mf = manifest if manifest is not None else load_manifest(
        cfg.paths.resolve(cfg.paths.manifest), raw_dir=raw
    )

    pdf_list = [Path(p) for p in pdfs]
    _assert_manifest_covers(mf, pdf_list, raw)

    owned = conn is None
    conn = conn or connect(cfg)
    out: list[IngestStats] = []
    total = len(pdf_list)
    try:
        for i, (path, outcome) in enumerate(
            iter_parsed(pdf_list, cfg, cache_dir=cache_dir, use_cache=use_cache, jobs=jobs),
            1,
        ):
            record = mf.lookup(path, raw)
            if record is None:  # pragma: no cover —— 预校验与这里同步，正常到不了
                raise ManifestError(f"清单里没有 {path}（预校验未拦住？）")

            if not outcome.result.ok:
                # 解析失败 / 准入未过都由上游判定，这里只记录，不重新判一次。
                stats = IngestStats(
                    source_path=path,
                    doc_id=record.doc_id,
                    status="failed" if outcome.result.fatal else "rejected",
                    reason=outcome.result.fatal or outcome.result.admission.reason,
                )
            else:
                try:
                    chunked = chunk_document(outcome.result.doc, cfg)
                    stats = ingest_document(
                        conn, outcome.result.doc, chunked, record, cfg,
                        force=force, use_cache=use_cache,
                    )
                    # 分块警告只影响观测：挂到 stats 上，供报告列明细。不进任何指纹。
                    stats.warnings = list(chunked.warnings)
                    if chunked.warnings:
                        logger.info(
                            "%s 分块警告 %d 条：%s",
                            path.name, len(chunked.warnings), "；".join(chunked.warnings),
                        )
                except Exception as exc:  # noqa: BLE001 —— 单份失败不得中断整批
                    logger.warning(
                        "入库失败（不中断整批）：%s（%s: %s）",
                        path.name, type(exc).__name__, exc,
                    )
                    stats = IngestStats(
                        source_path=path,
                        doc_id=record.doc_id,
                        status="failed",
                        reason=f"{type(exc).__name__}: {exc}",
                    )
            out.append(stats)
            if progress is not None:
                progress(i, total, stats)
    finally:
        if owned:
            conn.close()
    return out


# ------------------------------------------------------------------
# 报告用的一层措辞
# ------------------------------------------------------------------
_STATUS_LABELS: Final[dict[str, str]] = {
    "written": "已写入",
    "skipped": "跳过（指纹一致）",
    "rejected": "拒绝入库（准入未过）",
    "failed": "失败（解析或入库异常）",
}


def status_label(status: str) -> str:
    """入库状态的展示写法；未知取值原样返回。"""
    return _STATUS_LABELS.get(status, status)