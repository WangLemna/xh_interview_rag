"""BGE-M3 编码器：一次前向同时产出 dense 与 sparse 两种表示，供双路召回使用。

缓存键为 ``sha256(模型标识 + 文本)``，命中与否只看输入变没变；``max_length``
与分块层共用同一个 tokenizer，此处仅作兜底上限。

向量缓存是 **SQLite 文件**（键值表），不是"一片一个 pickle" —— 数十万个小文件在 NTFS 上的
open/stat 开销远大于一次索引查找。缓存只是记忆化：删掉只会让下次编码慢，不改变任何产物。

**分为两个库**（见 :data:`CACHE_KIND_DOC` / :data:`CACHE_KIND_QUERY`）：文档块向量与查询
向量分开落文件。原因是查询是长尾一次性文本（线上每来一条新问句就 +1 行、单调增长），若与
文档库混在一起，会让"入库产物稳定"的文档库被无谓撑大；拆开后两者互不影响、可各自清理。
"""

from __future__ import annotations

import hashlib
import logging
import pickle
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Sequence

from finrag.config import AppConfig, get_config

__all__ = [
    "CACHE_KIND_DOC",
    "CACHE_KIND_QUERY",
    "ENCODER_VERSION",
    "EmbeddingSet",
    "cache_hit_count",
    "clear_cache",
    "encode_query",
    "encode_texts",
    "format_dense",
    "format_sparse",
    "get_encoder",
    "migrate_legacy_cache",
]

logger = logging.getLogger(__name__)

# 缓存文件格式版本，信封结构变化时才递增
CACHE_SCHEMA: Final[int] = 1

# 缓存子目录，与解析缓存并列
CACHE_SUBDIR: Final[str] = "embeddings"

# 输出约定（如 sparse 字面量格式）变更时手工升版；模型权重变化由模型标识负责
ENCODER_VERSION: Final[str] = "1"

# 加载锁：get_encoder 是进程级单例，并发首调会各建一份 2.2GB 权重
_LOAD_LOCK = threading.Lock()


@dataclass(frozen=True)
class EmbeddingSet:
    """一批文本的两种表示，必须来自同一次前向，避免两路各有一套看法。"""

    dense: list[Any]                      # 每项 ndarray[dim]，L2 归一化
    sparse: list[dict[int, float]]         # 每项 {token_id(1-based): weight}
    dim: int
    sparse_dim: int
    #: 这批里**命中缓存**的条数。由 :func:`encode_texts` 在过缓存时顺带数出 ——
    #: 于是调用方（``writer._encode_for``）不必再单独扫一遍缓存来报命中数（那是第二遍磁盘读）。
    n_cached: int = 0

    def __len__(self) -> int:
        return len(self.dense)

    def sparse_literals(self) -> list[str]:
        """全部稀疏向量转成可直接写库的 ``sparsevec`` 字面量，必须带 ``sparse_dim`` 后缀。"""
        return [format_sparse(w, self.sparse_dim) for w in self.sparse]

    def dense_literals(self) -> list[str]:
        """全部稠密向量转成可直接写库的 ``vector`` 字面量，格式与 :func:`format_sparse` 不同。"""
        return [format_dense(d) for d in self.dense]


def format_dense(vec: Any) -> str:
    """ndarray → ``vector`` 字面量（无后缀，形如 ``'[0.1,0.2]'``）；用 ``repr(float(x))`` 保留
    float32 的最短往返表示，避免截断污染余弦距离。
    """
    return "[" + ",".join(repr(float(x)) for x in vec) + "]"


def format_sparse(weights: dict[int, float], dim: int) -> str:
    """``{token_id: weight}`` → ``sparsevec`` 字面量（形如 ``'{1:0.1,5:2}/250002'``）；索引 1-based
    且必须带 ``/维度`` 后缀，``dim`` 不能由最大索引推断；空字典写作 ``'{}'``（非 ``'0'``），语义是"无词命中"。
    """
    if not weights:
        return f"{{}}/{int(dim)}"
    body = ",".join(f"{int(idx)}:{float(w):.6g}" for idx, w in sorted(weights.items()))
    return f"{{{body}}}/{int(dim)}"


# 已加载的编码器单例
_ENCODER: Any = None

# 词表大小缓存：tokenizer.json 有 50MB 量级，读到一次即缓存
_VOCAB_CACHE: dict[str, int] = {}


def _model_identity(cfg: AppConfig) -> str:
    """模型标识（模型名 + 维度），进缓存键；不能并入稀疏维度 —— 读它会碰 tokenizer，
    而该处也要用模型标识，会互相递归。
    """
    return f"{Path(cfg.embed.model_path).name}|d{cfg.embed.dim}"


def _sparse_dim(cfg: AppConfig) -> int:
    """稀疏维度 = BGE-M3 的词表大小，必须与 ``chunks.sparse_vec`` 的声明维度一致。"""
    key = f"{_model_identity(cfg)}|vocab"
    if key in _VOCAB_CACHE:
        return _VOCAB_CACHE[key]
    try:
        from finrag.chunk.tokens import resolve_tokenizer_file
        from tokenizers import Tokenizer

        tok = Tokenizer.from_file(str(resolve_tokenizer_file(cfg.embed.model_path)))
        size = int(tok.get_vocab_size())
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"读不到 BGE-M3 词表大小，无法确定 sparse_vec 维度：{exc}。"
            "它必须与 chunks.sparse_vec 的声明一致，否则报错会指向数据库、"
            "而真正的原因在模型侧。"
        ) from exc
    _VOCAB_CACHE[key] = size
    return size


def get_encoder(cfg: AppConfig | None = None) -> Any:
    """进程内单例的 ``BGEM3FlagModel``，加载 2.2GB 权重只应发生一次。"""
    cfg = cfg or get_config()
    global _ENCODER
    with _LOAD_LOCK:
        if _ENCODER is None:
            from FlagEmbedding import BGEM3FlagModel

            logger.info(
                "加载 BGE-M3 权重：%s（device=%s, fp16=%s）",
                cfg.embed.model_path, cfg.embed.device, cfg.embed.use_fp16,
            )
            _ENCODER = BGEM3FlagModel(
                cfg.embed.model_path,
                use_fp16=cfg.embed.use_fp16,
                device=cfg.embed.device,
            )
        return _ENCODER


# ------------------------------------------------------------------
# 缓存
# ------------------------------------------------------------------
def _cache_dir(cfg: AppConfig) -> Path:
    return cfg.embed.cache_path / CACHE_SUBDIR


def _text_key(cfg: AppConfig, text: str) -> str:
    """文本的缓存键 = ``CACHE_SCHEMA | ENCODER_VERSION | 模型标识 | 文本`` 的 sha256。

    ⚠ 键里**不含** ``embed.max_length`` / ``use_fp16`` / ``batch_size`` / 推理后端 ——
    改这些维度会改变向量，却不会让缓存失效。要改就必须先 :func:`clear_cache`（或递增
    :data:`ENCODER_VERSION`），否则会静默复用旧向量。
    """
    h = hashlib.sha256()
    h.update(f"{CACHE_SCHEMA}|{ENCODER_VERSION}|{_model_identity(cfg)}\n".encode("utf-8"))
    h.update(text.encode("utf-8"))
    return h.hexdigest()[:32]


#: 向量缓存分两个类别，落**不同文件**，避免"一次性查询"把文档库撑大：
#: - ``CACHE_KIND_DOC``：文档块向量（入库时写入，条目数与语料同量级、稳定）
#: - ``CACHE_KIND_QUERY``：查询向量（线上每来一条新问句就 +1 行，长尾、单调增长）
#: 两者缓存键算法相同，但互不干扰，可各自清理。
CACHE_KIND_DOC: Final[str] = "doc"
CACHE_KIND_QUERY: Final[str] = "query"

#: 各类别对应的库文件名。**文档库保持原文件名** ``vectors.sqlite``，兼容已有缓存 ——
#: 拆分不会让文档库失效、不触发重新编码；查询则另起一个文件。**单文件** SQLite 取代原先
#: "一片一个 ``.pkl``"的目录树（数十万个小文件）。WAL 模式允许多进程/多线程并发读、单写。
_CACHE_DB_NAMES: Final[dict[str, str]] = {
    CACHE_KIND_DOC: "vectors.sqlite",
    CACHE_KIND_QUERY: "vectors.query.sqlite",
}

#: 建表语句。``key`` 即 :func:`_text_key`；``dense`` / ``sparse`` 各存一份 pickle 字节。
_CACHE_DDL: Final[str] = (
    "CREATE TABLE IF NOT EXISTS vectors ("
    " key TEXT PRIMARY KEY,"
    " schema INTEGER NOT NULL,"
    " dense BLOB NOT NULL,"
    " sparse BLOB NOT NULL)"
)

#: 每线程一个连接（``sqlite3.Connection`` 默认不可跨线程共享，而服务端同步路由跑在
#: 线程池里 —— ``encode_query`` 会被多个线程调用）。线程本地省掉加锁，也避免
#: ``check_same_thread=False`` 下的并发写风险。``_DB_INIT_LOCK`` 只保护"建连接 + 建表"。
_TLS = threading.local()
_DB_INIT_LOCK = threading.Lock()


def _cache_db_path(cfg: AppConfig, kind: str = CACHE_KIND_DOC) -> Path:
    """某类别的向量缓存库路径（单个 SQLite 文件，落在原缓存目录下）。

    ``kind`` 取 :data:`CACHE_KIND_DOC` / :data:`CACHE_KIND_QUERY`；未知值直接报错，
    避免因笔误悄悄落到一个"野库"上。
    """
    try:
        name = _CACHE_DB_NAMES[kind]
    except KeyError:
        raise ValueError(
            f"未知的向量缓存类别：{kind!r}；可选 {sorted(_CACHE_DB_NAMES)}"
        ) from None
    return _cache_dir(cfg) / name


def _connect_cache(cfg: AppConfig, kind: str = CACHE_KIND_DOC) -> sqlite3.Connection:
    """取当前线程、指定类别的缓存连接；首次调用时建目录、建库、建表。

    连接按**库文件路径**分桶缓存在线程本地（``_TLS.conns``），故同一进程里可同时打开
    文档库与查询库、互不干扰 —— 拆库正是靠这一点，无需额外加锁。

    ``timeout=30`` 让并发写（入库进程 vs 服务进程）在锁上等待而非立刻报 ``database is locked``。
    """
    key = str(_cache_db_path(cfg, kind))
    conns = getattr(_TLS, "conns", None)
    if conns is None:
        conns = {}
        _TLS.conns = conns
    conn = conns.get(key)
    if conn is not None:
        return conn
    with _DB_INIT_LOCK:
        Path(key).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(key, timeout=30.0)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute(_CACHE_DDL)
        conn.commit()
    conns[key] = conn
    return conn


def _load_cached(
    cfg: AppConfig, text: str, kind: str = CACHE_KIND_DOC
) -> tuple[Any, dict[int, float]] | None:
    """读缓存。任何异常都退回 ``None``（当作未命中），绝不向上抛。"""
    key = _text_key(cfg, text)
    try:
        row = _connect_cache(cfg, kind).execute(
            "SELECT schema, dense, sparse FROM vectors WHERE key = ?", (key,)
        ).fetchone()
    except Exception as exc:  # noqa: BLE001
        logger.warning("向量缓存不可读，将重新编码（%s: %s）", type(exc).__name__, exc)
        return None
    if row is None or row[0] != CACHE_SCHEMA:
        return None
    try:
        dense = pickle.loads(row[1])
        sparse = pickle.loads(row[2])
    except Exception as exc:  # noqa: BLE001
        logger.warning("向量缓存条目损坏，将重新编码（%s: %s）", type(exc).__name__, exc)
        return None
    return dense, sparse


def _save_many_cached(
    cfg: AppConfig,
    rows: Sequence[tuple[str, Any, dict[int, float]]],
    *,
    kind: str = CACHE_KIND_DOC,
    batch: int = 1000,
) -> int:
    """批量写缓存：``rows = [(文本, dense, sparse), ...]``。一个事务写完，返回写入条数。

    ``kind`` 决定落到哪个库（文档库 / 查询库）。失败只记日志、**不抛**（缓存写失败不该影响
    本次编码结果）。单条写入走 :func:`_save_cached`。
    """
    if not rows:
        return 0
    try:
        conn = _connect_cache(cfg, kind)
        payload = [
            (key, CACHE_SCHEMA,
             pickle.dumps(dense, protocol=pickle.HIGHEST_PROTOCOL),
             pickle.dumps(sparse, protocol=pickle.HIGHEST_PROTOCOL))
            for _text, dense, sparse in rows
            for key in (_text_key(cfg, _text),)
        ]
        for start in range(0, len(payload), batch):
            conn.executemany(
                "INSERT OR REPLACE INTO vectors (key, schema, dense, sparse) VALUES (?, ?, ?, ?)",
                payload[start : start + batch],
            )
        conn.commit()
        return len(payload)
    except Exception as exc:  # noqa: BLE001
        logger.warning("向量缓存写入失败（不影响本次结果）：%s", exc)
        return 0


def _save_cached(
    cfg: AppConfig,
    text: str,
    dense: Any,
    sparse: dict[int, float],
    *,
    kind: str = CACHE_KIND_DOC,
) -> None:
    """写单条缓存（``INSERT OR REPLACE``）；热路径请用 :func:`_save_many_cached` 批量写。"""
    _save_many_cached(cfg, [(text, dense, sparse)], kind=kind)


def clear_cache(cfg: AppConfig | None = None, *, kind: str | None = None) -> int:
    """清空向量缓存，返回清除的条数（库内条目 + 顺带清掉的遗留 ``.pkl`` 文件）。

    ``kind=None``（默认）清**所有类别**（文档库 + 查询库）—— 改了模型 / 维度 / 精度
    （``use_fp16``）/ 截断长度（``max_length``）后两类缓存的向量都已失效，必须一起清。
    只想回收查询库（例如线上跑久了单独清理）时传 ``kind=CACHE_KIND_QUERY``。

    ⚠ 这些维度**不在缓存键里**（见 :func:`_text_key` 的说明），不清就会复用旧向量。
    """
    cfg = cfg or get_config()
    if kind is not None and kind not in _CACHE_DB_NAMES:
        raise ValueError(f"未知的向量缓存类别：{kind!r}；可选 {sorted(_CACHE_DB_NAMES)}")
    n = 0
    for k in ([kind] if kind is not None else list(_CACHE_DB_NAMES)):
        try:
            conn = _connect_cache(cfg, k)
            n += int(conn.execute("SELECT count(*) FROM vectors").fetchone()[0])
            conn.execute("DELETE FROM vectors")
            conn.commit()
        except Exception as exc:  # noqa: BLE001
            logger.warning("清空向量缓存库失败（%s）：%s", k, exc)

    # 兼容：把历史遗留的 ``*.pkl`` 目录一并清掉（迁移完成后通常已为空）。
    d = _cache_dir(cfg)
    if d.exists():
        for f in d.rglob("*.pkl"):
            try:
                f.unlink()
                n += 1
            except OSError:
                pass
    return n


def cache_hit_count(
    texts: Sequence[str], cfg: AppConfig | None = None, *, kind: str = CACHE_KIND_DOC
) -> int:
    """这批文本里有多少条现在就能命中缓存；须在编码之前调用才准。

    ⚠ 编码**热路径不再用它**：:func:`encode_texts` 已在过缓存时顺带数出命中数并放进
    :attr:`EmbeddingSet.n_cached`（原先在这里预扫一遍 = 对每条缓存读两遍）。本函数保留给
    脚本 / 自检使用。
    """
    cfg = cfg or get_config()
    return sum(1 for t in texts if _load_cached(cfg, t, kind) is not None)


def migrate_legacy_cache(cfg: AppConfig | None = None, *, batch: int = 1000) -> int:
    """把历史遗留的"一片一个 ``.pkl``"缓存迁进 SQLite 库，返回迁入条数。

    一次性动作：旧文件里存的信封是 ``{"schema", "dense", "sparse"}``，文件名 stem 即缓存键
    （:func:`_text_key` 的值）。**旧文件不删**（确认迁移无误后由调用方自行清理）。找不到旧
    文件时返回 ``0``。写库走 ``INSERT OR REPLACE``，可安全重复执行。

    迁入的是**文档库**（``.pkl`` 时代查询未单独缓存；即便偶有查询键，落到文档库也只是永不
    命中的死条目，无害）。
    """
    cfg = cfg or get_config()
    root = _cache_dir(cfg)
    if not root.exists():
        return 0
    conn = _connect_cache(cfg, CACHE_KIND_DOC)
    n = 0
    buf: list[tuple[str, int, bytes, bytes]] = []
    for p in root.rglob("*.pkl"):
        try:
            with p.open("rb") as fh:
                env = pickle.load(fh)
        except Exception:  # noqa: BLE001 —— 坏文件跳过，不中断迁移
            continue
        if not isinstance(env, dict) or env.get("schema") != CACHE_SCHEMA:
            continue
        dense, sparse = env.get("dense"), env.get("sparse")
        if dense is None or sparse is None:
            continue
        buf.append((
            p.stem, CACHE_SCHEMA,
            pickle.dumps(dense, protocol=pickle.HIGHEST_PROTOCOL),
            pickle.dumps(sparse, protocol=pickle.HIGHEST_PROTOCOL),
        ))
        if len(buf) >= batch:
            conn.executemany(
                "INSERT OR REPLACE INTO vectors (key, schema, dense, sparse) VALUES (?, ?, ?, ?)",
                buf,
            )
            conn.commit()
            n += len(buf)
            buf = []
    if buf:
        conn.executemany(
            "INSERT OR REPLACE INTO vectors (key, schema, dense, sparse) VALUES (?, ?, ?, ?)",
            buf,
        )
        conn.commit()
        n += len(buf)
    return n


# ------------------------------------------------------------------
# 编码
# ------------------------------------------------------------------
def _l2_normalize(vec: Any) -> Any:
    """L2 归一化，零向量原样返回（不除零）；归一化后模长恒为 1，零向量说明输入为空或全空白。"""
    import numpy as np

    norm = float(np.linalg.norm(vec))
    if norm == 0.0:
        return vec
    return vec / norm


def encode_texts(
    texts: Sequence[str],
    cfg: AppConfig | None = None,
    *,
    use_cache: bool = True,
    batch_size: int | None = None,
    progress: Any = None,
    cache_kind: str = CACHE_KIND_DOC,
) -> EmbeddingSet:
    """编码一批文本，返回稠密 + 稀疏两种表示；``use_cache=False`` 时不读也不写，
    且不要在外层再按批切片，否则缓存命中会错位。

    ``cache_kind`` 决定读写哪个缓存库：文档块走默认的 :data:`CACHE_KIND_DOC`，查询走
    :data:`CACHE_KIND_QUERY`（由 :func:`encode_query` 自动指定）。两端**必须一致**，
    否则会出现"这一侧写了、那一侧读不到"的空命中。
    """
    cfg = cfg or get_config()
    items = list(texts)
    if not items:
        return EmbeddingSet(dense=[], sparse=[], dim=cfg.embed.dim,
                            sparse_dim=_sparse_dim(cfg))

    dense_out: list[Any] = []
    sparse_out: list[dict[int, float]] = []

    # ---- 先过一遍缓存，剩下的才真编码 ----
    # 命中数在这里顺带数出（B1）：不再由调用方单独预扫一遍缓存。
    n_cached = 0
    todo: list[int] = []
    for i, text in enumerate(items):
        if use_cache:
            hit = _load_cached(cfg, text, cache_kind)
            if hit is not None:
                dense_out.append(hit[0])
                sparse_out.append(hit[1])
                n_cached += 1
                continue
        dense_out.append(None)  # 占位，下面统一填
        sparse_out.append({})
        todo.append(i)

    if todo:
        encoder = get_encoder(cfg)
        # return_sparse 必须显式为 True：默认 False 会让 lexical_weights 为 None 且不报错
        raw = encoder.encode(
            [items[i] for i in todo],
            batch_size=batch_size or cfg.embed.batch_size,
            max_length=cfg.embed.max_length,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,  # 本项目不用 late interaction，省一份显存
        )
        lex = raw.get("lexical_weights")
        if lex is None:
            raise RuntimeError(
                "BGE-M3 未产出 lexical_weights。encode() 的 return_sparse 必须为 True —— "
                "默认 False 会返回 None 且不报错。检查是否被中间层改写了参数。"
            )
        to_save: list[tuple[str, Any, dict[int, float]]] = []
        for slot, i in enumerate(todo):
            d = _l2_normalize(raw["dense_vecs"][slot])
            s = {int(k): float(v) for k, v in lex[slot].items()}
            dense_out[i] = d
            sparse_out[i] = s
            if use_cache:
                to_save.append((items[i], d, s))
        if to_save:
            # 一批（本文档）一个新算的向量，**一个事务**写完，避免逐条提交。
            _save_many_cached(cfg, to_save, kind=cache_kind)
        if progress is not None:
            progress(len(items), len(items), len(todo))

    return EmbeddingSet(
        dense=dense_out, sparse=sparse_out,
        dim=cfg.embed.dim, sparse_dim=_sparse_dim(cfg), n_cached=n_cached,
    )


def encode_query(text: str, cfg: AppConfig | None = None) -> EmbeddingSet:
    """编码一条查询，与批量入库共用同一个函数，但**落查询专属的缓存库**。

    查询文本是一次性长尾：线上每来一条新问句就新增一行。若与文档块混在同一个库里，
    会让"入库产物稳定"的文档库被无谓撑大；故此处显式走 :data:`CACHE_KIND_QUERY`，
    与文档库分离，可单独清理。
    """
    return encode_texts([text], cfg, cache_kind=CACHE_KIND_QUERY)