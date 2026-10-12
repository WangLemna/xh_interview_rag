"""结构感知分块：把解析层的 IR 切成扁平的检索单元（:class:`~finrag.parse.ir.ParsedDoc` → chunks）。

不碰数据库、不加载编码模型，只依赖可注入的 token 计数函数。切分走取块、按节累积、
超长块硬切、合并碎片四阶段，再配父块与表头片并编号；只有硬切带 overlap。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Iterable, Iterator, NamedTuple

from finrag.chunk.tokens import CountTokens, bge_m3_counter
from finrag.config import AppConfig, ChunkConfig, get_config
from finrag.parse import layout as L
from finrag.parse.ir import Block, ParsedDoc

__all__ = [
    "HEADING_SEP",
    "ROLE_CHILD",
    "ROLE_PARENT",
    "Chunk",
    "ChunkResult",
    "chunk_document",
]

logger = logging.getLogger(__name__)

# heading_path 入库时的拼接分隔符，口径只定义这一处：writer 直接拿
# ``Chunk.heading_text`` 写进 ``chunks.heading_path``。
HEADING_SEP: Final[str] = " > "

# 块的角色，落进 ``chunks.meta.role``。检索期据此把父块排除出召回 ——
# 父块只用来给命中的子块补上下文，自身不参与排名。
ROLE_CHILD: Final[str] = "child"
ROLE_PARENT: Final[str] = "parent"

# 硬切切点的性质，"有损"与否跟着切分当场记下，不事后从文本反推（见 _Piece.split_kind）。
KIND_CHAR: Final[str] = "char"          # 切在文本中间 —— 有损
KIND_SENTENCE: Final[str] = "sentence"  # 切在句末标点上 —— 无损
KIND_LINE: Final[str] = "line"          # 切在行边界上 —— 无损，但有两个来源
# KIND_LINE 的两个来源"无损"成色不同：结构化表格的 markdown 行（一行 = 一条记录）
# 是真无损；经 ``_rebuild_text`` 重建的换行是几何猜的（容差 1.0×字号），多分行处属误报无损。

# 唯一需要分块层特殊对待的块类型。表格的语义单元是"行 / 记录"，单元格里的句号
# 不构成边界，故不能对它按句切（否则"有损"会变成"误报无损"）。本常量只约束
# "表格的单个不可分行"不按句切，结构化表的 markdown 行仍可无损地按行切。
KIND_TABLE: Final[str] = "table"

# 句边界吸附用的句末标点。只收全角：半角 "." / ";" 在年报里高频出现在小数点
# 与编号缩写里，拿它当句末会把切点放进数字中间。
SENTENCE_ENDS: Final[str] = "。！？；"

# 句边界吸附的下限：句末标点必须落在字符二分切点的这个比例之后才采纳，
# 否则一个靠左的句号会把片切成碎片，把压力推给合并阶段。
SENTENCE_SNAP_MIN_RATIO: Final[float] = 0.5


@dataclass(slots=True)
class Chunk:
    """一个检索单元，字段与 ``chunks`` 表的列一一对应（向量列由编码层填）。
    ``heading_path`` 存 tuple 以便展示层拆深路径（入库的一维形式见 :attr:`heading_text`）。
    """

    content: str
    chunk_index: int
    heading_path: tuple[str, ...]
    page_from: int
    page_to: int
    token_count: int
    role: str = ROLE_CHILD
    # 指向同文档内回溯目标的 chunk_index，而非数据库的 ``chunks.id``（后者入库时才生成）。
    # 目标可能是节父块（上下文回溯）或表头片（列名回溯），两者都排在子块之前。
    # 注意它未必指向父块：表头片自己是 child、要参与排名。
    parent_index: int | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def heading_text(self) -> str | None:
        """入库用的一维路径。空路径返回 ``None``：NULL 表示内容不属于任何标题之下。"""
        return HEADING_SEP.join(self.heading_path) if self.heading_path else None


@dataclass(slots=True)
class ChunkResult:
    """一份文档的分块结果。统计量不预计算成字段，只提供按角色筛选的辅助方法。"""

    source_path: Path
    chunks: list[Chunk] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def children(self) -> list[Chunk]:
        return [c for c in self.chunks if c.role == ROLE_CHILD]

    def parents(self) -> list[Chunk]:
        return [c for c in self.chunks if c.role == ROLE_PARENT]

    def table_headers(self) -> list[Chunk]:
        """表头片：``role`` 是 child（参与排名），只是额外被同表数据片指为回溯目标。
        判别看 ``meta['table_header']`` 而非 ``role``。
        """
        return [c for c in self.chunks if c.meta.get("table_header")]

    @property
    def total_tokens(self) -> int:
        """子块 token 总量，不含父块（父块是子块的拼接，计入即双重计数）。"""
        return sum(c.token_count for c in self.children())


class _Item(NamedTuple):
    """预处理后的一个 IR 块。只留切分需要的字段。"""

    heading_path: tuple[str, ...]
    text: str
    page: int
    kind: str
    # 同表标记，只有结构化表格块有值，段落块恒为空串。由本层按块流顺序分配
    # （第 N 个结构化表格块 → "tN"），不改 IR 契约。用途两条：表片不参与合并；
    # 让"同表"成为可判定的键，供检索取整表、供表头片把同表的片归组。
    # 未结构化的表格块拿不到它 —— 那类形态下"一张表 = 多个块"，本层无从归组。
    table_id: str = ""


@dataclass(slots=True)
class _Piece:
    """切分中途的一片：还没编号、也还没配父块。"""

    text: str
    heading_path: tuple[str, ...]
    page_from: int
    page_to: int
    token_count: int
    n_blocks: int = 0
    is_split: bool = False
    # 硬切落点："" 片内无断点 / "char" 切在文本中间（有损）/ "sentence" 切在句末
    # 标点上（无损）/ "line" 切在行边界上（无损）。只有 "char" 会丢语义单元。
    # 它是"有损"的准确判据，必须跟着切分当场记下，不事后从文本反推。
    # "line" 有两个来源：结构化表格的 markdown 行（一行 = 一条记录）是真无损；
    # ``_rebuild_text`` 重建的换行是几何猜的（容差 1.0×字号），多分行处属误报无损。
    split_kind: str = ""
    # 片主体来自哪种 IR 块，用于分清"表格数据被切开"与"长段落被切开"（两者修法不同）。
    # 取值 "table" 的含义是"片主体落在某张表格的矩形内"，不是"这片就是那张表" ——
    # get_text("dict") 把表格每一行切成一个块，一张表 = 多个块。
    # 口径上限：标不上的表格块仍记 "paragraph"，故它是 find_tables 认下的成分。
    source_kind: str = ""
    # 同表标记，取自 ``_Item.table_id``。非空即断言"这一片就是那一张表"：
    # 硬切片直接继承；累积片仅在该片只含一个块时继承，否则记空串
    # （见 ``_try_merge``：带标记的片不进入任何合并结果）。
    table_id: str = ""


# ------------------------------------------------------------------
# 主入口
# ------------------------------------------------------------------
def chunk_document(
    doc: ParsedDoc,
    cfg: AppConfig | None = None,
    count_tokens: CountTokens | None = None,
) -> ChunkResult:
    """把一份 IR 切成检索单元。
    ``count_tokens`` 缺省按 BGE-M3 tokenizer 计数，长度须与编码器对齐，勿在生产传别的实现。
    """
    cfg = cfg or get_config()
    ccfg = cfg.chunk
    count = count_tokens or bge_m3_counter(cfg.embed.model_path)
    warnings: list[str] = []

    items = list(_iter_items(doc, warnings))
    if not items:
        warnings.append("没有可切分的内容块（整份文档可能只含图片占位块）")
        return ChunkResult(source_path=doc.source_path, warnings=warnings)

    pieces = _split_sections(items, ccfg, count)
    pieces = _merge_fragments(pieces, ccfg, count)
    chunks = _assemble(pieces, ccfg, count, warnings)

    # 与解析层的低置信门控呼应，但要分清**标题层级究竟从哪来**：只有真走了版式兜底
    # （``has_toc=False``）才谈得上"标题不可信"；有 TOC 时标题来自书签、与字号无关，
    # 此时 body_share 低只说明字号分布偏斜 —— 不能照抄"来自版式兜底"（那会误报）。
    # 阈值从配置读，不在此重写一个常量；本块只记 warning、不改变切分行为，
    # 供下游（评测分组、人工复核）筛出低置信文档。
    body_share = doc.meta.body_share
    if body_share and body_share < cfg.parse.min_body_share:
        if doc.meta.has_toc:
            warnings.append(
                f"字号分布偏斜：body_share = {body_share:.2f} "
                f"< {cfg.parse.min_body_share}（标题层级来自 TOC，未受影响）"
            )
        else:
            warnings.append(
                f"heading 可信度低：body_share = {body_share:.2f} "
                f"< {cfg.parse.min_body_share}，无 TOC，标题层级来自版式兜底 / 虚拟根"
            )

    return ChunkResult(source_path=doc.source_path, chunks=chunks, warnings=warnings)


# ------------------------------------------------------------------
# 阶段 1：取块
# ------------------------------------------------------------------
def _iter_items(doc: ParsedDoc, warnings: list[str]) -> Iterator[_Item]:
    """IR 块流 → 待切分文本流：跳过 image 占位块与空文本块，表格形态的块按行几何重建。"""
    n_images = 0
    n_blank = 0
    n_tables = 0
    for heading_path, block in doc.iter_blocks():
        if block.kind == "image":
            n_images += 1
            continue
        text = _rebuild_text(block, warnings)
        if not text:
            n_blank += 1
            continue
        # 判别结构化表格看 ``block.table is not None``，不能看 kind（两种形态的 kind
        # 都是 "table"）。序号按块流顺序分配 ⇒ 同一份 IR 每次得到的序号相同。
        table_id = ""
        if block.table is not None:
            n_tables += 1
            table_id = f"t{n_tables}"
        yield _Item(heading_path, text, block.page, block.kind, table_id)

    if n_blank:
        # 意味着解析层扫到了既无内容也无表题的表格块。
        warnings.append(f"丢弃 {n_blank} 个空文本块")
    logger.debug(
        "%s：跳过 %d 个图片占位块，留下 %d 个待切分块",
        doc.source_path.name,
        n_images,
        doc.block_count() - n_images,
    )


def _block_text(block: Block) -> str:
    """块的待切分文本；表题与表体用换行连接、表题独立成行（表题分支当前不可达）。"""
    return _with_caption(block, block.text.strip())


def _with_caption(block: Block, body: str) -> str:
    """把表题拼到 ``body`` 前面（换行连接）；分支当前不可达（``Block.caption`` 恒为 ``None``）。"""
    if block.kind == "table" and block.caption:
        cap = block.caption.strip()
        return f"{cap}\n{body}" if body else cap
    return body


def _squeezed_len(text: str) -> int:
    """去掉所有空白后的字符数 —— 重建守门用的量。"""
    return len("".join(text.split()))


def _rebuild_text(block: Block, warnings: list[str]) -> str:
    """块的待切分文本：对"表格形态"的块按行几何重建（容差 1.0×字号），其余原样返回。
    仅在 ``has_glued_cells`` 命中时重建，只插分隔符不丢字符，去空白字符数不符即退回并记 warning。
    """
    lines = block.lines
    if not lines or not L.has_glued_cells(lines):
        return _block_text(block)
    tol = L.row_cluster_tol(block.font_size)
    body = "\n".join(
        " ".join(line.text for line in cluster) for cluster in L.y_clusters(lines, tol)
    ).strip()
    # 守门：去空白后的字符数必须与原文一致，不等说明重建丢了内容（比粘连严重得多），故直接降级。
    if _squeezed_len(body) != _squeezed_len(block.text):
        warnings.append(
            f"行重建后字符数不符（{_squeezed_len(body)} != "
            f"{_squeezed_len(block.text)}）⇒ 退回原文本"
        )
        return _block_text(block)
    return _with_caption(block, body)


# ------------------------------------------------------------------
# 阶段 2 + 3：分节累积与硬切
# ------------------------------------------------------------------
def _split_sections(items: list[_Item], ccfg: ChunkConfig, count: CountTokens) -> list[_Piece]:
    """同标题下的块顺序累积、装不下就换片，单块超长才硬切。
    累积整块进出；唯一例外是结构化表格块（``chunk.table_units``）独占片以保住 table_id。
    """
    pieces: list[_Piece] = []
    cur: list[_Item] = []
    cur_tokens = 0
    cur_path: tuple[str, ...] | None = None

    def flush() -> None:
        nonlocal cur, cur_tokens
        if cur:
            pieces.append(_make_piece(cur, count))
            cur = []
            cur_tokens = 0

    for item in items:
        # respect_headings=False 时忽略标题边界，只按长度切（消融项）。
        # 表格的独占片不受本开关影响：片边界与表边界是一回事，与标题边界无关。
        if ccfg.respect_headings and item.heading_path != cur_path:
            flush()

        t = count(item.text)
        # 结构化表格块独占片：片边界须与表边界一致，否则"这张表"在片这一级拿不到
        # table_id。代价是片数上升。它不是"表格比段落重要"，而是"一张表的片得能被辨认出来"。
        exclusive = ccfg.table_units and bool(item.table_id)

        if exclusive or t > ccfg.max_tokens:
            # 先落片、不更新 cur_path：当前块仍属当前节，下一块若同节还要接着累积。
            flush()
            if t > ccfg.max_tokens:
                # 把块类型一路传下去：切点的"有无损"取决于块是什么 ——
                # 句末标点对段落是语义边界，对表格不是（见 KIND_TABLE）。
                # ``structured_table`` 用于补表头，不能拿 ``source_kind`` 反推
                # （kind 对结构化与未结构化两种形态都成立，而只有结构化表 text 首行才是列名）。
                for txt, mode in _hard_split(
                    item.text, ccfg, count, item.kind,
                    structured_table=bool(item.table_id),
                ):
                    pieces.append(
                        _Piece(
                            text=txt,
                            heading_path=item.heading_path,
                            page_from=item.page,
                            page_to=item.page,
                            token_count=count(txt),
                            n_blocks=1,
                            is_split=True,
                            split_kind=mode,
                            source_kind=item.kind,
                            # 超长表格块切出来的每一片都属同一张表 —— 直接继承。
                            table_id=item.table_id,
                        )
                    )
            else:
                # 独占片：只有这一个块，``_make_piece`` 会照常认领它的 table_id。
                pieces.append(_make_piece([item], count))
            cur_path = item.heading_path
            continue

        if cur and cur_tokens + t > ccfg.max_tokens:
            flush()

        cur.append(item)
        cur_tokens += t
        cur_path = item.heading_path

    flush()
    return pieces


def _make_piece(items: list[_Item], count: CountTokens) -> _Piece:
    """把攒下的一批块落成一片。
    ``token_count`` 重新数（块间换行也占 token，累加会低报）；页区间取 min/max 而非首末块。
    """
    text = "\n".join(it.text for it in items)
    pages = [it.page for it in items]
    return _Piece(
        text=text,
        heading_path=items[0].heading_path,
        page_from=min(pages),
        page_to=max(pages),
        token_count=count(text),
        n_blocks=len(items),
        # 认领同表标记的唯一条件：这一片恰好只含一个块（混合片不是"某一张表"）。
        table_id=items[0].table_id if len(items) == 1 else "",
    )


def _hard_split(
    text: str, ccfg: ChunkConfig, count: CountTokens, source_kind: str = "paragraph",
    *, structured_table: bool = False,
) -> list[tuple[str, str]]:
    """把单个超长块切成多片，返回 ``[(片文本, 切分方式)]``，方式是切点性质（见 KIND_*）。
    唯一会切断语义单元的切分，故独占 overlap；预算取 ``max_tokens - overlap_tokens``。
    """
    budget = max(1, ccfg.max_tokens - ccfg.overlap_tokens)

    # 逐行标好切点性质，这个信息只在切的那一刻拿得到。
    # 循环体跑几次取决于是哪类块：段落块 / 未结构化表格块文本无换行，只跑一次；
    # 结构化表格块按行数循环，每行各自判是否超预算 —— 这就是 ``line`` 从不可达变可达的机理。
    lines: list[tuple[str, str]] = []
    for raw in text.split("\n"):
        if count(raw) > budget:
            lines.extend(
                _split_long_line(raw, budget, count, source_kind, ccfg.split_on_sentence)
            )
        else:
            lines.append((raw, KIND_LINE))

    pieces: list[tuple[str, str]] = []
    cur: list[tuple[str, str]] = []
    cur_tokens = 0

    for line, _mode in lines:
        t = count(line)
        if cur and cur_tokens + t > budget:
            pieces.append(("\n".join(x for x, _ in cur), _worse_split(m for _, m in cur)))
            # headroom = 新片装下当前行后还能挪给回带的余量；不给约束会让新片一开头
            # 就顶到预算线上，紧接着 append 的当前行无处安放（越界会被静默截断）。
            cur = _carry(cur, ccfg.overlap_tokens, budget - t, count)
            cur_tokens = sum(count(x) for x, _ in cur)
        cur.append((line, _mode))
        cur_tokens += t

    if cur:
        pieces.append(("\n".join(x for x, _ in cur), _worse_split(m for _, m in cur)))

    # 补表头只对结构化表格块做，不能拿 ``source_kind`` 反推（重建块首行未必是列名）。
    if ccfg.repeat_header and structured_table:
        pieces = _prepend_table_header(pieces, ccfg, count)
    return pieces


def _prepend_table_header(
    pieces: list[tuple[str, str]], ccfg: ChunkConfig, count: CountTokens
) -> list[tuple[str, str]]:
    """给被切开的表格片的非首片补上 markdown 首行（表头行），只用 text 首行、不用 ``TableData.header``。
    补完超 ``max_tokens`` 就按行递归再切（判据用 ``max_tokens`` 而非硬切的 ``budget``）。
    """
    if len(pieces) < 2:
        return pieces
    header = pieces[0][0].split("\n")[0]
    if not header.strip():
        return pieces

    out: list[tuple[str, str]] = [pieces[0]]
    for text, kind in pieces[1:]:
        out.extend(_fit_with_header(text, kind, header, ccfg, count))
    return out


def _fit_with_header(
    text: str,
    kind: str,
    header: str,
    ccfg: ChunkConfig,
    count: CountTokens,
) -> list[tuple[str, str]]:
    """给一片补上列头，装不下就按行递归再切，返回可能多于一片。
    切点性质取更严重者（:func:`_worse_split`），不把有损报成无损。
    """
    cand = f"{header}\n{text}"
    if count(cand) <= ccfg.max_tokens:
        # ``kind`` 原样带过：补列头是无损的，降级会让"有损"的观测失真。
        return [(cand, kind)]

    lines = text.split("\n")
    if len(lines) >= 2:
        # 预算扣掉列头，它自己也占 token。
        per = max(1, ccfg.max_tokens - count(header) - 1)
        cur: list[str] = []
        cur_tok = 0
        out: list[tuple[str, str]] = []
        for line in lines:
            t = count(line)
            if cur and cur_tok + t > per:
                out.append(("\n".join(cur), kind))
                cur, cur_tok = [], 0
            cur.append(line)
            cur_tok += t
        if cur:
            out.append(("\n".join(cur), kind))
    else:
        # 单行片带列头后可能超预算（宽表的长数据行），按字符再切而不是放弃补列头。
        # ``on_sentence=False``：表格的语义单元是行而非句子（同 ``_hard_split`` 对 KIND_TABLE 的处理）。
        segs = _split_long_line(
            text,
            max(1, ccfg.max_tokens - count(header) - 1),
            count,
            KIND_TABLE,
            False,
        )
        out = [
            (seg, _worse_split((kind, m))) for seg, m in segs
        ]

    # 对新切出的每一片都要再走一次本函数，否则列头只补到第一层。
    res: list[tuple[str, str]] = []
    for t, k in out:
        res.extend(_fit_with_header(t, k, header, ccfg, count))
    return res


def _worse_split(kinds: Iterable[str]) -> str:
    """取更严重的切分方式：``char`` > ``sentence`` > ``line`` > 未切（漏报有损比误报危险）。"""
    kinds = tuple(kinds)
    if KIND_CHAR in kinds:
        return KIND_CHAR
    if KIND_SENTENCE in kinds:
        return KIND_SENTENCE
    if KIND_LINE in kinds:
        return KIND_LINE
    return ""


def _carry(
    prev: list[tuple[str, str]], overlap: int, headroom: int, count: CountTokens
) -> list[tuple[str, str]]:
    """上一片末尾可回带的单元，连同切分标记；``headroom`` = budget - 当前单元 token，回带总量须落在其中。
    首单元只受 ``headroom`` 限制不受 ``overlap`` 限制；单单元连 headroom 都塞不下就整个不带。
    """
    if overlap <= 0 or headroom <= 0:
        return []
    limit = min(overlap, headroom)
    out: list[tuple[str, str]] = []
    total = 0
    for line, mode in reversed(prev):
        t = count(line)
        if t > headroom:
            break
        if out and total + t > limit:
            break
        out.insert(0, (line, mode))
        total += t
    return out


def _split_long_line(
    line: str,
    budget: int,
    count: CountTokens,
    source_kind: str = "paragraph",
    on_sentence: bool = True,
) -> list[tuple[str, str]]:
    """单行超预算时的硬切，返回 ``[(片文本, 切点性质)]``；切点优先吸附句末标点，挪不动才转 ``KIND_CHAR``。
    表格块不做句切（语义单元是行）；吸附下标有下限 ``SENTENCE_SNAP_MIN_RATIO``，标点太靠左不采纳。
    """
    out: list[tuple[str, str]] = []
    rest = line
    last_kind = ""
    while count(rest) > budget:
        cut = max(1, _cut_at(rest, budget, count))
        kind = KIND_CHAR
        if on_sentence and source_kind != KIND_TABLE:
            snapped = _snap_to_sentence(rest[:cut], int(cut * SENTENCE_SNAP_MIN_RATIO))
            if snapped is not None:
                cut = snapped
                kind = KIND_SENTENCE
        last_kind = kind
        out.append((rest[:cut], kind))
        rest = rest[cut:]
    if rest:
        # 尾段内部没有断点，但它的起点就是刚切出来的那个断点，故按最近一次切点的性质
        # 标注；不标 ""，否则一片真正的 lossy 数据会在统计里显得干净。
        out.append((rest, last_kind))
    return out


def _snap_to_sentence(prefix: str, floor: int) -> int | None:
    """在 ``prefix`` 里找最后一个句末标点，返回其后下标（即切点）；无可用句边界返回 ``None``。
    ``floor`` 是切点的最小下标，标点落得更靠左就不采纳。
    """
    for i in range(len(prefix) - 1, floor - 1, -1):
        if prefix[i] in SENTENCE_ENDS:
            return i + 1
    return None


def _cut_at(text: str, budget: int, count: CountTokens) -> int:
    """二分找切点下标，使前缀的 token 数不超过 ``budget``。
    BPE 计数对前缀不严格单调，故不保证最长；只在 ``count(前缀) <= budget`` 时推进，结果绝不超预算。
    """
    lo, hi = 1, len(text)
    best = 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if count(text[:mid]) <= budget:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return best


# ------------------------------------------------------------------
# 阶段 4：合并碎片
# ------------------------------------------------------------------
def _merge_fragments(
    pieces: list[_Piece], ccfg: ChunkConfig, count: CountTokens
) -> list[_Piece]:
    """把低于 ``min_tokens`` 的片向右吸收相邻片，直到够长或不能再并；合并后 ``heading_path`` 退到共同前缀。
    达 ``min_tokens`` 即停，不填满 ``max_tokens``（否则小节会滚成大片、路径退到根）。
    """
    out: list[_Piece] = []
    i = 0
    n = len(pieces)

    while i < n:
        cur = pieces[i]
        while cur.token_count < ccfg.min_tokens and i + 1 < n:
            merged = _try_merge(cur, pieces[i + 1], ccfg, count)
            if merged is None:
                break
            cur = merged
            i += 1
        out.append(cur)
        i += 1

    return out


def _try_merge(
    cur: _Piece, nxt: _Piece, ccfg: ChunkConfig, count: CountTokens
) -> _Piece | None:
    """两片合并的条件与结果。不满足条件返回 ``None`` —— 保持原样胜过强行合。"""
    # 带同表标记的片不参与合并：粘上邻居后就不再指向某一张表了（table_id 被清空）。
    # ``table_units`` 开时更严，一律不合；关时退回只禁跨表。代价是小表碎片数上升，有意为之。
    if cur.table_id or nxt.table_id:
        if ccfg.table_units:
            return None
        if cur.table_id and nxt.table_id and cur.table_id != nxt.table_id:
            return None

    common = _common_prefix(cur.heading_path, nxt.heading_path)

    # 跨节合并需要非空的共同祖先来承载 heading_path：共同前缀为空意味着两片分属
    # 不同的一级章节，合并后无路径可标，不如让它保持独立。同节合并不受此限。
    if not common and cur.heading_path != nxt.heading_path:
        return None

    text = f"{cur.text}\n{nxt.text}"
    tokens = count(text)
    if tokens > ccfg.max_tokens:
        # 消碎片而超上限是拆东墙补西墙：超上限会被编码器静默截断尾巴（即刚并进来的那片）。
        return None

    return _Piece(
        text=text,
        heading_path=common or cur.heading_path,
        page_from=min(cur.page_from, nxt.page_from),
        page_to=max(cur.page_to, nxt.page_to),
        token_count=tokens,
        n_blocks=cur.n_blocks + nxt.n_blocks,
        is_split=cur.is_split or nxt.is_split,
        split_kind=_worse_split((cur.split_kind, nxt.split_kind)),
        source_kind=cur.source_kind or nxt.source_kind,
        # 两侧标记相同才保留（同一张表的两个片被并到一起），否则清空 —— 合并后的片不再指向某一张表。
        table_id=cur.table_id if cur.table_id == nxt.table_id else "",
    )


def _common_prefix(a: tuple[str, ...], b: tuple[str, ...]) -> tuple[str, ...]:
    """两个标题路径的共同前缀。"""
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return a[:n]


# ------------------------------------------------------------------
# 阶段 5：配父块与编号
# ------------------------------------------------------------------
def _assemble(
    pieces: list[_Piece], ccfg: ChunkConfig, count: CountTokens, warnings: list[str]
) -> list[Chunk]:
    """配父块与表头片，然后全局连续编号。父块排在子块之前（writer 单趟插入靠此拿到 parent_id）。

    父块 = 一个标题节，节切成 ≥2 片才配。节内总量 ≤ ``chunk.parent_max_tokens`` → 整节一个父块；
    超预算 → **按片数窗口拆成多个父块**（相邻窗口重叠 ``chunk.parent_window_overlap`` 片），
    每个窗口另受 ``parent_max_tokens`` 的 token 上限兜底。不用改写入层：父块依旧全部排在子块之前，
    ``writer`` 靠同文档内 ``chunk_index`` 解析 ``parent_id``。
    """
    groups: dict[tuple[str, ...], list[_Piece]] = {}
    for p in pieces:
        groups.setdefault(p.heading_path, []).append(p)

    # 按表归集片。``pieces`` 的顺序就是阅读顺序，``group[0]`` 恒为含表头行的首片
    # （表头片的内容直接从它取）；dict 的插入顺序即 table_id 首次出现的顺序。
    # 关掉 ``table_units`` 时这里为空（回退开关须整份退回）。
    table_of: dict[str, list[_Piece]] = {}
    if ccfg.table_units:
        for p in pieces:
            if p.table_id:
                table_of.setdefault(p.table_id, []).append(p)

    # table_id → (表头行, 首片, 数据片数)。只有被切成 ≥2 片的表进来。
    headers: dict[str, tuple[str, _Piece, int]] = {}
    for tid, group in table_of.items():
        if len(group) >= 2:
            headers[tid] = (group[0].text.split("\n", 1)[0], group[0], len(group))

    # 表头片按它所属的节归集：父块的 ``n_children`` 要把挂上来的表头片算进去。
    headers_by_path: dict[tuple[str, ...], list[_Piece]] = {}
    for _line, first, _n in headers.values():
        headers_by_path.setdefault(first.heading_path, []).append(first)

    chunks: list[Chunk] = []
    # 片 → 父块 chunk_index。用 id(_Piece) 作键：``_Piece`` 是可变 dataclass（不可哈希），
    # 而这些对象在整个 pieces 生命周期内都活着，id 稳定。
    piece_parent: dict[int, int] = {}
    n_windowed = 0
    n_window_parents = 0

    for path, group in groups.items():
        if len(group) < 2:
            continue
        hdr_pieces = headers_by_path.get(path, [])
        section_tokens = count("\n".join(p.text for p in group))
        windowed = bool(ccfg.parent_max_tokens and section_tokens > ccfg.parent_max_tokens)
        segs = _section_windows(group, ccfg) if windowed else [group]

        # 一遍前向：决定哪些窗口真正留得下。判据是"**至少有一个子块会指向它**"——
        # 不能拿"窗口里有几片"当判据：表数据片只要其表被切成 ≥2 片就改挂表头片
        # （列名回溯比整节上下文更具体），于是"只含表数据片"的窗口发出来就是没人指向的空父块。
        # 留不住的窗口**不认领**、把片让给后面的窗口（重叠保证后续窗口仍覆盖得到）。
        surviving: list[tuple[list[_Piece], int, int]] = []  # (窗口, 片儿子数, 表头儿子数)
        claimed: set[int] = set()
        for seg in segs:
            fresh = [p for p in seg if id(p) not in claimed]
            if not fresh:
                continue
            fresh_ids = {id(p) for p in fresh}
            n_piece = sum(1 for p in fresh if _uses_section_parent(p, headers))
            n_hdr = sum(1 for fp in hdr_pieces if id(fp) in fresh_ids)
            if n_piece + n_hdr == 0:
                continue
            claimed.update(fresh_ids)
            surviving.append((seg, n_piece, n_hdr))

        # 重叠区里的片归**第一个**认领它的窗口（先到先得），归属唯一且确定。
        base = len(chunks)
        for k, (seg, _np, _nh) in enumerate(surviving):
            for p in seg:
                if id(p) not in piece_parent:
                    piece_parent[id(p)] = base + k

        for k, (seg, n_piece, n_hdr) in enumerate(surviving):
            content = "\n".join(p.text for p in seg)
            chunks.append(
                Chunk(
                    content=content,
                    chunk_index=base + k,
                    heading_path=path,
                    page_from=min(p.page_from for p in seg),
                    page_to=max(p.page_to for p in seg),
                    token_count=count(content),
                    role=ROLE_PARENT,
                    # n_children = **实际会指向本条的子块数**（片儿子 + 表头儿子），
                    # 不是"窗口覆盖的片数"，两者在含表格的窗口上不等。
                    meta={"n_children": n_piece + n_hdr},
                )
            )
        if windowed:
            n_windowed += 1
            n_window_parents += len(surviving)

    if n_windowed:
        warnings.append(
            f"{n_windowed} 个节超父块预算 {ccfg.parent_max_tokens} token，"
            f"已按窗口拆成 {n_window_parents} 个父块"
            f"（每窗 ≤ {max(2, ccfg.parent_window_pieces)} 片，重叠 {_window_overlap(ccfg)} 片）"
        )

    emitted_header: dict[str, int] = {}  # table_id → 表头片的 chunk_index

    for p in pieces:
        tid = p.table_id

        # 表头片排在它的数据片之前，且**只发一次**（首片处发）。
        if tid and tid in headers and tid not in emitted_header:
            line, first, n_data = headers[tid]
            emitted_header[tid] = len(chunks)
            chunks.append(
                Chunk(
                    content=line,
                    chunk_index=len(chunks),
                    heading_path=first.heading_path,
                    page_from=first.page_from,
                    page_to=first.page_to,
                    token_count=count(line),
                    role=ROLE_CHILD,
                    # 挂在该节父块上（若有）。这一层保证"节父块仍有人指"。
                    parent_index=piece_parent.get(id(first)),
                    meta={
                        "table_id": tid,
                        # 判别标记，检索层取整表、取表头都靠它 + table_id（与 role 无关）。
                        "table_header": True,
                        "n_children": n_data,
                        "n_blocks": 1,
                        "is_split": False,
                        "split_kind": "",
                        "source_kind": first.source_kind,
                    },
                )
            )

        # 数据片的回溯目标：**先看表头片**（"同表"比"同节"更具体），没有才落回节父块。
        target = emitted_header.get(tid) if tid else None
        if target is None:
            target = piece_parent.get(id(p))

        chunks.append(
            Chunk(
                content=p.text,
                chunk_index=len(chunks),
                heading_path=p.heading_path,
                page_from=p.page_from,
                page_to=p.page_to,
                token_count=p.token_count,
                role=ROLE_CHILD,
                parent_index=target,
                meta={
                    "n_blocks": p.n_blocks,
                    "is_split": p.is_split,
                    "split_kind": p.split_kind,
                    "source_kind": p.source_kind,
                    # 同表标记：非空 ⇒ 这一片就是那张表（未结构化的表拿不到它）。
                    # 落进 chunks.meta 供检索层"取整表"用，不靠片在编号上相邻（那只是当前实现的巧合）。
                    "table_id": p.table_id,
                },
            )
        )

    return chunks


def _uses_section_parent(piece: _Piece, headers: dict[str, tuple[str, _Piece, int]]) -> bool:
    """该片最终会不会挂到**节父块**上？

    表数据片只要其表被切成 ≥2 片（``table_id`` 命中 ``headers``）就改挂**表头片**
    （列名回溯比整节上下文更具体），不消耗 ``piece_parent``，故不计入父块的 ``n_children``。
    """
    return not (piece.table_id and piece.table_id in headers)


def _window_overlap(ccfg: ChunkConfig) -> int:
    """相邻窗口的重叠片数；钳到"窗口片数 − 1"（否则步进为 0 ⇒ 死循环）。"""
    size = max(2, ccfg.parent_window_pieces)
    return max(0, min(ccfg.parent_window_overlap, size - 1))


def _section_windows(group: list[_Piece], ccfg: ChunkConfig) -> list[list[_Piece]]:
    """把超预算节按**片数**切成若干窗口（相邻窗口重叠），并保证每窗 token ≤ ``parent_max_tokens``。

    片本身 ≤ ``max_tokens``，故默认 ``parent_window_pieces=4`` 时窗口约 ≤ 2048 token；
    token 上限是兜底，防日后把窗口片数调大后单窗超预算。窗口内至少一片，故必然推进。
    """
    size = max(2, ccfg.parent_window_pieces)
    overlap = _window_overlap(ccfg)
    step = max(1, size - overlap)
    budget = ccfg.parent_max_tokens
    out: list[list[_Piece]] = []
    n = len(group)
    start = 0
    while start < n:
        end = start
        tokens = 0
        while end < n and end - start < size:
            t = group[end].token_count
            if end > start and budget and tokens + t > budget:
                break
            tokens += t
            end += 1
        out.append(group[start:end])
        if end >= n:
            break
        # 下一窗的起点。取 min 是为了**不留覆盖空隙**：窗口被 token 预算截短时
        # （``end`` 提前），若仍按 ``start + step`` 前进，``[start+step, end)`` 这一段
        # 会落在所有窗口之外 ⇒ 那些片永远拿不到父块。回退到"本窗末尾往前 overlap 片"即可。
        start = min(start + step, max(start + 1, end - overlap))
    return out
