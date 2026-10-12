"""解析层中间表示（IR）契约：页面定位、文本块、层级树与文档级元数据。

解析层只产出结构 IR、不产出 chunk，本模块零内部依赖：只有数据、类型与若干纯文本工具。
``Block.page`` 一律是 1-based 物理页序、非印刷页码（后者见 ``PageRef.label``）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Literal, NamedTuple

__all__ = [
    "IR_SCHEMA",
    "PARSER_VERSION",
    "BlockKind",
    "HeadingSource",
    "BBox",
    "LineGeo",
    "TableData",
    "Block",
    "Section",
    "PageRef",
    "DocMeta",
    "ParsedDoc",
    "sanitize_text",
]

# 结构版本：字段增删改 → +1。下游读取代码必须同步，否则会读到不存在的属性。
IR_SCHEMA = 4
# 解析行为版本：结构不变、算法或启发式改了 → +1。用于判断"已入库的数据要不要重跑"。
PARSER_VERSION = 7

# 块类型。第一维只区分"要不要在检索里出现"；table 的两种形态见 ``Block.table``。
BlockKind = Literal["paragraph", "table", "list", "caption", "image"]

# heading 来源：toc（书签 / TOC）为主路径，layout（字号兜底）服务无书签文档。
HeadingSource = Literal["toc", "layout"]


# ------------------------------------------------------------------
# 文本清洗：进 IR 的文本必须能落进 PostgreSQL
# ------------------------------------------------------------------
# PostgreSQL 的 ``text`` / ``jsonb`` **硬禁 NUL 字节（``U+0000``）**：含它的字符串连
# INSERT 都发不出去，驱动在发送阶段就抛
# ``DataError: PostgreSQL text fields cannot contain NUL (0x00) bytes``。
#
# PyMuPDF 在字体编码异常的 PDF（子集字体 / ToUnicode CMap 缺失或把字形映射到 0）上，
# 会把下面**三个入口**都抽成含 ``\x00`` 的字符串 —— 三者全在解析层，故清洗也留在解析层：
#   ① ``page.get_text("dict")`` 的 span 文本            → pageio
#   ② ``find_tables().extract()`` 的单元格 / 表头名      → pdf_parser._read_table / _header_info
#   ③ ``doc.get_toc(simple=True)`` 的书签标题            → heading.toc_hits
def sanitize_text(text: str) -> str:
    """剔除 PostgreSQL 无法存储的 NUL（``U+0000``）；无 NUL 时**原样返回**（不做无谓拷贝）。

    **只删 ``\\x00``**：``\\t`` / ``\\n`` 等其余控制字符 PG 本就允许，而多删会悄悄改动
    "有效字符数"口径（``guard.count_page_chars`` 是准入判据、也是消融实验的自变量），
    不能顺手一起改。
    """
    return text.replace("\x00", "") if "\x00" in text else text


class BBox(NamedTuple):
    """外接矩形：**PDF 坐标**，原点在左上、y 轴向下、单位 pt（PyMuPDF 约定）。"""

    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0


class LineGeo(NamedTuple):
    """PDF 里一条 line 的文本与几何，供分块层重建块文本用（不动 ``Block.text``）。

    ``text`` 口径同 ``scan_page`` 拼块文本：``"".join(l.text for l in lines)``。
    """

    text: str
    y0: float
    y1: float
    x0: float
    x1: float

    @property
    def y(self) -> float:
        """y 中心 —— "同一基线"的判定量。"""
        return (self.y0 + self.y1) / 2


@dataclass(slots=True)
class TableData:
    """结构化表格的单元格数据（决策 D21）。只挂在 ``Block.table`` 上；不存 bbox。"""

    # 裁掉全空行列之后的单元格矩阵。它是**序列化文本的形状**，不是 ``find_tables`` 的原始输出。
    rows: list[list[str]]
    # ``find_tables`` 的原始 row_count / col_count（未裁），用于度量过分割（多切出的列数）。
    n_rows_raw: int
    n_cols_raw: int
    # 表头，与 ``rows`` 逐列对齐；``header_external=True`` 时内容多为噪声，不得进检索文本。
    header: list[str] = field(default_factory=list)
    # 表头在表格矩形**之外**（PyMuPDF 判定）：``True`` 意味 ``header`` 大概率不是列名。
    header_external: bool = False
    # ``len(extract 去空白) / len(块文本去空白)``，口径见 :func:`parse.table.reconcile`；恒 ≥ 1.0。
    align_recall: float = 1.0


@dataclass(slots=True)
class Block:
    """文档级文本块 —— 阅读顺序已还原、页眉页脚已剔除之后的最小内容单元。"""

    # 块内文本。段落/图片块不含行结构（无换行）；结构化表格块是整表的 markdown，含换行。
    text: str
    page: int  # 1-based 物理页
    bbox: BBox
    kind: BlockKind = "paragraph"
    font_size: float = 0.0
    # 块内多数文字加粗。不要用它单独判标题，仅用于版式加权 / 前端高亮。
    bold: bool = False
    # 块内非空行数。区分标题与段落最便宜有效的信号：标题 1~2 行、段落 3 行以上。
    n_lines: int = 1
    # 块内各 line 的文本与几何，默认空列表（仅 scan_page 扫出的块有值，结构化表格块不填）。
    lines: list[LineGeo] = field(default_factory=list)
    # 结构化表格数据（决策 D21）。判"是不是结构化表格"要看本字段而非 kind：``None`` = 未结构化。
    table: TableData | None = None
    # kind == "table" 时的表题。**当前恒为 None**：表题本是独立 IR 块，已参与检索。
    caption: str | None = None


@dataclass(slots=True)
class Section:
    """层级树的节点。一个标题 + 它名下直属的 Block + 子 Section。"""

    title: str
    level: int  # 1 起；root 虚拟节点为 0
    page: int  # 标题所在的 1-based 物理页
    source: HeadingSource
    blocks: list[Block] = field(default_factory=list)
    children: list["Section"] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class PageRef:
    """物理页 ↔ 印刷页码的对照项。**只服务展示，不参与定位**；``label`` 可为 None。"""

    index: int  # 1-based 物理页
    label: str | None = None  # 印刷页码，如 "128" / "Ⅶ" / "附录-3"


@dataclass(slots=True)
class DocMeta:
    """解析期可确定的文档级元数据（``code`` / ``company`` / ``year`` 不在此，归 corpus）。"""

    ir_schema: int
    parser_version: int
    probe_version: int  # 判定准入选用的判据快照（guard.PROBE_VERSION）
    doc_type: str  # 文本型 / 混合件 / 纯扫描件 / 未知，见 guard.DOC_*
    has_toc: bool
    page_count: int  # 物理页总数
    # "疑似多栏页"占比 —— 页级代理指标（非真实栏数），仅供对齐体检字段，不可用于判定重排。
    multicol_ratio: float
    # 全局正文字号占全部字符的比例；越低，"比正文大就是标题"越不成立。不落库、不做判定。
    body_share: float = 0.0


@dataclass(slots=True)
class ParsedDoc:
    """一份文档的完整 IR。``root`` 是虚拟根（``level = 0``、``title = ""``），承载首个标题前的内容。"""

    source_path: Path
    root: Section
    pages: list[PageRef]
    meta: DocMeta

    def iter_blocks(self) -> Iterator[tuple[tuple[str, ...], Block]]:
        """深度优先遍历，产出 ``(heading_path, block)``。

        ``heading_path`` 是父到子的标题元组（root 直属内容为空元组），长度不固定。
        """

        def walk(section: Section, path: tuple[str, ...]) -> Iterator[tuple[tuple[str, ...], Block]]:
            for block in section.blocks:
                yield path, block
            for child in section.children:
                # root（level 0、title 空）不贡献路径层级
                child_path = path + (child.title,) if child.level > 0 else path
                yield from walk(child, child_path)

        yield from walk(self.root, ())

    def iter_sections(self) -> Iterator[tuple[tuple[str, ...], Section]]:
        """遍历所有真实 section（不含虚拟根），供调试与结构抽查。"""

        def walk(section: Section, path: tuple[str, ...]) -> Iterator[tuple[tuple[str, ...], Section]]:
            if section.level > 0:
                path = path + (section.title,)
                yield path, section
            for child in section.children:
                yield from walk(child, path)

        yield from walk(self.root, ())

    def block_count(self) -> int:
        return sum(1 for _ in self.iter_blocks())

    def validate(self) -> list[str]:
        """结构自检。返回问题清单（空列表表示通过），刻意不抛异常。"""
        problems: list[str] = []
        page_count = self.meta.page_count

        if len(self.pages) != page_count:
            problems.append(
                f"pages 长度 {len(self.pages)} 与 meta.page_count {page_count} 不一致"
            )
        for i, ref in enumerate(self.pages, 1):
            if ref.index != i:
                problems.append(f"pages[{i - 1}].index = {ref.index}，应为 {i}（必须连续且 1-based）")
                break

        if self.root.level != 0:
            problems.append(f"root.level = {self.root.level}，虚拟根必须为 0")

        def walk(section: Section, parent: Section | None, path: str) -> None:
            if section.level > 0 and not section.title.strip():
                problems.append(f"{path}：level {section.level} 的 section 标题为空")
            if section.level < 0 or section.level > 9:
                problems.append(f"{path}：level {section.level} 超出合理范围 0~9")
            # 严格递进：允许跳级，但禁止同级或降级（否则 heading_path 语义崩坏）。
            if parent is not None and parent.level > 0 and section.level <= parent.level:
                problems.append(
                    f"{path}：level {section.level} 未大于父节点 {parent.title!r} 的 {parent.level}"
                )
            if page_count and not (1 <= section.page <= page_count):
                problems.append(f"{path}：page {section.page} 超出 1~{page_count}")
            for j, block in enumerate(section.blocks):
                if page_count and not (1 <= block.page <= page_count):
                    problems.append(f"{path} 第 {j} 块：page {block.page} 超出 1~{page_count}")
                b = block.bbox
                if b.x0 > b.x1 or b.y0 > b.y1:
                    problems.append(f"{path} 第 {j} 块：bbox 上下/左右颠倒 {tuple(b)}")
                # 负坐标**刻意不判为问题**：PDF 允许内容越出 CropBox（出血），非 IR 结构错。
                if block.kind == "table" and block.text == "" and not block.caption:
                    problems.append(f"{path} 第 {j} 块：表格块既无内容也无表题")
                # ---- 结构化表格的自检（D21）：拦形态标记与数据对不上的静默失败 ----
                if block.table is not None:
                    if block.kind != "table":
                        problems.append(
                            f"{path} 第 {j} 块：带了结构化表格数据却未标 kind=table"
                            f"（实际 {block.kind!r}）—— 下游按 kind 分流会漏掉它"
                        )
                    if not block.table.rows:
                        problems.append(
                            f"{path} 第 {j} 块：结构化表格的 rows 为空 —— "
                            f"守门应当已把它退回未结构化形态"
                        )
                    if block.table.align_recall < 1.0:
                        # 恒等不变式：align_recall < 1 意味块文本里有 extract 没有的字符（内容已丢）。
                        problems.append(
                            f"{path} 第 {j} 块：结构化表格 align_recall = "
                            f"{block.table.align_recall:.3f} < 1.0，守门被绕过（内容已丢）"
                        )
            for k, child in enumerate(section.children):
                walk(child, section, f"{path}/{k}")

        # 从**根节点本身**开始走（不是 root.children）：``root.blocks`` 装了首个标题前的内容。
        walk(self.root, None, "root")

        return problems
