"""单页提取：把 PyMuPDF 的页面对象变成结构化的 :class:`PageScan`。

只调一次 ``get_text("dict")``，字符数 / 字号 / 版式信息全部由这一次结果派生；产出的是**页级**事实，不带页码、不管层级、不去页眉页脚。
"有效字符数"的口径在此唯一定义，体检与解析器共用。
"""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass, field
from typing import Any

from finrag.parse.guard import (
    SIZE_ROUND,
    ZONE_FOOTER,
    count_page_chars,
    edge_zone,
    in_content_zone,
)
from finrag.parse.ir import BBox, LineGeo, sanitize_text

__all__ = [
    "BOLD_FLAG",
    "TextBlock",
    "PageScan",
    "scan_page",
    "lines_of",
    "footer_text",
    "printed_page_label",
]

# PyMuPDF 的 span flags 位标志：第 4 位（值 16）为粗体。
# 其余：1 上标 / 2 斜体 / 4 衬线 / 8 等宽。
BOLD_FLAG = 1 << 4

# 页脚印刷页码的写法，实测四种都已出现，顺序有意义：
#   美的  "- 66 -"                茅台  "61 / 143"
#   多数  "第 12 页 共 291 页"      平安  "中国平安保险（集团）股份有限公司 197"
# 先认最明确的"第 X 页"，再认 "X / Y"，最后才认"文本尾部一串数字" ——
# 最后一条最宽松，放前面会把"共291页"这类总数当成页码。
_PRINTED_PAGE_PATTERNS = (
    re.compile(r"第(\d{1,4})页"),
    re.compile(r"(\d{1,4})/(\d{1,4})"),
    re.compile(r"[-–—](\d{1,4})[-–—]"),
    re.compile(r"^[^\d]*(\d{1,4})$"),
)

# 块内"多数文字加粗"的判定线。不用 any —— 正文里偶有几个加粗小标题就会把整个段落
# 标成粗体，而字号路径失效的文档正是靠这个标志找标题。
_BOLD_CHAR_SHARE = 0.6


@dataclass(slots=True)
class TextBlock:
    """一个页内文本块。图片块不进这里，只累加到 :attr:`PageScan.n_images`。"""

    text: str
    bbox: BBox
    size: float  # 块内字号中位数，按 SIZE_ROUND 取整
    bold: bool = False
    # 块内的非空行数。**这是区分"标题"与"段落"最便宜、最有效的信号**：
    # 字号和加粗都会失灵，但"标题通常 1~2 行、段落 3 行以上"几乎不随模板变化。
    n_lines: int = 1
    # 块内各 line 的文本与几何。与 ``n_lines`` 同源（都出自同一次
    # ``get_text("dict")`` 的 line 序列）⇒ 恒有 ``len(lines) == n_lines``。
    #
    # 存在的理由：``text`` 把行边界压平了，而"同一视觉行的多个单元格被粘在一起"
    # 只有几何诊断得出、也只有几何修得了。
    # 它**只加信息、不改产出**：``text`` 仍是 ``"".join`` 的结果，字符数口径一字不动。
    lines: list[LineGeo] = field(default_factory=list)


@dataclass(slots=True)
class PageScan:
    """一页的扫描结果。字段全部由一次 ``get_text("dict")`` 派生。

    ``error`` 非空时，其余字段是**可用的空值而非垃圾值**：该页计 0 字符（即判为空页），宽高仍有效。
    """

    page_no: int  # 1-based 物理页
    width: float
    height: float
    blocks: list[TextBlock] = field(default_factory=list)
    n_chars: int = 0
    n_images: int = 0
    # 图片块的外接矩形。**必须留下**：图文混排会打散块顺序，解析层要产出图片占位块
    # 才能解释"这里的块为什么断开了"，而占位块需要 bbox 才能在原文上定位。
    # 长度**可以小于** n_images：bbox 退化的图（width/height 为 0、坐标颠倒）被挡在外面。
    image_boxes: list[BBox] = field(default_factory=list)
    median_rel_width: float = 0.0
    error: str = ""

    @property
    def n_blocks(self) -> int:
        return len(self.blocks)


def _median(values: list[float], default: float = 0.0) -> float:
    return round(statistics.median(values), 3) if values else default


def lines_of(blk: dict[str, Any]) -> list[LineGeo]:
    """从 ``get_text("dict")`` 的一个块里取出各非空 line 的文本与几何。**实现只此一处。**

    line 含至少一个非空白 span 才算一条（与 ``TextBlock.n_lines`` 同源）、其 ``text`` 只含非空白 span 原文；``bbox`` 退化（长度不为 4）时填全零、不跳过该行。
    """
    out: list[LineGeo] = []
    for line in blk.get("lines", []):
        texts: list[str] = []
        for span in line.get("spans", []):
            # PyMuPDF 在坏字体上会把字形抽成 NUL，而 PG 的 text 禁收 NUL ⇒ 文本进系统即清洗。
            text = sanitize_text(span.get("text", ""))
            if text.strip():
                texts.append(text)
        if not texts:
            continue
        bb = line.get("bbox") or ()
        if len(bb) == 4:
            x0, y0, x1, y1 = (float(v) for v in bb)
        else:
            x0 = y0 = x1 = y1 = 0.0
        out.append(LineGeo("".join(texts), y0, y1, x0, x1))
    return out


def scan_page(page: object, page_no: int) -> PageScan:
    """扫描一页。``page`` 是 ``fitz.Page``（用 object 标注以免本模块被 fitz 绑架）。"""
    width = float(page.rect.width)  # type: ignore[attr-defined]
    height = float(page.rect.height)  # type: ignore[attr-defined]

    try:
        raw = page.get_text("dict")  # type: ignore[attr-defined]
    except Exception as exc:  # 个别页损坏不应中断整份文档
        return PageScan(
            page_no=page_no,
            width=width,
            height=height,
            error=f"{type(exc).__name__}: {exc}",
        )

    blocks: list[TextBlock] = []
    n_images = 0
    image_boxes: list[BBox] = []
    for blk in raw.get("blocks", []):
        raw_bbox = blk.get("bbox") or ()
        # PyMuPDF 的 bbox 恒为 4 元组。个别损坏页可能给出别的长度或 None，此时退回
        # 全零而不是抛异常 —— 抛了会让这一块被丢弃，块数偏少会低估该页字符数，
        # 最终影响的是**准入判定**本身。
        bbox = (
            BBox(*(float(v) for v in raw_bbox))
            if len(raw_bbox) == 4
            else BBox(0.0, 0.0, 0.0, 0.0)
        )

        if blk.get("type") == 1:
            n_images += 1
            # 只有**可定位**的图进 image_boxes。PyMuPDF 对一批图会给出退化的 bbox：
            # x1/y1 被填 0、甚至出现负的 x1，且 width/height 均为 0（来自跨页复用的
            # XObject 的累积坐标系），即左右/上下颠倒的矩形，落进 IR 会让校验报错。
            #
            # 计数与列表**刻意分开**：n_images 是体检口径（已落基线），改它等于悄悄
            # 改动已公布的语料契约；image_boxes 只服务解析器的占位块，过滤无副作用。
            if bbox.x0 <= bbox.x1 and bbox.y0 <= bbox.y1:
                image_boxes.append(bbox)
            continue

        # 行文本与几何由 ``lines_of`` **单独产出**；下面那一遍只为字号与加粗 ——
        # 那两项要**逐 span**（块级字号取所有 span 的中位数），粒度与"行"不同，合并不进去。
        geo = lines_of(blk)
        parts = [line.text for line in geo]
        sizes: list[float] = []
        bold_chars = 0
        total_chars = 0
        for line in blk.get("lines", []):
            for span in line.get("spans", []):
                # 必须与 lines_of 同一口径：不清洗会让字号/加粗统计把 NUL 也算进字符数。
                text = sanitize_text(span.get("text", ""))
                if not text.strip():
                    continue
                sizes.append(float(span.get("size", 0.0)))
                n = len(text)
                total_chars += n
                if int(span.get("flags", 0)) & BOLD_FLAG:
                    bold_chars += n
        n_lines = len(geo)

        # 块内 span **直接拼接**，不加分隔符 —— 中文没有词间空格，加空格会凭空多出字符，
        # 这一句与 guard.count_page_chars 的口径指向同一件事。
        #
        # 代价是**行边界一并被丢掉**：拼回来的 text 不含任何换行符。这是有意的取舍 ——
        # text 要服务 count_page_chars 的字符数口径，插分隔符会直接改动准入门槛。
        # 行结构另有出口（``TextBlock.lines``），语义重建下沉到分块层。
        # （D21 之后"IR 块文本不含换行"只对非表格块成立：结构化表格块的 text 必含换行。）
        block_text = "".join(parts).strip()
        if not block_text:
            continue

        # 字号取中位数而不是最大/最小：块内混排（正文 + 小一号的脚注）时中位数更能
        # 代表这个块"看上去多大"。
        size = round(statistics.median(sizes) / SIZE_ROUND) * SIZE_ROUND if sizes else 0.0
        blocks.append(
            TextBlock(
                text=block_text,
                bbox=bbox,
                size=size,
                bold=total_chars > 0 and bold_chars >= total_chars * _BOLD_CHAR_SHARE,
                n_lines=n_lines,
                lines=geo,
            )
        )

    # 版式：正文块（排除页边）的相对宽度中位数。单栏页正文块通常横跨大半页宽，
    # 双栏页则明显收窄。这是启发式信号，只用于给人工复核指路与多栏占比统计。
    # "算不算正文"走 guard.in_content_zone —— 与页眉页脚判据（edge_zone）用同一组常量
    # 但语义不同，所以是两个函数而不是一个。
    content_rel_widths = [
        b.bbox.width / width
        for b in blocks
        if width > 0 and in_content_zone(b.bbox.y0, b.bbox.y1, height) and b.bbox.width > 0
    ]

    return PageScan(
        page_no=page_no,
        width=width,
        height=height,
        blocks=blocks,
        n_chars=count_page_chars(blocks),
        n_images=n_images,
        image_boxes=image_boxes,
        median_rel_width=_median(content_rel_widths),
    )


def footer_text(scan: PageScan) -> str:
    """页脚区的第一个块文本。没有页脚块时返回空串。

    只取第一个：页脚可能由"公司名"与"页码"两块拼成，取全部再拼接会把两块数字混在一起。
    """
    for b in scan.blocks:
        if b.text.strip() and edge_zone(b.bbox.y0, scan.height) == ZONE_FOOTER:
            return b.text
    return ""


def printed_page_label(scan: PageScan) -> str | None:
    """抠出这一页的**印刷页码**，抠不出返回 ``None``。

    ``None`` 是常态；不要用"物理页 − 固定偏移"反推（各文档偏移不一致）；返回值只填 ``PageRef.label``，不参与定位。
    """
    t = re.sub(r"\s+", "", footer_text(scan))
    if not t or len(t) > 40:
        return None
    for pat in _PRINTED_PAGE_PATTERNS:
        m = pat.search(t)
        if not m:
            continue
        v = int(m.group(1))
        if 0 < v <= 9999:
            return str(v)
    return None
