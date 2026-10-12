"""表格结构化的口径 —— 裁空、序列化、对账守门；三条判据都不依赖 PyMuPDF，只操作二维字符串矩阵。
:func:`drop_empty` / :func:`serialize_md` / :func:`reconcile` 由解析器与探针反向引用同一份实现。"""

from __future__ import annotations

from collections import Counter
from typing import Any, Final, Iterable, NamedTuple, Sequence

__all__ = [
    "CJK_CLASS",
    "CJK_RANGES",
    "MD_CELL_SEP",
    "Reconcile",
    "align_header",
    "drop_empty",
    "flatten_header_names",
    "fold_cell",
    "is_cjk",
    "norm_ws",
    "reconcile",
    "serialize_md",
]

# markdown 管道的单元格分隔符。**格式定义只此一处** —— 展示层要靠它解析回单元格。
MD_CELL_SEP: Final[str] = " | "

#: "算一个词内字符"的区间：CJK 表意文字 + CJK 标点 + 全角形式。
#:
#: **判据只此一处**（判据下沉、探针反向引用）。带上标点与全角形式是有意的：
#: ``元）\n其中`` 这种折行落在括号边界上，按"仅表意文字"判会退化成插空格。
#:
#: **写成区间元组而不是拼接好的正则字符类**：直接 ``frozenset()`` 一个
#: ``"\u4e00-\u9fff"`` 字符串得到的是 ``{'\', 'u', '4', 'e', ...}`` —— 一个字都判不出来，
#: 且**不报错**。⇒ 区间只定义一次，:func:`is_cjk` 与探针各自展开。
CJK_RANGES: Final[tuple[tuple[int, int], ...]] = (
    (0x4E00, 0x9FFF),  # CJK 表意文字
    (0x3000, 0x303F),  # CJK 标点（。、〈〉「」…）
    (0xFF01, 0xFF60),  # 全角形式（！到｠）
    (0xFFE0, 0xFFE6),  # 全角货币符号
)


def _expand(ranges: Iterable[tuple[int, int]]) -> frozenset[str]:
    return frozenset(
        chr(cp) for lo, hi in ranges for cp in range(lo, hi + 1)
    )


_CJK_SET: Final[frozenset[str]] = _expand(CJK_RANGES)

#: 给正则用的字符类写法（``[...]`` 里的内容）。**由 :data:`CJK_RANGES` 生成**，不另写一份。
CJK_CLASS: Final[str] = "".join(
    f"{chr(lo)}-{chr(hi)}" if lo != hi else chr(lo) for lo, hi in CJK_RANGES
)


def is_cjk(ch: str) -> bool:
    """单个字符是否属于 :data:`CJK_RANGES`。空串返回 ``False``。"""
    return ch in _CJK_SET


def norm_ws(s: str) -> str:
    """去掉全部空白字符。对账只关心**字符在不在**，不关心排版。"""
    return "".join(s.split())


def drop_empty(data: list[list[str]]) -> tuple[list[list[str]], list[int]]:
    """裁掉全空的行与全空列，返回 ``(裁后的矩阵, 保留的列下标)``。
    只裁"全空"：部分空的列必须保留（合并单元格的形态）；返回的列下标是表头与列对齐的唯一依据。"""
    kept_rows = [r for r in data if any(c.strip() for c in r)]
    if not kept_rows:
        return [], []
    # 列数取各行最大长度：extract 的短行是"尾部单元格不存在"，按缺失补空串
    ncol = max(len(r) for r in kept_rows)
    keep = [j for j in range(ncol) if any(j < len(r) and r[j].strip() for r in kept_rows)]
    return [[r[j] if j < len(r) else "" for j in keep] for r in kept_rows], keep


def fold_cell(cell: str) -> str:
    """单元格内部折行 → 单个分隔符，逐个折行处按两侧字符性质决定：
    两侧都是 :data:`CJK_RANGES` 则不插，任一侧是数字 / 西文则插空格。"""
    if "\n" not in cell:
        return cell.strip()
    out: list[str] = []
    for seg in cell.split("\n"):
        s = seg.strip()
        if not s:
            # 空段（连续折行 / 首尾折行）不产出分隔符，否则会在词中多插一个空格。
            continue
        if out:
            # 上一个已保留段的末字符 × 本段首字符 —— 判"该插什么"。
            prev_ch = out[-1][-1]
            next_ch = s[0]
            if not (is_cjk(prev_ch) and is_cjk(next_ch)):
                out.append(" ")
        out.append(s)
    return "".join(out)


def serialize_md(data: list[list[str]]) -> str:
    """二维单元格 → markdown 管道文本（每行一条，``| a | b |``），列边界显式可见；单元格内部折行交由 :func:`fold_cell` 处理。"""
    return "\n".join(
        "| " + MD_CELL_SEP.join(fold_cell(c) for c in row) + " |" for row in data
    )


def flatten_header_names(names: Iterable[Any]) -> list[str]:
    """``t.header.names``（每列一组）→ 一列一名（按列取第一项），长度等于列数。
    空名留占位、不过滤 —— 过滤会丢掉列位置，让后面所有名字整体前移（表头不参与对账，会静默生效）。"""
    out: list[str] = []
    for group in names:
        if isinstance(group, (tuple, list)):
            out.append(str(group[0]) if group else "")
        else:
            out.append(str(group) if group else "")
    return out


def align_header(names: Sequence[str], keep: Sequence[int], n_cols: int) -> list[str]:
    """把表头从 ``extract()`` 的列坐标系按 ``keep`` 投影到裁空后的坐标系，长度对齐 ``n_cols``。
    ``keep`` 与 ``names`` 长度对不上时返回空表 —— 宁可没列名，也不交可能错位的表头。"""
    if not names:
        return []
    if len(keep) != n_cols:
        return []
    return [names[j] if j < len(names) else "" for j in keep]


class Reconcile(NamedTuple):
    """对账结果。``ok`` 是唯一被用来决策的字段，其余三个是留痕与诊断用。"""

    ok: bool
    # ``len(extract 去空白) / len(块文本去空白)``。**恒 ≥ 1.0**。明显大于 1 说明
    # ``extract()`` 多拿了矩形之外的内容（表题、表外标题行）—— 那是冗余不是收益。
    align_recall: float
    # 块文本里有、extract 里没有的字符数。``ok`` 时恒为 0。
    missing: int
    # extract 里有、块文本里没有的字符数。它**不影响 ok**。
    extra: int


def reconcile(rows: list[list[str]], block_texts: Sequence[str]) -> Reconcile:
    """``extract()`` 的单元格是否一字不差覆盖了被罩住的块文本（两边去空白后按字符多重集求差，不看顺序）。
    口径上接受漏报（同字出现在别处即可顶掉一次真缺失）；用原始 ``rows``，空单元格不贡献字符。"""
    ex = norm_ws("".join(c for row in rows for c in row))
    bl = norm_ws("".join(block_texts))
    if not bl:
        # 没有块就没什么可吸收的，别把空对空算成通过
        return Reconcile(ok=False, align_recall=0.0, missing=0, extra=len(ex))
    # 用两个 Counter 一次算清两侧：差集只在正数侧保留，正好是"谁多了谁少了"
    ce, cb = Counter(ex), Counter(bl)
    missing = sum((cb - ce).values())
    extra = sum((ce - cb).values())
    return Reconcile(
        ok=missing == 0,
        align_recall=len(ex) / max(1, len(bl)),
        missing=missing,
        extra=extra,
    )
