"""标题层级：TOC 主路径与版式兜底两条来源收束到同一棵树。

两条路都归一成 :class:`HeadingHit` 序列、由同一个 :class:`TreeBuilder` 建树，``source`` 只作留痕。
``get_toc()`` 的 page 是 1-based 物理页，不做偏移校正；版式兜底可信度由 ``meta.body_share`` 标出。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Final, Iterable, Sequence

from finrag.parse.ir import HeadingSource, Section, sanitize_text

__all__ = [
    "HeadingHit",
    "LAYOUT_MODE_BODY_WINDOW",
    "body_size_stats",
    "local_body_sizes",
    "toc_hits",
    "layout_hits",
    "layout_candidates",
    "TreeBuilder",
    "ENDS_SENTENCE",
]

# 标题候选的长度上限。超过的多半是正文段落。
CAND_MAX_CHARS: Final[int] = 40
CAND_MAX_LINES: Final[int] = 2  # 超过此行数视为段落
CAND_SIZE_DELTA: Final[float] = 0.5
ENDS_SENTENCE: Final[re.Pattern[str]] = re.compile(r"[。；！？，、,;.]$")
# 局部正文字号的滑动窗口（±N 页）。仅在"本页正文太少"时启用，见 local_body_sizes。
LAYOUT_MODE_BODY_WINDOW: Final[int] = 3
LOCAL_BODY_MIN_CHARS: Final[int] = 200

# 字号差 → 层级。分档依据是实测的字号档间距（比亚迪正文 9pt：16pt 一级、12pt 二级、
# 10.5pt 三级）。
_LEVEL_DELTA_MAJOR: Final[float] = 6.0
_LEVEL_DELTA_MINOR: Final[float] = 3.0
_LEVEL_MAX: Final[int] = 3


@dataclass(frozen=True, slots=True)
class HeadingHit:
    """一条标题证据。两条来源归一成同一种东西，好让建树逻辑只有一份。"""

    page: int  # 1-based 物理页
    y0: float  # 块上沿；TOC 路径无 y（只知道页码）→ -1，表示"排在该页所有块之前"
    level: int  # 1 起
    title: str
    source: HeadingSource


# ------------------------------------------------------------------
# 路径一：TOC
# ------------------------------------------------------------------
def _needle(title: str) -> str:
    return re.sub(r"\s+", "", title or "").strip(".．·…、，,：:;；")


def clamp_toc_pages(
    toc: Iterable[Sequence[Any]], page_count: int
) -> tuple[list[tuple[int, str, int]], list[str]]:
    """书签页码的单调性守卫：把违反单调（``page`` 小于前一条）的页号钳制到前一条。

    返回 ``(清洗后的 (level, title, page) 序列, 修正说明)``。
    """
    entries: list[tuple[int, str, int]] = []
    fixes: list[str] = []
    last = 0
    n_clamped = 0
    first_example = ""

    for item in toc:
        if len(item) < 3:
            continue
        try:
            level, title, page = int(item[0]), str(item[1]), int(item[2])
        except (TypeError, ValueError):
            continue
        if not (1 <= page <= page_count):
            continue
        if page < last:
            # 钳制而不是丢弃：丢掉会让整节失去标题；钳制的代价只是页码偏前几页，
            # 树结构与内容归属仍能落回正确的章节区间。
            if not n_clamped:
                first_example = f"{title.strip()[:24]!r} p{page} → p{last}"
            n_clamped += 1
            page = last
        entries.append((max(1, min(level, 9)), title.strip(), page))
        last = page

    if n_clamped:
        fixes.append(
            f"书签页码非单调：{n_clamped} 条被钳制到前一条的页码（首例 {first_example}）"
            f" —— 源 PDF 书签数据有误，不钳制则这些节点会把前置内容吸进自己的子树"
        )
    return entries, fixes


def toc_hits(
    toc: Iterable[Sequence[Any]],
    page_count: int,
    notes: list[str] | None = None,
) -> list[HeadingHit]:
    """书签 → :class:`HeadingHit`。三道过滤：页号越界丢弃、非单调交由
    :func:`clamp_toc_pages` 钳制（修正经 ``notes`` 上抛）、同页同归一化标题去重保留先出现者。
    """
    entries, fixes = clamp_toc_pages(toc, page_count)
    if notes is not None:
        notes.extend(fixes)

    out: list[HeadingHit] = []
    seen: set[tuple[int, str]] = set()
    for level, title, page in entries:
        # 书签标题直出 ``get_toc()``、没过 pageio ⇒ 在此单独清洗：NUL 会随
        # ``heading_path`` 进 chunk 文本，落库时同样会被 PG 拒绝。
        title = sanitize_text(title)
        key_title = _needle(title)
        if not key_title:
            continue
        key = (page, key_title)
        if key in seen:
            continue
        seen.add(key)
        out.append(
            HeadingHit(
                page=page,
                y0=-1.0,
                level=level,
                title=title,
                source="toc",
            )
        )
    return out


# ------------------------------------------------------------------
# 路径二：版式兜底
# ------------------------------------------------------------------
def body_size_stats(blocks_by_page: Iterable[Iterable[Any]]) -> tuple[float, float]:
    """全文的（字符加权主字号, 主字号字符占比）。

    按字符加权而非按块计票，避免短标题块与长正文块计票相当而让标题胜出。
    """
    counter: dict[float, int] = {}
    total = 0
    for blocks in blocks_by_page:
        for b in blocks:
            counter[b.size] = counter.get(b.size, 0) + len(b.text)
            total += len(b.text)
    if not counter or total <= 0:
        return 0.0, 0.0
    size, chars = max(counter.items(), key=lambda kv: kv[1])
    return size, chars / total


def local_body_sizes(blocks_by_page: list[list[Any]]) -> list[float]:
    """每页的局部正文字号（字符加权众数）。

    页内正文不足 ``LOCAL_BODY_MIN_CHARS`` 时退回 ±``LAYOUT_MODE_BODY_WINDOW`` 页窗口。
    """
    n = len(blocks_by_page)
    out: list[float] = []
    for i in range(n):
        own = blocks_by_page[i]
        if sum(len(b.text) for b in own) >= LOCAL_BODY_MIN_CHARS:
            counter: dict[float, int] = {}
            for b in own:
                counter[b.size] = counter.get(b.size, 0) + len(b.text)
            out.append(max(counter.items(), key=lambda kv: kv[1])[0])
            continue
        window: dict[float, int] = {}
        for j in range(max(0, i - LAYOUT_MODE_BODY_WINDOW), min(n, i + LAYOUT_MODE_BODY_WINDOW + 1)):
            for b in blocks_by_page[j]:
                window[b.size] = window.get(b.size, 0) + len(b.text)
        out.append(max(window.items(), key=lambda kv: kv[1])[0] if window else 0.0)
    return out


def layout_candidates(blocks: list[Any], local_body: float) -> list[Any]:
    """一页里满足三条判据的块：字号 ≥ 局部正文 + 0.5pt、行数 ≤ 2、不以句末标点结尾。"""
    out = []
    for b in blocks:
        t = b.text.strip()
        if not t or len(t) > CAND_MAX_CHARS:
            continue
        if getattr(b, "n_lines", 1) > CAND_MAX_LINES:
            continue
        if ENDS_SENTENCE.search(t):
            continue
        if local_body and b.size >= local_body + CAND_SIZE_DELTA:
            out.append(b)
    return out


def _level_of(delta: float) -> int:
    if delta >= _LEVEL_DELTA_MAJOR:
        return 1
    if delta >= _LEVEL_DELTA_MINOR:
        return 2
    return min(3, _LEVEL_MAX)


def layout_hits(blocks_by_page: list[list[Any]]) -> list[HeadingHit]:
    """版式兜底 → :class:`HeadingHit` 序列（按页、按 y 排序）。

    ``level`` 是**字号差的推断**、非权威层级，故只在无 TOC 的文档上生效。
    """
    bodies = local_body_sizes(blocks_by_page)
    hits: list[HeadingHit] = []
    for i, blocks in enumerate(blocks_by_page):
        local = bodies[i] or 0.0
        if not local:
            continue
        for b in layout_candidates(blocks, local):
            hits.append(
                HeadingHit(
                    page=i + 1,
                    y0=float(b.bbox.y0),
                    level=_level_of(b.size - local),
                    title=b.text.strip(),
                    source="layout",
                )
            )
    hits.sort(key=lambda h: (h.page, h.y0))
    return hits


# ------------------------------------------------------------------
# 建树（两条路径共用）
# ------------------------------------------------------------------
class TreeBuilder:
    """按阅读顺序喂入 :class:`HeadingHit`，用栈维护层级：栈中 level 严格递增。

    新 hit 到来时弹出 level 不小于它的祖先，挂到剩余栈顶之下；天然容忍跳级。
    """

    def __init__(self, source: HeadingSource = "toc") -> None:
        self.root = Section(title="", level=0, page=1, source=source)
        self._stack: list[Section] = [self.root]

    def push(self, hit: HeadingHit) -> Section:
        while len(self._stack) > 1 and self._stack[-1].level >= hit.level:
            self._stack.pop()
        node = Section(
            title=hit.title, level=hit.level, page=hit.page, source=hit.source
        )
        self._stack[-1].children.append(node)
        self._stack.append(node)
        return node

    @property
    def current(self) -> Section:
        """当前负责接收块的节点（没有标题时就是虚拟根）。"""
        return self._stack[-1]
