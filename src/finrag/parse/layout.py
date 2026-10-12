"""几何判据：**页级的栏结构** + **块级的行结构**。判据只在这里定义一处。

纯几何函数，输入 duck-typed、输出判据 —— 解析器与探针都反向引用。
栏重排默认关闭（``parse.reorder_columns`` 为 **False**），检测能力保留供消融实验。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence, TypeVar

#: 行几何对象的类型变量。刻意不写死 ``ir.LineGeo``，只要求对象有 ``.y`` / ``.x0`` / ``.text``。
L = TypeVar("L")

__all__ = [
    "BIN_PT",
    "GUTTER_MIN_PT",
    "GUTTER_MIN_BLOCKS_PER_COL",
    "ROW_ALIGN_MIN",
    "ROW_CLUSTER_SIZE_FACTOR",
    "SAME_ROW_TOL",
    "PageColumns",
    "content_bounds",
    "find_gutters",
    "col_of",
    "row_alignment",
    "y_monotonic",
    "row_cluster_tol",
    "y_clusters",
    "n_y_clusters",
    "has_glued_cells",
    "has_side_by_side",
    "glued_pairs",
    "analyze",
    "column_major_order",
    "should_reorder",
]

BIN_PT = 2.0  # x 覆盖率直方图的分箱宽度
# 内容区边界分位数。用分位而非极值：个别越界块会把边界撑开、把真栏沟推出搜索范围。
CONTENT_BOUND_Q_LO = 0.05
CONTENT_BOUND_Q_HI = 0.95
CONTENT_BOUND_INSET = 2.0
# 栏沟最小宽度（pt）。实测栏沟通常 15~30pt，低于 12pt 的空白多为缩进，认成栏沟会切碎单栏页。
GUTTER_MIN_PT = 12.0
# 每栏最少块数。单个孤块会在其左侧造出贯穿整页的空白带，不设此限会把几乎每页都判成多栏。
GUTTER_MIN_BLOCKS_PER_COL = 2
ROW_ALIGN_Y_TOL = 3.0  # 判定"同一行"的 y 容差
ROW_ALIGN_MIN = 0.60  # 对齐比例达此值 → 行对齐型
Y_MONOTONIC_TOL = 3.0  # 默认顺序 y 单调的容差：同一行内的单元格顺序会有微小抖动


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    vs = sorted(values)
    i = min(len(vs) - 1, max(0, int(round(q * (len(vs) - 1)))))
    return vs[i]


def content_bounds(blocks: Iterable[Any]) -> tuple[float, float]:
    """内容区左右边界，从本页块的实际范围（分位数）推，不用固定页边比例。"""
    bs = list(blocks)
    return _quantile([b.bbox.x0 for b in bs], CONTENT_BOUND_Q_LO), _quantile(
        [b.bbox.x1 for b in bs], CONTENT_BOUND_Q_HI
    )


def find_gutters(blocks: list[Any], min_pt: float = GUTTER_MIN_PT) -> list[tuple[float, float]]:
    """在 x 覆盖率上找"零覆盖空白带"，即栏沟。返回 [(起, 止)]（PDF pt）。

    调用方必须**先剔除页眉页脚**（它们通栏，会盖住栏沟）；块数 < 4 时直接返回 []。
    """
    if len(blocks) < 4:
        return []
    left, right = content_bounds(blocks)
    lo = left + CONTENT_BOUND_INSET
    hi = right - CONTENT_BOUND_INSET
    if hi - lo < min_pt * 2:
        return []

    nbins = int(hi / BIN_PT) + 2
    occ = bytearray(nbins)
    for b in blocks:
        i0 = max(0, int(b.bbox.x0 / BIN_PT))
        i1 = min(nbins - 1, int(b.bbox.x1 / BIN_PT))
        for i in range(i0, i1 + 1):
            occ[i] = 1

    gutters: list[tuple[float, float]] = []
    run: float | None = None
    for i in range(nbins):
        x = i * BIN_PT
        if not occ[i] and lo <= x <= hi:
            if run is None:
                run = x
            continue
        if run is not None:
            end = min(x, hi)
            if end - run >= min_pt:
                gutters.append((run, end))
            run = None
    if run is not None and hi - run >= min_pt:
        gutters.append((run, hi))

    if not gutters:
        return []
    # 防单块造栏：某个"栏"里只有 1 个块 → 这条栏沟更像偶然空隙，整页退回单栏。
    # 保守方向是明确的：不重排只是没优化，错排会打乱正文。
    cols = [col_of(b.bbox.x0, gutters) for b in blocks]
    counts = {c: cols.count(c) for c in set(cols)}
    if min(counts.values()) < GUTTER_MIN_BLOCKS_PER_COL:
        return []
    return gutters


def col_of(x0: float, gutters: list[tuple[float, float]]) -> int:
    """块的左边缘越过几条栏沟（以栏沟**中点**为界）就是第几栏。"""
    return sum(1 for gs, ge in gutters if x0 > (gs + ge) / 2)


def row_alignment(blocks: list[Any], cols: list[int]) -> float:
    """行对齐度：非首栏的块里，能在首栏找到 y 差 ≤3pt 伙伴的比例（表格 vs 分栏正文信号）。

    它**不是充分判据**：合并单元格的复杂表格会使其失效，还需 ``find_tables`` 兜底。
    """
    base = [b.bbox.y0 for i, b in enumerate(blocks) if cols[i] == 0]
    total = 0
    aligned = 0
    for i, b in enumerate(blocks):
        if cols[i] == 0:
            continue
        total += 1
        if any(abs(b.bbox.y0 - y) <= ROW_ALIGN_Y_TOL for y in base):
            aligned += 1
    return aligned / total if total else 0.0


def y_monotonic(blocks: list[Any]) -> bool:
    """默认块顺序的 y 是否单调不降 —— 即它**是否已经是行优先**（容差 ``Y_MONOTONIC_TOL``）。"""
    ys = [b.bbox.y0 for b in blocks]
    return all(ys[i] >= ys[i - 1] - Y_MONOTONIC_TOL for i in range(1, len(ys)))


@dataclass(slots=True)
class PageColumns:
    """一页的栏结构分析结果。"""

    gutters: list[tuple[float, float]]
    cols: list[int]  # 每个块的栏号，与传入的 blocks 同序
    row_align: float
    y_monotonic: bool

    @property
    def n_columns(self) -> int:
        return len(self.gutters) + 1


def analyze(blocks: list[Any]) -> PageColumns | None:
    """分析一页的栏结构。检不出栏沟返回 ``None``（即单栏，不必做任何重排）。"""
    gutters = find_gutters(blocks)
    if not gutters:
        return None
    cols = [col_of(b.bbox.x0, gutters) for b in blocks]
    return PageColumns(
        gutters=gutters,
        cols=cols,
        row_align=row_alignment(blocks, cols),
        y_monotonic=y_monotonic(blocks),
    )


def column_major_order(blocks: list[Any], cols: list[int]) -> list[int]:
    """按 (栏, y0) 给出块的下标顺序 —— 即契约 C5 要求的列优先读法。"""
    return sorted(range(len(blocks)), key=lambda i: (cols[i], blocks[i].bbox.y0))


def should_reorder(pc: PageColumns | None, table_covers_gutter: bool) -> bool:
    """最终门控：检出栏沟 + 非行对齐型（非表格）+ 栏沟未被表格覆盖，三者同时成立才重排。"""
    if pc is None:
        return False
    if pc.row_align >= ROW_ALIGN_MIN:
        return False
    return not table_covers_gutter


# ------------------------------------------------------------------
# 行几何：块内各 line 的位置
# ------------------------------------------------------------------
# ``pageio.scan_page`` 用 ``"".join`` 拼 ``Block.text``，行边界在那一刻丢掉：散文无碍，
# 但表格会把同一视觉行的多个单元格粘成一个数（如 ...13,456,607,476.0014,951,202,185.14）。
#: 判定"同一基线"的容差（pt）：同一视觉行各单元格 y 逐位相同，而行间距实测 ≥ 10pt。
SAME_ROW_TOL = 1.5

#: 行重建的容差系数（× 字号），见 :func:`row_cluster_tol`。
ROW_CLUSTER_SIZE_FACTOR = 1.0


# ``LineGeo`` 定义在 ``ir``（属 IR 契约）；本模块只消费其 ``.y`` / ``.x0`` / ``.text``。
def row_cluster_tol(font_size: float) -> float:
    """行重建的容差 = ``1.0 × 字号``（字号 ≤ 0 时退回 :data:`SAME_ROW_TOL`）。

    不依赖 find_tables；失败方向是"多分行"（跨基线折行被当新行），但不影响数字配对。
    """
    return font_size * ROW_CLUSTER_SIZE_FACTOR if font_size > 0 else SAME_ROW_TOL


def y_clusters(
    lines: Sequence[L], tol: float = SAME_ROW_TOL
) -> list[list[L]]:
    """按 y 中心把 line 聚成**视觉行**（滑动式聚簇）。簇按 y 升序，簇内按 x0 升序。"""
    out: list[list[L]] = []
    for line in sorted(lines, key=lambda ln: ln.y):
        if out and line.y - out[-1][-1].y <= tol:
            out[-1].append(line)
        else:
            out.append([line])
    for cluster in out:
        cluster.sort(key=lambda ln: ln.x0)
    return out


def n_y_clusters(lines: Sequence[L], tol: float = SAME_ROW_TOL) -> int:
    """聚簇后的视觉行数。"""
    return len(y_clusters(lines, tol))


def has_side_by_side(lines: Sequence[L], tol: float = SAME_ROW_TOL) -> bool:
    """块内是否存在"一条视觉行里有多条 line"——即拼接时会粘连。

    召回型几何判据：散文也可能命中（如"编号 + 标题"共基线），但未命中可放心。
    """
    return len(lines) >= 2 and n_y_clusters(lines, tol) < len(lines)


def glued_pairs(lines: Sequence[L], tol: float = SAME_ROW_TOL) -> list[tuple[str, str]]:
    """同一视觉行内**首尾都是数字**的相邻 line 配对 —— 即"直接测量"到的粘连。

    判据限定"首尾都是数字"是有意的：数字粘在一起不可逆，故门控对准它（窄于几何判据）。
    """
    out: list[tuple[str, str]] = []
    for cluster in y_clusters(lines, tol):
        ordered = sorted(cluster, key=lambda ln: ln.x0)
        for a, b in zip(ordered, ordered[1:]):
            ta, tb = a.text.strip(), b.text.strip()
            if ta and tb and ta[-1].isdigit() and tb[0].isdigit():
                out.append((ta, tb))
    return out


def has_glued_cells(lines: Sequence[L], tol: float = SAME_ROW_TOL) -> bool:
    """块内是否存在**粘连的数字**。这是分块层文本重建的门控（门控与病同义，无假阳性）。"""
    return bool(glued_pairs(lines, tol))
