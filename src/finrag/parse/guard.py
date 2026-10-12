"""语料判据：准入门槛与版式噪声判定（纯逻辑，无 I/O）。

体检与入库共用同一套阈值、判定函数与"有效字符数"口径；``PROBE_VERSION`` 是全部阈值的快照。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Final, Iterable, Iterator, Protocol

# ------------------------------------------------------------------
# 判据版本：本文件里所有阈值合起来构成一个"判据快照"，改其中任何一个都要 +1。
# ------------------------------------------------------------------
PROBE_VERSION: Final[int] = 3

# ------------------------------------------------------------------
# 阈值
# ------------------------------------------------------------------
# "空页"判据：< 100 字符即算空页。语义是"文本不足"，不是"取不到文本层"，
# 于是同一判据可一并捕获扫描件与文本层残缺的页（只剩水印、图注、页码，或表格未进文本层）。
MIN_TEXT_CHARS: Final[int] = 100
REPEAT_MIN_RATIO: Final[float] = 0.30  # 某块文本出现在 >=30% 的有文本页上 → 页眉/页脚候选
SIZE_ROUND: Final[float] = 0.5  # 字号统计取整精度
# 页边阈值按全量实测留足余量。按块上沿判定，不加容差。
EDGE_TOP: Final[float] = 0.10  # 上沿在此线之上（含）→ 页眉
EDGE_BOTTOM: Final[float] = 0.90  # 上沿在此线之下（含）→ 页脚
# 统计"正文块宽度"（供多栏判定）时排除的页边范围。语义与上面两个**不同**：
# EDGE_TOP/BOTTOM 判"某个重复文本算不算页眉页脚"，这里判"哪些块算正文、参与版式统计"。
# 取值目前一致，但必须独立命名。
CONTENT_ZONE_TOP: Final[float] = 0.10
CONTENT_ZONE_BOTTOM: Final[float] = 0.90
MULTICOL_WIDTH_RATIO: Final[float] = 0.62  # 页内正文块中位相对宽度低于此值 → 疑似多栏
BLOCK_TEXT_MAXLEN: Final[int] = 60  # 归一化后的块文本上限（超长的多半是正文，不必统计）

# 语料准入：扫描页占比达阈值的文档整份排除，不进入解析环节。
# 全扫描件取不到任何文本；半扫描件缺的那一半通常是财务报表，检索会稳定地答不全，
# 建库前拒绝优于事后打补丁。
SCAN_PURE_RATIO: Final[float] = 0.95  # 空页占比 >= 此值 → 纯扫描件
# 半扫描线取 20%。它定在 20% 而非更低的 5%，是因为实测分布在中间留了一大片空白：
# 19 份准入文档的空页率集中在 0~5.3%，两个扫描类样本是 50.2% 与 100%，5.3%~50% 无落点。
# 取"宁可放行、不要误杀"一侧：真实年报偶尔有几页印刷留白，按 5% 判会因几页空白废掉一整份。
SCAN_MIXED_RATIO: Final[float] = 0.20

# ------------------------------------------------------------------
# 类型常量
# ------------------------------------------------------------------
# 用模块级字符串常量而不是 Enum：doc_type 直接落库并存进基线 JSON，
# 而 str 子类 Enum 的 json.dumps 序列化行为在不同版本间变过。
DOC_TEXT: Final[str] = "文本型"
DOC_MIXED: Final[str] = "混合件"
DOC_SCAN: Final[str] = "纯扫描件"
DOC_UNKNOWN: Final[str] = "未知"

ZONE_HEADER: Final[str] = "页眉"
ZONE_FOOTER: Final[str] = "页脚"
# 重复但不在页边 —— 高频噪声里有一批不带公司名的纯版式文本（"□适用√不适用"、
# "单位：元币种：人民币"），它们既跨页重复又落在正文区，只能靠重复率识别。
ZONE_BODY: Final[str] = "正文（重复但非页边）"


class HasText(Protocol):
    """带 ``text`` 字段即可参与字符统计；避免本模块反向依赖提取层的数据类。"""

    text: str


# ------------------------------------------------------------------
# 文本归一化：判据依赖它（跨页重复率的 key 就是归一化后的文本）
# ------------------------------------------------------------------
def norm_block_text(text: str, max_len: int = BLOCK_TEXT_MAXLEN) -> str:
    """归一化块文本：去空白，并把连续数字串折叠成 ``#``（"2024" → "#"），跨页重复率的 key 即此。"""
    t = re.sub(r"\s+", "", text)
    t = re.sub(r"\d+", "#", t)
    return t[:max_len]


def iter_repeat_keys(blocks: Iterable[HasText]) -> Iterator[tuple[str, HasText]]:
    """一页里参与"跨页重复"统计的归一化 key，**页内已按 key 去重**，成对产出原块。

    长度区间 ``[2, BLOCK_TEXT_MAXLEN]`` 与去重规则一并算作口径，体检与解析器必须共用。
    """
    seen: set[str] = set()
    for b in blocks:
        key = norm_block_text(b.text)
        if not (2 <= len(key) <= BLOCK_TEXT_MAXLEN):
            continue
        if key in seen:
            continue
        seen.add(key)
        yield key, b


# ------------------------------------------------------------------
# 口径定义与单页判定
# ------------------------------------------------------------------
def count_page_chars(blocks: Iterable[HasText]) -> int:
    """一页的**有效字符数**，全项目唯一的定义。

    口径：块内 span 文本直接拼接后取 ``len``（不加分隔符、不剔页眉页脚、不计图片块）。
    """
    return sum(len(b.text) for b in blocks)


def is_empty_page(n_chars: int) -> bool:
    """该页是否算"空页"。注意语义是"文本不足"，不是"取不到文本层"。"""
    return n_chars < MIN_TEXT_CHARS


def in_content_zone(y0: float, y1: float, page_height: float) -> bool:
    """块是否落在正文区（即未被页边切掉），用于版式统计与多栏判定。

    与 :func:`edge_zone` 判的不是同一件事；用块的 y **区间**与正文带求交，而非上沿单点比。
    """
    if page_height <= 0:
        return False
    return y1 > page_height * CONTENT_ZONE_TOP and y0 < page_height * CONTENT_ZONE_BOTTOM


def edge_zone(y0: float, page_height: float) -> str:
    """按块**上沿**判它落在页眉 / 页脚 / 正文区。

    按上沿而非下沿：部分页眉是"公司名 + 附注说明"组成的长块，拿下沿判会错推给正文。
    """
    if page_height <= 0:
        return ZONE_BODY
    if y0 <= page_height * EDGE_TOP:
        return ZONE_HEADER
    if y0 >= page_height * EDGE_BOTTOM:
        return ZONE_FOOTER
    return ZONE_BODY


def classify_doc_type(empty_pages: int, scanned_pages: int) -> str:
    """按空页占比判形态。扫描件与混合件都会被准入守卫拒绝。"""
    if scanned_pages <= 0:
        return DOC_UNKNOWN
    ratio = empty_pages / scanned_pages
    if ratio >= SCAN_PURE_RATIO:
        return DOC_SCAN
    if ratio >= SCAN_MIXED_RATIO:
        return DOC_MIXED
    return DOC_TEXT


@dataclass(frozen=True, slots=True)
class Admission:
    """准入判定结果。"""

    eligible: bool
    doc_type: str
    empty_pages: int
    scanned_pages: int
    reason: str = ""

    @property
    def empty_ratio(self) -> float:
        return self.empty_pages / self.scanned_pages if self.scanned_pages else 0.0


def judge_admission(
    empty_pages: int, scanned_pages: int, text_page_ranges: str = ""
) -> Admission:
    """语料准入判定：只有"文本型"才进解析。

    ``reason`` 会被写进体检报告并入库，改动措辞等于改契约。
    """
    doc_type = classify_doc_type(empty_pages, scanned_pages)
    if doc_type == DOC_TEXT:
        return Admission(True, doc_type, empty_pages, scanned_pages)
    if doc_type == DOC_UNKNOWN:
        reason = "未取得任何页面统计，无法判定形态"
        return Admission(False, doc_type, empty_pages, scanned_pages, reason)

    detail = f"{empty_pages}/{scanned_pages} 页有效字符 < {MIN_TEXT_CHARS}"
    if text_page_ranges:
        detail += f"，内容充足页 = {text_page_ranges}"
    return Admission(
        False, doc_type, empty_pages, scanned_pages, f"{doc_type}：{detail}"
    )


def can_early_reject(empty_pages: int, total_pages: int) -> bool:
    """扫描途中判断"剩余页已不必再看"，可立即停止。

    空页只增不减、总数固定，``empty_so_far / total`` 是最终空页率下界；越过 ``SCAN_PURE_RATIO`` 即纯扫描件（用 ``>=`` 与 :func:`classify_doc_type` 一致）。
    """
    if total_pages <= 0:
        return False
    return empty_pages / total_pages >= SCAN_PURE_RATIO


def threshold_snapshot() -> dict[str, Any]:
    """判据快照，写进体检 JSON 的 ``thresholds`` 段。

    键名须与既有基线逐字一致 —— 比对脚本靠它区分「数据变了」与「判据变了」。
    """
    return {
        "min_text_chars": MIN_TEXT_CHARS,
        "repeat_min_ratio": REPEAT_MIN_RATIO,
        "edge_top": EDGE_TOP,
        "edge_bottom": EDGE_BOTTOM,
        "content_zone_top": CONTENT_ZONE_TOP,
        "content_zone_bottom": CONTENT_ZONE_BOTTOM,
        "multicol_width_ratio": MULTICOL_WIDTH_RATIO,
        "scan_pure_ratio": SCAN_PURE_RATIO,
        "scan_mixed_ratio": SCAN_MIXED_RATIO,
    }
