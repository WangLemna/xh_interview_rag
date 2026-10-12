"""别名归一的**纯函数**工具，以及查询期的**模糊匹配原语**。

两块的共同点：**只负责给出候选，不做裁决**。名字 → doc_ids 的裁决始终在别名表
（``store.resolve``），LLM 只有提议权。

**别名归一（入库期+ 查询期共用）**
:func:`normalize_alias` / :func:`alias_variants` —— 只做一件事：去掉别名前导的
``_`` / ``*``（Windows 文件名非法字符的落盘痕迹）。``*ST高斯`` 落盘成 ``_ST高斯``，
不归一的话用户写 ``ST高斯`` 就命中不了。

**模糊匹配原语（查询期，Q2）**
:func:`pinyin_full` / :func:`pinyin_initials` / :func:`edit_distance` —— 精确子串之外的容错。
顺序铁律是"先拼音索引缩小候选，再对缩小后的候选算距离"，反了才会真的慢。

⚠ **已删除的自动抽取（2026-10-11）**：本模块原先还有 :func:`company_like` /
:func:`full_name_candidates` / :func:`ends_with_short` / :func:`strip_suffixes` 与
``LEGIT_PREFIX`` 白名单，构成"从 PDF 正文粗提公司全称 + 归属判断"的一整条自动抽取链。
现已整体移除—— **别名一律来自语料清单（人工核对）**，`writer._merge_aliases` 只保留
``record.aliases`` 一路。理由：那层自动抽取是**不可靠的来源**（会捞进「就比亚迪」这类句法碎片），
而清单侧已人工核对，不需要它在旁自动化。移除后**行为零变化**（公司表正常装载时本来就不走这层），
且**不需要重灌库**（已入库的 ``meta.aliases`` 不受影响，只是不再有新数据从这条路进来）。

⚠ **勿把 :func:`ends_with_short` 那套思路挪到查询期**：它判的是"全称 → 短名"（需合法前缀），
而短前缀过匹配（Q10）要判的是"库内短名 → 库外长名"（需识别尾部独立实体后缀），**方向相反、
判据不同**，硬套会引入新 bug。
"""

from __future__ import annotations

from functools import lru_cache
from typing import Final

__all__ = [
    "alias_variants",
    "edit_distance",
    "normalize_alias",
    "pinyin_available",
    "pinyin_full",
    "pinyin_initials",
]

#: 文件名里非法字符的落盘痕迹：``*ST高斯`` 落盘成 ``_ST高斯``（``*`` 不合法被换成前导 ``_``）。
#: 归一即"去掉前导的这串字符"——否则别名表里只有 ``_ST高斯``，用户写 ``ST高斯`` 命中不了。
_PREFIX_NOISE: Final[str] = "_*"


def normalize_alias(name: str) -> str:
    """去掉别名前导的 ``_`` / ``*``（Windows 文件名非法字符留下的痕迹）。

    只动**前导**：``_ST高斯`` → ``ST高斯``；``沈化``/``ST复华`` 原样返回。
    **不剥 ST/XD 本身** —— "是不是同一家"由调用方按证券代码归组判断，这里只保证
    ``_ST高斯`` 与 ``ST高斯`` 都能落到同一份文档上。
    """
    return (name or "").lstrip(_PREFIX_NOISE).strip()


def alias_variants(name: str) -> list[str]:
    """一个写法 → 它自己 + 归一后的写法（去前导 ``_``/``*``），去重保序。

    单一出口：查询期（``store.resolve`` 建别名表）、入库期（``writer._merge_aliases`` 预计算）
    与离线生成器都调它，口径只留一份。
    """
    out: list[str] = []
    for cand in ((name or "").strip(), normalize_alias(name)):
        if cand and cand not in out:
            out.append(cand)
    return out


# ------------------------------------------------------------------
# 查询期模糊匹配（Q2：别名匹配零容错）
# ------------------------------------------------------------------
# 设计前提（勿回退）：
#
# 1. **只缩小候选、不裁决**。本节所有函数都只回答"哪些名字可能是它"，最终采不采纳由
#    ``resolve`` 决定 —— 别名表仍是唯一的裁决者，LLM 只有提议权。
# 2. **顺序不能反**：先用拼音索引把候选集缩小，再对缩小后的候选算编辑距离。反过来对全表
#    879个别名逐个算距离才是真慢（与已作废的 AC 自动机犯的是同一个错：优化了不慢的环节）。
# 3. **没装 pypinyin 就整体退化为"无拼音"**，不是抛错 —— 容错是增强项，不得成为单点故障。

def pinyin_available() -> bool:
    """本机是否装了 ``pypinyin``。未装时下面两个函数返回空串，模糊匹配自动失效。"""
    try:
        import pypinyin  # noqa: F401
    except Exception:  # noqa: BLE001 —— 装坏了也当没装，不让检索崩
        return False
    return True


@lru_cache(maxsize=4096)
def pinyin_full(name: str) -> str:
    """中文名 → 全拼（无调号、无分隔、小写）：``"格力电器"`` → ``"gelidianqi"``。

    非汉字字符原样保留（不做音译）⇒ 混入英文/数字的错别字也能比对。
    未装 ``pypinyin`` 返回空串。**带缓存**：同一名字在别名前后各查一次，只算一次。
    """
    if not name:
        return ""
    try:
        from pypinyin import Style, lazy_pinyin
    except Exception:  # noqa: BLE001
        return ""
    try:
        return "".join(lazy_pinyin(name, style=Style.NORMAL)).lower()
    except Exception:  # noqa: BLE001
        return ""


@lru_cache(maxsize=4096)
def pinyin_initials(name: str) -> str:
    """中文名 → 首字母串：``"格力电器"`` → ``"gldq"``。未装 ``pypinyin`` 返回空串。"""
    if not name:
        return ""
    try:
        from pypinyin import Style, lazy_pinyin
    except Exception:  # noqa: BLE001
        return ""
    try:
        return "".join(lazy_pinyin(name, style=Style.FIRST_LETTER)).lower()
    except Exception:  # noqa: BLE001
        return ""


def edit_distance(a: str, b: str, *, cap: int | None = None) -> int:
    """Levenshtein 编辑距离（滚动两行，O(len(a)) 空间）。

    ``cap`` 给定时做**剪枝**：距离已超过 ``cap`` 就提前返回 ``cap + 1``。
    模糊匹配里候选集已缩小，但保守起见仍传 cap —— 单个别名不可能差到几十个字。
    """
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    if cap is not None and abs(len(a) - len(b)) > cap:
        return cap + 1  # 长度差就超了，必然超cap

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        row_min = i
        for j, cb in enumerate(b, 1):
            # 相等时走对角（免一次 `ca != cb`），否则三者取最小；用整数变量避开 min() 的开销。
            v = prev[j - 1] if ca == cb else min(prev[j - 1] + 1, cur[j - 1] + 1, prev[j] + 1)
            cur.append(v)
            if v < row_min:
                row_min = v
        if cap is not None and row_min > cap:
            return cap + 1
        prev = cur
    return prev[-1]