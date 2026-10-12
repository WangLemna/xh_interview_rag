"""生成侧组装：检索片段 → prompt，答案 → 数字溯源审计。

消费 ``rerank.retrieve_and_rerank`` 产出的 ``RerankResult.items``（顺序即引用顺序），
只读不改。拒答由 system 提示词中的固定话术承担，且判据可机械识别。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Final, Iterable, Sequence

from finrag.config import AppConfig, get_config

__all__ = [
    "REFUSAL_MARKER",
    "Citation",
    "NumberAudit",
    "audit_numbers",
    "build_citations",
    "build_context",
    "build_messages",
    "is_refusal",
    "system_prompt",
]

#: 拒答的固定话术（评测契约）：判据是"答案里是否含这个串"。
REFUSAL_MARKER: Final[str] = "未能从所给年报片段中找到该信息。"

#: system 提示词版本号：换措辞即换行为，历史分数不可比。
PROMPT_VERSION: Final[str] = "gen-sys-1"

_SYSTEM_PROMPT: Final[str] = f"""你是财务年报问答助手。只能依据 <片段> 中给出的原文回答问题。

回答规则：

1. **只依据片段**。不得使用片段之外的知识，不得补一句"通常情况是这样"。
   如果片段里没有，就说没有 —— 你不知道不等于可以推测。

2. **禁止推算与派生计算**（最重要的一条）。
   不得用片段中的两个或多个数字，通过加、减、乘、除、求平均等方式，
   得出一个片段里没有给出的结果。典型反例：
   - 片段给了"应付职工薪酬总额"与"员工人数" ⇒ **不能**据此计算人均薪酬；
   - 片段给了"营业收入"与"销售量" ⇒ **不能**据此计算平均单价；
   - 片段给了"研发人员数"与"员工总数" ⇒ **不能**据此计算研发人员占比。
   这类派生指标若未在片段中**逐字给出**，一律按**未披露**处理。
   只有结果本身**逐字出现在片段中**时，才允许引用它。

3. **数字照抄**。引用数字时保留原文的写法与单位，不换算、不四舍五入、
   不补齐位数（原文写 456,451,731 千元，就不要改写成 4564.5 亿元）。

4. **标注引用**。每个事实性陈述的末尾用 [编号] 标注来源，多个来源写 [2][5]。
   编号对应 <片段> 的方括号编号，不要自己编号。

5. **未披露就说不披露**。若片段中没有回答问题所需的信息，你的回答必须
   以这一句开头：「{REFUSAL_MARKER}」
   然后用一句话说明片段里**有什么**相近内容、缺的是什么。
   不要拿相近内容硬答，不要给区间或估计值。

6. **口径冲突要并列**。若不同片段对同一事项给出不同数字（年报表内多处披露
   不同口径是常态），逐一列出并说明各自出处，不要只挑一个。

回答用简体中文，简洁，不要复述问题，不要写"根据您提供的片段"这类开场白。"""


@dataclass(frozen=True, slots=True)
class Citation:
    """一条注入 prompt 的片段，带它在 prompt 里的编号；``index`` 为 1-based，与 ``[n]`` 逐字对应。"""

    index: int
    chunk: object  # RetrievedChunk；用 object 避免跨层 import 环
    label: str

    @property
    def text(self) -> str:
        return str(getattr(self.chunk, "content", ""))


def build_citations(
    items: Sequence[object],
    cfg: AppConfig | None = None,
) -> list[Citation]:
    """``RerankResult.items`` → 带编号的引用列表；``heading_path`` 是否进标签由 ``cfg.gen.include_heading_path`` 决定。"""
    cfg = cfg or get_config()
    out: list[Citation] = []
    for i, it in enumerate(items, 1):
        chunk = getattr(it, "chunk", it)
        label = str(getattr(chunk, "citation", lambda: "")())
        if not cfg.gen.include_heading_path:
            label = _label_without_heading(chunk)
        out.append(Citation(index=i, chunk=chunk, label=label))
    return out


def _label_without_heading(chunk: object) -> str:
    """去掉 ``heading_path`` 的标签，仅用于消融对照（``gen.include_heading_path=false``）。"""
    title = getattr(chunk, "title", None) or getattr(chunk, "source_path", "")
    pf = getattr(chunk, "page_from", None)
    pt = getattr(chunk, "page_to", None)
    if not pf:
        return str(title)
    pages = f"p{pf}" if pf == pt else f"p{pf}-{pt}"
    return f"{title} ｜ {pages}"


def build_context(citations: Iterable[Citation]) -> str:
    """编号片段块 → 注入 prompt 的 ``<片段>`` 正文；格式为"一行出处 + 空行 + 原文"，
    出处单独成行以免被模型当作可引用内容。
    """
    blocks: list[str] = []
    for c in citations:
        blocks.append(f"[{c.index}] {c.label}\n{c.text}")
    return "\n\n".join(blocks)


def build_messages(
    query: str,
    items: Sequence[object],
    cfg: AppConfig | None = None,
    *,
    missing: Sequence[str] = (),
) -> list[dict[str, str]]:
    """组装 chat 消息，纯函数（不碰库、不调模型）；不注入历史轮次，以免上一轮幻觉被下一轮当事实引用。

    ``missing`` 是"问句里提到、但本次片段覆盖不到"的公司（库外 / 无对应年份）。多实体问句只要
    还有一家能答就不会被拦，故必须把缺谁**显式告诉模型**，否则它会拿有资料的那家硬答对比。
    """
    cfg = cfg or get_config()
    cits = build_citations(items, cfg)
    context = build_context(cits) if cits else "（未检索到任何片段）"
    note = ""
    if missing:
        names = "、".join(missing)
        note = (
            f"<缺失资料>本次片段里没有这些公司的资料：{names}。"
            f"不要为它们编造任何数字；若问题要求与它们对比，照常回答有资料的部分，"
            f"并在答案末尾用一句话说明这几家公司的资料库中没有。</缺失资料>\n\n"
        )
    user = f"{note}<片段>\n{context}\n</片段>\n\n问题：{query}"
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def system_prompt() -> str:
    """导出 system 提示词，供断言拒答话术确实包含在其中。"""
    return _SYSTEM_PROMPT


def is_refusal(answer: str) -> bool:
    """答案是否是一次拒答；用 ``in`` 而非 ``startswith``（真实输出可能带前导换行等前缀），只用于读数。"""
    return REFUSAL_MARKER in (answer or "")


# ------------------------------------------------------------------
# 数字溯源审计
# ------------------------------------------------------------------
#: 抓数字主体，刻意不抓中文数字（"一百二十万"）。
_NUM_RE: Final[re.Pattern[str]] = re.compile(r"(?<![\d.])(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)")

#: 引用标记 ``[1]`` / ``[2][5]``，须先去掉：它们是数字，会被整批判成"未支持"。
_CITE_RE: Final[re.Pattern[str]] = re.compile(r"\[\s*\d+\s*\]")

#: 页码 ``p9`` / ``p137-138``，同理去掉：它是标签里的元信息。
_PAGE_RE: Final[re.Pattern[str]] = re.compile(r"(?<![A-Za-z0-9])p\d+(?:-\d+)?")


@dataclass(slots=True)
class NumberAudit:
    r"""答案里的数字逐个核对它在注入片段中的状态：``matched`` / ``glued`` / ``unsupported`` 三档。
    ``glued`` = 原文存在但被相邻单元格粘住（非幻觉），``unsupported`` = M5 faithfulness 初筛看的档；仅观测、不拒答，中文数字不参与匹配。
    """

    #: 作为独立数字出现在片段里（保序去重）—— 最强的一档
    matched: list[str] = field(default_factory=list)
    #: 在原文里出现但被相邻单元格粘住，非幻觉，是文本层的形态
    glued: list[str] = field(default_factory=list)
    #: 三档皆无 ⇒ 真正可疑的一档，M5 的 faithfulness 看这个
    unsupported: list[str] = field(default_factory=list)
    #: 去重前抽到的数字总数
    n_extracted: int = 0

    @property
    def n_unique(self) -> int:
        return len(self.matched) + len(self.glued) + len(self.unsupported)

    @property
    def exact_ratio(self) -> float:
        """严格口径：只有独立出现才算。"""
        return len(self.matched) / self.n_unique if self.n_unique else 1.0

    @property
    def found_ratio(self) -> float:
        """宽松口径：含"被粘连"那一档，即"能否在原文中找到"的比例。"""
        n = self.n_unique
        return (len(self.matched) + len(self.glued)) / n if n else 1.0

    def to_dict(self) -> dict[str, object]:
        return {
            "matched": self.matched,
            "glued": self.glued,
            "unsupported": self.unsupported,
            "n_extracted": self.n_extracted,
            "exact_ratio": round(self.exact_ratio, 4),
            "found_ratio": round(self.found_ratio, 4),
        }


def _normalize_number(s: str) -> str:
    """千分位逗号去掉，其余原样；仅用于比对，不用于展示。"""
    return s.replace(",", "")


#: 候选含这些字符才允许走 ``glued`` 档：是"粘连档"与"子串泛滥"的分界线。
_SEPARATORS: Final[str] = ",."


def audit_numbers(
    answer: str,
    items: Sequence[object],
    cfg: AppConfig | None = None,
) -> NumberAudit:
    """抽取答案里的数字，逐个核它在注入片段里处于哪一档；``items`` 须与 ``build_messages``
    传同一批片段，否则审计会把模型没看到的片算成"它知道"。
    """
    cfg = cfg or get_config()
    if not cfg.gen.audit_numbers:
        return NumberAudit()

    haystack = "\n".join(
        str(getattr(getattr(it, "chunk", it), "content", "")) for it in items
    )
    # 标签（标题/章节名）也注入了 prompt，故也算"模型看得到"
    if cfg.gen.include_heading_path:
        haystack += "\n" + "\n".join(c.label for c in build_citations(items, cfg))

    # 片段侧同样抽一遍数字、归一化成集合，再比集合成员：
    # 若改用子串匹配，答案 "3" 会命中片段 "456451731"，unsupported 恒为空集。
    haystack_numbers = {
        _normalize_number(n) for n in _NUM_RE.findall(haystack)
    }
    # 去逗号后的原文，供 `glued` 档做子串匹配（只对含分隔符的候选开放）。
    haystack_flat = _normalize_number(haystack)

    stripped = _CITE_RE.sub(" ", answer)
    stripped = _PAGE_RE.sub(" ", stripped)

    seen: set[str] = set()
    matched: list[str] = []
    glued: list[str] = []
    unsupported: list[str] = []
    n_total = 0
    for raw in _NUM_RE.findall(stripped):
        n_total += 1
        key = _normalize_number(raw)
        if key in seen:
            continue
        seen.add(key)
        if key in haystack_numbers:
            matched.append(raw)
        elif any(ch in raw for ch in _SEPARATORS) and key in haystack_flat:
            # 含 , 或 . ⇒ 原文里逐字存在，只是被相邻单元格粘住；
            # 纯整数不走这一档，否则会走上子串泛滥的老路。
            glued.append(raw)
        else:
            unsupported.append(raw)

    return NumberAudit(
        matched=matched, glued=glued, unsupported=unsupported, n_extracted=n_total
    )
