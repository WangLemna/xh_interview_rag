"""多轮问句改写 + 实体抽取：补全指代，并列出问句涉及的公司。

设计要点（与 :mod:`finrag.gen.prompt`、:mod:`finrag.store.resolve` 的分工）：

1. **话题判定并入同一次调用**：让模型自己决定"最新问句是否依赖上文" —— 依赖则用上文补全、
   不依赖则原样输出。因此不需要第二个模型/分类器。
2. **顺带做实体抽取（NL级 NER 的轻替代）**：模型同时列出问句里提到的公司名，**含库里没有的**
   （如"宁德时代"）。这是 `resolve` 识别"用户点了库外公司的名"的唯一来源，用来避免误继承。
   **首轮（无历史）也调**：此时没有指代可补，实体是唯一但关键的产出——缺了它，首轮问句里的
   库外公司名就检测不到，会静默退回全库检索。首轮同时做**轻度归一化**（剥口语外壳、补句式），
   但禁止改数字 / 改实体名 / 扩问题维度（见 ``_REWRITE_SYSTEM`` 的归一化规则）。
3. **不注入上一轮答案**：喂给模型的只有"用户问句 + 已解析出的公司/年份"，**模型侧的回答正文
   永不进入任何 prompt**。理由见 ``prompt.py`` 的反幻觉不变式，以及数字审计
   （``audit_numbers``）只认本轮片段 —— 混入历史答案会让审计口径失效。
4. **失败即可降级**：无 Key / 超时 / 解析不出问句行，一律回落原问句、实体列表为空。
   改写是增强项，不是必经路径；调用方（``service.prepare``）随后仍有确定性的范围继承兜底。
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Final, Sequence

from finrag.config import AppConfig, get_config
from finrag.gen.client import LLMError, chat

__all__ = [
    "HistoryTurn",
    "RewriteResult",
    "build_rewrite_messages",
    "rewrite_query",
]

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class HistoryTurn:
    """一轮历史：用户原问句 + 该轮改写出的自足问句 + 解析出的公司/年份/实体。

    刻意不含该轮的模型回答 —— 记忆只记"用户说了什么、系统认出了什么"，不记答案。

    ``standalone`` 是这一轮改写后的自足问句（未改写或改写失败时为空）。喂给下一次改写时
    **优先用它**：让历史每条都独立可读，避免模型沿链条把指代一层层叠回去（叠错一层、后面全错）。
    ``entities`` 是这一轮问句提到的公司名（含库外），供范围继承兜底时判断"该不该往回找"。
    """

    query: str
    company: str | None = None
    year: int | None = None
    #: 该轮改写出的自足问句；空 = 未改写 / 改写失败（此时等价于原始 ``query``）
    standalone: str = ""
    #: 该轮问句涉及的公司名（含库外）；空 = 没提公司
    entities: tuple[str, ...] = ()


@dataclass(slots=True)
class RewriteResult:
    """一次改写的产物。``query`` 才是交给下游的问句；失败时它等于 ``original``。

    ``used_context`` 是**观测值而非模型自报**：判据是"输出是否与输入不同"。
    ⚠️ 首轮（无历史）做的是**归一化**，输出同样可能与输入不同 ⇒ 此时该字段为 ``True``
    不代表"用到了上文"，只代表"问句被规范化过"。判定是否真用上历史要看 ``used_turns``。
    ``entities`` 是模型抽出的公司名（含库外），供 :func:`finrag.store.resolve.resolve`
    区分"用户点了库外公司的名"与"用户啥也没说" —— **首轮场景下它是唯一的判据**。
    ``used_turns`` 是模型**自报**用到了哪几轮历史（窗口由 ``multiturn.rewrite_turns`` 给出）——
    纯观测，不参与任何逻辑，用于核对"它到底参考了哪一轮"。
    """

    original: str
    query: str
    used_context: bool = False
    entities: tuple[str, ...] = ()
    used_turns: tuple[int, ...] = ()
    #: 非空 = 改写失败并已回落原问句（``disabled`` / ``LLMError.kind`` / ``empty`` / ``too_long``）
    error: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "original": self.original,
            "standalone": self.query,
            "used": self.used_context,
            "entities": list(self.entities),
            "used_turns": list(self.used_turns),
            "error": self.error,
        }


#: 改写提示词。给定历史（用户问句 + 公司/年份）+ 最新问句，输出"问句 + 实体 + 依据"三行。
_REWRITE_SYSTEM: Final[str] = """你是多轮问答里的「问句归一化」组件，只做改写与实体抽取，绝不回答问题。

给你若干轮历史（每轮含：该轮问句（已尽量补全为自足问句，可能已替你补齐了指代）、系统识别出的公司、年份），以及用户的最新问句。

你要输出**恰好三行**，格式严格如下：

问句：<归一化后的自足问句>
实体：<公司名 1>|<公司名 2>|...
依据：<用到的历史轮号，如 3|5；没用到就写 无>

先分清你处在哪种情形，再决定怎么做：

**A. 无历史（历史为空）—— 做「问句归一化 + 抽实体」，不做指代补全。**
没有可依赖的上文，此时你的任务只有两件：
  - 问句行：做**轻度归一化**，把口语问句整理成正式问句（见下方归一化规则）。
  - 实体行：**必须**尽力列出问句里出现的全部公司名（含库里没有的）。

**B. 有历史，且最新问句依赖上文** —— 补全指代（见规则 2）。

**C. 有历史，但最新问句自身已完整** —— **原样输出**最新问句，一个字都不要增删（见规则 2）。

规则：

1. **先挑轮次，再改写**（仅情形 B）。历史可能很长，但只有少数几轮与最新问句相关。先判断最新问句依赖哪几轮
   —— 出现代词（它 / 它们 / 该 / 这家 / 那家 / 上述等）、省略主语、或只给年份（如"2023年呢"）
   即视为依赖；最新问句自身已经完整（自己点了名、说全了指标）则不依赖。然后**只用挑中的轮次**
   补全。挑轮次的原则是"越近越优先"：通常 1~3 轮就够，**不要因为历史里有就全用上**。
2. 情形 B：用挑中的轮次把问句补成一句**不依赖上下文也能独立看懂**的完整问句。
   情形 C：**原样输出**最新问句，一个字都不要增删。
   情形 A：按下方「归一化规则」处理。

   —— **归一化规则（仅情形 A）——**
   归一化只允许做这两件事：
     · **剥口语外壳**：删掉「帮我查一下 / 请问 / 我想知道 / 麻烦问下」这类无检索意义的客套前缀与语气词。
     · **补句式**：把「宁德时代2023年净利润呢」补成「宁德时代2023年净利润是多少」这类完整问句。
   归一化**禁止**做以下三件事（做了就等于编造）：
     · **禁止改动或补写数字与年份**（「去年营收」不要改成"2022 年营收"；无历史时你没有年份锚点，改了就凭空编）。
     · **禁止改动公司名**（不要把「平安」改成"中国平安"，不要换全称简称）—— 改什么公司由别名表裁决，不由你决定。
     · **禁止扩展问题维度**（用户问「营收」不要改成"营收及营业成本"；不要增加用户没问的指标、年份、口径）。
   若问句本身已经规范，**原样输出**即可，不必为了"看起来改过"而改写。

3. 实体行只列**公司名**：把（补全/归一化后的）问句里提到的公司**全部**列出，用 `|` 分隔；
   即使是库里没见过、历史里也没有的公司（如"宁德时代"）也要列。
   不要列年份、指标名（营业总收入 / 净利润）、货币、人名。一个公司都没有就写 `无`。
   **情形 A 下这一步最重要**：首轮没有历史兜底，你在实体行写下的名字是范围层唯一的"用户点了名"证据，
   漏列会让系统拿别家公司的数字来回答。
4. 依据行只写**轮号**（阿拉伯数字，多个用 `|` 分隔，不要写字）；最新问句自足时、以及情形 A 时为 `无`。
5. 情形 B 下**只能**使用挑中的轮次里出现过的公司名与年份；不得引入历史里没有的公司、数字或事实。
6. **对比类问句**（"A 和 B 对比""A 与 B 谁高"）必须把 A、B 都列进实体行。
7. 不做单位换算、不给估计值、不把事情问到历史里没有的维度上。
8. 除这三行外**不要输出任何其他内容**。
"""

#: 问句行 / 实体行 / 依据行的前缀（容忍中英文冒号、几种叫法）
_ASK_RE: Final[re.Pattern[str]] = re.compile(
    r"^\s*(?:问句|改写后问句|改写|问题)\s*[:：]\s*(.+)$"
)
_ENT_RE: Final[re.Pattern[str]] = re.compile(
    r"^\s*(?:实体|公司|涉及公司|公司名)\s*[:：]\s*(.+)$"
)
_USED_RE: Final[re.Pattern[str]] = re.compile(
    r"^\s*(?:依据|轮次|使用轮次|参考轮次|用了)\s*[:：]\s*(.+)$"
)
#: 实体分隔符：中文顿号/逗号、英文逗号、竖线、分号、斜杠
_ENT_SPLIT: Final[re.Pattern[str]] = re.compile(r"[|｜、,，;；/]+")
#: 表示"没有实体"的占位
_ENT_NONE: Final[frozenset[str]] = frozenset({"无", "none", "无实体", "-", "—", "n/a"})


def _fmt_turn(i: int, t: HistoryTurn) -> str:
    # 优先用该轮**改写后的自足问句**：历史里每条都独立可读，模型不必沿链条叠指代。
    # 没改写成功时 ``standalone`` 为空，自动回落到用户原问句。
    bits = [f"第{i}轮 用户问：{t.standalone or t.query}"]
    if t.company:
        bits.append(f"（识别出的公司：{t.company}）")
    if t.year:
        bits.append(f"（年份：{t.year}）")
    return " ".join(bits)


def build_rewrite_messages(
    query: str, history: Sequence[HistoryTurn]
) -> list[dict[str, str]]:
    """组装改写的 chat 消息；**纯函数**，只吃用户问句与结构化实体，不吃任何文档。

    无历史时也**照常组装**（此时走"情形 A"：只做归一化 + 抽实体，不做指代补全）——
    首轮同样需要模型抽出实体，见 :func:`rewrite_query` 的 docstring。
    """
    if history:
        lines = [_fmt_turn(i, t) for i, t in enumerate(history, 1)]
        user = "历史：\n" + "\n".join(lines) + f"\n\n最新问句：{query}"
    else:
        # ⚠️ 用拼接而非 ``.format(query=...)``：问句里出现 ``{`` / ``}``（如含花括号的
        # 科目、代码片段）会让 format 抛 KeyError/IndexError，而这里**绝不能因拼 prompt 失败**。
        user = (
            "历史：（无）\n\n"
            f"最新问句：{query}\n"
            "（无历史：不做指代补全。只做轻度归一化 + 抽出问句里的公司名，依据行写 无。）"
        )
    return [
        {"role": "system", "content": _REWRITE_SYSTEM},
        {"role": "user", "content": user},
    ]


def _strip_wrap(s: str) -> str:
    """去首尾空白与常见包裹（引号 / 书名号 / "改写为：" 前缀）。"""
    line = (s or "").strip()
    for prefix in ("改写为：", "改写后：", "改写：", "输出："):
        if line.startswith(prefix):
            line = line[len(prefix):].strip()
    return line.strip("「」“”‘’\"'《》【】 ")


def _parse_entities(raw: str) -> tuple[str, ...]:
    """实体行 → 去重后的公司名元组，过滤空值 / "无" 占位 / 过长片段。"""
    out: list[str] = []
    for piece in _ENT_SPLIT.split(raw or ""):
        name = _strip_wrap(piece)
        if not name or name.lower() in _ENT_NONE:
            continue
        if len(name) > 40:
            # 公司名不会这么长：多半是模型把整句塞进来了，丢弃。
            continue
        out.append(name)
    return tuple(dict.fromkeys(out))


def _parse_used_turns(raw: str) -> tuple[int, ...]:
    """依据行 → 轮号元组（按出现顺序去重）；``无`` / 空 → 空元组。

    只抽阿拉伯数字：模型偶尔会写成"第3轮（美的集团）"，把数字捞出来即可，其余是噪声。
    纯观测值，不参与任何逻辑。
    """
    if not raw or _strip_wrap(raw).lower() in _ENT_NONE:
        return ()
    return tuple(dict.fromkeys(int(m) for m in re.findall(r"\d+", raw)))


def _parse_reply(text: str) -> tuple[str, tuple[str, ...], tuple[int, ...]]:
    """解析模型输出 → ``(自足问句, 实体元组, 用到的轮号元组)``。

    容错：优先认带前缀的三行；认不到"问句行"时退回第一段非空行、其余为空
    （宁可不用，也不要解析出半截脏数据去污染范围解析）。
    """
    query_line = ""
    ent_raw = ""
    used_raw = ""
    first_other = ""

    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        m = _ASK_RE.match(line)
        if m and not query_line:
            query_line = _strip_wrap(m.group(1))
            continue
        m = _ENT_RE.match(line)
        if m and not ent_raw:
            ent_raw = m.group(1)
            continue
        m = _USED_RE.match(line)
        if m and not used_raw:
            used_raw = m.group(1)
            continue
        if not first_other:
            first_other = _strip_wrap(line)

    if not query_line:
        # 没有规范前缀：把第一段非空行当问句（模型没按格式来时的兜底）。
        query_line = first_other
    return query_line, _parse_entities(ent_raw), _parse_used_turns(used_raw)


def rewrite_query(
    query: str,
    history: Sequence[HistoryTurn],
    cfg: AppConfig | None = None,
) -> RewriteResult:
    """把问句归一化/改写成自足问句并抽出实体；任何异常都回落原问句，**绝不抛给调用方**。

    **首轮（无 history）也会调用**：此时没有指代可补，模型只做两件事——轻度归一化问句
    （剥口语外壳、补句式；禁止改数字/改实体名/扩维度，见 ``_REWRITE_SYSTEM``）与**抽出实体**。
    抽出实体是首轮**唯一但关键**的产出：它是 :func:`finrag.store.resolve.resolve` 判定
    "用户点了库外公司"的唯一依据（``missing`` 的唯一来源）。没有它，首轮问"宁德时代"会因
    ``missing`` 为空而不触发拦截，静默退回全库检索、拿别家公司的数字作答。

    开关关闭时原样返回（``error="disabled"``）。调用失败一律回落原问句，随后仍有
    ``resolve(prior=...)`` 的确定性范围继承兜底。
    """
    cfg = cfg or get_config()
    original = query or ""

    if not cfg.multiturn.enabled:
        return RewriteResult(original=original, query=original, error="disabled")
    # 注意：**无 history 不再早退**。首轮同样要走模型——为的是抽实体（见 docstring）。

    messages = build_rewrite_messages(original, history)
    try:
        reply = chat(messages, cfg)
    except LLMError as exc:
        # 改写失败不该影响这一次问答：降级为原问句，交给确定性继承兜底。
        logger.warning("问句改写失败（%s），回落原问句：%s", exc.kind, exc)
        return RewriteResult(original=original, query=original, error=exc.kind)
    except Exception as exc:  # noqa: BLE001
        logger.warning("问句改写异常（%s），回落原问句：%s", type(exc).__name__, exc)
        return RewriteResult(original=original, query=original, error=type(exc).__name__)

    out, entities, used_turns = _parse_reply(reply.text)
    if not out:
        return RewriteResult(original=original, query=original, error="empty")
    if len(out) > cfg.multiturn.max_chars:
        # 超长多半是模型没听话、把解释也写进来了：宁可不用，也不要污染检索。
        return RewriteResult(
            original=original, query=original, entities=entities,
            used_turns=used_turns, error="too_long",
        )

    return RewriteResult(
        original=original,
        query=out,
        used_context=(out != original.strip()),
        entities=entities,
        used_turns=used_turns,
    )
