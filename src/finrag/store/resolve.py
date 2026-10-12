"""查询侧的公司名/年份解析：把问句里的专名转成可用的 ``doc_ids``。

库内多家公司讲同一套科目，问"某公司"时易被别家公司的片挤掉；本模块从问句认出公司名
（及年份/证券代码）并定位到文档，把 ``doc_ids`` 交给 :func:`search.retrieve` —— 只收窄范围，不改排序。

别名表**首选公司表**（``companies`` / ``company_aliases``，由 ``scripts/load_companies.py``
装载）：按证券代码把"同一家公司的多种写法"收敛成一家一行，法定全称取自**年报封面**、
别名已人工剪枝。公司表不存在或为空时**自动兜底**到旧口径（各文档 ``documents.meta.aliases``）。
两条路径都只做一次全库查询 + 进程内缓存，不逐文档扫 chunks。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Final, Sequence

from finrag.alias import (
    alias_variants,
    edit_distance,
    pinyin_full,
    pinyin_initials,
)

__all__ = [
    "DocGroup",
    "QueryScope",
    "build_alias_table",
    "clear_alias_cache",
    "resolve",
    "scope_of",
]

# 年份：2000~2099，四位；要求不在数字串中间，避免抓到"128,2024"的一部分。
_YEAR_RE = re.compile(r"(?<!\d)(20\d{2})(?!\d)")

#: 抽取"证券代码"用。六位数字，边界与年份同理。
_CODE_RE = re.compile(r"(?<!\d)(\d{6})(?!\d)")


@dataclass(frozen=True, slots=True)
class DocGroup:
    """一组文档 + 它的显示标签；多实体问句里每个公司一组。

    检索侧据此做**配额配平**：按组分别召回，再按组保底选取 —— 否则片段多的一家会把
    对比问句里的另一方整个挤出 top-k。
    """

    label: str
    doc_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class QueryScope:
    """一条问句的检索范围。``company`` 空 = 问句无公司名（闭卷）；``company`` 有值而 ``doc_ids``
    空 = 认出公司但库里没有它（须报错、不可静默退回全库）；``doc_ids`` 有值即开卷（限定到那几份文档）。

    **多实体**：``companies`` 是问句里命中的全部公司（"A 和 B 对比"这种问句会有多个），
    ``doc_ids`` 是它们文档的并集。``missing`` 是"提到了、但本次覆盖不到"的公司名（库里没有，
    或没有对应年份）—— 用来把"用户点名了库外公司"这件事**显式报告**，而不是静默丢掉。
    """

    query: str
    company: str | None = None
    year: int | None = None
    code: str | None = None
    doc_ids: tuple[str, ...] = ()
    #: 匹配用的别名（可能包含法定全称），报告里要印出来以便核对
    matched_alias: str | None = None
    #: 为什么没解析出结果（``doc_ids`` 为空且无 missing 时必有值）
    reason: str = ""
    #: 本轮问句**没认出**实体，公司/年份是由上一轮补全的（多轮记忆）。前端/评测据此
    #: 区分"真命中"与"继承"，避免"看着对、其实串了上一轮"的隐性错误。
    inherited: bool = False
    #: 问句命中的**全部**公司别名（多实体）。单实体时长度为 1，无命中为空。
    companies: tuple[str, ...] = ()
    #: 提到了、但本次覆盖不到的公司名（库外，或库内没有该年份）。非空时须在答复里说明。
    missing: tuple[str, ...] = ()
    #: 按公司分组的文档集合（多实体配额配平用）。单实体时长度为 1，无命中为空。
    groups: tuple[DocGroup, ...] = ()

    @property
    def is_open_book(self) -> bool:
        return bool(self.doc_ids)


# ------------------------------------------------------------------
# 别名表
# ------------------------------------------------------------------
#: **公司表**（首选来源）：按证券代码收敛的"一家一行"权威映射。
#: 三条查询各司其职 —— 名字（简称/全称）、别名、以及把别名展开成 ``(code, year, doc_id)`` 所需的文档。
#: ⚠ `pinyin` / `pinyin_initials` 由``scripts/build_companies.py`` 用 pypinyin 从**简称**生成
#: （未装 pypinyin 时留空）。它们**只服务于查询期模糊匹配的候选缩小**，不进别名表本身 ——
#: 别名表的键仍是字面写法。少取这两列会让 Q2 的拼音索引整个失效。
_COMPANY_NAMES_QUERY: Final[str] = (
    "SELECT code, short_name, full_name, pinyin, pinyin_initials FROM companies"
)
_COMPANY_ALIAS_QUERY: Final[str] = "SELECT code, alias FROM company_aliases"
_COMPANY_DOCS_QUERY: Final[str] = (
    "SELECT meta->>'code', meta->>'year', doc_id FROM documents"
    " WHERE meta->>'code' IS NOT NULL AND meta->>'code' <> ''"
)

#: 旧口径（**兜底**）：别名表所需列。``meta.aliases`` 是入库时预计算好的公司全称候选。
#: 只在公司表不存在或为空时才走这里 —— 见 :func:`build_alias_table`。
_ALIAS_QUERY: Final[str] = (
    "SELECT doc_id, meta->>'code', meta->>'year', meta->>'company',"
    "       meta->'aliases', title"
    "  FROM documents"
)

#: 进程内别名表缓存：``{"v": (语料版本, 别名表)}``。版本取 ``(count(*), max(updated_at))``，
#: 入库提交后即变 —— 服务进程无需重启就能感知语料更新。公司表装载后其读数也进版本键，
#: 否则 ``load_companies.py`` 跑完，服务仍会用旧别名表。
_ALIAS_CACHE: dict[str, tuple[Any, dict[str, list[tuple[str, str, int | None]]]]] = {}


def clear_alias_cache() -> None:
    """清空别名表缓存（测试、``load_companies.py`` 之后强制刷新用）。"""
    _ALIAS_CACHE.clear()
    _PINYIN_CACHE.clear()


#: 拼音倒排索引（Q2）：键 = ``<键值><槽位后缀>``，一个 dict 装三类桶。只用于**缩小候选**，
#: 本身不做裁决。进程内缓存，失效键与别名表一致（`_alias_version`）。
_PINYIN_CACHE: dict[str, tuple[Any, dict[str, tuple[str, ...]]]] = {}
_SLOT_FULL: Final[str] = "\x00p"       # 全拼精确桶
_SLOT_INITIALS: Final[str] = "\x00i"   # 首字母精确桶
_SLOT_BUCKET: Final[str] = "\x00b"     # 全拼首字母分桶（兜底）


def _build_pinyin_index(
    table: dict[str, list[tuple[str, str, int | None]]]
) -> dict[str, tuple[str, ...]]:
    """别名表 → 倒排索引。键 = ``<键值> + <槽位后缀>``，一个 dict 装三类桶：

      * ``全拼 + "\\x00p"``  → 该读音的别名
      * ``首字母 + "\\x00i"`` → 该首字母串的别名
      * ``全拼首字母 + "\\x00b"`` → 按读音首字母分桶（兜底用）

    只收**长度 ≥ 2** 的键：1 个音节（"力"→li）在几百个别名里撞车严重，收进来只会把候选集
    撑大、把阈值判定的区分度冲掉。

    无 ``pypinyin`` 时全表为空 ⇒ 上层自然退回纯精确匹配（fail-closed，见
    :func:`finrag.alias.pinyin_available`）。
    """
    full: dict[str, list[str]] = {}
    initials: dict[str, list[str]] = {}
    buckets: dict[str, list[str]] = {}
    for alias in table:
        if not alias:
            continue
        pf = pinyin_full(alias)
        if len(pf) >= 2:
            full.setdefault(pf, []).append(alias)
            buckets.setdefault(pf[0], []).append(alias)
        pi = pinyin_initials(alias)
        if len(pi) >= 2:
            initials.setdefault(pi, []).append(alias)
    idx: dict[str, tuple[str, ...]] = {}
    for k, v in full.items():
        idx[k + _SLOT_FULL] = tuple(v)
    for k, v in initials.items():
        idx[k + _SLOT_INITIALS] = tuple(v)
    for k, v in buckets.items():
        idx[k + _SLOT_BUCKET] = tuple(v)
    return idx


def _pinyin_index(
    conn: Any, table: dict[str, list[tuple[str, str, int | None]]]
) -> dict[str, tuple[str, ...]]:
    """取拼音倒排索引（带进程内缓存，失效键与别名表一致）。"""
    ver = _alias_version(conn)
    hit = _PINYIN_CACHE.get("v")
    if hit is not None and hit[0] == ver:
        return hit[1]
    idx = _build_pinyin_index(table)
    _PINYIN_CACHE["v"] = (ver, idx)
    return idx


def _fuzzy_candidates(
    name: str,
    idx: dict[str, tuple[str, ...]],
    cap: int,
) -> list[str]:
    """拼音索引 → 候选别名列表（**缩小后**，至多 ``cap`` 个）。

    走三步，每步都先查后扩：① 全拼精确桶 → ② 首字母精确桶 → ③ 全拼首字母分桶（兜底，
    用于错到整串读音都对不上的情况）。**这一步就是"缩小候选"**。

    顺序不能反：对全表 879 个别名逐个算编辑距离才是真慢（与已作废的 AC 自动机同类错误）。
    """
    if not name or cap <= 0:
        return []
    out: list[str] = []
    seen: set[str] = set()

    def take(key: str) -> bool:
        """把一个桶里的别名收进候选；返回是否已满。"""
        for alias in idx.get(key, ()):
            if alias not in seen:
                seen.add(alias)
                out.append(alias)
                if len(out) >= cap:
                    return True
        return False

    pf = pinyin_full(name)
    for key in (pf + _SLOT_FULL, pinyin_initials(name) + _SLOT_INITIALS,
                (pf[:1] + _SLOT_BUCKET) if pf else ""):
        if key and take(key):
            return out
    return out


def _best_fuzzy_match(
    name: str,
    table: dict[str, list[tuple[str, str, int | None]]],
    idx: dict[str, tuple[str, ...]],
    cfg: Any,
) -> str | None:
    """错别字 → 最可能的别名；**判不出来就返回 None**（宁可漏匹配，不误匹配）。

    裁决顺序（每一关都能否决）：
      1. 候选非空，且带得回来的别名**必须真的在别名表里**（索引与表可能不同版本）；
      2. 拼音距离 ≤ ``max_pinyin_dist``；
      3. 汉字距离 ≤ ``max_char_dist``（默认 0，即只认"拼音对得上"的同音错别字）；
      4. **唯一最优**：分数并列时按 ``tie_break`` 决定取最短还是全丢。
    """
    if not name or len(name) < cfg.min_len:
        return None  # 太短的别名撞车严重，宁可漏过

    cands = _fuzzy_candidates(name, idx, cfg.pinyin_candidates)
    if not cands:
        return None

    pname = pinyin_full(name)
    scored: list[tuple[int, int, int, str]] = []
    for alias in cands:
        if alias not in table or len(alias) < cfg.min_len:
            continue  # 索引与别名表不同版本，或太短易撞车
        d_py = edit_distance(pname, pinyin_full(alias), cap=cfg.max_pinyin_dist)
        if d_py > cfg.max_pinyin_dist:
            continue
        d_ch = edit_distance(name, alias, cap=cfg.max_char_dist)
        if d_ch > cfg.max_char_dist:
            continue
        scored.append((d_py, d_ch, len(alias), alias))

    if not scored:
        return None
    scored.sort()
    best = scored[0]
    # 并列检测：距离对完全相同的另一个候选。
    tied = [s for s in scored[1:] if s[0] == best[0] and s[1] == best[1]]
    if tied:
        return min(tied, key=lambda s: s[2])[3] if cfg.tie_break else None
    return best[3]


def _alias_version(conn: Any) -> Any:
    """别名表的廉价版本号：文档读数 + （公司表存在时的）公司表读数。

    用 ``to_regclass`` 判存在而不是直接查 —— 查不存在的表会把事务置为 aborted，
    后续所有语句都会失败（这类失败很难一眼看出原因）。
    """
    with conn.cursor() as cur:
        cur.execute(
            "SELECT to_regclass('public.companies'), to_regclass('public.company_aliases')"
        )
        comps, aliases = cur.fetchone()
        cur.execute("SELECT count(*), max(updated_at) FROM documents")
        docs = cur.fetchone()
        if comps and aliases:
            cur.execute("SELECT count(*), max(updated_at) FROM companies")
            comp = cur.fetchone()
            cur.execute("SELECT count(*) FROM company_aliases")
            alias_n = cur.fetchone()[0]
            return (docs[0], docs[1], comp[0], comp[1], alias_n)
    return (docs[0], docs[1], None)


def build_alias_table(
    conn: Any, *, use_cache: bool = True, prefer_company: bool = True
) -> dict[str, list[tuple[str, str, int | None]]]:
    """库内公司名 → ``别名 -> [(代码, 年份, doc_id), ...]``。

    **首选公司表**（``companies`` / ``company_aliases``，由 ``scripts/load_companies.py`` 装载）：
    它按证券代码把"同一家公司的多种写法"收敛成一行，是别名解析的**唯一裁决来源**
    （全称来自年报封面、别名已人工剪枝）。

    **兜底**：公司表不存在或为空时，自动退回旧的 ``documents.meta.aliases`` 口径
    （``prefer_company=False`` 可强制走兜底，用于对照/回滚）。``use_cache`` 打开时
    按版本键复用进程内缓存。

    两种口径都过 :func:`finrag.alias.alias_variants`，所以"去前导 ``_``/``*``"这条归一
    在两条路径上口径一致。
    """
    ver = _alias_version(conn) if use_cache else None
    if use_cache:
        hit = _ALIAS_CACHE.get("v")
        if hit is not None and hit[0] == ver:
            return hit[1]

    table = _query_company_table(conn) if prefer_company else {}
    if not table:
        table = _query_documents_table(conn)
    if use_cache:
        _ALIAS_CACHE["v"] = (ver, table)
    return table


def _query_company_table(conn: Any) -> dict[str, list[tuple[str, str, int | None]]]:
    """公司表 → 别名表（无缓存）。表不存在时抛异常，由调用方回落。"""
    with conn.cursor() as cur:
        cur.execute(_COMPANY_NAMES_QUERY)
        comps = cur.fetchall()
        cur.execute(_COMPANY_ALIAS_QUERY)
        aliases = cur.fetchall()
        cur.execute(_COMPANY_DOCS_QUERY)
        docs = cur.fetchall()

    # code → 该公司的全部文档（别名要按 code 展开成"这条写法能命中哪几份文档"）
    docs_by_code: dict[str, list[tuple[str, int | None, str]]] = {}
    for code, year, doc_id in docs:
        key = (code or "").strip()
        y = int(year) if year and str(year).isdigit() else None
        docs_by_code.setdefault(key, []).append((key, y, doc_id))

    names: dict[str, list[str]] = {}
    seen: dict[str, set[str]] = {}
    for code, short, full, _py, _ini in comps:
        # ⚠ 解包必须与 `_COMPANY_NAMES_QUERY` 的**列数一致**。该 SQL 多取了
        #   `pinyin` / `pinyin_initials` 两列（Q2 拼音索引用），这里必须一并接住 ——
        #   列数对不上时会在**每次** `build_alias_table` 里抛 ValueError，且不在 try 内，
        #   会一路炸穿 resolve（检索完全不可用，不是降级）。改 SQL 务必回看这里。
        for n in (short, full):
            for name in alias_variants(n or ""):
                s = seen.setdefault(code, set())
                if name and name not in s:
                    s.add(name)
                    names.setdefault(code, []).append(name)
    for code, alias in aliases:
        for name in alias_variants(alias or ""):
            s = seen.setdefault(code, set())
            if name and name not in s:
                s.add(name)
                names.setdefault(code, []).append(name)

    table: dict[str, list[tuple[str, str, int | None]]] = {}
    for code, code_names in names.items():
        entries = docs_by_code.get(code)
        if not entries:
            continue
        for name in code_names:
            table.setdefault(name, []).extend(entries)
    return table


def _query_documents_table(conn: Any) -> dict[str, list[tuple[str, str, int | None]]]:
    """旧口径：从 ``documents.meta.aliases`` 建别名表（**兜底路径**，无缓存）。"""
    with conn.cursor() as cur:
        cur.execute(_ALIAS_QUERY)
        docs = cur.fetchall()

    table: dict[str, list[tuple[str, str, int | None]]] = {}

    def add(alias: str | None, code: str, year: int | None, doc_id: str) -> None:
        # 一个写法可能派生出多个键（``_ST高斯`` → 同时收 ``_ST高斯`` 与 ``ST高斯``）：
        # 归一在这里做，**查询期即时生效**，无需重灌库或改 ``documents.meta``。
        for name in alias_variants(alias or ""):
            table.setdefault(name, []).append((code, year, doc_id))

    for doc_id, code, year, company, aliases, title in docs:
        y = int(year) if year and str(year).isdigit() else None
        code = code or ""
        short = (company or "").strip() or _short_from_title(title)
        # 清单面短名无条件收录，是别名体系里唯一可信的一层。
        add(short, code, y, doc_id)
        add(_short_from_title(title), code, y, doc_id)
        # 全称候选已在入库时按归属过滤过，直接收录。
        for name in (aliases or []):
            add(name, code, y, doc_id)

    return table


def _short_from_title(title: str | None) -> str | None:
    """``"美的集团 2024年年度报告"`` → ``"美的集团"``。"""
    if not title:
        return None
    return title.split(" ", 1)[0].strip() or None


def _fuzzy_cfg(cfg: Any = None) -> Any:
    """取 ``FuzzyConfig``；拿不到就返回 ``None``（**调用方必须能处理 None**）。

    ⚠ ``resolve`` 是纯查询层、不强依赖 ``AppConfig``（测试常直接传 ``alias_table`` 进来）。
    所以这里刻意**不硬 import** ``get_config``，而是让调用方决定 —— 拿不到就等于关掉模糊，
    退化到 Q2 之前的行为，而不是抛异常。
    """
    if cfg is None:
        return None
    return getattr(cfg, "fuzzy", None)


# ------------------------------------------------------------------
# 解析
# ------------------------------------------------------------------
def resolve(
    query: str,
    conn: Any,
    alias_table: dict[str, list[tuple[str, str, int | None]]] | None = None,
    *,
    prior: QueryScope | None = None,
    entities: Sequence[str] = (),
    cfg: Any = None,
) -> QueryScope:
    """从问句里解析公司名与年份，定位到文档集合；支持**一句话多个公司**。

    - ``prior``：上一轮已解析出的范围（多轮记忆）。只有在本轮既没命中库内公司、也没有六位代码、
      又没有任何"点了名"的证据时，才沿用上一轮。
    - ``entities``：改写层抽出的公司名（含**库外**）。库里对不上的会进 ``missing`` —— 这让
      "用户点名了一家库外公司"不再被误当作"用户啥也没说"，从而**不会触发继承**（继承会把
      对比/多实体问句答成上一家公司）。
    - ``cfg``：``AppConfig``，取其中的 ``fuzzy`` 段。``None`` 时现取全局配置。

    命中多家时 ``doc_ids`` 取并集，由生成侧依据多份片段作答；``missing`` 非空时不可静默丢弃。
    """
    q = query or ""
    year_m = _YEAR_RE.search(q)
    year = int(year_m.group(1)) if year_m else None

    code_m = _CODE_RE.search(q)
    code = code_m.group(1) if code_m else None

    table = alias_table if alias_table is not None else build_alias_table(conn)

    # 年份口径：本轮有就用本轮的，否则沿用上一轮 —— "宁德时代和它对比呢"里的"它"通常指同一期。
    eff_year = year if year is not None else (prior.year if prior is not None else None)

    # ---- ① 收集问句里命中的**全部**公司（问句字面 + 改写层实体列表），按年份筛文档 ----
    # 模糊匹配（Q2）只在**精确一个都没命中**的名字上启用；索引按需构建并缓存，
    # 语料版本一变自动重建（与别名表同一把失效键）。
    fcfg = _fuzzy_cfg(cfg)
    pidx = _pinyin_index(conn, table) if fcfg is not None and fcfg.enabled else None
    groups, ids, missing = _collect_companies(
        q, table, entities, eff_year, fuzzy_cfg=fcfg, pinyin_idx=pidx
    )
    if ids:
        return QueryScope(
            query=q, company=groups[0].label if groups else None,
            year=eff_year, code=code, doc_ids=ids,
            matched_alias="、".join(g.label for g in groups) or None,
            companies=tuple(g.label for g in groups), groups=groups, missing=missing,
        )

    if code:
        # 有六位代码但无名字时直接按代码找；同一代码可能对应多份年报，按年份收窄。
        hits = [h for hits in table.values() for h in hits if h[0] == code]
        if eff_year is not None:
            hits = [h for h in hits if h[1] == eff_year]
        cids = tuple(sorted({h[2] for h in hits}))
        if cids:
            return QueryScope(query=q, company=None, year=eff_year, code=code,
                              doc_ids=cids, matched_alias=f"代码 {code}",
                              groups=(DocGroup(label=f"代码 {code}", doc_ids=cids),))
        have = sorted({h[1] for h in hits if h[1] is not None})
        return QueryScope(
            query=q, company=None, year=eff_year, code=code, doc_ids=(),
            matched_alias=f"代码 {code}",
            reason=(f"库里没有 {code} {eff_year} 年那份（现有：{have or '无'}）" if eff_year
                    else f"库里没有代码 {code} 的文档"),
        )

    # ---- ③ 点了名但本次覆盖不到 ⇒ 明确报告，**不继承**（继承会把问句答成上一家公司）----
    if missing:
        return QueryScope(
            query=q, company=None, year=eff_year, code=code, doc_ids=(),
            matched_alias="、".join(missing), missing=missing,
            reason=_missing_reason(missing, table, eff_year),
        )

    # ---- ④ 啥也没认出 ⇒ 多轮记忆：沿用上一轮的实体（年份本轮优先、否则沿用）----
    if prior is not None and prior.company:
        return _inherit_scope(q, table, prior, year, code)

    return QueryScope(query=q, company=None, year=eff_year, code=code, doc_ids=())


def _collect_companies(
    q: str,
    table: dict[str, list[tuple[str, str, int | None]]],
    entities: Sequence[str],
    year: int | None,
    *,
    fuzzy_cfg: Any = None,
    pinyin_idx: dict[str, tuple[str, ...]] | None = None,
) -> tuple[tuple[DocGroup, ...], tuple[str, ...], tuple[str, ...]]:
    """收集问句涉及的公司 → ``(命中的公司组, 其文档并集, 覆盖不到的缺失名)``。

    名字来源两处：① 问句**字面**命中的别名；② 改写层给的**实体列表**（与别名表对账，能对上
    库的也算命中）。同一家公司在别名表里有很多写法（短名 / 全称 / 去后缀变体），故**按文档集合
    归组**：落进同一批文档的别名只留一个标签，优先级是"问句里出现过的" > "更短的"。
    "库内但对不上年份"与"库里根本没有的实体"都进 ``missing``。

    第三条来源（Q2 模糊匹配）：**只对"精确匹配一个都没命中"的名字**做，且命中后仍按同一套
    归组/年份规则处理。精确匹配永远优先—— 模糊只填补空缺，绝不改变已确定的结果。
    这样"库外公司"不会被模糊拉成库内公司（它本来就没有任何精确命中，但拼音/首字母都对不上）。

    返回的组是**按公司分的**（每组一个 ``DocGroup``），不是把所有文档并成一组 —— 检索侧要靠
    这个划分做配额配平。
    """
    names: list[str] = [a for a in table if a and a in q]
    matched_entities: set[str] = set()
    unmatched_entities: list[str] = []
    for e in entities:
        hits = [a for a in table if a and (a in e or e in a)]
        names += hits
        if hits:
            matched_entities.add(e)
        else:
            # 一个都没命中才进模糊候选。**没有这一步，模糊就会抢走精确匹配的活** ——
            # 而"能精确命中却去猜"正是最容易答错公司的场景。
            unmatched_entities.append(e)

    # ---- Q2 模糊匹配：只补空缺，且必须**唯一**才采纳 ----
    if fuzzy_cfg is not None and getattr(fuzzy_cfg, "enabled", False) and pinyin_idx:
        for e in unmatched_entities:
            hit = _best_fuzzy_match(e, table, pinyin_idx, fuzzy_cfg)
            if hit:
                names.append(hit)

    uniq = list(dict.fromkeys(names))

    groups: dict[tuple[str, ...], list[str]] = {}
    year_groups: dict[tuple[str, ...], list[str]] = {}
    for alias in uniq:
        sel = [h for h in table[alias] if year is None or h[1] == year]
        if sel:
            groups.setdefault(tuple(sorted({h[2] for h in sel})), []).append(alias)
        else:
            # 年份对不上：按"全量文档"归组，免得同一家公司的全称变体把 missing 刷屏。
            year_groups.setdefault(
                tuple(sorted({h[2] for h in table[alias]})), []
            ).append(alias)

    ids: set[str] = set()
    out: list[DocGroup] = []
    for key, aliases in groups.items():
        ids |= set(key)
        out.append(DocGroup(label=_pick_label(aliases, q), doc_ids=key))
    uncovered = [_pick_label(v, q) for v in year_groups.values()]

    # 实体列表里与别名表**完全对不上**的（库外公司），是"用户点了名"的直接证据。
    #⚠ 判定必须与上面**同源**：精确对不上、模糊也没救回来的才算库外。若这里另起一套判据，
    #   会出现"模糊已把它接上、却仍被判为库外"的自相矛盾（实体进了 missing 又进了 groups）。
    outside = [e for e in entities if e not in matched_entities]
    if outside and fuzzy_cfg is not None and getattr(fuzzy_cfg, "enabled", False) and pinyin_idx:
        outside = [
            e for e in outside
            if _best_fuzzy_match(e, table, pinyin_idx, fuzzy_cfg) is None
        ]
    missing = tuple(dict.fromkeys(list(outside) + uncovered))
    return tuple(out), tuple(sorted(ids)), missing


def _pick_label(aliases: Sequence[str], q: str) -> str:
    """同一批文档的多个别名里挑一个显示名：优先问句里出现过的，其次更短的。"""
    in_query = [a for a in aliases if a in q]
    return min(in_query or list(aliases), key=len)


def _missing_reason(
    missing: Sequence[str],
    table: dict[str, list[tuple[str, str, int | None]]],
    year: int | None,
) -> str:
    """缺失公司 → 给人看的原因；库里能对上名的补一句"现有哪些年份"。"""
    parts: list[str] = []
    for name in missing:
        years = sorted({h[1] for h in table.get(name, []) if h[1] is not None})
        if years and year is not None:
            parts.append(f"库里没有 {name} {year} 年那份（现有：{years}）")
        else:
            parts.append(f"库里没有 {name} 的文档")
    return "；".join(parts)


def _inherit_scope(
    q: str,
    table: dict[str, list[tuple[str, str, int | None]]],
    prior: QueryScope,
    year: int | None,
    code: str | None,
) -> QueryScope:
    """本轮问句没认出实体时，沿用上一轮的 ``company``/``code`` 定位文档。

    年份口径：**本轮有就用本轮，否则沿用上一轮** —— "2023年呢"这类换年份的追问靠这条生效。
    沿用后仍定位不到（如库里没有该公司该年份）时，返回带 ``reason`` 的空范围，由
    ``service.prepare`` 拦住并说明，**不静默退回全库**。
    """
    alias = prior.company
    y = year if year is not None else prior.year
    c = code or prior.code
    hits = list(table.get(alias) or [])
    if not hits:
        return QueryScope(
            query=q, company=alias, year=y, code=c, doc_ids=(),
            matched_alias=alias, inherited=True, companies=(alias,),
            reason=f"沿用上一轮的「{alias}」，但库里已无该文档",
        )

    sel = [h for h in hits if y is None or h[1] == y]
    ids = tuple(sorted({h[2] for h in sel}))
    if ids:
        return QueryScope(
            query=q, company=alias, year=y, code=c, doc_ids=ids,
            matched_alias=alias, inherited=True, companies=(alias,),
            groups=(DocGroup(label=alias, doc_ids=ids),),
        )

    have = sorted({h[1] for h in hits if h[1] is not None})
    return QueryScope(
        query=q, company=alias, year=y, code=c, doc_ids=(),
        matched_alias=alias, inherited=True, companies=(alias,),
        reason=(f"沿用上一轮的「{alias}」：库里没有 {y} 年那份（现有：{have}）" if y
                else f"沿用上一轮的「{alias}」，但未能定位到文档"),
    )


def scope_of(
    query: str,
    conn: Any,
    prior: QueryScope | None = None,
    entities: Sequence[str] = (),
) -> QueryScope:
    """``resolve`` 的便捷包装：单次查询用这个，别自己管别名表缓存。"""
    return resolve(query, conn, prior=prior, entities=entities)
