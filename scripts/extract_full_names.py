"""从年报封面抽「公司法定全称」：``code → 全称``（只读库；不加载模型、不碰语料文件）。

**为什么用年报封面**：封面第一页就是公司自己披露的法定全称（与工商登记一致），
是**离语料最近、又最权威**的来源 —— 不必依赖外部网站（无需联网），也不会与语料脱节。

抽取顺序：
1. 标签优先：``公司全称：`` / ``公司法定中文名称：`` / ``中文名称：`` 等；
2. 无标签（多数 A 股年报封面就是不写标签、名称单独成行）→ 在**头部窗口**里找
   以 ``股份有限公司`` / ``有限公司`` / ``集团`` 结尾、且**含清单简称**的最长中文串；
3. 仍无 → 记为缺失，进人工复核队列。

**交叉校验**：同一 ``code`` 的多个年份必须抽到**同一个**全称；不一致即报警进人工队列。
这道校验能把"封面排版特殊/扫描件"这类个别情况自动挑出来，**人工只需看冲突项**。

产物（UTF-8）::

    <out>                   逐文档抽取结果（code,year,简称,全称,来源,候选）
    <out>.needs_review.csv  **待勾选表**：建议全称 / 建议旧名(进别名) / 各年份值 —— 人工扫一眼
    <out>.review.txt        人工复核队列（缺失 / 多年冲突）+ 每家的候选清单
    <out>.summary.txt       汇总读数

**人工介入点只有一个**：本文件的 :data:`MANUAL_OVERRIDES`（``code → 定稿全称``）。
看完待勾选表把确认值填进去即可；**别改生成的 CSV**（重跑会被覆盖）。

用法::

    C:\\Python312\\python.exe scripts\\extract_full_names.py
    C:\\Python312\\python.exe scripts\\extract_full_names.py --out data/company_full_names.csv

退出码：0 已写出；1 连不上库。
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import traceback
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from finrag.config import get_config  # noqa: E402
from finrag.console import use_utf8_stdout  # noqa: E402

DEFAULT_OUT = PROJECT_ROOT / "data" / "company_full_names.csv"

#: **唯一人工介入点**：``code → 定稿法定全称``。
#:
#: 只在 `看 <out>.needs_review.csv`（待勾选表）之后填这里，别改生成的 CSV
#: （重跑会被覆盖）。填了的值**覆盖**自动选择，而且会把该公司的每一行都写成定稿值，
#: 于是下游"唯一值"校验必然通过。格式::
#:
#:     MANUAL_OVERRIDES = {
#:         "600257": "大湖健康产业股份有限公司",   # 2025 更名（原 大湖水殖股份有限公司）
#:     }
MANUAL_OVERRIDES: dict[str, str] = {
    # 2026-10-10 人工定稿（依据 <out>.needs_review.csv，用户逐条确认）：
    # 9 家更名 —— 取最新法定名；4 家封面抽不到 —— 人工给出。
    "000401": "金隅冀东水泥集团股份有限公司",      # 2025（原 唐山冀东水泥股份有限公司）
    "002245": "江苏蔚蓝锂芯集团股份有限公司",      # 2024（原 江苏蔚蓝锂芯股份有限公司）
    "002372": "浙江伟星新型建材股份有限公司",      # 2023（2024 封面抽到了子公司）
    "002457": "青龙管业集团股份有限公司",          # 2025（原 宁夏青龙管业集团股份有限公司）
    "002493": "荣盛石化股份有限公司",              # 封面抽不到，人工确认
    "300225": "上海金力泰化工股份有限公司",        # 封面抽不到；真实简称「金力泰」（清单误写为「金泰」）
    "300242": "广东佳云科技股份有限公司",          # 2025（原 广东佳兆业佳云科技股份有限公司）
    "600257": "大湖健康产业股份有限公司",          # 2025（原 大湖水殖股份有限公司）
    "600743": "北京华远新航控股股份有限公司",      # 2025（原 华远地产股份有限公司）
    "600844": "内蒙古金煤化工科技股份有限公司",    # 2025（原 丹化化工科技股份有限公司）
    "601628": "中国人寿保险股份有限公司",          # 封面抽不到，人工确认
    "601881": "中国银河证券股份有限公司",          # 封面抽不到，人工确认
    "603001": "浙江奥康鞋业股份有限公司",          # 2025（2024 封面抽到了子公司）
}

#: 取每个文档前几个 child 块拼成"头部窗口"（封面在最前面）。
HEAD_CHUNKS = 6
#: 头部窗口的字符上限（封面 + 第一节能覆盖住"证券简称/公司全称"）。
HEAD_CHARS = 3000

#: 带标签的全称行（容忍中英文冒号与几种叫法）。
_FULL_LABEL = re.compile(
    r"(?:公司全称|公司法定中文名称|法定中文名称|公司中文名称|中文名称|注册名称|公司名称)"
    r"\s*[:：]\s*([^\s，,。;；、]{4,30})"
)

#: 像公司名的串：中文（含括号）/字母数字 + 常见公司后缀。
_NAME_RE = re.compile(r"[\u4e00-\u9fff（）()A-Za-z0-9]{2,26}(?:股份有限公司|有限公司|集团)")

#: 噪声词：命中即丢弃（会计师/审计/委员会/年份报告等，不是本公司名）。
_NOISE = (
    "年度报告", "年报", "摘要", "目录", "释义", "会计师事务所", "审计",
    "委员会", "证券", "公告", "公司简称", "股票", "章程",
)

#: **句子碎片**标记：候选里含这些词，说明它是从一句话里被正则"吞"出来的
#: （如「共同为雅戈尔建世界级时尚集团」「公司拟向全体股东（剔除…」），一律丢弃。
#: 刻意只收"强标记"，不收单字「和/与/如」——它们在合法公司名里也会出现。
_STOP_MARKERS = (
    "的", "为", "亦", "拟", "包括", "剔除", "上述", "本公司", "共同",
    "依托", "秉承", "控股股东", "全体股东", "部分子公司", "世界级", "月完成",
    # `指`：把"释义"章节的「简称指全称」行（如「丹化科技指丹化化工科技股份有限公司」）
    # 整串吞进来的脏候选一并挡掉 —— 合法公司名里不会出现「指」。
    "指",
)

#: 法定全称的长度下限：短于此的多半是被截断的碎片（如「管理有限公司」）。
_MIN_NAME_LEN = 7

#: 法定后缀的优先层级：越靠前越像"本公司法定全称"。
_LEGAL_TIERS = ("股份有限公司", "有限公司", "公司", "集团", "控股")


def _tier(name: str) -> int:
    for i, suf in enumerate(_LEGAL_TIERS):
        if name.endswith(suf):
            return i
    return len(_LEGAL_TIERS)


def _head_window(content: str) -> str:
    """头部窗口：只保留前 ``HEAD_CHARS`` 个字符（封面区域）。"""
    return (content or "")[:HEAD_CHARS]


def _clean(name: str) -> str:
    return (name or "").strip().strip("：: 　")


def _candidates(head: str) -> list[str]:
    """头部窗口里的"像公司名"的串，按出现顺序去重（过滤噪声 / 句子碎片 / 过短）。"""
    out: list[str] = []
    for raw in _NAME_RE.findall(head):
        n = _clean(raw)
        if not n or len(n) < _MIN_NAME_LEN:
            continue
        if any(w in n for w in _NOISE) or any(w in n for w in _STOP_MARKERS):
            continue
        if n not in out:
            out.append(n)
    return out


def _extract(head: str, short: str) -> tuple[str, str]:
    """``(全称, 来源)``；来源 ∈ ``label`` / ``head`` / ``head_noword`` / ``none``。

    无标签时：在候选里挑 —— ① 优先**含清单简称**的（强约束）；② 按**后缀层级**
    （股份有限公司 > 有限公司 > … > 集团）排序；③ 同层取**头部窗口里出现最早**的
    （封面标题=本公司名，永远排在前面的子公司提及之前）。
    """
    m = _FULL_LABEL.search(head)
    if m:
        cand = _clean(m.group(1))
        if cand.endswith(("公司", "集团")) and not any(w in cand for w in _NOISE):
            return cand, "label"

    cands = _candidates(head)
    if not cands:
        return "", "none"
    with_short = [c for c in cands if short and short in c]
    pool = with_short or cands
    # 出现顺序即索引（``cands`` 已保序）
    pos = {c: i for i, c in enumerate(cands)}
    best = min(pool, key=lambda c: (_tier(c), pos[c]))
    return best, ("head" if with_short else "head_noword")


def _read_heads(conn: Any) -> dict[str, tuple[str, str]]:
    """库内每个文档 → ``(code, 头部窗口文本)``（只取前 ``HEAD_CHUNKS`` 个 child 块）。"""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT t.doc_id, t.code, t.content FROM (
                SELECT c.doc_id, d.meta->>'code' AS code, c.content,
                       ROW_NUMBER() OVER (PARTITION BY c.doc_id ORDER BY c.chunk_index) AS rn
                FROM chunks c JOIN documents d ON d.doc_id = c.doc_id
                WHERE c.role = 'child'
            ) t
            WHERE t.rn <= %s
            ORDER BY t.doc_id, t.rn
            """,
            (HEAD_CHUNKS,),
        )
        rows = cur.fetchall()
    agg: dict[str, tuple[str, str]] = {}
    for doc_id, code, content in rows:
        old = agg.get(doc_id)
        text = (old[1] if old else "") + "\n" + (content or "")
        agg[doc_id] = ((code or "").strip() or (old[0] if old else ""), text)
    return agg


def _read_doc_meta(conn: Any) -> list[dict[str, Any]]:
    """documents → 逐文档 ``{doc_id, code, year, company}``（清单短名即 company）。"""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT doc_id, meta->>'code', meta->>'year', meta->>'company' FROM documents"
        )
        rows = cur.fetchall()
    out: list[dict[str, Any]] = []
    for doc_id, code, year, company in rows:
        out.append({
            "doc_id": doc_id,
            "code": (code or "").strip(),
            "year": int(year) if year and str(year).isdigit() else None,
            "company": (company or "").strip(),
        })
    return out


def main() -> int:
    use_utf8_stdout()
    cfg = get_config()

    ap = argparse.ArgumentParser(description="从年报封面抽法定全称（只读库）")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()
    out_path = Path(args.out)

    try:
        from finrag.store.writer import connect

        with connect(cfg) as conn:
            heads = _read_heads(conn)
            docs = _read_doc_meta(conn)
    except Exception:  # noqa: BLE001
        print("连库或查询失败（未写文件）：")
        print(traceback.format_exc())
        return 1

    # 逐文档抽取
    per_doc: list[dict[str, Any]] = []
    for d in docs:
        doc_id = d["doc_id"]
        _, text = heads.get(doc_id, ("", ""))
        head = _head_window(text)
        full, src = _extract(head, d["company"])
        per_doc.append({
            "code": d["code"], "year": d["year"] or "",
            "简称": d["company"], "全称": full, "来源": src,
            "候选": "|".join(_candidates(head)[:6]),
        })

    # ---- 按 code 汇总：定稿全称 + 多年一致性校验 ----
    by_code: dict[str, list[dict[str, Any]]] = {}
    for r in per_doc:
        if r["code"]:
            by_code.setdefault(r["code"], []).append(r)

    final: dict[str, str] = {}
    settled: list[dict[str, str]] = []     # 已由 MANUAL_OVERRIDES 人工定稿
    need_manual: list[dict[str, str]] = [] # 仍需人工
    for code in sorted(by_code):
        rows = by_code[code]
        shorts = sorted({r["简称"] for r in rows if r["简称"]})
        short = shorts[0] if shorts else ""
        pairs = [
            (int(r["year"]) if str(r["year"]).isdigit() else -1, r["全称"])
            for r in rows if r["全称"]
        ]
        distinct = sorted({n for _y, n in pairs})

        override = (MANUAL_OVERRIDES.get(code) or "").strip()
        if override:
            chosen, reason = override, "人工定稿"
        elif not pairs:
            chosen, reason = "", "未抽到"
        elif len(distinct) == 1:
            chosen, reason = distinct[0], "多年一致"
        else:
            # 多个不同写法（更名 / 或抽到子公司）：层级优先，同层取最新年份
            chosen = min(pairs, key=lambda t: (_tier(t[1]), -t[0]))[1]
            reason = "多年不一致（层级+最新年份）"
        final[code] = chosen

        if reason != "多年一致":
            row = {
                "code": code,
                "简称": short,
                "建议全称": chosen,
                "建议旧名(进别名)": "|".join(n for n in distinct if n != chosen),
                "原因": reason,
                "各年份值": " / ".join(f"{y if y > 0 else '?'}:{n}" for y, n in pairs) or "（无）",
            }
            (settled if reason == "人工定稿" else need_manual).append(row)

    # 人工定稿覆盖：把该 code 的每一行都写成定稿值 —— 下游按"唯一值"取用即得定稿
    for r in per_doc:
        ov = (MANUAL_OVERRIDES.get(r["code"]) or "").strip()
        if ov:
            r["全称"] = ov
            r["来源"] = "manual"

    # 写逐文档结果
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(
            fh, fieldnames=["code", "year", "简称", "全称", "来源", "候选"],
            lineterminator="\n",
        )
        w.writeheader()
        w.writerows(per_doc)

    # 写"待勾选表"（CSV）：人工只需扫一眼、把确认值搬进 MANUAL_OVERRIDES
    pick = out_path.with_suffix(out_path.suffix + ".needs_review.csv")
    cols = ["code", "简称", "建议全称", "建议旧名(进别名)", "原因", "各年份值"]
    with pick.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, lineterminator="\n")
        w.writeheader()
        w.writerows(need_manual)

    # 写人工复核队列（含候选清单，便于直接挑）；`待勾选表` 只收**仍需人工**的
    review = out_path.with_suffix(out_path.suffix + ".review.txt")
    lines = [
        f"逐文档 {len(per_doc)} 条；公司 {len(by_code)} 家",
        f"自动定稿（多年一致）：{len(final) - len(settled) - len(need_manual)} 家",
        f"人工定稿（MANUAL_OVERRIDES）：{len(settled)} 家",
        f"仍待人工：{len(need_manual)} 家",
        "",
        "⚠ `.needs_review.csv` 的「建议旧名(进别名)」是**该 code 出现过的其它写法**，",
        "   其中可能混着**子公司名**（如 000401 的「涞水金隅冀东环保科技有限公司」）——",
        "   进别名前**须人工剪枝**：只留「更名前的本公司法定名」，子公司一律丢弃。",
        "",
        "== 仍待人工（code / 简称 / 原因 / 候选）==",
    ]
    for m in need_manual:
        cands: list[str] = []
        for r in by_code[m["code"]]:
            for c in (r["候选"] or "").split("|"):
                if c and c not in cands:
                    cands.append(c)
        lines.append(f"{m['code']}  {m['简称']}  —— {m['原因']}；建议={m['建议全称'] or '（空）'}")
        lines.append(f"    候选：{' | '.join(cands[:8]) or '（无）'}")
    if settled:
        lines += ["", "== 已人工定稿（留痕，勿再改这里；改请改 MANUAL_OVERRIDES）=="]
        for m in settled:
            lines.append(
                f"{m['code']}  {m['简称']}  → {m['建议全称']}"
                f"（各年份值：{m['各年份值']}）"
            )
    review.write_text("\n".join(lines) + "\n", encoding="utf-8")

    summary = [
        f"逐文档 {len(per_doc)} 条 → {out_path}",
        f"公司 {len(by_code)} 家：自动 {len(final) - len(settled) - len(need_manual)}"
        f" / 人工定稿 {len(settled)} / 仍待人工 {len(need_manual)}",
        f"待勾选表：{pick}",
        f"复核队列：{review}",
    ]
    (out_path.with_suffix(out_path.suffix + ".summary.txt")).write_text(
        "\n".join(summary) + "\n", encoding="utf-8"
    )
    print("\n".join(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
