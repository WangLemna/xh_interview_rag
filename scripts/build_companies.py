"""建「公司级权威信息表」（离线，只读库 + 读清单）。

把散在各份"documents.meta"里的公司写法（清单短名/正文抽出的全称/曾用名/标题短名）
按证券代码收敛成"一家一行"，作为后续"resolve"别名解析的数据基础。

它不是全市场词典：只覆盖库内公司，识别不了库外公司。

全称的来源有优先级：先取"data/company_full_names.csv"（由"scripts/extract_full_names.py"
从年报封面抽出的公司法定全称 —— 权威且与语料对齐），没有该文件或该 code 缺失时，
才退回"从meta.aliases里挑"的启发式（_pick_full_name）。

输出 CSV:

    code,简称,全称,别名,年份,n_docs,拼音,拼音首字母

- 简称：清单面短名（别名体系里唯一可信的一层）。
- 全称：按出现顺序取第一个"像法定全称"的写法（以 公司/集团/控股 结尾、且比简称长）。
  正文抽出的全称在"meta.aliases"里是按频次降序的，所以取第一个 —— 不取最长：
  子公司名往往更长（带地名/业务词），"最长"会把子公司误当母公司全称。没有则空。
- 别名：全部写法去重后以"|"连接（含简称、全称，短的在前）；每个写法另经
  "finrag.alias.alias_variants()"补出去前导 "_"/"*"的归一形态
  （"_ST高斯"→ 同时收 "ST高斯"），与查询期 / 入库期同口径。
- 年份 / n_docs：库内该公司的年份集合（"|"连接）与文档数
  （清单有、库内无时 "n_docs=0"，如被拒的扫描件）。
- 拼音 / 拼音首字母：由简称生成。

同时写一份复核清单 "<out>.review.txt"：只在一侧的 code（清单有库无 / 库有清单无）、缺全称的行。

用法:

    C:\\Python312\\python.exe scripts\\build_companies.py
    C:\\Python312\\python.exe scripts\\build_companies.py --out data/companies.csv

退出码：0 已写出；1 清单不可读 / 连不上库。
"""

from __future__ import annotations

import argparse
import csv
import sys
import traceback
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from finrag.config import get_config
from finrag.console import use_utf8_stdout
from finrag.alias import alias_variants
from finrag.manifest import load_manifest

DEFAULT_OUT = PROJECT_ROOT / "data" / "companies.csv"
# 法定全称的权威来源（由 "scripts/extract_full_names.py"从年报封面抽出）。存在就优先用它。
DEFAULT_NAMES = PROJECT_ROOT / "data" / "company_full_names.csv"

# 人工补充的追加别名："code → 额外写法"（如更名前的旧名）。
# "extract_full_names.MANUAL_OVERRIDES"决定"定稿全称"，
# "EXTRA_ALIASES"补"还能命中的写法"（可多个）。
# 格式:
#   EXTRA_ALIASES = {
#       "600257": ("大湖水殖股份有限公司",),   # 2025 更名前
#   }
EXTRA_ALIASES: dict[str, tuple[str, ...]] = {
    # 更名前的本公司法定名（旧名也要能命中）。子公司名一律不收。
    "000401": ("唐山冀东水泥股份有限公司",),
    "002245": ("江苏蔚蓝锂芯股份有限公司",),
    "002457": ("宁夏青龙管业集团股份有限公司",),
    "300225": ("金力泰",),
    "300242": ("广东佳兆业佳云科技股份有限公司",),
    "600257": ("大湖水殖股份有限公司",),
    "600743": ("华远地产股份有限公司",),
    "600844": ("丹化化工科技股份有限公司",),
}

# 像"法定全称"的结尾（用于从别名里挑"全称"）。
_FULL_TAILS: tuple[str, ...] = ("公司", "集团", "控股")

# 法定后缀的优先层级：越靠前越像"本公司法定全称"。带"有限公司"的通常就是法人主体，
# 而裸"集团"/"控股"常只是集团简称或控股平台名。
_LEGAL_TIERS: tuple[str, ...] = ("股份有限公司", "有限公司", "公司", "集团", "控股")


def _full_tier(name: str) -> int:
    for i, suf in enumerate(_LEGAL_TIERS):
        if name.endswith(suf):
            return i
    return len(_LEGAL_TIERS)


def _has_pinyin() -> bool:
    try:
        import pypinyin
    except ImportError:
        return False
    return True


def _pinyin(name: str) -> tuple[str, str]:
    """简称 → (全拼, 首字母)；未装 pypinyin 或名称为空时返回空串。"""
    if not name:
        return "", ""
    try:
        from pypinyin import Style, lazy_pinyin
    except ImportError:
        return "", ""
    full = "".join(lazy_pinyin(name, style=Style.NORMAL))
    initials = "".join(lazy_pinyin(name, style=Style.FIRST_LETTER))
    return full, initials


def _add(target: list[str], seen: set[str], name: str | None) -> None:
    """把名字及其归一形态去重保序地塞进 "target"。

    归一走:"finrag.alias.alias_variants()：
    "_ST高斯"会同时产出"_ST高斯"与"ST高斯"。
    """
    for n in alias_variants(name or ""):
        if n not in seen:
            seen.add(n)
            target.append(n)


def _pick_full_name(names: Iterable[str], short: str) -> str:
    """从全部写法里挑"最像法定全称"的一个。

    排序键 = 法定后缀层级（"股份有限公司">"有限公司">"公司">"集团"/"控股"），
    同层内取出现顺序最早的（"meta.aliases"里正文抽出的全称是按频次降序的）；
    优先"比简称长"的候选。
    """
    cands = [(i, n) for i, n in enumerate(names) if n and n.endswith(_FULL_TAILS)]
    if not cands:
        return ""
    longer = [x for x in cands if len(x[1]) > len(short or "")]
    pool = longer or cands
    pool.sort(key=lambda x: _full_tier(x[1]))  # 稳定排序 ⇒ 同层保持出现顺序
    return pool[0][1]


def _short_from_title(title: str | None) -> str | None:
    """ "美的集团 2024年年度报告"→"美的集团"；与 "finrag.store.resolve._short_from_title()"同口径。"""
    if not title:
        return None
    return title.split(" ", 1)[0].strip() or None


def _read_full_names(path: Path) -> tuple[dict[str, str], int]:
    """读"extract_full_names.py"的产物 → "(code → 法定全称, 多年不一致的家数)"。

    这是权威来源（年报封面，公司自己披露）。
    同一code的多年报若抽到不同全称，取最新年份那个
    更名后的法定名就是"当前"权威名；同时把家数回报出来（明细见"company_full_names.csv.review.txt"），
    需人工确认（旧名若也要能命中，应作为别名保留，而不是当全称）。
    """
    if not path.is_file():
        return {}, 0
    by_code: dict[str, list[tuple[int, str]]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            code = (row.get("code") or "").strip()
            name = (row.get("全称") or "").strip()
            y = str(row.get("year") or "")
            if code and name:
                by_code.setdefault(code, []).append((int(y) if y.isdigit() else -1, name))

    out: dict[str, str] = {}
    n_conflict = 0
    for code, vals in by_code.items():
        distinct = {n for _y, n in vals}
        if len(distinct) > 1:
            n_conflict += 1
        # 多个不同写法（多为更名，也可能抽到子公司）时：法定后缀层级优先
        # （股份有限公司 > 有限公司 > … > 集团），同层再取最新年份。
        out[code] = min(vals, key=lambda t: (_full_tier(t[1]), -t[0]))[1]
    return out, n_conflict


def _read_library(conn: Any) -> list[dict[str, Any]]:
    """库内 documents → 逐文档一行（code / company / year / aliases / title）。"""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT meta->>'code', meta->>'company', meta->>'year',"
            "       meta->'aliases', title FROM documents"
        )
        rows = cur.fetchall()
    out: list[dict[str, Any]] = []
    for code, company, year, aliases, title in rows:
        out.append({
            "code": (code or "").strip(),
            "company": (company or "").strip() or None,
            "year": int(year) if year and str(year).isdigit() else None,
            "aliases": [str(a).strip() for a in (aliases or []) if str(a).strip()],
            "title": title,
        })
    return out


def build_rows(
    manifest: Any, docs: list[dict[str, Any]], full_names: dict[str, str] | None = None
) -> tuple[list[dict[str, str]], int]:
    """清单 + 库内文档 → 按 code 归并的公司行（已排序），并回报"全称取自权威源"的家数。

    "full_names"（"code → 法定全称"，来自年报封面）非空时优先用它；
    否则退回 :func:`_pick_full_name` 的启发式（从 "meta.aliases"里挑）。
    """
    full_names = full_names or {}
    # code → {简称候选, 别名(保序), 年份集合, 清单文档数, 库内文档数}
    agg: dict[str, dict[str, Any]] = {}

    def bucket(code: str) -> dict[str, Any]:
        return agg.setdefault(
            code,
            {
                "shorts": [], "short_seen": set(),
                "names": [], "seen": set(), "years": set(),
                "n_manifest": 0, "n_db": 0,
            },
        )

    for rec in manifest:
        b = bucket(rec.code)
        _add(b["shorts"], b["short_seen"], rec.company)   # 清单短名是权威，直接进
        _add(b["names"], b["seen"], rec.company)
        for a in rec.aliases:
            _add(b["names"], b["seen"], a)
        if rec.year:
            b["years"].add(rec.year)
        b["n_manifest"] += 1

    for d in docs:
        code = d["code"]
        if not code:
            continue
        b = bucket(code)
        if d["company"]:
            _add(b["names"], b["seen"], d["company"])
        for a in d["aliases"]:
            _add(b["names"], b["seen"], a)
        _add(b["names"], b["seen"], _short_from_title(d["title"]))
        if d["year"]:
            b["years"].add(d["year"])
        b["n_db"] += 1

    rows: list[dict[str, str]] = []
    n_authoritative = 0
    for code in sorted(agg):
        b = agg[code]
        names: list[str] = b["names"]
        short = (b["shorts"][0] if b["shorts"] else (names[0] if names else ""))
        # 全称：权威源优先（年报封面），没有才退回"从 meta.aliases 里挑"的启发式
        full = full_names.get(code) or _pick_full_name(names, short)
        if full_names.get(code):
            n_authoritative += 1
        if full and full not in names:
            names = names + [full]   # 权威全称也必须进别名
        for extra in EXTRA_ALIASES.get(code, ()):
            if extra and extra not in names:
                names = names + [extra]   # 人工补充（如更名前的旧名）
        # 别名列：短的在前（越短越可能是"用户口语写法"，也更易命中），全称与简称一定在内
        ordered = sorted(dict.fromkeys(names), key=lambda n: (len(n), n))
        py, ini = _pinyin(short or full)
        rows.append({
            "code": code,
            "简称": short,
            "全称": full,
            "别名": "|".join(ordered),
            "年份": "|".join(str(y) for y in sorted(b["years"])),
            # n_docs = 库内文档数（清单有、库内无的会是 0，如被拒的扫描件）
            "n_docs": str(b["n_db"]),
            "拼音": py,
            "拼音首字母": ini,
        })
    return rows, n_authoritative


def write_csv(rows: list[dict[str, str]], out_path: Path) -> None:
    """写 CSV（UTF-8-sig）。"""
    cols = ["code", "简称", "全称", "别名", "年份", "n_docs", "拼音", "拼音首字母"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def write_review(rows: list[dict[str, str]], manifest_codes: set[str], doc_codes: set[str], path: Path) -> None:
    """复核清单：两侧 code 差异 + 缺全称的行（供人工核对，不改数据）。"""
    lines: list[str] = []
    only_manifest = sorted(manifest_codes - doc_codes)
    only_db = sorted(doc_codes - manifest_codes)
    no_full = [r for r in rows if not r["全称"]]
    lines.append(f"公司数（按 code 归并）：{len(rows)}")
    lines.append(f"清单有、库内无：{len(only_manifest)}  {only_manifest}")
    lines.append(f"库内无、清单无（应为空）：{len(only_db)}  {only_db}")
    lines.append(f"缺「全称」：{len(no_full)} 家（别名表里没有像法定全称的写法）")
    for r in no_full[:50]:
        lines.append(f"  - {r['code']} {r['简称']}  别名={r['别名']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    use_utf8_stdout()
    cfg = get_config()

    ap = argparse.ArgumentParser(description="建公司级权威信息表（离线；只读库 + 读清单）")
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="输出 CSV 路径")
    ap.add_argument("--manifest", default=None, help="清单路径（默认取 config/.env 的 paths.manifest）")
    ap.add_argument("--names", default=str(DEFAULT_NAMES),
                    help="法定全称权威源（extract_full_names.py 的产物）；存在即优先采用")
    args = ap.parse_args()

    manifest_path = Path(args.manifest) if args.manifest else cfg.paths.resolve(cfg.paths.manifest)
    raw_dir = cfg.paths.resolve(cfg.paths.raw_dir)
    out_path = Path(args.out)
    names_path = Path(args.names)

    if not manifest_path.is_file():
        print(f"清单不存在：{manifest_path}（用 --manifest 或 FINRAG_PATHS__MANIFEST 指定）")
        return 1

    try:
        manifest = load_manifest(manifest_path, raw_dir=raw_dir)
    except Exception:  # noqa: BLE001
        print("清单不可读（未写文件）：")
        print(traceback.format_exc())
        return 1

    try:
        from finrag.store.writer import connect

        with connect(cfg) as conn:
            docs = _read_library(conn)
    except Exception:  # noqa: BLE001
        print("连库或查询失败（未写文件）：")
        print(traceback.format_exc())
        return 1

    full_names, n_conflict = _read_full_names(names_path)
    rows, n_auth = build_rows(manifest, docs, full_names)
    write_csv(rows, out_path)

    manifest_codes = {r.code for r in manifest}
    doc_codes = {d["code"] for d in docs if d["code"]}
    review_path = out_path.with_suffix(out_path.suffix + ".review.txt")
    write_review(rows, manifest_codes, doc_codes, review_path)

    has_py = _has_pinyin()
    summary = [
        f"公司表已写出：{out_path}",
        f"  公司数 {len(rows)}  库内文档 {len(docs)}  清单文档 {len(manifest)}",
        f"  全称：权威源(年报封面) {n_auth} 家"
        + (f"（源文件 {names_path}）" if full_names else "（未找到全称源文件，退回启发式）"),
        f"  逐文档仍存在多年不一致（更名未定稿）：{n_conflict} 家"
        + ("（已全部人工定稿）" if n_conflict == 0 else f"（见 {names_path.name}.needs_review.csv）"),
        f"  缺全称 {sum(1 for r in rows if not r['全称'])} 家；"
        f"拼音：{'已生成' if has_py else '未生成（未装 pypinyin）'}",
        f"  复核清单：{review_path}",
    ]
    if not has_py:
        summary.append("  要补拼音：C:\\Python312\\python.exe -m pip install --user pypinyin")
    summary_path = out_path.with_suffix(out_path.suffix + ".summary.txt")
    summary_path.write_text("\n".join(summary) + "\n", encoding="utf-8")
    print("\n".join(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
