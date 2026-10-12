"""生成语料清单：PDF 目录 → manifest CSV。

这是新语料入库的**前置步骤** —— ``ingest.py`` 是 fail-closed，没有清单一步都走不动。
纯离线：不连库、不加载模型、不需要 GPU。

默认命名约定（可用 ``--pattern`` 覆盖）::

    <代码>_<名称简称>_<年份>年年度报告.pdf     例：000061_农产品_2023年年度报告.pdf

``company`` 会剥掉 ``ST`` / ``XD`` 等**临时标签**（``*ST`` 落盘时 ``*`` 常被替换成前导 ``_``），
并把剥标签前的**原始名**写进清单 ``aliases`` 列 —— 这样用户写"沈化"或"ST沈化"都能命中。

fail-closed：解析不出的文件、或 ``doc_key`` 撞车，一律**拒绝写出正式清单**，改写到 ``--review``
供人工处理；确要强行放行再加 ``--allow-unparsed``（此时清单只含解析成功的那些）。

用法：
    python scripts/make_manifest.py --raw-dir "D:/Agent/爬取年报/data/pdf" \
        --out "D:/Agent/爬取年报/data/corpus_manifest.csv"

退出码：0 清单已写出；1 有解析不了/撞车的文件（未写正式清单）、或参数/环境问题。
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Final, NamedTuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from finrag.config import get_config  # noqa: E402
from finrag.console import use_utf8_stdout  # noqa: E402
from finrag.manifest import ManifestError, ManifestRecord, load_manifest, write_manifest  # noqa: E402

#: 默认文件名模板：锚定到"年年度报告"结尾，故名字段里出现 ``_`` 也不影响切分。
DEFAULT_STEM_PATTERN: Final[str] = r"^(?P<code>\d{6})_(?P<name>.+)_(?P<year>20\d{2})年年度报告$"

#: 需要从 ``company`` 剥掉的前缀标签。``*ST`` 会因 ``*`` 不合法而落盘成前导 ``_``，故先剥 ``_``/``*``。
TAG_PREFIXES: Final[tuple[str, ...]] = ("ST", "XD")

#: ``--pattern`` 里未使用组时的兜底组名。
_GROUPS: Final[tuple[str, ...]] = ("code", "name", "year")


class Parsed(NamedTuple):
    """一个文件名解析出的三元组。"""

    code: str
    name: str
    year: int


def strip_tags(name: str) -> str:
    """剥掉前导标签（``*``/``_`` 与 ``ST``/``XD``），返回清洗后的简称。

    剥没了就退回原名 —— 宁可留个带标签的名字，也不要造出一个空名字。
    """
    raw = name.strip()
    s = raw.lstrip("_*").lstrip()
    for tag in TAG_PREFIXES:
        if s.startswith(tag) and len(s) > len(tag):
            s = s[len(tag):].lstrip("_*").lstrip()
            break
    return s or raw


def parse_stem(stem: str, pattern: re.Pattern[str]) -> Parsed | None:
    """文件名（不含扩展名）→ :class:`Parsed`；不匹配返回 ``None``。"""
    m = pattern.match(stem)
    if not m:
        return None
    try:
        return Parsed(m.group("code"), m.group("name"), int(m.group("year")))
    except (IndexError, ValueError):
        return None


def build_records(
    raw_dir: Path, pattern: re.Pattern[str], *, strip: bool
) -> tuple[list[ManifestRecord], list[str], list[str]]:
    """扫目录 → ``(records, 解析不了的文件名, doc_key 撞车描述)``。"""
    unparsed: list[str] = []
    records: list[ManifestRecord] = []
    by_key: dict[str, list[str]] = {}

    for pdf in sorted(raw_dir.rglob("*.pdf")):
        parsed = parse_stem(pdf.stem, pattern)
        if parsed is None:
            unparsed.append(str(pdf.relative_to(raw_dir)))
            continue
        company = strip_tags(parsed.name) if strip else parsed.name
        doc_key = f"{parsed.code}_{parsed.year}"
        rel = pdf.relative_to(raw_dir).as_posix()
        by_key.setdefault(doc_key, []).append(rel)
        records.append(ManifestRecord(
            doc_key=doc_key,
            code=parsed.code,
            company=company,
            year=parsed.year,
            path=rel,
            # 剥了标签就把原始名留成别名；没剥则 company 已是原名，无需重复。
            aliases=(parsed.name,) if company != parsed.name else (),
        ))

    dups = [f"{k} ← {' / '.join(v)}" for k, v in sorted(by_key.items()) if len(v) > 1]
    records.sort(key=lambda r: (r.code, r.year or 0, r.path))
    return records, unparsed, dups


def _report(
    raw_dir: Path,
    pattern_src: str,
    n_files: int,
    records: list[ManifestRecord],
    unparsed: list[str],
    dups: list[str],
    stripped: int,
) -> str:
    """给人看的报告（落盘，因为本机 PowerShell 不回显 stdout）。"""
    years = Counter(r.year for r in records if r.year is not None)
    lines = [
        "FinRAG 语料清单生成",
        f"语料目录：{raw_dir}",
        f"文件名模板：{pattern_src}",
        f"PDF 总数：{n_files}｜解析成功 {len(records)}｜解析失败 {len(unparsed)}",
        f"doc_key 撞车：{len(dups)}",
        f"剥标签（ST/XD）：{stripped} 份",
        "",
        "按年份：" + "｜".join(f"{y} {n}" for y, n in sorted(years.items())),
        "",
    ]
    if dups:
        lines += ["撞车明细（同一 code_year 多份文件，需人工定 doc_key 判别）："]
        lines += [f"  {d}" for d in dups]
        lines += [""]
    if unparsed:
        lines += [f"解析失败（共 {len(unparsed)}）："]
        lines += [f"  {u}" for u in unparsed[:50]]
        if len(unparsed) > 50:
            lines += [f"  … 另有 {len(unparsed) - 50} 份"]
        lines += [""]
    return "\n".join(lines)


def main() -> int:
    use_utf8_stdout()
    cfg = get_config()
    ap = argparse.ArgumentParser(description="生成语料清单（PDF 目录 → manifest CSV）")
    ap.add_argument("--raw-dir", default=str(cfg.paths.resolve(cfg.paths.raw_dir)),
                    help="语料目录（递归扫 *.pdf），默认取 paths.raw_dir")
    ap.add_argument("--out", default=str(cfg.paths.resolve(cfg.paths.manifest)),
                    help="清单输出路径，默认取 paths.manifest")
    ap.add_argument("--pattern", default=DEFAULT_STEM_PATTERN,
                    help="文件名模板（正则，匹配不含扩展名的 stem；须含 code/name/year 三个命名组）")
    ap.add_argument("--keep-tags", action="store_true",
                    help="不剥 ST/XD 标签（默认剥，并把原始名写进 aliases）")
    ap.add_argument("--allow-unparsed", action="store_true",
                    help="有解析不了的文件时仍写出清单（默认拒绝写出）")
    ap.add_argument("--review", default="", help="待处理说明落盘路径，默认 <out>.review.txt")
    ap.add_argument("--report", default="", help="报告落盘路径，默认 <out>.report.txt")
    args = ap.parse_args()

    raw_dir = Path(args.raw_dir)
    if not raw_dir.is_dir():
        print(f"语料目录不存在：{raw_dir}")
        return 1

    try:
        pattern = re.compile(args.pattern)
    except re.error as exc:
        print(f"--pattern 不是合法正则：{exc}")
        return 1
    hit = [g for g in _GROUPS if g in pattern.groupindex]
    if hit != list(_GROUPS):
        print(f"--pattern 必须含命名组 {list(_GROUPS)}，实际只有 {hit}")
        return 1

    out = Path(args.out)
    review_path = Path(args.review) if args.review else out.with_suffix(out.suffix + ".review.txt")
    report_path = Path(args.report) if args.report else out.with_suffix(out.suffix + ".report.txt")

    n_files = sum(1 for _ in raw_dir.rglob("*.pdf"))
    if n_files == 0:
        print(f"目录里没有 PDF：{raw_dir}")
        return 1

    records, unparsed, dups = build_records(raw_dir, pattern, strip=not args.keep_tags)
    stripped = sum(1 for r in records if r.aliases)
    report = _report(raw_dir, args.pattern, n_files, records, unparsed, dups, stripped)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")

    blocking = bool(unparsed or dups)
    if blocking and not args.allow_unparsed:
        note = ["", "发现问题，**未写出正式清单**（fail-closed）。", f"报告：{report_path}"]
        if unparsed:
            note.append(f"· {len(unparsed)} 份文件名不符合模板（改名或调 --pattern）")
        if dups:
            note.append(f"· {len(dups)} 处 doc_key 撞车（同一代码同一年份有多份文件）")
        note.append("确认可放行后加 --allow-unparsed 重跑。")
        review_path.parent.mkdir(parents=True, exist_ok=True)
        review_path.write_text(report + "\n".join(note), encoding="utf-8")
        print(f"清单未写出：{len(unparsed)} 份解析失败、{len(dups)} 处撞车。详见 {review_path}")
        return 1

    try:
        write_manifest(records, out)
        load_manifest(out)  # 对称出口回读自校验
    except (ManifestError, OSError) as exc:
        print(f"写出或回读校验失败：{exc}")
        return 1

    if review_path.exists():
        review_path.unlink()  # 上一轮的待处理说明已失效
    print(f"清单已写出：{out}（{len(records)} 条）；报告：{report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
