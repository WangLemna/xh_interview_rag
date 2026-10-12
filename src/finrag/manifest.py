"""语料清单：``doc_key`` ↔ 文件 ↔ 元数据（code / company / year）的唯一可信来源。

清单随语料走，是**文档身份（``doc_id``）**与**检索范围（company / year）**的裁决依据。
设计上是 fail-closed：清单缺失、或磁盘上的文件在清单里找不到，一律报错，
**不退回"按文件名猜"** —— 那是扩容到几千家时最危险的静默降级。

CSV 表头（必填 ``doc_key,code,company,year,path``，可选 ``category,title,aliases``）::

    doc_key,code,company,year,path,category,title,aliases
    000698_2025,000698,沈化,2025,000698_ST沈化_2025年年度报告.pdf,,,ST沈化

``path`` 相对 ``raw_dir``；``company`` 填**清单面短名**（别名体系里唯一可信的一层）。
``aliases`` 是**额外别名**（``|`` 分隔），用于承载"清洗前的原始写法"（如带 ST/XD 标签的
``ST沈化``）与曾用名 —— 这样用户写"沈化"或"ST沈化"都能命中。
"""

from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Iterable, Sequence

__all__ = [
    "ALIAS_SEP",
    "DOC_ID_LEN",
    "Manifest",
    "ManifestError",
    "ManifestRecord",
    "doc_id_for",
    "load_manifest",
    "missing_from_manifest",
    "write_manifest",
]

#: ``doc_id`` 取 sha256 前多少位十六进制。
DOC_ID_LEN: Final[int] = 16

#: 清单必填列。
REQUIRED_COLUMNS: Final[tuple[str, ...]] = ("doc_key", "code", "company", "year", "path")

#: 可选列。
OPTIONAL_COLUMNS: Final[tuple[str, ...]] = ("category", "title", "aliases")

#: ``aliases`` 列内多个别名的分隔符（别名里几乎不可能出现 ``|``）。
ALIAS_SEP: Final[str] = "|"


class ManifestError(ValueError):
    """清单本身不合法，或语料与清单对不上。"""


def doc_id_for(doc_key: str) -> str:
    """``doc_key`` → 稳定 ``doc_id``。

    只认 ``doc_key``、不认路径也不认文件名 —— 于是**重排目录、换文件名都不会改变文档身份**，
    不会出现"旧行成孤儿、新写入变重复"。
    """
    key = (doc_key or "").strip()
    if not key:
        raise ManifestError("doc_key 不能为空")
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:DOC_ID_LEN]


def _norm_posix(p: str | Path) -> str:
    """规范化成"无前导 ``./`` 的正斜杠相对路径"，用于清单内路径与磁盘路径对齐。"""
    s = Path(str(p)).as_posix()
    return s[2:] if s.startswith("./") else s


@dataclass(frozen=True, slots=True)
class ManifestRecord:
    """清单里的一行。``company`` 是清单面短名；``year`` 可为 ``None``（未知报告期）。

    ``aliases`` 是额外别名（原始写法 / 曾用名），入库时并入 ``documents.meta.aliases``。
    """

    doc_key: str
    code: str
    company: str
    year: int | None
    path: str
    category: str | None = None
    title: str | None = None
    aliases: tuple[str, ...] = ()

    @property
    def doc_id(self) -> str:
        return doc_id_for(self.doc_key)


class Manifest:
    """一份已校验的清单；按 ``doc_key`` / 路径 / 文件名三种方式检索。"""

    def __init__(self, source: Path, records: Sequence[ManifestRecord]) -> None:
        self.source = Path(source)
        self.records: tuple[ManifestRecord, ...] = tuple(records)
        self.by_doc_key: dict[str, ManifestRecord] = {}
        self.by_path: dict[str, ManifestRecord] = {}
        self.by_name: dict[str, ManifestRecord] = {}
        for r in self.records:
            self.by_doc_key[r.doc_key] = r
            self.by_path[_norm_posix(r.path)] = r
            self.by_name[Path(r.path).name] = r

    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self) -> Iterable[ManifestRecord]:
        return iter(self.records)

    def lookup(self, pdf_path: str | Path, raw_dir: str | Path | None = None) -> ManifestRecord | None:
        """按文件找记录：先按"相对 ``raw_dir`` 的路径"，再按文件名（清单只有 basename 时）。"""
        p = Path(pdf_path)
        if raw_dir is not None:
            try:
                rel = p.resolve().relative_to(Path(raw_dir).resolve())
                hit = self.by_path.get(_norm_posix(rel))
                if hit is not None:
                    return hit
            except ValueError:
                pass
        return self.by_path.get(_norm_posix(p)) or self.by_name.get(p.name)

    def relative_paths(self) -> set[str]:
        """清单里登记的全部（规范化）相对路径，供调用方与磁盘目录比对。"""
        return set(self.by_path)


def load_manifest(path: str | Path, *, raw_dir: str | Path | None = None) -> Manifest:
    """读入并校验清单；任何不合规都抛 :class:`ManifestError`（fail-closed）。

    ``raw_dir`` 只用于给出更清楚的报错（缺失文件相对于哪个目录），校验逻辑本身不依赖它。
    """
    src = Path(path)
    if not src.is_file():
        raise ManifestError(
            f"语料清单不存在：{src}。清单是扩容后**必需**的身份与元数据来源；"
            f"请生成清单（表头：{','.join(REQUIRED_COLUMNS)}）并用 FINRAG_PATHS__MANIFEST 指向它。"
        )

    # utf-8-sig：容忍 Excel 另存 CSV 时写入的 BOM。
    with src.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        header = [h.strip() for h in (reader.fieldnames or [])]
        missing_cols = [c for c in REQUIRED_COLUMNS if c not in header]
        if missing_cols:
            raise ManifestError(
                f"清单缺列：{missing_cols}（表头实为 {header}；必需列 {list(REQUIRED_COLUMNS)}）"
            )

        records: list[ManifestRecord] = []
        seen_key: set[str] = set()
        seen_path: set[str] = set()
        for i, row in enumerate(reader, start=2):  # 第 1 行是表头
            rec = _parse_row(row, i)
            if rec.doc_key in seen_key:
                raise ManifestError(f"第 {i} 行：doc_key 重复 —— {rec.doc_key!r}")
            if rec.path in seen_path:
                raise ManifestError(f"第 {i} 行：path 重复 —— {rec.path!r}")
            seen_key.add(rec.doc_key)
            seen_path.add(rec.path)
            records.append(rec)

    if not records:
        raise ManifestError(f"清单为空：{src}")
    return Manifest(src, records)


def missing_from_manifest(
    manifest: Manifest, pdfs: Iterable[str | Path], raw_dir: str | Path | None = None
) -> list[str]:
    """磁盘上有、清单里没有的 PDF 文件名（basename）。空列表 = 清单全覆盖。"""
    return [Path(p).name for p in pdfs if manifest.lookup(p, raw_dir) is None]


def _parse_row(row: dict[str, str], lineno: int) -> ManifestRecord:
    """一行 → :class:`ManifestRecord`；缺必填或年份非法即抛。"""
    def cell(name: str) -> str:
        return (row.get(name) or "").strip()

    for col in ("doc_key", "code", "company", "path"):
        if not cell(col):
            raise ManifestError(f"第 {lineno} 行：{col} 为空")

    year_raw = cell("year")
    year: int | None
    if not year_raw:
        year = None
    elif year_raw.isdigit():
        year = int(year_raw)
    else:
        raise ManifestError(f"第 {lineno} 行：year 不是整数 —— {year_raw!r}")

    return ManifestRecord(
        doc_key=cell("doc_key"),
        code=cell("code"),
        company=cell("company"),
        year=year,
        path=_norm_posix(cell("path")),
        category=cell("category") or None,
        title=cell("title") or None,
        aliases=_split_aliases(cell("aliases")),
    )


def _split_aliases(cell: str) -> tuple[str, ...]:
    """``aliases`` 单元格 → 去重保序的别名元组（``|`` 分隔）。"""
    out: list[str] = []
    seen: set[str] = set()
    for part in (cell or "").split(ALIAS_SEP):
        name = part.strip()
        if name and name not in seen:
            seen.add(name)
            out.append(name)
    return tuple(out)


#: 写盘用的列顺序（必填在前，可选在后）。
CSV_COLUMNS: Final[tuple[str, ...]] = REQUIRED_COLUMNS + OPTIONAL_COLUMNS


def write_manifest(records: Sequence[ManifestRecord], path: str | Path) -> Path:
    """把记录写成清单 CSV（``load_manifest`` 的对称出口），返回落盘路径。

    用 ``utf-8-sig``：Excel 双击打开中文不乱码，而 ``load_manifest`` 又能容忍该 BOM。
    ``lineterminator="\\n"`` 让文件在 Windows 上也不带 ``\\r``，方便 diff。
    """
    dst = Path(path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(CSV_COLUMNS)
        for r in records:
            writer.writerow([
                r.doc_key, r.code, r.company,
                "" if r.year is None else r.year,
                r.path, r.category or "", r.title or "",
                ALIAS_SEP.join(r.aliases),
            ])
    return dst
