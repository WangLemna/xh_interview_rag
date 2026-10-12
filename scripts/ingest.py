"""批量入库：解析 → 分块 → 编码 → 落库。

写入是唯一会改库的动作，单独成脚本便于重跑与观察代价。
``--force`` 只绕过入库层的跳过判定，不碰解析/向量两处缓存；要连缓存一起重算用 ``--no-cache``。

语料元数据与文档身份由**清单**（``paths.manifest`` / ``--manifest``）裁决；清单缺失、
或磁盘上有清单外的文件，一律拒绝入库（fail-closed）。

用法：
    python scripts/ingest.py            # 增量（指纹一致则跳过）
    python scripts/ingest.py --force    # 全部重写（换 embed 模型后用）

退出码：0 全部处理完（被准入拒绝/单份失败的不算失败）；1 参数/环境/清单问题。
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from finrag.config import get_config  # noqa: E402
from finrag.console import use_utf8_stdout  # noqa: E402
from finrag.manifest import (  # noqa: E402
    ManifestError,
    load_manifest,
    missing_from_manifest,
)
from finrag.parse.cache import default_cache_dir, resolve_jobs  # noqa: E402
from finrag.store.writer import (  # noqa: E402
    IngestStats,
    connect,
    ingest_many,
    status_label,
)

COLUMNS = (
    "文档",
    "状态",
    "片数",
    "子块",
    "父块",
    "表头片",
    "编码",
    "缓存命中",
    "说明",
)


def _fmt(v: Any) -> str:
    return str(v)


def _reason_with_warnings(s: IngestStats) -> str:
    """说明列：原有原因 + 分块警告条数（警告文本在报告末尾单独列明细）。"""
    parts = [p for p in (s.reason,) if p]
    if s.warnings:
        parts.append(f"{len(s.warnings)} 条分块警告")
    return "；".join(parts)


def _verdict(conn: Any, stats: list[IngestStats]) -> list[str]:
    """入库后的库内核对，只查不判。

    两条写法是为规避"大表上的病态计划"——表涨到几十万行时，原来那条查询会从秒级
    变成"跑到天亮"（实测：40 万行时该查询 25 分钟未完，且仍有 3 个实例在跑）：

    ① **5 条带条件的 count 合并成一次顺序扫描**（``count(*) FILTER``）。分开写会让
       ``chunks`` 被整表扫 5 遍，而这张表是 GB 级的（每行带 1024 维向量）。
    ② 悬空 ``parent_id`` 用 ``NOT EXISTS``（走 ``chunks_pkey`` 索引）。**不要**写
       ``parent_id NOT IN (SELECT id FROM chunks)``：40 万 id 的哈希一旦超出默认
       ``work_mem=4MB``，规划器就退回 ``Materialize`` + 逐行重扫，退化成"每行重扫
       几十万 id"，跑到天亮也跑不完。

    ``statement_timeout`` 是硬上限：核对只是"读数"，不该把整批拖住。
    """
    with conn.cursor() as cur:
        cur.execute("SET LOCAL statement_timeout = '300s'")
        cur.execute("SET LOCAL work_mem = '64MB'")
        cur.execute(
            """
            SELECT
                (SELECT count(*) FROM documents),
                t.n_all,
                t.n_child,
                t.n_parent,
                t.n_root,
                t.n_child_dense_null,
                t.n_child_sparse_null,
                (SELECT count(*) FROM chunks c
                   WHERE c.parent_id IS NOT NULL
                     AND NOT EXISTS (SELECT 1 FROM chunks p WHERE p.id = c.parent_id))
            FROM (
                SELECT
                    count(*)                                                      AS n_all,
                    count(*) FILTER (WHERE role = 'child')                        AS n_child,
                    count(*) FILTER (WHERE role = 'parent')                       AS n_parent,
                    count(*) FILTER (WHERE parent_id IS NULL)                     AS n_root,
                    count(*) FILTER (WHERE role = 'child' AND dense_vec  IS NULL) AS n_child_dense_null,
                    count(*) FILTER (WHERE role = 'child' AND sparse_vec IS NULL) AS n_child_sparse_null
                FROM chunks
            ) t
            """
        )
        row = cur.fetchone()
    labels = (
        "documents 行数",
        "chunks 行数",
        "  其中 role='child'",
        "  其中 role='parent'",
        # NULL 是正常形态，不是异常：① 父块自己当然没有父块；② **只含 1 个子块的节
        # 不配父块**（chunker 的 `if len(group) < 2: continue`），那个片就是孤儿子块。
        "parent_id 为 NULL（= 父块自身 + 单片节的孤儿子块，均正常）",
        "child 但 dense_vec 为 NULL（不该有）",
        "child 但 sparse_vec 为 NULL（不该有）",
        "parent_id 指向不存在的行（外键在，不可能非 0）",
    )
    return [f"  {lab}：{n}" for lab, n in zip(labels, row)]


def main() -> int:
    use_utf8_stdout()
    cfg = get_config()
    ap = argparse.ArgumentParser(description="批量入库（解析 → 分块 → 编码 → 落库）")
    ap.add_argument("--raw-dir", default=str(cfg.paths.resolve(cfg.paths.raw_dir)),
                    help="语料目录，默认取 paths.raw_dir")
    ap.add_argument("--manifest", default=str(cfg.paths.resolve(cfg.paths.manifest)),
                    help="语料清单 CSV（身份与元数据来源，必填），默认取 paths.manifest")
    ap.add_argument("--out", default=str(cfg.paths.resolve(cfg.paths.log_dir) / "ingest.txt"),
                    help="报告落盘路径，默认取 paths.log_dir")
    ap.add_argument("--only", default="", help="只跑文件名含该子串的")
    ap.add_argument("--force", action="store_true", help="忽略 content_hash 跳过判定，全部重写")
    ap.add_argument("--dry-run", action="store_true", help="只走解析与分块，不连库、不编码")
    ap.add_argument("--no-cache", action="store_true", help="绕过解析缓存与向量缓存（不读也不写）")
    ap.add_argument("--jobs", type=int, default=0, help="并行解析的进程数，0=自动（上限 4）")
    ap.add_argument("--cache-dir", default="", help="解析缓存目录，默认取 <paths.cache_dir>/parse")
    args = ap.parse_args()

    raw_dir = Path(args.raw_dir)
    pdfs = sorted(raw_dir.rglob("*.pdf"))  # 递归：语料按公司/年份分子目录时也要扫到
    if args.only:
        pdfs = [p for p in pdfs if args.only in p.name]
    if not pdfs:
        print(f"未找到 PDF：{raw_dir}" + (f"（--only {args.only}）" if args.only else ""))
        return 1

    # 清单是身份(doc_id)与元数据(code/company/year)的唯一来源：读不到就 fail-closed。
    manifest_path = Path(args.manifest)
    try:
        manifest = load_manifest(manifest_path, raw_dir=raw_dir)
    except ManifestError as exc:
        print(f"语料清单不可用：{exc}")
        return 1
    missing = missing_from_manifest(manifest, pdfs, raw_dir)
    if missing:
        shown = "、".join(missing[:10])
        more = f" 等 {len(missing)} 份" if len(missing) > 10 else ""
        print(f"以下文件不在语料清单里，拒绝入库：{shown}{more}")
        print(f"  清单：{manifest_path}")
        return 1

    jobs = resolve_jobs(args.jobs, len(pdfs))
    cache_dir = Path(args.cache_dir) if args.cache_dir else default_cache_dir(cfg)

    out: list[str] = []
    add = out.append
    add("FinRAG 批量入库")
    add(f"语料目录：{raw_dir}")
    add(f"清单：{manifest_path}（{len(manifest)} 条）")
    add(f"解析：进程 {jobs} ｜ 解析缓存：{'（本次绕过）' if args.no_cache else cache_dir}")
    add(f"分块：max_tokens={cfg.chunk.max_tokens} min_tokens={cfg.chunk.min_tokens} "
        f"overlap_tokens={cfg.chunk.overlap_tokens} table_units={cfg.chunk.table_units}")
    add(f"编码：{cfg.embed.dense_model} dim={cfg.embed.dim} device={cfg.embed.device} "
        f"batch={cfg.embed.batch_size}")
    add(f"模式：{'--dry-run（不连库、不编码）' if args.dry_run else ('--force（全部重写）' if args.force else '增量（指纹一致则跳过）')}")
    add("")

    stats: list[IngestStats] = []
    if args.dry_run:
        # dry-run 只走到入库前一步，不加载编码器（约 2.2GB 权重）。
        from finrag.chunk.chunker import chunk_document
        from finrag.parse.cache import iter_parsed

        t0 = time.perf_counter()
        for path, outcome in iter_parsed(
            pdfs, cfg, cache_dir=cache_dir, use_cache=not args.no_cache, jobs=jobs,
        ):
            record = manifest.lookup(path, raw_dir)
            doc_id = record.doc_id if record else ""
            if not outcome.result.ok:
                stats.append(IngestStats(
                    path, doc_id,
                    "failed" if outcome.result.fatal else "rejected",
                    reason=outcome.result.fatal or outcome.result.admission.reason,
                ))
                continue
            res = chunk_document(outcome.result.doc, cfg)
            stats.append(IngestStats(
                source_path=path,
                doc_id=doc_id,
                status="dry-run",
                n_rows=len(res.chunks),
                n_children=len(res.children()),
                n_parents=len(res.parents()),
                n_table_headers=len(res.table_headers()),
                warnings=list(res.warnings),
            ))
        add(f"解析 + 分块用时 {time.perf_counter() - t0:.1f}s（dry-run 未编码）")
    else:
        t0 = time.perf_counter()
        stats = ingest_many(
            pdfs, cfg, cache_dir=cache_dir, jobs=jobs,
            force=args.force, use_cache=not args.no_cache,
            manifest=manifest, raw_dir=raw_dir,
        )
        add(f"解析 + 编码 + 入库用时 {time.perf_counter() - t0:.1f}s")

    add("")
    add("\t".join(COLUMNS))
    for s in stats:
        add("\t".join((
            s.source_path.name,
            status_label(s.status),
            _fmt(s.n_rows),
            _fmt(s.n_children),
            _fmt(s.n_parents),
            _fmt(s.n_table_headers),
            _fmt(s.n_vectors),
            _fmt(s.n_cached),
            _reason_with_warnings(s),
        )))

    tally = Counter(s.status for s in stats)
    add("")
    add("汇总：" + "｜ ".join(f"{status_label(k)} {v}" for k, v in sorted(tally.items())))
    add(f"片合计 {sum(s.n_rows for s in stats)}"
        f"（子 {sum(s.n_children for s in stats)} / 父 {sum(s.n_parents for s in stats)}"
        f" / 表头 {sum(s.n_table_headers for s in stats)}）")
    if not args.dry_run:
        add(f"编码 {sum(s.n_vectors for s in stats)} 条，"
            f"其中缓存命中 {sum(s.n_cached for s in stats)} 条")

    warned = [s for s in stats if s.warnings]
    if warned:
        add("")
        add(f"分块警告明细（{len(warned)} 份文档，共 {sum(len(s.warnings) for s in warned)} 条）：")
        for s in warned:
            add(f"  {s.source_path.name}")
            for w in s.warnings:
                add(f"    - {w}")

    if not args.dry_run:
        conn = connect(cfg)
        try:
            add("")
            add("库内核对：")
            try:
                out.extend(_verdict(conn, stats))
            except Exception as exc:  # noqa: BLE001 —— 核对只是"读数"，失败不该连报告一起丢
                add(f"  （库内核对失败：{type(exc).__name__}: {exc}）")
        finally:
            conn.close()

    text = "\n".join(out)
    Path(args.out).write_text(text, encoding="utf-8")
    print(f"报告已写入 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())