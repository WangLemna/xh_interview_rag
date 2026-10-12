"""把公司表 CSV 装载进 PG：``companies`` / ``company_aliases``（幂等，全量替换）。

数据来源链：``build_companies.py``（合并清单短名 + 年报封面权威全称 + 人工定稿别名 + 拼音）
→ 本脚本 → PG。装载后 ``store/resolve.py`` 即以公司表为**唯一裁决来源**建别名表。

**全量替换而非增量**：CSV 是从库 + 清单**重新生成**的完整快照，增量同步只会留下
"上一版有、这一版删了"的残影。全量替换放进**一个事务**，失败即回滚，不会留下半装载状态。

⚠ 本脚本**会清空并重写**这两张表（不影响 ``documents`` / ``chunks``）。

用法::

    C:\\Python312\\python.exe scripts\\load_companies.py
    C:\\Python312\\python.exe scripts\\load_companies.py --csv data/companies.csv

退出码：0 已装载；1 CSV 不可读 / 连不上库 / 装载失败（已回滚）。
"""

from __future__ import annotations

import argparse
import csv
import sys
import traceback
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from finrag.config import get_config  # noqa: E402
from finrag.console import use_utf8_stdout  # noqa: E402

DEFAULT_CSV = PROJECT_ROOT / "data" / "companies.csv"

_REQUIRED = ("code", "简称", "全称", "别名")


def _read_csv(path: Path) -> list[dict[str, str]]:
    """读公司表 CSV（UTF-8-sig）；缺列即报错（fail-closed），不静默跳过。"""
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        head = [h.strip() for h in (reader.fieldnames or [])]
        missing = [c for c in _REQUIRED if c not in head]
        if missing:
            raise ValueError(f"公司表缺列：{missing}（表头实为 {head}）")
        rows = []
        for row in reader:
            code = (row.get("code") or "").strip()
            if not code:
                continue
            rows.append({k: (row.get(k) or "").strip() for k in head})
        if not rows:
            raise ValueError(f"公司表为空：{path}")
        return rows


def _load(conn: Any, rows: list[dict[str, str]]) -> dict[str, int]:
    """一个事务里全量替换两张表；返回读数。"""
    n_alias = 0
    with conn.transaction():
        with conn.cursor() as cur:
            # 顺序：有 FK，先删子表再删父表
            cur.execute("DELETE FROM company_aliases")
            cur.execute("DELETE FROM companies")

            cur.executemany(
                "INSERT INTO companies"
                " (code, short_name, full_name, pinyin, pinyin_initials, n_docs)"
                " VALUES (%s, %s, %s, %s, %s, %s)",
                [
                    (
                        r["code"],
                        r["简称"],
                        r["全称"] or None,
                        r.get("拼音") or None,
                        r.get("拼音首字母") or None,
                        int(r["n_docs"]) if str(r.get("n_docs") or "0").isdigit() else 0,
                    )
                    for r in rows
                ],
            )

            pairs = [
                (r["code"], a)
                for r in rows
                for a in dict.fromkeys(
                    [x.strip() for x in (r["别名"] or "").split("|") if x.strip()]
                )
            ]
            # PK(code, alias) 去重后再插，避免同一 code 内重复别名触发冲突
            cur.executemany(
                "INSERT INTO company_aliases (code, alias) VALUES (%s, %s) ON CONFLICT DO NOTHING",
                pairs,
            )
            n_alias = len(pairs)

            cur.execute("SELECT count(*) FROM companies")
            n_comp = int(cur.fetchone()[0])
            cur.execute("SELECT count(*) FROM company_aliases")
            n_alias_db = int(cur.fetchone()[0])
            cur.execute("SELECT count(*) FROM companies WHERE full_name IS NULL OR full_name = ''")
            n_no_full = int(cur.fetchone()[0])
    return {
        "companies": n_comp,
        "aliases": n_alias_db,
        "aliases_try": n_alias,
        "no_full_name": n_no_full,
    }


def main() -> int:
    use_utf8_stdout()
    cfg = get_config()

    ap = argparse.ArgumentParser(description="装载公司表到 PG（全量替换；幂等）")
    ap.add_argument("--csv", default=str(DEFAULT_CSV), help="公司表 CSV（build_companies.py 的产物）")
    args = ap.parse_args()
    csv_path = Path(args.csv)

    if not csv_path.is_file():
        print(f"公司表不存在：{csv_path}（先跑 scripts/build_companies.py）")
        return 1

    try:
        rows = _read_csv(csv_path)
    except Exception:  # noqa: BLE001
        print("CSV 不可读（未写库）：")
        print(traceback.format_exc())
        return 1

    try:
        from finrag.store.writer import connect

        with connect(cfg) as conn:
            stat = _load(conn, rows)
    except Exception:  # noqa: BLE001
        print("装载失败（事务已回滚，两张表保持原状）：")
        print(traceback.format_exc())
        return 1

    summary = [
        f"公司表已装载 ← {csv_path}",
        f"  companies={stat['companies']}  company_aliases={stat['aliases']}"
        f"（CSV 侧 {stat['aliases_try']} 条，PK 去重后落库）",
        f"  缺法定全称：{stat['no_full_name']} 家",
        "  提示：服务侧需 clear_alias_cache() 或重启才会重建别名表缓存",
    ]
    (csv_path.with_suffix(csv_path.suffix + ".load.txt")).write_text(
        "\n".join(summary) + "\n", encoding="utf-8"
    )
    print("\n".join(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())