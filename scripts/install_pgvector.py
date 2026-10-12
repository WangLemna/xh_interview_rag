"""pgvector Windows 部署工具（预编译包路线）。

官方不提供 Windows 二进制，故拉取第三方预编译包；脚本会打印 SHA256 供核对留档。

用法：
    python scripts/install_pgvector.py --pg-root "C:\\Program Files\\PostgreSQL\\17"
    python scripts/install_pgvector.py --pg-root "C:\\Program Files\\PostgreSQL\\17" --apply

``--apply`` 会写入 Program Files 并重启服务，需管理员权限。
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

PACKAGE_URL = (
    "https://github.com/levovs/pgvector_compiled/releases/download/"
    "v0.17.2/pgvector-x86_64-pc-windows-msvc-pg17.zip"
)
PACKAGE_NAME = "pgvector-x86_64-pc-windows-msvc-pg17.zip"
EXPECTED_MIN_BYTES = 50 * 1024  # 实测约 159KB，过小说明下到了错误页

# 压缩包内文件 -> PostgreSQL 安装目录下的相对位置
DISPATCH_RULES: tuple[tuple[str, str], ...] = (
    ("*.dll", "lib"),
    ("*.control", "share/extension"),
    ("*--*.sql", "share/extension"),
    ("*.h", "include/server/extension/vector"),
)


def log(msg: str = "") -> None:
    print(msg, flush=True)


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def download(dest_dir: Path, url: str = PACKAGE_URL) -> Path:
    dest = dest_dir / PACKAGE_NAME
    log(f"下载: {url}")
    try:
        with urllib.request.urlopen(url, timeout=120) as resp, dest.open("wb") as fh:
            shutil.copyfileobj(resp, fh)
    except Exception as exc:
        log(f"下载失败: {type(exc).__name__}: {exc}")
        log("若为网络问题，可换用镜像前缀后重试，例如：")
        log(f"  --url https://ghfast.top/{url}")
        raise SystemExit(2)

    size = dest.stat().st_size
    log(f"完成: {dest} ({size / 1024:.1f} KB)")
    if size < EXPECTED_MIN_BYTES:
        log("文件体积异常偏小，可能被重定向到了 HTML 页面，请核对来源。")
        raise SystemExit(2)
    return dest


def inspect(zip_path: Path) -> list[str]:
    """校验压缩包结构并打印成员清单。"""
    if not zipfile.is_zipfile(zip_path):
        log("不是合法 zip 文件。")
        raise SystemExit(2)

    log("")
    log("SHA256 (请留档): " + sha256_of(zip_path))
    log("")

    with zipfile.ZipFile(zip_path) as zf:
        names = [n for n in zf.namelist() if not n.endswith("/")]
        log(f"压缩包内含 {len(names)} 个文件:")
        for name in names:
            info = zf.getinfo(name)
            log(f"  {info.file_size:>8,} B  {name}")

        dlls = [n for n in names if n.lower().endswith(".dll")]
        if not dlls:
            log("未找到 .dll，压缩包结构不符合预期。")
            raise SystemExit(2)

        log("")
        log("PE 头校验:")
        for name in dlls:
            head = zf.read(name)[:2]
            ok = head == b"MZ"
            log(f"  {name}: {'MZ 合法' if ok else '头部异常 -> ' + repr(head)}")
            if not ok:
                raise SystemExit(2)
    return names


def plan_dispatch(names: list[str], pg_root: Path) -> list[tuple[str, Path]]:
    """把压缩包成员映射为目标路径。"""
    plan: list[tuple[str, Path]] = []
    for name in names:
        base = Path(name).name
        for pattern, rel_dir in DISPATCH_RULES:
            if Path(name).match(pattern) or Path(base).match(pattern.replace("*--*", "*--*")):
                if Path(base).match(pattern):
                    target_dir = pg_root / rel_dir
                    plan.append((name, target_dir / base))
                    break
        else:
            log(f"跳过（无匹配规则）: {name}")
    return plan


def dispatch(plan: list[tuple[str, Path]], zip_path: Path, apply: bool) -> int:
    log("")
    log("部署计划:" if apply else "部署计划（干跑，不写入）:")
    for name, target in plan:
        marker = "存在，将覆盖" if target.exists() else "新增"
        log(f"  {name}  ->  {target}   [{marker}]")

    if not apply:
        log("")
        log("以上为预览。确认无误后加 --apply 实际执行。")
        return 0

    failures = 0
    with zipfile.ZipFile(zip_path) as zf:
        for name, target in plan:
            try:
                target.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(name) as src, target.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
                log(f"  写入成功: {target}")
            except PermissionError:
                log(f"  权限不足: {target}")
                log("    -> 请以管理员身份重新打开终端后再运行")
                failures += 1
            except Exception as exc:
                log(f"  失败 {target}: {type(exc).__name__}: {exc}")
                failures += 1
    return 1 if failures else 0


def restart_service(pg_root: Path, apply: bool) -> None:
    version = pg_root.name
    service = f"postgresql-x64-{version}"
    log("")
    log(f"重启服务: {service}")
    if not apply:
        log(f"  （干跑跳过）实际执行: net stop {service} && net start {service}")
        return
    try:
        r = subprocess.run(
            ["net", "stop", service], capture_output=True, text=True, timeout=60
        )
        log(f"  stop  -> exit={r.returncode} {r.stdout.strip()[:200]}")
        r = subprocess.run(
            ["net", "start", service], capture_output=True, text=True, timeout=60
        )
        log(f"  start -> exit={r.returncode} {r.stdout.strip()[:200]}")
    except Exception as exc:
        log(f"  自动重启失败({type(exc).__name__})，请手动重启：services.msc -> {service}")


def verify(pg_root: Path) -> None:
    log("")
    log("落盘校验:")
    lib = pg_root / "lib" / "vector.dll"
    ext_dir = pg_root / "share" / "extension"
    log(f"  {lib}: {'存在' if lib.exists() else '缺失'}")
    if ext_dir.exists():
        extras = sorted(p.name for p in ext_dir.glob("vector*"))
        log(f"  {ext_dir} 下 vector 相关文件: {extras or '无'}")


def main() -> int:
    parser = argparse.ArgumentParser(description="pgvector Windows 部署（预编译包）")
    parser.add_argument(
        "--pg-root",
        required=True,
        help='PostgreSQL 安装根目录，例如 "C:\\Program Files\\PostgreSQL\\17"',
    )
    parser.add_argument("--apply", action="store_true", help="实际写入并重启服务")
    parser.add_argument("--url", default=PACKAGE_URL, help="覆盖下载地址")
    args = parser.parse_args()

    pg_root = Path(args.pg_root)
    log("=" * 64)
    log("pgvector 部署（预编译包路线）")
    log("=" * 64)

    if not pg_root.exists():
        log(f"PostgreSQL 根目录不存在: {pg_root}")
        log("请先安装 PostgreSQL 17（EDB 官方安装包），再回来跑本脚本。")
        return 2

    log(f"PG 根目录: {pg_root}")
    with tempfile.TemporaryDirectory(prefix="pgvector_") as tmp:
        tmp_dir = Path(tmp)
        zip_path = download(tmp_dir, args.url)
        names = inspect(zip_path)
        plan = plan_dispatch(names, pg_root)
        if not plan:
            log("没有可部署的文件，终止。")
            return 2
        code = dispatch(plan, zip_path, args.apply)
        if code:
            return code
        restart_service(pg_root, args.apply)
        if args.apply:
            verify(pg_root)

    log("")
    if args.apply:
        log("下一步：")
        log('  psql -U postgres -d finrag -c "CREATE EXTENSION vector;"')
        log("  python scripts/check_env.py")
    else:
        log("干跑结束。确认上面的 SHA256 与来源后可加 --apply。")
    return 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass
    raise SystemExit(main())
