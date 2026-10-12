"""解析结果缓存 —— 把"解析一遍"与"验一遍"解耦。

只做记忆化、不做时间策略：缓存能不能用，只看输入（PDF 字节 × 解析层代码 × 解析配置）变没变。
出口 :func:`parse_cached`（单份）、:func:`iter_parsed`（批量、可并行、**逐份产出不一次收齐**）与
:func:`parse_many`（在上面套一层"全收齐"），缓存位置 ``<paths.cache_dir>/parse/``（按 PDF 路径分片）；
调用方须把来源如实写进报告（:data:`SOURCE_*` / :func:`source_tally`）。
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import logging
import os
import pickle
import sys
import time
from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, NamedTuple, Sequence

from finrag.config import AppConfig, get_config
from finrag.parse.guard import DOC_UNKNOWN, PROBE_VERSION, Admission
from finrag.parse.ir import IR_SCHEMA, PARSER_VERSION
from finrag.parse.pdf_parser import ParseResult, parse_pdf

__all__ = [
    "CACHE_SCHEMA",
    "SOURCE_BROKEN",
    "SOURCE_ERROR",
    "SOURCE_HIT",
    "SOURCE_MISS",
    "SOURCE_OFF",
    "ParseOutcome",
    "cache_file",
    "cache_key",
    "default_cache_dir",
    "fatal_result",
    "iter_parsed",
    "parse_cached",
    "parse_many",
    "resolve_jobs",
    "source_label",
    "source_tally",
]

logger = logging.getLogger(__name__)

# 缓存**文件格式**版本。与 IR_SCHEMA / PARSER_VERSION 不是一回事：那两个描述被缓存的
# 数据，本常量描述"装数据的信封"（字典键、pickle 协议）。信封改了而数据没改时只动它。
CACHE_SCHEMA = 1

CACHE_SUBDIR = "parse"
CACHE_SUFFIX = ".pkl"

# 这一份结果是哪来的。写进验收报告，理由见模块说明。
SOURCE_HIT = "hit"        # 读到了缓存
SOURCE_MISS = "miss"      # 缓存里没有，现算
SOURCE_BROKEN = "broken"  # 缓存文件在，但读不出来（损坏 / 格式不符），现算
SOURCE_OFF = "off"        # 调用方要求绕过缓存
SOURCE_ERROR = "error"    # 解析本身失败（坏 PDF / 加密 / 内存不足），不写缓存

_SOURCE_LABELS = {
    SOURCE_HIT: "缓存",
    SOURCE_MISS: "现算",
    SOURCE_BROKEN: "现算(缓存损坏)",
    SOURCE_OFF: "现算(绕过缓存)",
    SOURCE_ERROR: "解析失败",
}

# 自动并行时进程数的上限。不是"越多越快"：单份年报的 PDF 与 IR 都常驻内存，
# 再把核数吃满只会与 PyMuPDF 自身的缓冲抢内存，而收益早已递减。
AUTO_JOBS_CAP = 4


class ParseOutcome(NamedTuple):
    """一次取解析结果的产出：结果本身 + 它的来源。"""

    result: ParseResult
    source: str


# ------------------------------------------------------------------
# 键
# ------------------------------------------------------------------
@lru_cache(maxsize=1)
def _parse_source_fingerprint() -> str:
    """解析层源码指纹，覆盖 ``parse/`` 包全部模块 + 同级 ``config.py``，是缓存键能成立的前提。"""
    h = hashlib.sha256()
    files = sorted(Path(__file__).parent.glob("*.py"))
    files.append(Path(__file__).parents[1] / "config.py")
    for f in files:
        h.update(f.name.encode("utf-8"))
        h.update(f.read_bytes())
    return h.hexdigest()[:32]


@lru_cache(maxsize=1)
def _engine_fingerprint() -> str:
    """底层引擎版本。块切分由 PyMuPDF 决定 —— 升级它可能让同一份 PDF 切出不同的块。"""
    for dist in ("PyMuPDF", "pymupdf", "fitz"):
        try:
            return f"{dist}={importlib.metadata.version(dist)}"
        except Exception:  # noqa: BLE001 —— 版本读不到不该让验收失败
            continue
    return "PyMuPDF=unknown"


def cache_key(pdf_path: str | Path, cfg: AppConfig | None = None) -> str:
    """一份 PDF 的缓存键 = PDF 字节 × 解析层代码 × 解析配置 的哈希。

    PDF 一侧用 ``(绝对路径, 大小, mtime_ns)`` 而非内容哈希；配置一侧序列化整段 ``cfg.parse``。
    """
    cfg = cfg or get_config()
    p = Path(pdf_path)
    st = p.stat()
    h = hashlib.sha256()
    h.update(
        f"cache{CACHE_SCHEMA}|ir{IR_SCHEMA}|parser{PARSER_VERSION}|probe{PROBE_VERSION}"
        # 解释器版本也进键：pickle 的可用性与类的布局都跟着它走。真换了解释器时
        # 该是一次未命中，而不是一次意外。
        f"|py{sys.version_info[:3]}\n".encode("utf-8")
    )
    h.update(f"{_parse_source_fingerprint()}|{_engine_fingerprint()}\n".encode("utf-8"))
    h.update(cfg.parse.model_dump_json().encode("utf-8"))
    h.update(f"\n{p.resolve()}|{st.st_size}|{st.st_mtime_ns}".encode("utf-8"))
    return h.hexdigest()[:16]


def default_cache_dir(cfg: AppConfig | None = None) -> Path:
    """默认缓存根（``<paths.cache_dir>/parse``）。"""
    cfg = cfg or get_config()
    return cfg.paths.resolve(cfg.paths.cache_dir) / CACHE_SUBDIR


def _shard(resolved: str) -> str:
    """缓存分片子目录名（取路径 sha256 前 2 位）。

    按 **PDF 路径**而非缓存键分片：同一份 PDF 在改代码 / 改配置后键会变，但恒落在同一分片，
    :func:`_prune` 才清得掉旧键。分片是为了避免万份语料把文件全平铺在一个目录里。
    """
    return hashlib.sha256(resolved.encode("utf-8")).hexdigest()[:2]


def cache_file(
    pdf_path: str | Path, cfg: AppConfig | None = None, cache_dir: str | Path | None = None
) -> Path:
    """这份 PDF 在当前输入下的缓存文件路径（按 PDF 路径分片存放）。"""
    root = Path(cache_dir) if cache_dir else default_cache_dir(cfg)
    p = Path(pdf_path)
    return root / _shard(str(p.resolve())) / f"{p.stem}.{cache_key(p, cfg)}{CACHE_SUFFIX}"


# ------------------------------------------------------------------
# 读写
# ------------------------------------------------------------------
def _load(path: Path) -> ParseResult | None:
    """读缓存；任何异常都退回 ``None``（= 当作未命中），绝不向上抛。"""
    try:
        with path.open("rb") as fh:
            envelope = pickle.load(fh)
    except FileNotFoundError:
        return None
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "解析缓存不可读，将重新解析：%s（%s: %s）", path.name, type(exc).__name__, exc
        )
        return None

    if not isinstance(envelope, dict) or envelope.get("schema") != CACHE_SCHEMA:
        logger.warning("解析缓存格式不符（期望 schema=%s），将重新解析：%s", CACHE_SCHEMA, path.name)
        return None
    result = envelope.get("result")
    if not isinstance(result, ParseResult):
        logger.warning("解析缓存内容不是 ParseResult，将重新解析：%s", path.name)
        return None
    return result


def _save(path: Path, result: ParseResult, pdf_path: Path, elapsed: float) -> None:
    """写缓存，**先写临时文件再原子替换**。

    信封里额外记下的 ``elapsed_s`` / ``created_at`` 只供排查，**不参与任何判定**。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        with tmp.open("wb") as fh:
            pickle.dump(
                {
                    "schema": CACHE_SCHEMA,
                    "source_pdf": str(pdf_path),
                    "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "elapsed_s": round(elapsed, 3),
                    "result": result,
                },
                fh,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
    _prune(path)


def _prune(path: Path) -> None:
    """清掉同一份 PDF 的旧键文件（改代码 / 换 PDF / 改配置都会让键变）。"""
    # 文件名形如 ``<pdf 名>.<键>.pkl``；带走 ``.`` 前缀匹配，避免把
    # ``000333_2025_年报补充`` 这种以 ``000333_2025_年报`` 开头的另一份 PDF 误删。
    stem = path.name[: -len(CACHE_SUFFIX)]
    if "." not in stem:
        return
    prefix = stem.rsplit(".", 1)[0]
    try:
        siblings = list(path.parent.iterdir())
    except OSError:
        return
    for stale in siblings:
        if stale == path:
            continue
        if stale.name.startswith(f"{prefix}.") and stale.name.endswith(CACHE_SUFFIX):
            try:
                stale.unlink()
            except OSError:
                pass


# ------------------------------------------------------------------
# 出口
# ------------------------------------------------------------------
def fatal_result(pdf: Path, exc: BaseException) -> ParseResult:
    """把"解析在框架层失败"包成一个可入库的 :class:`ParseResult`（``doc`` 为 None、``fatal`` 非空）。

    与"被准入拒绝"不同：后者是判据给出的结论，前者是根本没跑起来（坏 PDF / 加密 / 内存不足）。
    """
    detail = f"{type(exc).__name__}: {exc}"
    return ParseResult(
        path=pdf,
        admission=Admission(False, DOC_UNKNOWN, 0, 0, f"解析失败：{detail}"),
        warnings=[f"解析失败：{detail}"],
        fatal=detail,
    )


def _parse_guarded(pdf: Path, cfg: AppConfig, source_on_ok: str) -> ParseOutcome:
    """跑 ``parse_pdf`` 并兜住任何异常 —— 单份解析失败绝不能中断整批。"""
    try:
        return ParseOutcome(parse_pdf(pdf, cfg), source_on_ok)
    except Exception as exc:  # noqa: BLE001
        logger.warning("解析失败（不中断整批）：%s（%s: %s）", pdf.name, type(exc).__name__, exc)
        return ParseOutcome(fatal_result(pdf, exc), SOURCE_ERROR)


def parse_cached(
    pdf_path: str | Path,
    cfg: AppConfig | None = None,
    *,
    cache_dir: str | Path | None = None,
    use_cache: bool = True,
) -> ParseOutcome:
    """``parse_pdf`` 的记忆化版本，返回 ``(结果, 来源)``；``use_cache=False`` 时**不读也不写**。

    解析失败不抛异常：返回带 ``fatal`` 的结果、来源标 ``error``、**不写缓存**（免得把一次失败固化）。
    """
    cfg = cfg or get_config()
    pdf = Path(pdf_path)
    if not use_cache:
        return _parse_guarded(pdf, cfg, SOURCE_OFF)

    path = cache_file(pdf, cfg, cache_dir)
    cached = _load(path)
    if cached is not None:
        return ParseOutcome(cached, SOURCE_HIT)

    existed = path.exists()  # 文件在却读不出来 ⇒ 来源是"损坏"而不是"从没缓存过"
    t0 = time.perf_counter()
    try:
        result = parse_pdf(pdf, cfg)
    except Exception as exc:  # noqa: BLE001
        logger.warning("解析失败（不中断整批）：%s（%s: %s）", pdf.name, type(exc).__name__, exc)
        return ParseOutcome(fatal_result(pdf, exc), SOURCE_ERROR)
    elapsed = time.perf_counter() - t0
    try:
        _save(path, result, pdf, elapsed)
    except Exception as exc:  # noqa: BLE001
        logger.warning("解析缓存写入失败（不影响本次结果）：%s（%s）", path.name, exc)
    return ParseOutcome(result, SOURCE_BROKEN if existed else SOURCE_MISS)


def _worker(task: tuple[str, str, bool]) -> ParseOutcome:
    """进程池 worker，**必须是模块级函数**（Windows spawn 下按名字解析）。

    入参刻意压成三个基本类型、不传 ``AppConfig``。
    """
    pdf, cache_dir, use_cache = task
    return parse_cached(pdf, None, cache_dir=cache_dir or None, use_cache=use_cache)


def resolve_jobs(requested: int, n_files: int) -> int:
    """定并行度。``requested <= 0`` 表示自动。

    自动值取 ``min(AUTO_JOBS_CAP, 核数 - 1)`` 且不超过文件数（留一颗核给主进程）。
    """
    if n_files <= 0:
        return 1
    if requested > 0:
        return min(requested, n_files)
    return max(1, min(AUTO_JOBS_CAP, (os.cpu_count() or 1) - 1, n_files))


def _result_of(fut: Any, path: Path) -> ParseOutcome:
    """取一个在飞 future 的结果；失败（含 worker 进程内异常）也收成 ``error`` 来源，不抛。"""
    try:
        return fut.result()
    except Exception as exc:  # noqa: BLE001
        logger.warning("解析任务失败（不中断整批）：%s（%s: %s）", path.name, type(exc).__name__, exc)
        return ParseOutcome(fatal_result(path, exc), SOURCE_ERROR)


def iter_parsed(
    pdfs: Sequence[str | Path],
    cfg: AppConfig | None = None,
    *,
    cache_dir: str | Path | None = None,
    use_cache: bool = True,
    jobs: int = 1,
    window: int = 0,
) -> Iterator[tuple[Path, ParseOutcome]]:
    """逐份产出 ``(path, ParseOutcome)``，**顺序与入参一致**；``jobs > 1`` 时走进程池。

    与 :func:`parse_many` 的唯一差别是**不一次性收齐**：并行时用**有界在飞窗口**（``window``，
    默认取 ``jobs``），主进程最多同时持有这么多份结果 —— 既压住峰值内存，也让"解析（CPU 进程池）"
    与"编码（主进程 GPU）"能重叠。调用方边收边处理即可。
    """
    items = [Path(p) for p in pdfs]
    cfg = cfg or get_config()
    if not items:
        return

    if jobs > 1 and len(items) > 1:
        # worker 不接 cfg，只认进程内的单例；这里拦一次"传了自定义配置还要求并行"，
        # 否则两边会各自算出不同的缓存键，命中率无声归零。
        if cfg.parse.model_dump_json() != get_config().parse.model_dump_json():
            raise ValueError(
                "jobs > 1 时不能传自定义解析配置：worker 只认进程内的 get_config()。"
                "请改为串行（jobs=1），或在环境变量 / configs/default.yaml 里统一配置。"
            )
        limit = max(1, min(window if window > 0 else jobs, len(items)))
        pending = iter(
            [(str(p), str(cache_dir) if cache_dir else "", use_cache) for p in items]
        )
        with ProcessPoolExecutor(max_workers=min(jobs, len(items))) as pool:
            inflight: deque[tuple[Path, Any]] = deque()
            while len(inflight) < limit:
                task = next(pending, None)
                if task is None:
                    break
                inflight.append((Path(task[0]), pool.submit(_worker, task)))
            while inflight:
                path, fut = inflight.popleft()
                yield path, _result_of(fut, path)
                task = next(pending, None)
                if task is not None:
                    inflight.append((Path(task[0]), pool.submit(_worker, task)))
        return

    for path in items:
        yield path, parse_cached(path, cfg, cache_dir=cache_dir, use_cache=use_cache)


def parse_many(
    pdfs: Sequence[str | Path],
    cfg: AppConfig | None = None,
    *,
    cache_dir: str | Path | None = None,
    use_cache: bool = True,
    jobs: int = 1,
    progress: Callable[[int, int, Path, ParseOutcome], None] | None = None,
) -> list[ParseOutcome]:
    """``iter_parsed`` 的"全收齐"版本，**返回顺序与入参一致**；``progress`` 按入参顺序回调。

    只在确实需要一次拿到全部结果时用（如 ingest 的 ``--dry-run`` 报告）；主入库路径请用
    :func:`iter_parsed` 边收边写，避免万份语料把全部文档压进内存。
    """
    items = [Path(p) for p in pdfs]
    out: list[ParseOutcome] = []
    for i, (path, res) in enumerate(
        iter_parsed(items, cfg, cache_dir=cache_dir, use_cache=use_cache, jobs=jobs), 1
    ):
        out.append(res)
        if progress is not None:
            progress(i, len(items), path, res)
    return out


# ------------------------------------------------------------------
# 报告用的一层措辞
# ------------------------------------------------------------------
def source_label(source: str) -> str:
    """来源的展示写法。未知取值原样返回 —— 宁可印出一个没见过的字符串，
    也不要悄悄统一成某个"看起来正常"的标签。"""
    return _SOURCE_LABELS.get(source, source)


def source_tally(outcomes: Iterable[ParseOutcome]) -> str:
    """来源统计的一行文字。

    **措辞只定义这一处**，两个验收脚本共用。
    """
    n = Counter(o.source for o in outcomes)
    parts = [
        f"{_SOURCE_LABELS[k]} {n[k]}"
        for k in (SOURCE_HIT, SOURCE_MISS, SOURCE_BROKEN, SOURCE_OFF, SOURCE_ERROR)
        if n[k]
    ]
    return " ｜ ".join(parts) if parts else "（无）"
