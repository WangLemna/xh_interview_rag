"""FinRAG 环境自检：确认依赖、配置、数据库与模型就绪。

用法：
    python scripts/check_env.py
    python scripts/check_env.py --online   # 额外验证 DeepSeek 连通性

退出码：0 全部通过（允许存在 WARN）；1 存在 FAIL 项。
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import platform
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

LOG_FILENAME = "check_env.log"

#: 配置不可用时的兜底日志目录 —— 自检的本职就是查配置，它坏了这里也得能落盘。
FALLBACK_LOG_DIR = PROJECT_ROOT / "logs"


def log_dir() -> Path:
    """报告落盘目录：读 ``paths.log_dir``，配置加载失败则退回 :data:`FALLBACK_LOG_DIR`。"""
    try:
        from finrag.config import get_config

        cfg = get_config()
        return cfg.paths.resolve(cfg.paths.log_dir)
    except Exception:  # noqa: BLE001
        return FALLBACK_LOG_DIR


def log_file() -> Path:
    return log_dir() / LOG_FILENAME


# 必需依赖：缺失直接 FAIL
REQUIRED_PACKAGES: dict[str, str] = {
    "pydantic": "配置模型",
    "pydantic_settings": "环境变量加载",
    "yaml": "yaml 配置文件解析（包名 PyYAML）",
    "fastapi": "API 服务",
    "uvicorn": "ASGI 服务器",
    "fitz": "PDF 解析（包名 PyMuPDF）",
    "psycopg": "PostgreSQL 驱动",
    "pgvector": "pgvector 类型适配",
    "openai": "LLM 客户端（DeepSeek 走 OpenAI 协议）",
    "httpx": "HTTP 客户端",
    "numpy": "数值计算",
}

# 重依赖：缺失会 WARN，只影响向量化环节
HEAVY_PACKAGES: dict[str, str] = {
    "FlagEmbedding": "BGE-M3 官方推理库",
    "torch": "深度学习后端",
    "transformers": "模型加载",
    "sentencepiece": "BGE-M3 分词器",
    "tqdm": "进度条",
}

_HAS_HF_HUB = importlib.util.find_spec("huggingface_hub") is not None


@dataclass
class CheckResult:
    name: str
    status: str  # OK / WARN / FAIL / SKIP
    detail: str = ""
    hint: str = ""

    @property
    def symbol(self) -> str:
        return {"OK": "[ OK ]", "WARN": "[WARN]", "FAIL": "[FAIL]", "SKIP": "[SKIP]"}[
            self.status
        ]


class Reporter:
    """同时输出到控制台和日志文件。"""

    def __init__(self) -> None:
        self._buf: list[str] = []

    def emit(self, line: str = "") -> None:
        self._buf.append(line)
        try:
            print(line)
        except UnicodeEncodeError:
            print(line.encode("utf-8", "replace").decode("utf-8", "replace"))

    def flush_log(self) -> None:
        try:
            d = log_dir()
            d.mkdir(parents=True, exist_ok=True)
            (d / LOG_FILENAME).write_text("\n".join(self._buf) + "\n", encoding="utf-8")
        except OSError:
            pass


def check_python() -> list[CheckResult]:
    major, minor = sys.version_info[:2]
    ok = (major, minor) >= (3, 10)
    results = [
        CheckResult(
            "Python 版本",
            "OK" if ok else "FAIL",
            f"{platform.python_version()} @ {sys.executable}",
            "" if ok else "需要 3.10+，本项目在 3.12 上开发",
        )
    ]
    if "Windows" in platform.platform():
        results.append(
            CheckResult("运行平台", "OK", f"{platform.system()} {platform.release()}")
        )
    return results


def _probe(module: str) -> tuple[bool, str]:
    spec = importlib.util.find_spec(module)
    if spec is None:
        return False, ""
    try:
        mod = importlib.import_module(module)
        version = getattr(mod, "__version__", "")
        return True, str(version)
    except Exception as exc:  # 已安装但 import 失败
        return False, f"import 失败: {type(exc).__name__}: {exc}"


def check_packages() -> list[CheckResult]:
    results: list[CheckResult] = []

    missing_req = [name for name in REQUIRED_PACKAGES if importlib.util.find_spec(name) is None]
    if missing_req:
        results.append(
            CheckResult(
                "必需依赖",
                "FAIL",
                f"缺失 {len(missing_req)} 个: {', '.join(missing_req)}",
                "装依赖：README「安装」一节，注意本机需加 --user 与清华镜像",
            )
        )
    else:
        results.append(CheckResult("必需依赖", "OK", f"{len(REQUIRED_PACKAGES)} 个全部就位"))

    missing_heavy = [name for name in HEAVY_PACKAGES if importlib.util.find_spec(name) is None]
    if missing_heavy:
        results.append(
            CheckResult(
                "向量化依赖",
                "WARN",
                f"缺失: {', '.join(missing_heavy)}",
                "不装无法做 embedding，检索链路跑不通",
            )
        )
    else:
        ok, version = _probe("torch")
        detail = f"FlagEmbedding + torch {version}" if version else "全部就位"
        results.append(CheckResult("向量化依赖", "OK", detail))

    return results


def check_env_file() -> list[CheckResult]:
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        return [
            CheckResult(
                ".env 配置文件",
                "FAIL",
                "不存在",
                f"复制 {PROJECT_ROOT / '.env.example'} 为 .env"
                f"（Key 仍要另设环境变量，模板里有说明）",
            )
        ]

    text = env_path.read_text(encoding="utf-8", errors="ignore")
    results = [CheckResult(".env 配置文件", "OK", str(env_path))]

    # 通过配置系统读 Key（Key 在环境变量，不在 .env 文本里），避免 grep 文件误报 FAIL。
    from finrag.config import get_config

    try:
        api_key = get_config().llm.api_key
    except Exception as exc:  # noqa: BLE001
        api_key = None
        results.append(
            CheckResult("LLM Key", "FAIL", f"配置加载失败：{type(exc).__name__}")
        )
    if api_key is None:
        results.append(
            CheckResult(
                "LLM Key",
                "FAIL",
                "未读到（环境变量 DEEPSEEKFLASH_KEY 或 FINRAG_LLM__API_KEY）",
                "设置其一；细路见 scripts/check_llm_key.py",
            )
        )
    else:
        v = api_key.get_secret_value()
        if not v.startswith("sk-") or "xxxx" in v:
            results.append(
                CheckResult("LLM Key", "FAIL", "值不像是真实 Key", f"{v[:7]}...")
            )
        else:
            results.append(CheckResult("LLM Key", "OK", f"{v[:7]}...（{len(v)} 位）"))

    # 模型名写错只会拿到语义不明的 400/404、不体现在本地报错里，故单独核。
    try:
        model = get_config().llm.model
        results.append(
            CheckResult(
                "LLM 模型名",
                "OK" if model == "deepseek-flash" else "WARN",
                model,
                None if model == "deepseek-flash"
                else "现役的 API 名是 deepseek-flash",
            )
        )
    except Exception:  # noqa: BLE001
        pass

    if "FINRAG_PG_DSN" not in text:
        results.append(CheckResult("FINRAG_PG_DSN", "WARN", "未配置，将使用默认 DSN"))
    else:
        results.append(CheckResult("FINRAG_PG_DSN", "OK", "已配置"))

    results.extend(_check_config_overridable())
    return results


def _check_config_overridable() -> list[CheckResult]:
    """行为探测 env 能否覆盖 yaml——配置系统曾静默失效过，只看文件不可靠。"""
    probe_field, probe_value = "table_units", False
    try:
        from finrag.config import load_config

        baseline = getattr(load_config().chunk, probe_field)
    except Exception as exc:
        return [CheckResult("配置可被 env 覆盖", "FAIL", f"配置加载失败: {exc}")]

    env_key = f"FINRAG_CHUNK__{probe_field.upper()}"
    old = os.environ.get(env_key)
    # 挑一个与 yaml 相反的值：yaml 里是 true，故用 false；反之亦然。
    os.environ[env_key] = "false" if baseline is True else "true"
    try:
        after = getattr(load_config().chunk, probe_field)
    except Exception as exc:
        return [CheckResult("配置可被 env 覆盖", "FAIL", f"覆盖探测失败: {exc}")]
    finally:
        if old is None:
            os.environ.pop(env_key, None)
        else:
            os.environ[env_key] = old

    if after != baseline:
        return [CheckResult(
            "配置可被 env 覆盖", "OK",
            f"{env_key} 生效（消融开关可用）",
        )]
    return [CheckResult(
        "配置可被 env 覆盖", "FAIL",
        f"{env_key} 被忽略 —— env 覆盖链路已失效",
        "消融实验会量成同一形态。查 config.py 的 settings_customise_sources",
    )]


def _load_dsn() -> str:
    try:
        from finrag.config import get_config

        return get_config().pg_dsn.get_secret_value()
    except Exception:
        return "postgresql://postgres:postgres@localhost:5432/finrag"


def _connect(dsn: str):
    """统一的连接入口，显式指定 client_encoding=UTF8（content 为中文文本，避免乱码）。"""
    import psycopg

    return psycopg.connect(dsn, connect_timeout=5, client_encoding="utf8")


def _dsn_endpoint(dsn: str) -> tuple[str, int]:
    """从 DSN 解析 host/port，供 TCP 可达性探测使用。"""
    from psycopg.conninfo import conninfo_to_dict

    try:
        info = conninfo_to_dict(dsn)
    except Exception:
        return "localhost", 5432
    return info.get("host") or "localhost", int(info.get("port") or 5432)


def _tcp_reachable(host: str, port: int, timeout: float = 3.0) -> bool:
    import socket

    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _pg_failure_result(dsn: str, exc: Exception) -> CheckResult:
    """把连接失败翻译成可读诊断：按端口是否可达分层定位，不依赖服务端措辞。"""
    host, port = _dsn_endpoint(dsn)
    safe_dsn = dsn.split("@")[-1] if "@" in dsn else dsn

    if not _tcp_reachable(host, port):
        return CheckResult(
            "PostgreSQL 连接",
            "FAIL",
            f"{safe_dsn} -> {host}:{port} 无法建立 TCP 连接 ({type(exc).__name__})",
            "PostgreSQL 服务未启动，或 DSN 中的主机/端口不对",
        )

    return CheckResult(
        "PostgreSQL 连接",
        "FAIL",
        f"{safe_dsn} -> 端口可达，但连接被拒 ({type(exc).__name__})",
        "服务在运行。请核对 DSN 里的用户名/密码，以及库 finrag 是否已创建；"
        "可用 psql -U postgres -l 手工确认",
    )


def check_postgres() -> list[CheckResult]:
    if importlib.util.find_spec("psycopg") is None:
        return [CheckResult("PostgreSQL 连接", "SKIP", "psycopg 未安装")]

    import psycopg

    dsn = _load_dsn()
    safe_dsn = dsn.split("@")[-1] if "@" in dsn else dsn

    try:
        with _connect(dsn) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT version()")
                version = cur.fetchone()[0]
                cur.execute("SHOW server_version")
                server_version = cur.fetchone()[0]
                cur.execute("SELECT current_database(), current_user")
                db_name, db_user = cur.fetchone()
    except Exception as exc:
        return [_pg_failure_result(dsn, exc)]

    return [
        CheckResult("PostgreSQL 连接", "OK", f"{safe_dsn} (PG {server_version})"),
        CheckResult("当前库/用户", "OK", f"{db_name} / {db_user}"),
        CheckResult("服务端版本", "OK", version.split(",")[0]),
    ]


def check_pgvector() -> list[CheckResult]:
    if importlib.util.find_spec("psycopg") is None:
        return [CheckResult("pgvector 扩展", "SKIP", "psycopg 未安装")]

    import psycopg

    dsn = _load_dsn()
    try:
        with _connect(dsn) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT extversion FROM pg_extension WHERE extname = 'vector'"
                )
                row = cur.fetchone()
    except Exception as exc:
        return [
            CheckResult(
                "pgvector 扩展",
                "FAIL",
                f"查询失败（{type(exc).__name__}）",
                "先解决上面的 PostgreSQL 连接问题",
            )
        ]

    if row is None:
        return [
            CheckResult(
                "pgvector 扩展",
                "FAIL",
                "扩展未创建",
                "在 finrag 库执行 CREATE EXTENSION vector; "
                "若报找不到 vector.control，说明扩展文件尚未安装到 PG 目录",
            )
        ]

    results = [CheckResult("pgvector 扩展", "OK", f"已启用 v{row[0]}")]

    # 探测 sparsevec 是否可用（混合检索的稀疏路依赖它）
    import psycopg

    try:
        with _connect(dsn) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT '{1:0.5,3:0.2}/5'::sparsevec")
                cur.fetchone()
        results.append(
            CheckResult("sparsevec 类型", "OK", "可用，稀疏检索路就绪")
        )
    except Exception as exc:
        results.append(
            CheckResult(
                "sparsevec 类型",
                "WARN",
                f"不可用: {type(exc).__name__}",
                "需要 pgvector >= 0.7.0，低于此版本无法存 BGE-M3 的稀疏向量",
            )
        )

    return results


def _find_weight_file(path: Path) -> Path | None:
    """定位权重文件：优先顶层常见命名，最后才递归兜底。"""
    for name in ("pytorch_model.bin", "model.safetensors"):
        f = path / name
        if f.exists():
            return f
    return next(
        (
            f
            for f in path.rglob("*")
            if f.is_file() and f.suffix in (".bin", ".safetensors", ".onnx")
        ),
        None,
    )


#: (展示名, 原始配置读取器, 解析后路径读取器, 用途, 额外必需文件)。位置**不在这里写死**
#: —— 一律走配置的 ``model_path``（即 ``config.resolve_model_path``），与运行时同源。
MODEL_SPECS: list[
    tuple[str, Callable[[Any], str], Callable[[Any], str], str, tuple[str, ...]]
] = [
    # sparse_linear.pt 是稀疏检索路的命门：缺失则混合检索静默降为稠密一路，必须拦住。
    ("BGE-M3 编码器", lambda c: c.embed.dense_model, lambda c: c.embed.model_path,
     "dense 1024 维 + sparse 词权重", ("sparse_linear.pt",)),
    ("bge-reranker-v2-m3", lambda c: c.rerank.model, lambda c: c.rerank.model_path,
     "Cross-Encoder 精排", ()),
]


def check_local_models() -> list[CheckResult]:
    """检查检索链路依赖的两个权重：位置**按运行时口径解析**（``model_path``，即
    ``resolve_model_path``），命中不到再查 HF 缓存。

    此前这里写死 ``PROJECT_ROOT/"models"/<名字>`` 并假设 HF 缓存目录叫
    ``models--BAAI--<名字>``，于是把权重配到项目外时会出现"服务跑得起来、自检却说
    未找到"的分裂。自检的口径必须与运行时同源，否则它比不检查更误导。
    """
    hf_hub = Path.home() / ".cache" / "huggingface" / "hub"
    results: list[CheckResult] = []

    try:
        from finrag.config import get_config

        cfg = get_config()
    except Exception as exc:  # noqa: BLE001
        return [CheckResult("本地模型", "SKIP", f"配置加载失败: {exc}")]

    for label, raw_getter, path_getter, role, extra_required in MODEL_SPECS:
        raw = raw_getter(cfg)
        resolved = Path(path_getter(cfg))
        # HF 缓存的目录名规则：BAAI/bge-m3 -> models--BAAI--bge-m3
        cached = hf_hub / ("models--" + raw.replace("/", "--"))

        if resolved.is_dir():
            # 落在项目外 = 权重配的是项目外的绝对路径。报出来 —— "从哪加载"不该靠猜
            src = "本地" if resolved.is_relative_to(PROJECT_ROOT) else "外部"
            path = resolved
        elif cached.is_dir():
            src, path = "HF 缓存", cached
        else:
            hint = (
                f"配置里写的是绝对路径 {raw!r}，但该目录不存在 —— "
                "绝对路径**不会**回落到联网下载，加载时会直接报找不到"
                if Path(raw).is_absolute()
                else f"配置里是 {raw!r}：项目内没有这个目录，于是它会被**原样当 HF repo id** "
                     "交给加载器（所以误写 models/bge-m3 会报 repo 找不到）。"
                     "权重放项目外请直接写绝对路径；要走 HF 下载就写 BAAI/bge-m3 这类"
                     "合法 id（国内建议先设 HF_ENDPOINT=https://hf-mirror.com）"
            )
            results.append(CheckResult(label, "WARN", f"未找到（用途：{role}）", hint))
            continue

        weight = _find_weight_file(path)
        missing = [name for name in extra_required if not (path / name).exists()]

        if weight is not None and not missing:
            size_mb = weight.stat().st_size / 1024**2
            results.append(
                CheckResult(label, "OK", f"{src} · {weight.name} ({size_mb:.0f} MB)")
            )
            continue

        problems = ([] if weight else ["无权重文件"]) + [f"缺 {name}" for name in missing]
        results.append(
            CheckResult(
                label,
                "FAIL" if src in ("本地", "外部") else "WARN",
                f"{src} · {', '.join(problems)}",
                "本地权重不完整，需重新拷贝或下载"
                if src in ("本地", "外部")
                else "缓存不完整会被联网重下；改用本地目录可避免",
            )
        )

    return results


def check_deepseek() -> list[CheckResult]:
    if importlib.util.find_spec("httpx") is None:
        return [CheckResult("DeepSeek API", "SKIP", "httpx 未安装")]

    try:
        from finrag.config import get_config

        cfg = get_config()
    except Exception as exc:
        return [CheckResult("DeepSeek API", "SKIP", f"配置加载失败: {exc}")]

    if cfg.llm.api_key is None:
        return [CheckResult("DeepSeek API", "FAIL", "未配置 API Key")]

    import httpx

    url = cfg.llm.base_url.rstrip("/") + "/models"
    headers = {"Authorization": f"Bearer {cfg.llm.api_key.get_secret_value()}"}
    try:
        resp = httpx.get(url, headers=headers, timeout=15)
    except Exception as exc:
        return [
            CheckResult(
                "DeepSeek API", "FAIL", f"请求异常: {type(exc).__name__}: {exc}"
            )
        ]

    if resp.status_code == 200:
        try:
            models = [m["id"] for m in resp.json().get("data", [])]
        except Exception:
            models = []
        return [CheckResult("DeepSeek API", "OK", f"可用模型: {', '.join(models) or '未知'}")]

    return [
        CheckResult(
            "DeepSeek API",
            "FAIL",
            f"HTTP {resp.status_code}: {resp.text[:160]}",
            "检查 Key 是否有效、账户是否有余额",
        )
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="FinRAG 环境自检")
    parser.add_argument(
        "--online", action="store_true", help="额外验证 DeepSeek API 连通性（会发一次网络请求）"
    )
    args = parser.parse_args()

    reporter = Reporter()
    reporter.emit("=" * 64)
    reporter.emit("FinRAG 环境自检")
    reporter.emit(f"项目根目录: {PROJECT_ROOT}")
    reporter.emit("=" * 64)

    sections: list[tuple[str, list[CheckResult]]] = [
        ("运行时", check_python()),
        ("依赖包", check_packages()),
        ("配置", check_env_file()),
        ("数据库", check_postgres() + check_pgvector()),
        ("模型", check_local_models()),
    ]
    if args.online:
        sections.append(("外部服务", check_deepseek()))

    all_results: list[CheckResult] = []
    for title, results in sections:
        reporter.emit("")
        reporter.emit(f"--- {title} ---")
        for r in results:
            all_results.append(r)
            line = f"{r.symbol} {r.name}: {r.detail}"
            reporter.emit(line)
            if r.hint and r.status in ("FAIL", "WARN"):
                reporter.emit(f"       -> {r.hint}")

    counts = {s: sum(1 for r in all_results if r.status == s) for s in ("OK", "WARN", "FAIL", "SKIP")}
    reporter.emit("")
    reporter.emit("=" * 64)
    reporter.emit(
        f"结果: OK {counts['OK']} | WARN {counts['WARN']} | "
        f"FAIL {counts['FAIL']} | SKIP {counts['SKIP']}"
    )

    if counts["FAIL"]:
        reporter.emit("存在阻塞项，请按上面的 -> 提示逐条解决。")
        exit_code = 1
    elif counts["WARN"]:
        reporter.emit("无阻塞项。WARN 项在进入对应环节前处理即可。")
        exit_code = 0
    else:
        reporter.emit("环境完全就绪。")
        exit_code = 0
    reporter.emit("=" * 64)
    reporter.emit(f"日志已写入: {log_file()}")

    reporter.flush_log()
    return exit_code


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass
    raise SystemExit(main())
