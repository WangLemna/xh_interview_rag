"""启动 FinRAG 服务（FastAPI + SSE）。

用法：
    python scripts/serve.py
    python scripts/serve.py --port 8080 --no-preload   # 跳过模型预热，启动快、首问慢
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from finrag.config import get_config  # noqa: E402
from finrag.console import use_utf8_stdout  # noqa: E402


def main() -> int:
    use_utf8_stdout()
    ap = argparse.ArgumentParser(description="启动 FinRAG HTTP 服务")
    ap.add_argument("--host", default=None, help="默认取 configs/default.yaml 的 api.host")
    ap.add_argument("--port", type=int, default=None, help="默认取 api.port（8000）")
    ap.add_argument("--no-preload", action="store_true",
                    help="跳过模型预热（启动快，但首个请求要等模型加载）")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    cfg = get_config()
    host = args.host or cfg.api.host
    port = args.port or cfg.api.port

    # 预热开关走环境变量：app 在 server 模块导入时即构造，无法从命令行传参。
    if args.no_preload:
        import os

        os.environ["FINRAG_SKIP_PRELOAD"] = "1"
        print("[serve] 已关闭模型预热（首个请求会触发加载）")

    print(f"[serve] FinRAG → http://{host}:{port}/")
    print(f"[serve] 模型 llm={cfg.llm.model} · rerank={cfg.rerank.enabled} "
          f"· 库 {cfg.pg_dsn.get_secret_value().split('@')[-1]}")

    import uvicorn

    from finrag.api.server import create_app

    uvicorn.run(create_app(cfg), host=host, port=port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
