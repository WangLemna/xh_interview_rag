"""HTTP 层（FastAPI + SSE）。

本包不在 ``__init__`` 里 import ``server``：``server.py`` 末尾的 ``app = create_app()``
会构造配置树，做成 import 副作用会让只用 ``service`` 的脚本也顺带建一遍 app。
"""
