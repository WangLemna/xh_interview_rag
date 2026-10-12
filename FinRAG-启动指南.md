# FinRAG 启动指南

> 前置 → 建库 → 灌数据 → 自检 → 启动 → 就绪判据 → 接口 → 停服务 → 故障速查 → 配置优先级。

---

## 0. 本机约定（只本机适用）

| 项 | 值 |
|---|---|
| Python | `C:\Python312\python.exe`（下文一律简写为 `python`） |
| 项目根 | `D:\AgentGit`（下文相对路径均以此为基准） |
| 权重 / 语料 | 不在仓库内，由 `.env` 指向项目外（见 §1） |

---

## 1. 前置条件

| 项 | 要求 | 怎么验 |
| --- | --- | --- |
| Python | 3.12 | `python -V` |
| PostgreSQL | 17 | `Get-Service postgresql-x64-17` |
| pgvector | 0.8.0（含 `sparsevec`）；Windows 装法见 §1.3 | `python scripts\check_env.py` |
| 依赖包 | `requirements.txt` 全装 | 同上 |
| 数据库 | 库 `finrag`，DSN 写在 `.env` 的 `FINRAG_PG_DSN` | 同上 |
| 模型权重 | `bge-m3`（\~2.1GB）、`bge-reranker-v2-m3`（\~2.1GB） | 同上 |
| 计算设备 | `cuda`；无 GPU 改 `.env` 的 `FINRAG_EMBED__DEVICE=cpu` | 同上 |
| `.env` | 由 `.env.example` 复制，至少填 `FINRAG_PG_DSN` | - |
| LLM Key | 系统环境变量 `FINRAG_LLM__API_KEY`（user） | - |
| 语料 | PDF命名 `<代码>_<简称>_<年份>年年度报告.pdf` | - |
| 清单 | `make_manifest.py` 生成 `corpus_manifest.csv`，**语料变更后必重跑** | - |

### 1.1**装依赖**

```powershell
python -m pip install --user -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
```

### 1.2**权重/语料**：改 `.env`

```ini
FINRAG_EMBED__DENSE_MODEL=D:/Models/bge-m3
FINRAG_RERANK__MODEL=D:/Models/bge-reranker-v2-m3
FINRAG_PATHS__RAW_DIR=D:/corpus/pdf
FINRAG_PATHS__MANIFEST=D:/corpus/corpus_manifest.csv
```

### 1.3**装 pgvector**（Windows）

官方**不提供 Windows 预编译二进制**

```powershell
# 1) 试跑：只下载 + 校验，不改 PG 目录（会打印 SHA256 供核对）
python scripts\install_pgvector.py --pg-root "C:\Program Files\PostgreSQL\17"

# 2) 核对无误后写入并重启服务（需管理员权限）
python scripts\install_pgvector.py --pg-root "C:\Program Files\PostgreSQL\17" --apply
```

- `--pg-root` 为 PostgreSQL 安装根目录（版本号按实际改）。
- 装完 `python scripts\check_env.py` 应显示 `pgvector 扩展: 已启用 v0.8.0` 与 `sparsevec 类型: 可用`。

---

## 2. 建库（只做一次）

```powershell
createdb -U postgres finrag
psql -U postgres -d finrag -f src\finrag\store\schema.sql
```

---

## 3. 灌数据

### 3.1 语料 → 库

```powershell
python scripts\make_manifest.py                          # PDF 目录 → corpus_manifest.csv
python scripts\ingest.py                                 # 增量入库（指纹一致则跳过）
python scripts\ingest.py --force                         # 全部重写（换 embed 模型后用）
python scripts\ingest.py --dry-run                       # 只解析 + 分块，不连库、不编码
psql -U postgres -d finrag -f scripts\build_index.sql    # 建 HNSW 索引（全量入库后）
```

- 清单缺失、或磁盘上有清单外的文件 ⇒ 入库**直接失败**（不按文件名猜）。
- 有解析不了/`doc_key` 撞车的文件时，`make_manifest.py` 不写清单，只在 `<清单>.review.txt` 列出待处理项。

### 3.2 公司别名表（须在入库之后）

```powershell
python scripts\extract_full_names.py     # 读库：从年报封面抽法定全称 → data\company_full_names.csv
python scripts\build_companies.py        # 库 + 清单 + 上一步 → data\companies.csv
python scripts\load_companies.py         # 装载进 PG 的 companies / company_aliases
```

---

## 4. 环境自检

```powershell
python scripts\check_env.py
python scripts\check_env.py --online # 额外验 DeepSeek 连通性
```

期望结尾：

```log
结果: OK 16 | WARN 0 | FAIL 0 | SKIP 0
环境完全就绪。
```

报告落 `logs\check_env.log`；退出码 `0` = 通过，`1` = 有 FAIL。加 `--online` 会多一项，共 17 项。

---

## 5. 启动

```powershell
python scripts\serve.py
```

浏览器打开 **http://127.0.0.1:8000/**

| 参数 | 作用 |
|---|---|
| `--host <ip>` | 覆盖监听地址（默认配置里的 `127.0.0.1`） |
| `--port <n>` | 覆盖端口（默认 8000） |
| `--no-preload` | 跳过模型预热，启动 37s → 1s（只调前端时用；首问会等模型加载） |

```powershell
python scripts\serve.py --no-preload # 调前端：起得快，改完刷新即可
python scripts\serve.py --port 8080 # 换端口
```

- **不要**给 uvicorn 开 `reload`。
- 改前端**不用重启**：`index.html` 每次现读（响应头 `Cache-Control: no-store`）。

---

## 6. 就绪判据

启动日志出现这三行：

```log
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
预热完成，用时 37.2s
```

`预热完成` 是「可以提问了」的真正信号，**最晚出现**；预热没完就提问，前端表现为"发出去没反应"。

探针：

```powershell
Invoke-RestMethod http://127.0.0.1:8000/api/health
```

```json
{
  "ok": true,
  "model": "deepseek-flash",
  "has_key": true,
  "scope_modes": ["auto", "closed"],
  "final_top_k": 15,
  "rerank_enabled": true,
  "n_child_chunks": 12345,
  "n_documents": 234
}
```

`ok:true` = 真的连上库了（每次实连并数一遍）。`n_documents`/`n_child_chunks` 按实际读数为准。

---

## 7. 接口一览

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/` | 单页前端（前端用相对路径打 `/api/*`，不需要 CORS） |
| GET | `/api/health` | 探针，用 HTTP 状态码（异常时 503） |
| POST | `/api/chat` | **SSE 流式问答**，前端走这条。事件序列：`meta` → `reasoning`* → `delta`* → `audit` → `done` |
| POST | `/api/answer` | 非流式，一次拿完整答案 + 审计 + 思考过程。给脚本和人工核对用 |
| GET | `/api/corpus` | 库内文档清单，前端用它拼示例问题 |

请求体（`/api/chat` 与 `/api/answer` 相同）：

```json
{
  "query": "格力电器2025年的基本每股收益是多少？",
  "mode": "auto",
  "top_k": null,
  "backtrack": false,
  "show_reasoning": true,
  "history": [
    {"query": "美的集团2024年营业总收入是多少", "standalone": "美的集团2024年营业总收入是多少",
     "company": "美的集团", "year": 2024, "entities": ["美的集团"]}
  ]
}
```

- `mode`：只支持 **`auto`**（线上真实口径）与 `closed`（全库不过滤）。`open` 是评测作弊口径，传了返回 **400**。
- `top_k`：覆盖注入 prompt 的条数，**范围 1~20，超出返回 422**。不传时多实体走放大 `min(top_k_cap, final_top_k × 组数)`（`final_top_k=15` / `cap=20` ⇒ 两家 = 20 条）。**传了就以请求为准、不放大**。
- `backtrack`：是否额外取整节父块，默认关。
- `show_reasoning`：是否把思考过程流给前端，默认开。关掉**只停转发**，请求参数与答案一字不变。
- `history`：**多轮记忆**，前端持有并回传，按时间升序、**每轮都回传**（含被拒答 / 被拦的轮次）。每项含 `query` / `standalone` / `company` / `year` / `entities`，**不含答案**。不传 = 单轮。
  - ⚠ 单轮同样会调改写模型（无历史时只做归一化 + 抽实体，≈1000ms）。
  - 两个窗口：`multiturn.max_turns`（默认 100，服务端保留上限）、`multiturn.rewrite_turns`（默认 10，送进改写 prompt 的候选窗口，模型自报依据轮次到 `meta.rewrite.used_turns`）。
- 响应 `meta` 的路线级读数：`recall.routes`（两路射程内的共同/稠密独有/稀疏独有命中数）、`citations[].rrf_rank`（融合池位次）、`citations[].routes`（被哪几路召回、各路名次）、`rerank.n_dropped`（进池但没进 prompt 的条数）；另有 `effective_query` / `rewrite` / `scope.*` / `rerank.grouped`。

命令行验证（不流式）：

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:8000/api/answer -Method Post `
  -ContentType "application/json" `
  -Body '{"query":"格力电器2025年基本每股收益是多少","mode":"auto"}'
```

---

## 8. 停服务

- **前台启动**：那个窗口按 `Ctrl+C`。
- **后台启动**：按端口杀进程。

---

## 9. 故障速查

| 症状 | 原因 | 方案 |
|---|---|---|
| 启动即报端口占用 | 8000 被占 | `--port 8080`，或清掉占用的进程 |
| 前端发出去没反应 | 模型预热还没完 | 看日志等 `预热完成` |
| 提问后返回 401 | Key 没读到 | 检查环境变量长度是不是 35 |
| 提问后无引用/答不出东西 | 库空或检索层挂了 | 跑 `check_env.py`，再看`/api/health`的`n_documents` |
| 启动日志有 `预热失败` | 显存不够 | 服务仍能起；把 `.env` 的 `FINRAG_EMBED__DEVICE` 改成`cpu` |
| `/api/health` 返回 503 | 库连不上 | 看返回里的 `db_error`，再查PG服务是否 Running |
| 启动日志报模型加载失败/"repo 名非法" | 权重没就位（相对路径不存在时会被原样当 HF repo id） | 按 §1 配好权重路径，再跑 `check_env.py` |
| 页面显示"数据库不可用" | `.env` 不在（副本默认不带），DSN 退回代码默认 | 复制源项目的 `.env`，或改 `FINRAG_PG_DSN` |
| `check_env.py` 报 "端口可达，但连接被拒" | PG 在跑，但库名/用户名/密码对不上 | 对照 `.env` 的 DSN；`psql -U postgres -l` 看有没有 `finrag` |
| `check_env.py` 报 pgvector "扩展未创建" / "找不到 `vector.control`" | pgvector 未装进 PG 目录 | 跑 `scripts\install_pgvector.py`（见 §1.3），再重启 PG 服务 |

---

## 10. 配置优先级

```txt
系统环境变量 / .env > configs/default.yaml > 代码默认值
```

- 日常改端口/换设备/换DSN：改 `.env`。
- 改检索/分块/重排参数：改 `configs/default.yaml`。
- 改**多轮记忆**：`configs/default.yaml` 的 `multiturn` 段（`enabled` / `inherit_scope` / `max_turns` / `rewrite_turns` / `max_chars`），或环境变量 `FINRAG_MULTITURN__*`。
  - ⚠ `enabled=false` 或 `rewrite_turns=0` 会**一并失去库外公司识别与多实体分组**，只适合做纯召回。
- 改**多实体**：`retrieve`段的`group_pool_full`/`full_pool_max_groups`/`top_k_cap`。
- 改语料/日志/缓存目录：`paths` 段，或 `FINRAG_PATHS__RAW_DIR`/`FINRAG_PATHS__LOG_DIR`/`FINRAG_PATHS__CACHE_DIR`。

⚠ 嵌套字段用**双层下划线**：`FINRAG_LLM__MODEL`、`FINRAG_EMBED__DEVICE`。

---

## 11. 公网访问

无公网访问,`server.py`**没有**加`allow_origins=["*"]`。
