# FinRAG

面向 A 股上市公司年报的检索增强问答系统。**PDF进，带引用的答案出。**

- 启动见 [FinRAG-启动指南.md](FinRAG-启动指南.md)

---

## 一、链路

```txt
PDF
 └─ parse      准入守卫 → 单页提取 → 栏结构 → 标题层级 → 表格结构化 → 结构 IR
     └─ chunk   结构感知分块：按标题切节 → 合并碎片 → 硬切（吸附句末）→ 父子块
         └─ embed   BGE-M3 单次前向产出 dense + sparse 两种表示
             └─ store   PostgreSQL 17+pgvector：documents/chunks幂等落库

问题
 └─ rewrite    多轮问句改写：补指代/省略主语，抽出公司实体（含库外）→ 自足问句
     └─ resolve    问句 → 公司（可多家）/年份 → 文档分组；认不出则沿用上一轮
         └─ retrieve   按组（公司）双路召回 → RRF融合 → BGE-Reranker重排 → 按组保底取top-k
             └─ generate   组装 prompt（引用标签 + `<缺失资料>`）→ DeepSeek 流式生成 → 数字审计
                 └─ api    FastAPI + SSE
```

## 二、目录结构

```txt
.
├── README.md
├── FinRAG-启动指南.md
├── requirements.txt              # 依赖 
├── .env.example                  # 环境变量模板（Key 走系统环境变量，不写文件）
├── configs/
│   └── default.yaml              # 检索链路全部参数
├── eval/                         # 测评集
├── scripts/
│   ├── serve.py                  # 启动服务
│   ├── ingest.py                 # 批量入库
│   ├── check_env.py              # 环境自检：依赖 / 配置 / 数据库 / 模型
│   ├── check_llm_key.py          # 验 LLM Key 的读取路径
│   ├── make_manifest.py          # 生成语料清单（PDF 目录 → manifest CSV）
│   ├── build_index.sql           # 全量入库后建 HNSW 索引（不随 schema 建）
│   ├── install_pgvector.py       # pgvector Windows 部署（预编译包）
│   ├── extract_full_names.py     # 从年报封面抽法定全称 → data/company_full_names.csv（只读）
│   ├── build_companies.py        # 公司级权威信息表 → data/companies.csv（只读库 + 读清单）
│   ├── load_companies.py         # 把公司表装载进 PG 的 companies / company_aliases（全量替换）
└── src/finrag/
    ├── config.py                 # 三层配置合并：环境变量 > yaml > 代码默认值
    ├── console.py                # 终端编码：入口处把 stdout/stderr 切 UTF-8（Windows GBK 环境）
    ├── manifest.py               # 语料清单：doc_key ↔ 文件 ↔ 元数据（身份与 company/year 的唯一来源）
    ├── alias.py                  # 公司名候选的纯函数（入库时预计算全称，供别名表用）
    ├── service.py                # 组合根：多轮改写 → 检索 → 重排 → 组装 → 生成
    ├── parse/                    # PDF → 结构 IR
    │   ├── guard.py              # 准入判据（阈值 + 空页 / 扫描件判定）
    │   ├── pageio.py             # 单页提取，"有效字符数"与"印刷页码"的唯一口径
    │   ├── ir.py                 # IR 契约（ParsedDoc / Section / Block / PageRef）
    │   ├── layout.py             # 栏结构分析
    │   ├── heading.py            # 标题层级（TOC 主路径 + 版式兜底）
    │   ├── table.py              # 表格结构化（表头对账 / 折叠单元格）
    │   ├── pdf_parser.py         # 主流程
    │   └── cache.py              # 解析结果记忆化 + 并行批量
    ├── chunk/                    # 结构感知分块
    │   ├── chunker.py
    │   └── tokens.py             # 长度口径统一走 BGE-M3 tokenizer
    ├── embed/bge_m3.py           # dense + sparse 双输出
    ├── rerank/                   # RRF 融合 + Cross-Encoder 重排
    │   ├── pipeline.py
    │   └── bge_reranker.py
    ├── store/                    # 持久化与多路检索
    │   ├── schema.sql            # 表结构单点定义（documents / chunks / companies / company_aliases）
    │   ├── pool.py               # 进程级连接池（优先 psycopg_pool，缺席回落内置极简池）
    │   ├── writer.py             # 幂等写入
    │   ├── search.py             # 双路召回 + RRF 融合
    │   └── resolve.py            # 问句 → 公司（可多家）/ 年份 → 文档分组
    ├── gen/                      # LLM 客户端 + 多轮改写 + prompt 组装 + 数字审计
    │   ├── client.py
    │   ├── rewrite.py            # 多轮问句改写（含话题判定，单次调用）
    │   └── prompt.py
    └── api/
        ├── server.py             # FastAPI + SSE
        └── static/index.html     # 单页前端
```

## 三、快速开始

```powershell
# 1. 装依赖
python -m pip install --user -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt

# 2. 配环境变量：复制 .env.example 为 .env 并填好 DSN
#    API Key 走系统环境变量，不写进文件
[Environment]::SetEnvironmentVariable("FINRAG_LLM__API_KEY", "sk-你的key", "User")

# 3. 建库并建表（HNSW 不在此建，见第 8 步）
createdb -U postgres finrag
psql -U postgres -d finrag -f src/finrag/store/schema.sql

# 4. 放语料 + 生成清单
#    命名约定：<代码>_<名称简称>_<年份>年年度报告.pdf，如 000061_农产品_2023年年度报告.pdf
#    语料目录与清单路径写进 .env（清单是身份与元数据的唯一来源，缺失即拒绝入库）：
#      FINRAG_PATHS__RAW_DIR=D:/corpus/pdf
#      FINRAG_PATHS__MANIFEST=D:/corpus/corpus_manifest.csv
python scripts\make_manifest.py     # PDF 目录 → corpus_manifest.csv
#    有解析不了 / doc_key 撞车的文件时它不写清单，只在 <清单>.review.txt 列出待处理项

# 5. 下模型：BAAI/bge-m3 与 BAAI/bge-reranker-v2-m3 放到 models/ 下（默认配置就指向这里）
#    想放到项目外、或改成走 HF 缓存，请写 .env —— 别改 configs/default.yaml（那个进版本库）：
#      FINRAG_EMBED__DENSE_MODEL=D:/Models/bge-m3      # 项目外的本地目录
#      FINRAG_RERANK__MODEL=BAAI/bge-reranker-v2-m3    # 不落地，走 HF 下载

# 6. 自检 → 入库
python scripts\check_env.py
python scripts\ingest.py

# 7. 生成公司别名表
#    三步都只读库/清单 ⇒ 必须在入库之后跑
python scripts\extract_full_names.py     # 读库：从年报封面抽法定全称 → data\company_full_names.csv
python scripts\build_companies.py        # 库 + 清单 + 上一步 → data\companies.csv
python scripts\load_companies.py         # 装载进 PG 的 companies / company_aliases

# 8. 灌完后建向量索引（HNSW 边插边建极慢，故与建表分开）→ 启动
psql -U postgres -d finrag -f scripts\build_index.sql
python scripts\serve.py
```

启动后访问 **http://127.0.0.1:8000/** 。

前置条件（Python 3.12 / PostgreSQL 17 + pgvector 0.8.0 / 磁盘约 7GB / GPU 可选 / DeepSeek Key）、
参数说明与故障定位见 [FinRAG-启动指南.md](FinRAG-启动指南.md)。
