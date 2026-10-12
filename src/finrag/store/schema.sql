-- ============================================================
-- FinRAG 数据库结构
--
-- 单表 chunks 同时承载稠密向量 / 稀疏向量 / 全文索引 / 元数据，
-- 使多路召回可在单个 SQL 内完成。
--
-- 执行：psql -U postgres -d finrag -f src/finrag/store/schema.sql
-- ============================================================

CREATE EXTENSION IF NOT EXISTS vector;

-- ------------------------------------------------------------
-- 文档层：一份原始文件对应一行
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS documents (
    id            BIGSERIAL PRIMARY KEY,
    -- 稳定指纹：sha256(规范化相对路径) 前 16 位，重复入库时用于 upsert
    doc_id        TEXT        NOT NULL UNIQUE,
    source_path   TEXT        NOT NULL,
    title         TEXT,
    doc_type      TEXT        NOT NULL,               -- pdf（当前语料全部为年报 PDF）
    -- 内容指纹：正文 sha256，与库内不一致时触发重新分块与重新编码
    content_hash  TEXT        NOT NULL,
    category      TEXT,                               -- 财报 / 研报 / 交易笔记
    -- 自由元数据：市场、标的代码、报告期、来源机构等，供检索期做前置过滤
    meta          JSONB       NOT NULL DEFAULT '{}'::jsonb,
    n_chunks      INTEGER     NOT NULL DEFAULT 0,
    char_count    INTEGER     NOT NULL DEFAULT 0,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ------------------------------------------------------------
-- 分块层：检索与生成的最小单元
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS chunks (
    id           BIGSERIAL PRIMARY KEY,
    doc_id       TEXT    NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    chunk_index  INTEGER NOT NULL,
    content      TEXT    NOT NULL,
    token_count  INTEGER,

    -- 结构感知分块的产物：记录该块在原文中的位置，用于展示引用来源
    heading_path TEXT,                                -- 例如 "第二章 > 2.3 收入确认政策"
    page_from    INTEGER,
    page_to      INTEGER,

    -- 父块回溯：命中细粒度块时可上溯到更大上下文，缓解"碎片信息答不全"
    parent_id    BIGINT REFERENCES chunks(id) ON DELETE SET NULL,

    -- 块角色：child 参与检索与排名；parent 只服务回溯、不参与排名
    -- （否则同一段内容会因父块出现两次，挤占 final_top_k）。
    -- 召回 SQL 里仍显式写 role = 'child'：父块本就没有向量，但显式过滤比"靠无向量排除"更稳。
    role         TEXT    NOT NULL DEFAULT 'child' CHECK (role IN ('child', 'parent')),

    -- 稠密向量：BGE-M3 dense，1024 维，语义召回
    dense_vec    vector(1024),

    -- 稀疏向量：BGE-M3 lexical weights，维度 = XLM-RoBERTa 词表大小，
    -- 承担关键词召回，规避中文分词扩展的编译依赖。换模型时此处维度须与
    -- src/finrag/embed/bge_m3.py 一致。
    -- 文本字面量必须带 `/维度数` 后缀（如 '{1:1,5:2}/250002'），索引从 1 开始；
    -- 缺后缀或含小数点都会被拒。运算符：`<#>` 负内积、`<->` 欧氏、`<=>` 余弦、`<+>` 曼哈顿。
    sparse_vec   sparsevec(250002),

    -- 原生全文检索列（tsv）已随 lexical 路一并移除：``simple`` parser 不识别中文词边界，
    -- 全文路在中文语料上是净负收益，而生成列会为每行白算一次 to_tsvector。真要用再作为迁移加回。

    meta         JSONB   NOT NULL DEFAULT '{}'::jsonb,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),

    UNIQUE (doc_id, chunk_index)
);

-- ------------------------------------------------------------
-- 公司层：一家公司一行（名称 → 证券代码 的权威映射）
--
-- 为什么独立于 documents：同一公司的多种写法（清单简称 / 法定全称 / 曾用名 /
-- ST 时期的带标签写法）散落在各份 ``documents.meta.aliases`` 里，既难剪枝，
-- 也无法保证"同一家只有一行"。这里按**证券代码**收敛成一家一行，是别名解析的
-- **唯一裁决来源**。
--
-- 数据来源：``scripts/build_companies.py`` 生成 CSV → ``scripts/load_companies.py`` 装载。
-- 法定全称取自**年报封面**（公司自己披露，见 ``scripts/extract_full_names.py``）。
-- 人工定稿只留两处：``extract_full_names.MANUAL_OVERRIDES``（定稿全称）与
-- ``build_companies.EXTRA_ALIASES``（追加别名，如更名前旧名）。
--
-- ⚠ 本表**只管名字**，不管文档；代码 → 年份 → doc_id 的展开仍走 ``documents``。
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS companies (
    code             TEXT PRIMARY KEY,                 -- 证券代码，如 600519
    short_name       TEXT    NOT NULL,                 -- 清单面简称（用户主用写法）
    full_name        TEXT,                              -- 法定全称（年报封面权威源），可空
    pinyin           TEXT,                              -- 简称全拼（拼音容错匹配用）
    pinyin_initials  TEXT,                              -- 简称拼音首字母
    n_docs           INTEGER NOT NULL DEFAULT 0,       -- 库内该公司的年报份数
    updated_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS company_aliases (
    code    TEXT NOT NULL REFERENCES companies(code) ON DELETE CASCADE,
    alias   TEXT NOT NULL,                             -- 该公司的任一写法（简称/全称/曾用名/带标签写法）
    PRIMARY KEY (code, alias)
);

-- ------------------------------------------------------------
-- 索引
-- ------------------------------------------------------------

-- 别名 → 代码 是查询期的反查方向（问句命中别名后要拿到 code），无索引会每问句全表扫。
CREATE INDEX IF NOT EXISTS idx_company_aliases_alias
    ON company_aliases (alias);

-- 公司代码 → 文档：建别名表时要按 code 把别名展开成 (code, year, doc_id)，
-- 缺索引会退化成 documents 全表扫（几千家公司时尤其明显）。
-- chunks 的召回 SQL 不走这条路径，故索引只加在 documents 上。
CREATE INDEX IF NOT EXISTS idx_documents_code
    ON documents ((meta->>'code'));

-- 稠密路 HNSW **不在此处建**：边插边建极慢、图质量也差。全量灌完后跑
-- ``scripts/build_index.sql`` 一次性创建（见该文件）。
--
-- 稀疏路不建 ANN 索引：稀疏向量非零元素仅百余个，精确内积代价很低
-- （十万级 chunk 仍是毫秒级），可换 100% 召回并避免近似误差叠加。
-- ⚠ 若语料扩到千万级 chunk，这条"毫秒级"假设会失效，届时需为 sparse_vec 另建 HNSW。

CREATE INDEX IF NOT EXISTS idx_chunks_doc_id
    ON chunks (doc_id);

-- 自引用外键 ``chunks.parent_id`` 的支撑索引。**不可省**：入库每次重写都要
-- ``DELETE FROM chunks WHERE doc_id = ...``，而 ``ON DELETE SET NULL`` 会为每个被删的父块
-- 反查引用它的子行；没有这个索引就是"每删一行全表扫一次"，重入库直接不可用。
-- 用部分索引：只有带父的片才需要被反查。
CREATE INDEX IF NOT EXISTS idx_chunks_parent_id
    ON chunks (parent_id) WHERE parent_id IS NOT NULL;

-- role 不建索引：子块约占 85%，`WHERE role = 'child'` 选择性极低，PG 会顺序扫描。
-- 若要建，应为 `WHERE role = 'parent'` 的部分索引，服务列出父块这类反向查询。

-- ============================================================
-- 参考：RRF（Reciprocal Rank Fusion）融合
--
-- 选 RRF 而非加权分数求和：稠密路输出余弦相似度、稀疏路输出内积，量纲不可比，
-- RRF 只看排名，天然免疫量纲差异。
--
-- 本段仅为概念示意，非当前实现：search.py 改为一路一条 SQL、在应用层融合，
-- 以避免某一路无命中时占位符个数与顺序错乱。
-- ============================================================
--
-- 各路都带 role = 'child'，且向量列显式 IS NOT NULL：`ORDER BY dense_vec <=> q`
-- 对 NULL 行给 NULL 距离并默认排到最后，LIMIT 够大时会混入无法向量召回的片。
-- 稠密路还需 SET LOCAL hnsw.ef_search（默认 40，偏小）。
--
-- WITH dense AS (
--     SELECT id, ROW_NUMBER() OVER (ORDER BY d) AS rank, d AS score
--     FROM (
--         SELECT id, dense_vec <=> :q_dense AS d
--         FROM chunks
--         WHERE role = 'child' AND dense_vec IS NOT NULL
--         ORDER BY d
--         LIMIT :dense_top_k
--     ) t
-- ),
-- sparse AS (
--     -- <#> 返回负内积，升序即等价于内积降序（与 <=> 同向，无需取负）
--     SELECT id, ROW_NUMBER() OVER (ORDER BY s) AS rank, s AS score
--     FROM (
--         SELECT id, sparse_vec <#> :q_sparse AS s
--         FROM chunks
--         WHERE role = 'child' AND sparse_vec IS NOT NULL
--         ORDER BY s
--         LIMIT :sparse_top_k
--     ) t
-- )
-- -- RRF 逐 id 累加各路 1/(rrf_k + rank)；分数是唯一可跨路相加的量
-- -- （dense_dist 与 sparse_negip 量纲不可比，任何排序都不得引入）。
-- SELECT id, SUM(1.0 / (:rrf_k + rank)) AS rrf_score, COUNT(*) AS n_hits
-- FROM (
--     SELECT id, rank FROM dense
--     UNION ALL
--     SELECT id, rank FROM sparse
-- ) u
-- GROUP BY id
-- ORDER BY rrf_score DESC
-- LIMIT :rerank_candidates;
