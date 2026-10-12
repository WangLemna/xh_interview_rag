-- ============================================================
-- 全量入库**之后**创建向量索引（稠密路 HNSW）。
--
-- 为什么不在 schema.sql 里建：HNSW 边插边建极慢，且新点插入时图是增量生长的、
-- 质量不如一次性 bulk 构建。千万级 chunk 下这个差别是数量级的。
--
-- 执行：
--   psql -U postgres -d finrag -f scripts/build_index.sql
--
-- maintenance_work_mem 需按机器内存调（默认 64MB，建千万级 HNSW 会走磁盘排序、很慢）。
-- 有多个 CPU 核时 PostgreSQL 会并行建 B-tree，但 HNSW 的并行度受 hnsw 扩展实现限制。
--
-- ⚠ **上限是 2GB − 1kB，写 4GB 会直接报错**。`maintenance_work_mem` 的硬上限是
--   ``MAX_KILOBYTES`` = ``INT_MAX / 1024`` = **2097151 kB ≈ 2047.999 MB**，这是**所有**
--   PostgreSQL 版本的硬限制（不限本机、不限 Windows）。所以：
--     · 写 '4GB'（4194304 kB）→ 报 `4194304 kB is outside the valid range ... (64 kB .. 2097151 kB)`；
--     · 写 '2GB'（2097152 kB）→ **只差 1 kB 也被拒**；
--     · 取 '2047MB'（= 2096128 kB）才稳。
--   本项目实测：343,693 条 × 1024 维，HNSW 构建需常驻约 1.5GB，'2047MB' 够用。
-- ============================================================

SET maintenance_work_mem = '2047MB';

CREATE INDEX IF NOT EXISTS idx_chunks_dense_hnsw
    ON chunks USING hnsw (dense_vec vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- 稀疏路暂不建 ANN 索引（见 schema.sql 的说明）；若语料扩到千万级再评估
-- 为 sparse_vec 建 HNSW（pgvector 0.7+ 支持 sparsevec 索引）。
