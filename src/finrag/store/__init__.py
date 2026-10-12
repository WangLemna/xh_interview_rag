"""持久化层：PostgreSQL 17 + pgvector。

表结构由 ``schema.sql`` 单点定义：``writer`` 把解析 / 分块 / 编码的产物按
``doc_id`` 幂等写入 ``documents`` / ``chunks``，``search`` 做多路召回与 RRF 融合。
"""