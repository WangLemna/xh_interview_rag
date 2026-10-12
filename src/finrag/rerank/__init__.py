"""重排层：bge-reranker-v2-m3（Cross-Encoder）精排。

``bge_reranker`` 负责打分（模型加载与 ``(query, passage)`` 配对打分），``pipeline``
负责组合（重排候选、截到 ``final_top_k``、``retrieve_and_rerank``）。依赖方向只有
``pipeline`` → ``store.search`` / ``bge_reranker``。
"""

from finrag.rerank.bge_reranker import get_reranker, score_pairs
from finrag.rerank.pipeline import (
    RerankResult,
    RerankedChunk,
    rerank,
    retrieve_and_rerank,
)

__all__ = [
    "RerankResult",
    "RerankedChunk",
    "get_reranker",
    "rerank",
    "retrieve_and_rerank",
    "score_pairs",
]
