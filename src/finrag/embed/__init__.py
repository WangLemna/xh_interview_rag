"""向量化层：BGE-M3 双输出（dense + sparse），入口 :mod:`finrag.embed.bge_m3`。"""

from finrag.embed.bge_m3 import (
    CACHE_KIND_DOC,
    CACHE_KIND_QUERY,
    EmbeddingSet,
    clear_cache,
    encode_query,
    encode_texts,
    format_sparse,
    get_encoder,
)

__all__ = [
    "CACHE_KIND_DOC",
    "CACHE_KIND_QUERY",
    "EmbeddingSet",
    "clear_cache",
    "encode_query",
    "encode_texts",
    "format_sparse",
    "get_encoder",
]