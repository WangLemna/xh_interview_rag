"""bge-reranker-v2-m3 打分器 —— Cross-Encoder 精排。

对双路召回的候选池做 ``(query, passage)`` 配对打分，回答"谁该排第一"。分数是原始 logit（不过 sigmoid）；
``max_length`` 是配对后的总长上限，超长从片尾截；设备走 ``devices=``（复数），传 ``device=`` 会被静默忽略而退回自动探测。
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Sequence

from finrag.config import AppConfig, get_config

__all__ = ["get_reranker", "score_pairs"]

logger = logging.getLogger(__name__)

# 加载锁：get_reranker 是进程级单例，并发首调会各建一份 1.2GB 权重
_LOAD_LOCK = threading.Lock()

_RERANKER: Any = None


def get_reranker(cfg: AppConfig | None = None) -> Any:
    """进程内单例的 ``FlagReranker``（加载 1.2GB 权重只应发生一次）；设备走 ``devices=``（复数），
    传 ``device=`` 会被静默忽略而退回自动探测。
    """
    cfg = cfg or get_config()
    global _RERANKER
    with _LOAD_LOCK:
        if _RERANKER is None:
            from FlagEmbedding import FlagReranker

            logger.info(
                "加载 reranker 权重：%s（devices=%s, fp16=%s, max_length=%d）",
                cfg.rerank.model_path, cfg.rerank.device,
                cfg.rerank.use_fp16, cfg.rerank.max_length,
            )
            _RERANKER = FlagReranker(
                cfg.rerank.model_path,
                use_fp16=cfg.rerank.use_fp16,
                devices=cfg.rerank.device,
                # 配对后总长上限，非片长上限；超长从片尾截断
                max_length=cfg.rerank.max_length,
                # 显式传，不吃默认值：默认值一变阈值的量纲就变了
                normalize=False,
            )
        return _RERANKER


def score_pairs(
    query: str,
    passages: Sequence[str],
    cfg: AppConfig | None = None,
    *,
    reranker: Any = None,
) -> list[float]:
    """给 ``[(query, passage)]`` 逐对打分，返回原始 logit、与 ``passages`` 同序；``reranker`` 可注入，
    返回前断言条数与输入相等，防止分数与片错位。
    """
    cfg = cfg or get_config()
    items = list(passages)
    if not items:
        return []

    rr = reranker if reranker is not None else get_reranker(cfg)
    pairs = [[query, p] for p in items]
    raw = rr.compute_score(
        pairs, batch_size=cfg.rerank.batch_size, normalize=False
    )

    # compute_score 对单条输入可能返回标量而非列表：按"能否迭代"判别，
    # 不按类型判别（np.float32 不可迭代）。
    try:
        scores = list(raw)
    except TypeError:
        scores = [raw]
    out = [float(x) for x in scores]

    if len(out) != len(items):
        raise RuntimeError(
            f"reranker 返回 {len(out)} 个分数，但有 {len(items)} 条候选 —— "
            "分数与片已经错位。这类错**不会**从结果长相上暴露（排出来的顺序"
            "完全正常，只是每个分数挂给了别人的内容），所以在这里直接停。"
        )
    return out
