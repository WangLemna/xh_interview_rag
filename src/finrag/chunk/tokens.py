"""分块层的 token 计数口径：默认使用 BGE-M3 的 tokenizer。

分块长度必须与编码器对齐 —— ``embed.max_length`` 以该 tokenizer 的 token 为单位，按别的
口径切到 512 到编码层可能已超长而被静默截断；口径只此一份，并提供可注入的 callable 以脱离模型文件测试。
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Final

__all__ = [
    "CountTokens",
    "TOKENIZER_FILENAME",
    "TokenizerNotFound",
    "bge_m3_counter",
    "char_counter",
    "resolve_tokenizer_file",
]

# 计数签名：文本 → token 数。分块层只依赖这个签名，不依赖任何具体实现。
CountTokens = Callable[[str], int]

TOKENIZER_FILENAME: Final[str] = "tokenizer.json"


class TokenizerNotFound(FileNotFoundError):
    """找不到 tokenizer 文件。刻意不静默降级成字符近似。"""


def resolve_tokenizer_file(model_dir: str | Path) -> Path:
    """在模型目录里定位 ``tokenizer.json``；也允许直接传文件路径。"""
    p = Path(model_dir)
    if p.is_file():
        return p
    cand = p / TOKENIZER_FILENAME
    if not cand.exists():
        raise TokenizerNotFound(
            f"未找到 {cand}。分块层需要 BGE-M3 的 tokenizer 来对齐长度口径"
            f"（见 finrag/chunk/tokens.py 的模块说明）。"
            f"若当前只想跑不依赖模型的结构性冒烟，请显式注入 count_tokens，"
            f"例如 finrag.chunk.tokens.char_counter。"
        )
    return cand


@lru_cache(maxsize=4)
def _load(path: str) -> Any:
    """加载并缓存 tokenizer。按路径缓存 —— 同路径重复调用只读一次盘。"""
    from tokenizers import Tokenizer

    return Tokenizer.from_file(path)


def bge_m3_counter(model_dir: str | Path) -> CountTokens:
    """按 BGE-M3 的 tokenizer 计数，是分块层的默认口径；``add_special_tokens=False`` 只数内容长度、
    不含边界标记；返回闭包供注入替换。
    """
    tokenizer = _load(str(resolve_tokenizer_file(model_dir)))

    def count(text: str) -> int:
        if not text:
            return 0
        return len(tokenizer.encode(text, add_special_tokens=False).ids)

    return count


def char_counter(text: str) -> int:
    """字符数近似。仅供测试与结构性冒烟、**不可用于生产分块** —— 字符数与 token 数无稳定换算倍率，
    用它跑出的分块边界与生产口径相差难估。
    """
    return len(text)
