"""控制台输出统一按 UTF-8 编码。

Windows 中文版终端的默认编码是 GBK(cp936)，而本项目的中文注释里大量出现
``⚠ ✓ ⇒`` 这类符号——它们在 GBK 里没有对应码位（只有``→ · ≥ — ｜`` 等能编）。
一旦 Python 未开启 UTF-8 模式（``PYTHONUTF8``/``-X utf8``，两者都不是默认），
``print()`` 就会抛 ``UnicodeEncodeError: 'gbk' codec can't encode character``。

与 :mod:`finrag.config` 的 ``FILE_ENCODING`` 是两件事：
那边管**读**配置文件（UTF-8 字节 → str），这边管**写**终端（str → 终端字节）。
两个方向都得钉死，否则换个终端就会随机炸一个方向。
"""

from __future__ import annotations

import sys


def use_utf8_stdout() -> None:
    """把 stdout / stderr 切到 UTF-8。幂等，失败静默（流不支持 reconfigure 时放弃即可）。

    必须在 ``print`` / ``logging.basicConfig`` **之前**调用：logging 的 StreamHandler
    会缓存创建时的 ``sys.stderr``，晚于它调用只改一半。
    """
    for name in ("stdout", "stderr"):
        stream = getattr(sys, name, None)
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            # 已被替换成别的流对象（如 pytest 的 capture 流），或已关闭 —— 都不是错误。
            pass
