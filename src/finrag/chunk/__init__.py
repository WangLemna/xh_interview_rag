"""分块层：结构感知切分，IR → 检索单元。

``tokens`` 提供 token 计数口径（默认 BGE-M3 tokenizer，可注入替换），``chunker``
是切分主流程。本层不依赖 PyMuPDF 与 torch，只依赖 ``parse.ir`` 契约与 ``config``。
"""
