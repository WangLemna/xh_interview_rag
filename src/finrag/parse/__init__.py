"""解析层：把 PDF 转成结构 IR。

链路上位于入库之前。guard 提供判据与"有效字符数"口径，pageio 做单页提取，ir 定义
结构契约，layout 分析栏结构，heading 收束标题层级，pdf_parser 串起主流程，
cache 负责记忆化与批量入口。子模块按需导入，本文件不预先 import。
"""
