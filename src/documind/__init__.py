"""DocuMind 公开 API 面（P0-03）。

只有 `__all__` 里的名字是**对外契约**。其余模块（config / db / embed / rerank /
retrieve / generate / chunking / pipeline / app / evals）与任何下划线开头的名字都是内部实现，
可以随时改，不构成下游可依赖的接口。契约由 `tests/test_documind_contract.py` 钉住，
下游（bidpilot）的用法见 `bidpilot/docs/DEPENDENCY.md` §1。

五个能力 → 七个名字：
    C1 格式解析    parse_file / ParsedDocument / EmptyDocumentError（P0-04 的公开失败模式）
    C2 段落结构    Paragraph
    C3 页文本      ParsedDocument.pages / page_count（属性，随类导出）
    C4 span→原文   ParsedDocument.slice_text（方法，随类导出）
    C5 中文词法+BM25  tokenize / IndexItem / BM25Index

`__all__` 的个数比"五个能力"多，是因为：C3/C4 是方法不是模块级符号，C5 由三个名字构成，
而 C1 的成功/失败两侧都必须可从顶层捕获（bidpilot `sections.py` catch 的就是
EmptyDocumentError，见 DEPENDENCY.md §2 第 2 行）。DEPENDENCY.md §4.1 要求
parse_file/ParsedDocument/Paragraph/tokenize/IndexItem/BM25Index 全部可从顶层导入，
所以按能力收敛、按可导入面展开。
"""

from __future__ import annotations

from .index import BM25Index, IndexItem, tokenize
from .parsing import EmptyDocumentError, Paragraph, ParsedDocument, parse_file

__version__ = "0.1.0"

__all__ = [
    "BM25Index",
    "EmptyDocumentError",
    "IndexItem",
    "Paragraph",
    "ParsedDocument",
    "parse_file",
    "tokenize",
]