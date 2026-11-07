"""P1-01：`Retriever` 死代码换成了两个真端口，实现类与端口必须对得上。

这些用例不是"文档"，是钉住签名的：端口与实现的参数名一旦漂移，`isinstance` 仍会通过
（`runtime_checkable` 只查属性存在），所以额外比一次 `inspect.signature`。
"""

from __future__ import annotations

import inspect
import typing

import numpy as np

import documind.index as index_mod
from documind.index import BM25Index, IndexItem, KeywordStore, SearchIndex, VectorIndex, VectorStore


def _items() -> list[IndexItem]:
    return [IndexItem(id="a", text="员工每年享有带薪年假五天", doc_id="d1", strategy="structure")]


def test_bm25_implements_keyword_store():
    idx = BM25Index(_items())
    assert isinstance(idx, KeywordStore)
    assert inspect.signature(BM25Index.search).parameters.keys() == (
        inspect.signature(KeywordStore.search).parameters.keys()
    )


def test_vector_index_implements_vector_store():
    matrix = np.zeros((1, 4), dtype=np.float32)
    idx = VectorIndex(_items(), matrix)
    assert isinstance(idx, VectorStore)
    assert inspect.signature(VectorIndex.search).parameters.keys() == (
        inspect.signature(VectorStore.search).parameters.keys()
    )


def test_vector_port_takes_a_query_vector_not_a_string():
    """签名必须是 query_vec：收 str 的向量端口谁实现都用不上（旧 Retriever 的病）。"""
    assert list(inspect.signature(VectorStore.search).parameters)[1] == "query_vec"


def test_search_index_depends_on_ports_not_implementations():
    hints = typing.get_type_hints(SearchIndex)
    assert hints["bm25"] == KeywordStore | None
    assert hints["vector"] == VectorStore | None


def test_dead_retriever_protocol_is_gone():
    assert not hasattr(index_mod, "Retriever")