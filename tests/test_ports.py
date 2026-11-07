"""P1-01：`Retriever` 死代码换成了两个真端口，实现类与端口必须对得上。

这些用例不是"文档"，是钉住签名的：端口与实现的参数名一旦漂移，`isinstance` 仍会通过
（`runtime_checkable` 只查属性存在），所以额外比一次 `inspect.signature`。

Q3 审计补的第三条（`test_retrieval_runs_on_injected_fake_stores`）：签名对上不等于可替换。
用两个不依赖 numpy/BM25 的内存假实现注入 `IndexStore.get`，跑通 `retrieve()` 全链路，
并断言假实现真的被构造、被查询、且拿到的命中顺序就是 `retrieve()` 的结果顺序。
"""

from __future__ import annotations

import inspect
import typing
from pathlib import Path

import numpy as np

import documind.index as index_mod
from documind.index import BM25Index, IndexItem, KeywordStore, SearchIndex, VectorIndex, VectorStore
from documind.pipeline import ingest_document
from documind.retrieve import retrieve


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


class FakeKeywordStore:
    """内存假关键词后端：不 import BM25、不碰 numpy，只按汉字集合重合度打分。

    存在的意义不是"再测一遍 BM25"，而是证明端口后面换什么都能被 `retrieve()` 用起来。
    """

    def __init__(self, items: list[IndexItem]) -> None:
        self.items = list(items)
        self.searches = 0
        self.allowed_seen: list[set[str] | None] = []
        self.last_hits: list[tuple[str, float]] = []

    def search(self, query: str, top_n: int, allowed: set[str] | None = None) -> list[tuple[str, float]]:
        self.searches += 1
        self.allowed_seen.append(allowed)
        q = set(query)
        hits = [
            (it.id, float(len(q & set(it.text))))
            for it in self.items
            if (allowed is None or it.doc_id in allowed) and (q & set(it.text))
        ]
        hits.sort(key=lambda x: (-x[1], x[0]))
        self.last_hits = hits[:top_n]
        return self.last_hits


class FakeVectorStore:
    """内存假向量后端：永远返回空（本轮只证明它被构造并被调用，不判相似度）。"""

    def __init__(self, items: list[IndexItem], matrix: np.ndarray) -> None:
        self.items = list(items)
        self.matrix = matrix
        self.searches = 0

    def search(
        self, query_vec: np.ndarray, top_n: int, allowed: set[str] | None = None
    ) -> list[tuple[str, float]]:
        self.searches += 1
        return []


async def test_retrieval_runs_on_injected_fake_stores(db, corpus_md: Path):
    """换后端只改注入：`retrieve()` 编排、ACL、span 装配一行不改也跑得通。"""
    doc = await ingest_document(db, corpus_md, acl_roles=["all"])

    made: dict[str, list] = {"kw": [], "vec": []}

    def build_kw(items):
        inst = FakeKeywordStore(items)
        made["kw"].append(inst)
        return inst

    def build_vec(items, matrix):
        inst = FakeVectorStore(items, matrix)
        made["vec"].append(inst)
        return inst

    res = await retrieve(
        db,
        "员工年假",
        role="employee",
        top_k=3,
        rerank=False,
        keyword_builder=build_kw,
        vector_builder=build_vec,
    )

    # 1) 假实现真的被用上了（构造 + 查询都计数，不是"注入后被忽略"）
    assert len(made["kw"]) == 1 and len(made["vec"]) == 1
    kw, vec = made["kw"][0], made["vec"][0]
    assert kw.searches > 0 and vec.searches > 0
    # 2) 端口拿到了 ACL 集合（契约：过滤在端口内完成，不是调用方事后挑）
    assert kw.allowed_seen and kw.allowed_seen[0] is not None
    # 3) 命中顺序 = 假端口返回的顺序；字段来自库里的行
    assert res.hits
    assert [h.chunk_id for h in res.hits] == [cid for cid, _ in kw.last_hits][: len(res.hits)]
    assert [h.bm25_rank for h in res.hits] == list(range(len(res.hits)))
    assert all(h.vec_rank is None for h in res.hits)  # 真 VectorIndex 会命中，假的不该
    assert res.debug["vector"] == 0
    # 4) 装配仍然正确：doc/span/score 由 retrieve 侧填，假端口只给 id+分数
    assert {h.doc_id for h in res.hits} == {doc["document_id"]}
    assert all(h.doc_title for h in res.hits)
    assert all(h.page_start >= 1 and h.char_end > h.char_start for h in res.hits)
    assert [h.score for h in res.hits] == [round(1 / (i + 1), 4) for i in range(len(res.hits))]
    # 5) 注入不污染默认路径：同一 store 换回默认后端时仍走真实现（缓存键含后端身份）
    real = await retrieve(db, "员工年假", role="employee", top_k=3, rerank=False)
    assert real.debug["vector"] > 0, "假后端建的索引被默认路径复用了——缓存键没算后端身份"