"""检索索引：BM25（字符 2-gram）＋ 向量（numpy 暴力搜索）＋ RRF 融合。

为什么自己做 BM25 而不是用 SQLite FTS5：FTS5 的 unicode61 分词器把连续中文当成一个 token，
"年假可以折现吗"整串匹配不到"员工享有年假"，中文检索直接失效。字符 bigram 是把中文当无空格语言
处理的标准做法，且不需要分词词典。
换 OpenSearch / pgvector 时要实现的是下面两个端口（`KeywordStore` / `VectorStore`），
注入点是 `IndexStore.get(keyword_builder=..., vector_builder=...)`（`SearchIndex.build` 转发）。
但"接口不变"只覆盖检索这一层：dev 侧的组装（`IndexStore` 全表 vstack、`chunks.embedding` BLOB）
仍要改，真实清单见 docs/adr/0002-hybrid-retrieval.md 与 docs/ARCH-REVIEW.md §4。
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np

from .embed import cosine_search


def tokenize(text: str) -> list[str]:
    t = re.sub(r"\s+", "", text.lower())
    grams = [t[i : i + 2] for i in range(max(len(t) - 1, 0))]
    words = re.findall(r"[a-z0-9]+", t)
    return grams + words


@dataclass(slots=True)
class IndexItem:
    id: str
    text: str
    doc_id: str
    strategy: str
    page_start: int = 0
    char_start: int = 0
    page_end: int = 0
    char_end: int = 0
    section_path: str = ""
    parent_id: str | None = None
    is_parent: bool = False


class BM25Index:
    def __init__(self, items: list[IndexItem], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.ids = [it.id for it in items]
        self.doc_ids = np.array([it.doc_id for it in items], dtype=object)
        self.tf: list[Counter[str]] = []
        self.lengths = np.zeros(len(items), dtype=np.float32)
        df: Counter[str] = Counter()
        for i, it in enumerate(items):
            toks = tokenize(it.text)
            c = Counter(toks)
            self.tf.append(c)
            self.lengths[i] = len(toks)
            df.update(c.keys())
        self.n = len(items)
        self.avgdl = float(self.lengths.mean()) if self.n else 0.0
        self.idf = {t: math.log(1 + (self.n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self._row = {id_: i for i, id_ in enumerate(self.ids)}

    def search(self, query: str, top_n: int, allowed: set[str] | None = None) -> list[tuple[str, float]]:
        if not self.n:
            return []
        qtoks = tokenize(query)
        scores = np.zeros(self.n, dtype=np.float32)
        for tok in set(qtoks):
            idf = self.idf.get(tok)
            if idf is None:
                continue
            for i, c in enumerate(self.tf):
                f = c.get(tok)
                if not f:
                    continue
                dl = self.lengths[i] or 1.0
                scores[i] += idf * (f * (self.k1 + 1)) / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
        if allowed is not None:
            # 输出侧掩码：整表已经算完 BM25，这里把无权行置 -1，结果里既不会出现也不会以低分上榜；
            # 省算力的前置子矩阵过滤在向量侧做（VectorIndex.search），BM25 侧不做（见 ADR-0003）
            mask = np.array([d in allowed for d in self.doc_ids])
            scores = np.where(mask, scores, -1.0)
        k = min(top_n, self.n)
        idx = np.argpartition(-scores, k - 1)[:k]
        idx = idx[np.argsort(-scores[idx])]
        return [(self.ids[i], float(scores[i])) for i in idx if scores[i] > 0]


class VectorIndex:
    def __init__(self, items: list[IndexItem], matrix: np.ndarray) -> None:
        self.ids = [it.id for it in items]
        self.matrix = matrix
        self.doc_ids = np.array([it.doc_id for it in items], dtype=object)

    def search(self, query_vec: np.ndarray, top_n: int, allowed: set[str] | None = None) -> list[tuple[str, float]]:
        if self.matrix.size == 0:
            return []
        if allowed is not None:
            mask = np.array([d in allowed for d in self.doc_ids])
            if not mask.any():
                return []
            sub_ids = [i for i, m in zip(self.ids, mask, strict=True) if m]
            sub = self.matrix[mask]
            return cosine_search(query_vec, sub, sub_ids, top_n)
        return cosine_search(query_vec, self.matrix, self.ids, top_n)


@runtime_checkable
class KeywordStore(Protocol):
    """关键词检索端口。`BM25Index` 是 dev 实现，生产换 OpenSearch。

    契约两条：① 返回 (chunk_id, score) 且按分数降序；② `allowed` 必须在端口内生效，
    无权文档的 chunk_id 不得出现在返回值里（实现可以是打分前滤行、也可以是打分后置负分掩码，
    但**不能**返回后再由调用方隐藏——否则无权文档的名次会侧漏，见 ADR-0003）。
    注入点：`IndexStore.get(keyword_builder=...)`，默认就是 `BM25Index`。
    """

    def search(self, query: str, top_n: int, allowed: set[str] | None = None) -> list[tuple[str, float]]: ...


@runtime_checkable
class VectorStore(Protocol):
    """向量检索端口。入参是**已编码的查询向量**：向量化留在调用方，端口不认 embedder。

    这里刻意收 `np.ndarray` 而不是 str —— 旧 `Retriever` 声明 `search(query: str, ...)`，
    与 `VectorIndex.search(query_vec, ...)` 根本对不上，是零实现零引用的死代码（P1-01 已删）。
    """

    def search(
        self, query_vec: np.ndarray, top_n: int, allowed: set[str] | None = None
    ) -> list[tuple[str, float]]: ...


# 后端工厂：组合根（`IndexStore.get`）只认这两个类型，换实现 = 换一个可调用对象。
# 参数就是端口实现的构造入参，`BM25Index(items)` / `VectorIndex(items, matrix)` 是默认值。
KeywordBuilder = Callable[[list[IndexItem]], KeywordStore]
VectorBuilder = Callable[[list[IndexItem], np.ndarray], VectorStore]


@dataclass(slots=True)
class SearchIndex:
    """一个切块策略对应一套索引。"""

    strategy: str
    items: dict[str, IndexItem] = field(default_factory=dict)
    # 字段类型是端口而不是实现：换 OpenSearch/pgvector 时 SearchIndex 不需要改
    bm25: KeywordStore | None = None
    vector: VectorStore | None = None

    @classmethod
    def build(
        cls,
        strategy: str,
        items: list[IndexItem],
        matrix: np.ndarray | None,
        keyword_builder: KeywordBuilder = BM25Index,
        vector_builder: VectorBuilder = VectorIndex,
    ) -> SearchIndex:
        idx = cls(strategy=strategy, items={it.id: it for it in items})
        idx.bm25 = keyword_builder(items)
        if matrix is not None and len(matrix):
            idx.vector = vector_builder(items, matrix)
        return idx


def rrf_fuse(
    runs: list[list[tuple[str, float]]], k: int = 60, weights: list[float] | None = None
) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion：只用名次不用分数，避免 BM25 与 cosine 的分数量纲打架。"""
    weights = weights or [1.0] * len(runs)
    acc: dict[str, float] = defaultdict(float)
    for w, run in zip(weights, runs, strict=True):
        for rank, (doc_id, _score) in enumerate(run):
            acc[doc_id] += w / (k + rank + 1)
    return sorted(acc.items(), key=lambda x: -x[1])
