"""检索索引：BM25（字符 2-gram）＋ 向量（numpy 暴力搜索）＋ RRF 融合。

为什么自己做 BM25 而不是用 SQLite FTS5：FTS5 的 unicode61 分词器把连续中文当成一个 token，
"年假可以折现吗"整串匹配不到"员工享有年假"，中文检索直接失效。字符 bigram 是把中文当无空格语言
处理的标准做法，且不需要分词词典。
生产环境这两个索引分别换 OpenSearch / pgvector，接口不变（见 docs/adr/0002-hybrid-retrieval.md）。
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Protocol

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
            # 权限在打分之前生效：无权文档的分数直接不存在，不是"生成后再隐藏"
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


@dataclass(slots=True)
class SearchIndex:
    """一个切块策略对应一套索引。"""

    strategy: str
    items: dict[str, IndexItem] = field(default_factory=dict)
    bm25: BM25Index | None = None
    vector: VectorIndex | None = None

    @classmethod
    def build(cls, strategy: str, items: list[IndexItem], matrix: np.ndarray | None) -> SearchIndex:
        idx = cls(strategy=strategy, items={it.id: it for it in items})
        idx.bm25 = BM25Index(items)
        if matrix is not None and len(matrix):
            idx.vector = VectorIndex(items, matrix)
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


class Retriever(Protocol):  # pragma: no cover - 接口文档
    def search(self, query: str, top_n: int, allowed: set[str] | None) -> list[tuple[str, float]]: ...