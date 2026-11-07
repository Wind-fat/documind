"""重排：bge-reranker-base（cross-encoder），无模型时降级为词面重排。

重排的作用在消融里看得见：RRF 融合后的前 30 条里，正确答案常常排在第 7-12 位，
cross-encoder 能把它提到前 3 —— 这是 Recall@5 与 faithfulness 的主要来源。
"""

from __future__ import annotations

import logging
import math
import os
import re
import threading
from collections.abc import Sequence
from functools import lru_cache

from .config import get_settings


class Reranker:
    name = "base"

    def score(self, query: str, docs: Sequence[str]) -> list[float]:  # pragma: no cover
        raise NotImplementedError

    def rerank(self, query: str, items: Sequence[tuple[str, str]], top_k: int) -> list[tuple[str, float]]:
        """items = [(id, text)] → [(id, score)]，按分数降序取前 top_k。"""
        if not items:
            return []
        scores = self.score(query, [t for _, t in items])
        ranked = sorted(zip([i for i, _ in items], scores, strict=False), key=lambda x: -x[1])
        return ranked[:top_k]


class CrossEncoderReranker(Reranker):
    name = "bge-reranker-base"

    def __init__(self, model_name: str | None = None, cache_dir: str | None = None) -> None:
        s = get_settings()
        os.environ.setdefault("HF_ENDPOINT", s.hf_endpoint)
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        self._model = TextCrossEncoder(
            model_name or s.rerank_model, cache_dir=str(cache_dir or s.model_cache_dir)
        )
        self.name = model_name or s.rerank_model
        self._lock = threading.Lock()

    def score(self, query: str, docs: Sequence[str]) -> list[float]:
        if not docs:
            return []
        with self._lock:
            return [float(x) for x in self._model.rerank(query, list(docs))]


class LexicalReranker(Reranker):
    """词面重排兜底：BM25-ish 覆盖率打分，映射到与 cross-encoder 可比的对数域。

    分数域刻意贴着 cross-encoder 的 logits 区间，否则拒答阈值没法共用一套标定。
    """

    name = "lexical"

    @staticmethod
    def _grams(text: str) -> set[str]:
        t = re.sub(r"\s+", "", text.lower())
        return {t[i : i + 2] for i in range(max(len(t) - 1, 0))}

    def score(self, query: str, docs: Sequence[str]) -> list[float]:
        q = self._grams(query)
        out = []
        for d in docs:
            g = self._grams(d)
            if not g:
                out.append(-12.0)
                continue
            cover = len(q & g) / max(len(q), 1)
            out.append(round(-12.0 + 12.0 * (cover**0.5), 4))
        _ = math
        return out


@lru_cache
def get_reranker() -> Reranker:
    s = get_settings()
    if s.rerank_backend == "lexical":
        return LexicalReranker()
    try:
        return CrossEncoderReranker(cache_dir=str(s.model_cache_dir))
    except Exception as exc:  # pragma: no cover
        logging.getLogger(__name__).warning("cross-encoder 不可用，降级为词面重排: %s", exc)
        return LexicalReranker()