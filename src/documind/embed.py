"""向量化：本地 BGE（离线、免费）＋ 哈希兜底（测试 / 无模型环境）。

为什么本地而不是调 API：企业知识库的文档内容不能出门，embedding 也要能私有化；
bge-small-zh-v1.5（512 维）在中文制度/条款检索上够用，CPU 单条 ~8ms。
"""

from __future__ import annotations

import hashlib
import os
import re
import threading
from collections.abc import Sequence
from functools import lru_cache

import numpy as np

from .config import get_settings


class Embedder:
    dim: int = 512
    name: str = "base"

    def embed(self, texts: Sequence[str]) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError

    @staticmethod
    def _normalize(mat: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (mat / norms).astype(np.float32)


class BgeEmbedder(Embedder):
    name = "bge-small-zh-v1.5"

    def __init__(self, model_name: str | None = None, cache_dir: str | None = None) -> None:
        s = get_settings()
        os.environ.setdefault("HF_ENDPOINT", s.hf_endpoint)
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
        from fastembed import TextEmbedding

        self._model = TextEmbedding(
            model_name or s.embed_model, cache_dir=str(cache_dir or s.model_cache_dir)
        )
        self.name = model_name or s.embed_model
        self.dim = 512
        self._lock = threading.Lock()

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        with self._lock:
            mat = np.array(list(self._model.embed(list(texts))), dtype=np.float32)
        if mat.shape[1] != self.dim:
            self.dim = mat.shape[1]
        return self._normalize(mat)


class HashEmbedder(Embedder):
    """字符 2-gram 哈希到固定维度。不需要任何依赖，测试与离线 demo 用。

    它当然不如 BGE 准，但**评测报告里会明写用的是哪个 embedder**，
    不自欺欺人是这个项目的第一原则。
    """

    name = "hash-bigram"
    dim = 512

    @staticmethod
    def _tokens(text: str) -> list[str]:
        t = re.sub(r"\s+", "", text.lower())
        grams = [t[i : i + 2] for i in range(max(len(t) - 1, 0))]
        return grams + re.findall(r"[a-z0-9]+", t)

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        mat = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for tok in self._tokens(text):
                h = int.from_bytes(hashlib.blake2b(tok.encode(), digest_size=8).digest(), "big")
                mat[row, h % self.dim] += 1.0
        return self._normalize(mat)


@lru_cache
def get_embedder() -> Embedder:
    """优先本地 BGE；模型拉不下来就降级到哈希，并且降级这件事会被记录。"""
    s = get_settings()
    if s.embed_backend == "hash":
        return HashEmbedder()
    try:
        return BgeEmbedder(cache_dir=str(s.model_cache_dir))
    except Exception as exc:  # pragma: no cover - 依赖模型下载
        import logging

        logging.getLogger(__name__).warning("BGE 不可用，降级为 hash embedder: %s", exc)
        return HashEmbedder()


def cosine_search(query: np.ndarray, matrix: np.ndarray, ids: list[str], top_n: int) -> list[tuple[str, float]]:
    if matrix.size == 0:
        return []
    q = query.astype(np.float32).ravel()
    n = np.linalg.norm(q) or 1.0
    sims = matrix @ (q / n)
    k = min(top_n, len(ids))
    order = np.argpartition(-sims, k - 1)[:k]
    order = order[np.argsort(-sims[order])]
    return [(ids[i], float(sims[i])) for i in order]