"""向量化：本地 BGE（离线、免费）＋ 哈希兜底（测试 / 无模型环境）。

为什么本地而不是调 API：企业知识库的文档内容不能出门，embedding 也要能私有化；
bge-small-zh-v1.5（512 维）在中文制度/条款检索上够用，CPU 单条 ~8ms。
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from .config import get_settings

log = logging.getLogger(__name__)


class EmbeddingSpaceMismatch(RuntimeError):
    """库里的向量与当前 embedder 不在同一个向量空间（模型或维度不同）。

    这个异常存在的意义是**不要尽力而为**：不同模型/不同降级档产出的向量之间做余弦相似度
    不会报错，只会给出语义上不可比的排名。宁可让检索失败，也不静默返回错结果（P0-02）。
    """


@dataclass(slots=True)
class EmbedderStatus:
    """当前 embedder 的运行时状态，给 /health 与评测报告用（降级必须可观测）。"""

    backend: str = "unknown"  # bge | hash
    degraded: bool = False  # 是否发生过自动降级（不是显式配置 hash）
    reason: str = ""  # 降级原因（异常类型 + 摘要）

    def as_dict(self) -> dict:
        return {"backend": self.backend, "degraded": self.degraded, "reason": self.reason}


_STATUS = EmbedderStatus()


class Embedder:
    dim: int = 512
    name: str = "base"

    def embed(self, texts: Sequence[str]) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError

    @property
    def space(self) -> str:
        """向量空间标识：`model@dim`。写进 documents.embed_space 与 chunks.embed_model/embed_dim。"""
        return f"{self.name}@{self.dim}"

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

    它当然不如 BGE 准，所以**降级不可静默**：backend/degraded 会进 /health 与评测报告，
    `space` 也写进库里——换回 bge 时会因为空间不一致直接 fail-fast，而不是混着比。
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
    """优先本地 BGE；模型拉不下来就降级到哈希。降级写进 _STATUS（/health 与报告可见）。"""
    s = get_settings()
    if s.embed_backend == "hash":
        # 显式配置 hash（CI / 离线）不算降级，但 backend 仍然要标出来
        _STATUS.backend, _STATUS.degraded, _STATUS.reason = "hash", False, ""
        return HashEmbedder()
    try:
        emb = BgeEmbedder(cache_dir=str(s.model_cache_dir))
    except Exception as exc:  # pragma: no cover - 依赖模型下载
        _STATUS.backend = "hash"
        _STATUS.degraded = True
        _STATUS.reason = f"{type(exc).__name__}: {exc}"[:300]
        log.warning("BGE 不可用，降级为 hash embedder: %s", exc)
        return HashEmbedder()
    _STATUS.backend, _STATUS.degraded, _STATUS.reason = "bge", False, ""
    return emb


def embedder_status() -> EmbedderStatus:
    """确保 embedder 已初始化后返回状态（调用它就会触发一次真实的模型加载尝试）。"""
    get_embedder()
    return _STATUS


def embedder_health() -> dict:
    """给 /health、评测报告、EvalRun.metrics 共用的一份 embedder 事实。"""
    emb = get_embedder()
    st = embedder_status()
    return {
        "name": emb.name,
        "dim": emb.dim,
        "space": emb.space,
        "backend": st.backend,
        "degraded": st.degraded,
        "degraded_reason": st.reason,
    }


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