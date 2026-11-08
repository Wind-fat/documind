"""向量化：本地 BGE（离线、免费）＋ 哈希兜底（测试 / 无模型环境）＋ 远端 OpenAI 兼容 /embeddings。

默认走本地：企业知识库的文档内容不能出门，embedding 也要能私有化；
bge-small-zh-v1.5（512 维）在中文制度/条款检索上够用，CPU 单条 ~8ms。

远端档（`DOCUMIND_EMBED_BACKEND=dashscope|openai`）是给"要真供应商、也要能换空间"的场景：
一次 HTTP 批量调 `/embeddings`（DashScope 的 text-embedding-v3，1024 维，单请求上限 10 条）。
它与本地档共用同一套空间指纹（`model@dim`）：换后端就是换空间，换空间就必须重建索引，
绝不静默降级到 hash。
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

import httpx
import numpy as np

from .config import get_settings

log = logging.getLogger(__name__)


class EmbeddingSpaceMismatch(RuntimeError):
    """库里的向量与当前 embedder 不在同一个向量空间（模型或维度不同）。

    这个异常存在的意义是**不要尽力而为**：不同模型/不同降级档产出的向量之间做余弦相似度
    不会报错，只会给出语义上不可比的排名。宁可让检索失败，也不静默返回错结果（P0-02）。
    """


class EmbeddingBackendError(RuntimeError):
    """远端 embedding 后端不可用。

    **绝不降级到 hash**：hash 是另一个向量空间，换过去不会报错，只会让新查询与旧向量
    算出一堆语义上不可比的余弦相似度。取舍与 `EmbeddingSpaceMismatch` 一致：
    宁可让请求失败并在 /health 里显示 degraded，也不静默换空间。
    """


def _scrub(text: str, secret: str) -> str:
    """错误信息里不许出现 key（供应商 4xx 偶尔回显请求头）。"""
    if secret and secret in text:
        return text.replace(secret, "***")
    return text


def _set_status(backend: str, degraded: bool, reason: str = "") -> None:
    _STATUS.backend, _STATUS.degraded, _STATUS.reason = backend, degraded, reason


# 测试注入点：httpx.MockTransport 从这里进（生产恒为 None = 用默认传输）。
_HTTP_TRANSPORT: httpx.BaseTransport | None = None


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


class OpenAICompatEmbedder(Embedder):
    """远端 OpenAI 兼容 `/embeddings`（DashScope compatible-mode / 自建 vLLM / OpenAI 均可）。

    四条硬约束，每条都被真实事故逼出来过：
    1. `trust_env=False`：本机系统代理会把请求接走，现象是"供应商超时"而不是"代理挂了"；
    2. **批量入参**：一次请求带 `batch_size` 条（DashScope v3 上限 10，超过直接 400
       `batch size is invalid`），一次入库只花几十个请求而不是每块一个；
    3. **维度必须与配置一致**：返回维度不符时报错，不把 `self.dim` 悄悄改成实际维度——
       那等于换了一个向量空间，而库里写着的 `model@dim` 指纹还是旧的；
    4. 失败**不降级**：抛 `EmbeddingBackendError`，同时把状态标进 /health。
    """

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        dim: int = 1024,
        batch_size: int = 10,
        timeout_s: float = 60.0,
        transport: httpx.BaseTransport | None = None,
        backend: str = "openai",
    ) -> None:
        if not base_url or not api_key:
            raise EmbeddingBackendError(
                "远端 embedding 未配置：需要 DOCUMIND_EMBED_API_BASE 与 DOCUMIND_EMBED_API_KEY"
                "（key 只在运行时从环境变量读，不要写进仓库或提交的 .env）"
            )
        if int(dim) <= 0:
            raise EmbeddingBackendError(f"DOCUMIND_EMBED_API_DIM 必须是正整数，当前 {dim!r}")
        self.name = model
        self.dim = int(dim)
        self.batch_size = max(1, int(batch_size))
        self.backend = backend
        self._base = base_url.rstrip("/")
        self._api_key = api_key
        self._client = httpx.Client(trust_env=False, timeout=timeout_s, transport=transport)
        # 运行时计数：入库结束后能直接报"几条文本 / 几个请求 / 多少 embedding token"
        # （token 数取供应商返回的 usage.prompt_tokens，不是字数估算）
        self.requests = 0
        self.prompt_tokens = 0

    def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        payload = {"model": self.name, "input": batch, "dimensions": self.dim}
        try:
            r = self._client.post(
                f"{self._base}/embeddings",
                headers={"Authorization": f"Bearer {self._api_key}"},
                json=payload,
            )
        except Exception as exc:  # 连接/超时/TLS 都归这里
            reason = f"{type(exc).__name__}: {_scrub(str(exc), self._api_key)}"[:300]
            _set_status(self.backend, True, reason)
            raise EmbeddingBackendError(f"embedding 请求失败（不降级）：{reason}") from exc
        if r.status_code >= 400:
            body = _scrub(r.text.strip(), self._api_key)[:300]
            reason = f"HTTP {r.status_code}: {body}"
            _set_status(self.backend, True, reason[:300])
            raise EmbeddingBackendError(f"embedding 上游拒绝请求（不降级）：{reason}")
        self.requests += 1
        try:
            body = r.json()
            self.prompt_tokens += int((body.get("usage") or {}).get("prompt_tokens") or 0)
            data = body.get("data") or []
            # OpenAI 兼容端点不保证顺序；带 index 时按 index 归位，否则按返回顺序
            items = sorted(data, key=lambda d: d.get("index", 0)) if all("index" in d for d in data) else data
            vecs = [d["embedding"] for d in items]
        except (ValueError, KeyError, TypeError, AttributeError) as exc:
            reason = f"响应不是预期的 embeddings 结构：{type(exc).__name__}"
            _set_status(self.backend, True, reason)
            raise EmbeddingBackendError(reason) from exc
        if len(vecs) != len(batch):
            reason = f"请求 {len(batch)} 条、返回 {len(vecs)} 条向量（不降级）"
            _set_status(self.backend, True, reason)
            raise EmbeddingBackendError(reason)
        return vecs

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        rows: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            rows.extend(self._embed_batch(list(texts[i : i + self.batch_size])))
        mat = np.asarray(rows, dtype=np.float32)
        if mat.ndim != 2 or mat.shape != (len(texts), self.dim):
            # 维度不符就是另一个空间：报错，不接受"实际 1024 但我们按 768 存"这种妥协
            got = mat.shape[1] if mat.ndim == 2 else "非二维数组"
            reason = (
                f"向量维度不符：配置 DOCUMIND_EMBED_API_DIM={self.dim}，{self.name} 实际返回 {got}。"
                "改配置后必须重建索引（旧向量属于另一个空间）"
            )
            _set_status(self.backend, True, reason[:300])
            raise EmbeddingBackendError(reason)
        _set_status(self.backend, False, "")  # 成功即恢复：degraded 是"最近一次事实"，不是历史污点
        return self._normalize(mat)


REMOTE_BACKENDS = frozenset({"openai", "dashscope"})


@lru_cache
def get_embedder() -> Embedder:
    """按 `DOCUMIND_EMBED_BACKEND` 选后端。

    降级规则按后端分开，因为它们的代价不同：
    - `auto`（默认，本地 bge）：模型拉不下来才降级 hash——**降级写进 _STATUS**，
      /health 的 `degraded`/`degraded_reason` 与评测报告都能看到（历史行为）；
    - `hash`：显式选择，不算降级；
    - `bge`：用户点名了本地模型，起不来就报错，不偷偷换 hash（换了就是换空间）；
    - `openai` / `dashscope`：远端 /embeddings，配置缺失或请求失败一律抛
      `EmbeddingBackendError`，并把 degraded 标进 /health。**任何情况下都不降级到 hash。**
    """
    s = get_settings()
    backend = (s.embed_backend or "auto").strip().lower()
    if backend == "hash":
        # 显式配置 hash（CI / 离线）不算降级，但 backend 仍然要标出来
        _set_status("hash", False, "")
        return HashEmbedder()
    if backend in REMOTE_BACKENDS:
        try:
            emb = OpenAICompatEmbedder(
                base_url=s.embed_api_base,
                api_key=s.embed_api_key,
                model=s.embed_api_model,
                dim=s.embed_api_dim,
                batch_size=s.embed_api_batch,
                timeout_s=s.embed_api_timeout_s,
                transport=_HTTP_TRANSPORT,
                backend=backend,
            )
        except Exception as exc:
            # 构建失败也要留痕：/health 里 degraded 有值，同时异常照常往上抛
            _set_status(backend, True, f"{type(exc).__name__}: {exc}"[:300])
            log.error("远端 embedding 后端不可用（%s）：%s", backend, exc)
            raise
        _set_status(backend, False, "")
        log.info("embedding 后端：%s（%s@%s）", backend, emb.name, emb.dim)
        return emb
    if backend == "bge":
        emb = BgeEmbedder(cache_dir=str(s.model_cache_dir))
        _set_status("bge", False, "")
        return emb
    try:
        emb = BgeEmbedder(cache_dir=str(s.model_cache_dir))
    except Exception as exc:  # pragma: no cover - 依赖模型下载
        _set_status("hash", True, f"{type(exc).__name__}: {exc}"[:300])
        log.warning("BGE 不可用，降级为 hash embedder: %s", exc)
        return HashEmbedder()
    _set_status("bge", False, "")
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