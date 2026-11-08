"""远端 OpenAI 兼容 `/embeddings` 后端：批量、空间指纹、失败不降级、降级可观测。

这一组用例全部用 `httpx.MockTransport` 打桩，不出网：钉的是**契约**（批量入参、`model@dim`
指纹、维度不符就报错、失败不换空间），不是供应商的语义质量。
"""

from __future__ import annotations

import json
import math
import os

import httpx
import numpy as np
import pytest
import sqlalchemy as sa

from documind import embed as embed_mod
from documind.app import app
from documind.config import get_settings
from documind.db import Chunk, Document
from documind.embed import (
    EmbeddingBackendError,
    EmbeddingSpaceMismatch,
    HashEmbedder,
    OpenAICompatEmbedder,
)
from documind.pipeline import ingest_document
from documind.retrieve import retrieve, store

API_BASE = "https://embedding.example.invalid/compatible-mode/v1"
API_KEY = "sk-test-not-a-real-key"


class Recorder:
    """记下每个请求体，按 OpenAI 兼容格式造响应。"""

    def __init__(self, dim: int = 1024, status: int = 200, short: bool = False) -> None:
        self.dim = dim
        self.status = status
        self.short = short  # True = 少返回一条向量（上游截断的典型表现）
        self.calls: list[dict] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.calls.append({"url": str(request.url), "headers": dict(request.headers), **body})
        if self.status >= 400:
            return httpx.Response(self.status, text='{"error":{"message":"boom"}}')
        n = len(body["input"]) - (1 if self.short else 0)
        data = [{"index": i, "embedding": self._vec(t)} for i, t in enumerate(body["input"][:n])]
        return httpx.Response(
            200, json={"object": "list", "model": body["model"], "data": data, "usage": {"prompt_tokens": n}}
        )

    def _vec(self, text: str) -> list[float]:
        seed = int.from_bytes(text.encode("utf-8")[:8].ljust(8, b"\0"), "big") % (2**32)
        v = np.random.default_rng(seed).random(self.dim).astype(np.float32) + 0.01
        return (v / np.linalg.norm(v)).tolist()


@pytest.fixture(autouse=True)
def _clean_embedder_state():
    """embedder 是进程级单例、状态是模块级：每个用例前后都清干净，避免互相污染。"""
    embed_mod.get_embedder.cache_clear()
    embed_mod._HTTP_TRANSPORT = None
    yield
    embed_mod._HTTP_TRANSPORT = None
    embed_mod.get_embedder.cache_clear()
    # 测试环境显式配置的是 hash（conftest），把状态恢复到那个事实
    embed_mod._STATUS.backend, embed_mod._STATUS.degraded, embed_mod._STATUS.reason = "hash", False, ""


@pytest.fixture()
def api_embedder(monkeypatch):
    """把 settings 指到远端后端 + 装上 MockTransport；返回请求记录器。"""
    rec = Recorder()
    monkeypatch.setattr(embed_mod, "_HTTP_TRANSPORT", httpx.MockTransport(rec))
    s = get_settings()
    monkeypatch.setattr(s, "embed_backend", "dashscope")
    monkeypatch.setattr(s, "embed_api_base", API_BASE)
    monkeypatch.setattr(s, "embed_api_key", API_KEY)
    monkeypatch.setattr(s, "embed_api_model", "text-embedding-v3")
    monkeypatch.setattr(s, "embed_api_dim", 1024)
    monkeypatch.setattr(s, "embed_api_batch", 10)
    embed_mod.get_embedder.cache_clear()
    return rec


@pytest.fixture()
async def client():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


def _make(dim: int = 1024, batch: int = 10, rec: Recorder | None = None) -> OpenAICompatEmbedder:
    recorder = rec or Recorder(dim=dim)
    return OpenAICompatEmbedder(
        base_url=API_BASE,
        api_key=API_KEY,
        model="text-embedding-v3",
        dim=dim,
        batch_size=batch,
        transport=httpx.MockTransport(recorder),
    )


# ---------------------------------------------------------------- 批量 + 指纹


def test_batches_inputs_into_single_requests_and_stamps_space():
    rec = Recorder()
    emb = _make(rec=rec)
    texts = [f"第{i}条 测试条款" for i in range(25)]

    mat = emb.embed(texts)

    # 25 条 / 每批 10 条 = 3 个请求（10 + 10 + 5），不是 25 个
    assert [len(c["input"]) for c in rec.calls] == [10, 10, 5]
    assert all(c["model"] == "text-embedding-v3" for c in rec.calls)
    assert all(c["dimensions"] == 1024 for c in rec.calls)
    assert mat.shape == (25, 1024)
    # 计数来自供应商返回的 usage，不是字数估算
    assert (emb.requests, emb.prompt_tokens) == (3, 25)
    assert emb.space == "text-embedding-v3@1024"
    assert math.isclose(float(np.linalg.norm(mat[0])), 1.0, rel_tol=1e-5)


def test_key_goes_in_header_not_in_logs(caplog):
    """key 只进 Authorization 头；失败信息里也不能出现它。"""
    rec = Recorder(status=401)
    emb = _make(rec=rec)
    with caplog.at_level("DEBUG"):
        with pytest.raises(EmbeddingBackendError) as ei:
            emb.embed(["x"])
    assert rec.calls[0]["headers"]["authorization"] == f"Bearer {API_KEY}"
    assert API_KEY not in str(ei.value)
    assert API_KEY not in caplog.text


# ---------------------------------------------------------------- 失败不降级


def test_dim_mismatch_raises_instead_of_resizing():
    """返回 512 维但配置 1024：报错，且不把 self.dim 改成 512（那等于悄悄换空间）。"""
    rec = Recorder(dim=512)
    emb = _make(dim=1024, rec=rec)

    with pytest.raises(EmbeddingBackendError) as ei:
        emb.embed(["一", "二"])

    assert "向量维度不符" in str(ei.value)
    assert "1024" in str(ei.value) and "512" in str(ei.value)
    assert emb.dim == 1024
    assert emb.space == "text-embedding-v3@1024"


def test_truncated_response_raises():
    rec = Recorder(short=True)
    emb = _make(rec=rec)
    with pytest.raises(EmbeddingBackendError) as ei:
        emb.embed(["一", "二", "三"])
    assert "返回 2 条向量" in str(ei.value)


def test_missing_config_raises_and_is_not_hash(monkeypatch):
    """没配 base/key 时是错误，不是 hash：换空间比报错更糟。"""
    s = get_settings()
    monkeypatch.setattr(s, "embed_backend", "dashscope")
    monkeypatch.setattr(s, "embed_api_base", "")
    monkeypatch.setattr(s, "embed_api_key", "")
    embed_mod.get_embedder.cache_clear()

    with pytest.raises(EmbeddingBackendError) as ei:
        embed_mod.get_embedder()

    assert "DOCUMIND_EMBED_API_KEY" in str(ei.value)
    assert embed_mod._STATUS.degraded is True
    assert embed_mod._STATUS.backend == "dashscope"


def test_unknown_backend_name_is_rejected(monkeypatch):
    """`dashscop` 这种拼写错误不许静默变成 bge。"""
    from documind.config import Settings

    monkeypatch.setenv("DOCUMIND_EMBED_BACKEND", "dashscop")
    with pytest.raises(Exception) as ei:
        Settings()
    assert "embed_backend" in str(ei.value)


# ---------------------------------------------------------------- 可观测（/health）


async def test_health_shows_degraded_after_backend_failure(api_embedder, client):
    """失败 → 抛异常 + /health.degraded 有值；空间仍是远端模型，没变成 hash。"""
    api_embedder.status = 500
    emb = embed_mod.get_embedder()
    assert emb.space == "text-embedding-v3@1024"  # 失败也没换空间

    with pytest.raises(EmbeddingBackendError):
        emb.embed(["一"])

    body = (await client.get("/health")).json()
    assert body["degraded"] == ["embedder"]
    assert body["embedder_backend"] == "dashscope"
    assert body["embedder"] == "text-embedding-v3"
    assert body["embed_space"] == "text-embedding-v3@1024"
    assert "HTTP 500" in body["degraded_reason"]


async def test_health_is_clean_when_backend_works(api_embedder, client):
    embed_mod.get_embedder().embed(["一"])
    body = (await client.get("/health")).json()
    assert body["degraded"] == []
    assert body["embedder_backend"] == "dashscope"
    assert body["embed_space"] == "text-embedding-v3@1024"


# ---------------------------------------------------------------- 空间指纹 + 409


async def test_ingest_with_api_embedder_then_hash_query_is_409(db, corpus_md, api_embedder, client):
    """用远端后端入库 → 换回本地 hash 查询：必须 409，而不是拿两个空间的向量比余弦。"""
    res = await ingest_document(db, corpus_md, acl_roles=["all"])
    assert res["embed_space"] == "text-embedding-v3@1024"
    doc = (await db.execute(sa.select(Document).where(Document.id == res["document_id"]))).scalar_one()
    assert doc.embed_space == "text-embedding-v3@1024"
    rows = (await db.execute(sa.select(Chunk).where(Chunk.doc_id == doc.id))).scalars().all()
    assert {r.embed_model for r in rows} == {"text-embedding-v3"}
    assert {r.embed_dim for r in rows} == {1024}
    assert all(len(r.embedding or b"") == 1024 * 4 for r in rows)
    assert len(api_embedder.calls) >= 1

    # 换回建库前的 embedder（测试默认 hash）——这一换就是换空间
    s = get_settings()
    s.embed_backend = "hash"
    embed_mod.get_embedder.cache_clear()
    store.invalidate()

    r = await client.post("/query", json={"question": "年假有几天"}, headers={"X-User": "alice"})
    assert r.status_code == 409
    detail = r.json()["detail"]
    assert detail["code"] == "embedding_space_mismatch"
    assert "text-embedding-v3@1024" in detail["message"]
    assert "hash-bigram@512" in detail["message"]


async def test_retrieve_raises_space_mismatch_without_http(db, corpus_md, api_embedder):
    await ingest_document(db, corpus_md, acl_roles=["all"])
    store.invalidate()

    with pytest.raises(EmbeddingSpaceMismatch) as ei:
        await retrieve(db, "年假有几天", role="employee", embedder=HashEmbedder())
    assert "向量空间不一致" in str(ei.value)
    assert "text-embedding-v3@1024" in str(ei.value)


# ---------------------------------------------------------------- 与旧机制一致


def test_space_string_matches_existing_convention():
    """`model@dim` 与 documents.embed_space / chunks.embed_model+embed_dim 的口径一致。"""
    emb = _make()
    assert emb.space == f"{emb.name}@{emb.dim}" == "text-embedding-v3@1024"
    assert os.environ.get("DOCUMIND_EMBED_BACKEND", "hash") == "hash"  # conftest 的显式配置