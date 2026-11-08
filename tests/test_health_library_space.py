"""`/health` 要同时报**进程**的向量空间与**库**里的向量空间（P1-11）。

事故形状（实测踩过）：库里是 `text-embedding-v3@1024`，进程默认 `bge-small@512`。
当时 `/health` 只有 `embed_space`（进程的），运维看到 `source=builtin` 却没有任何线索指向
"库里根本没有这个空间的数据，下一句提问就 409"。这里钉住新增的两个字段与那句警告，
同时钉住"老字段一个不少、语义不变"。
"""

from __future__ import annotations

import httpx
import pytest

from documind.app import app
from documind.db import Chunk, Document

SPACE = "hash-bigram@512"  # conftest 固定 DOCUMIND_EMBED_BACKEND=hash
OTHER = "text-embedding-v3@1024"


async def _health() -> dict:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test", trust_env=False) as c:
        r = await c.get("/health")
    assert r.status_code == 200, r.text
    return r.json()


async def _add_docs(db, space: str, n_docs: int = 1, n_chunks: int = 2, title: str = "制度") -> None:
    for i in range(n_docs):
        d = Document(
            title=f"{title}{i}",
            filename=f"{title}{i}.md",
            sha256="0" * 64,
            page_count=1,
            embed_space=space,
            outline="[]",
        )
        db.add(d)
        await db.flush()
        model = space.split("@")[0] if space else ""
        dim = int(space.split("@")[1]) if space else 0
        for k in range(n_chunks):
            db.add(
                Chunk(
                    doc_id=d.id,
                    strategy="fixed",
                    ordinal=k,
                    text=f"第 {k} 块",
                    tokens=5,
                    embed_model=model,
                    embed_dim=dim,
                )
            )
    await db.commit()


# ------------------------------------------------------- ① 库里的空间 == 进程空间


async def test_health_reports_library_spaces_with_counts(db) -> None:
    """`library_spaces` 是库里的去重空间 + 每个空间的文档/块计数；同空间不算 mismatch。"""
    await _add_docs(db, SPACE, n_docs=2, n_chunks=3, title="本公司")
    await _add_docs(db, OTHER, n_docs=1, n_chunks=4, title="别家")

    body = await _health()
    assert body["embed_space"] == SPACE, "进程空间的字段语义没变"
    assert body["library_spaces"] == [
        {"embed_space": SPACE, "documents": 2, "chunks": 6, "legacy": False},
        {"embed_space": OTHER, "documents": 1, "chunks": 4, "legacy": False},
    ]
    assert body["space_mismatch"] is False
    # 老字段一个不少（语义不变）
    for key in ("status", "documents", "chunks", "embedder", "embed_dim", "embed_space",
                "embedder_backend", "degraded", "degraded_reason", "index_version",
                "index_fingerprint", "llm", "refusal_threshold", "streaming"):
        assert key in body, key
    assert body["documents"] == 3 and body["chunks"] == 10


# ------------------------------------------------------- ② 库里没有进程的空间


async def test_space_mismatch_true_and_warning_says_retrieval_will_409(db) -> None:
    """进程=bge(512)、库里只有 dashscope(1024) → space_mismatch=true，警告直说会 409。"""
    await _add_docs(db, OTHER, n_docs=1, n_chunks=3, title="suhao")

    body = await _health()
    assert body["embed_space"] == SPACE
    assert body["space_mismatch"] is True
    assert body["library_spaces"] == [
        {"embed_space": OTHER, "documents": 1, "chunks": 3, "legacy": False}
    ]

    thr = body["refusal_threshold"]
    assert "库里没有你这个空间的数据" in thr["warning"]
    assert "检索会 409" in thr["warning"]
    assert "先入库或换回库里那个 backend" in thr["warning"]
    assert OTHER in thr["warning"] and SPACE in thr["warning"]
    # 只追加警告，不改 source/value 语义：没标定就还是 builtin 占位
    assert thr["source"] == "builtin" and thr["value"] == pytest.approx(-6.0)
    assert thr["calibrated"] is None


# ------------------------------------------------------- ③ 空库不算 mismatch


async def test_empty_library_is_not_a_mismatch(db) -> None:
    """空库没有"对不上"的数据，409 也不会发生：不许在刚部署时误报。"""
    body = await _health()
    assert body["library_spaces"] == []
    assert body["space_mismatch"] is False
    assert "409" not in body["refusal_threshold"]["warning"]


# ------------------------------------------------------- ④ 迁移前的无空间文档


async def test_legacy_documents_are_listed_separately(db) -> None:
    """`embed_space=''` 的历史文档单列并标 legacy：它们不是"某个后端的数据"。"""
    await _add_docs(db, "", n_docs=1, n_chunks=2, title="旧库")

    body = await _health()
    assert body["library_spaces"] == [
        {"embed_space": "", "documents": 1, "chunks": 2, "legacy": True}
    ]
    assert body["space_mismatch"] is True, "只有无指纹的历史数据时，进程空间同样对不上"
    assert "（无空间标注的历史数据）" in body["refusal_threshold"]["warning"]