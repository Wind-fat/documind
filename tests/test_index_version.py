"""P1-04：索引缓存的失效必须在**库**里，而不是在进程内存里。

旧实现只在 `store.invalidate()` 时清缓存，于是：CLI 入库 → 已在跑的 uvicorn 看不到新文档；
别的进程删掉的块仍被命中。下面每条用例对应一个曾经必然失败的行为。
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest
import sqlalchemy as sa

from documind.app import app
from documind.db import Chunk, Document, bump_index_version
from documind.pipeline import ingest_document
from documind.retrieve import (
    IndexStore,
    read_index_fingerprint,
    retrieve,
    store,
)


@pytest.fixture()
async def client(db):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


async def test_fingerprint_is_db_derived_and_moves_on_ingest(db, corpus_md: Path):
    before = await read_index_fingerprint(db)
    await ingest_document(db, corpus_md, acl_roles=["all"])
    after = await read_index_fingerprint(db)

    assert after.generation == before.generation + 1, "写路径没有 bump 库内版本"
    assert after.documents == before.documents + 1
    assert after.chunks > before.chunks
    assert after.tokens > before.tokens
    assert after != before


async def test_second_store_instance_sees_documents_added_elsewhere(db, corpus_md: Path, secret_md: Path):
    """两个 IndexStore 实例 = 两个进程：缓存不共享，只能靠库里的版本对齐。"""
    first_doc = await ingest_document(db, corpus_md, acl_roles=["all"])

    procs_b = IndexStore()  # 模拟"已经在跑的 uvicorn"
    idx_before = await procs_b.get(db, "structure")
    assert set(idx_before.items) and all(it.doc_id == first_doc["document_id"] for it in idx_before.items.values())

    # 进程 A 入库（旧实现里这里只 invalidate 了 A 自己的缓存）
    second = await ingest_document(db, secret_md, acl_roles=["all"])

    idx_after = await procs_b.get(db, "structure")
    doc_ids = {it.doc_id for it in idx_after.items.values()}
    assert second["document_id"] in doc_ids, "跨实例看不到新入库的文档"
    assert first_doc["document_id"] in doc_ids
    assert idx_after is not idx_before, "版本变了却复用了旧索引对象"


async def test_deleted_chunks_stop_being_hit(db, corpus_md: Path):
    """裸 SQL 删块（不 bump）：行数变化必须同样触发重建，否则已删内容仍被命中。"""
    await ingest_document(db, corpus_md, acl_roles=["all"])
    worker = IndexStore()
    idx = await worker.get(db, "structure")
    assert idx.items

    hits_before = await retrieve(db, "年假有几天", role="employee")
    assert hits_before.hits

    await db.execute(sa.delete(Chunk))
    await db.commit()

    idx_after = await worker.get(db, "structure")
    assert idx_after.items == {}
    hits_after = await retrieve(db, "年假有几天", role="employee")
    assert hits_after.hits == [], "已删的块仍被命中"


async def test_in_place_text_rewrite_is_detected(db, corpus_md: Path):
    """行数不变、文本被原地改写（嵌入重算/人工订正）：tokens 总数兜底。"""
    await ingest_document(db, corpus_md, acl_roles=["all"])
    worker = IndexStore()
    before = await worker.get(db, "structure")

    row = (
        await db.execute(
            sa.select(Chunk).where(Chunk.is_parent.is_(False), Chunk.strategy == "structure").limit(1)
        )
    ).scalar_one()
    row.text = row.text + "（2026 年修订，年假十天起）"
    row.tokens = len(row.text)
    await db.commit()

    after = await worker.get(db, "structure")
    assert after is not before
    assert "2026 年修订" in after.items[row.id].text


async def test_rebuild_only_when_version_changes(db, corpus_md: Path):
    """版本比对不能退化成"每请求都重建"：5 次请求只许建 1 次。"""
    worker = IndexStore()
    for _ in range(5):
        await worker.get(db, "structure")
    assert worker.builds == 1

    await ingest_document(db, corpus_md, acl_roles=["all"])
    for _ in range(3):
        await worker.get(db, "structure")
    assert worker.builds == 2, "版本只变了一次，却重建了多次"

    # 四个策略各建一次索引，但不会因为"多次请求"而叠加
    for strategy in ("fixed", "semantic", "parent_child", "structure"):
        await worker.get(db, strategy)
    builds_after_four = worker.builds
    for _ in range(3):
        for strategy in ("fixed", "semantic", "parent_child", "structure"):
            await worker.get(db, strategy)
    assert worker.builds == builds_after_four


async def test_health_exposes_db_index_version(client, db, corpus_md: Path):
    """/health 的 index_version 必须是**库里的**版本：别的进程入库后它才会变。"""
    before = (await client.get("/health")).json()
    assert isinstance(before["index_version"], int)

    await ingest_document(db, corpus_md, acl_roles=["all"])

    after = (await client.get("/health")).json()
    assert after["index_version"] == before["index_version"] + 1
    assert after["index_fingerprint"]["documents"] == before["index_fingerprint"]["documents"] + 1
    assert after["index_fingerprint"]["chunks"] > 0
    # 降级观测没有被这次改动挤掉（P0-02 的既有契约）
    assert "degraded" in after and "refusal_threshold" in after


async def test_bump_is_idempotent_within_one_transaction(db):
    v1 = await bump_index_version(db)
    v2 = await bump_index_version(db)
    await db.commit()
    assert v2 == v1 + 1
    assert (await read_index_fingerprint(db)).generation >= v2


async def test_hits_only_reference_live_documents(db, corpus_md: Path, secret_md: Path):
    """重建后每个命中都必须指向库里还活着的文档（缓存不得残留上一轮的 doc_id）。"""
    await ingest_document(db, corpus_md, acl_roles=["all"])
    await retrieve(db, "年假有几天", role="employee", rerank=False)
    await ingest_document(db, secret_md, acl_roles=["all"])

    res = await retrieve(db, "备用金额度", role="employee", rerank=False)
    alive = set((await db.execute(sa.select(Document.id))).scalars().all())
    assert res.hits
    assert all(h.doc_id in alive for h in res.hits)
    assert store.version == (await read_index_fingerprint(db)).generation