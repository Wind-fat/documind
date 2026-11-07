"""P0-02：向量空间指纹 + fail-fast + 降级可观测（health / 评测报告）。"""

from __future__ import annotations

import json
import os
from pathlib import Path

import httpx
import pytest
import sqlalchemy as sa
from sqlalchemy.ext.asyncio import create_async_engine

from documind import embed as embed_mod
from documind.app import app
from documind.db import Chunk, Document, ensure_schema_compat
from documind.embed import EmbeddingSpaceMismatch, HashEmbedder
from documind.pipeline import ingest_document
from documind.retrieve import retrieve, store

ADMIN = {"X-Admin-Token": os.environ.get("DOCUMIND_ADMIN_TOKEN", "dev-admin-token")}


class BgeStandIn(HashEmbedder):
    """同维（512）但不同名字的 embedder。

    它的向量是哈希算出来的——这里测的是**空间指纹守卫**，不是 bge 的语义质量；
    维度相同正是最危险的场景：余弦相似度照算，结果全错。
    """

    name = "BAAI/bge-small-zh-v1.5"


@pytest.fixture()
async def client(db):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


async def test_ingest_stamps_embed_space(db, corpus_md: Path):
    res = await ingest_document(db, corpus_md, acl_roles=["all"])
    doc = (await db.execute(sa.select(Document).where(Document.id == res["document_id"]))).scalar_one()
    assert doc.embed_space == "hash-bigram@512"

    rows = (await db.execute(sa.select(Chunk).where(Chunk.doc_id == doc.id))).scalars().all()
    assert rows
    assert {r.embed_model for r in rows} == {"hash-bigram"}
    assert {r.embed_dim for r in rows} == {512}
    assert all(len(r.embedding or b"") == 512 * 4 for r in rows)


async def test_index_store_fails_fast_on_space_mismatch(db, corpus_md: Path):
    await ingest_document(db, corpus_md, acl_roles=["all"])
    store.invalidate()

    with pytest.raises(EmbeddingSpaceMismatch) as ei:
        await retrieve(db, "年假有几天", role="employee", embedder=BgeStandIn())

    msg = str(ei.value)
    assert "向量空间不一致" in msg
    assert "BAAI/bge-small-zh-v1.5@512" in msg  # 当前 embedder
    assert "hash-bigram@512" in msg  # 库里块的来源

    # 失败不能污染缓存：换回建库时的 embedder 仍然可用
    res = await retrieve(db, "年假有几天", role="employee")
    assert res.hits


async def test_legacy_chunks_without_fingerprint_fail_fast(db, corpus_md: Path):
    """迁移前的历史块没有指纹：不许猜，直接报错并给出修复路径。"""
    await ingest_document(db, corpus_md, acl_roles=["all"])
    await db.execute(sa.update(Chunk).values(embed_model="", embed_dim=0))
    await db.commit()
    store.invalidate()

    with pytest.raises(EmbeddingSpaceMismatch) as ei:
        await retrieve(db, "年假有几天", role="employee")
    assert "缺少向量空间指纹" in str(ei.value)
    assert "backfill_embed_space.py" in str(ei.value)


async def test_health_exposes_degraded_embedder(client, monkeypatch):
    """/health 必须能看见 bge→hash 的降级（否则降级等于隐形）。"""
    settings = embed_mod.get_settings()
    monkeypatch.setattr(settings, "embed_backend", "auto")

    def boom(*args, **kwargs):
        raise RuntimeError("模型下载失败（测试注入）")

    monkeypatch.setattr(embed_mod, "BgeEmbedder", boom)
    embed_mod.get_embedder.cache_clear()
    try:
        body = (await client.get("/health")).json()
        assert body["degraded"] == ["embedder"]
        assert body["embedder_backend"] == "hash"
        assert body["embedder"] == "hash-bigram"
        assert "模型下载失败" in body["degraded_reason"]
        assert body["embed_space"] == "hash-bigram@512"
    finally:
        embed_mod.get_embedder.cache_clear()


async def test_health_without_degradation(client):
    body = (await client.get("/health")).json()
    assert body["degraded"] == []
    assert body["embedder_backend"] == "hash"  # 测试里显式配置 hash，不算降级
    assert body["embed_dim"] == 512


def test_report_records_embedder(tmp_path: Path, monkeypatch):
    from documind.evals import report as report_mod

    monkeypatch.setattr(report_mod, "REPORTS", tmp_path)
    summary = {"structure": {"n_cases": 3, "n_answerable": 2, "n_unanswerable": 1}}
    calibration = {
        "curve": [{"threshold": -6.0, "refusal_accuracy": 1.0, "false_refusal": 0.0}],
        "recommended": {"threshold": -6.0, "refusal_accuracy": 1.0, "false_refusal": 0.0},
        "rule": "误拒率 ≤ 15% 的前提下，取拒答准确率最高的阈值",
    }
    info = {
        "name": "hash-bigram",
        "dim": 512,
        "space": "hash-bigram@512",
        "backend": "hash",
        "degraded": True,
        "degraded_reason": "RuntimeError: 模型下载失败（测试）",
    }
    out = report_mod.write_reports("default", summary, {}, calibration, embedder=info)

    md = Path(out["markdown"]).read_text(encoding="utf-8")
    html_text = Path(out["html"]).read_text(encoding="utf-8")
    data = json.loads(Path(out["json"]).read_text(encoding="utf-8"))

    assert "hash-bigram" in md and "dim 512" in md and "backend hash" in md
    assert "已自动降级" in md
    assert "hash-bigram" in html_text and "dim 512" in html_text
    assert data["embedder"]["backend"] == "hash" and data["embedder"]["dim"] == 512


async def test_legacy_sqlite_database_gets_new_columns(tmp_path: Path):
    """老库平滑补列：不删库、原数据不丢、新列带可用默认值。"""
    db_path = tmp_path / "legacy.db"
    engine = create_async_engine(f"sqlite+aiosqlite:///{db_path.as_posix()}")
    async with engine.begin() as conn:
        await conn.exec_driver_sql(
            "CREATE TABLE documents (id VARCHAR(32) PRIMARY KEY, title VARCHAR(256), "
            "filename VARCHAR(256), sha256 VARCHAR(64), page_count INTEGER, acl_roles TEXT, "
            "source_path VARCHAR(512), created_at DATETIME)"
        )
        await conn.exec_driver_sql(
            "CREATE TABLE chunks (id VARCHAR(32) PRIMARY KEY, doc_id VARCHAR(32), strategy VARCHAR(32), "
            "ordinal INTEGER, is_parent BOOLEAN, parent_id VARCHAR(32), text TEXT, page_start INTEGER, "
            "char_start INTEGER, page_end INTEGER, char_end INTEGER, section_path VARCHAR(512), "
            "tokens INTEGER, embedding BLOB)"
        )
        await conn.exec_driver_sql(
            "INSERT INTO documents (id, title, filename, sha256, page_count, acl_roles, source_path) "
            "VALUES ('d1', '老文档', 'a.md', 'sha', 1, '[\"all\"]', '')"
        )
        await conn.exec_driver_sql(
            "INSERT INTO chunks (id, doc_id, strategy, ordinal, is_parent, text, page_start, char_start, "
            "page_end, char_end, section_path, tokens, embedding) VALUES "
            "('c1', 'd1', 'structure', 0, 0, '老块', 1, 0, 1, 2, '', 2, X'00000000')"
        )

    added = await ensure_schema_compat(engine)
    assert set(added) == {"chunks.embed_model", "chunks.embed_dim", "documents.embed_space"}

    async with engine.connect() as conn:
        chunk = (await conn.exec_driver_sql("SELECT embed_model, embed_dim, text FROM chunks")).one()
        assert chunk[0] == "" and chunk[1] == 0 and chunk[2] == "老块"
        doc = (await conn.exec_driver_sql("SELECT embed_space, title FROM documents")).one()
        assert doc[0] == "" and doc[1] == "老文档"

    # 幂等：第二次启动不重复补列
    assert await ensure_schema_compat(engine) == []
    await engine.dispose()