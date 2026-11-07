"""API 层测试：鉴权、ACL 不泄露、证据页取回、拒答字段。"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from documind import pipeline
from documind.app import app
from documind.pipeline import ingest_document

ADMIN = {"X-Admin-Token": "dev-admin-token"}


@pytest.fixture()
async def client(db):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.fixture()
async def loaded(db, corpus_md: Path, secret_md: Path):
    from documind.db import upsert_user

    await upsert_user(db, "alice", "employee")
    await upsert_user(db, "bob", "finance")
    await ingest_document(db, corpus_md, acl_roles=["all"], title="员工制度")
    await ingest_document(db, secret_md, acl_roles=["finance"], title="财务机密制度")
    return db


async def test_health(client):
    r = await client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


async def test_upload_requires_admin(client, corpus_md: Path):
    with corpus_md.open("rb") as f:
        r = await client.post("/documents", files={"file": ("a.md", f, "text/markdown")})
    assert r.status_code == 401


async def test_upload_and_list(client, corpus_md: Path):
    with corpus_md.open("rb") as f:
        r = await client.post(
            "/documents",
            files={"file": ("a.md", f, "text/markdown")},
            data={"acl_roles": json.dumps(["all"], ensure_ascii=False)},
            headers=ADMIN,
        )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["pages"] >= 1 and body["strategy_stats"]["structure"]["chunks"] > 0

    r2 = await client.get("/documents", headers={"X-User": "alice"})
    assert any(d["title"] == body["title"] for d in r2.json())


async def test_hidden_document_is_404_not_403(client, loaded):
    docs = (await client.get("/documents", headers={"X-User": "bob"})).json()
    secret = next(d for d in docs if d["title"] == "财务机密制度")
    # 员工身份取财务文档的页 → 404（不泄露存在性）
    r = await client.get(f"/documents/{secret['id']}/pages/1", headers={"X-User": "alice"})
    assert r.status_code == 404
    # 财务身份可以取
    r2 = await client.get(f"/documents/{secret['id']}/pages/1", headers={"X-User": "bob"})
    assert r2.status_code == 200 and "备用金" in r2.json()["text"]


async def test_query_refuses_without_llm_and_keeps_acl(client, loaded, monkeypatch):
    from documind.llm import LLMReply, Usage

    class Fake:
        async def chat(self, messages, **kwargs):
            return LLMReply(
                text=json.dumps({"answer": "文档中未找到相关条款", "citations": [], "not_found": True}),
                usage=Usage(model="fake"),
            )

    monkeypatch.setattr(pipeline, "get_llm", lambda: Fake())
    r = await client.post(
        "/query",
        json={"question": "备用金额度是多少", "strategy": "structure"},
        headers={"X-User": "alice"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["refused"] is True
    # 员工身份下，绝对不该出现财务文档的命中
    hits = body["debug"]["hits"]
    titles = {h["doc_title"] for h in hits}
    assert "财务机密制度" not in titles


async def test_query_answer_shape(client, loaded, monkeypatch):
    from documind.llm import LLMReply, Usage

    class Fake:
        async def chat(self, messages, **kwargs):
            return LLMReply(
                text=json.dumps(
                    {
                        "answer": "员工每年享有带薪年假五天 [1]",
                        "citations": [{"ref": 1, "quote": "员工每年享有带薪年假五天"}],
                        "not_found": False,
                    }
                ),
                usage=Usage(model="fake", prompt_tokens=100, completion_tokens=20, cost_usd=0.00001),
            )

    monkeypatch.setattr(pipeline, "get_llm", lambda: Fake())
    r = await client.post("/query", json={"question": "年假有几天"}, headers={"X-User": "alice"})
    body = r.json()
    assert body["refused"] is False
    assert body["citations"] and body["citations"][0]["char_end"] > body["citations"][0]["char_start"]
    assert body["latency_ms"] >= 0


async def test_stream_events(client, loaded, monkeypatch):
    from documind.llm import LLMReply, Usage

    class Fake:
        async def chat(self, messages, **kwargs):
            return LLMReply(text=json.dumps({"answer": "五天 [1]", "citations": [{"ref": 1, "quote": "员工每年享有带薪年假五天"}]}), usage=Usage(model="fake"))

        async def chat_stream(self, messages, *, json_mode=False, max_tokens=None, meta=None):
            # 单块流：只验事件名与载荷契约（真流式的时序在 test_stream_and_threshold.py）
            yield json.dumps(
                {
                    "answer": "五天 [1]",
                    "citations": [{"ref": 1, "quote": "员工每年享有带薪年假五天"}],
                    "not_found": False,
                },
                ensure_ascii=False,
            )

    monkeypatch.setattr(pipeline, "get_llm", lambda: Fake())
    async with client.stream(
        "GET", "/query/stream", params={"question": "年假有几天", "selfcheck": "false"}, headers={"X-User": "alice"}
    ) as resp:
        text = "".join([chunk async for chunk in resp.aiter_text()])
    assert "event: evidence" in text and "event: answer" in text
    # 向后兼容：evidence 仍是 Hit 数组，answer 仍是完整 Answer
    assert '"chunk_id"' in text and '"refused"' in text


async def test_outline_is_non_empty_for_non_structure_ingest(client, db, corpus_md: Path):
    """P1-08 验收：只用 fixed 策略入库（库里根本没有 structure 块），章节树不能为空。

    改前这里必然返回 `sections: []`——旧实现从 `chunk.strategy == "structure"` 反推。
    """
    import sqlalchemy as sa

    from documind.db import Chunk

    res = await ingest_document(db, corpus_md, acl_roles=["all"], title="非结构入库", strategies=("fixed",))
    doc_id = res["document_id"]
    assert "structure" not in res["strategy_stats"]
    n_structure = (
        await db.execute(
            sa.select(sa.func.count(Chunk.id)).where(Chunk.doc_id == doc_id, Chunk.strategy == "structure")
        )
    ).scalar_one()
    assert n_structure == 0

    r = await client.get(f"/documents/{doc_id}/outline", headers={"X-User": "alice"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["outline_source"] == "parsed"
    assert [s["title"] for s in body["sections"]] == [
        "员工制度汇编",
        "第一章 总则",
        "第二章 休假",
        "第三章 财务",
    ]
    assert all(s["page"] >= 1 for s in body["sections"])
    # 层级也随响应给出（老响应只有 `章节路径 → 页码` 的平铺串）
    assert body["tree"][1]["level"] == 1 and body["tree"][1]["page"] >= 1


async def test_outline_reports_missing_for_legacy_rows(client, db):
    """历史文档（outline 列为后加的，值为空串）显式标 missing，不偷偷退回"从块反推"。"""
    from documind.db import Document

    d = Document(title="迁移前的老文档", filename="old.pdf", sha256="0" * 64, outline="")
    db.add(d)
    await db.commit()

    r = await client.get(f"/documents/{d.id}/outline", headers={"X-User": "alice"})
    assert r.status_code == 200, r.text
    assert r.json() == {"doc_id": d.id, "sections": [], "tree": [], "outline_source": "missing"}


async def test_reuploading_a_legacy_document_backfills_its_outline(client, db, corpus_md: Path):
    """迁移前入库的文档（outline 为空串）重新上传一次就能拿到章节树，且不重复建块。

    这是 P1-08 留的升级路径：`documents.outline` 是后加的列，历史行无从回填；
    重新上传不会重算向量（sha256 相同 → 走 duplicate 分支），只补这一列。
    """
    import sqlalchemy as sa

    from documind.db import Chunk, Document

    first = await ingest_document(db, corpus_md, acl_roles=["all"], title="老文档")
    doc = (
        await db.execute(sa.select(Document).where(Document.id == first["document_id"]))
    ).scalar_one()
    n_chunks = (
        await db.execute(sa.select(sa.func.count(Chunk.id)).where(Chunk.doc_id == doc.id))
    ).scalar_one()
    # 模拟"迁移前入库的行走势"：列存在但值是空串
    doc.outline = ""
    await db.commit()

    r = await client.get(f"/documents/{doc.id}/outline", headers={"X-User": "alice"})
    assert r.json()["outline_source"] == "missing" and r.json()["sections"] == []

    again = await ingest_document(db, corpus_md, acl_roles=["all"], title="老文档")
    assert again["duplicate"] is True and again["outline_backfilled"] is True
    # 块没有重建（重复入库不重新切块/向量化）
    assert (
        await db.execute(sa.select(sa.func.count(Chunk.id)).where(Chunk.doc_id == doc.id))
    ).scalar_one() == n_chunks

    r2 = await client.get(f"/documents/{doc.id}/outline", headers={"X-User": "alice"})
    body = r2.json()
    assert body["outline_source"] == "parsed"
    assert [s["title"] for s in body["sections"]][:2] == ["员工制度汇编", "第一章 总则"]

    # 再传一次：已经有章节树，不重复补
    third = await ingest_document(db, corpus_md, acl_roles=["all"], title="老文档")
    assert third["duplicate"] is True and third["outline_backfilled"] is False
