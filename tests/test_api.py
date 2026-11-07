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

    monkeypatch.setattr(pipeline, "get_llm", lambda: Fake())
    async with client.stream(
        "GET", "/query/stream", params={"question": "年假有几天"}, headers={"X-User": "alice"}
    ) as resp:
        text = "".join([chunk async for chunk in resp.aiter_text()])
    assert "event: evidence" in text and "event: answer" in text