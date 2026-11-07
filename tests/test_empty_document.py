"""P0-04：空文本必须显式失败（ADR-0001 承诺的护栏），不能 200 建出空知识库。"""

from __future__ import annotations

import os
from pathlib import Path

import httpx
import pytest
import sqlalchemy as sa

from documind.app import app
from documind.db import Document
from documind.parsing import EmptyDocumentError, ParsedDocument, parse_file

ADMIN = {"X-Admin-Token": os.environ.get("DOCUMIND_ADMIN_TOKEN", "dev-admin-token")}


@pytest.fixture()
async def client(db):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


def test_empty_markdown_raises(tmp_path: Path):
    p = tmp_path / "空文件.md"
    p.write_text("   \n\n", encoding="utf-8")
    with pytest.raises(EmptyDocumentError) as ei:
        parse_file(p)
    assert "未提取到文本，需要 OCR" in str(ei.value)


def test_pdf_without_text_layer_raises(tmp_path: Path):
    """无文本层页：真实 PDF，页里只有一张空白画布，extract_text() 返回空串。"""
    from pypdf import PdfWriter

    p = tmp_path / "扫描件.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=595, height=842)
    with p.open("wb") as f:
        writer.write(f)

    with pytest.raises(EmptyDocumentError) as ei:
        parse_file(p)
    assert "未提取到文本，需要 OCR" in str(ei.value)
    assert "1 页" in str(ei.value)


def test_parsed_document_empty_flags():
    doc = ParsedDocument(title="t", filename="x.md", pages=["", " "], paragraphs=[])
    assert doc.is_empty is True
    with pytest.raises(EmptyDocumentError):
        doc.require_text()


async def test_ingest_rejects_custom_parser_empty_output(db, tmp_path: Path, monkeypatch):
    """自定义 Parser 绕过 parse_file 的守卫时，ingest_document 的二次断言必须接住。"""
    from documind import pipeline

    p = tmp_path / "绕过.md"
    p.write_text("有内容", encoding="utf-8")
    monkeypatch.setattr(
        pipeline,
        "parse_file",
        lambda _p: ParsedDocument(title="空", filename="绕过.md", pages=[""], paragraphs=[]),
    )

    with pytest.raises(EmptyDocumentError):
        await pipeline.ingest_document(db, p)

    count = (await db.execute(sa.select(sa.func.count(Document.id)))).scalar_one()
    assert count == 0, "空文档不能在库里留下 Document 行"


async def test_upload_empty_markdown_returns_422(client, tmp_path: Path):
    p = tmp_path / "空文件.md"
    p.write_text("", encoding="utf-8")
    with p.open("rb") as f:
        r = await client.post("/documents", files={"file": ("空文件.md", f, "text/markdown")}, headers=ADMIN)

    assert r.status_code == 422, r.text
    body = r.json()
    assert body["detail"]["code"] == "empty_document"
    assert "未提取到文本，需要 OCR" in body["detail"]["message"]

    # 不能是"200 + 空库"：文档列表里不该多出一条
    docs = (await client.get("/documents", headers={"X-User": "admin"})).json()
    assert all(d["filename"] != "空文件.md" for d in docs)


async def test_upload_blank_pdf_returns_422(client, tmp_path: Path):
    from pypdf import PdfWriter

    p = tmp_path / "扫描件.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=595, height=842)
    with p.open("wb") as f:
        writer.write(f)

    with p.open("rb") as f:
        r = await client.post("/documents", files={"file": ("扫描件.pdf", f, "application/pdf")}, headers=ADMIN)

    assert r.status_code == 422, r.text
    assert r.json()["detail"]["code"] == "empty_document"