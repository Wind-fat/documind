"""索引与检索测试：BM25 召回、RRF 融合、以及**权限在检索阶段生效**这条硬规则。"""

from __future__ import annotations

from pathlib import Path

from documind.index import BM25Index, IndexItem, rrf_fuse
from documind.pipeline import ingest_document
from documind.retrieve import allowed_doc_ids, retrieve


def _items() -> list[IndexItem]:
    return [
        IndexItem(id="a", text="员工每年享有带薪年假五天", doc_id="d1", strategy="structure"),
        IndexItem(id="b", text="公司注册资本为人民币一亿元", doc_id="d1", strategy="structure"),
        IndexItem(id="c", text="备用金额度为五十万元由财务总监审批", doc_id="d2", strategy="structure"),
    ]


def test_bm25_finds_keyword():
    idx = BM25Index(_items())
    hits = idx.search("年假有几天", 5)
    assert hits and hits[0][0] == "a"


def test_bm25_acl_filters_before_scoring():
    idx = BM25Index(_items())
    hits = idx.search("备用金额度", 5, allowed={"d1"})
    assert all(hit_id != "c" for hit_id, _ in hits)


def test_rrf_favours_agreement():
    run1 = [("x", 9.0), ("y", 8.0), ("z", 1.0)]
    run2 = [("y", 0.9), ("x", 0.8), ("w", 0.7)]
    fused = rrf_fuse([run1, run2], k=60)
    assert fused[0][0] in {"x", "y"}
    assert fused[-1][0] in {"z", "w"}


async def test_retrieval_respects_document_acl(db, corpus_md: Path, secret_md: Path):
    await ingest_document(db, corpus_md, acl_roles=["all"])
    secret = await ingest_document(db, secret_md, acl_roles=["finance"])

    assert secret["document_id"] not in await allowed_doc_ids(db, "employee")

    # 员工问"备用金额度"，检索阶段就该什么都拿不到
    res = await retrieve(db, "备用金额度是多少", role="employee")
    assert res.hits == [] or all(h.doc_id != secret["document_id"] for h in res.hits)

    # 财务角色问同一句，能拿到
    res_fin = await retrieve(db, "备用金额度是多少", role="finance")
    assert any(h.doc_id == secret["document_id"] for h in res_fin.hits)


async def test_retrieval_returns_spans(db, corpus_md: Path):
    await ingest_document(db, corpus_md, acl_roles=["all"])
    res = await retrieve(db, "年假有几天", role="employee")
    assert res.hits
    top = res.hits[0]
    assert top.page_start >= 1
    assert top.char_end > top.char_start
    assert "年假" in top.text


async def test_parent_child_uses_parent_context(db, corpus_md: Path):
    await ingest_document(db, corpus_md, acl_roles=["all"])
    res = await retrieve(db, "年假有几天", role="employee", strategy="parent_child")
    assert res.hits and res.contexts
    # 父子块：喂模型的上下文比命中的小块更长（父块）
    assert len(res.contexts[0].text) >= len(res.hits[0].text)


async def test_empty_corpus_returns_nothing(db):
    res = await retrieve(db, "年假有几天", role="employee")
    assert res.hits == []
    assert res.debug.get("reason") == "acl_empty" or res.hits == []