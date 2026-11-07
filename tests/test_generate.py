"""生成层测试：引用校验、quote 防伪造、拒答判定、阈值。

这些是"不做评测也必须成立"的安全网——它们比模型质量更该被测试。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import sqlalchemy as sa

from documind.db import Page
from documind.generate import answer_question, locate_quote
from documind.pipeline import ingest_document
from documind.retrieve import Hit, RetrievalResult, retrieve


@pytest.fixture()
async def ready(db, corpus_md: Path):
    await ingest_document(db, corpus_md, acl_roles=["all"])
    return db


def _retrieval_from(doc_text_holder):  # pragma: no cover - helper
    return doc_text_holder


async def test_citation_accepted_and_located(ready, fake_llm):
    db = ready
    res = await retrieve(db, "年假有几天", role="employee")
    quote = "员工每年享有带薪年假五天"
    llm = fake_llm([json.dumps({"answer": "五天 [1]", "citations": [{"ref": 1, "quote": quote}], "not_found": False})])
    ans = await answer_question("年假有几天", res, llm, page_texts={}, selfcheck=False)
    assert not ans.refused
    assert ans.citations and ans.citations[0].quote == quote
    assert ans.citations[0].char_end > ans.citations[0].char_start


async def test_fabricated_quote_is_rejected(ready, fake_llm):
    """模型编了一段原文 → 引用作废 → 没有有效引用 → 必须拒答。"""
    db = ready
    res = await retrieve(db, "年假有几天", role="employee")
    llm = fake_llm(
        [json.dumps({"answer": "可以折现 [1]", "citations": [{"ref": 1, "quote": "年假可以折现为现金"}], "not_found": False})]
    )
    ans = await answer_question("年假可以折现吗", res, llm, page_texts={}, selfcheck=False)
    assert ans.refused
    assert "引用" in ans.reason or "不足" in ans.reason


async def test_not_found_is_refusal(ready, fake_llm):
    db = ready
    res = await retrieve(db, "年假有几天", role="employee")
    llm = fake_llm([json.dumps({"answer": "文档中未找到相关条款", "citations": [], "not_found": True})])
    ans = await answer_question("年假可以折现吗", res, llm, page_texts={}, selfcheck=False)
    assert ans.refused


async def test_threshold_refuses_without_calling_llm(ready, fake_llm):
    """最高分低于阈值时不该花钱调模型。"""
    db = ready
    res = await retrieve(db, "年假有几天", role="employee")
    low = RetrievalResult(hits=res.hits[:1], contexts=res.contexts[:1], top_score=-99.0, latency_ms=1)
    llm = fake_llm([])
    ans = await answer_question("年假有几天", low, llm, page_texts={}, selfcheck=False)
    assert ans.refused and "阈值" in ans.reason
    assert llm.calls == []


async def test_empty_retrieval_refuses_and_says_why(db):
    empty = RetrievalResult(hits=[], contexts=[], top_score=-99.0)
    from documind.llm import LLMClient

    ans = await answer_question("年假可以折现吗", empty, LLMClient(), page_texts={}, selfcheck=False)
    assert ans.refused and "未找到" in ans.text


async def test_selfcheck_retry_path(ready, fake_llm):
    """自检判 false → 重试一次；重试结果有有效引用则采纳。"""
    db = ready
    res = await retrieve(db, "年假有几天", role="employee")
    first = json.dumps({"answer": "五天 [1]", "citations": [{"ref": 1, "quote": "员工每年享有带薪年假五天"}], "not_found": False})
    check = json.dumps({"supported": False, "unsupported_claims": ["隐含可以累积到明年"]})
    second = json.dumps({"answer": "五天 [1]", "citations": [{"ref": 1, "quote": "工作满十年者享有十天"}], "not_found": False})
    llm = fake_llm([first, check, second])
    ans = await answer_question("年假有几天", res, llm, page_texts={}, selfcheck=True)
    assert not ans.refused
    assert ans.selfcheck.get("retried") is True


async def test_selfcheck_unparsed_is_unknown_not_pass(ready, fake_llm):
    """审计 P0-5：自检返回非 JSON 时必须落到"未完成"（supported=None），不许当成通过。

    钉住的是"不知道 ≠ 没问题"：曾经 except 里写 `{"supported": True}`，
    前端据此渲染绿色"自检通过"，而那次检查根本没给出结论。
    """
    db = ready
    res = await retrieve(db, "年假有几天", role="employee")
    answer = json.dumps(
        {"answer": "五天 [1]", "citations": [{"ref": 1, "quote": "员工每年享有带薪年假五天"}], "not_found": False}
    )
    llm = fake_llm([answer, "OK，看起来没问题"])  # 自检返回自由文本，不是 JSON
    ans = await answer_question("年假有几天", res, llm, page_texts={}, selfcheck=True)

    assert ans.selfcheck.get("supported") is None, f"自检没解析出来却被判成 {ans.selfcheck.get('supported')!r}"
    assert ans.selfcheck.get("unparsed") is True
    assert "JSON" in str(ans.selfcheck.get("error"))
    assert not ans.refused, "自检没跑成不该把答案判成拒答（那是掩盖问题）"
    assert len(llm.calls) == 2, "解析失败不重试：重试一次也拿不到判定，只会掩盖'没跑成'"


async def test_citation_ref_out_of_range(ready, fake_llm):
    db = ready
    res = await retrieve(db, "年假有几天", role="employee")
    llm = fake_llm([json.dumps({"answer": "五天 [9]", "citations": [{"ref": 9, "quote": "员工每年享有带薪年假五天"}]})])
    ans = await answer_question("年假有几天", res, llm, page_texts={}, selfcheck=False)
    assert ans.refused


QUOTE = "员工每年享有带薪年假五天"


def _hit(page_start: int, page_end: int, text: str, char_start: int = 4, char_end: int | None = None) -> Hit:
    return Hit(
        chunk_id="c1",
        doc_id="d1",
        doc_title="员工制度",
        text=text,
        page_start=page_start,
        char_start=char_start,
        page_end=page_end,
        char_end=len(text) if char_end is None else char_end,
        section_path="第二章 休假",
        score=9.0,
    )


def test_locate_quote_is_anchored_to_the_cited_page():
    """Q3：同一句话在第 1 页目录也出现，但块声明在第 2 页 → 必须返回第 2 页。"""
    texts = {
        ("d1", 1): f"目录 {QUOTE} …………… 3",
        ("d1", 2): f"第三条 {QUOTE}，工作满十年者享有十天。",
    }
    hit = _hit(2, 2, texts[("d1", 2)])
    span = locate_quote(hit, QUOTE, texts)
    assert span is not None
    assert span.page == 2, f"被第 1 页的同文片段带跑了：{span}"
    assert span.source == "anchored"
    assert texts[("d1", 2)][span.char_start : span.char_end] == QUOTE


async def test_fallback_to_other_page_is_marked(ready, fake_llm):
    """被引页真的没有整句 → 允许退页，但必须标 span_source=fallback_page。"""
    texts = {
        ("d1", 1): f"第三条 {QUOTE}。",
        ("d1", 2): "第四条 员工请病假需提供二级以上医院出具的诊断证明。",
    }
    # 模拟"块文本对、页标错"：整句在块里（能过逐字校验），但第 2 页原文没有
    hit = _hit(2, 2, f"第三条 {QUOTE}。")
    assert locate_quote(hit, QUOTE, texts).source == "fallback_page"

    res = RetrievalResult(hits=[hit], contexts=[hit], top_score=9.0)
    llm = fake_llm([json.dumps({"answer": "五天 [1]", "citations": [{"ref": 1, "quote": QUOTE}]})])
    ans = await answer_question("年假有几天", res, llm, page_texts=texts, selfcheck=False, threshold=-99)
    assert not ans.refused
    cite = ans.citations[0]
    assert (cite.page_start, cite.page_end) == (1, 1)
    assert cite.span_source == "fallback_page"
    assert ans.debug["span_sources"] == {"fallback_page": 1}


async def test_cross_page_chunk_keeps_both_pages(ready, fake_llm):
    """Q3：跨页块定位失败时不得把 page_end 塌成 page_start（[100,30) 倒区间）。"""
    texts = {
        ("d1", 1): "甲" * 100 + "跨页块从第一页这里开始",
        ("d1", 2): "到第二页这里结束" + "乙" * 200,
    }
    hit = _hit(1, 2, texts[("d1", 1)][100:] + texts[("d1", 2)][:30], char_start=100, char_end=30)
    assert locate_quote(hit, "这句两页里都没有", texts) is None

    res = RetrievalResult(hits=[hit], contexts=[hit], top_score=9.0)
    quote = hit.text[:20]
    llm = fake_llm([json.dumps({"answer": "跨页 [1]", "citations": [{"ref": 1, "quote": quote}]})])
    ans = await answer_question("跨页", res, llm, page_texts=texts, selfcheck=False, threshold=-99)
    cite = ans.citations[0]
    assert cite.span_source == "chunk"
    assert (cite.page_start, cite.page_end) == (1, 2), "page_end 被塌成起始页了"
    assert (cite.char_start, cite.char_end) == (100, 30)
    assert not (cite.page_start == cite.page_end and cite.char_end < cite.char_start)


async def test_quote_with_fabricated_tail_is_rejected(ready, fake_llm):
    """只照抄前半句、后半句编造 → 整句校验不过 → 引用作废 → 必须拒答。"""
    db = ready
    res = await retrieve(db, "年假有几天", role="employee")
    head = res.contexts[0].text.strip()[:20]
    fabricated = head + "并且可以按三倍工资折现"
    llm = fake_llm(
        [json.dumps({"answer": "可以折现 [1]", "citations": [{"ref": 1, "quote": fabricated}], "not_found": False})]
    )
    ans = await answer_question("年假可以折现吗", res, llm, page_texts={}, selfcheck=False)
    assert ans.refused


async def test_span_slice_matches_real_page_text(ready, fake_llm):
    """用 pages 表的真实页文本断言 (page, char_start, char_end) 与原文切片逐字一致。"""
    db = ready
    pages = (await db.execute(sa.select(Page).order_by(Page.page_no))).scalars().all()
    page_texts = {(p.doc_id, p.page_no): p.text for p in pages}
    res = await retrieve(db, "年假有几天", role="employee")
    hit = next(h for h in res.contexts if h.page_start == h.page_end and "员工每年享有带薪年假五天" in h.text)
    page = page_texts[(hit.doc_id, hit.page_start)]
    quote = page[hit.char_start : hit.char_start + 40].strip()
    assert quote and quote in hit.text, "chunk 的 char_start 本身就对不上页文本"
    expected = page.find(quote)
    assert expected >= 0

    llm = fake_llm([json.dumps({"answer": "五天 [1]", "citations": [{"ref": 1, "quote": quote}]})])
    ans = await answer_question("年假有几天", res, llm, page_texts=page_texts, selfcheck=False, threshold=-99)
    assert not ans.refused
    cite = ans.citations[0]
    assert cite.span_source == "anchored"
    assert cite.page_start == hit.page_start and cite.page_end == hit.page_start
    assert (cite.char_start, cite.char_end) == (expected, expected + len(quote))
    assert page[cite.char_start : cite.char_end] == quote