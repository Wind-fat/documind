"""生成层测试：引用校验、quote 防伪造、拒答判定、阈值。

这些是"不做评测也必须成立"的安全网——它们比模型质量更该被测试。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from documind.generate import answer_question
from documind.pipeline import ingest_document
from documind.retrieve import RetrievalResult, retrieve


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