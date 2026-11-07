"""评测指标测试：口径是数字可信的前提，指标本身也要被测。"""

from __future__ import annotations

from documind.evals.metrics import aggregate, citation_validity, retrieval_metrics
from documind.evals.runner import calibrate_threshold
from documind.generate import Answer, Citation
from documind.retrieve import Hit, RetrievalResult


def _hit(page: int, text: str, score: float = 1.0, cid: str = "c1") -> Hit:
    return Hit(
        chunk_id=cid,
        doc_id="d1",
        doc_title="员工制度",
        text=text,
        page_start=page,
        char_start=0,
        page_end=page,
        char_end=len(text),
        section_path="第二章 休假",
        score=score,
    )


def test_recall_by_page_and_quote():
    case = {"gold_pages": [3], "gold_quote": "员工每年享有带薪年假五天", "answerable": True}
    res = RetrievalResult(hits=[_hit(2, "公司注册资本一亿元"), _hit(3, "员工每年享有带薪年假五天", cid="c2")], contexts=[])
    m = retrieval_metrics(res, case, k=5)
    assert m["recall_at_k"] == 1.0
    assert m["mrr"] == 0.5


def test_recall_miss():
    case = {"gold_pages": [9], "gold_quote": "某条不存在的条款", "answerable": True}
    res = RetrievalResult(hits=[_hit(2, "公司注册资本一亿元")], contexts=[])
    m = retrieval_metrics(res, case, k=5)
    assert m["recall_at_k"] == 0.0 and m["mrr"] == 0.0


def test_context_recall_uses_contexts_not_hits():
    case = {"gold_pages": [3], "gold_quote": "员工每年享有带薪年假五天", "answerable": True}
    res = RetrievalResult(
        hits=[_hit(3, "无关")],
        contexts=[_hit(3, "员工每年享有带薪年假五天，工作满十年者享有十天", cid="parent")],
    )
    m = retrieval_metrics(res, case, k=5)
    assert m["context_recall"] == 1.0


def test_aggregate_refusal_and_false_refusal():
    rows = [
        {"answerable": True, "refused": False, "latency_ms": 100, "cost_usd": 0.001, "recall_at_k": 1.0,
         "context_recall": 1.0, "mrr": 1.0, "faithfulness": 1.0, "relevancy": 1, "citation_validity": 1.0},
        {"answerable": True, "refused": True, "latency_ms": 300, "cost_usd": 0.001, "recall_at_k": 0.0,
         "context_recall": 0.0, "mrr": 0.0, "faithfulness": 0.0, "relevancy": 0, "citation_validity": 1.0},
        {"answerable": False, "refused": True, "latency_ms": 200, "cost_usd": 0.0, "recall_at_k": 0.0,
         "context_recall": 0.0, "mrr": 0.0, "faithfulness": None, "relevancy": None, "citation_validity": 1.0},
        {"answerable": False, "refused": False, "latency_ms": 400, "cost_usd": 0.0, "recall_at_k": 0.0,
         "context_recall": 0.0, "mrr": 0.0, "faithfulness": None, "relevancy": None, "citation_validity": 1.0},
    ]
    agg = aggregate(rows)
    assert agg["refusal_accuracy"] == 0.5       # 2 条无答案里拒了 1 条
    assert agg["false_refusal_rate"] == 0.5     # 2 条有答案里误拒 1 条
    assert agg["faithfulness"] == 0.5           # 只看可答问题
    assert agg["n_cases"] == 4


def test_calibrate_threshold_respects_false_refusal_budget():
    rows = []
    for score in (-10, -8, -6, -4):
        rows.append({"top_score": score, "answerable": False})
    for score in (-3, -2, 1, 3):
        rows.append({"top_score": score, "answerable": True})
    out = calibrate_threshold(rows)
    rec = out["recommended"]
    # 阈值取 -4：4 条无答案全拒（准确率 1.0），4 条可答全不拒（误拒 0）
    assert rec["refusal_accuracy"] == 1.0
    assert rec["false_refusal"] <= 0.15
    assert out["curve"]


def test_citation_validity_needs_quote_in_page():
    page = "第三条 员工每年享有带薪年假五天。"
    page_texts = {("d1", 3): page}
    start = page.find("员工每年")
    good = Answer(
        text="五天 [1]",
        citations=[Citation(1, "d1", "制度", "c1", 3, start, 3, start + 12, "员工每年享有带薪年假五天")],
    )
    bad = Answer(
        text="可以折现 [1]",
        citations=[Citation(1, "d1", "制度", "c1", 3, start, 3, start + 6, "年假可以折现")],
    )
    assert citation_validity(good, page_texts) == 1.0
    assert citation_validity(bad, page_texts) == 0.0


def test_citation_validity_rejects_fabricated_tail():
    """Q3：只命中前 20 字不算有效引用，整句都要在 span 里逐字存在。"""
    quote = "员工每年享有带薪年假五天，工作满十年者享有十天"
    page = f"第三条 {quote}。"
    start = page.find(quote)
    page_texts = {("d1", 3): page}
    real = Answer(
        text="五天 [1]",
        citations=[Citation(1, "d1", "制度", "c1", 3, start, 3, start + len(quote), quote)],
    )
    fabricated = quote[:25] + "，未休完可以按三倍工资折现"
    fake = Answer(
        text="可折现 [1]",
        citations=[Citation(1, "d1", "制度", "c1", 3, start, 3, start + len(quote), fabricated)],
    )
    assert citation_validity(real, page_texts) == 1.0
    assert citation_validity(fake, page_texts) == 0.0, "前半句命中、后半句编造，指标仍是绿的就是假绿"


def test_citation_validity_checks_span_bounds_and_cross_page():
    page1 = "甲" * 20 + "跨页引用的前半段"
    page2 = "后半段到第二页结束" + "乙" * 10
    page_texts = {("d1", 1): page1, ("d1", 2): page2}
    quote = page1[20:] + page2[:10]
    cross = Answer(
        text="跨页 [1]",
        citations=[Citation(1, "d1", "制度", "c1", 1, 20, 2, 10, quote)],
    )
    out_of_bounds = Answer(
        text="越界 [1]",
        citations=[Citation(1, "d1", "制度", "c1", 1, 20, 1, len(page1) + 5, quote)],
    )
    assert citation_validity(cross, page_texts) == 1.0
    assert citation_validity(out_of_bounds, page_texts) == 0.0