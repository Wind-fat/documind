"""评测指标：检索用确定性指标，生成质量用 LLM judge（prompt 与判定口径写在代码里，可审计）。

指标口径（报告里每个数字后面都跟着这套定义）：
- Recall@K：Top-K 命中任一 gold 页 / 任一引用片段逐字出现在召回上下文里，记为命中
- Context Recall：gold quote 是否逐字出现在喂给模型的上下文里（衡量检索/切块的"召回完整性"）
- Faithfulness：回答里每条事实性论断都能在上下文找到依据的比例（judge 判定）
- Answer Relevancy：回答是否切题（judge 0/1，不再用 embedding 相似度凑数）
- 拒答准确率：无答案问题上拒答的比例（宁拒不答的代价由这个数兑现）
- 误拒率：有答案问题上被拒的比例
- 引用有效率：citation 的 (page_start,char_start,page_end,char_end) 能切出原文，
  且**归一化后的整句** quote 出现在这段切片里（不是只比前 20 字/前缀）
"""

from __future__ import annotations

import re

from ..generate import Answer, Citation
from ..llm import LLMClient, parse_json
from ..retrieve import RetrievalResult

JUDGE_PROMPT = """你是严格的评审。给定【资料】【问题】【回答】，输出 JSON：
{"faithfulness": 0.0~1.0, "relevancy": 0 或 1, "unsupported_claims": ["..."]}

faithfulness = 回答中的事实性论断里，能在【资料】找到逐字依据的比例。
- 回答拒答（明确说资料里没有）时 faithfulness = 1.0，relevancy 按"是否恰当拒答"判 1。
- 常识性连接词（因此、综上）不算论断。
- 数字、期限、比例、主体名称必须与资料完全一致，不一致即视为无依据。
relevancy = 回答是否正面回答了【问题】（答非所问给 0）。"""


def _norm(s: str) -> str:
    return re.sub(r"\s+", "", s or "")


def retrieval_metrics(retrieval: RetrievalResult, case: dict, k: int = 5) -> dict:
    gold_pages = set(case.get("gold_pages") or [])
    gold_quote = _norm(case.get("gold_quote") or "")
    hits = retrieval.hits[:k]
    page_hit = any(set(range(h.page_start, h.page_end + 1)) & gold_pages for h in hits) if gold_pages else False
    quote_hit = bool(gold_quote) and any(gold_quote[:30] in _norm(h.text) for h in hits)
    ctx_quote = bool(gold_quote) and any(gold_quote[:30] in _norm(h.text) for h in retrieval.contexts)
    rank = None
    for i, h in enumerate(hits, start=1):
        if (set(range(h.page_start, h.page_end + 1)) & gold_pages) or (gold_quote and gold_quote[:30] in _norm(h.text)):
            rank = i
            break
    return {
        "recall_at_k": 1.0 if (page_hit or quote_hit) else 0.0,
        "context_recall": 1.0 if ctx_quote else 0.0,
        "mrr": (1.0 / rank) if rank else 0.0,
    }


async def judge_answer(llm: LLMClient, case: dict, ans: Answer, retrieval: RetrievalResult) -> dict:
    context = "\n\n".join(f"[{i}] {h.text[:800]}" for i, h in enumerate(retrieval.contexts[:5], start=1))
    reply = await llm.chat(
        [
            {"role": "system", "content": JUDGE_PROMPT},
            {
                "role": "user",
                "content": (
                    f"【资料】\n{context or '(空)'}\n\n【问题】\n{case['question']}\n\n"
                    f"【回答】\n{ans.text}\n\n只输出 JSON。"
                ),
            },
        ],
        json_mode=True,
        max_tokens=400,
    )
    if reply.usage.mocked:
        return {"faithfulness": 0.0, "relevancy": 0, "unsupported_claims": [], "cost_usd": 0.0}
    try:
        data = parse_json(reply.text)
    except Exception:
        data = {"faithfulness": 0.0, "relevancy": 0, "unsupported_claims": ["judge_output_invalid"]}
    return {
        "faithfulness": float(data.get("faithfulness", 0.0)),
        "relevancy": int(data.get("relevancy", 0)),
        "unsupported_claims": data.get("unsupported_claims") or [],
        "cost_usd": reply.usage.cost_usd,
        "prompt_tokens": reply.usage.prompt_tokens,
        "completion_tokens": reply.usage.completion_tokens,
    }


def _cited_span_text(c: Citation, page_texts: dict[tuple[str, int], str]) -> str | None:
    """按 citation 自己的页区间与字符偏移切出原文；越界/缺页返回 None。

    单页 citation 就是 page[char_start:char_end]。跨页 citation 的 char_end 是
    page_end 页内偏移：首页取 char_start 到页尾、中间页整页、末页取 0:char_end。
    """
    if c.page_start == c.page_end:
        page = page_texts.get((c.doc_id, c.page_start), "")
        if not page or not (0 <= c.char_start <= c.char_end <= len(page)):
            return None
        return page[c.char_start : c.char_end]

    if c.page_start > c.page_end:
        return None
    start = page_texts.get((c.doc_id, c.page_start), "")
    end = page_texts.get((c.doc_id, c.page_end), "")
    if not start or not end:
        return None
    if not (0 <= c.char_start <= len(start) and 0 <= c.char_end <= len(end)):
        return None
    parts = [start[c.char_start :]]
    for page in range(c.page_start + 1, c.page_end):
        middle = page_texts.get((c.doc_id, page), "")
        if not middle:
            return None
        parts.append(middle)
    parts.append(end[: c.char_end])
    return "".join(parts)


def citation_validity(ans: Answer, page_texts: dict[tuple[str, int], str]) -> float:
    if ans.refused:
        return 1.0
    if not ans.citations:
        return 0.0
    ok = 0
    for c in ans.citations:
        quote = _norm(c.quote)
        if not quote:
            continue
        span = _cited_span_text(c, page_texts)
        if span is None:
            continue
        if quote in _norm(span):
            ok += 1
    return ok / len(ans.citations)


def aggregate(rows: list[dict]) -> dict:
    def mean(key: str, subset: list[dict] | None = None) -> float:
        sub = subset if subset is not None else rows
        vals = [r[key] for r in sub if r.get(key) is not None]
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    ans_rows = [r for r in rows if r["answerable"]]
    unans_rows = [r for r in rows if not r["answerable"]]
    lats = sorted(r["latency_ms"] for r in rows)

    def pct(p: float) -> float:
        if not lats:
            return 0.0
        idx = min(len(lats) - 1, int(round(p * (len(lats) - 1))))
        return float(lats[idx])

    return {
        "n_cases": len(rows),
        "n_answerable": len(ans_rows),
        "n_unanswerable": len(unans_rows),
        # Recall@5 只在可答题上平均：不可答题没有 gold 页，把它们算进分母会把
        # "检索没命中" 和 "本来就没有答案" 混为一谈（context_recall / mrr 同口径）。
        "recall_at_5": mean("recall_at_k", ans_rows),
        "context_recall": mean("context_recall", ans_rows),
        "mrr": mean("mrr", ans_rows),
        "faithfulness": mean("faithfulness", ans_rows),
        "answer_relevancy": mean("relevancy", ans_rows),
        "citation_validity": mean("citation_validity", ans_rows),
        "refusal_accuracy": mean("refused", unans_rows),
        "false_refusal_rate": mean("refused", ans_rows),
        "latency_p50_ms": pct(0.5),
        "latency_p95_ms": pct(0.95),
        "cost_per_query_usd": round(sum(r["cost_usd"] for r in rows) / max(len(rows), 1), 6),
        "total_cost_usd": round(sum(r["cost_usd"] for r in rows), 4),
    }