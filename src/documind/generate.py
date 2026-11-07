"""生成层：引用强制、拒答判定、自检重试。

三条硬规则（都不是"提示词工程"，是代码里的判定）：
1. 上下文里的每一条都有编号 [1]..[n]，模型只能引用这些编号；
2. 引用的 quote 必须在上下文原文里逐字出现，否则这条引用作废（防模型编原文）；
3. 有效引用为空 → 拒答。宁可拒，不可错答。
"""

from __future__ import annotations

import re
import time
from dataclasses import asdict, dataclass, field

from .config import get_settings
from .llm import LLMClient, parse_json
from .retrieve import Hit, RetrievalResult

SYSTEM_PROMPT = """你是企业知识库问答助手。规则（违反任意一条即视为无效回答）：
1. 只能使用【资料】中出现的原文作答，禁止使用你自己的知识补充事实。
2. 每个事实性论断后面必须标注来源编号，形如 [1] 或 [1][3]。
3. 如果【资料】不足以回答问题，必须回答"文档中未找到相关条款"，不要猜测、不要用相邻条款类比。
4. 引用编号只能来自【资料】中真实存在的编号，不得编造。
5. 输出严格的 JSON，不要输出任何其它内容：
{"answer": "回答正文（含 [n] 标注）", "citations": [{"ref": 1, "quote": "该编号资料中支撑结论的原文片段（逐字复制）"}], "not_found": false}
如果资料不足，输出 {"answer": "文档中未找到相关条款", "citations": [], "not_found": true}。"""

SELFCHECK_PROMPT = """你是严格的审稿人。判断【回答】中的每一个事实性论断是否都能在【资料】里找到逐字依据。
只输出 JSON：{"supported": true/false, "unsupported_claims": ["..."]}
回答里的寒暄、编号标注不算论断。宁可判 false，也不要放过任何一句资料里没有的话。"""


@dataclass(slots=True)
class Citation:
    ref: int
    doc_id: str
    doc_title: str
    chunk_id: str
    page_start: int
    char_start: int
    page_end: int
    char_end: int
    quote: str
    section_path: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(slots=True)
class Answer:
    text: str
    citations: list[Citation] = field(default_factory=list)
    refused: bool = False
    reason: str = ""
    rewritten_query: str = ""
    latency_ms: int = 0
    cost_usd: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    retrieval_ms: int = 0
    generation_ms: int = 0
    top_score: float = -99.0
    selfcheck: dict = field(default_factory=dict)
    debug: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["citations"] = [c.to_dict() for c in self.citations]
        return d


def build_context_block(contexts: list[Hit], max_chars: int = 6000) -> tuple[str, dict[int, Hit]]:
    blocks: list[str] = []
    mapping: dict[int, Hit] = {}
    total = 0
    for i, hit in enumerate(contexts, start=1):
        head = f"[{i}] 《{hit.doc_title}》 第{hit.page_start}-{hit.page_end}页"
        if hit.section_path:
            head += f" · {hit.section_path}"
        body = hit.text.strip()
        if total + len(body) > max_chars and blocks:
            break
        blocks.append(f"{head}\n{body}")
        mapping[i] = hit
        total += len(body)
    return "\n\n".join(blocks), mapping


def _norm(s: str) -> str:
    return re.sub(r"\s+", "", s)


def locate_quote(
    hit: Hit, quote: str, page_texts: dict[tuple[str, int], str] | None
) -> tuple[int, int, int] | None:
    """把模型给的 quote 定位回"第几页、页内哪个字符区间"。

    必须真的在该页原文里找到，找不到就返回 None（退回整块 span），
    绝不能返回 (0, len) 这种"从页首开始高亮"的假定位——那是把错误藏起来。
    """
    if not quote or not page_texts:
        return None
    needle = quote.strip()[:40]
    for (doc_id, page), text in sorted(page_texts.items(), key=lambda kv: kv[0][1]):
        if doc_id != hit.doc_id:
            continue
        raw = text.find(needle)
        if raw >= 0:
            return (page, raw, raw + len(needle))
    return None


async def answer_question(
    query: str,
    retrieval: RetrievalResult,
    llm: LLMClient,
    page_texts: dict[tuple[str, int], str] | None = None,
    history: list[dict] | None = None,
    selfcheck: bool | None = None,
) -> Answer:
    s = get_settings()
    selfcheck = s.selfcheck_retry if selfcheck is None else selfcheck
    title = ""
    t0 = time.perf_counter()

    if not retrieval.hits:
        return Answer(
            text="文档中未找到相关条款。",
            refused=True,
            reason="检索为空（可能是文档里确实没有，或当前账号无权访问相关文档）",
            latency_ms=0,
            top_score=retrieval.top_score,
        )
    if retrieval.top_score < s.refusal_rerank_threshold:
        return Answer(
            text="文档中未找到相关条款。",
            refused=True,
            reason=f"最高重排分 {retrieval.top_score:.2f} 低于拒答阈值 {s.refusal_rerank_threshold:.2f}",
            top_score=retrieval.top_score,
            retrieval_ms=retrieval.latency_ms,
        )

    context, mapping = build_context_block(retrieval.contexts)
    user_prompt = f"【资料】\n{context}\n\n【问题】\n{query}\n\n请按 JSON 格式作答。"
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for h in (history or [])[-4:]:
        messages.append({"role": h["role"], "content": h["content"]})
    messages.append({"role": "user", "content": user_prompt})

    reply = await llm.chat(messages, json_mode=True, max_tokens=800)
    gen_ms = reply.usage.latency_ms
    if reply.usage.mocked:
        return Answer(
            text="LLM 不可用：请检查 DOCUMIND_LLM_BASE_URL / DOCUMIND_LLM_API_KEY（本地 mock 不编答案）。",
            refused=True,
            reason="llm_unavailable",
            top_score=retrieval.top_score,
            retrieval_ms=retrieval.latency_ms,
        )

    try:
        data = parse_json(reply.text)
    except Exception:
        return Answer(
            text="文档中未找到相关条款。",
            refused=True,
            reason="模型输出无法解析为引用结构（按拒答处理）",
            top_score=retrieval.top_score,
            retrieval_ms=retrieval.latency_ms,
        )

    cites = _validate_citations(data.get("citations") or [], mapping, page_texts)
    text = str(data.get("answer") or "").strip()
    not_found = bool(data.get("not_found"))

    ans = Answer(
        text=text,
        citations=cites,
        refused=not_found or not cites,
        reason="" if cites and not not_found else ("模型判定资料不足" if not_found else "回答未携带有效引用"),
        latency_ms=int((time.perf_counter() - t0) * 1000),
        cost_usd=reply.usage.cost_usd,
        prompt_tokens=reply.usage.prompt_tokens,
        completion_tokens=reply.usage.completion_tokens,
        retrieval_ms=retrieval.latency_ms,
        generation_ms=gen_ms,
        top_score=retrieval.top_score,
        debug={"contexts": len(mapping)},
    )

    if selfcheck and not ans.refused:
        chk = await llm.chat(
            [
                {"role": "system", "content": SELFCHECK_PROMPT},
                {"role": "user", "content": f"【资料】\n{context}\n\n【回答】\n{text}\n\n只输出 JSON。"},
            ],
            json_mode=True,
            max_tokens=300,
        )
        try:
            verdict = parse_json(chk.text)
            if not isinstance(verdict, dict):
                raise TypeError(f"自检输出不是 JSON 对象：{type(verdict).__name__}")
        except Exception as exc:
            # 校验没跑成功 ≠ 校验通过（审计 P0-5）。这里必须置 supported=None：
            # 回落到 True 会让前端渲染绿色"自检通过"，等于把"不知道"当成"已确认"。
            verdict = {
                "supported": None,
                "unparsed": True,
                "error": f"自检输出无法解析为 JSON：{str(exc)[:160]}",
                "unsupported_claims": [],
            }
        ans.cost_usd = round(ans.cost_usd + chk.usage.cost_usd, 8)
        ans.prompt_tokens += chk.usage.prompt_tokens
        ans.completion_tokens += chk.usage.completion_tokens
        ans.selfcheck = verdict
        if verdict.get("supported") is False:
            # 只有**明确判否**才重试。None（解析失败）不重试：那次调用本身没给出结论，
            # 重试只会掩盖"自检没跑成"这件事。
            bad = "；".join(verdict.get("unsupported_claims") or [])[:300]
            retry = await llm.chat(
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            f"【资料】\n{context}\n\n【问题】\n{query}\n\n"
                            f"上一次回答里这些内容在资料中找不到依据：{bad}\n"
                            "请删掉它们，只用资料原文重答，仍然输出 JSON。"
                        ),
                    },
                ],
                json_mode=True,
                max_tokens=800,
            )
            ans.cost_usd = round(ans.cost_usd + retry.usage.cost_usd, 8)
            try:
                d2 = parse_json(retry.text)
                c2 = _validate_citations(d2.get("citations") or [], mapping, page_texts)
                if c2 and not d2.get("not_found"):
                    ans.text = str(d2.get("answer") or ans.text)
                    ans.citations = c2
                    ans.reason = ""
                    ans.refused = False
                    ans.selfcheck["retried"] = True
                else:
                    ans.refused = True
                    ans.reason = "自检两次未通过，按拒答处理"
            except Exception:
                ans.refused = True
                ans.reason = "自检重试输出无效，按拒答处理"
    _ = title
    return ans


def _validate_citations(
    raw: list, mapping: dict[int, Hit], page_texts: dict[tuple[str, int], str] | None
) -> list[Citation]:
    out: list[Citation] = []
    seen: set[int] = set()
    for c in raw:
        try:
            ref = int(c.get("ref"))
        except (TypeError, ValueError):
            continue
        if ref in seen or ref not in mapping:
            continue
        hit = mapping[ref]
        quote = str(c.get("quote") or "").strip()
        # quote 必须在上下文原文里逐字出现，否则这条引用无效（防编造原文）
        if quote and _norm(quote)[:40] not in _norm(hit.text):
            continue
        span = locate_quote(hit, quote, page_texts) if quote else None
        if span:
            page, cs, ce = span
        else:
            page, cs, ce = hit.page_start, hit.char_start, hit.char_end
        out.append(
            Citation(
                ref=ref,
                doc_id=hit.doc_id,
                doc_title=hit.doc_title,
                chunk_id=hit.chunk_id,
                page_start=page,
                char_start=cs,
                page_end=page,
                char_end=ce,
                quote=quote or hit.text[:60],
                section_path=hit.section_path,
            )
        )
        seen.add(ref)
    return out