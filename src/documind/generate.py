"""生成层：引用强制、拒答判定、自检重试。

三条硬规则（都不是"提示词工程"，是代码里的判定）：
1. 上下文里的每一条都有编号 [1]..[n]，模型只能引用这些编号；
2. 引用的 quote 必须在上下文原文里逐字出现，否则这条引用作废（防模型编原文）；
3. 有效引用为空 → 拒答。宁可拒，不可错答。
"""

from __future__ import annotations

import re
import time
from collections.abc import AsyncIterator
from dataclasses import asdict, dataclass, field

from .config import get_settings
from .llm import LLMClient, StreamMeta, Usage, parse_json
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
    # span 是怎么来的：anchored=在被引页内命中；fallback_page=被引页没有、退到同文档其它页；
    # chunk=整句在文档里找不到，退回块自身 span。退页不允许静默（Q3 审计）。
    span_source: str = "chunk"

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


def _norm_index(text: str) -> tuple[str, list[int]]:
    """去空白后的文本 + 每个保留字符在原文里的下标。

    归一化只能用于比较，不能直接拿来当偏移：去掉空白后下标会整体前移，
    所以这里同时维护映射，命中后再换算回页内原始字符位置。
    """
    chars: list[str] = []
    index: list[int] = []
    for i, ch in enumerate(text):
        if ch.isspace():
            continue
        chars.append(ch)
        index.append(i)
    return "".join(chars), index


def _find_in_page(text: str, quote: str) -> tuple[int, int] | None:
    """在一页原文里找**整句** quote：先按原文精确找，再按去空白后的文本找。"""
    raw = text.find(quote)
    if raw >= 0:
        return raw, raw + len(quote)
    norm_text, index = _norm_index(text)
    needle = _norm(quote)
    if not needle:
        return None
    pos = norm_text.find(needle)
    if pos < 0:
        return None
    return index[pos], index[pos + len(needle) - 1] + 1


@dataclass(frozen=True, slots=True)
class QuoteSpan:
    page: int
    char_start: int
    char_end: int
    source: str  # anchored | fallback_page


def locate_quote(
    hit: Hit, quote: str, page_texts: dict[tuple[str, int], str] | None
) -> QuoteSpan | None:
    """把模型给的 quote 定位回"第几页、页内哪个字符区间"。

    锚点是被引块自己声明的页（跨页块是 page_start..page_end）：只有这些页里
    都找不到整句时，才允许退到同文档的其它页，并返回 source="fallback_page"
    —— 退页必须在 Citation/调试信息里留痕，不能静默。
    整份文档都找不到则返回 None（调用方退回块 span），
    绝不能返回 (0, len) 这种"从页首开始高亮"的假定位——那是把错误藏起来。
    """
    needle = (quote or "").strip()
    if not needle or not page_texts:
        return None
    all_pages = sorted(page for (doc_id, page) in page_texts if doc_id == hit.doc_id)
    cited = [p for p in range(hit.page_start, hit.page_end + 1) if (hit.doc_id, p) in page_texts]
    other = [p for p in all_pages if p not in cited]
    for pages, source in ((cited, "anchored"), (other, "fallback_page")):
        for page in pages:
            found = _find_in_page(page_texts[(hit.doc_id, page)], needle)
            if found is not None:
                return QuoteSpan(page, found[0], found[1], source)
    return None


@dataclass(slots=True)
class _Prompt:
    """一次生成的输入：消息体 + 引用编号 → Hit 的映射。"""

    messages: list[dict]
    context: str
    mapping: dict[int, Hit]


def prepare_prompt(query: str, retrieval: RetrievalResult, history: list[dict] | None = None) -> _Prompt:
    context, mapping = build_context_block(retrieval.contexts)
    user_prompt = f"【资料】\n{context}\n\n【问题】\n{query}\n\n请按 JSON 格式作答。"
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for h in (history or [])[-4:]:
        messages.append({"role": h["role"], "content": h["content"]})
    messages.append({"role": "user", "content": user_prompt})
    return _Prompt(messages=messages, context=context, mapping=mapping)


def _pre_llm_refusal(retrieval: RetrievalResult, threshold: float) -> Answer | None:
    """不花钱就能判定的两种拒答：检索为空、最高分低于阈值。"""
    if not retrieval.hits:
        return Answer(
            text="文档中未找到相关条款。",
            refused=True,
            reason="检索为空（可能是文档里确实没有，或当前账号无权访问相关文档）",
            latency_ms=0,
            top_score=retrieval.top_score,
        )
    if retrieval.top_score < threshold:
        return Answer(
            text="文档中未找到相关条款。",
            refused=True,
            reason=f"最高重排分 {retrieval.top_score:.2f} 低于拒答阈值 {threshold:.2f}",
            top_score=retrieval.top_score,
            retrieval_ms=retrieval.latency_ms,
        )
    return None


def effective_threshold(threshold: float | None = None, reranker_name: str | None = None) -> float:
    """生效阈值：调用方显式给的优先，否则按 reranker 分桶解析（P1-03）。"""
    if threshold is not None:
        return float(threshold)
    from .thresholds import default_threshold

    return default_threshold(reranker_name)


async def answer_question(
    query: str,
    retrieval: RetrievalResult,
    llm: LLMClient,
    page_texts: dict[tuple[str, int], str] | None = None,
    history: list[dict] | None = None,
    selfcheck: bool | None = None,
    threshold: float | None = None,
    reranker_name: str | None = None,
) -> Answer:
    s = get_settings()
    selfcheck = s.selfcheck_retry if selfcheck is None else selfcheck
    t0 = time.perf_counter()
    thr = effective_threshold(threshold, reranker_name)

    refusal = _pre_llm_refusal(retrieval, thr)
    if refusal is not None:
        return refusal

    prompt = prepare_prompt(query, retrieval, history)
    reply = await llm.chat(prompt.messages, json_mode=True, max_tokens=800)
    gen_ms = reply.usage.latency_ms
    if reply.usage.mocked:
        return Answer(
            text="LLM 不可用：请检查 DOCUMIND_LLM_BASE_URL / DOCUMIND_LLM_API_KEY（本地 mock 不编答案）。",
            refused=True,
            reason="llm_unavailable",
            top_score=retrieval.top_score,
            retrieval_ms=retrieval.latency_ms,
        )
    return await _finalize(
        query,
        retrieval,
        prompt,
        reply.text,
        reply.usage,
        page_texts=page_texts,
        llm=llm,
        selfcheck=selfcheck,
        t0=t0,
        gen_ms=gen_ms,
    )


async def _finalize(
    query: str,
    retrieval: RetrievalResult,
    prompt: _Prompt,
    raw: str,
    usage: Usage,
    *,
    page_texts: dict[tuple[str, int], str] | None,
    llm: LLMClient,
    selfcheck: bool,
    t0: float,
    gen_ms: int,
) -> Answer:
    """把模型的原始输出变成可审计的 Answer：解析 → 引用逐字校验 → 自检/重试。"""
    try:
        data = parse_json(raw)
    except Exception:
        return Answer(
            text="文档中未找到相关条款。",
            refused=True,
            reason="模型输出无法解析为引用结构（按拒答处理）",
            top_score=retrieval.top_score,
            retrieval_ms=retrieval.latency_ms,
        )

    cites = _validate_citations(data.get("citations") or [], prompt.mapping, page_texts)
    text = str(data.get("answer") or "").strip()
    not_found = bool(data.get("not_found"))

    ans = Answer(
        text=text,
        citations=cites,
        refused=not_found or not cites,
        reason="" if cites and not not_found else ("模型判定资料不足" if not_found else "回答未携带有效引用"),
        latency_ms=int((time.perf_counter() - t0) * 1000),
        cost_usd=usage.cost_usd,
        prompt_tokens=usage.prompt_tokens,
        completion_tokens=usage.completion_tokens,
        retrieval_ms=retrieval.latency_ms,
        generation_ms=gen_ms,
        top_score=retrieval.top_score,
        debug={"contexts": len(prompt.mapping), "span_sources": _span_source_counts(cites)},
    )

    if selfcheck and not ans.refused:
        chk = await llm.chat(
            [
                {"role": "system", "content": SELFCHECK_PROMPT},
                {"role": "user", "content": f"【资料】\n{prompt.context}\n\n【回答】\n{text}\n\n只输出 JSON。"},
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
                            f"【资料】\n{prompt.context}\n\n【问题】\n{query}\n\n"
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
                c2 = _validate_citations(d2.get("citations") or [], prompt.mapping, page_texts)
                if c2 and not d2.get("not_found"):
                    ans.text = str(d2.get("answer") or ans.text)
                    ans.citations = c2
                    ans.debug["span_sources"] = _span_source_counts(c2)
                    ans.reason = ""
                    ans.refused = False
                    ans.selfcheck["retried"] = True
                else:
                    ans.refused = True
                    ans.reason = "自检两次未通过，按拒答处理"
            except Exception:
                ans.refused = True
                ans.reason = "自检重试输出无效，按拒答处理"
    return ans


_ANSWER_KEY = re.compile(r'"answer"\s*:\s*"')


class _AnswerFieldExtractor:
    r"""从**增量** JSON 文本里抽出 `"answer"` 字段的可见部分。

    为什么需要它：会话契约是 JSON（答案 + 引用），一次性吐 JSON 给用户毫无意义；
    而引用必须等 JSON 完整才能校验。折中：把 answer 字符串边到边解出来先渲染，
    `answer` 事件到达时再用校验后的权威版本覆盖（前端本来就是这么处理的）。
    """

    def __init__(self) -> None:
        self._buf = ""
        self._scan = 0
        self._state = "seek"  # seek → value → done

    def feed(self, delta: str) -> str:
        self._buf += delta
        if self._state == "seek":
            m = _ANSWER_KEY.search(self._buf)
            if m is None:
                return ""
            self._scan = m.end()
            self._state = "value"
        if self._state != "value":
            return ""
        out: list[str] = []
        buf = self._buf
        i = self._scan
        simple = {"n": "\n", "t": "\t", "r": "\r", '"': '"', "\\": "\\", "/": "/", "b": "\b", "f": "\f"}
        while i < len(buf):
            ch = buf[i]
            if ch == "\\":
                if i + 1 >= len(buf):
                    break  # 转义序列跨块：等下一个 delta 再解
                nxt = buf[i + 1]
                if nxt == "u":
                    if i + 6 > len(buf):
                        break
                    try:
                        out.append(chr(int(buf[i + 2 : i + 6], 16)))
                    except ValueError:
                        pass
                    i += 6
                else:
                    out.append(simple.get(nxt, nxt))
                    i += 2
                continue
            if ch == '"':
                self._state = "done"
                i = len(buf)
                break
            out.append(ch)
            i += 1
        self._scan = i
        return "".join(out)


class AnswerStream:
    """流式生成：迭代出答案正文的增量；结束后 `.answer` 是校验后的最终 Answer。

    事件语义（见 docs/API.md）：
    - `token` 是**草稿**：引用未校验、自检未跑，很可能被最终答案覆盖；
    - `answer` 是**权威**：与 POST /query 的响应体完全同构。
    """

    def __init__(
        self,
        query: str,
        retrieval: RetrievalResult,
        llm: LLMClient,
        *,
        page_texts: dict[tuple[str, int], str] | None = None,
        history: list[dict] | None = None,
        selfcheck: bool | None = None,
        threshold: float | None = None,
        reranker_name: str | None = None,
    ) -> None:
        s = get_settings()
        self.query = query
        self.retrieval = retrieval
        self.llm = llm
        self.page_texts = page_texts
        self.history = history or []
        self.selfcheck = s.selfcheck_retry if selfcheck is None else selfcheck
        self.threshold = effective_threshold(threshold, reranker_name)
        self.reranker_name = reranker_name or ""
        self.answer: Answer | None = None
        self.meta = StreamMeta()
        self.degraded = ""  # 非空 = 上游流式失败后退回一次性返回（必须留痕，不许静默）
        self.tokens = 0

    def __aiter__(self) -> AsyncIterator[str]:
        return self._run()

    async def _run(self) -> AsyncIterator[str]:
        from .llm import LLMError

        t0 = time.perf_counter()
        refusal = _pre_llm_refusal(self.retrieval, self.threshold)
        if refusal is not None:
            self.answer = refusal
            return

        prompt = prepare_prompt(self.query, self.retrieval, self.history)
        extractor = _AnswerFieldExtractor()
        parts: list[str] = []
        try:
            async for piece in self.llm.chat_stream(
                prompt.messages, json_mode=True, max_tokens=800, meta=self.meta
            ):
                parts.append(piece)
                visible = extractor.feed(piece)
                if visible:
                    self.tokens += 1
                    yield visible
        except LLMError as exc:
            # 上游不支持/拒绝流式（本机 magpie 的 qwen-cn 偶尔 400）。退回一次非流式调用，
            # 事件顺序不变；但把原因写进 degraded，answer.debug 与 /health 都能看到。
            self.degraded = str(exc)[:300]
            reply = await self.llm.chat(prompt.messages, json_mode=True, max_tokens=800)
            if reply.usage.mocked:
                self.answer = Answer(
                    text="LLM 不可用：请检查 DOCUMIND_LLM_BASE_URL / DOCUMIND_LLM_API_KEY（本地 mock 不编答案）。",
                    refused=True,
                    reason="llm_unavailable",
                    top_score=self.retrieval.top_score,
                    retrieval_ms=self.retrieval.latency_ms,
                )
                return
            self.meta = StreamMeta(usage=reply.usage, usage_available=True, finished=True)
            self.meta.chunks = 1
            parts = [reply.text]
            visible = extractor.feed(reply.text)
            if visible:
                self.tokens += 1
                yield visible

        ans = await _finalize(
            self.query,
            self.retrieval,
            prompt,
            "".join(parts),
            self.meta.usage,
            page_texts=self.page_texts,
            llm=self.llm,
            selfcheck=self.selfcheck,
            t0=t0,
            gen_ms=self.meta.usage.latency_ms,
        )
        ans.debug["streaming"] = {
            "mode": "sse",
            "chunks": self.meta.chunks,
            "ttft_ms": self.meta.ttft_ms,
            "token_events": self.tokens,
            "degraded": self.degraded,
        }
        self.answer = ans


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
        # quote 必须在上下文原文里**整句**逐字出现，否则这条引用无效（防编造原文）。
        # 只比前 40 字会放过"前半句照抄、后半句编造"的引用（Q3 审计）。
        if quote and _norm(quote) not in _norm(hit.text):
            continue
        span = locate_quote(hit, quote, page_texts) if quote else None
        if span is not None:
            page_start, char_start, char_end = span.page, span.char_start, span.char_end
            page_end = span.page
            span_source = span.source
        else:
            # 定位不到就退回块自身 span。跨页块必须同时保留 page_start 与 page_end：
            # 把 page_end 塌成 page_start 会得到 char_start > char_end 的倒区间（Q3 审计）。
            page_start, char_start = hit.page_start, hit.char_start
            page_end, char_end = hit.page_end, hit.char_end
            span_source = "chunk"
            if page_start == page_end and char_end < char_start:
                char_end = char_start
        out.append(
            Citation(
                ref=ref,
                doc_id=hit.doc_id,
                doc_title=hit.doc_title,
                chunk_id=hit.chunk_id,
                page_start=page_start,
                char_start=char_start,
                page_end=page_end,
                char_end=char_end,
                quote=quote or hit.text[:60],
                section_path=hit.section_path,
                span_source=span_source,
            )
        )
        seen.add(ref)
    return out


def _span_source_counts(cites: list[Citation]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for c in cites:
        counts[c.span_source] = counts.get(c.span_source, 0) + 1
    return counts