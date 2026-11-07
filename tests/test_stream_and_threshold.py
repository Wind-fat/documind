"""P1-05 真流式 + P1-03 阈值回写的验收测试。

两条都是"承诺不实"类问题，所以测试必须能区分"真"与"看起来像"：
1. 流式：假模型每块延迟 100ms，断言 evidence 先到、≥2 个 token 在 answer 之前到达，
   且首个 token 到达时模型**还没跑完**——旧实现（等全量再一次吐两个事件）
   一个 token 事件都产不出来，`finished_at` 也已经写完。
2. 阈值：两个 reranker 各标各的桶，写回后生效值与标定值一致；
   环境变量覆盖时 /health 的 mismatch 必须亮。
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import httpx
import pytest

from documind import pipeline
from documind.app import app
from documind.config import get_settings
from documind.db import EvalRun
from documind.evals.runner import calibrate_threshold
from documind.generate import answer_question
from documind.pipeline import ingest_document
from documind.retrieve import Hit, RetrievalResult
from documind.thresholds import resolve_refusal_threshold


@pytest.fixture()
async def loaded(db, corpus_md: Path):
    from documind.db import upsert_user

    await upsert_user(db, "alice", "employee")
    await ingest_document(db, corpus_md, acl_roles=["all"], title="员工制度")
    return db


@pytest.fixture()
def clean_settings():
    """环境变量改过之后必须清 lru_cache，否则读到的是上一个用例的 Settings。"""
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


ANSWER_JSON = json.dumps(
    {
        "answer": "员工每年享有带薪年假五天，工作满十年者享有十天 [1]",
        "citations": [{"ref": 1, "quote": "员工每年享有带薪年假五天"}],
        "not_found": False,
    },
    ensure_ascii=False,
)


class SlowStreamLLM:
    """每块延迟 delay 秒的假流：块数、到达时间、结束时刻都可观测。"""

    def __init__(self, pieces: list[str], delay: float = 0.1) -> None:
        self.pieces = pieces
        self.delay = delay
        self.finished_at: float | None = None
        self.started_at: float | None = None
        self.stream_calls = 0
        self.chat_calls = 0

    async def chat_stream(self, messages, *, json_mode=False, max_tokens=None, meta=None):
        self.stream_calls += 1
        self.started_at = time.perf_counter()
        for piece in self.pieces:
            await asyncio.sleep(self.delay)
            if meta is not None:
                meta.chunks += 1
                if meta.ttft_ms < 0:
                    meta.ttft_ms = int(self.delay * 1000)
            yield piece
        self.finished_at = time.perf_counter()

    async def chat(self, messages, **kwargs):  # pragma: no cover - 流式用例不该退回非流式
        self.chat_calls += 1
        raise AssertionError("流式路径退化成了非流式调用")


@pytest.fixture()
async def live_server(loaded):
    """真 uvicorn（同进程、同一事件循环，真 socket）：只有它能把 SSE 的到达时间测准。

    httpx 的 ASGITransport 会把整个 body 收齐再返回，用它测 TTFT 只能得到
    "全部事件同一时刻到达"——那正是我们要区分的假流式。
    """
    import uvicorn

    config = uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning", lifespan="on")
    server = uvicorn.Server(config)
    task = asyncio.create_task(server.serve())
    for _ in range(200):
        if server.started:
            break
        await asyncio.sleep(0.05)
    assert server.started, "uvicorn 未能在 10s 内启动"
    port = server.servers[0].sockets[0].getsockname()[1]
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        await asyncio.wait_for(task, timeout=15)


def _split(payload: str, n: int) -> list[str]:
    size = max(len(payload) // n, 1)
    return [payload[i : i + size] for i in range(0, len(payload), size)]


async def _collect_events(base_url: str, params: dict) -> list[dict]:
    """读 SSE 并按到达顺序打时间戳（收到即记，不等整条流结束）。"""
    events: list[dict] = []
    buf = ""
    t0 = time.perf_counter()
    async with (
        httpx.AsyncClient(base_url=base_url, timeout=30, trust_env=False) as http,
        http.stream("GET", "/query/stream", params=params, headers={"X-User": "alice"}) as resp,
    ):
        assert resp.status_code == 200, await resp.aread()
        async for chunk in resp.aiter_bytes():
            buf += chunk.decode("utf-8")
            while "\n\n" in buf:
                block, buf = buf.split("\n\n", 1)
                name, data = "", ""
                for line in block.split("\n"):
                    if line.startswith("event:"):
                        name = line[6:].strip()
                    elif line.startswith("data:"):
                        data += line[5:].strip()
                if name:
                    events.append({"event": name, "data": json.loads(data), "t": time.perf_counter() - t0})
    return events


async def test_stream_evidence_then_tokens_before_answer(live_server, monkeypatch):
    """真流式的判定：token 在答案完成**之前**到达，且 evidence 不等生成。"""
    llm = SlowStreamLLM(_split(ANSWER_JSON, 6), delay=0.1)
    monkeypatch.setattr(pipeline, "get_llm", lambda: llm)

    events = await _collect_events(
        live_server, {"question": "年假有几天", "strategy": "structure", "selfcheck": "false"}
    )
    names = [e["event"] for e in events]
    assert names[0] == "evidence", f"第一个事件必须是 evidence，实际 {names}"
    assert names[-1] == "answer"
    tokens = [e for e in events if e["event"] == "token"]
    answers = [e for e in events if e["event"] == "answer"]
    assert len(tokens) >= 2, f"只收到 {len(tokens)} 个 token 事件——这就是假流式"

    # ① token 全部早于 answer（不是攒完一起发）
    assert tokens[0]["t"] < answers[0]["t"]
    # ② 首个 token 到达时，模型还在生成（延迟 0.1s/块，还有 5 块没吐）
    assert answers[0]["t"] - tokens[0]["t"] >= 0.2, (
        f"token 与 answer 相差仅 {answers[0]['t'] - tokens[0]['t']:.3f}s = 伪流式"
    )
    # ③ evidence 早于任何 token，且早于生成结束
    evidence = next(e for e in events if e["event"] == "evidence")
    assert evidence["t"] < tokens[0]["t"]
    # ④ TTFT 显著小于端到端（门槛取 50%，实测远小于此）
    assert tokens[0]["t"] < answers[0]["t"] * 0.5, (tokens[0]["t"], answers[0]["t"])

    # 载荷契约：token 是正文增量，answer 是校验后的权威版本
    assert all(isinstance(e["data"].get("text"), str) for e in tokens)
    body = answers[0]["data"]
    assert body["refused"] is False and body["citations"]
    assert body["debug"]["streaming"]["chunks"] >= 2
    assert body["debug"]["streaming"]["token_events"] >= 2
    assert llm.chat_calls == 0, "上游流式是通的，不该退回非流式"
    assert "".join(t["data"]["text"] for t in tokens).startswith("员工每年享有带薪年假五天")


async def test_stream_falls_back_to_single_shot_and_says_so(live_server, monkeypatch):
    """上游拒绝流式时必须退回非流式，但要在 degraded 里说清楚（不许静默假装在流）。"""
    from documind.llm import LLMError, LLMReply, Usage

    class NoStream:
        async def chat_stream(self, messages, *, json_mode=False, max_tokens=None, meta=None):
            raise LLMError("上游拒绝流式请求（HTTP 400）：Required body invalid")
            yield ""  # pragma: no cover

        async def chat(self, messages, **kwargs):
            return LLMReply(
                text=ANSWER_JSON,
                usage=Usage(model="fake", prompt_tokens=10, completion_tokens=5),
            )

    monkeypatch.setattr(pipeline, "get_llm", lambda: NoStream())
    events = await _collect_events(
        live_server, {"question": "年假有几天", "strategy": "structure", "selfcheck": "false"}
    )
    answer = events[-1]
    assert answer["event"] == "answer" and answer["data"]["refused"] is False
    degraded = answer["data"]["debug"]["streaming"]["degraded"]
    assert "Required body invalid" in degraded, degraded


# ----------------------------- P1-03 阈值 -----------------------------


def _rows(scores: list[tuple[float, bool]]) -> list[dict]:
    return [{"top_score": s, "answerable": a} for s, a in scores]


async def test_calibration_grid_covers_actual_score_range():
    """网格必须由实际分数决定：[min-1, max+1]，且能扫到 cross-encoder 的正值侧。"""
    rows = _rows([(-3.0, False), (-1.5, False), (0.4, True), (2.2, True), (6.0, True)])
    out = calibrate_threshold(rows, reranker="bge-reranker-base")
    grid = out["grid"]
    assert grid["min"] == pytest.approx(-4.0)  # min - 1
    assert grid["max"] == pytest.approx(7.0)  # max + 1
    assert grid["max"] > 0, "旧网格 [-14,0] 扫不到 logits > 0 的区间"
    assert out["reranker"] == "bge-reranker-base"
    # 推荐值必须落在网格里，而不是网格外的常数
    assert out["recommended"]["threshold"] in [c["threshold"] for c in out["curve"]]


async def test_thresholds_are_bucketed_per_reranker(db, clean_settings):
    """两个 reranker 各自标定 → 两个桶，互不串；写回后生效值 == 标定值。"""
    lexical_rows = _rows([(-12.0, False), (-10.0, False), (-8.0, False), (-3.0, True), (-1.0, True)])
    cross_rows = _rows([(-2.0, False), (-0.5, False), (1.5, True), (3.0, True), (5.5, True)])

    lex = calibrate_threshold(lexical_rows, reranker="lexical")
    cross = calibrate_threshold(cross_rows, reranker="bge-reranker-base")
    lex_th, cross_th = lex["recommended"]["threshold"], cross["recommended"]["threshold"]
    assert lex_th != cross_th, "两个分数域标出了同一个阈值，说明还在共用一个常量"
    assert -13.0 <= lex_th <= 0.0
    assert -3.0 <= cross_th <= 6.5

    # 回写：EvalRun 按 reranker 各存一份
    for name, value in (("lexical", lex_th), ("bge-reranker-base", cross_th)):
        db.add(EvalRun(strategy="structure", dataset="default", n_cases=5, reranker=name, recommended_threshold=value))
    await db.commit()

    st_lex = await resolve_refusal_threshold(db, "lexical")
    st_cross = await resolve_refusal_threshold(db, "bge-reranker-base")
    assert st_lex.source == "calibrated" and st_lex.value == pytest.approx(lex_th)
    assert st_cross.source == "calibrated" and st_cross.value == pytest.approx(cross_th)
    assert st_lex.value != st_cross.value
    # 未标定过的第三个后端拿不到别人的阈值
    st_other = await resolve_refusal_threshold(db, "some-future-reranker")
    assert st_other.source == "builtin" and st_other.value == pytest.approx(-6.0)
    assert "还没有标定值" in st_other.warning


async def test_env_override_beats_calibration_and_flags_mismatch(db, monkeypatch, clean_settings):
    db.add(EvalRun(strategy="structure", dataset="default", n_cases=5, reranker="lexical", recommended_threshold=-4.0))
    await db.commit()

    monkeypatch.setenv("DOCUMIND_REFUSAL_THRESHOLD", "1.25")
    get_settings.cache_clear()
    st = await resolve_refusal_threshold(db, "lexical")
    assert st.value == pytest.approx(1.25) and st.source == "env"
    assert st.calibrated == pytest.approx(-4.0)
    assert st.mismatch is True and "不一致" in st.warning

    # 覆盖值与标定值一致时不该报"打架"
    monkeypatch.setenv("DOCUMIND_REFUSAL_THRESHOLD", "-4.0")
    get_settings.cache_clear()
    st2 = await resolve_refusal_threshold(db, "lexical")
    assert st2.mismatch is False and st2.warning == ""


async def test_effective_threshold_changes_the_refusal_decision(db, clean_settings, monkeypatch):
    """写回之后，拒答判定用的是标定值，而不是那个 -6.0。"""
    monkeypatch.delenv("DOCUMIND_REFUSAL_THRESHOLD", raising=False)
    monkeypatch.delenv("DOCUMIND_REFUSAL_RERANK_THRESHOLD", raising=False)
    get_settings.cache_clear()
    db.add(EvalRun(strategy="structure", dataset="default", n_cases=5, reranker="lexical", recommended_threshold=-4.0))
    await db.commit()
    st = await resolve_refusal_threshold(db, "lexical")
    assert st.value == pytest.approx(-4.0)

    text = "第三条 员工每年享有带薪年假五天，工作满十年者享有十天。"
    hit = Hit(
        chunk_id="c1",
        doc_id="d1",
        doc_title="员工制度",
        text=text,
        page_start=1,
        char_start=0,
        page_end=1,
        char_end=len(text),
        section_path="第二章 休假",
        score=-5.0,
    )

    class OneShot:
        def __init__(self, payload: str) -> None:
            self.payload = payload

        async def chat(self, messages, **kwargs):
            from documind.llm import LLMReply, Usage

            return LLMReply(text=self.payload, usage=Usage(model="fake"))

    low = RetrievalResult(hits=[hit], contexts=[hit], top_score=-5.0, latency_ms=1)
    refused = await answer_question("年假有几天", low, OneShot(ANSWER_JSON), page_texts={}, selfcheck=False, threshold=st.value)
    assert refused.refused is True
    assert "-4.00" in refused.reason, refused.reason  # 提示的是生效阈值，不是常量
    assert "阈值" in refused.reason

    high = RetrievalResult(hits=[hit], contexts=[hit], top_score=-3.0, latency_ms=1)
    answered = await answer_question("年假有几天", high, OneShot(ANSWER_JSON), page_texts={}, selfcheck=False, threshold=st.value)
    assert answered.refused is False and answered.citations