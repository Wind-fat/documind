"""LLM 请求体的成本契约：`enable_thinking` 三态 + 流式 usage。

为什么要钉这一层：思考开关是**请求体里的一个布尔值**，它不是提示词、也不在代码里显式出现——
一旦被悄悄改掉，token 账单翻倍而不报错。这里用 MockTransport 直接看请求体。
"""

from __future__ import annotations

import json

import httpx

from documind.config import get_settings
from documind.llm import LLMClient, StreamMeta

OK_BODY = {
    "model": "qwen3.8-flash",
    "choices": [{"message": {"content": '{"answer":"x"}'}}],
    "usage": {"prompt_tokens": 11, "completion_tokens": 7},
}

SSE = (
    'data: {"model":"qwen3.8-flash","choices":[{"delta":{"content":"{\\"an"}}]}\n\n'
    'data: {"model":"qwen3.8-flash","choices":[{"delta":{"content":"swer\\":\\"x\\"}"}}]}\n\n'
    'data: {"model":"qwen3.8-flash","choices":[],"usage":{"prompt_tokens":11,"completion_tokens":7}}\n\n'
    "data: [DONE]\n\n"
)


def _client(handler) -> LLMClient:
    c = LLMClient()
    c._client = httpx.AsyncClient(transport=httpx.MockTransport(handler), trust_env=False)
    return c


async def test_thinking_flag_is_sent_explicitly():
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200, json=OK_BODY)

    c = _client(handler)
    await c.chat([{"role": "user", "content": "x"}], thinking=False)
    await c.chat([{"role": "user", "content": "x"}], thinking=True)

    assert seen[0]["enable_thinking"] is False
    assert seen[1]["enable_thinking"] is True


async def test_thinking_none_omits_field_for_unknown_providers(monkeypatch):
    """三态里的 None = 不发送该字段：接不认 `enable_thinking` 的兼容端点时不会 400。"""
    monkeypatch.setattr(get_settings(), "llm_enable_thinking", None)
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(200, json=OK_BODY)

    await _client(handler).chat([{"role": "user", "content": "x"}])
    assert "enable_thinking" not in seen


async def test_default_policy_answer_on_short_tasks_off():
    """默认策略：短任务关思考、答案生成开思考（都是配置项，不是散落在调用点）。"""
    s = get_settings()
    assert s.llm_enable_thinking is False
    assert s.llm_answer_thinking is True


async def test_stream_asks_for_usage_and_reports_it():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, content=SSE.encode("utf-8")
        )

    meta = StreamMeta()
    parts = [p async for p in _client(handler).chat_stream([{"role": "user", "content": "x"}], meta=meta)]

    assert seen["stream"] is True
    assert seen["stream_options"] == {"include_usage": True}
    assert "".join(parts) == '{"answer":"x"}'
    assert meta.usage_available is True
    assert (meta.usage.prompt_tokens, meta.usage.completion_tokens) == (11, 7)
    assert meta.finished is True