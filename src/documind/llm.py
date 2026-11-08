"""LLM 客户端（OpenAI 兼容）+ 成本计量 + trace。

默认指向本机 magpie 网关（OpenAI 兼容），也可以一行切到项目三的 aigate 网关：
    DOCUMIND_LLM_BASE_URL=http://127.0.0.1:8080/v1  DOCUMIND_LLM_API_KEY=sk-ag-...
这就是"底层模型可替换"的证据——DocuMind 不知道自己背后是哪家供应商。
"""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from .config import get_settings


@dataclass(slots=True)
class Usage:
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: int = 0
    mocked: bool = False


@dataclass(slots=True)
class LLMReply:
    text: str
    usage: Usage = field(default_factory=Usage)


@dataclass(slots=True)
class StreamMeta:
    """一次流式调用的旁路观测：用量、首 token 耗时、块数。

    这些字段是给 /query/stream 的 answer 事件和 trace 用的——没有它们就无法区分
    "真流式"与"等全量再一次性吐出来"（P1-05）。
    """

    usage: Usage = field(default_factory=Usage)
    chunks: int = 0
    ttft_ms: int = -1
    usage_available: bool = False
    finished: bool = False


class LLMError(RuntimeError):
    pass


class LLMClient:
    def __init__(self) -> None:
        self.s = get_settings()
        self._client: httpx.AsyncClient | None = None

    def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            # trust_env=False：本机网关不能走系统代理（Clash 会把 127.0.0.1 也接走）
            self._client = httpx.AsyncClient(trust_env=False, timeout=self.s.llm_timeout_s)
        return self._client

    def cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        return round(
            prompt_tokens / 1e6 * self.s.price_prompt_per_mtok
            + completion_tokens / 1e6 * self.s.price_completion_per_mtok,
            8,
        )

    def _payload(
        self,
        messages: list[dict],
        *,
        json_mode: bool,
        temperature: float | None,
        max_tokens: int | None,
        thinking: bool | None = None,
    ) -> dict:
        s = self.s
        payload: dict = {
            "model": s.llm_model,
            "messages": messages,
            "temperature": s.llm_temperature if temperature is None else temperature,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        if max_tokens:
            payload["max_tokens"] = max_tokens
        # 思考开关（DashScope/Qwen3 的 `enable_thinking`，请求体顶层）。三态：
        #   None = 不发送（交给供应商默认，兼容不认这个字段的端点）；
        #   False/True = 显式关闭/开启。
        # 这是纯粹的成本/延迟杠杆：实测同一问题 59→31 completion token、1.7s→1.1s，
        # 但多步推理能力会下降，所以调用点分开传（见 config.llm_enable_thinking 的注释）。
        mode = s.llm_enable_thinking if thinking is None else thinking
        if mode is not None:
            payload["enable_thinking"] = bool(mode)
        return payload

    async def chat(
        self,
        messages: list[dict],
        *,
        json_mode: bool = False,
        temperature: float | None = None,
        max_tokens: int | None = None,
        allow_mock: bool = True,
        thinking: bool | None = None,
    ) -> LLMReply:
        s = self.s
        payload = self._payload(
            messages,
            json_mode=json_mode,
            temperature=temperature,
            max_tokens=max_tokens,
            thinking=thinking,
        )
        t0 = time.perf_counter()
        try:
            r = await self._http().post(
                f"{s.llm_base_url.rstrip('/')}/chat/completions",
                headers={"Authorization": f"Bearer {s.llm_api_key}"},
                json=payload,
            )
            r.raise_for_status()
            data = r.json()
        except Exception as exc:
            if not allow_mock:
                raise LLMError(f"LLM 调用失败: {exc}") from exc
            return LLMReply(text="", usage=Usage(model="unavailable", mocked=True))
        content = (data.get("choices") or [{}])[0].get("message", {}).get("content") or ""
        u = data.get("usage") or {}
        pt, ct = int(u.get("prompt_tokens", 0)), int(u.get("completion_tokens", 0))
        usage = Usage(
            model=data.get("model", s.llm_model),
            prompt_tokens=pt,
            completion_tokens=ct,
            cost_usd=self.cost(pt, ct),
            latency_ms=int((time.perf_counter() - t0) * 1000),
        )
        record_trace("llm", {"model": usage.model, "prompt_tokens": pt, "completion_tokens": ct, "ms": usage.latency_ms})
        return LLMReply(text=content, usage=usage)

    async def chat_stream(
        self,
        messages: list[dict],
        *,
        json_mode: bool = False,
        temperature: float | None = None,
        max_tokens: int | None = None,
        meta: StreamMeta | None = None,
        thinking: bool | None = None,
    ) -> AsyncIterator[str]:
        """OpenAI 兼容 SSE：逐块 yield `delta.content`（真流式，P1-05）。

        与 `chat` 的差别只有传输方式：同样的 payload + `stream: true`。
        `reasoning_content` 不当答案吐出来（那是模型的思考过程，不是给用户的正文）。

        失败一律抛 `LLMError`（不静默 mock）：调用方要么降级到非流式，要么把错误
        作为事件发出去。流式静默变"一次性全量"就是 P1-05 要修的那种假象。
        """
        s = self.s
        meta = meta or StreamMeta()
        payload = self._payload(
            messages,
            json_mode=json_mode,
            temperature=temperature,
            max_tokens=max_tokens,
            thinking=thinking,
        )
        payload["stream"] = True
        # 流式默认**不回** usage（OpenAI 兼容协议）：显式要，否则 /query/stream 的
        # token 用量只能空着，报告里就只剩估算。DashScope 支持该字段（实测最后一块带 usage）。
        payload["stream_options"] = {"include_usage": True}
        t0 = time.perf_counter()
        try:
            async with self._http().stream(
                "POST",
                f"{s.llm_base_url.rstrip('/')}/chat/completions",
                headers={"Authorization": f"Bearer {s.llm_api_key}"},
                json=payload,
            ) as r:
                if r.status_code >= 400:
                    body = (await r.aread()).decode("utf-8", "replace").strip()[:300]
                    raise LLMError(f"上游拒绝流式请求（HTTP {r.status_code}）：{body}")
                async for line in r.aiter_lines():
                    line = line.strip()
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        obj = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    u = obj.get("usage") or {}
                    if u:
                        pt, ct = int(u.get("prompt_tokens", 0)), int(u.get("completion_tokens", 0))
                        meta.usage = Usage(
                            model=obj.get("model", s.llm_model),
                            prompt_tokens=pt,
                            completion_tokens=ct,
                            cost_usd=self.cost(pt, ct),
                            latency_ms=int((time.perf_counter() - t0) * 1000),
                        )
                        meta.usage_available = True
                    for ch in obj.get("choices") or []:
                        piece = (ch.get("delta") or {}).get("content") or ""
                        if not piece:
                            continue
                        if meta.ttft_ms < 0:
                            meta.ttft_ms = int((time.perf_counter() - t0) * 1000)
                        meta.chunks += 1
                        yield piece
        except LLMError:
            raise
        except Exception as exc:
            raise LLMError(f"流式调用失败: {exc}") from exc
        meta.finished = True
        meta.usage.latency_ms = int((time.perf_counter() - t0) * 1000)
        record_trace(
            "llm_stream",
            {
                "model": meta.usage.model or s.llm_model,
                # token 数与 llm 事件同口径：否则流式请求在 trace 里只剩耗时，成本无从对账
                "prompt_tokens": meta.usage.prompt_tokens,
                "completion_tokens": meta.usage.completion_tokens,
                "chunks": meta.chunks,
                "ttft_ms": meta.ttft_ms,
                "ms": meta.usage.latency_ms,
                "usage_available": meta.usage_available,
            },
        )

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()


def parse_json(text: str) -> dict:
    """模型偶尔把 JSON 包在 markdown fence 里，或前后带一句废话。"""
    t = text.strip()
    if t.startswith("```"):
        t = t.split("```")[1] if len(t.split("```")) > 1 else t
        t = t.replace("json", "", 1).lstrip() if t.lower().startswith("json") else t
    start, end = t.find("{"), t.rfind("}")
    if start >= 0 and end > start:
        t = t[start : end + 1]
    try:
        return json.loads(t)
    except json.JSONDecodeError as exc:
        raise LLMError(f"模型输出不是合法 JSON: {text[:200]}") from exc


def record_trace(kind: str, payload: dict) -> None:
    """trace 落本地 JSONL。字段与 Langfuse 的 observation 对齐，配了 key 就能直接转发。"""
    s = get_settings()
    path: Path = s.trace_dir / f"{time.strftime('%Y-%m-%d')}.jsonl"
    try:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": time.time(), "kind": kind, **payload}, ensure_ascii=False) + "\n")
    except OSError:  # pragma: no cover - 磁盘问题不该拖垮请求
        pass


_client: LLMClient | None = None


def get_llm() -> LLMClient:
    global _client
    if _client is None:
        _client = LLMClient()
    return _client