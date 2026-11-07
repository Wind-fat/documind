"""LLM 客户端（OpenAI 兼容）+ 成本计量 + trace。

默认指向本机 magpie 网关（OpenAI 兼容），也可以一行切到项目三的 aigate 网关：
    DOCUMIND_LLM_BASE_URL=http://127.0.0.1:8080/v1  DOCUMIND_LLM_API_KEY=sk-ag-...
这就是"底层模型可替换"的证据——DocuMind 不知道自己背后是哪家供应商。
"""

from __future__ import annotations

import json
import time
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

    async def chat(
        self,
        messages: list[dict],
        *,
        json_mode: bool = False,
        temperature: float | None = None,
        max_tokens: int | None = None,
        allow_mock: bool = True,
    ) -> LLMReply:
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