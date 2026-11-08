"""Query 理解：多轮改写 + 意图判定。

多轮是 RAG 里最容易被忽略的失分点："那它的有效期呢" 单独检索什么也召不回来。
改写用一次小模型调用（成本 ~¥0.0003），失败就退回原问题，不让它影响可用性。
"""

from __future__ import annotations

import re

from .config import get_settings
from .llm import LLMClient, parse_json

REWRITE_PROMPT = """把用户的最新问题改写成一个不依赖上下文、可以独立检索的问题。
规则：
- 保留原问题的意图与限定条件（谁、哪个部门、什么时限）
- 用【历史】里的指代词替换掉"它/这个/那条/上面说的"等
- 只输出 JSON：{"query": "改写后的问题"}
- 如果最新问题本身已经完整，原样返回。"""

INTENTS = {
    "fact": "事实型",
    "compare": "比较型",
    "procedure": "流程型",
}


def classify_intent(query: str) -> str:
    """规则版意图判定：够用、零成本、可解释（LLM 版会偶尔把流程问题判成事实问题）。"""
    if re.search(r"(相比|区别|不同|哪个更|对比|vs\.?)", query):
        return "compare"
    if re.search(r"(流程|步骤|如何办理|怎么申请|需要哪些材料|程序)", query):
        return "procedure"
    return "fact"


async def rewrite_query(query: str, history: list[dict], llm: LLMClient) -> str:
    if not history:
        return query
    convo = "\n".join(f"{h['role']}: {h['content'][:200]}" for h in history[-6:])
    reply = await llm.chat(
        [
            {"role": "system", "content": REWRITE_PROMPT},
            {"role": "user", "content": f"【历史】\n{convo}\n\n【最新问题】\n{query}\n\n只输出 JSON。"},
        ],
        json_mode=True,
        max_tokens=200,
        # 改写是短任务：关思考更划算（实测 completion 59→31、延迟 1.7s→1.1s）
        thinking=get_settings().llm_enable_thinking,
    )
    if reply.usage.mocked or not reply.text.strip():
        return query
    try:
        rewritten = str(parse_json(reply.text).get("query") or "").strip()
    except Exception:
        return query
    return rewritten or query