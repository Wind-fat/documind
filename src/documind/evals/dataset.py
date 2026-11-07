"""评测集构建：从语料里合成问答对 + 固定的"答不出"问题集。

诚实边界（README 与报告里都会写）：
- 问题由 LLM 从 chunk 合成，gold 依据是该 chunk 的原文（可逐字校验）；
- 我人工抽检了 20 条，删掉 3 条"问题本身有歧义"和 1 条"答案不唯一"；
- 没有用真实用户日志（没有真实用户），所以不主张"线上分布一致"，只主张"可复现"。
"""

from __future__ import annotations

import asyncio
import json
import random
import re
from pathlib import Path

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncSession

from ..db import Chunk, Document
from ..llm import LLMClient, parse_json

GEN_PROMPT = """你在为一份企业制度文档构建检索评测集。
给定一段原文，请写 1 个**只能靠这段原文回答**的中文问题，并给出支撑答案的原文片段。

要求：
- 问题要像员工真实会问的，口语化，不要出现"根据上文""这段文字"之类的指代
- 问题不能包含答案本身
- 答案必须能从原文逐字找到依据，quote 必须是原文的连续子串（不要改写、不要省略号）
- 只输出 JSON：{"question": "...", "quote": "..."}"""

# 语料里确实没有的内容（章程里没有年假、报销、报销额度、工资日）
UNANSWERABLE = [
    "年假可以折现吗？",
    "员工每年的带薪年假有几天？",
    "出差住宿费的报销标准是多少？",
    "公司的年终奖一般在几月发放？",
    "员工每月工资几号发？",
    "试用期最长可以是多久？",
    "公司是否提供免费午餐或餐补？",
    "加班可以调休吗？",
    "公司为员工缴纳五险一金的基数是多少？",
    "股权激励计划的具体行权价格是多少？",
    "公司是否允许远程办公？",
    "出差坐高铁可以坐商务座吗？",
    "公司年会的预算是多少？",
    "内部推荐的奖金是多少？",
    "员工的绩效考核周期是季度还是半年？",
    "公司对吸烟区有什么规定？",
    "员工离职需要提前多少天提出？",
    "公司是否报销打车费？",
    "培训期间是否有额外补贴？",
    "公司的注册资本是多少美元？",
]


async def build_dataset(
    session: AsyncSession,
    llm: LLMClient,
    n_answerable: int = 80,
    seed: int = 20261107,
    out: Path | None = None,
    per_doc: int | None = None,
) -> Path:
    """从 structure 策略的块里抽样本合成问答对。"""
    docs = (await session.execute(sa.select(Document))).scalars().all()
    rnd = random.Random(seed)
    cases: list[dict] = []
    for doc in docs:
        rows = (
            (
                await session.execute(
                    sa.select(Chunk)
                    .where(Chunk.doc_id == doc.id, Chunk.strategy == "structure", Chunk.is_parent.is_(False))
                    .order_by(Chunk.ordinal)
                )
            )
            .scalars()
            .all()
        )
        # 跳过封面/目录这类没有实质内容的块
        rows = [r for r in rows if len(r.text) > 150 and "目录" not in r.text[:20]]
        take = per_doc or max(10, n_answerable // max(len(docs), 1))
        for row in rnd.sample(rows, min(take, len(rows))):
            reply = await llm.chat(
                [
                    {"role": "system", "content": GEN_PROMPT},
                    {"role": "user", "content": f"【原文】\n{row.text[:1200]}\n\n只输出 JSON。"},
                ],
                json_mode=True,
                max_tokens=300,
            )
            if reply.usage.mocked or not reply.text.strip():
                continue
            try:
                data = parse_json(reply.text)
            except Exception:
                continue
            q = str(data.get("question") or "").strip()
            quote = str(data.get("quote") or "").strip()
            if not q or not quote:
                continue
            if _norm(quote)[:30] not in _norm(row.text):
                continue
            cases.append(
                {
                    "id": f"{doc.id}-{len(cases):03d}",
                    "question": q,
                    "answerable": True,
                    "gold_doc": doc.title,
                    "gold_pages": list(range(row.page_start, row.page_end + 1)),
                    "gold_quote": quote,
                    "source_chunk": row.id,
                    "section": row.section_path,
                }
            )
    for i, q in enumerate(UNANSWERABLE):
        cases.append({"id": f"unans-{i:03d}", "question": q, "answerable": False, "gold_doc": None, "gold_pages": [], "gold_quote": ""})

    out = out or Path(__file__).resolve().parents[3] / "evals" / "datasets" / "qa_default.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for c in cases:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    return out


def load_dataset(path: str | Path) -> list[dict]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"评测集不存在: {p}（先跑 `documind.evals.dataset.build_dataset`）")
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def _norm(s: str) -> str:
    return re.sub(r"\s+", "", s or "")


async def build_if_missing(session: AsyncSession, llm: LLMClient, path: Path, n: int = 80) -> Path:
    if path.exists():
        return path
    return await build_dataset(session, llm, n_answerable=n, out=path)


_ = asyncio