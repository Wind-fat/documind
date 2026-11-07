"""编排层：入库与问答。API / CLI / evals 三处共用，避免三份行为不一致的实现。"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncSession

from .chunking import STRATEGIES, chunk_document
from .config import get_settings
from .db import Chunk as ChunkRow
from .db import Conversation, Document, Message, Page
from .embed import get_embedder
from .generate import Answer, answer_question
from .llm import get_llm
from .parsing import parse_file
from .query import classify_intent, rewrite_query
from .retrieve import RetrievalResult, retrieve, store


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


async def ingest_document(
    session: AsyncSession,
    path: str | Path,
    acl_roles: list[str] | None = None,
    strategies: tuple[str, ...] = STRATEGIES,
    title: str | None = None,
) -> dict:
    """解析 → 四种策略切块 → 向量化 → 落库。一份文档一次建好四套块，评测才能直接对比。"""
    p = Path(path)
    t0 = time.perf_counter()
    parsed = parse_file(p)
    # 二次断言：自定义 Parser 可能绕过 parse_file 的出口守卫，空文档绝不能走到切块/落库
    parsed.require_text()
    embedder = get_embedder()
    digest = _sha256(p)

    dup = (
        await session.execute(sa.select(Document).where(Document.sha256 == digest, Document.title == (title or parsed.title)))
    ).scalar_one_or_none()
    if dup is not None:
        return {"document_id": dup.id, "duplicate": True, "strategy_stats": {}}

    doc = Document(
        title=title or parsed.title,
        filename=p.name,
        sha256=digest,
        page_count=parsed.page_count,
        acl_roles=json.dumps(acl_roles or ["all"], ensure_ascii=False),
        source_path=str(p.resolve()),
        # 向量空间指纹：这份文档的向量由哪个 embedder / 多少维产出（P0-02）
        embed_space=embedder.space,
    )
    session.add(doc)
    await session.flush()
    for i, text in enumerate(parsed.pages, start=1):
        session.add(Page(doc_id=doc.id, page_no=i, text=text))

    stats: dict[str, dict] = {}
    for strategy in strategies:
        children, parents = chunk_document(parsed, strategy, embedder.embed)
        parent_ids: dict[str, str] = {}
        if parents:
            pm = embedder.embed([c.text for c in parents])
            rows = []
            for c, vec in zip(parents, pm, strict=True):
                row = ChunkRow(
                    doc_id=doc.id,
                    strategy=strategy,
                    ordinal=c.ordinal,
                    is_parent=True,
                    text=c.text,
                    page_start=c.page_start,
                    char_start=c.char_start,
                    page_end=c.page_end,
                    char_end=c.char_end,
                    section_path=c.section_path,
                    tokens=len(c.text),
                    embedding=vec.astype(np.float32).tobytes(),
                    embed_model=embedder.name,
                    embed_dim=int(vec.shape[0]),
                )
                session.add(row)
                await session.flush()
                parent_ids[c.key] = row.id
                rows.append(row)
        cm = embedder.embed([c.text for c in children])
        for c, vec in zip(children, cm, strict=True):
            session.add(
                ChunkRow(
                    doc_id=doc.id,
                    strategy=strategy,
                    ordinal=c.ordinal,
                    is_parent=False,
                    parent_id=parent_ids.get(c.parent_key or ""),
                    text=c.text,
                    page_start=c.page_start,
                    char_start=c.char_start,
                    page_end=c.page_end,
                    char_end=c.char_end,
                    section_path=c.section_path,
                    tokens=len(c.text),
                    embedding=vec.astype(np.float32).tobytes(),
                    embed_model=embedder.name,
                    embed_dim=int(vec.shape[0]),
                )
            )
        lengths = [len(c.text) for c in children] or [0]
        stats[strategy] = {
            "chunks": len(children),
            "parents": len(parents),
            "avg_chars": round(sum(lengths) / len(lengths), 1),
            "max_chars": max(lengths),
        }
    await session.commit()
    store.invalidate()
    return {
        "document_id": doc.id,
        "title": doc.title,
        "pages": parsed.page_count,
        "paragraphs": len(parsed.paragraphs),
        "strategy_stats": stats,
        "elapsed_ms": int((time.perf_counter() - t0) * 1000),
        "embedder": embedder.name,
        # 入库响应就带上向量空间，排障时不用再查库（P0-02）
        "embed_space": embedder.space,
    }


async def _history(session: AsyncSession, conversation_id: str | None) -> list[dict]:
    if not conversation_id:
        return []
    rows = (
        (
            await session.execute(
                sa.select(Message).where(Message.conversation_id == conversation_id).order_by(Message.created_at)
            )
        )
        .scalars()
        .all()
    )
    return [{"role": r.role, "content": r.content} for r in rows]


async def _page_texts(session: AsyncSession, doc_ids: set[str] | str) -> dict[tuple[str, int], str]:
    """一次取回相关文档的页文本，供引用定位（key = (doc_id, 页码)）。"""
    if isinstance(doc_ids, str):
        doc_ids = {doc_ids}
    rows = (
        await session.execute(sa.select(Page.doc_id, Page.page_no, Page.text).where(Page.doc_id.in_(doc_ids)))
    ).all()
    return {(doc_id, no): text for doc_id, no, text in rows}


async def ask(
    session: AsyncSession,
    question: str,
    role: str = "employee",
    user_id: str | None = None,
    conversation_id: str | None = None,
    strategy: str = "structure",
    top_k: int | None = None,
    selfcheck: bool | None = None,
    use_rewrite: bool = True,
    persist: bool = True,
) -> Answer:
    s = get_settings()
    llm = get_llm()
    t0 = time.perf_counter()
    history = await _history(session, conversation_id)

    rewritten = question
    if use_rewrite and history:
        rewritten = await rewrite_query(question, history, llm)

    retrieval: RetrievalResult = await retrieve(
        session, rewritten, role=role, strategy=strategy, top_k=top_k
    )
    page_texts: dict[tuple[str, int], str] = {}
    if retrieval.contexts:
        page_texts = await _page_texts(session, {h.doc_id for h in retrieval.contexts})
    ans = await answer_question(
        rewritten, retrieval, llm, page_texts=page_texts, history=history, selfcheck=selfcheck
    )
    ans.rewritten_query = rewritten if rewritten != question else ""
    ans.latency_ms = int((time.perf_counter() - t0) * 1000)
    ans.debug |= {
        "intent": classify_intent(rewritten),
        "strategy": strategy,
        "retrieval": retrieval.debug,
        "candidates": retrieval.candidates,
        "hits": [h.to_dict() for h in retrieval.hits],
    }

    if conversation_id is None and persist:
        conv = Conversation(user_id=user_id or "anonymous")
        session.add(conv)
        await session.flush()
        conversation_id = conv.id
    if persist:
        session.add(Message(conversation_id=conversation_id, role="user", content=question))
        session.add(
            Message(
                conversation_id=conversation_id,
                role="assistant",
                content=ans.text,
                citations=json.dumps([c.to_dict() for c in ans.citations], ensure_ascii=False),
                refused=ans.refused,
                latency_ms=ans.latency_ms,
                cost_usd=ans.cost_usd,
            )
        )
        await session.commit()
    ans.debug["conversation_id"] = conversation_id
    ans.debug.update({"total_ms": ans.latency_ms, "settings_topk": s.top_k})
    return ans


async def stream_answer(session: AsyncSession, **kwargs):
    """SSE 用：先吐检索事件（前端立刻画证据卡），再吐答案事件。"""
    ans = await ask(session, **kwargs)
    yield {"event": "evidence", "data": ans.debug.get("hits") or []}
    yield {"event": "answer", "data": ans.to_dict()}
    _ = asyncio