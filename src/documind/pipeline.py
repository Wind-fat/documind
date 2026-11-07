"""编排层：入库与问答。API / CLI / evals 三处共用，避免三份行为不一致的实现。"""

from __future__ import annotations

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
from .db import Conversation, Document, Message, Page, bump_index_version, load_page_texts
from .embed import get_embedder
from .generate import Answer, AnswerStream, answer_question
from .llm import get_llm
from .parsing import parse_file
from .query import classify_intent, rewrite_query
from .retrieve import RetrievalResult, retrieve, store
from .thresholds import ThresholdState, resolve_refusal_threshold


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
        # 同一份文件不重复建块。但 `documents.outline` 是后加的列（P1-08）：迁移前的行
        # 没有章节树，`/outline` 只能报 missing。重新上传一次就把章节树补上——不需要
        # 重算向量，也不需要重建块（哈希与标题都没变，块内容必然相同）。
        backfilled = False
        if not (dup.outline or "").strip():
            dup.outline = json.dumps(parsed.outline(), ensure_ascii=False)
            await session.commit()
            backfilled = True
        return {
            "document_id": dup.id,
            "duplicate": True,
            "strategy_stats": {},
            "outline_backfilled": backfilled,
        }

    doc = Document(
        title=title or parsed.title,
        filename=p.name,
        sha256=digest,
        page_count=parsed.page_count,
        acl_roles=json.dumps(acl_roles or ["all"], ensure_ascii=False),
        source_path=str(p.resolve()),
        # 向量空间指纹：这份文档的向量由哪个 embedder / 多少维产出（P0-02）
        embed_space=embedder.space,
        # 章节树在解析期就定型落库（P1-08）：这是 `ParsedDocument.outline()` 的唯一出口，
        # /documents/{id}/outline 之后只读这一列。与用哪种切块策略入库**无关**——同一个
        # ParsedDocument 产出的章节树不该因为 strategy=fixed 就消失。
        outline=json.dumps(parsed.outline(), ensure_ascii=False),
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
    # 版本号与数据在同一个事务里提交：别的进程（uvicorn）下一次 get 就会重建索引（P1-04）
    await bump_index_version(session)
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


async def _prepare(
    session: AsyncSession,
    question: str,
    *,
    role: str,
    strategy: str,
    top_k: int | None,
    conversation_id: str | None,
    use_rewrite: bool,
    llm,
) -> tuple[str, RetrievalResult, dict[tuple[str, int], str], list[dict]]:
    """检索前置的全部动作：历史 → 改写 → 检索 → 引用定位用页文本。

    /query 与 /query/stream 共用这一条路径，否则两条路的检索行为迟早漂移。
    """
    history = await _history(session, conversation_id)
    rewritten = question
    if use_rewrite and history:
        rewritten = await rewrite_query(question, history, llm)
    retrieval: RetrievalResult = await retrieve(session, rewritten, role=role, strategy=strategy, top_k=top_k)
    page_texts: dict[tuple[str, int], str] = {}
    if retrieval.contexts:
        page_texts = await load_page_texts(session, {h.doc_id for h in retrieval.contexts})
    return rewritten, retrieval, page_texts, history


async def _threshold(session: AsyncSession) -> ThresholdState:
    """当前 reranker 的生效拒答阈值（按 name 分桶，P1-03）。"""
    from .rerank import get_reranker

    return await resolve_refusal_threshold(session, get_reranker().name)


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
    rewritten, retrieval, page_texts, history = await _prepare(
        session,
        question,
        role=role,
        strategy=strategy,
        top_k=top_k,
        conversation_id=conversation_id,
        use_rewrite=use_rewrite,
        llm=llm,
    )
    thr = await _threshold(session)
    ans = await answer_question(
        rewritten,
        retrieval,
        llm,
        page_texts=page_texts,
        history=history,
        selfcheck=selfcheck,
        threshold=thr.value,
        reranker_name=thr.reranker,
    )
    ans.rewritten_query = rewritten if rewritten != question else ""
    ans.latency_ms = int((time.perf_counter() - t0) * 1000)
    ans.debug |= _debug_payload(rewritten, retrieval, strategy)
    ans.debug["refusal_threshold"] = thr.to_dict()

    if conversation_id is None and persist:
        conv = Conversation(user_id=user_id or "anonymous")
        session.add(conv)
        await session.flush()
        conversation_id = conv.id
    if persist:
        await _persist_exchange(session, conversation_id, question, ans)
    ans.debug["conversation_id"] = conversation_id
    ans.debug.update({"total_ms": ans.latency_ms, "settings_topk": s.top_k})
    return ans


def _debug_payload(rewritten: str, retrieval: RetrievalResult, strategy: str) -> dict:
    return {
        "intent": classify_intent(rewritten),
        "strategy": strategy,
        "retrieval": retrieval.debug,
        "candidates": retrieval.candidates,
        "hits": [h.to_dict() for h in retrieval.hits],
    }


async def _persist_exchange(
    session: AsyncSession, conversation_id: str | None, question: str, ans: Answer
) -> None:
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


async def stream_answer(
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
):
    """SSE 事件流：检索完成即发 evidence，随后逐块发 token，最后发 answer（P1-05）。

    顺序是契约：`evidence` → 0..n × `token` → `answer`。
    evidence 不等生成，token 不等全量——这两句就是"真流式"的定义，也是
    docs/STREAM.md 里 TTFT 数字的来源。
    """
    s = get_settings()
    llm = get_llm()
    t0 = time.perf_counter()
    rewritten, retrieval, page_texts, history = await _prepare(
        session,
        question,
        role=role,
        strategy=strategy,
        top_k=top_k,
        conversation_id=conversation_id,
        use_rewrite=use_rewrite,
        llm=llm,
    )
    thr = await _threshold(session)

    # ① 检索一完成就发证据：这一行之前没有任何模型调用
    yield {"event": "evidence", "data": [h.to_dict() for h in retrieval.hits]}

    stream = AnswerStream(
        rewritten,
        retrieval,
        llm,
        page_texts=page_texts,
        history=history,
        selfcheck=selfcheck,
        threshold=thr.value,
        reranker_name=thr.reranker,
    )
    # ② 逐块转发答案正文的增量
    async for delta in stream:
        yield {"event": "token", "data": {"text": delta}}

    ans = stream.answer
    if ans is None:  # pragma: no cover - AnswerStream 必定留下 Answer
        raise RuntimeError("AnswerStream 结束时没有产出 Answer")
    ans.rewritten_query = rewritten if rewritten != question else ""
    ans.latency_ms = int((time.perf_counter() - t0) * 1000)
    ans.debug |= _debug_payload(rewritten, retrieval, strategy)
    ans.debug["refusal_threshold"] = thr.to_dict()
    ans.debug.setdefault("streaming", {})["degraded_reason"] = stream.degraded

    if conversation_id is None and persist:
        conv = Conversation(user_id=user_id or "anonymous")
        session.add(conv)
        await session.flush()
        conversation_id = conv.id
    if persist:
        await _persist_exchange(session, conversation_id, question, ans)
    ans.debug["conversation_id"] = conversation_id
    ans.debug.update({"total_ms": ans.latency_ms, "settings_topk": s.top_k})

    # ③ 权威结果（未被拦截时它可能与 token 草稿不同，前端应以此覆盖）
    yield {"event": "answer", "data": ans.to_dict()}