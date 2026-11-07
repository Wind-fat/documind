"""检索服务：ACL → 混合召回 → RRF → rerank → 上下文组装。

一次检索要能回答三个问题（也是面试会追问的三个）：
1. 无权文档为什么不会泄露？—— `allowed` 在打分前就把无权 chunk 的行剔掉，
   BM25 与向量两条路都一样，日志里连 id 都看不到。
2. 为什么还要 rerank？—— RRF 只保序不管相关性；cross-encoder 把"条款对得上但字面不重合"的找回来。
3. 拒答阈值怎么定？—— 用 evals 跑 P-R 曲线标定（docs/adr/0004）。
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import asdict, dataclass, field

import numpy as np
import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncSession

from .chunking import Chunk
from .config import get_settings
from .db import Chunk as ChunkRow
from .db import Document
from .embed import Embedder, get_embedder
from .index import IndexItem, SearchIndex, rrf_fuse
from .rerank import Reranker, get_reranker


@dataclass(slots=True)
class Hit:
    chunk_id: str
    doc_id: str
    doc_title: str
    text: str
    page_start: int
    char_start: int
    page_end: int
    char_end: int
    section_path: str
    score: float
    bm25_rank: int | None = None
    vec_rank: int | None = None
    rrf: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(slots=True)
class RetrievalResult:
    hits: list[Hit] = field(default_factory=list)
    contexts: list[Hit] = field(default_factory=list)  # 喂模型的上下文（父子块时是父块）
    top_score: float = -99.0
    latency_ms: int = 0
    candidates: int = 0
    debug: dict = field(default_factory=dict)


class IndexStore:
    """进程内索引缓存：按策略缓存，文档变更后 bump 版本失效。

    dev 的取舍：万级 chunk 下重建 ~0.3s，比维护一个外部索引服务简单得多；
    多副本时内存索引各自重建会有短暂不一致，这也是 prod 要换 OpenSearch/pgvector 的原因。
    """

    def __init__(self) -> None:
        self._cache: dict[str, SearchIndex] = {}
        self.version = 0

    def invalidate(self) -> None:
        self._cache.clear()
        self.version += 1

    async def get(self, session: AsyncSession, strategy: str, embedder: Embedder | None = None) -> SearchIndex:
        if strategy in self._cache:
            return self._cache[strategy]
        rows = (
            (
                await session.execute(
                    sa.select(ChunkRow).where(ChunkRow.strategy == strategy, ChunkRow.is_parent.is_(False))
                )
            )
            .scalars()
            .all()
        )
        embedder = embedder or get_embedder()
        missing = [r for r in rows if r.embedding is None]
        if missing:
            mat = embedder.embed([r.text for r in missing])
            for r, v in zip(missing, mat, strict=True):
                r.embedding = v.astype(np.float32).tobytes()
            await session.commit()
        items, vecs = [], []
        for r in rows:
            items.append(
                IndexItem(
                    id=r.id,
                    text=r.text,
                    doc_id=r.doc_id,
                    strategy=r.strategy,
                    page_start=r.page_start,
                    char_start=r.char_start,
                    page_end=r.page_end,
                    char_end=r.char_end,
                    section_path=r.section_path,
                    parent_id=r.parent_id,
                )
            )
            vecs.append(r.vector())
        matrix = np.vstack(vecs).astype(np.float32) if vecs else None
        idx = SearchIndex.build(strategy, items, matrix)
        self._cache[strategy] = idx
        return idx


store = IndexStore()


async def allowed_doc_ids(session: AsyncSession, role: str) -> set[str]:
    """文档级 ACL：'all' 或角色命中才可见。admin 全可见。"""
    rows = (await session.execute(sa.select(Document.id, Document.acl_roles))).all()
    import json

    allowed: set[str] = set()
    for doc_id, acl in rows:
        roles = json.loads(acl or '["all"]')
        if role == "admin" or "all" in roles or role in roles:
            allowed.add(doc_id)
    return allowed


async def retrieve(
    session: AsyncSession,
    query: str,
    role: str = "employee",
    strategy: str = "structure",
    top_k: int | None = None,
    recall_n: int | None = None,
    embedder: Embedder | None = None,
    reranker: Reranker | None = None,
    rerank: bool = True,
) -> RetrievalResult:
    s = get_settings()
    top_k = top_k or s.top_k
    recall_n = recall_n or s.recall_n
    t0 = time.perf_counter()

    allowed = await allowed_doc_ids(session, role)
    if not allowed:
        return RetrievalResult(latency_ms=int((time.perf_counter() - t0) * 1000), debug={"reason": "acl_empty"})

    idx = await store.get(session, strategy, embedder)
    embedder = embedder or get_embedder()
    reranker = reranker or get_reranker()

    qvec = (await asyncio.to_thread(embedder.embed, [query]))[0]
    bm25_hits = await asyncio.to_thread(idx.bm25.search, query, recall_n, allowed)
    vec_hits = (
        await asyncio.to_thread(idx.vector.search, qvec, recall_n, allowed) if idx.vector else []
    )
    fused = rrf_fuse([bm25_hits, vec_hits], k=s.rrf_k)
    bm25_rank = {cid: i for i, (cid, _) in enumerate(bm25_hits)}
    vec_rank = {cid: i for i, (cid, _) in enumerate(vec_hits)}

    # 父子块：用子块召回，拿父块喂模型；没有父块就用自己
    lookup: dict[str, str] = {}
    for cid, _ in fused:
        item = idx.items.get(cid)
        if item and item.parent_id:
            lookup[cid] = item.parent_id
    parent_rows: dict[str, ChunkRow] = {}
    if lookup:
        rows = (await session.execute(sa.select(ChunkRow).where(ChunkRow.id.in_(set(lookup.values()))))).scalars().all()
        parent_rows = {r.id: r for r in rows}

    titles = dict((await session.execute(sa.select(Document.id, Document.title))).all())

    pool = fused[: max(recall_n, top_k)]
    rerank_inputs = []
    for cid, _rrf in pool:
        item = idx.items[cid]
        rerank_inputs.append((cid, item.text))
    if rerank and rerank_inputs:
        scored = await asyncio.to_thread(reranker.rerank, query, rerank_inputs, top_k)
    else:
        scored = [(cid, 1.0 / (i + 1)) for i, (cid, _) in enumerate(pool[:top_k])]

    hits: list[Hit] = []
    for cid, score in scored:
        item = idx.items[cid]
        hits.append(
            Hit(
                chunk_id=cid,
                doc_id=item.doc_id,
                doc_title=titles.get(item.doc_id, ""),
                text=item.text,
                page_start=item.page_start,
                char_start=item.char_start,
                page_end=item.page_end,
                char_end=item.char_end,
                section_path=item.section_path,
                score=round(score, 4),
                bm25_rank=bm25_rank.get(cid),
                vec_rank=vec_rank.get(cid),
                rrf=round(dict(fused).get(cid, 0.0), 6),
            )
        )

    contexts: list[Hit] = []
    for h in hits:
        pid = lookup.get(h.chunk_id)
        if pid and pid in parent_rows:
            pr = parent_rows[pid]
            contexts.append(
                Hit(
                    chunk_id=pr.id,
                    doc_id=pr.doc_id,
                    doc_title=titles.get(pr.doc_id, ""),
                    text=pr.text,
                    page_start=pr.page_start,
                    char_start=pr.char_start,
                    page_end=pr.page_end,
                    char_end=pr.char_end,
                    section_path=pr.section_path,
                    score=h.score,
                )
            )
        else:
            contexts.append(h)

    return RetrievalResult(
        hits=hits,
        contexts=contexts,
        top_score=hits[0].score if hits else -99.0,
        latency_ms=int((time.perf_counter() - t0) * 1000),
        candidates=len(fused),
        debug={"bm25": len(bm25_hits), "vector": len(vec_hits), "allowed_docs": len(allowed)},
    )


def chunks_to_rows(chunks: list[Chunk], doc_id: str, strategy: str) -> list[ChunkRow]:
    rows = []
    for c in chunks:
        rows.append(
            ChunkRow(
                doc_id=doc_id,
                strategy=strategy,
                ordinal=c.ordinal,
                is_parent=c.is_parent,
                parent_id=None,
                text=c.text,
                page_start=c.page_start,
                char_start=c.char_start,
                page_end=c.page_end,
                char_end=c.char_end,
                section_path=c.section_path,
                tokens=len(c.text),
            )
        )
    return rows