"""检索服务：ACL → 混合召回 → RRF → rerank → 上下文组装。

一次检索要能回答三个问题（也是面试会追问的三个）：
1. 无权文档为什么不会泄露？—— `allowed` 在打分前就把无权 chunk 的行剔掉，
   BM25 与向量两条路都一样，日志里连 id 都看不到。
2. 为什么还要 rerank？—— RRF 只保序不管相关性；cross-encoder 把"条款对得上但字面不重合"的找回来。
3. 拒答阈值怎么定？—— 用 evals 跑 P-R 曲线标定（docs/adr/0004）。
"""

from __future__ import annotations

import asyncio
import re
import time
from dataclasses import asdict, dataclass, field

import numpy as np
import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncSession

from .chunking import Chunk
from .config import get_settings
from .db import INDEX_STATE_KEY, Document, IndexState
from .db import Chunk as ChunkRow
from .embed import Embedder, EmbeddingSpaceMismatch, get_embedder
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


@dataclass(frozen=True, slots=True)
class IndexFingerprint:
    """"库里的索引长什么样"的廉价指纹：版本号 + 行数 + token 总数。

    三者任一变化都意味着内存索引过期：
    - generation：写路径（pipeline / 迁移脚本）在同一事务里 bump；
    - documents/chunks：捕获绕过写路径的裸 SQL 插入与删除；
    - tokens：捕获原地改写块文本（行数不变）。
    """

    generation: int
    documents: int
    chunks: int
    tokens: int


async def read_index_fingerprint(session: AsyncSession) -> IndexFingerprint:
    """一条语句读出库内索引指纹（每请求一次，O(1) 级别的单行聚合）。

    刻意走 `select` 的标量子查询而不是三次往返：/query 的延迟预算不允许在这里加 RTT。
    """
    row = (
        await session.execute(
            sa.select(
                sa.select(sa.func.coalesce(sa.func.max(IndexState.version), 0))
                .where(IndexState.key == INDEX_STATE_KEY)
                .scalar_subquery(),
                sa.select(sa.func.count(Document.id)).scalar_subquery(),
                sa.select(sa.func.count(ChunkRow.id)).scalar_subquery(),
                sa.select(sa.func.coalesce(sa.func.sum(ChunkRow.tokens), 0)).scalar_subquery(),
            )
        )
    ).one()
    return IndexFingerprint(
        generation=int(row[0]), documents=int(row[1]), chunks=int(row[2]), tokens=int(row[3])
    )


class IndexStore:
    """进程内索引缓存：按 (策略, 向量空间) 缓存，**每次 get 都比对库内指纹**（P1-04）。

    旧实现只在 `invalidate()` 时清缓存，那只影响调用它的进程：CLI 入库后已在跑的
    uvicorn 永远看不到新文档，删掉的块仍会被命中，且日志里没有任何异常。现在
    读者每请求读一次指纹（单行 + 两个 count），不一致就整表丢弃重建。

    dev 的取舍：万级 chunk 下重建 ~0.3s，比维护一个外部索引服务简单得多；
    多副本之间仍有"最多一次请求"的窗口不一致（读到指纹到读完 rows 之间），
    这是 prod 要换 OpenSearch/pgvector 的原因，也是这里不做强一致的原因。

    P0-02：建索引前先核对**向量空间指纹**。库里的块与当前 embedder 不是同一个 `model@dim`
    时直接抛 EmbeddingSpaceMismatch —— 不同空间的向量算余弦不会报错，只会静默给出错的排名。
    """

    def __init__(self) -> None:
        self._cache: dict[str, SearchIndex] = {}
        self._fingerprint: IndexFingerprint | None = None
        self.version = 0  # 最近一次见到的库内版本号（仅用于观测，不参与判定）
        self.builds = 0  # 重建次数：测试钉住"不许每请求都重建"

    def invalidate(self) -> None:
        """本进程主动丢弃缓存。跨进程失效**不靠它**，靠库里的指纹。"""
        self._cache.clear()

    @staticmethod
    def _mismatch_error(space: str, by_space: dict[str, int], docs: list[str]) -> EmbeddingSpaceMismatch:
        found = "、".join(f"{s}（{n} 个块）" for s, n in sorted(by_space.items()))
        # 标题可能是正文首 40 字的回退值，压掉换行避免错误信息读不出来
        names = [re.sub(r"\s+", " ", d).strip()[:40] for d in docs[:5] if d.strip()]
        where = f"；涉及文档：{', '.join(names)}" if names else ""
        return EmbeddingSpaceMismatch(
            f"向量空间不一致：当前 embedder={space}，索引中的块来自 {found}{where}。"
            "继续检索会得到语义上不可比的结果，已中止。"
            "处理：① 用当前 embedder 重新入库这些文档；"
            "② 或把 DOCUMIND_EMBED_BACKEND / DOCUMIND_EMBED_MODEL 改回建库时的配置。"
        )

    async def get(self, session: AsyncSession, strategy: str, embedder: Embedder | None = None) -> SearchIndex:
        embedder = embedder or get_embedder()
        space = embedder.space
        key = f"{strategy}::{space}"

        # --- 跨进程失效：每次都比对库内指纹（旧实现在这里直接 return 缓存） ---
        fp = await read_index_fingerprint(session)
        if fp != self._fingerprint:
            self._cache.clear()
            self._fingerprint = fp
            self.version = fp.generation

        if key in self._cache:
            return self._cache[key]
        rows = (
            (
                await session.execute(
                    sa.select(ChunkRow).where(ChunkRow.strategy == strategy, ChunkRow.is_parent.is_(False))
                )
            )
            .scalars()
            .all()
        )

        # --- 向量空间核对（先核对，再补算，绝不把新向量塞进旧空间） ---
        by_space: dict[str, int] = {}
        legacy: list[ChunkRow] = []
        for r in rows:
            if r.embedding is None:
                continue  # 没向量，谈不上空间；下面统一补算
            row_space = r.embed_space
            if not row_space:
                legacy.append(r)
            elif row_space != space:
                by_space[row_space] = by_space.get(row_space, 0) + 1
        if by_space:
            doc_ids = sorted({r.doc_id for r in rows if r.embed_space and r.embed_space != space})
            titles = (
                (await session.execute(sa.select(Document.title).where(Document.id.in_(doc_ids)))).scalars().all()
            )
            raise self._mismatch_error(space, by_space, list(titles))
        if legacy:
            raise EmbeddingSpaceMismatch(
                f"索引中有 {len(legacy)} 个块缺少向量空间指纹（迁移前的历史数据，embed_model/embed_dim 为空），"
                f"无法判断它们由哪个 embedder 生成，拒绝与当前 {space} 混用。"
                "处理：运行 scripts/backfill_embed_space.py --space <建库时的 model@dim> 显式声明，"
                "或重新入库这些文档（脚本不会删库）。"
            )

        missing = [r for r in rows if r.embedding is None]
        if missing:
            mat = embedder.embed([r.text for r in missing])
            for r, v in zip(missing, mat, strict=True):
                r.embedding = v.astype(np.float32).tobytes()
                r.embed_model = embedder.name
                r.embed_dim = int(v.shape[0])
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
        self._cache[key] = idx
        self.builds += 1
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

    embedder = embedder or get_embedder()
    idx = await store.get(session, strategy, embedder)
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
    # CE 是本机最贵的一段（~1.4ms/token，见 docs/RERANK-PERF.md），送进去的条数另设上限：
    # 30 条候选全部打分要 21s，而 RRF 顺序里第 10 名之后的块进 Top-5 的机会极小。
    rerank_pool = pool[: max(top_k, s.rerank_candidates)]
    rerank_inputs = []
    for cid, _rrf in rerank_pool:
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