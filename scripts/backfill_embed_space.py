"""给历史块补上向量空间指纹（P0-02 迁移的显式一步，不删库、不重算向量）。

背景：chunks 新增 embed_model/embed_dim、documents 新增 embed_space 之后，
迁移前入库的块没有指纹。IndexStore 无法证明这些向量属于当前 embedder，会 fail-fast。
这个脚本让你**显式声明**它们属于哪个空间（而不是让代码猜）：

    # 历史库是 hash 建的（CI / 离线 demo 常见）
    .venv/Scripts/python.exe scripts/backfill_embed_space.py --space hash-bigram@512

    # 不写 --space 就用当前 embedder 的 model@dim
    .venv/Scripts/python.exe scripts/backfill_embed_space.py

    # 先看会改什么
    .venv/Scripts/python.exe scripts/backfill_embed_space.py --dry-run

脚本会校验声明的维度与实际 BLOB 长度（float32）一致，不一致直接拒绝，
避免"贴错标签"把不可比的向量伪装成同一个空间。
"""

from __future__ import annotations

import argparse
import asyncio

import sqlalchemy as sa

from documind.db import Chunk, Document, get_session_factory, init_db
from documind.embed import embedder_health


async def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="给历史块补向量空间指纹")
    ap.add_argument("--space", default="", help="历史向量所属空间 model@dim；默认取当前 embedder")
    ap.add_argument("--dry-run", action="store_true", help="只统计，不写库")
    args = ap.parse_args(argv)

    space = args.space.strip()
    if space and "@" not in space:
        print(f"--space 格式应为 model@dim，收到 {space!r}")
        return 2
    if not space:
        space = embedder_health()["space"]
    model, dim_s = space.rsplit("@", 1)
    dim = int(dim_s)

    await init_db()
    async with get_session_factory()() as s:
        rows = (
            (
                await s.execute(
                    sa.select(Chunk)
                    .where(Chunk.embedding.is_not(None))
                    .where(sa.or_(Chunk.embed_model == "", Chunk.embed_dim == 0))
                )
            )
            .scalars()
            .all()
        )
        if not rows:
            print("无需补：所有块都已有向量空间指纹")
            return 0

        bad = [r.id for r in rows if len(r.embedding or b"") != dim * 4]
        if bad:
            print(
                f"拒绝写入：声明空间 {space}（{dim} 维 = {dim * 4} 字节），"
                f"但有 {len(bad)} 个块的向量长度不符，例如 {bad[0]} "
                f"（{len(dict.fromkeys(len(r.embedding or b'') for r in rows))} 种长度）。"
                "说明 --space 选错了，请改成建库时实际用的 embedder。"
            )
            return 3

        doc_ids = {r.doc_id for r in rows}
        print(f"将标注 {len(rows)} 个块 / {len(doc_ids)} 份文档 → {space}")
        if args.dry_run:
            print("--dry-run：未写库")
            return 0

        for r in rows:
            r.embed_model = model
            r.embed_dim = dim
        await s.execute(
            sa.update(Document)
            .where(Document.id.in_(doc_ids), Document.embed_space == "")
            .values(embed_space=space)
        )
        await s.commit()

    from documind.retrieve import store

    store.invalidate()
    print(f"完成：{len(rows)} 个块已标注 {space}（documind.db 未删除，数据未重算）")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))