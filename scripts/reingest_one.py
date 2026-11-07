"""重灌指定文档（改了解析器之后用），不动其它文档与评测集。"""

import asyncio
import sys

import sqlalchemy as sa

from documind.db import Chunk, Document, Page, bump_index_version, get_session_factory, init_db
from documind.pipeline import ingest_document
from documind.retrieve import store


async def main() -> None:
    filename = sys.argv[1] if len(sys.argv) > 1 else "zhicheng_hengxuan.pdf"
    acl = sys.argv[2].split(",") if len(sys.argv) > 2 else ["all"]
    await init_db()
    async with get_session_factory()() as s:
        docs = (await s.execute(sa.select(Document).where(Document.filename == filename))).scalars().all()
        for d in docs:
            await s.execute(sa.delete(Chunk).where(Chunk.doc_id == d.id))
            await s.execute(sa.delete(Page).where(Page.doc_id == d.id))
            await s.execute(sa.delete(Document).where(Document.id == d.id))
        # 删旧块也是"库变了"：bump 让别的进程（uvicorn）下一次查询就重建（P1-04）
        await bump_index_version(s)
        await s.commit()
        store.invalidate()
        res = await ingest_document(s, f"evals/datasets/{filename}", acl_roles=acl)
        print(res["title"], res["strategy_stats"])


asyncio.run(main())