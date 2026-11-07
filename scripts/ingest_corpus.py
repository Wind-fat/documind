"""把公开语料灌进 dev 库（固定 ACL 演示：美诺华章程仅财务/管理员可见）。"""
import asyncio
import json

from documind.db import get_session_factory, init_db, upsert_user
from documind.pipeline import ingest_document

DOCS = [
    ("evals/datasets/zhicheng_suhao.pdf", ["all"]),
    ("evals/datasets/zhicheng_hengxuan.pdf", ["all"]),
    ("evals/datasets/zhicheng_meinuohua.pdf", ["finance", "admin"]),
]

async def main():
    await init_db()
    async with get_session_factory()() as s:
        for name, role in [("alice","employee"),("bob","finance"),("admin","admin")]:
            await upsert_user(s, name, role)
        for path, acl in DOCS:
            res = await ingest_document(s, path, acl_roles=acl)
            print(path, json.dumps(res.get("strategy_stats"), ensure_ascii=False), res.get("elapsed_ms"), "ms", flush=True)
asyncio.run(main())
