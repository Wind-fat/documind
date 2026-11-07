"""一次性把 dev 环境搭好：建库 → 灌 3 份公开语料 → 生成评测集。

用法：python scripts/bootstrap_dev.py
"""

import asyncio
import json
from pathlib import Path

from documind.config import ROOT
from documind.db import get_session_factory, init_db, upsert_user
from documind.evals.dataset import build_dataset
from documind.llm import get_llm
from documind.pipeline import ingest_document

DOCS = [
    ("evals/datasets/zhicheng_suhao.pdf", ["all"]),
    ("evals/datasets/zhicheng_hengxuan.pdf", ["all"]),
    ("evals/datasets/zhicheng_meinuohua.pdf", ["finance", "admin"]),
]
QA = ROOT / "evals" / "datasets" / "qa_default.jsonl"


async def main() -> None:
    await init_db()
    async with get_session_factory()() as s:
        for name, role in [("alice", "employee"), ("bob", "finance"), ("admin", "admin")]:
            await upsert_user(s, name, role)
        for path, acl in DOCS:
            res = await ingest_document(s, path, acl_roles=acl)
            print(path, res.get("title"), json.dumps(res.get("strategy_stats"), ensure_ascii=False), flush=True)
        path = await build_dataset(s, get_llm(), n_answerable=78, out=QA)
    print("dataset:", path, len(Path(path).read_text(encoding="utf-8").splitlines()), "条", flush=True)


asyncio.run(main())