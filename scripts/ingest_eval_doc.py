"""把单份 PDF 灌进**独立的**评测库（不碰 `data/real_dashscope.db`），并报入库成本。

为什么单独一个脚本：`ingest_corpus.py` 固定灌 3 份文档、`reingest_one.py` 只删重灌同库里的同名文档，
两者都会写进「当前库」。交叉验证要求第二份文档有自己的库（否则两套 chunk 混在一个检索空间里，
分数不可比，也会污染上一轮的标定数据）。

入库成本直接取 `OpenAICompatEmbedder` 的运行时计数（`requests` / `prompt_tokens` 来自供应商
`usage.prompt_tokens`，不是字数估算），耗时用 `perf_counter`，块数与字数来自 `strategy_stats`。

用法：
    DOCUMIND_DATABASE_URL="sqlite+aiosqlite:///D:/code/documind/data/meinuohua_dashscope.db" \
    ./scripts/with_dashscope.sh ./.venv/Scripts/python.exe scripts/ingest_eval_doc.py \
        --pdf evals/datasets/zhicheng_meinuohua.pdf --strategy structure
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


async def main(args: argparse.Namespace) -> dict:
    from documind.config import get_settings
    from documind.db import get_session_factory, init_db, upsert_user
    from documind.embed import get_embedder
    from documind.pipeline import ingest_document

    await init_db()
    embedder = get_embedder()
    t0 = time.perf_counter()
    async with get_session_factory()() as s:
        for name, role in (("bob", "finance"), ("admin", "admin")):
            await upsert_user(s, name, role)
        res = await ingest_document(s, args.pdf, acl_roles=args.acl.split(","), strategies=(args.strategy,))
    wall = time.perf_counter() - t0

    stats = res.get("strategy_stats", {}).get(args.strategy, {})
    out = {
        "db": get_settings().database_url,
        "pdf": args.pdf,
        "title": res.get("title"),
        "pages": res.get("pages"),
        "paragraphs": res.get("paragraphs"),
        "duplicate": res.get("duplicate", False),
        "strategy": args.strategy,
        "chunks": stats.get("chunks"),
        "parents": stats.get("parents"),
        "avg_chars": stats.get("avg_chars"),
        "max_chars": stats.get("max_chars"),
        "embed_model": res.get("embedder"),
        "embed_space": res.get("embed_space"),
        "embed_requests": getattr(embedder, "requests", None),
        "embed_prompt_tokens": getattr(embedder, "prompt_tokens", None),
        "elapsed_ms": res.get("elapsed_ms"),
        "wall_ms": int(wall * 1000),
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True)
    ap.add_argument("--strategy", default="structure")
    ap.add_argument("--acl", default="finance,admin")
    asyncio.run(main(ap.parse_args()))