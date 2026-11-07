"""给迁移前的文档补 `documents.outline`（P1-08 的一次性回填）。

`documents.outline` 是后加的列，历史行补出来是空串，`/outline` 只能报 `outline_source="missing"`。
两条补法：

1. 重新上传同一份文件 → 走 `ingest_document` 的 duplicate 分支，只补这一列，不重算向量；
2. 文件还在 `source_path`（本机部署的常态）→ 本脚本直接重解析，**更省事且不需要重新上传**。

两者都不碰 `chunks`，所以不会引起索引重建。

用法：
    python scripts/backfill_outline.py            # 干跑，只打印会改哪些行
    python scripts/backfill_outline.py --apply    # 真正写入
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

import sqlalchemy as sa

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from documind.db import Document, get_session_factory, init_db  # noqa: E402
from documind.parsing import parse_file  # noqa: E402


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="写入数据库（默认只干跑）")
    args = ap.parse_args()

    await init_db()
    changed = skipped_missing_file = already = 0
    async with get_session_factory()() as session:
        rows = (
            (await session.execute(sa.select(Document).order_by(Document.created_at))).scalars().all()
        )
        for doc in rows:
            if (doc.outline or "").strip():
                already += 1
                continue
            src = Path(doc.source_path or "")
            if not doc.source_path or not src.is_file():
                skipped_missing_file += 1
                print(f"[skip] 源文件不在本机: {doc.title} ({doc.source_path or '空'})")
                continue
            tree = parse_file(src).outline()
            changed += 1
            print(f"[fill] {doc.title}: {len(tree)} 个根节点")
            if args.apply:
                doc.outline = json.dumps(tree, ensure_ascii=False)
        if args.apply and changed:
            await session.commit()

    print(
        f"完成：待补 {changed}、已有 {already}、源文件缺失 {skipped_missing_file}"
        f"（{'已写入' if args.apply else '干跑，未写入'}）"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))