"""给迁移前的 `eval_runs` 行补上向量空间（P1-10）。

背景：`eval_runs` 新增 `embed_space` 之后，迁移前的行是空串。空串**不是**通配符：
`resolve_refusal_threshold` 会把它当"历史通用值"——只告警、不生效。原因是
一条无空间标注的行无法证明它属于哪个空间，自动猜一个就等于重现"跨库继承旧阈值"的事故
（2026-11-08 实测：同一条 1.5055 换一份文档后误拒率从 0/15 变成 2/26，同域难例拒答只剩 10/24）。

`--space` 省略时**不取当前 embedder 的空间**，而是读这个库自己的 `documents.embed_space`：
库里只有一个空间时用它，有多个/没有时拒绝执行。理由：跑脚本时进程的 embedder 很可能
不是这个库的空间（换过 backend、换了维度），拿进程的空间回填等于把错的空间写进历史行——
正好是这条命令要修的病。

    # 先看有哪些老行、库里是什么空间（不写库）
    ./.venv/Scripts/python.exe scripts/backfill_calibration_space.py --dry-run

    # 库里只有一个空间时：直接回填（脚本会把它打印出来）
    ./.venv/Scripts/python.exe scripts/backfill_calibration_space.py

    # 库里空间不唯一/为空，或你要覆盖判断：显式指定
    ./.venv/Scripts/python.exe scripts/backfill_calibration_space.py --space text-embedding-v3@1024

    # 只想回填某个 reranker 的行
    ./.venv/Scripts/python.exe scripts/backfill_calibration_space.py --reranker BAAI/bge-reranker-base

幂等：只更新 `embed_space=''` 且有 `recommended_threshold` 的行，重复跑是 no-op。
显式 `--space` 与库里的空间不一致时照样写入（有人可能故意这么做），但会**打印警告**。
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import sqlalchemy as sa  # noqa: E402
from sqlalchemy.ext.asyncio import AsyncSession  # noqa: E402

from documind.db import Document, EvalRun, ensure_schema_compat, get_session_factory  # noqa: E402
from documind.thresholds import LEGACY_SPACE, current_embed_space  # noqa: E402


async def _library_spaces(session: AsyncSession) -> list[str]:
    """这个库里实际存在的向量空间（来自 documents.embed_space，空串不算）。"""
    stmt = (
        sa.select(Document.embed_space)
        .where(Document.embed_space != LEGACY_SPACE)
        .distinct()
        .order_by(Document.embed_space)
    )
    return [s for s in (await session.execute(stmt)).scalars().all() if s]


async def main(args: argparse.Namespace) -> int:
    added = await ensure_schema_compat()
    if added:
        print(f"[schema] 已为现有库补齐列: {', '.join(added)}", flush=True)

    async with get_session_factory()() as session:
        lib_spaces = await _library_spaces(session)
        print(f"[library] 库里的向量空间: {lib_spaces or '（无：没有文档或全是迁移前数据）'}", flush=True)

        space = args.space
        if not space:
            if len(lib_spaces) == 1:
                space = lib_spaces[0]
            else:
                print(
                    "[error] 库里没有唯一的向量空间，不能替你猜。\n"
                    f"        库里实际: {lib_spaces}\n"
                    "        显式给一个：--space <model@dim>，例如 --space text-embedding-v3@1024",
                    file=sys.stderr,
                )
                return 2
        elif lib_spaces and space not in lib_spaces:
            # 不是错误，但必须让人看见：这行的空间和库里实际的空间不一致时，
            # 回填之后它仍然不会生效（/health 会把它报成"别的空间的标定"）
            print(
                f"[warn] --space {space} 不在库里的空间 {lib_spaces} 中；"
                "回填后这条标定依然不会在本库生效，确认这是你要的",
                file=sys.stderr,
            )

        process_space = current_embed_space()
        if process_space and process_space != space:
            print(f"[warn] 当前进程的 embedder 空间是 {process_space}，与回填空间 {space} 不同（不影响写入）")

        stmt = sa.select(EvalRun).where(
            EvalRun.embed_space == LEGACY_SPACE,
            EvalRun.recommended_threshold.is_not(None),
        )
        if args.reranker:
            stmt = stmt.where(EvalRun.reranker == args.reranker)
        rows = (await session.execute(stmt.order_by(EvalRun.created_at.desc()))).scalars().all()

        if not rows:
            print(f"[ok] 没有需要回填的行（embed_space='{LEGACY_SPACE}' 且 recommended_threshold 非空）")
            return 0

        print(f"[plan] 回填空间 {space}，影响 {len(rows)} 行：")
        for r in rows:
            print(
                f"  - {r.id} strategy={r.strategy} dataset={r.dataset} "
                f"reranker={r.reranker} threshold={r.recommended_threshold} "
                f"created_at={r.created_at.isoformat() if r.created_at else '-'}"
            )

        if args.dry_run:
            print("[dry-run] 未写入。确认这些行的空间确实是上面那个后去掉 --dry-run")
            return 0

        # 逐行改而不是 bulk update：id 列表已经打在上面，事后可精确回滚
        for r in rows:
            r.embed_space = space
        await session.commit()
        print(
            f"[write] 已回填 {len(rows)} 行 → embed_space={space}；"
            "/health 的 refusal_threshold 下次解析即生效"
        )
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="给 eval_runs 的历史行补 embed_space")
    ap.add_argument("--space", default="", help="要写入的向量空间 model@dim；省略则取库里唯一的空间")
    ap.add_argument("--reranker", default="", help="只回填这个 reranker 的行")
    ap.add_argument("--dry-run", action="store_true", help="只列出将要改的行，不写库")
    raise SystemExit(asyncio.run(main(ap.parse_args())))