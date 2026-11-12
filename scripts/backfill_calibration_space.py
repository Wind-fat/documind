"""给迁移前的 `eval_runs` 行补上向量空间与语料指纹（P1-10 / P1-12）。

背景：`eval_runs` 先新增 `embed_space`、后新增 `corpus_fp`，两条列出现之前写入的行是空串。
空串**不是**通配符：`resolve_refusal_threshold` 会把它们当"历史通用值"——只告警、不生效。
原因是：一条没有空间/语料标注的标定无法证明它属于哪个桶，自动猜一个就等于重现
"跨库继承旧阈值"的事故（2026-11-08 实测：同一条 1.5055 换一份文档后误拒率从 0/15 变成 2/26，
同域难例拒答只剩 10/24；而这两份文档用的是**同一个** embedder——空间相同，只有语料不同）。

两个桶键维度都由这个脚本回填，因为它们回答的是同一个问题：**这条标定属于哪个库**。

    # 先看有哪些老行、库里是什么空间/什么语料指纹（不写库）
    ./.venv/Scripts/python.exe scripts/backfill_calibration_space.py --dry-run

    # 缺省：空间取库里唯一的 `documents.embed_space`，语料指纹取当前库的
    # `corpus_fingerprint()`（documents.sha256 排序后再 sha256 的前 16 位）
    ./.venv/Scripts/python.exe scripts/backfill_calibration_space.py

    # 库里空间不唯一/为空，或你要覆盖判断：显式指定
    ./.venv/Scripts/python.exe scripts/backfill_calibration_space.py --space text-embedding-v3@1024
    ./.venv/Scripts/python.exe scripts/backfill_calibration_space.py --corpus-fp 0a1b2c3d4e5f6a7b

    # 只想回填某个 reranker 的行
    ./.venv/Scripts/python.exe scripts/backfill_calibration_space.py --reranker BAAI/bge-reranker-base

`--space` 省略时**不取当前 embedder 的空间**，而是读这个库自己的 `documents.embed_space`：
库里只有一个空间时用它，有多个/没有时拒绝执行。理由：跑脚本时进程的 embedder 很可能
不是这个库的空间（换过 backend、换了维度），拿进程的空间回填等于把错的空间写进历史行——
正好是这条命令要修的病。`--corpus-fp` 同理缺省取**库**的指纹（不是进程状态）。

空库（没有 documents）时语料指纹是空串：脚本拒绝用它回填（写空串等于没写，只是把
"不知道"伪装成"已回填"），要求先入库或显式给 `--corpus-fp`。

幂等：只更新"缺哪维补哪维"的行（`embed_space=''` / `corpus_fp=''` 且有
`recommended_threshold`），重复跑是 no-op。显式值与库实际不符时照样写入（有人可能
故意这么做），但会**打印警告**。
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
from documind.thresholds import (  # noqa: E402
    LEGACY_CORPUS,
    LEGACY_SPACE,
    corpus_fingerprint,
    current_embed_space,
)


async def _library_spaces(session: AsyncSession) -> list[str]:
    """这个库里实际存在的向量空间（来自 documents.embed_space，空串不算）。"""
    stmt = (
        sa.select(Document.embed_space)
        .where(Document.embed_space != LEGACY_SPACE)
        .distinct()
        .order_by(Document.embed_space)
    )
    return [s for s in (await session.execute(stmt)).scalars().all() if s]


def _short(fp: str) -> str:
    return fp[:8] if fp else "（无）"


async def main(args: argparse.Namespace) -> int:
    added = await ensure_schema_compat()
    if added:
        print(f"[schema] 已为现有库补齐列: {', '.join(added)}", flush=True)

    async with get_session_factory()() as session:
        lib_spaces = await _library_spaces(session)
        print(f"[library] 库里的向量空间: {lib_spaces or '（无：没有文档或全是迁移前数据）'}", flush=True)

        lib_fp = await corpus_fingerprint(session)
        print(f"[library] 当前库语料指纹: {lib_fp or '（空串：库里没有文档）'}", flush=True)

        target_fp = args.corpus_fp or lib_fp

        stmt = sa.select(EvalRun).where(
            EvalRun.recommended_threshold.is_not(None),
            sa.or_(
                EvalRun.embed_space == LEGACY_SPACE,
                EvalRun.corpus_fp == LEGACY_CORPUS,
            ),
        )
        if args.reranker:
            stmt = stmt.where(EvalRun.reranker == args.reranker)
        rows = (await session.execute(stmt.order_by(EvalRun.created_at.desc()))).scalars().all()

        if not rows:
            print(
                "[ok] 没有需要回填的行"
                f"（embed_space='{LEGACY_SPACE}' 或 corpus_fp='{LEGACY_CORPUS}'，且有 recommended_threshold）"
            )
            return 0

        need_space = [r for r in rows if not r.embed_space]
        need_corpus = [r for r in rows if not r.corpus_fp]

        space = args.space
        if need_space and not space:
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

        if need_corpus and not target_fp:
            print(
                "[error] 当前库没有文档 → 语料指纹为空串，不能拿它回填。\n"
                "        写空串等于没写（空串的语义仍是\"没有语料标注\"，这行照样不生效）；\n"
                "        先入库，或用 --corpus-fp <16 位十六进制> 显式指定。",
                file=sys.stderr,
            )
            return 2

        if space and lib_spaces and space not in lib_spaces:
            # 不是错误，但必须让人看见：这行的空间和库里实际的空间不一致时，
            # 回填之后它仍然不会生效（/health 会把它报成"别的空间的标定"）
            print(
                f"[warn] --space {space} 不在库里的空间 {lib_spaces} 中；"
                "回填后这条标定依然不会在本库生效，确认这是你要的",
                file=sys.stderr,
            )
        if target_fp and lib_fp and target_fp != lib_fp:
            print(
                f"[warn] --corpus-fp {_short(target_fp)} 与本库当前语料指纹 {_short(lib_fp)} 不同；"
                "回填后这条标定在本库不会生效（它属于另一份语料），确认这是你要的",
                file=sys.stderr,
            )

        process_space = current_embed_space()
        if space and process_space and process_space != space:
            print(f"[warn] 当前进程的 embedder 空间是 {process_space}，与回填空间 {space} 不同（不影响写入）")

        print(
            f"[plan] 空间目标 {space or '（无需回填）'}，语料目标 {target_fp or '（无法回填）'}，"
            f"影响 {len(rows)} 行（缺空间 {len(need_space)} 行 / 缺语料 {len(need_corpus)} 行）："
        )
        for r in rows:
            print(
                f"  - {r.id} strategy={r.strategy} dataset={r.dataset} "
                f"reranker={r.reranker} threshold={r.recommended_threshold} "
                f"embed_space={r.embed_space or '（空）'}→{r.embed_space or space or '-'} "
                f"corpus_fp={_short(r.corpus_fp)}→{_short(r.corpus_fp or target_fp)} "
                f"created_at={r.created_at.isoformat() if r.created_at else '-'}"
            )

        if args.dry_run:
            print("[dry-run] 未写入。确认这些行的空间/语料确实是上面那组后去掉 --dry-run")
            return 0

        # 逐行改而不是 bulk update：id 列表已经打在上面，事后可精确回滚
        filled_space = 0
        filled_corpus = 0
        for r in rows:
            if space and not r.embed_space:
                r.embed_space = space
                filled_space += 1
            if target_fp and not r.corpus_fp:
                r.corpus_fp = target_fp
                filled_corpus += 1
        await session.commit()
        print(
            f"[write] 已回填 {len(rows)} 行：embed_space ×{filled_space}，corpus_fp ×{filled_corpus}；"
            "/health 的 refusal_threshold 下次解析即生效（source 会变 calibrated）"
        )
        print(
            "[note] 想让别的库/别的语料用上这个值，必须在**那个库**重跑一次标定——"
            "阈值不跨空间、也不跨语料继承"
        )
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="给 eval_runs 的历史行补 embed_space 与 corpus_fp")
    ap.add_argument("--space", default="", help="要写入的向量空间 model@dim；省略则取库里唯一的空间")
    ap.add_argument(
        "--corpus-fp",
        default="",
        help="要写入的语料指纹（16 位十六进制）；省略则取当前库的 corpus_fingerprint()",
    )
    ap.add_argument("--reranker", default="", help="只回填这个 reranker 的行")
    ap.add_argument("--dry-run", action="store_true", help="只列出将要改的行，不写库")
    raise SystemExit(asyncio.run(main(ap.parse_args())))