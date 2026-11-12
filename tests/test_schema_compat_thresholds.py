"""老库补列：eval_runs 的阈值相关列必须被 ensure_schema_compat 补上（P1-03/P1-10/P1-12 的隐形前置）。

真事故（2026-11-07 主控复跑时踩到）：
`documind.db` 是 P1-03 之前建的，`eval_runs` 没有 `reranker` / `recommended_threshold`；
应用启动会 init_db 补列，但 `scripts/eval_retrieval.py` 这类**不走启动路径**的脚本没人替它迁移，
于是 `resolve_refusal_threshold` 直接抛
`sqlalchemy.exc.OperationalError: no such column: eval_runs.reranker`。

这里钉住四件事：
1. `ensure_schema_compat` 对 eval_runs 也生效（不是只有 documents/chunks）；
2. 补列之后 `resolve_refusal_threshold` 能正常解析（老库没有标定 → builtin 来源），不抛异常；
   ——包括"这个库里根本没有 documents 表"的极端情形（语料指纹未知 → 空串 → 标定一律不生效）；
3. P1-10 补的 `embed_space`、P1-12 补的 `corpus_fp`：老行都取 `''`，且**不会**被当成当前桶的标定
   （老行按"历史通用值"处理：只告警、不生效；回填走 scripts/backfill_calibration_space.py）；
4. 三个桶键（reranker / embed_space / corpus_fp）都写对的行 → 解析读到 calibrated。
"""

from __future__ import annotations

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from documind.db import Base, Document, ensure_schema_compat
from documind.thresholds import corpus_fingerprint_from_hashes, resolve_refusal_threshold

LEGACY_EVAL_RUNS = (
    "CREATE TABLE eval_runs (id VARCHAR(32) PRIMARY KEY, strategy VARCHAR(32), "
    "dataset VARCHAR(128), n_cases INTEGER, metrics TEXT, report_path VARCHAR(512), "
    "created_at DATETIME)"
)

DOC_SHA = "ab" * 32
DOC_FP = corpus_fingerprint_from_hashes([DOC_SHA])


@pytest.mark.asyncio
async def test_legacy_eval_runs_gets_threshold_columns(tmp_path) -> None:
    db = tmp_path / "legacy.db"
    engine = create_async_engine(f"sqlite+aiosqlite:///{db}")
    async with engine.begin() as conn:
        await conn.exec_driver_sql(LEGACY_EVAL_RUNS)
        await conn.exec_driver_sql(
            "INSERT INTO eval_runs (id, strategy, dataset, n_cases, metrics, report_path) "
            "VALUES ('old1', 'fixed', 'default', 28, '{}', '')"
        )

    added = await ensure_schema_compat(engine)
    assert {
        "eval_runs.reranker",
        "eval_runs.recommended_threshold",
        "eval_runs.embed_space",
        "eval_runs.corpus_fp",
    } <= set(added)

    # 老行不被破坏，新列取默认（空串 = 迁移前的历史行）
    async with engine.connect() as conn:
        row = (
            await conn.exec_driver_sql(
                "SELECT strategy, reranker, embed_space, corpus_fp, recommended_threshold "
                "FROM eval_runs WHERE id='old1'"
            )
        ).one()
    assert row[0] == "fixed" and row[1] == "" and row[2] == "" and row[3] == "" and row[4] is None

    # 补列后阈值解析不再炸：没有标定 → builtin，并带警告。
    # 这个库里连 documents 表都没有 → 语料指纹读不到（空串），标定一律不生效（安全的失败方向）。
    async with AsyncSession(engine) as session:
        state = await resolve_refusal_threshold(session, "lexical")
    assert state.source == "builtin"
    assert state.calibrated is None
    assert state.corpus_fp == ""
    assert state.warning

    # 幂等
    assert await ensure_schema_compat(engine) == []
    await engine.dispose()


@pytest.mark.asyncio
async def test_calibrated_threshold_is_read_back(tmp_path) -> None:
    """三键齐备的行 → 解析读到 calibrated（证明回写链在真库结构上闭合）。"""
    db = tmp_path / "cal.db"
    engine = create_async_engine(f"sqlite+aiosqlite:///{db}")
    async with engine.begin() as conn:
        await conn.exec_driver_sql(LEGACY_EVAL_RUNS)
    await ensure_schema_compat(engine)
    # documents 表：语料指纹的**唯一**来源（/health 与解析都读它）
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    async with AsyncSession(engine) as session:
        session.add(Document(title="cal", filename="cal.md", sha256=DOC_SHA, embed_space="hash-bigram@512"))
        await session.commit()

    async with engine.begin() as conn:
        # run1 = 三键齐备；run0 = 迁移前的历史行（无空间、无语料：只告警、不生效）
        await conn.exec_driver_sql(
            "INSERT INTO eval_runs (id, strategy, dataset, n_cases, metrics, report_path, "
            "reranker, embed_space, corpus_fp, recommended_threshold) VALUES "
            "('run1', 'structure', 'default', 28, '{}', '', 'lexical', 'hash-bigram@512', "
            f"'{DOC_FP}', -4.5), "
            "('run0', 'structure', 'default', 28, '{}', '', 'lexical', '', '', -9.5)"
        )

    async with AsyncSession(engine) as session:
        state = await resolve_refusal_threshold(session, "lexical", "hash-bigram@512")
    assert state.source == "calibrated"
    assert state.value == pytest.approx(-4.5), "历史行(-9.5)不许盖过同桶的新标定，也不许被采用"
    assert state.calibrated_run == "run1"
    assert state.embed_space == "hash-bigram@512"
    assert state.corpus_fp == DOC_FP
    assert state.legacy_calibration is None, "命中同空间标定时不必再报老行"

    # 另一个桶不许串（reranker 不同）
    async with AsyncSession(engine) as session:
        other = await resolve_refusal_threshold(session, "BAAI/bge-reranker-base", "hash-bigram@512")
    assert other.source == "builtin"
    assert other.value != pytest.approx(-4.5)

    # 同 reranker、别的空间：新标定不生效，老行只告警
    async with AsyncSession(engine) as session:
        elsewhere = await resolve_refusal_threshold(session, "lexical", "text-embedding-v3@1024")
    assert elsewhere.source == "builtin"
    assert elsewhere.value == pytest.approx(-6.0)
    assert elsewhere.other_space_calibration["value"] == pytest.approx(-4.5)
    assert elsewhere.legacy_calibration["value"] == pytest.approx(-9.5)
    assert "未生效" in elsewhere.warning and "历史通用值" in elsewhere.warning

    # 同空间、同 reranker、但语料指纹对不上（本测试库若有第二篇文档就会变成这样）：
    # 这里直接传一个假指纹模拟"另一份语料"
    async with AsyncSession(engine) as session:
        stranger = await resolve_refusal_threshold(
            session, "lexical", "hash-bigram@512", corpus_fp="ffffffffffffffff"
        )
    assert stranger.source == "builtin"
    assert stranger.calibrated is None
    assert stranger.other_corpus_calibration["value"] == pytest.approx(-4.5)
    assert "另一份语料" in stranger.warning

    await engine.dispose()


@pytest.mark.asyncio
async def test_corpus_fingerprint_is_order_independent(tmp_path) -> None:
    """同一个文档集合、不同入库顺序 → 同一个指纹（否则并行 ingest 会让标定失效）。"""
    assert corpus_fingerprint_from_hashes(["b" * 64, "a" * 64]) == corpus_fingerprint_from_hashes(
        ["a" * 64, "b" * 64]
    )
    assert corpus_fingerprint_from_hashes([]) == ""
    assert len(DOC_FP) == 16