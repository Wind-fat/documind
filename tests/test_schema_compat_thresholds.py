"""老库补列：eval_runs 的两列必须被 ensure_schema_compat 补上（P1-03 的隐形前置）。

真事故（2026-11-07 主控复跑时踩到）：
`documind.db` 是 P1-03 之前建的，`eval_runs` 没有 `reranker` / `recommended_threshold`；
应用启动会 init_db 补列，但 `scripts/eval_retrieval.py` 这类**不走启动路径**的脚本没人替它迁移，
于是 `resolve_refusal_threshold` 直接抛
`sqlalchemy.exc.OperationalError: no such column: eval_runs.reranker`。

这里钉住两件事：
1. `ensure_schema_compat` 对 eval_runs 也生效（不是只有 documents/chunks）；
2. 补列之后 `resolve_refusal_threshold` 能正常解析（老库没有标定 → builtin 来源），不抛异常。
"""

from __future__ import annotations

import pytest
import sqlalchemy as sa
from sqlalchemy.ext.asyncio import create_async_engine

from documind.db import ensure_schema_compat
from documind.thresholds import resolve_refusal_threshold

LEGACY_EVAL_RUNS = (
    "CREATE TABLE eval_runs (id VARCHAR(32) PRIMARY KEY, strategy VARCHAR(32), "
    "dataset VARCHAR(128), n_cases INTEGER, metrics TEXT, report_path VARCHAR(512), "
    "created_at DATETIME)"
)


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
    assert {"eval_runs.reranker", "eval_runs.recommended_threshold"} <= set(added)

    # 老行不被破坏，新列取默认
    async with engine.connect() as conn:
        row = (
            await conn.exec_driver_sql(
                "SELECT strategy, reranker, recommended_threshold FROM eval_runs WHERE id='old1'"
            )
        ).one()
    assert row[0] == "fixed" and row[1] == "" and row[2] is None

    # 补列后阈值解析不再炸：没有标定 → builtin，并带警告
    async with sa.ext.asyncio.AsyncSession(engine) as session:
        state = await resolve_refusal_threshold(session, "lexical")
    assert state.source == "builtin"
    assert state.calibrated is None
    assert state.warning

    # 幂等
    assert await ensure_schema_compat(engine) == []
    await engine.dispose()


@pytest.mark.asyncio
async def test_calibrated_threshold_is_read_back(tmp_path) -> None:
    """写回行 → 解析读到 calibrated（证明 P1-03 的回写链在真库结构上闭合）。"""
    db = tmp_path / "cal.db"
    engine = create_async_engine(f"sqlite+aiosqlite:///{db}")
    async with engine.begin() as conn:
        await conn.exec_driver_sql(LEGACY_EVAL_RUNS)
    await ensure_schema_compat(engine)

    async with engine.begin() as conn:
        await conn.exec_driver_sql(
            "INSERT INTO eval_runs (id, strategy, dataset, n_cases, metrics, report_path, "
            "reranker, recommended_threshold) VALUES "
            "('run1', 'structure', 'default', 28, '{}', '', 'lexical', -4.5)"
        )

    async with sa.ext.asyncio.AsyncSession(engine) as session:
        state = await resolve_refusal_threshold(session, "lexical")
    assert state.source == "calibrated"
    assert state.value == pytest.approx(-4.5)
    assert state.calibrated_run == "run1"

    # 另一个桶不许串
    async with sa.ext.asyncio.AsyncSession(engine) as session:
        other = await resolve_refusal_threshold(session, "BAAI/bge-reranker-base")
    assert other.source == "builtin"
    assert other.value != pytest.approx(-4.5)
    await engine.dispose()