"""标定样本量门槛（P1-11）与"空库也能跑评测脚本"。

两件事被这里钉住：
1. `recommended_threshold` 只有 `n_cases >= 20` 才算标定；小样本行是 provisional——
   写侧拒绝（退出码 2），读侧不生效（`/health.refusal_threshold.source` 不许变 calibrated）。
2. `scripts/eval_retrieval.py` 在**全新空 sqlite 文件**上必须能跑（先 create_all 再补列），
   不许再出现 `no such table: eval_runs`。
"""

from __future__ import annotations

import argparse
import importlib.util
import sqlite3
from pathlib import Path

import httpx
import pytest
from conftest import TEST_CORPUS_FP, seed_corpus

from documind.app import app
from documind.db import EvalRun
from documind.evals.runner import calibrate_threshold
from documind.thresholds import (
    MIN_CALIBRATION_CASES,
    calibration_gate,
    resolve_refusal_threshold,
)

SPACE = "hash-bigram@512"  # conftest 固定 DOCUMIND_EMBED_BACKEND=hash


def _rows(scores: list[tuple[float, bool]]) -> list[dict]:
    return [{"top_score": s, "answerable": a} for s, a in scores]


async def _health() -> dict:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test", trust_env=False) as c:
        r = await c.get("/health")
    assert r.status_code == 200, r.text
    return r.json()


# ------------------------------------------------------------ ① 纯粹的样本量判定


def test_min_calibration_cases_is_twenty() -> None:
    assert MIN_CALIBRATION_CASES == 20, "门槛变了就要同步改 README/ADR 里的口径"


def test_calibration_gate_refuses_below_floor() -> None:
    ok, reason = calibration_gate(1)
    assert ok is False
    assert "n_cases=1" in reason and "20" in reason and "provisional" in reason
    ok19, _ = calibration_gate(19)
    assert ok19 is False
    ok20, reason20 = calibration_gate(20)
    assert ok20 is True and reason20 == ""


def test_calibrate_threshold_flags_small_sample_as_provisional() -> None:
    small = _rows([(-12.0, False), (-9.0, False), (-3.0, True), (-1.0, True), (-0.5, True)])
    out = calibrate_threshold(small, reranker="lexical")
    assert out["n_cases"] == 5 and out["min_cases"] == 20
    assert out["provisional"] is True
    # 推荐值仍然算得出来（报告里要能看），只是不配被当成标定
    assert isinstance(out["recommended"]["threshold"], float)

    big = _rows([(-12.0 + i * 0.5, i % 2 == 0) for i in range(20)])
    out_big = calibrate_threshold(big, reranker="lexical")
    assert out_big["n_cases"] == 20 and out_big["provisional"] is False


# ------------------------------------------- ② 小样本行不许被解析成 calibrated


async def test_small_sample_row_is_provisional_not_calibrated(db) -> None:
    """直接插入一条 n_cases=5 的行（模拟老脚本/裸 SQL）：读侧必须仍然给 builtin。"""
    await seed_corpus(db)
    db.add(
        EvalRun(
            strategy="structure",
            dataset="default",
            n_cases=5,
            reranker="lexical",
            embed_space=SPACE,
            corpus_fp=TEST_CORPUS_FP,
            recommended_threshold=-4.0,
        )
    )
    await db.commit()

    st = await resolve_refusal_threshold(db, "lexical", SPACE)
    assert st.source == "builtin", "n_cases=5 的推荐值只是一条噪声，不许变成线上行为"
    assert st.value == pytest.approx(-6.0)
    assert st.calibrated is None
    assert st.calibrated_spaces == [], "provisional 行不算'已知有标定的空间'"
    assert st.provisional_calibration is not None
    assert st.provisional_calibration["value"] == pytest.approx(-4.0)
    assert st.provisional_calibration["n_cases"] == 5
    assert "provisional" in st.warning and "不生效" in st.warning

    body = await _health()
    thr = body["refusal_threshold"]
    assert thr["source"] != "calibrated"
    assert thr["calibrated"] is None
    assert thr["provisional_calibration"]["n_cases"] == 5


async def test_small_sample_row_does_not_shadow_a_real_calibration(db) -> None:
    """正式标定 + 一条更新的小样本行：生效的必须是正式那条，且警告解释小样本行为何被跳过。"""
    await seed_corpus(db)
    db.add(
        EvalRun(
            strategy="structure",
            dataset="default",
            n_cases=28,
            reranker="lexical",
            embed_space=SPACE,
            corpus_fp=TEST_CORPUS_FP,
            recommended_threshold=-4.5,
        )
    )
    await db.commit()
    db.add(
        EvalRun(
            strategy="structure",
            dataset="default",
            n_cases=5,
            reranker="lexical",
            embed_space=SPACE,
            corpus_fp=TEST_CORPUS_FP,
            recommended_threshold=-2.0,
        )
    )
    await db.commit()

    st = await resolve_refusal_threshold(db, "lexical", SPACE)
    assert st.source == "calibrated"
    assert st.value == pytest.approx(-4.5)


# ------------------------------------------- ③ 空库也能跑脚本主路径


async def test_eval_retrieval_main_runs_on_fresh_empty_db(tmp_path, monkeypatch) -> None:
    """全新空 sqlite 文件（表都不存在）上跑脚本主路径：不许 `no such table: eval_runs`。"""
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "eval_retrieval_script", root / "scripts" / "eval_retrieval.py"
    )
    assert spec is not None and spec.loader is not None
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)

    db_path = tmp_path / "fresh.db"
    assert not db_path.exists()

    from documind import db as dbmod
    from documind.config import get_settings
    from documind.retrieve import store

    monkeypatch.setenv("DOCUMIND_DATABASE_URL", f"sqlite+aiosqlite:///{db_path.as_posix()}")
    get_settings.cache_clear()

    old_engine, old_factory = dbmod._engine, dbmod._session_factory
    if old_engine is not None:
        await old_engine.dispose()
    dbmod._engine, dbmod._session_factory = None, None
    store.invalidate()
    try:
        args = argparse.Namespace(
            dataset="default",
            strategy="structure",
            limit=None,
            ans_limit=2,
            unans_limit=1,
            json=None,
            dump_hits=0,
            write=False,
        )
        out = await script.main(args)
    finally:
        if dbmod._engine is not None:
            await dbmod._engine.dispose()
        dbmod._engine, dbmod._session_factory = old_engine, old_factory
        get_settings.cache_clear()
        store.invalidate()

    summary = out["strategies"]["structure"]["summary"]
    assert summary["n_cases"] == 3, "2 题可答 + 1 题不可答"
    assert summary["provisional"] is True, "3 题当然不够 20 题门槛"

    con = sqlite3.connect(db_path)
    try:
        tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        n_rows = con.execute("SELECT COUNT(*) FROM eval_runs").fetchone()[0]
    finally:
        con.close()
    assert {"eval_runs", "documents", "chunks"} <= tables
    assert n_rows == 0, "没带 --write 就不该写标定行"