"""测试夹具：离线（hash embedder + 词面重排），不依赖模型下载与网络。"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

os.environ.setdefault("DOCUMIND_EMBED_BACKEND", "hash")
os.environ.setdefault("DOCUMIND_RERANK_BACKEND", "lexical")
TEST_DB = Path(__file__).resolve().parents[1] / "test.db"
os.environ.setdefault("DOCUMIND_DATABASE_URL", f"sqlite+aiosqlite:///{TEST_DB.as_posix()}")

import sqlalchemy as sa  # noqa: E402

from documind.db import Base, get_engine, get_session_factory  # noqa: E402
from documind.retrieve import store  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def _schema():
    TEST_DB.unlink(missing_ok=True)
    import asyncio

    async def setup() -> None:
        engine = get_engine()
        async with engine.begin() as conn:
            await conn.exec_driver_sql("PRAGMA journal_mode=WAL")
            await conn.run_sync(Base.metadata.create_all)
        await engine.dispose()

    asyncio.run(setup())
    yield
    asyncio.run(get_engine().dispose())


@pytest.fixture()
async def db():
    async with get_session_factory()() as session:
        store.invalidate()
        yield session
        for table in reversed(Base.metadata.sorted_tables):
            await session.execute(sa.delete(table))
        await session.commit()


@pytest.fixture()
def corpus_md(tmp_path: Path) -> Path:
    """两页的中文制度样例，结构与真实章程一致（第 X 章 / 第 X 条）。"""
    text = """# 员工制度汇编

## 第一章 总则

第一条 为规范公司管理，制定本制度。

第二条 本制度适用于公司全体在职员工。

## 第二章 休假

第三条 员工每年享有带薪年假五天，工作满十年者享有十天。

第四条 员工请病假需提供二级以上医院出具的诊断证明。

第五条 未经批准连续旷工三日以上者，公司可解除劳动合同。

## 第三章 财务

第六条 公司注册资本为人民币一亿元，由股东按认缴比例出资。

第七条 公司年度财务报告应在次年四月底前完成审计。
"""
    p = tmp_path / "员工制度.md"
    p.write_text(text, encoding="utf-8")
    return p


@pytest.fixture()
def secret_md(tmp_path: Path) -> Path:
    p = tmp_path / "财务机密制度.md"
    p.write_text(
        "# 财务机密制度\n\n## 第一章 资金\n\n第一条 公司备用金额度为人民币五十万元，由财务总监审批。\n",
        encoding="utf-8",
    )
    return p


class FakeLLM:
    """按剧本返回的假模型：用于验证"引用校验/拒答"这些判定逻辑，不验证模型质量。"""

    def __init__(self, payloads: list[str] | None = None) -> None:
        self.payloads = payloads or []
        self.calls: list[list[dict]] = []

    async def chat(self, messages, **kwargs):
        from documind.llm import LLMReply, Usage

        self.calls.append(messages)
        text = self.payloads.pop(0) if self.payloads else '{"answer":"文档中未找到相关条款","citations":[],"not_found":true}'
        return LLMReply(text=text, usage=Usage(model="fake", prompt_tokens=10, completion_tokens=5, cost_usd=0.0))


@pytest.fixture()
def fake_llm():
    return FakeLLM