"""数据模型与连接。

dev = SQLite（aiosqlite），prod = PostgreSQL + pgvector，同一套 ORM。
向量存 float32 blob：dev 后端在进程内用 numpy 暴力搜索（万级 chunk 下 <10ms），
prod 换 pgvector 的 `<=>` 算子，检索代码不动（见 index.VectorIndex 接口）。
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import uuid
from collections.abc import AsyncIterator

import numpy as np
import sqlalchemy as sa
from sqlalchemy import ForeignKey, Index, String, Text, func
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from .config import get_settings


def _uid() -> str:
    return uuid.uuid4().hex[:16]


def _now() -> dt.datetime:
    return dt.datetime.now(dt.UTC)


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uid)
    name: Mapped[str] = mapped_column(String(64), unique=True)
    role: Mapped[str] = mapped_column(String(32), default="employee")  # employee / finance / hr / admin
    created_at: Mapped[dt.datetime] = mapped_column(default=_now)


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uid)
    title: Mapped[str] = mapped_column(String(256))
    filename: Mapped[str] = mapped_column(String(256))
    sha256: Mapped[str] = mapped_column(String(64), index=True)
    page_count: Mapped[int] = mapped_column(default=0)
    # 文档级 ACL：哪些角色可见。检索阶段就按它过滤，不是生成后过滤。
    acl_roles: Mapped[str] = mapped_column(Text, default='["all"]')
    source_path: Mapped[str] = mapped_column(String(512), default="")
    # 向量空间指纹 `model@dim`：这份文档的向量由哪个 embedder 产出。
    # 空串 = 迁移前的历史数据（无指纹），检索时会被 fail-fast 拦下（见 index_store）。
    embed_space: Mapped[str] = mapped_column(String(96), default="")
    created_at: Mapped[dt.datetime] = mapped_column(default=_now)

    def roles(self) -> list[str]:
        return json.loads(self.acl_roles)


class Page(Base):
    __tablename__ = "pages"

    doc_id: Mapped[str] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"), primary_key=True)
    page_no: Mapped[int] = mapped_column(primary_key=True)  # 1-based
    text: Mapped[str] = mapped_column(Text)


class Chunk(Base):
    """一个检索单元。

    span 定位：page_start/char_start → page_end/char_end 的**页内字符偏移**，
    前端拿 pages 表的页文本就能精确高亮（这是"引用到 span"能落地的前提）。
    """

    __tablename__ = "chunks"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uid)
    doc_id: Mapped[str] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"), index=True)
    strategy: Mapped[str] = mapped_column(String(32), index=True)
    ordinal: Mapped[int] = mapped_column(default=0)
    is_parent: Mapped[bool] = mapped_column(default=False)
    parent_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    text: Mapped[str] = mapped_column(Text)
    page_start: Mapped[int] = mapped_column(default=1)
    char_start: Mapped[int] = mapped_column(default=0)
    page_end: Mapped[int] = mapped_column(default=1)
    char_end: Mapped[int] = mapped_column(default=0)
    section_path: Mapped[str] = mapped_column(String(512), default="")
    tokens: Mapped[int] = mapped_column(default=0)
    embedding: Mapped[bytes | None] = mapped_column(sa.LargeBinary, nullable=True)
    # 每个块自带向量空间指纹：换了 embedder 之后能逐块判定，而不是只信文档级一行（P0-02）
    embed_model: Mapped[str] = mapped_column(String(96), default="")
    embed_dim: Mapped[int] = mapped_column(default=0)

    def vector(self) -> np.ndarray:
        assert self.embedding is not None
        return np.frombuffer(self.embedding, dtype=np.float32)

    @property
    def embed_space(self) -> str:
        """`model@dim`；空串表示迁移前的历史块（无指纹，不可与其他版本混用）。"""
        if not self.embed_model or not self.embed_dim:
            return ""
        return f"{self.embed_model}@{self.embed_dim}"


class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uid)
    user_id: Mapped[str] = mapped_column(String(32), index=True)
    created_at: Mapped[dt.datetime] = mapped_column(default=_now)


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uid)
    conversation_id: Mapped[str] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE"), index=True
    )
    role: Mapped[str] = mapped_column(String(16))
    content: Mapped[str] = mapped_column(Text)
    citations: Mapped[str] = mapped_column(Text, default="[]")
    refused: Mapped[bool] = mapped_column(default=False)
    latency_ms: Mapped[int] = mapped_column(default=0)
    cost_usd: Mapped[float] = mapped_column(default=0.0)
    created_at: Mapped[dt.datetime] = mapped_column(default=_now)


class EvalRun(Base):
    __tablename__ = "eval_runs"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uid)
    strategy: Mapped[str] = mapped_column(String(32), index=True)
    dataset: Mapped[str] = mapped_column(String(128), default="")
    n_cases: Mapped[int] = mapped_column(default=0)
    metrics: Mapped[str] = mapped_column(Text, default="{}")
    report_path: Mapped[str] = mapped_column(String(512), default="")
    # 这次评测用的是哪个 reranker（阈值的桶键）与它标出来的建议阈值。
    # 没有这两列，"标定值"只活在报告 HTML 里，代码无从读取（P1-03）。
    reranker: Mapped[str] = mapped_column(String(96), default="", index=True)
    recommended_threshold: Mapped[float | None] = mapped_column(nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(default=_now)


Index("ix_chunks_doc_strategy", Chunk.doc_id, Chunk.strategy)


_engine: sa.ext.asyncio.AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def get_engine() -> sa.ext.asyncio.AsyncEngine:
    global _engine, _session_factory
    if _engine is None:
        s = get_settings()
        _engine = create_async_engine(
            s.database_url,
            echo=False,
            # SQLite 高并发写会 database is locked；WAL + 30s 超时是必须的
            connect_args={"timeout": 30} if s.database_url.startswith("sqlite") else {},
        )
        _session_factory = async_sessionmaker(_engine, expire_on_commit=False)
    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    get_engine()
    assert _session_factory is not None
    return _session_factory


def _column_default_sql(col: sa.Column) -> str | None:
    """老库补列时给已有行用的字面默认值（只覆盖本项目实际新增的标量列）。"""
    if isinstance(col.type, (String, Text)):
        return "''"
    if isinstance(col.type, sa.Integer):
        return "0"
    if isinstance(col.type, sa.Boolean):
        return "0"
    return None


def _add_column_sql(table: sa.Table, col: sa.Column, dialect: sa.Dialect) -> str:
    prep = dialect.identifier_preparer
    sql = (
        f"ALTER TABLE {prep.format_table(table)} ADD COLUMN "
        f"{prep.format_column(col)} {col.type.compile(dialect=dialect)}"
    )
    default = _column_default_sql(col)
    if not col.nullable and default is not None:
        sql += f" NOT NULL DEFAULT {default}"
    return sql


def _ensure_columns(sync_conn: sa.Connection) -> list[str]:
    """启动期补列：create_all 不会给已存在的表加列，这里逐列补齐（不删库、不动数据）。"""
    inspector = sa.inspect(sync_conn)
    existing_tables = set(inspector.get_table_names())
    added: list[str] = []
    for table in Base.metadata.sorted_tables:
        if table.name not in existing_tables:
            continue
        have = {c["name"] for c in inspector.get_columns(table.name)}
        for col in table.columns:
            if col.name in have:
                continue
            sync_conn.exec_driver_sql(_add_column_sql(table, col, sync_conn.dialect))
            added.append(f"{table.name}.{col.name}")
    return added


async def ensure_schema_compat(engine: sa.ext.asyncio.AsyncEngine | None = None) -> list[str]:
    """对现有库补列并返回补了哪些列（对空库是 no-op）。"""
    eng = engine or get_engine()
    async with eng.begin() as conn:
        return await conn.run_sync(_ensure_columns)


async def init_db() -> None:
    engine = get_engine()
    added: list[str] = []
    async with engine.begin() as conn:
        if engine.url.get_backend_name() == "sqlite":
            await conn.exec_driver_sql("PRAGMA journal_mode=WAL")
            await conn.exec_driver_sql("PRAGMA foreign_keys=ON")
        await conn.run_sync(Base.metadata.create_all)
        added = await conn.run_sync(_ensure_columns)
    if added:
        # 老库平滑升级：补列成功即可继续用，历史向量另由 scripts/backfill_embed_space.py 标空间
        logging.getLogger(__name__).info("已为现有库补齐列: %s", ", ".join(added))


async def session_scope() -> AsyncIterator[AsyncSession]:
    async with get_session_factory()() as s:
        yield s


async def get_db() -> AsyncIterator[AsyncSession]:
    async with get_session_factory()() as s:
        yield s


async def upsert_user(session: AsyncSession, name: str, role: str = "employee") -> User:
    u = (await session.execute(sa.select(User).where(User.name == name))).scalar_one_or_none()
    if u is None:
        u = User(name=name, role=role)
        session.add(u)
        await session.commit()
    return u


async def count_documents(session: AsyncSession) -> int:
    return (await session.execute(sa.select(func.count(Document.id)))).scalar_one()