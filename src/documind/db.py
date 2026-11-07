"""数据模型与连接。

dev = SQLite（aiosqlite），prod = PostgreSQL + pgvector，同一套 ORM。
向量存 float32 blob：dev 后端在进程内用 numpy 暴力搜索（万级 chunk 下 <10ms），
prod 换 pgvector 时**不是**"检索代码不动"：`chunks.embedding` 列类型、`IndexStore` 的
`np.vstack` 组装、`VectorStore` 实现的 `<=>` 查询三处都要改（端口见 index.KeywordStore/VectorStore，
切换清单见 docs/ARCH-REVIEW.md §4）。
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
    # 解析期落库的章节树（`ParsedDocument.outline()` 的 JSON 数组，P1-08）。
    # `/documents/{id}/outline` 只读这一列，不再从 `strategy == "structure"` 的块反推——
    # 否则用非 structure 策略入库的文档拿不到章节树，而"章节树"这个语义会有两份实现。
    # `''` = 迁移前的历史文档（这一列是后加的，没有回填）；`'[]'` = 解析后确实没有章节标记。
    outline: Mapped[str] = mapped_column(Text, default="[]")
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


class IndexState(Base):
    """索引版本（单行，key='index'）。

    P1-04：`IndexStore` 是进程内缓存，进程 A 入库/删块后进程 B 永远不刷新。
    修法是把"库变了"这件事变成库里的一个事实：写路径在同一事务里 `version+1`，
    读者每请求比一次（O(1) 单行读）。读者不缓存这个值以外的任何东西。

    `IndexStore` 还会叠加 row-count 指纹（见 `retrieve.read_index_fingerprint`），
    所以绕过 ORM 的裸 SQL 写入（迁移脚本、评测夹具）也能被发现。
    """

    __tablename__ = "index_state"

    key: Mapped[str] = mapped_column(String(32), primary_key=True)
    version: Mapped[int] = mapped_column(default=0)


INDEX_STATE_KEY = "index"


async def bump_index_version(session: AsyncSession) -> int:
    """在写事务里调用：让所有进程的索引缓存失效。返回 bump 后的版本号。

    必须在数据变更的**同一个事务**里调用（pipeline 在 commit 前调），
    否则读者可能看到新版本号但读不到新数据。

    首写用 `INSERT ... ON CONFLICT(key) DO UPDATE`，不写成「先 UPDATE、0 行再 INSERT」：
    后者在两个进程同时做第一次 bump 时会撞主键（IntegrityError），而这条路径正好发生在
    「部署后两个实例同时入库」这种最容易撞的时刻。SQLite ≥3.24 与 PostgreSQL 都支持 upsert。
    """
    values = {"key": INDEX_STATE_KEY, "version": 1}
    dialect = session.get_bind().dialect.name
    if dialect == "sqlite":
        from sqlalchemy.dialects.sqlite import insert as dialect_insert
    elif dialect == "postgresql":
        from sqlalchemy.dialects.postgresql import insert as dialect_insert
    else:  # pragma: no cover - 未支持的方言退回旧行为（单写者假设）
        res = await session.execute(
            sa.update(IndexState)
            .where(IndexState.key == INDEX_STATE_KEY)
            .values(version=IndexState.version + 1)
        )
        if not res.rowcount:
            session.add(IndexState(key=INDEX_STATE_KEY, version=1))
            await session.flush()
        row = await session.execute(
            sa.select(IndexState.version).where(IndexState.key == INDEX_STATE_KEY)
        )
        return int(row.scalar_one())

    stmt = dialect_insert(IndexState).values(**values)
    stmt = stmt.on_conflict_do_update(
        index_elements=[IndexState.key],
        set_={"version": IndexState.version + 1},
    )
    await session.execute(stmt)
    row = await session.execute(
        sa.select(IndexState.version).where(IndexState.key == INDEX_STATE_KEY)
    )
    return int(row.scalar_one())


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


async def load_page_texts(session: AsyncSession, doc_ids: set[str] | str) -> dict[tuple[str, int], str]:
    """一次取回相关文档的页文本，供引用定位（key = (doc_id, 页码)）。

    公开函数（P1-09）：`evals` 与 `pipeline` 都走它。放在 db 层而不是 pipeline 里，
    是因为它只查 `pages` 表、不含编排逻辑——评测链路要复用一段查询，不该去 import
    编排模块的私有名。
    """
    if isinstance(doc_ids, str):
        doc_ids = {doc_ids}
    if not doc_ids:
        return {}
    rows = (
        await session.execute(sa.select(Page.doc_id, Page.page_no, Page.text).where(Page.doc_id.in_(doc_ids)))
    ).all()
    return {(doc_id, no): text for doc_id, no, text in rows}