"""FastAPI 应用：文档入库、问答（含 SSE）、证据定位、评测看板、trace。"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from pathlib import Path

import sqlalchemy as sa
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from .config import get_settings
from .db import Document, EvalRun, Page, User, get_db, get_session_factory, init_db
from .embed import EmbeddingSpaceMismatch, embedder_health
from .parsing import EmptyDocumentError
from .pipeline import ask, ingest_document, stream_answer
from .retrieve import read_index_fingerprint
from .thresholds import ThresholdState, resolve_refusal_threshold

app = FastAPI(title="DocuMind", version="0.1.0", description="带证据链与拒答能力的企业知识库问答")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


async def _threshold_state(db: AsyncSession, embed_space: str | None = None) -> ThresholdState:
    """生效拒答阈值 + 它的来源。`reranker_name()` 不触发模型加载，/health 保持廉价。

    桶键是 `(reranker, embed_space)`（P1-10）：`/health` 已经算过 `embedder_health()`，
    空间直接传进来，不让阈值解析自己再问一遍。
    """
    from .rerank import reranker_name

    return await resolve_refusal_threshold(db, reranker_name(), embed_space)


@app.on_event("startup")
async def _startup() -> None:
    get_settings().ensure_dirs()
    await init_db()
    # 启动即解析一次阈值：标定值与环境变量不一致时，警告必须出现在服务启动日志里（P1-03）
    async with get_session_factory()() as session:
        emb = embedder_health()
        st = await _threshold_state(session, emb["space"])
    logging.getLogger("documind.thresholds").info(
        "拒答阈值生效: %.4f（reranker=%s，向量空间=%s，来源=%s，标定值=%s）",
        st.value,
        st.reranker,
        st.embed_space or "（未知）",
        st.source,
        st.calibrated,
    )


class QueryIn(BaseModel):
    question: str
    conversation_id: str | None = None
    strategy: str = "structure"
    top_k: int | None = None
    selfcheck: bool | None = None


class UserIn(BaseModel):
    name: str
    role: str = "employee"


class EvalIn(BaseModel):
    dataset: str = "default"
    strategies: list[str] = Field(default_factory=lambda: ["fixed", "semantic", "structure", "parent_child"])
    limit: int | None = None


async def current_user(
    authorization: str = Header(default=""),
    x_user: str = Header(default=""),
    db: AsyncSession = Depends(get_db),
) -> User:
    """身份只认签名令牌；dev 环境额外允许裸 X-User（不建号、角色以库为准）。

    生产（env=prod）下裸头直接 401 —— 否则一条 curl 就能冒充 admin（见 docs/ARCH-REVIEW.md P0-01）。
    """
    from .auth import dev_header_allowed, verify

    name, role_from_token = "", None
    if authorization.lower().startswith("bearer "):
        got = verify(authorization[7:].strip())
        if got is None:
            raise HTTPException(status_code=401, detail="令牌无效或已过期")
        name, role_from_token = got
    elif x_user and dev_header_allowed():
        name = x_user
    else:
        raise HTTPException(status_code=401, detail="缺少身份令牌（Authorization: Bearer ...）")

    u = (await db.execute(sa.select(User).where(User.name == name))).scalar_one_or_none()
    if u is None:
        raise HTTPException(status_code=401, detail="用户不存在")
    if role_from_token:
        u.role = role_from_token  # 令牌里的角色是签发时的事实，库里同步
    return u


def require_admin(x_admin_token: str = Header(default="")) -> None:
    if x_admin_token != get_settings().admin_token:
        raise HTTPException(status_code=401, detail="需要管理员令牌")


@app.get("/health")
async def health(db: AsyncSession = Depends(get_db)) -> dict:
    emb = embedder_health()
    thr = await _threshold_state(db, emb["space"])
    # 索引版本进 /health（P1-04）：它是**库里的**版本，不是本进程计数器，
    # 所以「进程 A 入库、进程 B 的 /health 还是旧值」这种事故在这里一眼可见。
    fp = await read_index_fingerprint(db)
    return {
        "status": "ok",
        "documents": fp.documents,
        "chunks": fp.chunks,
        "embedder": emb["name"],
        "embed_dim": emb["dim"],
        "embed_space": emb["space"],
        "embedder_backend": emb["backend"],
        # 降级必须是可观测的：bge→hash 只能在这里被看到（P0-02）
        "degraded": ["embedder"] if emb["degraded"] else [],
        "degraded_reason": emb["degraded_reason"],
        "index_version": fp.generation,
        # 指纹的其余两项：绕过写路径的裸 SQL 变更也能从 /health 看出来
        "index_fingerprint": {"documents": fp.documents, "chunks": fp.chunks, "tokens": fp.tokens},
        "llm": {"base_url": get_settings().llm_base_url, "model": get_settings().llm_model},
        # 实际生效的拒答阈值与它的来源（P1-03）：mismatch=true 表示 env 覆盖与标定值打架
        "refusal_threshold": thr.to_dict(),
        # 真流式的能力声明：客户端可据此判断 `/query/stream` 是否会先发 evidence（P1-05）
        "streaming": {"endpoint": "/query/stream", "events": ["evidence", "token", "answer"]},
    }


@app.exception_handler(EmptyDocumentError)
async def _empty_document_handler(_: Request, exc: EmptyDocumentError) -> JSONResponse:
    """空文档不是 500 也不是 200：给前端一个能直接展示的中文错误（P0-04）。"""
    return JSONResponse(
        status_code=422,
        content={"detail": {"code": "empty_document", "message": str(exc)}},
    )


@app.exception_handler(EmbeddingSpaceMismatch)
async def _space_mismatch_handler(_: Request, exc: EmbeddingSpaceMismatch) -> JSONResponse:
    """向量空间不一致是服务端配置/数据问题，不是查询写错了：409 + 可读修复建议（P0-02）。"""
    return JSONResponse(
        status_code=409,
        content={"detail": {"code": "embedding_space_mismatch", "message": str(exc)}},
    )


@app.post("/documents")
async def upload_document(
    file: UploadFile = File(...),
    acl_roles: str = Form(default='["all"]'),
    title: str | None = Form(default=None),
    _: None = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
) -> dict:
    suffix = Path(file.filename or "doc.pdf").suffix
    fd, tmp_name = tempfile.mkstemp(suffix=suffix, dir=str(get_settings().data_dir))
    os.close(fd)
    tmp = Path(tmp_name)
    tmp.write_bytes(await file.read())
    try:
        roles = json.loads(acl_roles)
    except json.JSONDecodeError as exc:
        raise HTTPException(400, f"acl_roles 不是合法 JSON: {acl_roles}") from exc
    try:
        return await ingest_document(db, tmp, acl_roles=roles, title=title)
    except EmptyDocumentError as exc:
        # 422 而不是 200 空库：扫描件/图片型 PDF 必须让用户看见"需要 OCR"
        raise HTTPException(
            status_code=422,
            detail={"code": "empty_document", "message": str(exc)},
        ) from exc
    finally:
        # Windows 上文件可能还被读句柄占着，删不掉不该让请求失败
        for _ in range(3):
            try:
                tmp.unlink(missing_ok=True)
                break
            except OSError:
                time.sleep(0.05)


@app.get("/documents")
async def list_documents(user: User = Depends(current_user), db: AsyncSession = Depends(get_db)) -> list[dict]:
    rows = (await db.execute(sa.select(Document).order_by(Document.created_at.desc()))).scalars().all()
    out = []
    for d in rows:
        roles = d.roles()
        visible = user.role == "admin" or "all" in roles or user.role in roles
        out.append(
            {
                "id": d.id,
                "title": d.title,
                "filename": d.filename,
                "pages": d.page_count,
                "acl_roles": roles,
                "visible": visible,
                "created_at": d.created_at.isoformat(),
            }
        )
    return out


async def _visible_doc(db: AsyncSession, doc_id: str, user: User) -> Document:
    d = (await db.execute(sa.select(Document).where(Document.id == doc_id))).scalar_one_or_none()
    if d is None:
        raise HTTPException(404, "文档不存在")
    roles = d.roles()
    if not (user.role == "admin" or "all" in roles or user.role in roles):
        # 无权访问不返回 403 细节，统一 404：不泄露"存在哪些文档"
        raise HTTPException(404, "文档不存在")
    return d


@app.get("/documents/{doc_id}/pages/{page_no}")
async def get_page(
    doc_id: str, page_no: int, user: User = Depends(current_user), db: AsyncSession = Depends(get_db)
) -> dict:
    await _visible_doc(db, doc_id, user)
    p = (
        await db.execute(sa.select(Page).where(Page.doc_id == doc_id, Page.page_no == page_no))
    ).scalar_one_or_none()
    if p is None:
        raise HTTPException(404, "页码超出范围")
    return {"doc_id": doc_id, "page": page_no, "text": p.text}


def _flatten_outline(nodes: list[dict], parent_path: str = "") -> list[dict]:
    """把落库的章节树摊平成 `章节路径 → 页码` 列表（保持 `/outline` 既有字段语义）。

    老响应里的 `sections[].title` 就是 `chunk.section_path`（`第一章 总则 > 第一节 ...`
    这种用 ` > ` 拼起来的字符串），前端拿它当章节路径显示。这里仍产出同样的字符串，
    只是数据来源换成解析期落库的章节树，而不是按 structure 块的 section_path 去重。
    """
    flat: list[dict] = []
    for node in nodes:
        label = str(node.get("title") or "").strip()
        if not label:
            continue
        path = f"{parent_path} > {label}" if parent_path else label
        flat.append(
            {
                "title": path,  # 兼容旧字段：章节路径
                "page": int(node.get("page") or 1),
                "level": int(node.get("level") or 1),
                "heading": label,  # 标题本身（不含父路径）
            }
        )
        flat.extend(_flatten_outline(node.get("children") or [], path))
    return flat


@app.get("/documents/{doc_id}/outline")
async def get_outline(
    doc_id: str, user: User = Depends(current_user), db: AsyncSession = Depends(get_db)
) -> dict:
    """章节树只读 `documents.outline`（解析期由 `ParsedDocument.outline()` 落库，P1-08）。

    这里**不再**从 `strategy == "structure"` 的块反推：那样一份用 fixed/semantic 策略入库的
    文档会返回空章节树，而它明明有章节——数据库里已经存了权威版本，没有理由再猜一次。
    """
    doc = await _visible_doc(db, doc_id, user)
    raw = (doc.outline or "").strip()
    tree: list[dict] = []
    if not raw:
        # 这一列是后加的：迁移前的历史文档没有章节树落库。不退回"从块反推"（那会让
        # 两份定义重新长出来），而是显式告诉调用方需要重新入库。
        source = "missing"
    else:
        try:
            loaded = json.loads(raw)
        except json.JSONDecodeError:
            logging.getLogger(__name__).warning("documents.outline 不是合法 JSON: doc_id=%s", doc_id)
            source = "invalid"
        else:
            tree = loaded if isinstance(loaded, list) else []
            source = "parsed"
    return {
        "doc_id": doc_id,
        "sections": _flatten_outline(tree),
        "tree": tree,
        "outline_source": source,
    }


@app.post("/query")
async def query(payload: QueryIn, user: User = Depends(current_user), db: AsyncSession = Depends(get_db)) -> dict:
    ans = await ask(
        db,
        payload.question,
        role=user.role,
        user_id=user.id,
        conversation_id=payload.conversation_id,
        strategy=payload.strategy,
        top_k=payload.top_k,
        selfcheck=payload.selfcheck,
    )
    return ans.to_dict()


@app.get("/query/stream")
async def query_stream(
    question: str = Query(...),
    conversation_id: str | None = None,
    strategy: str = "structure",
    selfcheck: bool | None = None,
    user: User = Depends(current_user),
    db: AsyncSession = Depends(get_db),
) -> StreamingResponse:
    """SSE：`evidence`（检索完即发）→ 0..n × `token`（逐块）→ `answer`（权威）。

    向后兼容：事件名与字段沿用 v0.1（`evidence` 是 Hit 数组，`answer` 是完整 Answer），
    `token` 是新增事件，旧客户端解析到未知事件时忽略即可，时序与载荷都不变（P1-05）。
    """

    async def gen():
        async for ev in stream_answer(
            db,
            question=question,
            role=user.role,
            user_id=user.id,
            conversation_id=conversation_id,
            strategy=strategy,
            selfcheck=selfcheck,
        ):
            yield f"event: {ev['event']}\ndata: {json.dumps(ev['data'], ensure_ascii=False)}\n\n"

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={
            # 三层缓冲都要关掉，否则"真流式"会在代理层被重新攒成一次性响应
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


class LoginIn(BaseModel):
    name: str
    role: str | None = None


@app.post("/auth/login")
async def login(payload: LoginIn, db: AsyncSession = Depends(get_db)) -> dict:
    """签发签名令牌。dev 里按用户库发；生产应替换为 IdP 回调（见 auth.py 顶部注释）。"""
    from .auth import sign

    u = (await db.execute(sa.select(User).where(User.name == payload.name))).scalar_one_or_none()
    if u is None:
        raise HTTPException(status_code=401, detail="用户不存在")
    if payload.role and payload.role != u.role:
        raise HTTPException(status_code=403, detail="不能自行声明角色")
    token = sign(u.name, u.role)
    return {"token": token, "role": u.role, "expires_in_s": 12 * 3600}


@app.get("/users")
async def list_users(_: None = Depends(require_admin), db: AsyncSession = Depends(get_db)) -> list[dict]:
    rows = (await db.execute(sa.select(User))).scalars().all()
    return [{"name": u.name, "role": u.role} for u in rows]


@app.post("/users")
async def create_user(payload: UserIn, _: None = Depends(require_admin), db: AsyncSession = Depends(get_db)) -> dict:
    u = (await db.execute(sa.select(User).where(User.name == payload.name))).scalar_one_or_none()
    if u is None:
        u = User(name=payload.name, role=payload.role)
        db.add(u)
    else:
        u.role = payload.role
    await db.commit()
    return {"name": u.name, "role": u.role}


@app.post("/evals/run")
async def run_eval(payload: EvalIn, _: None = Depends(require_admin), db: AsyncSession = Depends(get_db)) -> dict:
    from .evals.runner import run_eval as _run

    report = await _run(db, dataset=payload.dataset, strategies=payload.strategies, limit=payload.limit)
    return report


@app.get("/evals/runs")
async def eval_runs(db: AsyncSession = Depends(get_db)) -> list[dict]:
    rows = (await db.execute(sa.select(EvalRun).order_by(EvalRun.created_at.desc()).limit(30))).scalars().all()
    return [
        {
            "id": r.id,
            "strategy": r.strategy,
            "dataset": r.dataset,
            "n_cases": r.n_cases,
                        "metrics": json.loads(r.metrics),
                        "created_at": r.created_at.isoformat(),
                        "report_path": r.report_path,
                        # 标定回写：阈值现在是可查询的事实，而不是报告里的一段文字（P1-03）
                        "reranker": r.reranker,
                        # 空间是阈值桶键的另一半（P1-10）：空串 = 迁移前的历史行（不生效）
                        "embed_space": r.embed_space or "",
                        "recommended_threshold": r.recommended_threshold,
                    }
        for r in rows
    ]


@app.get("/evals/report", response_class=HTMLResponse)
async def eval_report() -> HTMLResponse:
    from .evals.report import REPORTS

    path = REPORTS / "ablation.html"
    if not path.exists():
        raise HTTPException(404, "还没有评测报告，先跑一次 POST /evals/run")
    return HTMLResponse(path.read_text(encoding="utf-8"))


@app.get("/stats")
async def stats(db: AsyncSession = Depends(get_db)) -> dict:
    from .db import Message

    rows = (
        await db.execute(
            sa.select(
                sa.func.count(Message.id),
                sa.func.avg(Message.latency_ms),
                sa.func.sum(Message.cost_usd),
                sa.func.sum(sa.case((Message.refused.is_(True), 1), else_=0)),
            ).where(Message.role == "assistant")
        )
    ).one()
    total, avg_lat, cost, refused = rows
    return {
        "answers": total or 0,
        "avg_latency_ms": round(float(avg_lat or 0), 1),
        "total_cost_usd": round(float(cost or 0), 6),
        "refusal_rate": round((refused or 0) / total, 3) if total else 0.0,
    }
