"""拒答阈值：按 `(reranker, embed_space)` 分桶、标定回写、启动/检索时解析（P1-03 / P1-10）。

之前的实情：`config.py` 一个 `-6.0` 被两个 reranker 共用，`evals/runner.py` 标定完
只把建议写进报告，代码里**没有任何写入路径**——ADR-0004 说的"由评测集标定"
在运行时并不成立。

这里把阈值变成一个可追溯的对象：
1. 桶的键是 `(reranker.name, embed_space)`。两个后端的分数域不同（cross-encoder 出 logits，
   LexicalReranker 是 [-12,0]）；更重要的是**同一套 reranker 换一份语料/一个向量空间，
   分数分布就整体平移**——2026-11-08 的交叉验证：同一条 1.5055 在 suhao 上"可答最低分 1.5759、
   0 误拒"，在美诺华上"可答最低分 0.0706、2/26 误拒"，同域难例只拒 10/24。
   只按 reranker 分桶会把这份差异静默继承下来，所以空间必须进桶键。
2. 优先级：`DOCUMIND_REFUSAL_THRESHOLD`（显式覆盖） > 当前空间的最近标定值 > 内置占位值。
   覆盖值与标定值不一致时**打警告并在 /health 暴露**——静默不一致就是下一次事故。
3. 标定结果写回 `EvalRun.recommended_threshold` + `EvalRun.embed_space`，
   所以"最近一次标定"是数据库里的事实，不是报告里的字符串。
4. 老库（迁移前的行）`embed_space=''`：**只告警、不生效**，按"历史通用值"处理。
   理由见 `resolve_refusal_threshold`：一条无空间标注的行无法证明它属于哪个空间，
   猜错就等于重现"跨库继承"这个事故。回填路径是显式的：`scripts/backfill_calibration_space.py`。
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Final

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncSession

from .config import get_settings
from .db import Chunk as ChunkRow
from .db import Document, EvalRun

# 标定样本量门槛：一份 5~10 题的"标定"不是标定，它标出的是噪声的形状。
# Recall@5 在 n=20 时的粒度就是 5 个百分点（见 README），阈值取 20 是同一个量级的下界：
# 低于它，P-R 曲线上相邻两个候选点的差别只由 1 道题决定，写回库后再被当成"标定值"读出来，
# 就等于把一个偶然数变成了线上拒答行为——这正是 P1-03 要消灭的那类"看起来有据可依"。
# 门槛对**读**也生效：任何写入路径（含裸 SQL）留下的小样本行都不许被解析成 calibrated。
MIN_CALIBRATION_CASES: Final[int] = 20

# 从未标定过时的占位值。它们**不是**标定结论：
# /health 会带 `calibrated: false` + 警告，避免把占位值当成标定值（这正是 P1-03 的病根）。
BUILTIN_THRESHOLDS: Final[dict[str, float]] = {
    "lexical": -6.0,  # 词面域 [-12, 0] 的中段
    "BAAI/bge-reranker-base": -6.0,  # cross-encoder logits 的保守下界（settings.rerank_model 的默认值）
    "bge-reranker-base": -6.0,  # 不带 org 前缀时的同一模型
}
UNKNOWN_RERANKER_FALLBACK: Final[float] = -6.0

# 老库行的空间标注（迁移前的行补列后就是这个值）。
LEGACY_SPACE: Final[str] = ""

log = logging.getLogger(__name__)


@dataclass(slots=True)
class ThresholdState:
    """一次阈值解析的完整结果——足以回答"这个数从哪来的、为什么不是别的数"。"""

    reranker: str
    value: float
    source: str  # env | calibrated | builtin
    calibrated: float | None = None
    calibrated_run: str = ""
    calibrated_at: str = ""
    # 这条标定是在哪份数据集/语料上标出来的：同空间不同语料是**已知未覆盖**的边界
    calibrated_dataset: str = ""
    mismatch: bool = False
    warning: str = ""
    # ---- P1-10 新增（纯增量字段，老字段语义不变）----
    # 解析时用的向量空间（桶键的另一半）。空串 = 解析时拿不到 embedder
    embed_space: str = ""
    # 该 reranker 已知有标定的空间（不含老库的空空间行）
    calibrated_spaces: list[str] = field(default_factory=list)
    # 其它空间上最新的一条标定：**仅供参考、未生效**
    other_space_calibration: dict | None = None
    # 无空间标注的历史标定行：**只告警、未生效**
    legacy_calibration: dict | None = None
    # 同空间但样本量不足的行（n_cases < MIN_CALIBRATION_CASES）：**provisional，未生效**
    provisional_calibration: dict | None = None
    # 为什么现在生效的是这个值（始终有值，含 source=calibrated 的正常路径）
    reason: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def calibrated_matches(self) -> bool | None:
        return None if self.calibrated is None else not self.mismatch


def env_override() -> float | None:
    """显式覆盖。`DOCUMIND_REFUSAL_THRESHOLD` 优先，旧名 `..._RERANK_THRESHOLD` 兼容。"""
    s = get_settings()
    if s.refusal_threshold is not None:
        return float(s.refusal_threshold)
    if s.refusal_rerank_threshold is not None:
        return float(s.refusal_rerank_threshold)
    return None


def builtin_threshold(reranker: str) -> float:
    return BUILTIN_THRESHOLDS.get(reranker, UNKNOWN_RERANKER_FALLBACK)


def default_threshold(reranker: str | None = None) -> float:
    """同步兜底：没有 session 时用（环境变量 > 内置占位）。

    同步路径**读不到库**，所以永远拿不到标定值——这不是降级：调用方要么给显式阈值，
    要么走 `resolve_refusal_threshold`。
    """
    if reranker is None:
        try:
            from .rerank import get_reranker

            reranker = get_reranker().name
        except Exception:  # pragma: no cover - 模型不可用不该让问答崩掉
            reranker = ""
    override = env_override()
    if override is not None:
        return override
    return builtin_threshold(reranker)


def current_embed_space() -> str:
    """当前进程的向量空间 `model@dim`；embedder 起不来时返回空串（调用方按"未知"处理）。

    `get_embedder` 是 lru_cache，检索路径已经实例化过它，这里不额外付代价。
    """
    try:
        from .embed import get_embedder

        return get_embedder().space
    except Exception as exc:  # 配置缺失/模型不可用：阈值解析不该因此 500
        log.warning("无法确定当前向量空间（%s: %s），阈值按未知空间处理", type(exc).__name__, exc)
        return ""


def _calibration_payload(row: EvalRun) -> dict:
    return {
        "embed_space": row.embed_space or LEGACY_SPACE,
        "value": float(row.recommended_threshold),  # type: ignore[arg-type]
        "run": row.id,
        # 这个值是几道题标出来的：小于 MIN_CALIBRATION_CASES 的行只配当 provisional
        "n_cases": int(row.n_cases or 0),
        # created_at 可能是 NULL（老库行 / 手工迁移）——不能让它把 /health 拖成 500，
        # 时间只是说明性字段。
        "at": row.created_at.isoformat() if row.created_at else "",
        "dataset": row.dataset or "",
    }


async def _latest(
    session: AsyncSession,
    *where: sa.ColumnElement[bool],
    min_cases: int = MIN_CALIBRATION_CASES,
    max_cases: int | None = None,
) -> EvalRun | None:
    """桶内最近一条**够样本量**的标定行。

    `n_cases` 门槛放在 SQL 里而不是 Python 里过滤：否则"最近一条是小样本行"会让
    查询先返回它、再被丢弃，结果变成"有标定却解析不到"，而不是"回退到上一条正式标定"。
    """
    conds: list[sa.ColumnElement[bool]] = [
        *where,
        EvalRun.recommended_threshold.is_not(None),
        EvalRun.n_cases >= min_cases,
    ]
    if max_cases is not None:
        conds.append(EvalRun.n_cases <= max_cases)
    stmt = (
        sa.select(EvalRun)
        .where(*conds)
        .order_by(EvalRun.created_at.desc())
        .limit(1)
    )
    return (await session.execute(stmt)).scalars().first()


async def latest_calibration(session: AsyncSession, reranker: str, embed_space: str) -> dict | None:
    """`(reranker, embed_space)` 桶里最近一次标定。

    空间为空串时**不查**：空串是"老库行/未知空间"的哨兵，不是可用的桶键。
    """
    if not embed_space:
        return None
    row = await _latest(
        session,
        EvalRun.reranker == reranker,
        EvalRun.embed_space == embed_space,
    )
    return _calibration_payload(row) if row is not None else None


async def provisional_calibration(
    session: AsyncSession, reranker: str, embed_space: str
) -> dict | None:
    """同空间、但样本量不足（n_cases < MIN）的最近一条：只用于解释"为什么它没生效"。"""
    if not embed_space:
        return None
    row = await _latest(
        session,
        EvalRun.reranker == reranker,
        EvalRun.embed_space == embed_space,
        min_cases=0,
        max_cases=MIN_CALIBRATION_CASES - 1,
    )
    return _calibration_payload(row) if row is not None else None


async def calibrated_spaces(session: AsyncSession, reranker: str) -> list[str]:
    """该 reranker 已知有**合格**标定的向量空间（不含老库空空间行、不含 provisional）。"""
    stmt = (
        sa.select(EvalRun.embed_space)
        .where(
            EvalRun.reranker == reranker,
            EvalRun.recommended_threshold.is_not(None),
            EvalRun.embed_space != LEGACY_SPACE,
            EvalRun.n_cases >= MIN_CALIBRATION_CASES,
        )
        .distinct()
        .order_by(EvalRun.embed_space)
    )
    return [s for s in (await session.execute(stmt)).scalars().all() if s]


async def other_space_calibration(
    session: AsyncSession, reranker: str, embed_space: str
) -> dict | None:
    """别的空间上最新的一条标定：只用于在 /health 里说明"有这个值，但没生效"。"""
    row = await _latest(
        session,
        EvalRun.reranker == reranker,
        EvalRun.embed_space != LEGACY_SPACE,
        # 未知空间时不排除任何空间：反正都用不上，宁可把候选值报出来
        EvalRun.embed_space != (embed_space or "\x00"),
    )
    return _calibration_payload(row) if row is not None else None


async def legacy_calibration(session: AsyncSession, reranker: str) -> dict | None:
    """无空间标注的历史标定行（迁移前写入）。只告警、不生效。"""
    row = await _latest(
        session,
        EvalRun.reranker == reranker,
        EvalRun.embed_space == LEGACY_SPACE,
    )
    return _calibration_payload(row) if row is not None else None


def _fmt(payload: dict) -> str:
    at = payload.get("at") or "时间未知"
    ds = payload.get("dataset") or "数据集未知"
    n = payload.get("n_cases")
    n_txt = f"，n_cases={n}" if n is not None else ""
    return f"value={payload['value']:.4f}（eval_run={payload['run']}，数据集 {ds}{n_txt}，{at}）"


def calibration_gate(n_cases: int) -> tuple[bool, str]:
    """`recommended_threshold` 能不能被当成正式标定写回/生效。

    选的是**写回前拒绝**，而不是"写一行标 provisional=true"：
    - 读侧已经有硬门槛（`_latest`），provisional 行永远不会生效，写进去只会在
      `/evals/runs` 里多出一个"看起来像标定、其实不是"的数字——那正是 P1-10 legacy 行的病根；
    - 拒绝发生在操作者眼前（终端 + 退出码 2），比在数据库里留一条需要再解释的行更容易纠正：
      加题重跑就行，不需要先清理脏行。
    返回 (ok, reason)；ok=False 时 reason 是给人看的说明。
    """
    if n_cases >= MIN_CALIBRATION_CASES:
        return True, ""
    return False, (
        f"样本量不足：n_cases={n_cases} < {MIN_CALIBRATION_CASES}。"
        f"这个推荐阈值只是 provisional（相邻候选点的差别由 1 道题决定），不写回、不生效；"
        f"用 --ans-limit/--unans-limit 或 --limit 凑到 ≥{MIN_CALIBRATION_CASES} 题再跑 --write"
    )


async def library_embed_spaces(session: AsyncSession) -> list[dict]:
    """库里**实际存在**的向量空间（`documents.embed_space` 去重）+ 每个空间的文档/块计数。

    为什么需要它：`/health.embed_space` 一直是**进程**的空间，而"检索会不会 409"取决于
    **库**里的空间。实测踩过这种组合——库里是 `text-embedding-v3@1024`、进程默认
    `bge-small@512`——只报进程空间时，人看不出为什么 source=builtin，也看不出下一句提问就会 409。

    空串（迁移前的历史文档）单列并标 `legacy`：它们没有指纹，检索会被 fail-fast 拦下，
    和"某个后端的数据"不是一回事。
    """
    stmt = (
        sa.select(
            Document.embed_space,
            sa.func.count(sa.distinct(Document.id)),
            sa.func.count(ChunkRow.id),
        )
        .select_from(Document)
        .outerjoin(ChunkRow, ChunkRow.doc_id == Document.id)
        .group_by(Document.embed_space)
        .order_by(Document.embed_space)
    )
    out: list[dict] = []
    for space, docs, chunks in (await session.execute(stmt)).all():
        out.append(
            {
                "embed_space": space or "",
                "documents": int(docs or 0),
                "chunks": int(chunks or 0),
                "legacy": not space,
            }
        )
    return out


def space_mismatch(library: list[dict], process_space: str) -> bool:
    """进程空间不在**有数据**的库内空间里 → 检索必然 409。

    空库不算 mismatch：没有数据可对不上，409 也不会发生。进程空间未知（空串）也不在这里报——
    那是 embedder 起不来的问题，有它自己的警告。
    """
    if not process_space:
        return False
    present = {item["embed_space"] for item in library if item.get("documents")}
    return bool(present) and process_space not in present


def _mismatch_note(library: list[dict], process_space: str) -> str:
    """mismatch 时追加到 refusal_threshold.warning 的那句话（运维第一句要读到它）。"""
    present = [i for i in library if i.get("documents")]
    detail = "、".join(
        f"{i['embed_space'] or '（无空间标注的历史数据）'}（{i['documents']} 篇 / {i['chunks']} 块）"
        for i in present
    )
    return (
        f"库里没有你这个空间的数据（进程空间 {process_space}，库里的空间：{detail}），"
        "检索会 409；先入库或换回库里那个 backend"
    )


async def resolve_refusal_threshold(
    session: AsyncSession,
    reranker: str,
    embed_space: str | None = None,
    library: list[dict] | None = None,
) -> ThresholdState:
    """解析当前**实际生效**的拒答阈值，并解释它与标定值的关系。

    桶键 = `(reranker, embed_space)`。落回 builtin 的情形各有专门字段与文案：
    - 当前空间无标定、别的空间有 → `other_space_calibration` + 警告（值**不生效**）；
    - 老库空空间行 → `legacy_calibration` + 警告（值**不生效**，给回填命令）；
    - 同空间只有小样本行 → `provisional_calibration` + 警告（**不生效**，见 MIN_CALIBRATION_CASES）；
    - 完全没标定 → 警告提示去跑标定。

    `embed_space=None` 时取当前 embedder 的空间；拿不到就是空串（未知空间，不匹配任何标定）。
    `library`（`/health` 传入）非 None 时额外判定"进程空间是否在库里有数据"，
    不一致就把"检索会 409"写进警告——不改任何 source 语义，只补一句人话。
    警告只在真有问题时打日志并写进返回对象（/health 直接透出），
    因为"日志里说过"等于没人知道。
    """
    space = current_embed_space() if embed_space is None else embed_space
    cal = await latest_calibration(session, reranker, space)
    spaces = await calibrated_spaces(session, reranker)
    override = env_override()
    bucket = f"reranker={reranker} · 向量空间 {space or '（未知）'}"

    if override is not None:
        state = ThresholdState(
            reranker=reranker,
            value=override,
            source="env",
            calibrated=cal["value"] if cal else None,
            calibrated_run=cal["run"] if cal else "",
            calibrated_at=cal["at"] if cal else "",
            calibrated_dataset=cal["dataset"] if cal else "",
            embed_space=space,
            calibrated_spaces=spaces,
            reason=(
                f"DOCUMIND_REFUSAL_THRESHOLD={override:.4f} 显式覆盖（优先级最高，{bucket}）；"
                + (f"该空间最近标定值 {cal['value']:.4f} 被覆盖" if cal else "该空间没有标定值可对比")
            ),
        )
        if cal and abs(override - cal["value"]) > 1e-9:
            state.mismatch = True
            state.warning = (
                f"拒答阈值被 DOCUMIND_REFUSAL_THRESHOLD={override:.4f} 覆盖，"
                f"与 {bucket} 最近一次标定值 {cal['value']:.4f}"
                f"（eval_run={cal['run']} @ {cal['at']}）不一致；实际生效 {override:.4f}"
            )
    elif cal is not None:
        state = ThresholdState(
            reranker=reranker,
            value=cal["value"],
            source="calibrated",
            calibrated=cal["value"],
            calibrated_run=cal["run"],
            calibrated_at=cal["at"],
            calibrated_dataset=cal["dataset"],
            embed_space=space,
            calibrated_spaces=spaces,
            reason=(
                f"{bucket} 最近一次标定（eval_run={cal['run']}，数据集 "
                f"{cal['dataset'] or '未知'}，{cal['at'] or '时间未知'}）写回 {cal['value']:.4f}；"
                "同空间没有更新的标定，环境变量未设置"
            ),
        )
    else:
        value = builtin_threshold(reranker)
        legacy = await legacy_calibration(session, reranker)
        provisional = await provisional_calibration(session, reranker, space)
        other = await other_space_calibration(session, reranker, space) if space else None

        if not space:
            warning = (
                f"无法确定当前向量空间（embedder 不可用），{bucket} 的标定值无从匹配，"
                f"拒答阈值退回内置占位 {value:.4f}；库里的标定值一律不生效，"
                "修好 embedder 配置后 /health 才会给出可解释的值"
            )
        elif other is not None:
            warning = (
                f"向量空间 {space} 还没有标定值，拒答阈值退回内置占位 {value:.4f}；"
                f"已知空间 {other['embed_space']} 有标定值 {_fmt(other)}，但**未生效**——"
                "阈值不跨空间继承（不同空间的 top_score 分布整体平移，"
                "同一数值落到这里会变成另一个误拒/漏拒水平）；"
                f"要在 {space} 上生效，需在该空间重跑一次标定"
            )
        else:
            warning = (
                f"{bucket} 还没有标定值，使用内置占位 {value:.4f}；"
                "跑一次 `documind eval` 才有可追溯的阈值"
            )

        if legacy is not None:
            warning += (
                f"。另有 1 条无空间标注的历史标定行 {_fmt(legacy)}，按“历史通用值”处理："
                "只告警、不生效（无标注的行无法证明属于哪个空间，猜错就等于跨库继承）。"
                "确认它属于哪个空间后跑 "
                "`python scripts/backfill_calibration_space.py --space <model@dim>` 回填"
            )

        if provisional is not None:
            warning += (
                f"。同空间还有一条样本量不足的标定 {_fmt(provisional)}，"
                f"按 provisional 处理：n_cases={provisional['n_cases']} < {MIN_CALIBRATION_CASES}，"
                "不写回、不生效（这个量级上相邻候选点的差别由 1 道题决定）"
            )

        state = ThresholdState(
            reranker=reranker,
            value=value,
            source="builtin",
            embed_space=space,
            calibrated_spaces=spaces,
            other_space_calibration=other,
            legacy_calibration=legacy,
            provisional_calibration=provisional,
            warning=warning,
            reason=(
                f"{bucket} 没有任何可生效的标定值（该空间无标定"
                + ("、其它空间有但按空间隔离不用" if other is not None else "")
                + ("、仅有无空间标注的历史行不用" if legacy is not None else "")
                + ("、仅有过样本量不足的 provisional 行" if provisional is not None else "")
                + f"），故使用内置占位 {value:.4f}"
            ),
        )

    # 库内空间与进程空间不一致：任何 source 下都要说清"下一句提问会 409"。
    # 这段不改 source/value/calibrated 语义，只在警告里追加一句可执行的解释。
    if library is not None and space_mismatch(library, space):
        note = _mismatch_note(library, space)
        state.warning = f"{state.warning}；{note}" if state.warning else note

    if state.warning:
        log.warning(state.warning)
    return state


def describe_calibration(
    reranker: str, recommended: dict, n_rows: int, embed_space: str = ""
) -> dict:
    """把标定结果整理成可写回的行字段（runner 用）。

    `embed_space` 必须与**入库时用的向量空间**一致，否则写回的行不会在该空间生效。
    """
    return {
        "reranker": reranker,
        "embed_space": embed_space,
        "recommended_threshold": float(recommended["threshold"]),
        "n_cases": n_rows,
    }