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
from .db import EvalRun

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
        # created_at 可能是 NULL（老库行 / 手工迁移）——不能让它把 /health 拖成 500，
        # 时间只是说明性字段。
        "at": row.created_at.isoformat() if row.created_at else "",
        "dataset": row.dataset or "",
    }


async def _latest(session: AsyncSession, *where: sa.ColumnElement[bool]) -> EvalRun | None:
    stmt = (
        sa.select(EvalRun)
        .where(*where, EvalRun.recommended_threshold.is_not(None))
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


async def calibrated_spaces(session: AsyncSession, reranker: str) -> list[str]:
    """该 reranker 已知有标定的向量空间（不含老库空空间行）。"""
    stmt = (
        sa.select(EvalRun.embed_space)
        .where(
            EvalRun.reranker == reranker,
            EvalRun.recommended_threshold.is_not(None),
            EvalRun.embed_space != LEGACY_SPACE,
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
    return f"value={payload['value']:.4f}（eval_run={payload['run']}，数据集 {ds}，{at}）"


async def resolve_refusal_threshold(
    session: AsyncSession, reranker: str, embed_space: str | None = None
) -> ThresholdState:
    """解析当前**实际生效**的拒答阈值，并解释它与标定值的关系。

    桶键 = `(reranker, embed_space)`。三种落回 builtin 的情形各有专门字段与文案：
    - 当前空间无标定、别的空间有 → `other_space_calibration` + 警告（值**不生效**）；
    - 老库空空间行 → `legacy_calibration` + 警告（值**不生效**，给回填命令）；
    - 完全没标定 → 警告提示去跑标定。

    `embed_space=None` 时取当前 embedder 的空间；拿不到就是空串（未知空间，不匹配任何标定）。
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

        state = ThresholdState(
            reranker=reranker,
            value=value,
            source="builtin",
            embed_space=space,
            calibrated_spaces=spaces,
            other_space_calibration=other,
            legacy_calibration=legacy,
            warning=warning,
            reason=(
                f"{bucket} 没有任何可生效的标定值（该空间无标定"
                + ("、其它空间有但按空间隔离不用" if other is not None else "")
                + ("、仅有无空间标注的历史行不用" if legacy is not None else "")
                + f"），故使用内置占位 {value:.4f}"
            ),
        )

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