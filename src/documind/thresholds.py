"""拒答阈值：按 reranker 分桶、标定回写、启动/检索时解析（P1-03）。

之前的实情：`config.py` 一个 `-6.0` 被两个 reranker 共用，`evals/runner.py` 标定完
只把建议写进报告，代码里**没有任何写入路径**——ADR-0004 说的"由评测集标定"
在运行时并不成立。

这里把阈值变成一个可追溯的对象：
1. 桶的键是 `reranker.name`（`lexical` / `bge-reranker-base` / 未来的任意名字）。
   两个后端的分数域不同（cross-encoder 出 logits，LexicalReranker 是 [-12,0]），
   共用一个数值等于给两个域各拍一次脑袋。
2. 优先级：`DOCUMIND_REFUSAL_THRESHOLD`（显式覆盖） > 最近一次标定值 > 内置占位值。
   覆盖值与标定值不一致时**打警告并在 /health 暴露**——静默不一致就是下一次事故。
3. 标定结果写回 `EvalRun.recommended_threshold`（按 reranker 行），
   所以"最近一次标定"是数据库里的事实，不是报告里的字符串。
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
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

log = logging.getLogger(__name__)


@dataclass(slots=True)
class ThresholdState:
    """一次阈值解析的完整结果——足以回答"这个数从哪来的"。"""

    reranker: str
    value: float
    source: str  # env | calibrated | builtin
    calibrated: float | None = None
    calibrated_run: str = ""
    calibrated_at: str = ""
    mismatch: bool = False
    warning: str = ""

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
    """同步兜底：没有 session 时用（环境变量 > 内置占位）。"""
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


async def latest_calibration(session: AsyncSession, reranker: str) -> tuple[float, str, str] | None:
    """该 reranker 最近一次标定：(阈值, run id, 时间字符串)。"""
    row = (
        (
            await session.execute(
                sa.select(EvalRun)
                .where(EvalRun.reranker == reranker, EvalRun.recommended_threshold.is_not(None))
                .order_by(EvalRun.created_at.desc())
                .limit(1)
            )
        )
        .scalars()
        .first()
    )
    if row is None or row.recommended_threshold is None:
        return None
    # created_at 可能是 NULL（老库行 / 手工迁移 / 老版本 default 未落库）——
    # 不能让它把整个 /health 拖成 500，时间只是说明性字段。
    created_at = row.created_at.isoformat() if row.created_at else ""
    return float(row.recommended_threshold), row.id, created_at


async def resolve_refusal_threshold(session: AsyncSession, reranker: str) -> ThresholdState:
    """解析当前**实际生效**的拒答阈值，并解释它与标定值的关系。

    警告只在真的不一致时打一次，且写进返回对象（/health 直接透出），
    因为"日志里说过"等于没人知道。
    """
    cal = await latest_calibration(session, reranker)
    calibrated, run_id, cal_at = cal if cal else (None, "", "")
    override = env_override()

    if override is not None:
        state = ThresholdState(
            reranker=reranker,
            value=override,
            source="env",
            calibrated=calibrated,
            calibrated_run=run_id,
            calibrated_at=cal_at,
        )
        if calibrated is not None and abs(override - calibrated) > 1e-9:
            state.mismatch = True
            state.warning = (
                f"拒答阈值被 DOCUMIND_REFUSAL_THRESHOLD={override:.4f} 覆盖，"
                f"与 reranker={reranker} 最近一次标定值 {calibrated:.4f}"
                f"（eval_run={run_id} @ {cal_at}）不一致；实际生效 {override:.4f}"
            )
    elif calibrated is not None:
        state = ThresholdState(
            reranker=reranker,
            value=calibrated,
            source="calibrated",
            calibrated=calibrated,
            calibrated_run=run_id,
            calibrated_at=cal_at,
        )
    else:
        value = builtin_threshold(reranker)
        state = ThresholdState(
            reranker=reranker,
            value=value,
            source="builtin",
            warning=(
                f"reranker={reranker} 还没有标定值，使用内置占位 {value:.4f}；"
                "跑一次 `documind eval` 才有可追溯的阈值"
            ),
        )

    if state.warning:
        log.warning(state.warning)
    return state


def describe_calibration(reranker: str, recommended: dict, n_rows: int) -> dict:
    """把标定结果整理成可写回的行字段（runner 用）。"""
    return {
        "reranker": reranker,
        "recommended_threshold": float(recommended["threshold"]),
        "n_cases": n_rows,
    }