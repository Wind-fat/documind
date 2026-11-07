"""评测执行器：一次跑多策略，产出可直接贴进 README 的报告。

用法：
    uv run documind eval --dataset default                  # 四种切块全跑
    uv run documind eval --dataset default --strategies structure --limit 20
"""

from __future__ import annotations

import asyncio
import json
import statistics
import time

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncSession

from ..config import ROOT, get_settings
from ..db import EvalRun, Page, get_session_factory
from ..embed import embedder_health
from ..generate import answer_question
from ..llm import LLMClient, get_llm
from ..retrieve import retrieve
from .metrics import aggregate, citation_validity, judge_answer, retrieval_metrics
from .report import write_reports

DATASETS = ROOT / "evals" / "datasets"
REPORTS = ROOT / "evals" / "reports"


async def _pages(session: AsyncSession, doc_ids: set[str]) -> dict[str, dict[int, str]]:
    rows = (await session.execute(sa.select(Page.doc_id, Page.page_no, Page.text).where(Page.doc_id.in_(doc_ids)))).all()
    out: dict[str, dict[int, str]] = {}
    for doc_id, no, text in rows:
        out.setdefault(doc_id, {})[no] = text
    return out


async def _one_case(case: dict, strategy: str, llm: LLMClient, selfcheck: bool) -> dict:
    from ..pipeline import _page_texts

    async with get_session_factory()() as session:
        ret = await retrieve(session, case["question"], role="admin", strategy=strategy)
        page_texts = await _page_texts(session, {h.doc_id for h in ret.hits}) if ret.hits else {}
        t0 = time.perf_counter()
        ans = await answer_question(case["question"], ret, llm, page_texts=page_texts, selfcheck=selfcheck)
        latency = int((time.perf_counter() - t0) * 1000) + ret.latency_ms
        rm = retrieval_metrics(ret, case)
        if case["answerable"]:
            judged = await judge_answer(llm, case, ans, ret)
        else:
            judged = {"faithfulness": None, "relevancy": None, "cost_usd": 0.0}
        return {
            "id": case["id"],
            "strategy": strategy,
            "question": case["question"],
            "answerable": case["answerable"],
            "refused": ans.refused,
            "answer": ans.text,
            "n_citations": len(ans.citations),
            "citation_validity": citation_validity(ans, page_texts),
            "top_score": round(ret.top_score, 3),
            "latency_ms": latency,
            "cost_usd": round(ans.cost_usd + judged.get("cost_usd", 0.0), 8),
            "faithfulness": judged.get("faithfulness"),
            "relevancy": judged.get("relevancy"),
            **rm,
        }


async def run_eval(
    session: AsyncSession,
    dataset: str = "default",
    strategies: list[str] | None = None,
    limit: int | None = None,
    concurrency: int = 4,
    selfcheck: bool = False,
) -> dict:
    from .dataset import load_dataset

    strategies = strategies or ["fixed", "semantic", "structure", "parent_child"]
    path = DATASETS / (dataset if dataset.endswith(".jsonl") else f"qa_{dataset}.jsonl")
    cases = load_dataset(path)
    if limit:
        # 保持可答案/不可答案比例，按答案组分别截断
        ans = [c for c in cases if c["answerable"]][: int(limit * 0.8)]
        unans = [c for c in cases if not c["answerable"]][: max(1, int(limit * 0.2))]
        cases = ans + unans
    llm = get_llm()
    sem = asyncio.Semaphore(concurrency)
    summary: dict[str, dict] = {}
    details: dict[str, list[dict]] = {}

    async def run_one(case: dict, strategy: str) -> dict:
        async with sem:
            return await _one_case(case, strategy, llm, selfcheck)

    for strategy in strategies:
        t0 = time.perf_counter()
        rows = await asyncio.gather(*(run_one(c, strategy) for c in cases))
        details[strategy] = list(rows)
        summary[strategy] = aggregate(list(rows))
        summary[strategy]["wall_s"] = round(time.perf_counter() - t0, 1)
        print(f"[{strategy}] faithfulness={summary[strategy]['faithfulness']} "
              f"recall@5={summary[strategy]['recall_at_5']} "
              f"refusal_acc={summary[strategy]['refusal_accuracy']} "
              f"p95={summary[strategy]['latency_p95_ms']}ms")

    calibration = calibrate_threshold([r for rows in details.values() for r in rows])
    embedder = embedder_health()
    # 每份报告都要能回答"这组数字是用哪个 embedder 跑的"：同时进报告与 EvalRun.metrics
    for strategy in summary:
        summary[strategy]["embedder"] = embedder
    report = write_reports(
        dataset=dataset, summary=summary, details=details, calibration=calibration, embedder=embedder
    )

    for strategy, m in summary.items():
        session.add(
            EvalRun(
                strategy=strategy,
                dataset=dataset,
                n_cases=m["n_cases"],
                metrics=json.dumps(m, ensure_ascii=False),
                report_path=str(report["markdown"]),
            )
        )
    await session.commit()
    _ = statistics
    _ = get_settings()
    return {"summary": summary, "calibration": calibration, **report, "details": details}


def calibrate_threshold(rows: list[dict]) -> dict:
    """用评测集画 P-R 曲线定拒答阈值。

    "检索分低于 0.6 就拒答"是拍脑袋；这里把 top_score 从低到高扫一遍，
    看每个阈值下"无答案问题被拒的比例"与"有答案问题被误拒的比例"怎么互换，
    再按业务规则（宁可多拒，不可错答）取点：误拒率 ≤ 15% 的前提下，拒答准确率最高。
    """
    scores = sorted({r["top_score"] for r in rows})
    grid = sorted(set([round(x, 2) for x in scores] + [round(-14 + 0.5 * i, 2) for i in range(29)]))
    curve = []
    for th in grid:
        unans = [r for r in rows if not r["answerable"]]
        ans = [r for r in rows if r["answerable"]]
        refused_unans = sum(1 for r in unans if r["top_score"] < th) / max(len(unans), 1)
        refused_ans = sum(1 for r in ans if r["top_score"] < th) / max(len(ans), 1)
        curve.append({"threshold": th, "refusal_accuracy": round(refused_unans, 4), "false_refusal": round(refused_ans, 4)})

    feasible = [c for c in curve if c["false_refusal"] <= 0.15]
    # 平台期（拒答准确率同为 100%）上取误拒率最低的点，而不是阈值最高的点：
    # 原来的 (refusal_accuracy, threshold) 会让阈值在平台上一直往上顶，
    # 选出一堆"拒得更多但没换来任何收益"的严格劣势点。
    best = (
        max(feasible, key=lambda c: (c["refusal_accuracy"], -c["false_refusal"], -c["threshold"]))
        if feasible
        else curve[0]
    )
    return {
        "curve": curve,
        "recommended": best,
        "rule": "误拒率 ≤ 15% 的前提下，取拒答准确率最高的阈值（宁可多拒，不可错答）",
    }