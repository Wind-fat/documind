"""只用确定性指标的检索评测：Recall@5 / Context Recall / MRR + 不花钱的拒答判定。

为什么单独有它：`documind eval` 每题要调一次生成模型 + 一次 judge，跑一轮 28 题要几十分钟，
而且在改重排/索引这种**纯检索**改动时，judge 的噪声会盖住真实差异（README 也写了 n=20 时
Recall@5 的粒度是 5 个百分点）。这个脚本只跑 `retrieve()` 与 `evals.metrics`，同一套指标口径
（`src/documind/evals/metrics.py`），但不调 LLM：
- 生成侧指标（faithfulness / answer_relevancy）不产出，报告里要写清"本次未测"
- 拒答只算 **pre-LLM** 那一段（检索为空 / 最高重排分低于阈值，见 `generate._pre_llm_refusal`），
  "模型自己说找不到"那一段需要 LLM，不在本脚本范围

用法：
    ./.venv/Scripts/python.exe scripts/eval_retrieval.py --strategy structure
    ./.venv/Scripts/python.exe scripts/eval_retrieval.py --ans-limit 20 --unans-limit 8 --json out.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from documind.config import ROOT as PKG_ROOT  # noqa: E402
from documind.evals.dataset import load_dataset  # noqa: E402
from documind.evals.metrics import retrieval_metrics  # noqa: E402
from documind.evals.runner import calibrate_threshold  # noqa: E402


def select_cases(dataset: str, limit: int | None, ans_limit: int | None, unans_limit: int | None) -> list[dict]:
    cases = load_dataset(PKG_ROOT / "evals" / "datasets" / f"qa_{dataset}.jsonl")
    ans = [c for c in cases if c["answerable"]]
    unans = [c for c in cases if not c["answerable"]]
    if limit is not None:  # 与 documind eval 的 --limit 同比例截断
        ans, unans = ans[: int(limit * 0.8)], unans[: max(1, int(limit * 0.2))]
    if ans_limit is not None:
        ans = ans[:ans_limit]
    if unans_limit is not None:
        unans = unans[:unans_limit]
    return ans + unans


def _pct(vals: list[float], p: float) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    return float(s[min(len(s) - 1, int(round(p * (len(s) - 1))))])


async def main(args: argparse.Namespace) -> dict:
    from documind.db import ensure_schema_compat, get_session_factory
    from documind.generate import _pre_llm_refusal
    from documind.rerank import get_reranker
    from documind.retrieve import retrieve
    from documind.thresholds import resolve_refusal_threshold

    # 老库补列必须在读 eval_runs 之前跑：它不是应用启动路径，没人替它做迁移，
    # 否则 `no such column: eval_runs.reranker` 会以裸 SQL 报错的形式炸出来。
    added = await ensure_schema_compat()
    if added:
        print(f"[schema] 已为现有库补齐列: {', '.join(added)}", flush=True)

    cases = select_cases(args.dataset, args.limit, args.ans_limit, args.unans_limit)
    reranker = get_reranker()
    out: dict = {"reranker": reranker.name, "strategies": {}}

    async with get_session_factory()() as session:
        thr = await resolve_refusal_threshold(session, reranker.name)
        print(f"[reranker] {reranker.name} · 拒答阈值 {thr.value:.4f}（来源 {thr.source}）"
              f" · max_tokens={getattr(reranker, 'max_tokens', '-')}"
              f" · batch={getattr(reranker, 'batch_size', '-')}", flush=True)

        for strategy in [s for s in args.strategy.split(",") if s]:
            rows: list[dict] = []
            lat: list[float] = []
            for case in cases:
                t0 = time.perf_counter()
                ret = await retrieve(session, case["question"], role="admin", strategy=strategy)
                wall = (time.perf_counter() - t0) * 1000
                rm = retrieval_metrics(ret, case)
                refused = _pre_llm_refusal(ret, thr.value) is not None
                rows.append(
                    {
                        "id": case["id"],
                        "answerable": case["answerable"],
                        "top_score": round(ret.top_score, 4),
                        "refused": refused,
                        "candidates": ret.candidates,
                        **rm,
                    }
                )
                lat.append(wall)

            ans = [r for r in rows if r["answerable"]]
            unans = [r for r in rows if not r["answerable"]]

            def mean(key: str, sub: list[dict]) -> float:
                return round(sum(r[key] for r in sub) / len(sub), 4) if sub else 0.0

            summary = {
                "n_cases": len(rows),
                "n_answerable": len(ans),
                "n_unanswerable": len(unans),
                "recall_at_5": mean("recall_at_k", ans),
                "context_recall": mean("context_recall", ans),
                "mrr": mean("mrr", ans),
                "refusal_accuracy_pre_llm": mean("refused", unans),
                "false_refusal_rate_pre_llm": mean("refused", ans),
                "top_score_min_answerable": round(min((r["top_score"] for r in ans), default=0.0), 4),
                "latency_p50_ms": round(_pct(lat, 0.5), 1),
                "latency_p95_ms": round(_pct(lat, 0.95), 1),
                "threshold": thr.value,
                "threshold_source": thr.source,
            }
            cal = calibrate_threshold(rows, reranker=reranker.name)
            summary["recommended_threshold"] = cal["recommended"]["threshold"]
            summary["recall_at_5_hits"] = f"{int(summary['recall_at_5'] * len(ans))}/{len(ans)}"
            summary["context_recall_hits"] = f"{int(summary['context_recall'] * len(ans))}/{len(ans)}"
            out["strategies"][strategy] = {"summary": summary, "rows": rows}
            print(f"[{strategy}] " + " ".join(f"{k}={v}" for k, v in summary.items() if not k.startswith("n_")), flush=True)

            if getattr(args, "write", False):
                # 标定回写：这是 P1-03 缺的那条路径——不调 judge 也能得到可追溯的阈值，
                # 写进 eval_runs 后 /health 的 source 就从 builtin 变 calibrated。
                from documind.db import EvalRun

                session.add(
                    EvalRun(
                        strategy=f"retrieval-calib:{strategy}",
                        dataset=args.dataset,
                        n_cases=summary["n_cases"],
                        metrics=json.dumps(summary, ensure_ascii=False),
                        report_path="scripts/eval_retrieval.py",
                        reranker=reranker.name,
                        recommended_threshold=float(summary["recommended_threshold"]),
                    )
                )
                await session.commit()
                print(
                    f"[write] 已写入标定行：reranker={reranker.name} "
                    f"recommended_threshold={summary['recommended_threshold']} n_cases={summary['n_cases']}",
                    flush=True,
                )

    if args.json:
        Path(args.json).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[json] {args.json}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="default")
    ap.add_argument("--strategy", default="structure")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--ans-limit", type=int, default=20)
    ap.add_argument("--unans-limit", type=int, default=8)
    ap.add_argument("--json", default=None)
    ap.add_argument(
        "--write",
        action="store_true",
        help="把标定出的推荐阈值写进 eval_runs（reranker 分桶），供 /health 与运行时读取",
    )
    asyncio.run(main(ap.parse_args()))