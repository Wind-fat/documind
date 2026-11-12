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
    from documind.db import get_session_factory, init_db
    from documind.generate import _pre_llm_refusal
    from documind.rerank import get_reranker
    from documind.retrieve import retrieve
    from documind.thresholds import (
        calibration_gate,
        corpus_fingerprint,
        current_embed_space,
        resolve_refusal_threshold,
    )

    # 表结构必须在读写 eval_runs 之前就位。这里用 init_db 而不是 ensure_schema_compat：
    # 后者只给**已存在的表**补列，在一个全新空 sqlite 文件上会直接
    # `no such table: eval_runs`（脚本不是应用启动路径，没人替它建表）。
    # init_db = create_all + 老库补列，两种库都覆盖。
    await init_db()

    cases = select_cases(args.dataset, args.limit, args.ans_limit, args.unans_limit)
    reranker = get_reranker()
    # 阈值的桶键是 (reranker, embed_space)：本轮跑的库用哪个向量空间，标定值就写回哪个空间。
    # 拿不到 embedder 时空串不会被当成桶键（`latest_calibration` 拒绝查空空间），
    # 所以这里必须显式报出来，而不是让它默默变成一个"通用值"。
    space = current_embed_space()
    out: dict = {"reranker": reranker.name, "embed_space": space, "strategies": {}}
    write_refusals: list[str] = []
    print(f"[space] 向量空间 {space or '（未知：标定行将无法在该库生效）'}", flush=True)

    async with get_session_factory()() as session:
        thr = await resolve_refusal_threshold(session, reranker.name, space)
        # 桶键的第三半（P1-12）：标定行必须记住"是在哪份语料上标的"，否则同一 embedder
        # 下换一份文档时这行会被当成可用的标定（suhao 的 1.5055 落到美诺华上误拒 2/26）。
        corpus = await corpus_fingerprint(session)
        print(f"[corpus] 语料指纹 {corpus or '（空库：标定行将无法归属任何语料）'}", flush=True)
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
                if args.dump_hits:
                    # 只给离线分析用（缺项信号要文本，不要分数）：与模型看到的上下文同序同源。
                    # 按 chunk_id 去重——父子块会重复指向同一个父块。
                    seen: set[str] = set()
                    dumped = []
                    for c in ret.contexts[: args.dump_hits]:
                        if c.chunk_id in seen:
                            continue
                        seen.add(c.chunk_id)
                        dumped.append({"chunk_id": c.chunk_id, "score": round(c.score, 4), "text": c.text})
                    rows[-1]["hits"] = dumped
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
                "embed_space": space,
            }
            cal = calibrate_threshold(rows, reranker=reranker.name)
            summary["recommended_threshold"] = cal["recommended"]["threshold"]
            # 样本量不足的推荐值只是 provisional：写回/生效都要过 calibration_gate
            summary["provisional"] = cal["provisional"]
            summary["min_calibration_cases"] = cal["min_cases"]
            summary["recall_at_5_hits"] = f"{int(summary['recall_at_5'] * len(ans))}/{len(ans)}"
            summary["context_recall_hits"] = f"{int(summary['context_recall'] * len(ans))}/{len(ans)}"
            out["strategies"][strategy] = {"summary": summary, "rows": rows}
            print(f"[{strategy}] " + " ".join(f"{k}={v}" for k, v in summary.items() if not k.startswith("n_")), flush=True)

            if getattr(args, "write", False):
                ok, gate_reason = calibration_gate(summary["n_cases"])
                if not ok:
                    # 选择"拒绝写回"而不是"写一行 provisional=true"：理由见 calibration_gate
                    write_refusals.append(f"{strategy}: {gate_reason}")
                    print(f"[write:拒绝] [{strategy}] {gate_reason}", flush=True)
                else:
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
                            # 空间必须落库：不带它，这一行在解析时属于"历史通用值"，永远不生效（P1-10）
                            embed_space=space,
                            # 语料指纹同理（P1-12）：同一 embedder 下换一份文档，这行仍然不该生效
                            corpus_fp=corpus,
                            recommended_threshold=float(summary["recommended_threshold"]),
                        )
                    )
                    await session.commit()
                    print(
                        f"[write] 已写入标定行：reranker={reranker.name} embed_space={space or '（空！不生效）'} "
                        f"recommended_threshold={summary['recommended_threshold']} n_cases={summary['n_cases']}",
                        flush=True,
                    )

    if args.json:
        Path(args.json).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[json] {args.json}")
    if write_refusals:
        print(f"[write] {len(write_refusals)} 条未写回：样本量不足的推荐阈值不算标定", flush=True)
        raise SystemExit(2)
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
        "--dump-hits",
        type=int,
        default=0,
        metavar="N",
        help="把每题前 N 条上下文（chunk_id/score/text）写进 json，供离线信号分析；0=不写",
    )
    ap.add_argument(
        "--write",
        action="store_true",
        help="把标定出的推荐阈值写进 eval_runs（reranker 分桶），供 /health 与运行时读取",
    )
    asyncio.run(main(ap.parse_args()))