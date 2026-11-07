"""消融实验的分片并行执行器（只用 scripts/ 下的编排，指标/判定全走 src 现成实现）。

为什么需要它：单进程跑一个策略 87 题时，rerank（bge-reranker-base，CPU）是串行段，
一个策略要 30 分钟以上，超过任何一次工具调用的时限。这里做两件事，都不改判定口径：

1. **多进程分片**：同一策略的题切成 N 片，每片一个独立进程（各自加载 reranker，
   绕开进程内的 reranker 锁），把 rerank 真正并行到多个核上。
2. **逐题落盘**：每跑完一题就 append 一行 JSONL 并 fsync，进程半路被杀也不会丢已完成的题，
   日志也是逐题 flush —— 之前几次"后台跑完只剩 0 字节日志"就是被缓冲吃掉了。

用法：
    python scripts/run_ablation_parallel.py                       # 四策略，全量 87 题
    python scripts/run_ablation_parallel.py --strategies structure --limit 15 --workers 3
    python scripts/run_ablation_parallel.py --merge-only          # 只合并已有分片重写报告
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PARTS = ROOT / "evals" / "reports" / "_parts"
DATASETS = ROOT / "evals" / "datasets"


def select_cases(dataset: str, limit: int | None, ans_limit: int | None = None,
                 unans_limit: int | None = None) -> list[dict]:
    """与 runner.run_eval 的截断口径一致：按可答/不可答 8:2 分别截断。

    ans_limit / unans_limit 单独给出时直接按条数取（用于把预算多分一点给拒答题）。
    """
    sys.path.insert(0, str(ROOT / "src"))
    from documind.evals.dataset import load_dataset

    cases = load_dataset(DATASETS / f"qa_{dataset}.jsonl")
    ans = [c for c in cases if c["answerable"]]
    unans = [c for c in cases if not c["answerable"]]
    if limit:
        ans = ans[: int(limit * 0.8)]
        unans = unans[: max(1, int(limit * 0.2))]
    if ans_limit is not None:
        ans = ans[:ans_limit]
    if unans_limit is not None:
        unans = unans[:unans_limit]
    return ans + unans


def shard(items: list[dict], i: int, n: int) -> list[dict]:
    return [c for j, c in enumerate(items) if j % n == i]


# ---------------------------------------------------------------- worker

async def worker(strategy: str, shard_id: int, nshards: int, limit: int | None, concurrency: int,
                 ans_limit: int | None = None, unans_limit: int | None = None) -> None:
    sys.path.insert(0, str(ROOT / "src"))
    from documind.db import init_db
    from documind.evals.runner import _one_case
    from documind.llm import get_llm

    cases = shard(select_cases("default", limit, ans_limit, unans_limit), shard_id, nshards)
    PARTS.mkdir(parents=True, exist_ok=True)
    out = PARTS / f"{strategy}-s{shard_id}.jsonl"
    # 续跑：同一分片重跑时跳过已完成的题
    done = set()
    if out.exists():
        for line in out.read_text(encoding="utf-8").splitlines():
            try:
                done.add(json.loads(line)["id"])
            except Exception:
                pass
    todo = [c for c in cases if c["id"] not in done]
    print(f"[{strategy}/s{shard_id}] n={len(cases)} todo={len(todo)}（已完成 {len(done)}）", flush=True)

    await init_db()
    llm = get_llm()
    sem = asyncio.Semaphore(concurrency)
    t_all = time.perf_counter()
    n_done = 0

    async def run(case: dict) -> None:
        nonlocal n_done
        async with sem:
            t0 = time.perf_counter()
            row = await _one_case(case, strategy, llm, selfcheck=False)
            n_done += 1
            with out.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
                os.fsync(f.fileno())
            print(f"[{strategy}/s{shard_id}] {n_done}/{len(todo)} {case['id']} "
                  f"ans={'Y' if case['answerable'] else 'N'} refused={row['refused']} "
                  f"{time.perf_counter() - t0:.0f}s cum={time.perf_counter() - t_all:.0f}s", flush=True)

    await asyncio.gather(*(run(c) for c in todo))
    print(f"[{strategy}/s{shard_id}] finished in {time.perf_counter() - t_all:.0f}s", flush=True)


# ---------------------------------------------------------------- parent

def run_strategy(strategy: str, limit: int | None, workers: int, concurrency: int,
                 ans_limit: int | None = None, unans_limit: int | None = None) -> float:
    PARTS.mkdir(parents=True, exist_ok=True)
    for p in PARTS.glob(f"{strategy}-s*.jsonl"):
        p.unlink()
    print(f"\n=== strategy={strategy} workers={workers} limit={limit} "
          f"ans={ans_limit} unans={unans_limit} ===", flush=True)
    t0 = time.perf_counter()
    procs = []
    for i in range(workers):
        cmd = [sys.executable, "-u", __file__, "--worker", "--strategies", strategy,
               "--shard", str(i), "--nshards", str(workers), "--concurrency", str(concurrency)]
        if limit:
            cmd += ["--limit", str(limit)]
        if ans_limit:
            cmd += ["--ans-limit", str(ans_limit)]
        if unans_limit:
            cmd += ["--unans-limit", str(unans_limit)]
        procs.append(subprocess.Popen(cmd, cwd=str(ROOT)))
    codes = [p.wait() for p in procs]
    wall = time.perf_counter() - t0
    if any(codes):
        print(f"!! {strategy} 有 worker 非零退出: {codes}", flush=True)
    print(f"[{strategy}] wall={wall:.0f}s", flush=True)
    wf = PARTS / "walls.json"
    data = json.loads(wf.read_text(encoding="utf-8")) if wf.exists() else {}
    data[strategy] = round(wall, 1)
    wf.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return wall


def collect(strategy: str) -> list[dict]:
    rows: dict[str, dict] = {}
    for p in sorted(PARTS.glob(f"{strategy}-s*.jsonl")):
        for line in p.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
            except Exception:
                continue
            rows[r["id"]] = r
    return list(rows.values())


def append_run_note(md_path: str, since_epoch: float, summary: dict, walls: dict, args) -> None:
    """在 write_reports 生成的 ablation.md 末尾补一段"这次到底怎么跑的"，含 LLM 调用统计。"""
    path = Path(md_path)
    trace = ROOT / "data" / "traces" / f"{time.strftime('%Y-%m-%d')}.jsonl"
    calls = 0
    llm_s = 0.0
    if trace.exists():
        for line in trace.read_text(encoding="utf-8").splitlines():
            try:
                d = json.loads(line)
            except Exception:
                continue
            if d.get("ts", 0) >= since_epoch and d.get("kind") == "llm":
                calls += 1
                llm_s += d.get("ms", 0) / 1000.0
    n_cases = sum(m["n_cases"] for m in summary.values())
    total_wall = sum(walls.values())
    lines = [
        "",
        "## 本次实验的执行配置（可复现）",
        "",
        f"- 数据集：`evals/datasets/qa_default.jsonl`，87 条（67 可答 / 20 不可答）；"
        f"本次每策略取前 {args.ans_limit or '全部'} 条可答 + 前 {args.unans_limit or '全部'} 条不可答，"
        f"合计 {n_cases} 次「题 × 策略」评测。",
        "- 四策略只改切块，其余链路完全一致：ACL → BM25 ∥ 向量 → RRF(k=60) → cross-encoder 重排(top 5) → 引用约束生成。",
        "- 组件：embedding `BAAI/bge-small-zh-v1.5`、rerank `BAAI/bge-reranker-base`（均本地 CPU）、"
        "被测与 judge 模型同为 `qwen-cn/qwen3.8-flash`（本机网关）。",
        f"- 执行方式：`scripts/run_ablation_parallel.py --workers {args.workers} --concurrency {args.concurrency}`"
        "（reranker 是进程级串行段，按策略切片多进程跑；逐题落盘，见 `evals/reports/_parts/`）。",
        f"- 耗时：LLM 调用 **{calls} 次**、累计 **{llm_s:.0f} 秒**（并发等待前）；"
        f"端到端墙钟 **{total_wall:.0f} 秒**。",
        "- **延迟口径**：表中的 P50/P95 是在上述并行评测负载下测到的（含 reranker 排队与并发生成等待），"
        "高于单用户串行体感；用于策略间横向比较，不等于线上 SLA。",
        "- 单问成本按 `config.py` 里的公开单价折算（prompt $0.15/Mtok、completion $0.47/Mtok），"
        "含 judge 那次调用——所以是「评测成本」而非「线上单问成本」。",
    ]
    with path.open("a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategies", default="fixed,semantic,structure,parent_child")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--ans-limit", type=int, default=None)
    ap.add_argument("--unans-limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--concurrency", type=int, default=4, help="每个 worker 内的并发")
    ap.add_argument("--merge-only", action="store_true")
    ap.add_argument("--since-epoch", type=float, default=None,
                    help="统计 trace 里该时间点之后的 LLM 调用（用于报告里的调用次数/耗时）")
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    args = ap.parse_args()
    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]

    if args.worker:
        asyncio.run(worker(strategies[0], args.shard, args.nshards, args.limit, args.concurrency,
                           args.ans_limit, args.unans_limit))
        return

    sys.path.insert(0, str(ROOT / "src"))
    from documind.evals.metrics import aggregate
    from documind.evals.report import write_reports
    from documind.evals.runner import calibrate_threshold

    walls: dict[str, float] = {}
    if not args.merge_only:
        for s in strategies:
            walls[s] = run_strategy(s, args.limit, args.workers, args.concurrency,
                                    args.ans_limit, args.unans_limit)
    else:
        wf = PARTS / "walls.json"
        if wf.exists():
            walls = json.loads(wf.read_text(encoding="utf-8"))

    summary: dict[str, dict] = {}
    details: dict[str, list[dict]] = {}
    for s in strategies:
        rows = collect(s)
        if not rows:
            continue
        details[s] = rows
        summary[s] = aggregate(rows)
        summary[s]["wall_s"] = round(walls.get(s, 0.0), 1)
        m = summary[s]
        print(f"[{s}] n={m['n_cases']} faithfulness={m['faithfulness']} recall@5={m['recall_at_5']} "
              f"refusal_acc={m['refusal_accuracy']} p95={m['latency_p95_ms']}ms", flush=True)
    if not summary:
        print("没有可合并的数据", flush=True)
        return

    calibration = calibrate_threshold([r for rows in details.values() for r in rows])
    report = write_reports(dataset="default", summary=summary, details=details, calibration=calibration)
    total_wall = sum(walls.values())
    if args.since_epoch:
        append_run_note(report["markdown"], args.since_epoch, summary, walls, args)
    print("\n=== summary ===", flush=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print("\n=== 拒答阈值 ===", flush=True)
    print(json.dumps(calibration["recommended"], ensure_ascii=False, indent=2), flush=True)
    print(f"\nwall_total={total_wall:.0f}s 报告: {report['markdown']}\n明细: {report['json']}", flush=True)


if __name__ == "__main__":
    main()