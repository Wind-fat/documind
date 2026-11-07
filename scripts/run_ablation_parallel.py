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


def select_cases(dataset: str, limit: int | None) -> list[dict]:
    """与 runner.run_eval 的截断口径完全一致：按可答/不可答 8:2 分别截断。"""
    sys.path.insert(0, str(ROOT / "src"))
    from documind.evals.dataset import load_dataset

    cases = load_dataset(DATASETS / f"qa_{dataset}.jsonl")
    if limit:
        ans = [c for c in cases if c["answerable"]][: int(limit * 0.8)]
        unans = [c for c in cases if not c["answerable"]][: max(1, int(limit * 0.2))]
        cases = ans + unans
    return cases


def shard(items: list[dict], i: int, n: int) -> list[dict]:
    return [c for j, c in enumerate(items) if j % n == i]


# ---------------------------------------------------------------- worker

async def worker(strategy: str, shard_id: int, nshards: int, limit: int | None, concurrency: int) -> None:
    sys.path.insert(0, str(ROOT / "src"))
    from documind.db import init_db
    from documind.evals.runner import _one_case
    from documind.llm import get_llm

    cases = shard(select_cases("default", limit), shard_id, nshards)
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

def run_strategy(strategy: str, limit: int | None, workers: int, concurrency: int) -> float:
    PARTS.mkdir(parents=True, exist_ok=True)
    for p in PARTS.glob(f"{strategy}-s*.jsonl"):
        p.unlink()
    print(f"\n=== strategy={strategy} workers={workers} limit={limit} ===", flush=True)
    t0 = time.perf_counter()
    procs = []
    for i in range(workers):
        cmd = [sys.executable, "-u", __file__, "--worker", "--strategies", strategy,
               "--shard", str(i), "--nshards", str(workers), "--concurrency", str(concurrency)]
        if limit:
            cmd += ["--limit", str(limit)]
        procs.append(subprocess.Popen(cmd, cwd=str(ROOT)))
    codes = [p.wait() for p in procs]
    wall = time.perf_counter() - t0
    if any(codes):
        print(f"!! {strategy} 有 worker 非零退出: {codes}", flush=True)
    print(f"[{strategy}] wall={wall:.0f}s", flush=True)
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategies", default="fixed,semantic,structure,parent_child")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--concurrency", type=int, default=4, help="每个 worker 内的并发")
    ap.add_argument("--merge-only", action="store_true")
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    args = ap.parse_args()
    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]

    if args.worker:
        asyncio.run(worker(strategies[0], args.shard, args.nshards, args.limit, args.concurrency))
        return

    sys.path.insert(0, str(ROOT / "src"))
    from documind.evals.metrics import aggregate
    from documind.evals.report import write_reports
    from documind.evals.runner import calibrate_threshold

    walls: dict[str, float] = {}
    if not args.merge_only:
        for s in strategies:
            walls[s] = run_strategy(s, args.limit, args.workers, args.concurrency)

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
    print("\n=== summary ===", flush=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print("\n=== 拒答阈值 ===", flush=True)
    print(json.dumps(calibration["recommended"], ensure_ascii=False, indent=2), flush=True)
    print(f"\nwall_total={total_wall:.0f}s 报告: {report['markdown']}\n明细: {report['json']}", flush=True)


if __name__ == "__main__":
    main()