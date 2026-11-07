"""跑四种切块的消融实验，产出 evals/reports/*（ablation.md / ablation.html / details-*.json）。"""

import asyncio
import json
import sys

from documind.db import get_session_factory, init_db
from documind.evals.runner import run_eval


async def main() -> None:
    strategies = ["fixed", "semantic", "structure", "parent_child"]
    limit = None
    if "--limit" in sys.argv:
        limit = int(sys.argv[sys.argv.index("--limit") + 1])
    await init_db()
    async with get_session_factory()() as s:
        report = await run_eval(s, dataset="default", strategies=strategies, limit=limit)
    print("\n=== summary ===")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print("\n=== 拒答阈值 ===")
    print(json.dumps(report["calibration"]["recommended"], ensure_ascii=False, indent=2))
    print(f"\n报告: {report['markdown']}\n看板: {report['html']}")


asyncio.run(main())