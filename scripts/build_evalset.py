"""生成评测集（合成问答对 + 20 条不可答问题）。已存在就跳过。"""

import asyncio
import sys
from pathlib import Path

from documind.config import ROOT
from documind.db import get_session_factory, init_db
from documind.evals.dataset import build_dataset
from documind.llm import get_llm

OUT = ROOT / "evals" / "datasets" / "qa_default.jsonl"


async def main() -> None:
    if OUT.exists() and "--force" not in sys.argv:
        print(f"已存在 {OUT}（加 --force 重生成）")
        return
    await init_db()
    async with get_session_factory()() as s:
        path = await build_dataset(s, get_llm(), n_answerable=78, out=OUT)
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    print(f"写出 {path}，共 {len(lines)} 条")


asyncio.run(main())