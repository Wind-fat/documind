"""CI 门禁：检索质量低于阈值就让流水线红。

与"跑一次评测看看"不同：阈值写死在仓库里，谁把检索或切块改坏了，PR 直接红，不用等人看报告。

CI 上只跑**检索类**指标（Recall@K / Context Recall / MRR）——它们不需要 LLM，
因此不烧额度、不受供应商抖动影响；生成质量（faithfulness 等）只在本地评测跑。
CI 用 hash embedder（不下载模型），阈值比本地 bge 低一档，两套阈值分开写，
禁止"CI 绿就等于线上好"的自欺。
"""

import asyncio
import sys

from documind.config import ROOT

FLOORS = {
    "hash": {"recall_at_5": 0.55, "context_recall": 0.55},
    "bge": {"recall_at_5": 0.70, "context_recall": 0.70},
}
CORPUS = [
    ("evals/datasets/zhicheng_suhao.pdf", ["all"]),
    ("evals/datasets/zhicheng_hengxuan.pdf", ["all"]),
    ("evals/datasets/zhicheng_meinuohua.pdf", ["finance", "admin"]),
]


async def main() -> int:
    import sqlalchemy as sa

    from documind.db import Document, get_session_factory, init_db
    from documind.embed import get_embedder
    from documind.evals.dataset import load_dataset
    from documind.evals.metrics import retrieval_metrics
    from documind.pipeline import ingest_document
    from documind.retrieve import retrieve

    qa = ROOT / "evals" / "datasets" / "qa_default.jsonl"
    if not qa.exists():
        print("SKIP: 没有评测集")
        return 0

    await init_db()
    async with get_session_factory()() as s:
        n_docs = (await s.execute(sa.select(sa.func.count(Document.id)))).scalar_one()
        if n_docs == 0:
            for rel, acl in CORPUS:
                if (ROOT / rel).exists():
                    await ingest_document(s, ROOT / rel, acl_roles=acl)
        cases = [c for c in load_dataset(qa) if c["answerable"]]
        rows = []
        for case in cases:
            res = await retrieve(s, case["question"], role="admin", strategy="structure")
            rows.append(retrieval_metrics(res, case))

    def mean(key: str) -> float:
        return round(sum(r[key] for r in rows) / max(len(rows), 1), 4)

    embedder = get_embedder().name
    floor = FLOORS["hash"] if embedder.startswith("hash") else FLOORS["bge"]
    got = {"recall_at_5": mean("recall_at_k"), "context_recall": mean("context_recall"), "mrr": mean("mrr")}
    print(f"embedder={embedder} n={len(rows)} metrics={got}")
    bad = [f"{k}={got[k]} < {v}" for k, v in floor.items() if got[k] < v]
    if bad:
        print("FAIL:", "; ".join(bad))
        return 1
    print("OK: 检索门禁通过")
    return 0


sys.exit(asyncio.run(main()))