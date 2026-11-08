# 评测集与回归流水线

三层结构：

```
evals/
  datasets/   语料（公开 PDF）+ 自建问答集 qa_default.jsonl
  reports/    每次跑的产物：ablation.md / ablation.html / details-*.json（带时间戳快照）
```

## 数据集怎么来的

| 项 | 做法 |
|---|---|
| 语料 | 3 份**公开**上市公司章程（巨潮资讯），共 166 页；doc_id 与 ACL 在 `scripts/ingest_corpus.py` 里固定 |
| 可答问题 | 从 structure 切块的块里抽样，让 LLM **只根据该块原文**出题，并要求给出逐字 quote 作为 gold 依据；quote 不能在该块里逐字找到的样本直接丢弃 |
| 不可答问题 | 20 条固定问题（年假折现、报销标准、工资发放日、试用期等），这些内容章程里确实没有——用来量拒答准确率 |
| 人工校验 | 抽检 20 条，删掉 3 条问题有歧义、1 条答案不唯一的样本 |
| holdout | 当前 87 条（67 可答 / 20 不可答）全量用于对比（样本量小），**没有**划分 holdout；这是已知限制，不宣称"线上分布一致" |

生成命令（幂等：已存在就不重跑）：

```bash
uv run python -c "import asyncio;from documind.evals.dataset import build_dataset;from documind.db import get_session_factory,init_db;from documind.llm import get_llm;
async def m():
    await init_db()
    async with get_session_factory()() as s: print(await build_dataset(s, get_llm(), 80))
asyncio.run(m())"
```

## 指标口径

定义写在 `src/documind/evals/metrics.py` 顶部，避免"数字口径随汇报变化"：

- **Recall@5**：Top-5 命中任一 gold 页，或 gold quote 的前 30 字逐字出现在召回块里
- **Context Recall**：gold quote 是否进了喂给模型的上下文（衡量切块+检索的召回完整性）
- **Faithfulness**：回答里的事实性论断能在上下文找到依据的比例（LLM judge，prompt 在代码里）
- **Answer Relevancy**：是否正面回答问题（judge 0/1）
- **拒答准确率**：20 条无答案问题里拒答的比例
- **误拒率**：可答问题里被拒的比例
- **引用有效率**：citation 的 (page_start,char_start,page_end,char_end) 能从 pages 原文切出 span，且**归一化后的整句** quote 在这段切片里逐字存在（不再只比前 20 字）
- **P50/P95 延迟**、**单问成本**

## 跑一次

```bash
uv run documind eval --dataset default                      # 四种切块全跑
uv run documind eval --dataset default --limit 20           # 冒烟
uv run documind eval --dataset default --strategies structure
```

**完整四策略跑法**：单个进程跑四策略 × 87 题时，rerank（bge-reranker-base，CPU）是进程内串行段，
一个策略要 30 分钟以上，容易被工具/会话的超时打断。用分片并行版（只做编排，指标与判定仍走 `src/documind/evals`）：

```bash
# 每策略 20 可答 + 8 不可答，3 个 worker 进程并行，逐题落盘到 evals/reports/_parts/
./.venv/Scripts/python.exe -u scripts/run_ablation_parallel.py \
    --strategies fixed,semantic,structure,parent_child \
    --ans-limit 20 --unans-limit 8 --workers 3 --concurrency 2

# 只合并已有分片、重写报告（跑挂了也不用重跑模型）
./.venv/Scripts/python.exe -u scripts/run_ablation_parallel.py --merge-only \
    --ans-limit 20 --unans-limit 8 --since-epoch <本次开跑的 epoch 秒>
```

产物：`evals/reports/ablation.md`（贴 README）、`ablation.html`（评测看板，`/evals/report` 直接返回它）、
`details-<时间戳>.json`（逐题明细，含每题的检索名次/引用/成本，便于定位回归）。

## 阈值标定（不调 judge，不花钱）

`scripts/eval_retrieval.py --write` 用检索指标反推拒答阈值并写回 `eval_runs`
（`reranker` 分桶），`/health` 的 `refusal_threshold.source` 随之从 `builtin` 变 `calibrated`。
规则：**误拒率 ≤ 15% 的前提下取拒答准确率最高的阈值**，平台期取最保守点
（`src/documind/evals/runner.py:161`）。

```bash
# 只保留与当前库匹配的可答题（默认集是 3 份 PDF 的题，库里只有 1 份）
./.venv/Scripts/python.exe scripts/build_calib_dataset.py \
    --doc "江苏苏豪时尚集团股份有限公司章 程" \
    --hard evals/datasets/hard_neg_suhao.jsonl -o evals/datasets/qa_suhao.jsonl

DOCUMIND_DATABASE_URL="sqlite+aiosqlite:///D:/code/documind/data/real_dashscope.db" \
./scripts/with_dashscope.sh ./.venv/Scripts/python.exe -u scripts/eval_retrieval.py \
    --dataset suhao --strategy structure --ans-limit 15 --unans-limit 22 --write \
    --json evals/reports/calib_suhao.json
```

远端 embedding（`text-embedding-v3@1024`）+ 1 份章程 + `structure` 切块的一次结果
（2026-11-08，n=37）与两套阈值的真问答对比见 `docs/REAL-PROVIDER.md` §8。摘要：

| 指标 | 值 |
|---|---|
| Recall@5 / Context Recall | 1.0（15/15） / 1.0（15/15） |
| MRR | 0.9222 |
| 拒答准确率（pre-LLM，旧阈值 -6.0） | 0.2273（5/22） |
| 误拒率（pre-LLM） | 0.0（0/15） |
| recommended_threshold | 1.5055（可答下界 1.5759 / 不可答上界 1.3999，间隔仅 0.176） |

**n 小不能外推**：15 条可答、22 条不可答（其中与章程同域的"有相似条款"难例只有 2 条），
且只有 1 份 PDF、一种切块。1.5055 只对这份语料 + 这个 reranker 成立；20 条固定不可答题是 HR 域，
与章程的距离偏大，拒答准确率会偏乐观。

## 诚实边界

- judge 与被测模型是同一个模型（自评有偏差），已在报告里标注。
- 评测集是合成的，不是真实用户日志；`false_refusal_rate` 的绝对值会偏乐观。
- 语料只有章程一类文本，切块结论不可外推到合同/代码/工单。