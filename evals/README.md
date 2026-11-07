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
| holdout | 当前 100 条全量用于对比（样本量小），**没有**划分 holdout；这是已知限制，不宣称"线上分布一致" |

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
- **引用有效率**：引用能定位到 span 且 quote 在原文里逐字存在
- **P50/P95 延迟**、**单问成本**

## 跑一次

```bash
uv run documind eval --dataset default                      # 四种切块全跑
uv run documind eval --dataset default --limit 20           # 冒烟
uv run documind eval --dataset default --strategies structure
```

产物：`evals/reports/ablation.md`（贴 README）、`ablation.html`（评测看板，`/evals/report` 直接返回它）、
`details-<时间戳>.json`（逐题明细，含每题的检索名次/引用/成本，便于定位回归）。

## 诚实边界

- judge 与被测模型是同一个模型（自评有偏差），已在报告里标注。
- 评测集是合成的，不是真实用户日志；`false_refusal_rate` 的绝对值会偏乐观。
- 语料只有章程一类文本，切块结论不可外推到合同/代码/工单。