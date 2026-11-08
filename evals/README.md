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
（`(reranker, embed_space)` 分桶，见文末"阈值桶键"一节；本节写作时桶键还只有 `reranker`），
`/health` 的 `refusal_threshold.source` 随之从 `builtin` 变 `calibrated`。
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

> **本节结论已被下一节取代**：只有 2 条同域难例时算出的"间隔 0.176"是样本量不足的产物。
> 扩到 28 条后间隔为 **-4.2269**（负值），1.5055 不再被任何数据支持为"安全阈值"。

## 同域难例扩样：28 条（本次）

上一节的间隔只由 2 条同域难例撑起来，其中 1 条标注还有争议。这一节把同域难例扩到 28 条并重跑标定，
目的是让"间隔"和"拒答率"有统计意义——结论是**没有**，间隔被证伪（详见"结论"）。

### 数据集怎么造的（可复核）

`evals/datasets/hard_neg_suhao.jsonl`，28 条，全部从 `zhicheng_suhao.pdf`（59 页）的**原文条款**里造：
挑"有章可循但没给数值/主体/时限"的条款，问题围绕缺失的那一项提问。每条都带：

| 字段 | 含义 | 怎么核 |
|---|---|---|
| `source_page` + `source_quote` | 那条相似条款在哪一页、逐字原文 | 必须能在该页原文里找到（归一化空白后比较） |
| `anchor` | 条款里紧邻缺失项的短语 | 必须是 `source_quote` 的子串；含 anchor 的那句**不得含数量词**（阿拉伯数字或"三十日/500 万元/25%"） |
| `abs_terms` | 一旦存在就说明"答案在文里"的词 | 全文 0 命中 |
| `ambiguous` | 标注有争议（可被解读成可答） | 单独列出，不自行裁定 |

```bash
./.venv/Scripts/python.exe scripts/verify_hard_neg.py     # 28/28 通过才算数
# [hard-neg] n=28 页数=59 全文=27893 字
# [分类] rate=5 · amount=8 · deadline=7 · subject=8
# [有争议] 4 条: hardneg-005, hardneg-010, hardneg-012, hardneg-026
# [OK] 28/28 条通过：出处逐字可核 · 缺失词全文零命中 · 条款无数量词（除显式豁免）
```

四类分布（每类至少 5 条，覆盖"税率/费率、金额/额度、时限/期限、主体/审批人"）：

| 类别 | n | 例 | 该类的典型陷阱 |
|---|---|---|---|
| rate 税率/费率 | 5 | 分红个税税率、任意公积金比例、企业所得税率 | 同条给了法定公积金 10%，易被误当答案 |
| amount 金额/额度 | 8 | 审计费用、独立董事津贴、关联交易门槛、对外捐赠额度 | 同页有 1,000 万/5,000 万交易门槛，但那是"交易"不是"关联交易" |
| deadline 时限/期限 | 7 | 股东查阅资料几日内提供、内部审计报告频次、党组织报告时限 | 条款只写"及时/视情节"，同章别处有 60 日/6 个月 |
| subject 主体/审批人 | 8 | 审计负责人由谁聘任、董秘空缺谁代行、独立董事候选人谁提名 | 条款只写"对董事会负责"，没写任免主体 |

### 重跑结果（不调 judge）

```bash
DOCUMIND_DATABASE_URL="sqlite+aiosqlite:///D:/code/documind/data/real_dashscope.db" \
./scripts/with_dashscope.sh ./.venv/Scripts/python.exe -u scripts/eval_retrieval.py \
    --dataset suhao --strategy structure --ans-limit 15 --unans-limit 48 \
    --json evals/reports/calib_suhao_hardneg.json
./.venv/Scripts/python.exe scripts/calib_report.py \
    --json evals/reports/calib_suhao_hardneg.json --threshold 1.5055 --threshold 2.7467
```

n=63（15 可答 / 48 不可答 = 28 同域难例 + 20 HR 固定），`structure` 切块，`text-embedding-v3@1024` +
`BAAI/bge-reranker-base`：

| 指标 | 值 |
|---|---|
| Recall@5 / Context Recall | 1.0（15/15） / 1.0（15/15） |
| MRR | 0.9222 |
| 拒答准确率（pre-LLM，**旧阈值 1.5055**） | 0.6875（33/48）—— 同域难例 13/28，HR 固定 20/20 |
| 误拒率（pre-LLM） | 0.0（0/15） |
| recommended_threshold | **2.7467**（脚本输出） |
| 延迟 P50 / P95 | 3248 ms / 4219 ms |

### 间隔：0.176 → **-4.2269**

| | 上一轮（n=37，同域难例 2） | 本次（n=63，同域难例 28） |
|---|---|---|
| 可答下界（min top_score，可答） | 1.5759 | 1.5759 |
| 不可答上界（max top_score，不可答） | 1.3999 | **5.8028** |
| 间隔 | **+0.176** | **-4.2269** |

不可答上界从 1.3999 冲到 5.8028 的是 `hardneg-009`（关联交易金额门槛，条款把门槛外指到交易所规则）——
问法与第 38 页正文几乎逐字重合，重排器就给了 5.80。**28 条里有 15 条分数高于"最容易的那条可答题"
（1.5759）**，即：任何"不误拒最低分可答题"的阈值都必须放弃这 15 条。间隔不是变厚，是**负的**。

分类分布（`top_score` 中位数）显示这不是个别噪声：`deadline` 中位数 3.70、`subject` 2.83，
而 `amount` 中位数 -1.06、`rate` -0.10——**"问法是否复述条款"比"答案在不在"更决定分数**。

### 两个候选阈值（同一批 63 题；未回写）

| 候选 | 来源 | 误拒率 | 同域难例拒答 | 全部不可答拒答 | 备注 |
|---|---|---|---|---|---|
| **A = 1.5055** | 上一轮标定（当前 `/health` 值） | 0/15 = 0.0000 | 13/28 = 0.4643 | 33/48 = 0.6875 | 零误拒，但近半数同域难例会被作答 |
| **B = 2.7467** | 本次脚本 `recommended_threshold` | 2/15 = 0.1333 | 16/28 = 0.5714 | 36/48 = 0.7500 | 已**高于**可答下界 1.5759，靠规则"允许 15% 误拒"才成立 |
| C = 3.1413 | 手算复核（网格没扫到） | 2/15 = 0.1333 | 17/28 = 0.6071 | 37/48 = 0.7708 | 与 B 同误拒率但多拒 1 条，说明 2.7467 是分位数网格的产物，不是 Pareto 点 |

`--write` **没有执行**：推荐值相对旧值变动 +82%（1.5055 → 2.7467），且 B/C 都建立在
"可答下界 < 不可答上界"不成立的前提下，属于"会推翻上一轮结论"的改动，按流程交主控定。
`eval_runs` 仍只有 2026-11-08 00:30 那行 1.5055，当时 `resolve_refusal_threshold` 复核返回
`value=1.5055 source=calibrated`。

> 2026-11-08 之后（P1-10）这行**不再自动生效**：它没有 `embed_space` 标注，按"历史通用值"只告警。
> 回填后仍是 1.5055/calibrated。见文末"阈值桶键：从 `reranker` 到 `(reranker, embed_space)`"。

### 结论

1. **间隔没变厚，是消失了**：0.176 → -4.2269。上一轮的 0.176 是 2 条同域难例的偶然，不能当分布看。
2. **1.5055 的实际能力**：零误拒、但同域难例只拒 46%；HR 域 20/20 全拒说明这层"拒答准确率"
   主要靠域外距离撑着，不是靠理解"条款有但没写值"。
3. **不存在单阈值解**：15/28 同域难例的分数高于最容易的可答题。要改善只能换信号
   （query 改写/条款级判别/让模型自己判"缺少哪一项"），不是调阈值。
4. **标注有争议的 4 条**（`hardneg-005/010/012/026`）单独列出、未计入任何"修正"；
   排除它们后（剩 24 条）不可答上界仍是 5.8028（`hardneg-009`），间隔结论不变，
   1.5055 下同域拒答 11/24 = 0.4583（原 13/28 = 0.4643）。
5. **n 仍小，不能外推**：28 条难例是 1 份 59 页章程、1 种切块、1 个 reranker 上的人工题；
   `deadline` 只有 7 条、`rate` 只有 5 条，分类中位数差 1–2 分不足以谈"类别效应"。
   换成合同/年报或换 reranker 必须重标。

## 第二信号实验：缺项形态（本次）

单阈值这条路的结论是"没有解"（见上一节）。这一轮试的是**不看分数**的第二信号：
问题在要一个具体值（比率 / 金额 / 时限 / 主体）时，去检索到的前 5 条块里找**那个形态**
（`%`、带货币单位的数字、`日/天/个月`、"由…决定/审批"）。找不到就判"该拒"。

- 实现：`src/documind/refusal.py`（`missing_specifics(query, hits, top_n, scope)`；纯函数、不调模型、不读库）
- 契约测试：`tests/test_refusal.py`（43 条）
- 复算脚本：`scripts/refusal_signal_report.py`（纯算术；输入是 `eval_retrieval.py --dump-hits 5` 的 json）
- **默认拒答路径未改**：`generate` 的私有 `_pre_llm_refusal` 仍只看分数（有测试钉住），信号是可选接入。

```bash
DOCUMIND_DATABASE_URL="sqlite+aiosqlite:///D:/code/documind/data/real_dashscope.db" \
./scripts/with_dashscope.sh ./.venv/Scripts/python.exe -u scripts/eval_retrieval.py \
    --dataset suhao --strategy structure --ans-limit 15 --unans-limit 48 --dump-hits 5 \
    --json evals/reports/calib_suhao_hardneg_hits.json
./.venv/Scripts/python.exe scripts/refusal_signal_report.py \
    --json evals/reports/calib_suhao_hardneg_hits.json --threshold 1.5055 --top-n 5 \
    --md evals/reports/refusal_signal.md
```

重跑后的检索数字与上一轮逐位一致（`refusal_accuracy_pre_llm=0.6875`、`false_refusal_rate_pre_llm=0.0`、
`top_score_min_answerable=1.5759`、`mrr=0.9222`），所以下面两表与上一节是同一批分。
`hits` 存的是 `ret.contexts`（喂模型的那批父块，已按 chunk_id 去重），不是原始召回结果。

### 四种组合 · `scope=block`（需求原始口径：形态出现在前 5 条块里任意位置）

| 组合 | 可答误拒 ↓ | 同域难例拒答 ↑ | HR 拒答 | 全部不可答拒答 | 拒答率−误拒率 |
|---|---|---|---|---|---|
| threshold 1.5055 | 0/15=0.0000 | 13/28=0.4643 | 20/20=1.0000 | 33/48=0.6875 | +0.4643 |
| signal | 0/15=0.0000 | 0/28=0.0000 | 0/20=0.0000 | 0/48=0.0000 | +0.0000 |
| OR | 0/15=0.0000 | 13/28=0.4643 | 20/20=1.0000 | 33/48=0.6875 | +0.4643 |
| AND | 0/15=0.0000 | 0/28=0.0000 | 0/20=0.0000 | 0/48=0.0000 | +0.0000 |

28 条难例的意图**全部认出来了**（4 类各 100%），但形态检查**一条都没判缺**：章程每页都有
百分比/日期/金额/"由…决定"，前 5 条块里总有一个"长得像值"的东西。块级口径下这个信号是恒假。

### 四种组合 · `scope=sentence`（形态必须出现在"与问题词面最重合的那一条"里）

| 组合 | 可答误拒 ↓ | 同域难例拒答 ↑ | HR 拒答 | 全部不可答拒答 | 拒答率−误拒率 |
|---|---|---|---|---|---|
| threshold 1.5055 | 0/15=0.0000 | 13/28=0.4643 | 20/20=1.0000 | 33/48=0.6875 | +0.4643 |
| signal | 1/15=0.0667 | 19/28=0.6786 | 4/20=0.2000 | 23/48=0.4792 | +0.6119 |
| **OR** | 1/15=0.0667 | **20/28=0.7143** | 20/20=1.0000 | 40/48=0.8333 | +0.6476 |
| AND | 0/15=0.0000 | 12/28=0.4286 | 4/20=0.2000 | 16/48=0.3333 | +0.4286 |

分意图（`sentence`）：`rate` 阈值 3/5 → 信号 5/5（全中）；`deadline` 2/7 → 6/7；
`amount` 6/8 → 6/8（漏 `009/011`，条款把门槛外指到交易所规则）；`subject` 2/8 → 2/8（几乎没动，
OR 只多 1 条：章程里"由…决定"太密）。唯一误拒是 `qa-007`：选中的条款写"在本章程规定的**合理期限**内
仍然有效"，真正的"六个月"在另一条里、不在 top-5 —— 信号只对"读到的那条"负责，检索没带回答案条款时
它救不了，反而会拒。

### 结论

1. **block 口径无效，不建议上线**：0/28，一条难例都没多拒，等于没接。
2. **sentence 口径有效但很薄**：同域难例 13/28 → 19/28（单独用）、20/28（与阈值 OR），
   代价是 0/15 → 1/15 误拒（`qa-007`）。净收益 +7 条难例 / +1 条误拒。
3. **不建议直接接入默认路径**：+7 条建立在 28 条同域难例上（1 份章程、1 种切块）；
   `block`/`sentence` 之间的选择是看过这批结果之后定的，**没有任何 held-out**；
   结果对"怎么选那一条条款"极其敏感——同一批题上另两种试法（每块取最重合句再合并 6/28、
   只查 rerank 第 1 块 9/28）差得很远，说明 19/28 里有多少是稳定增益、有多少是对这 28 条的形状，
   现在还分不清。要做成产品级，得先扩到第二份文档、并把"选条款"换成一个不依赖词面重合的判别器。
4. **它确实抓到了单阈值抓不到的东西**：15 条"分数高于可答下界"的难例里，有 7 条被 `sentence` 口径
   判了缺项——这正是上一节说"只能换信号"的那 15 条。

## 第二份文档交叉验证：信号在新文档上还成立吗（本次）

上一轮的结论卡在一句话上：19/28 里有多少是稳定增益、多少是对那 28 条题形状的拟合，分不清——
因为**没有第二份文档**。这一节把同一套信号搬到另一份章程（`zhicheng_meinuohua.pdf`，
宁波美诺华，50 页）上重跑一次。**信号代码一行没改**，改的只有语料和题。

### 1. 第二份文档入库（独立库，不碰 `real_dashscope.db`）

```bash
DOCUMIND_DATABASE_URL="sqlite+aiosqlite:///D:/code/documind/data/meinuohua_dashscope.db" \
./scripts/with_dashscope.sh ./.venv/Scripts/python.exe scripts/ingest_eval_doc.py \
    --pdf evals/datasets/zhicheng_meinuohua.pdf --strategy structure
```

| 项 | 值 |
|---|---|
| 库 | `data/meinuohua_dashscope.db`（新建；suhao 库与其 `eval_runs` 未动） |
| 页 / 段 / 字数 | 50 页 / 304 段 / 31 835 字（去空白 30 946） |
| 块（structure） | **52** 块，parents 0，平均 610.6 字符、最长 899 |
| embedding 成本 | **6 个请求 / 19 797 tokens**（供应商 `usage.prompt_tokens`，非估算） |
| 耗时 | **6 197 ms**（解析 + 切块 + 向量化 + 落库） |
| 空间指纹 | `text-embedding-v3@1024`（与 suhao 库同空间，但不能混库） |

`scripts/ingest_eval_doc.py` 是新加的：`ingest_corpus.py` 固定灌 3 份、`reingest_one.py` 只删重灌同库，
两者都会写进"当前库"；交叉验证要求第二份文档有自己的库，否则两套块混在一个检索空间里。

### 2. 题从哪来（可答 26 / 同域难例 24 / HR 20）

- **可答 26 条**：`qa_default.jsonl` 里 `gold_doc == 宁波美诺华药业股份有限公司章 程` 的题，**原样保留**，
  一条没改。逐条复核 `gold_quote` 是否在其 `gold_pages` 原文里：**26/26 命中**（机器校验，见下）。
  这批题是上一轮造题流程产出的、在**信号实现之前**就存在，所以它是成本端（误拒）的独立样本。

```bash
./.venv/Scripts/python.exe - <<'PY'
import json, re, sqlite3
con = sqlite3.connect("file:data/meinuohua_dashscope.db?mode=ro", uri=True)
pages = {int(n): (t or "") for n, t in con.execute("select page_no, text from pages")}
norm = lambda s: re.sub(r"\s+", "", s or "")  # noqa: E731
cases = [json.loads(l) for l in open("evals/datasets/qa_default.jsonl", encoding="utf-8") if l.strip()]
mine = [c for c in cases if c["answerable"] and c.get("gold_doc") == "宁波美诺华药业股份有限公司章 程"]
bad = [c["id"] for c in mine if not any(norm(c["gold_quote"]) in norm(pages.get(p, "")) for p in c["gold_pages"])]
print(f"可答 {len(mine)} 条 · gold_quote 不在 gold_pages 原文的: {len(bad)}", bad)
PY
# 可答 26 条 · gold_quote 不在 gold_pages 原文的: 0 []
```
- **同域难例 24 条**：新建 `evals/datasets/hard_neg_meinuohua.jsonl`，四类各 6 条，
  全部从美诺华章程的**原文条款**里造（做法与 `hard_neg_suhao.jsonl` 相同：条款在、值不在）。
  **没有改动**原有 28 条及其标注；`qa_default.jsonl` 也没动。
- **HR 固定 20 条**：原样拼进来，让"域外距离"那一列可比。

机检**复用**现有脚本（没有复制一份改改）：`verify_hard_neg.py` 本来就带 `--dataset` / `--db`。

```bash
./.venv/Scripts/python.exe scripts/verify_hard_neg.py \
    --dataset evals/datasets/hard_neg_meinuohua.jsonl --db data/meinuohua_dashscope.db
# [hard-neg] n=24 页数=50 全文=30946 字
# [分类] rate=6 · amount=6 · deadline=6 · subject=6
# [有争议] 3 条: hardneg-mnh-003, hardneg-mnh-009, hardneg-mnh-016
# [OK] 24/24 条通过：出处逐字可核 · 缺失词全文零命中 · 条款无数量词（除显式豁免）
```

四类样例（每条都带 `source_page` / `source_quote` / `anchor` / `abs_terms`）：

| 类 | 例 | 相似条款（页） | 缺的那一项 |
|---|---|---|---|
| rate | 任意公积金按什么比例提取 | 第一百五十五条（39） | 条款只写"还可以提取"，同页有法定公积金 10%/50% |
| amount | 员工持股计划最多授予多少股 | 第四十六条（14） | 只列股东会审议权，无授予数量 |
| deadline | 内审制度批准后几日内披露 | 第一百五十九条（43） | 只写"批准后实施并披露" |
| subject | 内审负责人由谁聘任解聘 | 第一百六十四条（43） | 只写"审计委员会参与考核" |

```bash
./.venv/Scripts/python.exe scripts/build_calib_dataset.py \
    --doc "宁波美诺华药业股份有限公司章 程" --hard evals/datasets/hard_neg_meinuohua.jsonl \
    -o evals/datasets/qa_meinuohua.jsonl
# [build] → 匹配 26 · 不可答 20 · 手写难例 24 · 合计 70
```

### 3. 检索 + 四组合（同一脚本，`--threshold 1.5055`）

```bash
DOCUMIND_DATABASE_URL="sqlite+aiosqlite:///D:/code/documind/data/meinuohua_dashscope.db" \
./scripts/with_dashscope.sh ./.venv/Scripts/python.exe -u scripts/eval_retrieval.py \
    --dataset meinuohua --strategy structure --ans-limit 26 --unans-limit 44 --dump-hits 5 \
    --json evals/reports/calib_meinuohua_hits.json
./.venv/Scripts/python.exe scripts/refusal_signal_report.py \
    --json evals/reports/calib_meinuohua_hits.json --dataset evals/datasets/qa_meinuohua.jsonl \
    --threshold 1.5055 --top-n 5 --md evals/reports/refusal_signal_meinuohua.md
```

检索：`recall_at_5=1.0（26/26）`、`context_recall=1.0`、`mrr=0.9808`、`latency_p50=2541 ms`。
新库没有 `eval_runs` 行，脚本自己解析到 `builtin -6.0`（日志里可见），所以下面的"threshold"列
**一律用手算的 1.5055** 表示"现行阈值"（就是 `/health` 现在对 suhao 库报的那个值）。

**`scope=block`（需求原始口径）——又一次恒假：**

| 组合 | 可答误拒 ↓ | 同域难例拒答 ↑ | HR 拒答 | 全部不可答拒答 | 拒答率−误拒率 |
|---|---|---|---|---|---|
| threshold 1.5055 | 2/26=0.0769 | 10/24=0.4167 | 19/20=0.9500 | 29/44=0.6591 | +0.3397 |
| signal | 0/26=0.0000 | **0/24=0.0000** | 0/20 | 0/44 | +0.0000 |
| OR | 2/26=0.0769 | 10/24=0.4167 | 19/20 | 29/44 | +0.3397 |
| AND | 0/26 | 0/24 | 0/20 | 0/44 | +0.0000 |

第二份文档复现了同一个结论：块级口径下形态**一条都不缺**（美诺华每页同样全是百分比/日期）。

**`scope=sentence`（形态必须在"与问题词面最重合的那一条"里）：**

| 组合 | 可答误拒 ↓ | 同域难例拒答 ↑ | HR 拒答 | 全部不可答拒答 | 拒答率−误拒率 |
|---|---|---|---|---|---|
| threshold 1.5055 | 2/26=0.0769 | 10/24=0.4167 | 19/20=0.9500 | 29/44=0.6591 | +0.3397 |
| signal | 3/26=0.1154 | 20/24=**0.8333** | 3/20=0.1500 | 23/44=0.5227 | +0.7179 |
| **OR** | 5/26=0.1923 | 22/24=**0.9167** | 19/20 | 41/44=0.9318 | +0.7244 |
| AND | 0/26=0.0000 | 8/24=0.3333 | 3/20 | 11/44=0.2500 | +0.3333 |

### 4. 两份文档并排（同一张表，同一口径）

| | suhao（59 页 / 28 难例 / 15 可答） | 美诺华（50 页 / 24 难例 / 26 可答） |
|---|---|---|
| signal 拒同域难例 | 19/28 = 0.6786 | **20/24 = 0.8333** |
| OR(阈值, signal) | 20/28 = 0.7143 | **22/24 = 0.9167** |
| signal 误拒可答 | 1/15 = 0.0667 | 3/26 = 0.1154 |
| OR 误拒可答 | 1/15 = 0.0667 | 5/26 = 0.1923 |
| 阈值误拒可答 | 0/15 = 0.0000 | **2/26 = 0.0769** |
| 可答最低 top_score | 1.5759 | **0.0706** |

配对（52 条难例合并，signal vs 阈值 1.5055）：两者都拒 20、**只 signal 拒 19**、只阈值拒 3、
都漏 10 → McNemar 精确双侧 **p = 0.00086**（3 vs 19 的失衡不是抽样噪声）。

分类（signal，两份文档各自独立）：`rate` 5/5 → 5/6；`amount` 6/8 → 6/6；
`deadline` 6/7 → 6/6；`subject` **2/8 → 3/6**。前两类（含时限）在两份文档上都稳，
`subject` 在两份文档上都弱——这是"形状"层面的复现，而不是单份文档的偶然。

### 5. 结论

**信号在新文档上仍有效，但幅度与代价都比第一份文档更保守地说明它不是一个"零成本"增益。**

具体数字：

1. `scope=sentence` 单独用：同域难例 **20/24 = 0.8333**（阈值 1.5055 只有 10/24 = 0.4167），
   与阈值 OR 后 **22/24 = 0.9167**。方向、量级、分类结构都与第一份文档一致
   （suhao 19/28 → OR 20/28；mnh 20/24 → OR 22/24）。
2. `scope=block` 在第二份文档上**仍然 0/24**——这条结论是稳定的（恒假，不是偶发）。
3. 增益的独立性有限：52 条难例里，signal 多拒 19 条、阈值多拒 3 条（p≈0.0009），
   但**其中 24 条难例是我读过信号实现之后按四类模板写的**，不是盲测；能算独立的那一侧只有
   误拒率（题在信号之前就存在），而它从 6.7% 升到 **11.5%**（OR 下 19.2%）。
4. 阈值本身不跨文档：同一份 1.5055 在 suhao 上 0 误拒，在美诺华上 2/26 误拒
   （可答最低分从 1.5759 掉到 0.0706）。**能迁移的是形态信号，不是分数**。

**是否建议上线：不建议（现在）。**

- 收益侧（多拒 20 条难例）成立，但成本侧关键指标变差了：OR 模式下每 5 个问题就有 1 个被误拒
  （美诺华 5/26），而误拒是用户直接可见的错误，比"该拒没拒"更贵。
- 4 条误拒（suhao 的 `qa-007`、美诺华的 `-043/-056/-065`）是**同一个缺陷**：答案条款在文档里，
  但"与问题词面最重合的那一条"不是它，形态检查就判了缺项。两份文档各出现一次以上，
  说明这不是这批题的噪声，而是 `_narrow_to_matching_clause` 这个启发式的系统性问题。
- 也就是说：信号抓的是"读到的那条没有值"，能不能上线取决于**先选对条款**。在条款选择换成
  不依赖词面重合的判别器（或改为对 top-k 全查、只在全都缺时才判拒）之前，默认路径不该接。
- 若一定要接，唯一可接受形态是"只做提示、不做拒答"（在回答里加"文档未给出具体数值"的免责），
  或把 `subject` 意图整个关掉（两份文档上 5/14，几乎等于没接）。

## 阈值桶键：从 `reranker` 到 `(reranker, embed_space)`（本次）

上一节最后一条结论是"阈值本身不跨文档"。这一节把那条结论变成**结构上的约束**：
`eval_runs` 加 `embed_space`，`resolve_refusal_threshold` 按 `(reranker, embed_space)` 查最近一次标定。

| 场景 | 现在的行为 | `/health.refusal_threshold` |
|---|---|---|
| 当前空间有标定 | 用它（`source=calibrated`） | `calibrated_spaces` 含当前空间，`warning=""` |
| 当前空间无标定、别的空间有 | **不用**别的空间的值，退回 builtin | `other_space_calibration` 报出那个值 + `warning` 写明"未生效" |
| 只有迁移前的行（`embed_space=''`） | **不用**（无法证明属于哪个空间），退回 builtin | `legacy_calibration` 报出那条行 + `warning` 给出回填命令 |
| `DOCUMIND_REFUSAL_THRESHOLD` 已设置 | 覆盖（优先级最高） | `source=env`；与本空间标定不一致时 `mismatch=true` |

`reason` 字段始终回答"为什么现在生效的是这个值"；`warning` 只在需要动作时非空。

### 迁移：老库的行不自动生效

`data/real_dashscope.db`（suhao 库）里那条 `1.5055` 是 P1-10 之前写的，`embed_space` 为空。
升级后 `/health` 会变成 `source=builtin`、`value=-6.0`，并报出那条历史行：

```bash
DOCUMIND_DATABASE_URL="sqlite+aiosqlite:///D:/code/documind/data/real_dashscope.db" \
./scripts/with_dashscope.sh ./.venv/Scripts/python.exe -m uvicorn documind.app:app --port 8961
curl -s http://127.0.0.1:8961/health | jq .refusal_threshold
# {"reranker":"BAAI/bge-reranker-base","value":-6.0,"source":"builtin",
#  "calibrated":null, ... ,
#  "warning":"... 另有 1 条无空间标注的历史标定行 value=1.5055（eval_run=04ea166191d245c4，数据集 suhao，
#  2026-11-08T00:30:36.058675），按“历史通用值”处理：只告警、不生效（无标注的行无法证明属于哪个空间，
#  猜错就等于跨库继承）。确认它属于哪个空间后跑
#  `python scripts/backfill_calibration_space.py --space <model@dim>` 回填",
#  "legacy_calibration":{"embed_space":"","value":1.5055,"run":"04ea166191d245c4","dataset":"suhao", ...}}
```

回填是一条命令，且**默认读库自己的空间**而不是进程的（换过 backend 时进程的空间常常不是库的空间，
拿它回填等于把错的空间写进历史行）：

```bash
DOCUMIND_DATABASE_URL="sqlite+aiosqlite:///D:/code/documind/data/real_dashscope.db" \
./.venv/Scripts/python.exe scripts/backfill_calibration_space.py --dry-run
# [library] 库里的向量空间: ['text-embedding-v3@1024']
# [warn] 当前进程的 embedder 空间是 BAAI/bge-small-zh-v1.5@512，与回填空间 text-embedding-v3@1024 不同（不影响写入）
# [plan] 回填空间 text-embedding-v3@1024，影响 1 行：
#   - 04ea166191d245c4 ... threshold=1.5055 ...
# [dry-run] 未写入。
```

去掉 `--dry-run` 之后 `/health` 变成 `source=calibrated`、`value=1.5055`、`calibrated_spaces=["text-embedding-v3@1024"]`。

### 三种 `/health` 输出（真实，2026-11-08）

**① 老库：历史行只告警不生效**（`data/real_dashscope.db` 的副本，未回填；embedder=dashscope）

```json
{"embed_space":"text-embedding-v3@1024",
 "refusal_threshold":{"reranker":"BAAI/bge-reranker-base","value":-6.0,"source":"builtin",
  "calibrated":null,"calibrated_run":"","calibrated_at":"","calibrated_dataset":"",
  "mismatch":false,
  "warning":"reranker=BAAI/bge-reranker-base · 向量空间 text-embedding-v3@1024 还没有标定值，使用内置占位 -6.0000；跑一次 `documind eval` 才有可追溯的阈值。另有 1 条无空间标注的历史标定行 value=1.5055（eval_run=04ea166191d245c4，数据集 suhao，2026-11-08T00:30:36.058675），按“历史通用值”处理：只告警、不生效（无标注的行无法证明属于哪个空间，猜错就等于跨库继承）。确认它属于哪个空间后跑 `python scripts/backfill_calibration_space.py --space <model@dim>` 回填",
  "embed_space":"text-embedding-v3@1024","calibrated_spaces":[],
  "other_space_calibration":null,
  "legacy_calibration":{"embed_space":"","value":1.5055,"run":"04ea166191d245c4","at":"2026-11-08T00:30:36.058675","dataset":"suhao"},
  "reason":"reranker=BAAI/bge-reranker-base · 向量空间 text-embedding-v3@1024 没有任何可生效的标定值（该空间无标定、仅有无空间标注的历史行不用），故使用内置占位 -6.0000"}}
```

**② 换空间：别的空间的 1.5055 不继承**（同一个库，但进程跑在 hash 空间）

```json
{"embed_space":"hash-bigram@512",
 "refusal_threshold":{"reranker":"BAAI/bge-reranker-base","value":-6.0,"source":"builtin",
  "calibrated":null,"mismatch":false,
  "warning":"向量空间 hash-bigram@512 还没有标定值，拒答阈值退回内置占位 -6.0000；已知空间 text-embedding-v3@1024 有标定值 value=1.5055（eval_run=04ea166191d245c4，数据集 suhao，2026-11-08T00:30:36.058675），但**未生效**——阈值不跨空间继承（不同空间的 top_score 分布整体平移，同一数值落到这里会变成另一个误拒/漏拒水平）；要在 hash-bigram@512 上生效，需在该空间重跑一次标定",
  "embed_space":"hash-bigram@512","calibrated_spaces":["text-embedding-v3@1024"],
  "other_space_calibration":{"embed_space":"text-embedding-v3@1024","value":1.5055,"run":"04ea166191d245c4","at":"2026-11-08T00:30:36.058675","dataset":"suhao"},
  "legacy_calibration":null,
  "reason":"reranker=BAAI/bge-reranker-base · 向量空间 hash-bigram@512 没有任何可生效的标定值（该空间无标定、其它空间有但按空间隔离不用），故使用内置占位 -6.0000"}}
```

**③ 同空间已标定：生效**（同一副本回填 `text-embedding-v3@1024` 之后）

```json
{"embed_space":"text-embedding-v3@1024",
 "refusal_threshold":{"reranker":"BAAI/bge-reranker-base","value":1.5055,"source":"calibrated",
  "calibrated":1.5055,"calibrated_run":"04ea166191d245c4","calibrated_at":"2026-11-08T00:30:36.058675",
  "calibrated_dataset":"suhao","mismatch":false,"warning":"",
  "embed_space":"text-embedding-v3@1024","calibrated_spaces":["text-embedding-v3@1024"],
  "other_space_calibration":null,"legacy_calibration":null,
  "reason":"reranker=BAAI/bge-reranker-base · 向量空间 text-embedding-v3@1024 最近一次标定（eval_run=04ea166191d245c4，数据集 suhao，2026-11-08T00:30:36.058675）写回 1.5055；同空间没有更新的标定，环境变量未设置"}}
```

### 回归测试

`tests/test_threshold_space_bucket.py`（7 条），其中两条是这一节的事实：

- `test_suhao_threshold_does_not_leak_into_meinuohua_space`：`1.5055`（无论带空间还是历史行）在
  另一个空间上解析时 `source=builtin`、`value=-6.0`、`calibrated=None`，且 `warning` 写明"未生效"。
- `test_same_space_different_corpus_still_shares`：**把边界也钉住** —— suhao 与美诺华用的是同一个
  `text-embedding-v3@1024`，所以空间分桶**拦不住**这条真实泄漏路径。它能生效的前提是那两批块在同一个库里；
  真跨库时 `eval_runs` 不共享，才没有实际泄漏。要在同一库里隔离两份语料，桶键还需要语料维度（未做）。

### 已知未覆盖（写在这里而不是藏起来）

1. `(reranker, embed_space)` 只区分"换 embedder"，不区分"同 embedder 换语料"。真实数据里 suhao 与美诺华
   同空间，同一份 1.5055 在后者上误拒 2/26、同域难例只拒 10/24。
2. 标定行没有"这份语料有多少题/多少页"的门槛，`n_cases` 只进 `metrics`，解析时不看。用小样本标出来的值
   与 63 题标出来的值在 `/health` 里长得一样（只有 `calibrated_dataset` 能提示来源）。
3. `eval_retrieval.py` 在一个**全新空库**上仍会因 `eval_runs` 不存在而报 `no such table`（它只补列不建表）；
   用已存在的库或在之前跑一次 `init_db` 不受影响。这是既有行为，本单没改。

## 诚实边界

- 美诺华那 24 条同域难例是**读过信号实现之后**按四类模板写的（非盲测）：增益侧（20/24、22/24）
  可能被高估；能算独立的是误拒侧（`qa_default` 里 26 条可答题早于信号存在）。
- judge 与被测模型是同一个模型（自评有偏差），已在报告里标注。
- 评测集是合成的，不是真实用户日志；`false_refusal_rate` 的绝对值会偏乐观。
- 缺项信号的词表/句式是中文、章程语境（`%`、元/万元、日/月、由…决定）；换英文合同或年报要重写形态，
  且 `2023 年` 这类年份会被"时限形态"误当成有期限。
- `sentence` 口径的"最重合那一条"是字符二元组重合度，不是语义；它在这份章程上有效，
  不代表在别处有效（见上面第 3 条）。
- 语料只有章程一类文本，切块结论不可外推到合同/代码/工单。