# ADR-0004 拒答：阈值由评测集 P-R 曲线标定，不是拍脑袋的 0.6

- 状态：已采纳（2026-11）

## 背景

"检索分数低于 0.6 就拒答"是最常见的写法，但：
- 重排分是 cross-encoder 的 logits（可正可负），没有 0-1 的含义；
- 阈值直接决定**漏答**与**错答**的兑换比例，是业务决策，不是技术参数。

## 决策

1. 拒答有三道判定，任一条命中即拒答：
   - a) 检索为空（ACL 过滤后无候选）；
   - b) 最高重排分 < `refusal_rerank_threshold` → 直接拒答，**不调用模型**（省钱且不给出编造机会）；
   - c) 模型输出 `not_found=true`，或输出的引用无法通过校验（ref 越界 / quote 在原文中不存在）。
2. 阈值 `refusal_rerank_threshold` 由 `evals` 的标定器扫描得到：在误拒率 ≤ 15% 的前提下取拒答准确率最高的点，
   规则写死在 `evals/runner.py:calibrate_threshold` 并可复现。
3. 评测集里固定包含 20 条**语料中确实没有答案**的问题（年假折现、报销标准、工资发放日等），
   拒答准确率就是对这 20 条的口径。

## 取舍

规则是"宁可多拒，不可错答"：制度问答里一次编造的"可以折现"会导致真实的用工风险；
而多拒一次只是用户换个问法或去看原文。这个偏好是业务给的，不是模型给的，所以写在代码里而不是 prompt 里。

## 放弃了什么

- 分层阈值（按文档类型/问题类型给不同阈值）：样本量还不够支撑分组标定，过拟合风险大于收益。
- 用 LLM 自己判断"要不要拒答"：模型对"资料里没有"的自我判断显著偏向过度自信，实测不如分数阈值可靠。

## 实现状态（P1-03 回写补丁，2026-11）

原实现只有决策、没有闭环：阈值是 `config.py` 里写死的 `-6.0`，标定器只把建议写进报告 HTML。
现在的落地方式（代码见 `src/documind/thresholds.py`）：

1. **按 reranker 分桶**：桶的键是 `reranker.name`（`lexical` / `BAAI/bge-reranker-base`）。
   两个分数域（[-12,0] 与 logits）不再共用一个数值。
2. **标定回写**：`calibrate_threshold` 的结果写入 `eval_runs.recommended_threshold` + `eval_runs.reranker`，
   服务启动/每次检索按 name 取最近一次标定值。
3. **优先级**：`DOCUMIND_REFUSAL_THRESHOLD`（显式覆盖） > 最近一次标定值 > 内置占位（`/health` 标 `builtin` + 警告）。
   覆盖值与标定值不一致时打 WARNING，并在 `/health.refusal_threshold.mismatch` 亮出。
4. **网格由数据决定**：按 top_score 的分位数生成，覆盖 `[min-1, max+1]`，
   不再写死 `[-14, 0]`（那个区间扫不到 cross-encoder 的正值侧）。

## 修订（P1-10，2026-11）：桶键加向量空间

**事实**：2026-11-08 的交叉验证把同一份标定值搬到第二份文档上，可答题最低分从 1.5759 掉到 0.0706，
同域难例只拒 10/24。**阈值不跨库**，而当时的桶键只有 `reranker`——换库时会静默继承旧值。

**决策**：桶键 = `(reranker.name, embed_space)`，`eval_runs` 增 `embed_space` 列。

1. 解析：当前空间有标定才用；没有就退回内置占位，并把"别的空间有值但未生效"作为
   `other_space_calibration` + `warning` 报出来（`/health`、启动日志、检索路径同一份解析）。
2. 迁移前的行 `embed_space=''`：**只告警、不生效**（无标注的行无法证明属于哪个空间，
   自动归属等于重现跨库继承）。回填是一条显式命令：`scripts/backfill_calibration_space.py`，
   默认取库自身的 `documents.embed_space` 而不是进程的空间。
3. 写路径：`evals/runner.py` 与 `scripts/eval_retrieval.py --write` 都带空间落库。

**当时没有解决**（P1-12 已修，见下一节）：两份语料共用同一个 embedder 时空间相同，标定值仍会互相生效
（真实数据就是这种情况）。**不要**用"已按空间分桶"当作跨库问题已解决的证据。

## 修订（P1-12，2026-11）：桶键加语料指纹

**事实**：suhao 与美诺华用的是同一个 embedder `text-embedding-v3@1024`，所以 `(reranker, embed_space)`
这一层根本不区分它们：1.5055 在后一份上误拒 2/26、同域难例只拒 10/24。空间分桶拦的是"换 embedder"，
不是"换语料"。

**决策**：桶键 = `(reranker.name, embed_space, corpus_fp)`，`eval_runs` 增 `corpus_fp` 列。

1. `corpus_fp` = 当前库 `documents.sha256` **排序后**再 sha256 的前 16 位；空库为空串。
   排序是必须的：同一份语料、不同入库顺序不该产生两个桶。
2. 解析：三者全中才用 calibrated；空间相同、语料不同 → 退回 builtin，`/health` 用
   `other_corpus_calibration` + `warning`（"因为语料变了"）解释，并用 `calibrated_corpora`
   列出已知语料（fp 前 8 位 + 数据集名 + 值）——**只列不用**。
3. 迁移前的行 `corpus_fp=''`（以及空空间行）：**只告警、不生效**，回填仍是显式命令
   `scripts/backfill_calibration_space.py`（`--corpus-fp` 缺省取库的指纹，幂等、有 `--dry-run`）。
4. 写路径（`evals/runner.py`、`scripts/eval_retrieval.py --write`）把标定时库的指纹一并落库。

**代价**：语料指纹是**整库**的集合指纹——同一个库里换掉一篇文档，其余文档的标定也一起失效。
这是有意的取舍：失效会被 `/health` 报出来、重跑一次标定即可恢复；静默继承一个属于别的语料的阈值
则是线上误拒，没人能从返回值里看出来。