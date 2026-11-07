# 重排性能：21.9s → 2.8s，召回不降

范围：`structure` 策略下的 cross-encoder 重排（`BAAI/bge-reranker-base`，本地 ONNX CPU）。
机器：i5-12500H（12 核 / 16 线程，2.5GHz 基频），Windows 11，ONNX Runtime 1.31.1，
`fastembed` 默认线程（= 物理核）。所有延迟均为本机实测、3 次以上取中位。

## 结论

1. **慢的原因不是"模型慢"，是"送进去的 token 太多"**：30 条候选每条都被截到 512 token
   （`tokenizer.json` 自带 `truncation.max_length=512`，而块文本平均约 710 字 > 512 token），
   合计 15 360 token。单批 forward 占 21 267 ms 里的 **99.1%**，分词只占 13 ms。
2. 本机 fp32 bge-reranker-base 的稳定吞吐约 **700 token/s（1.4ms/token）**，与批大小基本无关。
   于是 `延迟 ≈ token 数 / 700`，21.9s 就是 15 360 token 的算术结果。
   线程、模型加载、`inference_mode`、逐条 tokenize 全部排查过，**都不是病因**（见 §2）。
3. 把输入量压到 **1 930 token**（候选 30→10；送模型前按问题挑句窗口，256 字预算；
   长度聚合组批）：重排 P50 **21 267 ms → 2 711 ms（7.8x）**，
   `retrieve()` 端到端 P50 **21 679 ms → 2 821 ms**，首字节（TTFB）**18.8s → 1.6s**。
4. 召回没有下降：Recall@5 **0.900 → 0.950**（18/20 → 19/20），Context Recall
   **0.900 → 0.950**（18/20 → 19/20），MRR **0.7667 → 0.7792**。
   n=20，1 题 = 5 个百分点，所以正确说法是"**没有回归**"，不是"变好了"。
5. **"30 条候选 < 1s" 在这台机器上用 fp32 做不到**，账算得清：1s ≈ 700 token，
   30 条候选每条只剩 23 token。要做得到必须换精度或模型：int8 动态量化实测 **2.25x**
   （§5，已评估、未接入），叠加 128 字窗口可到 ~0.6-0.8s，但会动 MRR。

## 1. 分段计时：21 863 ms 花在哪

同一批 30 条真实候选（来自 `pools[0]`，合计 21 688 字，512 token × 30 = 15 360 token）：

| 项 | 耗时（中位，3 次） | 占比 |
|---|---|---|
| 只分词（一次 `encode_batch`，30 条） | 13 ms | 0.06% |
| 只 forward（单批 `(30, 512)`） | 21 086 ms | 99.1% |
| `fastembed.rerank()` 全量（含分词 + 组批 + forward） | 21 267 ms | 100% |
| 模型构造（`get_reranker` 命中 `lru_cache`） | 5 ms | — |

709 ms/条 = 512 token × 1.39 ms/token。**没有任何一项是"隐藏开销"**：这是纯算力。

参考算力基线：同机 numpy/OpenBLAS `1024³` 单精度 GEMM = 212.8 GFLOPS；
ORT 跑到约 121 GFLOPS（≈ 57% 的 GEMM 峰值，BERT 的 attention/softmax/layernorm 是访存瓶颈），
属于正常区间，不是"ORT 没吃满 CPU"。

## 2. 常见病因逐条排除

| 假设 | 实测 | 判定 |
|---|---|---|
| 每条候选都吃满 512 token | `tokenizer.json` 的 `truncation.max_length=512`；块平均 710 字，分词后长度 min/med/max = **512/512/512** | **命中**（输入量上限，也是本次优化的主战场） |
| batch=1 | fastembed 默认 `batch_size=64`，30 条一批 | 未命中 |
| 未用 `torch.inference_mode()` | 不走 torch：fastembed 用 onnxruntime，无 autograd、无梯度开关 | **不适用** |
| 线程数=1 | ORT 默认=物理核。显式 `threads` 扫描：1/2/4/8/16 线程 = 108/208/324/530/**713** token/s（近线性，16 线程最好） | 未命中（已饱和） |
| 反复加载模型 | `get_reranker` 是 `lru_cache`，一次进程内只构造一次（5 ms） | 未命中 |
| 每条候选单独 tokenize | 一次 `encode_batch(30 条)` = 13 ms | 未命中 |

结论：只剩下两个真变量——**送几条**、**每条送多少 token**。

## 3. 改了什么（按收益排序，每条都有前后对比）

| # | 改动 | 位置 | 收益 | 召回影响 |
|---|---|---|---|---|
| 1 | 候选数 30 → 10（RRF 顺序截断，不低于 `top_k`） | `retrieve.py:226`、`config.py:rerank_candidates` | 3.0x | 无（见 §4，反而 +1 题） |
| 2 | 送模型前按问题挑句窗口（`focus`，256 字预算） | `rerank.py:_window`、`config.py:rerank_truncate/rerank_max_tokens` | 2.6x（实际 193 token/条） | 无；**head 截断会掉**，见 §4 |
| 3 | 自己组批：长度排序 + 小批（`batch_size=8`） | `rerank.py:score` | 本语料 ~0（所有块都超上限，长度一样）；短块语料才省 padding | 无 |
| 4 | 线程数显式化（`rerank_threads`，默认 0=ORT 默认） | `config.py` | 0（已饱和，留成可配置） | 无 |
| 5 | tokenizer 批处理 | 本来就有（13 ms/30 条） | 0 | 无 |

改前改后的实现差别：原来一行 `self._model.rerank(query, docs)` 把 30 条塞进一批、
按 512 截断；现在复用同一个 fastembed 模型（同一份 tokenizer + 同一个 ONNX session，
不重复加载），自己控制"挑窗口 → 长度排序 → 8 条一批 → 拼 `input_ids/token_type_ids/attention_mask`"。

## 4. 召回：改前 / 改后并列

命令（不调 LLM judge，只用 `evals.metrics` 的确定性指标；20 可答 + 8 不可答 = 28 题，
与 `run_ablation_parallel.py --ans-limit 20 --unans-limit 8` 同一批）：

```bash
DOCUMIND_DATABASE_URL="sqlite+aiosqlite:///<你的库>" \
  ./.venv/Scripts/python.exe scripts/eval_retrieval.py --strategy structure
```

| 指标 | 改前（HEAD bd15490） | 改后（focus@256 × 10） | 变化 |
|---|---|---|---|
| Recall@5 | 0.900（18/20） | **0.950（19/20）** | +1 题 |
| Context Recall | 0.900（18/20） | **0.950（19/20）** | +1 题 |
| MRR | 0.7667 | **0.7792** | +0.0125 |
| 拒答准确率（pre-LLM，内置阈值 -6.0） | 0.375 | 0.500 | +0.125 |
| 误拒率（pre-LLM） | 0.000 | 0.000 | 0 |
| 重排分数分布（可答题最低 top_score） | 1.7161 | 2.9237 | 分数域整体上移 |
| 重标后推荐阈值 | 1.7971 | 2.9397 | **必须重标** |
| 重标后 拒答准确率 / 误拒率 | 1.000 / 0.050 | 1.000 / 0.050 | 一致 |
| `retrieve()` P50 / P95 | 21 679 / 21 962 ms | **2 821 / 3 063 ms** | 7.7x |

**为什么要"按问题挑窗口"而不是直接截前 N 个 token**（同一批 28 题，重放同一候选池）：

| 截断方式 | Recall@5 | Context Recall | MRR |
|---|---|---|---|
| head@512 × 30 条（改前） | 0.900 | 0.900 | 0.767 |
| head@256 × 30 条 | 0.900 | **0.650** | 0.637 |
| head@128 × 30 条 | **0.800** | **0.500** | 0.613 |
| focus@256 × 30 条 | 0.950 | 0.950 | 0.758 |
| focus@256 × 10 条（**采用**） | 0.950 | 0.950 | 0.779 |
| focus@192 × 10 条 | 0.950 | 0.900 | 0.742 |
| focus@128 × 10 条 | 0.950 | 0.950 | 0.750 |

直接 head 截到 256：Context Recall 0.900 → 0.650（20 题里掉 5 题）——因为块平均 710 字，
gold 句常常在第 256 字之后。`focus` 先按句切、取与问题字面最重合的句子再向两侧扩到预算，
所以窗口小了 2/3 还能覆盖到 gold 句。
`focus@128`（1.5s，见 §5）Recall 两项都不低于改前，但 MRR 0.750 < 0.7667，所以**默认没选它**，
留成 `DOCUMIND_RERANK_MAX_TOKENS=128` 的一行开关。

## 5. 延迟前沿（同一候选池，3 次中位）

| 配置 | 送模型 token | 重排 P50 | 相对改前 |
|---|---|---|---|
| head@512 × 30（改前） | 15 360 | 21 267 ms | 1.0x |
| focus@256 × 30 | 5 790 | 7 886 ms | 2.7x |
| focus@256 × 15 | 2 895 | 3 973 ms | 5.4x |
| **focus@256 × 10（采用）** | **1 930** | **2 711 ms** | **7.8x** |
| focus@192 × 10 | 1 520 | 2 320 ms | 9.2x |
| focus@128 × 10 | 1 110 | 1 491 ms | 14.3x |
| int8 focus@256 × 10（**评估用，未接入**） | 1 930 | 1 327 ms（同 harness fp32 2 396 ms） | 16x |

端到端（真网关 `qwen-cn/qwen3.8-flash`，`scripts/measure_stream.py --runs 3`）：

| 指标 | 改前 | 改后 |
|---|---|---|
| evidence（检索完成 / 首字节） | 18 824 ms | **1 609 ms** |
| first_token | 33 048 ms | **8 180 ms** |
| answer（端到端） | 34 658 ms | **9 490 ms** |
| first_token 占端到端 | 95.4%（FAIL） | **86.2%（仍 FAIL）** |

**为什么 86.2% 还是 FAIL**：改后检索只剩 1.6s，剩下的 6.6s 是网关自己的首 token 时间
（prompt 5 条上下文 + 网关排队）。也就是说 `<30%` 这条指标现在**被生成侧卡住**，不是检索侧的锅；
要继续压只能动 prompt 长度 / 换更快的模型 / 减少上下文条数，属于另一个议题。

## 6. 没解决的部分：<1s 的账

- 1s 预算 ÷ 700 token/s ≈ **700 token**。30 条候选 = 每条 23 token（约 23 个汉字），
  已不是重排，是关键词匹配。
- 本配置能做到 **2.7s**（10 条 × 193 token）。再往下只有三条路，都要付代价：
  1. **int8 动态量化**（已评估，`onnxruntime.quantization.quantize_dynamic`，29s 量化出 279MB 模型）：
     实测 fp32 6 673 ms → int8 2 969 ms（30 条，**2.25x**），10 条 2 396 → 1 327 ms（1.8x）；
     排序几乎不变（top-5 5/5 重合、top-10 10/10、Spearman 0.98）。
     **未接入**：量化需要 `onnx` 包（当前不是项目依赖），且会新增一个模型产物与一条构建路径。
     若接入，本配置 ~1.5s，叠加 `max_tokens=128` ~0.8s。
  2. **换更小的 reranker**：bge-reranker-base 是 12 层/768 维；换成 6 层级别（或中文蒸馏版）
     能直接砍一半算力，但分数域变了 → 拒答阈值必须整体重标，且要重跑 judge 侧指标。
  3. **降低 top_k 或召回条数**：动的是产品语义（引用条数），不是在提速。
- **未测**：judge 侧指标（faithfulness / answer_relevancy）本轮没跑——本次只改检索，
  且 judge 与生成模型同源、噪声会盖住 1 题的差异；要发版前仍应跑一次完整 `documind eval`。
- **阈值漂移**：`focus` 窗口让重排分数整体上移（可答题最低分 1.72 → 2.92），
  **换截断策略必须重标拒答阈值**（`documind eval` 会自动写回 `eval_runs.recommended_threshold`）。
  当前 dev 库对 `BAAI/bge-reranker-base` 还没有标定值（走内置占位 -6.0），这一点在改前改后都一样。
- **只测了 structure 策略**：其他三种切块策略的召回没重跑（重排改动与切块无关，按理同样受益）。
- **长度分桶组批的实际收益是 0**：本语料每个块都超过截断上限，长度全一样。
  保留它是为了短块语料（表格类、FAQ）不浪费 padding，但这台机器上没测出收益。

## 7. 复现

```bash
# 1) 延迟（分段计时 + 配置前沿）
./.venv/Scripts/python.exe -m pytest tests/test_rerank_perf.py -q      # 含 <5s 的回归钉子

# 2) 召回（不调 LLM judge）
./.venv/Scripts/python.exe scripts/eval_retrieval.py --strategy structure

# 3) 端到端 TTFT（服务先起来）
./.venv/Scripts/python.exe -m uvicorn documind.app:app --port 8080 &
./.venv/Scripts/python.exe scripts/measure_stream.py --runs 3

# 4) 想回到改前行为（对照用）：全量 30 条 × 512 token
DOCUMIND_RERANK_CANDIDATES=30 DOCUMIND_RERANK_MAX_TOKENS=512 \
DOCUMIND_RERANK_TRUNCATE=head ./.venv/Scripts/python.exe scripts/eval_retrieval.py
```