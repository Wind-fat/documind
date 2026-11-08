# 真实供应商实测：DashScope compatible-mode（LLM + embedding）

本文件记录**真调用**的一次完整跑法：LLM 直连 DashScope、embedding 走新增的远端后端、
用真 PDF 入库、真问题问答、引用 span 回来原文里逐字校验。所有数字都来自本次运行的响应与
trace，不是推算；key 不在此文件、也不在仓库任何文件里（运行时从本机凭据文件读进环境变量）。

日期：2026-11-08 · 机器：Windows 11 / i5-12500H · 供应商 endpoint：`https://dashscope.aliyuncs.com/compatible-mode/v1`
模型：`qwen3.8-flash`（LLM）、`text-embedding-v3`（embedding，1024 维）

## 1. 结论（先说结果，再说证据）

| 项 | 结果 |
|---|---|
| LLM 直连（非流式 `/query`） | 200，答案 + 1 条 citation 回到第 48 页 460–541 字符，`span_source=anchored` |
| LLM 直连（流式 `/query/stream`） | 200 `text/event-stream`，`evidence → 7×token → answer`，25 个上游 chunk，服务端 TTFT **4210 ms**，带 usage |
| 新增 embedder 后端 | `dashscope`（实现是通用 OpenAI 兼容 `/embeddings`），真入库 47 块 / **5 个请求** / **17 951 embedding token** / 5.2 s |
| 空间指纹 | `text-embedding-v3@1024` 同时写进 `documents.embed_space` 与每块 `embed_model/embed_dim`；换回 hash 查询 → **HTTP 409** |
| 失败不降级 | 后端 4xx/5xx、维度不符、配置缺失都抛 `EmbeddingBackendError` + `/health.degraded=["embedder"]`，**不会**换成 hash 空间 |
| 端到端引用校验 | 5 问（4 可答 + 1 应拒答）：span 命中 **4/4 = 1.0**、引用有效率 **1.0**、拒答 **1/1**、误拒 **0** |
| 思考开关（答案档） | 关掉后 completion **−66.1%**、单问成本 **−12.7%**、平均延迟 **−16.0%**；本批 4 条引用的 span 命中率与拒答判定**未变** |
| 测试 / 静态检查 | `pytest -q` → **139 passed**（基线 124 + 新增 15）；`ruff check .` → All checks passed |

一句话取舍：**思考开关在答案档保持默认开启**（本批样本太小，只证明"没变差"，不足以证明
"普遍不变"）；**短任务（query 改写、自检判定）默认关闭**。真正的成本大头不是 completion 而是
每问约 4.3k 的 prompt（上下文），见 §7。

## 2. 怎么接（不需要改任何硬编码）

三个环境变量就够，全部有前缀 `DOCUMIND_`：

| 变量 | 本次取值 | 说明 |
|---|---|---|
| `DOCUMIND_LLM_BASE_URL` | `https://dashscope.aliyuncs.com/compatible-mode/v1` | OpenAI 兼容端点 |
| `DOCUMIND_LLM_API_KEY` | 运行时注入 | 不落盘 |
| `DOCUMIND_LLM_MODEL` | `qwen3.8-flash` | |
| `DOCUMIND_EMBED_BACKEND` | `dashscope`（或 `openai`） | 新增档 |
| `DOCUMIND_EMBED_API_BASE` | 同 LLM base | `/embeddings` 挂在这里 |
| `DOCUMIND_EMBED_API_KEY` | 运行时注入 | |
| `DOCUMIND_EMBED_API_MODEL` | `text-embedding-v3` | |
| `DOCUMIND_EMBED_API_DIM` | `1024` | 与模型真实维度必须一致，不符直接报错 |
| `DOCUMIND_EMBED_API_BATCH` | `10` | DashScope v3 单请求上限 10 条（实测 25 条 → `400 batch size is invalid`） |
| `DOCUMIND_LLM_ANSWER_THINKING` | `true`（默认） | 答案生成的思考档 |
| `DOCUMIND_LLM_ENABLE_THINKING` | `false`（默认） | 短任务（改写/自检）的思考档 |

`scripts/with_dashscope.sh` 负责把本机 magpie 凭据（`~/.config/magpie/providers.json` 里
`name == "Qwen (China)"` 的 `chat`/`key`）在**运行时**读进子进程环境变量：不落盘、不打印、
不进 shell history、`git status` 里永远看不到。用法：

```bash
# 入库（1 份最小 PDF，只跑 /query 默认的 structure 策略以控成本）
DOCUMIND_DATABASE_URL="sqlite+aiosqlite:///D:/code/documind/data/real_dashscope.db" \
./scripts/with_dashscope.sh ./.venv/Scripts/python.exe - <<'PY'
import asyncio, sys; sys.path.insert(0, "src")
from documind.db import init_db, get_session_factory, upsert_user
from documind.pipeline import ingest_document
from documind.embed import get_embedder
async def main():
    await init_db()
    async with get_session_factory()() as s:
        for n, r in [("alice","employee"),("bob","finance"),("carol","hr"),("admin","admin")]:
            await upsert_user(s, n, r)
        print(await ingest_document(s, "evals/datasets/zhicheng_suhao.pdf", acl_roles=["all"], strategies=("structure",)))
        e = get_embedder(); print("embed requests/tokens:", e.requests, e.prompt_tokens)
asyncio.run(main())
PY

# 起服务
DOCUMIND_DATABASE_URL="sqlite+aiosqlite:///D:/code/documind/data/real_dashscope.db" \
./scripts/with_dashscope.sh ./.venv/Scripts/python.exe -m uvicorn documind.app:app --port 8913

# 引用 span 逐字回查（官方脚本）
./.venv/Scripts/python.exe scripts/verify_citation_spans.py --base-url http://127.0.0.1:8913 \
    --user admin --dataset evals/datasets/real_provider_qa.jsonl --limit 4

# 思考开/关对照（两个服务端各一次）
./.venv/Scripts/python.exe scripts/ab_thinking_e2e.py --base-url http://127.0.0.1:8913 --tag thinking_on
./.venv/Scripts/python.exe scripts/ab_thinking_e2e.py --base-url http://127.0.0.1:8914 --tag thinking_off
```

## 3. 改了什么（文件:行）

| 位置 | 内容 |
|---|---|
| `src/documind/config.py:19` | `EMBED_BACKENDS`（auto/bge/hash/openai/dashscope） |
| `src/documind/config.py:50-51` | `llm_enable_thinking=False`（短任务）、`llm_answer_thinking=True`（答案） |
| `src/documind/config.py:92-99` | `embed_api_base/key/model/dim/batch/timeout` |
| `src/documind/config.py:124-130` | `embed_backend` 白名单校验：拼错不静默变 bge |
| `src/documind/embed.py:39` | `EmbeddingBackendError`（不降级的语义） |
| `src/documind/embed.py:148` | `OpenAICompatEmbedder`：批量、`trust_env=False`、维度断言、失败即报错、请求/token 计数 |
| `src/documind/embed.py:246-283` | `get_embedder()` 按后端分派；`bge`/远端档都不许静默降级，`auto` 保留历史 bge→hash 但写 `/health` |
| `src/documind/llm.py:74-101` | `enable_thinking` 三态（true/false/不发送） |
| `src/documind/llm.py:177` | 流式 `stream_options.include_usage`：否则流式请求拿不到 usage |
| `src/documind/generate.py:231-234, 311, 348, 452` | 答案档/短任务档分别接上思考开关 |
| `src/documind/query.py:49` | query 改写按短任务档（关思考） |
| `tests/test_embed_api.py`（11 例） | 批量入参、`model@dim` 指纹、维度不符报错、截断报错、失败不降级、`/health` 降级可见、入库后换空间 409 |
| `tests/test_llm_payload.py`（4 例） | `enable_thinking` 三态、默认策略、流式 usage |
| `scripts/with_dashscope.sh` | 运行时读凭据的启动包装（不打印 key） |
| `scripts/ab_thinking_e2e.py` | 思考开/关的端到端对照（token/延迟/span/拒答） |

## 4. 实测证据

### 4.1 `/health`（真实后端已生效）

```bash
curl -s --noproxy '*' http://127.0.0.1:8913/health
```

```json
{"documents": 1, "chunks": 47,
 "embedder": "text-embedding-v3", "embed_dim": 1024, "embed_space": "text-embedding-v3@1024",
 "embedder_backend": "dashscope", "degraded": [], "degraded_reason": "", "index_version": 1,
 "llm": {"base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1", "model": "qwen3.8-flash"},
 "refusal_threshold": {"reranker": "BAAI/bge-reranker-base", "value": -6.0, "source": "builtin",
   "calibrated": null, "mismatch": false, "warning": "reranker=... 还没有标定值，使用内置占位 -6.0000"}}
```

### 4.2 非流式 `/query`（答案 + citations + 阈值 + token）

问题：`股东大会通过利润分配方案后，公司最迟要在多久内完成股利派发？`（`X-User: admin`）

```json
{"text": "公司股东大会对利润分配方案作出决议后，须在 2 个月内完成股利（或股份）的派发事项[1]。",
 "refused": false,
 "prompt_tokens": 4311, "completion_tokens": 519, "cost_usd": 0.00089058,
 "latency_ms": 19239, "retrieval_ms": 10428, "generation_ms": 8061, "top_score": 6.3142,
 "citations": [{"ref": 1, "page_start": 48, "char_start": 460, "char_end": 541,
                "span_source": "anchored",
                "quote": "公司股东大会对利润分配方案作出决议后，或公司董事会根据年度股东大会审议通过的下一年中期分红条件和上限制定具体方案后，须在 2 个月内完成股利（或股份）的派发事项。"}],
 "debug.refusal_threshold": {"value": -6.0, "source": "builtin", "reranker": "BAAI/bge-reranker-base"},
 "selfcheck": {"supported": true, "unsupported_claims": []}}
```

`latency_ms` 19.2 s 里 10.4 s 是**本机 CPU 上的 bge-reranker-base**（冷启动含模型加载），
不是供应商慢。预热之后的逐题查询是 4.4–10.7 s（见 §5）。

### 4.3 流式 `/query/stream`

问题：`董事会开会通知一般有哪些方式？`

```
status 200 content-type text/event-stream; charset=utf-8
events: {"evidence": 1, "token": 7, "answer": 1}   # 顺序 evidence → token×7 → answer
streaming: {"mode": "sse", "chunks": 25, "ttft_ms": 4210, "token_events": 7, "degraded": ""}
answer: 公司召开董事会的会议通知，以书面、电话、传真、电子邮件等方式通知。[3]
tokens: prompt=4376 completion=329 cost_usd=0.00081103
latency_ms: 8218 (retrieval 2388 / generation 4864) · 客户端 wall 8341 ms
citation [3] p52 chars 151..188 span_source=anchored
quote: "第四条 公司召开董事会的会议通知，以书面、电话、传真、电子邮件等方式通知。"
```

`usage` 能拿到是因为 `llm.py:177` 显式要了 `stream_options.include_usage`：不写这一行，
流式请求在协议里没有 usage，报告里的 token 就只能靠字数估。

### 4.4 引用 span 逐字回查（官方脚本）

```bash
./.venv/Scripts/python.exe scripts/verify_citation_spans.py \
  --base-url http://127.0.0.1:8913 --user admin --dataset evals/datasets/real_provider_qa.jsonl --limit 4
```

```json
{"questions": 4, "answered": 4, "refused": 0, "citations": 4,
 "span_ok": 4, "span_hit_rate": 1.0,
 "span_valid": 4, "span_valid_rate": 1.0,
 "span_sources": {"anchored": 4},
 "quote_verbatim": 4, "quote_verbatim_rate": 1.0,
 "citation_validity_mean": 1.0}
```

口径：`span_ok` = `page[char_start:char_end]` 确实是 quote 的开头；`span_valid` = 整句 quote
逐字落在切片里（= `citation_validity` 的判定）；`span_sources` 全是 `anchored`，没有
`fallback_page` / `chunk` 兜底，说明引用没有退页退块。

### 4.5 端到端 5 问（含拒答）

数据集 `evals/datasets/real_provider_qa.jsonl`：4 条可答（来自 `evals/datasets/qa_default.jsonl` 的真实题）+
1 条应拒答（`年假可以折现吗？`，公司章程里完全没有这个话题）。

| # | 问题（截断） | 期望 | refused | citations | span_ok/valid |
|---|---|---|---|---|---|
| 1 | 股利派发最迟多久 | 可答 | false | 1 | 1/1 |
| 2 | 董事会通知方式 | 可答 | false | 1 | 1/1 |
| 3 | 分立前债务找谁要 | 可答 | false | 1 | 1/1 |
| 4 | 董监高离职后能否卖股票 | 可答 | false | 1 | 1/1 |
| 5 | 年假可以折现吗 | 拒答 | **true**（`模型判定资料不足`） | 0 | — |

### 4.6 入库（远端正本）

```
title=江苏苏豪时尚集团股份有限公司章 程 pages=59 paragraphs=303
structure: chunks=47 parents=0 avg_chars=607.3 max_chars=899
embedder=text-embedding-v3 embed_space=text-embedding-v3@1024
elapsed_ms=5188 · embedding 请求数 5 · embedding prompt_tokens 17951（供应商返回的 usage）
```

## 5. 思考开关对照（同一批 5 问，两个服务端）

命令见 §2 最后两行；原始结果在 `data/real_run/ab_thinking_on.json` / `ab_thinking_off.json`
（`data/` 已在 .gitignore，数字已抄进本表）。

| 指标 | 思考开 | 思考关 | 变化 |
|---|---|---|---|
| prompt tokens（5 问合计） | 19 506 | 19 329 | −0.9% |
| completion tokens | 1 380 | 468 | **−66.1%** |
| 单问 completion（POST /query） | 219 / 256 / 199 / 608 / 98 | 134 / 101 / 118 / 93 / 22 | 全面下降 |
| 单问延迟 ms | 7301 / 7203 / 7142 / 10725 / 3877 | 12533 / 4848 / 5505 / 4449 / 3110 | 均值 7250 → 6089（−16.0%） |
| 引用 span 命中 | 4/4 | 4/4 | 不变 |
| 引用有效率（span_valid） | 4/4 | 4/4 | 不变 |
| 拒答（应拒 1 问） | 1/1 | 1/1 | 不变 |
| 误拒（可答 4 问） | 0 | 0 | 不变 |
| 自检判定 `supported` | 4×true | 4×true | 不变 |
| 成本（仓库价目表口径 USD） | 0.0035745 | 0.00311931 | **−12.7%** |
| 成本（百炼 qwen-flash 挂牌价口径 ¥） | 0.00500 | 0.00360 | −28% |

答案本身措辞有差异、语义一致（例：问 3 开=`…承担连带责任；但如果公司在分立前与债权人…另有约定的，从其约定`，
关=`…承担连带责任。但是，公司在分立前与债权人就债务清偿达成的书面协议另有约定的除外`），
两者都只引 1 条、都命中原文。

**结论与建议**：本批 4 条引用 + 1 条拒答没有观察到任何质量差异，所以"关掉思考省 66% completion"
在这批题上是安全的；但 4 条引用不足以证明**普遍**不退化（尤其多步条款推理题），因此：

- 答案生成默认**开**（`DOCUMIND_LLM_ANSWER_THINKING=true`，也是代码默认）；
- 自检判定/query 改写默认**关**（`DOCUMIND_LLM_ENABLE_THINKING=false`）；
- 要再省，先压上下文（每问 4.3k prompt 才是成本主体），不要靠关思考硬省。

## 6. token 用量与费用口径（本次全部真调用）

| 项目 | token（供应商 usage） | 延迟 |
|---|---|---|
| 入库 embedding | 17 951（5 请求，批量 10/请求） | 5.19 s |
| 查询 embedding | 约 17×~25 ≈ 0.4k（每次查询 1 条） | 单次 < 0.3 s |
| 非流式 `/query`（含自检） | 4 311 + 519 | 19.2 s（含 reranker 冷启动 10.4 s） |
| 流式 `/query/stream`（含自检） | 4 376 + 329 | 8.2 s / TTFT 4.2 s |
| 思考开 5 问 | 19 506 + 1 380 | 均值 7.25 s |
| 思考关 5 问 | 19 329 + 468 | 均值 6.09 s |
| 对话探测（探供应商字段/批量上限等） | < 300 | < 5 s |
| **本轮 LLM 合计** | **≈ 47.5k prompt + 2.7k completion**（不含探测与两次 trace 外的调用） | — |

费用两种口径（都是估算，**实际请以百炼控制台账单为准**）：

- 仓库价目表口径（`config.py` 默认 0.15 / 0.47 USD per Mtok）：本轮 LLM ≈ **$0.0084**；
  两档对照分别是 $0.00357 / $0.00312。
- 百炼挂牌价口径（qwen-flash 档：输入 0.15 元 / 输出 1.5 元 每百万 token；`qwen3.8-flash`
  未在公开价目页列出，按同档参照）：本轮 LLM ≈ **¥0.011**，两档对照 ¥0.0050 / ¥0.0036。
- embedding（text-embedding-v3，北京区 0.5 元/百万输入 token，批量调用再半价）：18.3k token
  ≈ **¥0.0092**。
- **本轮总计 ≈ ¥0.020（≈ $0.003）**。

两个口径差 ~5 倍，原因是仓库里的价目表是保守占位值；报告里同时给两个数就是为了不让读者把
`cost_usd` 当成账单。

## 7. 限制与不确定（必读）

1. **只入库 1 份 PDF、只跑 `structure` 策略**（`/query` 默认策略）。fixed / semantic /
   parent_child 三套块没在远端 embedding 上跑过，不能推断它们的行为一致。
2. **A/B 样本很小**（4 条可答 + 1 条应拒答，单次运行，无随机种子）。temperature=0 仍会
   有措辞差异（见 §5），完成度只能说明"这一批没退化"。
3. **延迟数字混入了本机 CPU rerank**（bge-reranker-base，10 候选 ×256 token）。供应商侧
   TTFT 4.2 s、单次生成 1–5 s；19 s 那条是冷启动。要量供应商延迟必须关掉 rerank 或换远端 reranker。
   思考关那组的第 1 问 12.5 s 是冷启动离群值（比思考开那组同一问还慢），把均值压成了 −16%；看其余 4 题是 −20%~−58%，第 1 问则变慢 72% —— 单次运行的冷启动噪声足以盖过 35% 的真实差，所以延迟只能当成"方向"而不是结论。
4. **拒答题原是「文档完全没有这个主题」**；本轮已补 2 条「文档有相似条款、答案不在」的手写难例
   并做了阈值标定（§8），但难例只有 2 条、可答只有 15 条，**不足以定论**。标定值与它的局限见 §8。
5. **没有换掉 reranker**，也没重跑 `documind eval` 全量评测，所以本文件的 span/拒答指标只在
   这 5 道题上成立，不能替代 `evals/reports/` 的评测结论。
6. **供应商约束只验证了一条**：text-embedding-v3 单请求最多 10 条（25 条直接 400
   `batch size is invalid`）。批量上限写进配置（`embed_api_batch=10`）。超长单条文本（>8192 token）
   没有测，理论上会被上游拒绝，代码会按"上游拒绝（不降级）"报错。
7. **`enable_thinking` 是 DashScope 的形状**，不是 OpenAI 标准。代码做成三态，空值即不发送；
   接别的兼容端点时请把两个 `*_THINKING` 设成空，别让陌生的字段把请求打成 400。
8. 本轮没有安装任何新组件（无 tiktoken / 无 dashscope SDK）：token 数全部取上游 `usage`，
   不需要本地分词器。

## 8. 拒答阈值在新 embedding 空间上的标定（2026-11-08，接 §4/§7 之后）

### 8.1 为什么标定、标的是什么

`/health` 里那个 `-6.0`（`source=builtin`）是旧 bge 分数域的**占位值**，不是任何一次评测的结论。
换到 `text-embedding-v3@1024` 之后，检索分与重排分的分布都变了（本次可答题 top_score 落在
1.58–7.57），`-6.0` 实际上等于「永不按分数拒答」，拒答全靠模型自己说"资料不足"——
概率性的，且每次都要付一次生成调用。

标定只动一件事：`EvalRun.recommended_threshold`（reranker 分桶），
`/health` 的 `source` 由此从 `builtin` 变 `calibrated`。不调 judge、不花 LLM 钱。

### 8.2 标定集怎么来的（**只做减法，不改标注**）

`qa_default.jsonl` 的 67 条可答题来自 3 份章程，而 `data/real_dashscope.db` 只有
`zhicheng_suhao.pdf` 一份（59 页 / 47 块）。直接拿全量跑，另两份文档的 gold 永远召不回来，
那是题与语料不匹配，不是检索退化。所以：

```bash
./.venv/Scripts/python.exe scripts/build_calib_dataset.py \
    --doc "江苏苏豪时尚集团股份有限公司章 程" \
    --hard evals/datasets/hard_neg_suhao.jsonl \
    -o evals/datasets/qa_suhao.jsonl
# [build] base=87 answerable(全量)=67 → 匹配 '江苏苏豪时尚集团股份有限公司章 程': 15 · 不可答 20 · 手写难例 2 · 合计 37
```

| 组 | n | 来源 |
|---|---|---|
| 可答 | 15 | `qa_default.jsonl` 里 `gold_doc == 苏豪章程` 的全部题，gold_pages / gold_quote 原样保留 |
| 不可答 | 20 | `qa_default.jsonl` 原本的固定 20 条（年假/报销/工资日…），这些内容章程里确实没有 |
| 手写难例 | 2 | `evals/datasets/hard_neg_suhao.jsonl`：**有相似条款、答案不在** |

两条难例都写了 `source` + `abs_check`（缺失验证），可复核"是不是为了让数字好看而加题"：

| id | 问题 | 相似条款在哪 | 为什么答案不在 |
|---|---|---|---|
| `hardneg-000` | 分红后股东个人按什么**税率**交个税？ | 第 48–49 页股利派发「须在 2 个月内完成」 | 59 页全文 `个人所得税`=0、`税率`=0 |
| `hardneg-001` | 每年给会计师事务所的**审计费用多少元**？ | 第 50 页「审计费用由股东大会或授权董事会决定」 | 只规定谁定，没有金额 |

### 8.3 指标（n=37，`structure` 策略，bge-reranker-base 本机重排）

```bash
DOCUMIND_DATABASE_URL="sqlite+aiosqlite:///D:/code/documind/data/real_dashscope.db" \
./scripts/with_dashscope.sh ./.venv/Scripts/python.exe -u scripts/eval_retrieval.py \
    --dataset suhao --strategy structure --ans-limit 15 --unans-limit 22 --write \
    --json evals/reports/calib_suhao.json
```

```
[reranker] BAAI/bge-reranker-base · 拒答阈值 -6.0000（来源 builtin） · max_tokens=256 · batch=8
[structure] recall_at_5=1.0 context_recall=1.0 mrr=0.9222 refusal_accuracy_pre_llm=0.2273
            false_refusal_rate_pre_llm=0.0 top_score_min_answerable=1.5759
            latency_p50_ms=2910.5 latency_p95_ms=3232.5 threshold=-6.0 threshold_source=builtin
            recommended_threshold=1.5055 recall_at_5_hits=15/15 context_recall_hits=15/15
[write] 已写入标定行：reranker=BAAI/bge-reranker-base recommended_threshold=1.5055 n_cases=37
```

| 指标 | 值 | 口径 |
|---|---|---|
| Recall@5 | **1.0（15/15）** | Top-5 命中 gold 页或 gold quote 前 30 字 |
| Context Recall | **1.0（15/15）** | gold quote 进了喂模型的上下文 |
| MRR | **0.9222** | 唯一失分是 1 题 gold 排到第 2 |
| 拒答准确率（pre-LLM） | **0.2273（5/22）** | 这是**在旧阈值 -6.0 下**的实测 |
| 误拒率（pre-LLM） | **0.0（0/15）** | 同上 |
| recommended_threshold | **1.5055** | 误拒 ≤15% 前提下拒答准确率最高、平台上取最保守点 |

分数分布（同一份 JSON 的 rows，可直接复算）：

```
可答 top_score 升序: 1.5759 1.9701 3.1413 3.2216 3.2999 4.0981 4.6983 5.4752 6.3142 6.3414 6.3831 6.5387 6.6369 6.8209 7.5705
不可答 top_score 降序: 1.3999 -0.1004 -0.1341 -0.2981 -1.9498 -3.1252 … -7.6743 -9.1707
```

即标定值落在 **(1.3999, 1.5759) 的间隔**里，间隔只有 **0.176**：换成两套阈值后
pre-LLM 拒答从 **5/22** 变成 **22/22**，误拒仍是 **0/15**：

```
th=-6.0000  拒答(不可答)=5/22   误拒(可答)=0/15
th= 1.5055  拒答(不可答)=22/22  误拒(可答)=0/15
```

### 8.4 `/health`：`builtin` → `calibrated`

```bash
curl -s --noproxy '*' http://127.0.0.1:8913/health
```

标定前：

```json
"refusal_threshold": {"reranker": "BAAI/bge-reranker-base", "value": -6.0, "source": "builtin",
  "calibrated": null, "calibrated_run": "", "calibrated_at": "", "mismatch": false,
  "warning": "reranker=BAAI/bge-reranker-base 还没有标定值，使用内置占位 -6.0000；…"}
```

`--write` 之后（服务不用重启：阈值是每次请求按库解析的）：

```json
"refusal_threshold": {"reranker": "BAAI/bge-reranker-base", "value": 1.5055, "source": "calibrated",
  "calibrated": 1.5055, "calibrated_run": "04ea166191d245c4",
  "calibrated_at": "2026-11-08T00:30:36.058675", "mismatch": false, "warning": ""}
```

`eval_runs` 里的那行：

```
('04ea166191d245c4', 'retrieval-calib:structure', 'suhao', 37, 'scripts/eval_retrieval.py',
 'BAAI/bge-reranker-base', 1.5055, '2026-11-08 00:30:36.058675')
```

另外验了覆盖路径：`DOCUMIND_REFUSAL_THRESHOLD=-6.0` 起第二个服务时
`source=env` 且 **`mismatch=true`**，warning 里写明覆盖值与标定值不一致——即"标定完了又被环境变量
悄悄拉回去"这件事在 `/health` 上看得到。

### 8.5 真问答：两套阈值各跑一次（5 题 + 1 条边界题）

题集 `evals/datasets/threshold_probe_suhao.jsonl`：3 条可答（`real_provider_qa.jsonl` 的
000/001/003）+ 2 条手写难例。两个服务同一份库：`-6.0` 用 `DOCUMIND_REFUSAL_THRESHOLD=-6.0`
起在 8914（"标定前"），1.5055 起在 8913（"标定后"）。请求一律 `selfcheck=false`、无历史
（不触发 query 改写），一题一次生成。

```bash
./.venv/Scripts/python.exe scripts/compare_thresholds_e2e.py --base-url http://127.0.0.1:8914 \
    --label before_builtin_-6.0 --dataset evals/datasets/threshold_probe_suhao.jsonl --out data/real_run/thr_before.json
./.venv/Scripts/python.exe scripts/compare_thresholds_e2e.py --base-url http://127.0.0.1:8913 \
    --label after_calibrated_1.5055 --dataset evals/datasets/threshold_probe_suhao.jsonl --out data/real_run/thr_after.json
```

| 题 | 期望 | `-6.0` 判定 | `1.5055` 判定 | top_score | 引用 span(ok/valid) |
|---|---|---|---|---|---|
| 股利派发最迟多久 | 可答 | 答（1 引，span 1/1） | 答（1 引，span 1/1） | 6.3142 | 1/1 · 1/1 |
| 董事会通知方式 | 可答 | 答（1 引） | 答（1 引） | 4.0981 | 1/1 · 1/1 |
| 董监高离职能否卖股 | 可答 | 答（1 引） | 答（1 引） | 6.3414 | 1/1 · 1/1 |
| **分红后个税税率** | 拒答 | 拒（`模型判定资料不足`，5.97 s） | 拒（`最高重排分 -0.10 低于拒答阈值 1.51`，2.16 s） | -0.1004 | — |
| **审计费用多少钱** | 拒答 | 拒（`模型判定资料不足`，5.57 s） | 拒（`最高重排分 -0.30 低于拒答阈值 1.51`，2.79 s） | -0.2981 | — |
| 离职提前多少天（边界） | 拒答 | 拒（`模型判定资料不足`，6.41 s） | 拒（`最高重排分 1.40 低于拒答阈值 1.51`，2.68 s） | 1.3999 | — |

汇总（脚本输出）：

```
before:  questions=5 n_citations=3 span_ok=3/3 span_valid=3/3  false_refusal_rate=0.0  refusal_accuracy=1.0
after :  questions=5 n_citations=3 span_ok=3/3 span_valid=3/3  false_refusal_rate=0.0  refusal_accuracy=1.0
```

**结论（老实说）：这一批题上两套阈值的最终判定和引用质量完全一样。**
差别只在"谁来拒"：

- 旧阈值：三题都是**走到生成之后**、由模型自己回"资料不足"（`模型判定资料不足`，5.6–6.4 s，各花一次生成）；
- 新阈值：同样的题在 **pre-LLM** 就被分数拦下（`最高重排分 1.40 低于拒答阈值 1.51`，2.2–2.8 s，**0 次生成**）。

也就是说，标定值的收益是**把概率性拒答换成确定性拒答 + 省掉生成调用**，不是"救回了原来答错的题"。
本批 4 条可答题的引用全部 `anchored`、`span_ok`/`span_valid` 3/3、误拒 0，与 §4.4 一致。

### 8.6 限制：n 小，不能外推

1. **n=15 可答 / 22 不可答，其中"同域难例"只有 2 条**。15 条可答里没有一条落在 1.4–1.58 之间，
   所以"间隔 0.176"只是这 15 条的间隔，不是分布的间隔——再来一条措辞更偏的可答题，很容易掉到
   1.5055 以下变成误拒。**这个数值只对这份语料 + 这套切块 + 这个 reranker 成立。**
2. **负例的域不对齐**：20 条固定不可答题是 HR 类（年假/报销/工资日），与章程的距离远大于
   真实业务里"同一份文档内相近但没写"的问题，因此拒答准确率会偏乐观（这一点已写进
   `evals/README.md` 的诚实边界）。
3. **只有 1 份 PDF、只有 `structure` 一种切块**（fixed/semantic/parent_child 没在远端 embedding 上入库）。
4. **阈值分桶的键是 reranker**，不是 embedding。换回 `lexical` 或换别的 cross-encoder 必须重标
   （两个分数域不同，代码里 `BUILTIN_THRESHOLDS` 也是按这个名字分桶的）。
5. **不知道"模型自己会不会拒"**：本轮没测生成侧拒答（那需要 judge）。因此不能说"关掉阈值也安全"，
   只能说"这批题上模型确实自己拒了"。
6. 6 题 × 2 套阈值只花了 **9 次生成调用**（`selfcheck=false`、无历史），未做重复采样，
   措辞与延迟单次即可波动（probe-000 26.7 s 那条是 reranker 冷启动）。

### 8.7 怎么补（下一轮）

1. **系统造同域难例**：把章程里"有章可循但没给数值/主体"的条款扫一遍（税率、金额、时限、
   审批人、违约后果），每条人工确认"答案确实不在"，目标 20–30 条；这一条比加可答题更能压住阈值。
2. **补齐语料**：把 `zhicheng_meinuohua.pdf` / `zhicheng_hengxuan.pdf` 也用 dashscope 入库，
   标定集就能回到 67 可答，间隔才有统计意义。
3. **分开报两段拒答**：pre-LLM（阈值）与 post-LLM（模型自判）各自的贡献要分别统计，
   否则"阈值有没有用"永远说不清——本轮已经看到两者的结论会重合。