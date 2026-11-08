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
4. **拒答题是"文档完全没有这个主题"**，没测"文档有相似条款但应当拒答"的难例；阈值仍是
   内置占位 `-6.0`（`source=builtin`），没有在本语料上标定。
5. **没有换掉 reranker**，也没重跑 `documind eval` 全量评测，所以本文件的 span/拒答指标只在
   这 5 道题上成立，不能替代 `evals/reports/` 的评测结论。
6. **供应商约束只验证了一条**：text-embedding-v3 单请求最多 10 条（25 条直接 400
   `batch size is invalid`）。批量上限写进配置（`embed_api_batch=10`）。超长单条文本（>8192 token）
   没有测，理论上会被上游拒绝，代码会按"上游拒绝（不降级）"报错。
7. **`enable_thinking` 是 DashScope 的形状**，不是 OpenAI 标准。代码做成三态，空值即不发送；
   接别的兼容端点时请把两个 `*_THINKING` 设成空，别让陌生的字段把请求打成 400。
8. 本轮没有安装任何新组件（无 tiktoken / 无 dashscope SDK）：token 数全部取上游 `usage`，
   不需要本地分词器。