# 真流式：`/query/stream` 的实测延迟（P1-05）

> 目标：`/query/stream` 不再"等全量再吐两个事件"。证据事件在检索完成后**立刻**发出，
> 答案正文**逐块**转发。本文只贴跑出来的数字，不贴推算。

## 1. 事件契约（前端按这个写）

```
event: evidence   data: [Hit, ...]              ← 检索完成即发，之前没有任何模型调用
event: token      data: {"text": "增量正文"}     ← 0..n 个，模型每吐一块就转发一块
event: answer     data: {Answer}                 ← 权威结果，与 POST /query 同构
```

- `token` 是**草稿**：引用还没校验、自检还没跑，最终可能被 `answer` 覆盖（拒答、自检重试都会覆盖）。
- `answer` 是**权威**：载荷字段与 v0.1 完全一致。
- 旧客户端不受影响：`evidence` 仍是 Hit 数组、`answer` 仍是完整 Answer，`token` 是新增事件，
  未知事件按 SSE 惯例忽略即可。前端的 SSE 解析器（`web/lib/api.ts`）本来就把未知事件丢给 `onEvent`。

## 2. 度量方式

服务端（真实进程，真实 socket）：

```bash
# 1) 用仓库里的语料副本起服务（不污染 documind.db）
./.venv/Scripts/python.exe -c "import sqlite3; sqlite3.connect('documind.db').execute(\"VACUUM INTO 'D:/tmp/measure.db'\")"
DOCUMIND_DATABASE_URL="sqlite+aiosqlite:///D:/tmp/measure.db" \
  ./.venv/Scripts/python.exe -m uvicorn documind.app:app --host 127.0.0.1 --port 8099

# 2) 同一问题分别走 /query 与 /query/stream，逐事件打时间戳
./.venv/Scripts/python.exe scripts/measure_stream.py \
  --base-url http://127.0.0.1:8099 --runs 3 --question "公司注册资本是多少？"
```

`scripts/measure_stream.py` 记录的是**到达时间**：
`evidence_ms`（首字节）→ `first_token_ms`（第一个正文块）→ `answer_ms`（端到端），
并统计 token 事件数与首尾跨度（跨度接近 0 = 攒完一次吐 = 伪流式）。

## 3. 受控上游：把"流式链路"单独量出来（PASS）

上游换成 `scripts/sse_stub.py`（OpenAI 兼容，固定 8 块 × 150ms，非流式路径同样等 1.2s，
所以两边总时长可比）。检索用 lexical（避免 cross-encoder 掩盖生成侧），3 轮中位数：

| 指标 | 数值 |
|---|---|
| 首字节 `evidence` | **22 ms** |
| 第一个正文 token | **177 ms**（占端到端 **12.6%**） |
| `answer`（端到端） | **1401 ms** |
| `POST /query` 端到端 | **1238 ms**（同样的上游耗时，一次性返回） |
| token 事件 | 3 个，首尾跨度 301 ms（≈2×150ms） |
| 判定 | `first_token < 30% × total` → **PASS**（177 < 420） |

逐事件时刻（run3）：`token @ 175ms / 325ms / 476ms`，`answer @ 1401ms` —
token 之间间隔 ≈150ms，与服务端每块延迟一致，说明是逐块转发而不是最后一刻批量 flush。

> 这一轮桩返回的引用与语料不逐字匹配，所以最终 `answer` 是**拒答**（引用强制的正确行为）。
> 它恰好演示了 `token` 是草稿、`answer` 才是权威。

## 4. 真网关（本机 magpie → qwen3.8-flash）：数字与瓶颈

### 4a. 默认配置（cross-encoder 重排）

`--runs 2 --question "公司注册资本是多少？"`：

| run | evidence(首字节) | 第一个 token | answer(端到端) | token 事件 | `/query` 总时长 |
|---|---|---|---|---|---|
| 1 | 25 998 ms | 53 067 ms | 54 942 ms | 17 | 47 448 ms |
| 2 | 21 079 ms | 32 450 ms | 34 220 ms | 18 | 33 664 ms |
| 中位数 | **23 538 ms** | **42 758 ms**（95.9%） | **44 581 ms** | 17.5 | **40 556 ms** |

**这一栏不满足"< 30%"**，原因是可测的、且与流式实现无关：

- `rerank` 单独跑 30 条候选（合计 21 246 字）实测 **21 863 ms**（`BAAI/bge-reranker-base`，CPU）；
  `bm25` 2 ms、`vector` 1 ms、query embed 7 ms、索引构建 64 ms。**检索 21s 全部花在 cross-encoder 上。**
- 上游是推理模型：正文之前先吐 `reasoning_content`（同一问题 9s~100s 不等），正文只占 641~661 ms。
  我们没有把思考过程当正文发给前端（不应把 CoT 当答案），所以"第一个正文 token"必然晚。

即：TTFT = 检索(21s) + 思考(9~25s)，端到端 = TTFT + 正文(0.65s)。在这个配置下，
**首字节/首 token 的比值由检索与模型思考决定，不是由流式实现决定**。

> 2026-11 更新：检索侧已从 21s 降到 1.6s（`docs/RERANK-PERF.md`），重测后
> evidence 1 609 ms / first_token 8 180 ms / answer 9 490 ms —— 比值 86.2%，仍不达标，
> 但**卡点已经从检索换成模型思考时间**。

### 4b. lexical 重排（检索不再是瓶颈）长回答 + 上游抖动

`--runs 2 --question "请逐条列出公司章程中与注册资本、股东出资有关的条款…300 字以上"`：

| run | evidence | 第一个 token | answer | token 事件 | 首尾跨度 | `/query` |
|---|---|---|---|---|---|---|
| 1 | 1 436 ms | 100 589 ms | 108 648 ms | 226 | 8 037 ms | 60 063 ms（超时） |
| 2 | 34 ms | 127 092 ms | 134 574 ms | 187 | 6 301 ms | 60 061 ms（超时） |

- 检索降到 **34 ms**，206 个 token 事件在 6~8 s 内陆续到达 —— 流式链路在真网关下也是逐块转发的。
- 但这一轮上游把"思考"拉到了 100 s+：`POST /query` 撞上 `llm_timeout_s=60`，
  两次都返回 **"LLM 不可用"**（`refused=true`，text 71 字），而同一时刻的流式请求已经拿到
  `evidence` + 200 个 token，最后 `answer` 也返回了。**这是非流式在这台机器上的真实失败模式**，
  也是流式最实际的价值：上游慢的时候，用户至少先看到证据和已生成的部分。

### 4c. 上游自身的流式质量（裸调，不经本服务）

同一个长回答 prompt，直接打网关数块：

| model | 总时长 | 首个正文块 | 块数 |
|---|---|---|---|
| `qwen-cn/qwen3.8-flash` | 22.1 s | 9.3 s | 171 |
| `qwen-cn/kimi-k3` | 22.2 s | 16.9 s | 82 |
| `qwen-cn/glm-5.2` | 10.8 s | 10.8 s | **1**（块全在最后 = 上游假流式） |

也就是说：`glm-5.2` 这类上游即便我们逐块转发，用户也只会看到"最后一次性出现"。
`/health.streaming` 只声明本服务的契约，管不了上游；判据是 `answer.debug.streaming.chunks`，
块数 = 1 且 `token_events` = 1 时应怀疑上游没真流。

## 5. 测试层面的真假判定

`tests/test_stream_and_threshold.py::test_stream_evidence_then_tokens_before_answer`
用真 uvicorn（同进程、真 socket；httpx 的 ASGITransport 会缓冲整个 body，测不出时序）
+ 假模型 6 块 × 100 ms，断言：

1. 第一个事件是 `evidence`；
2. `token` 事件 ≥ 2，且全部早于 `answer`；
3. `answer` 与首个 token 的时间差 ≥ 0.2 s（真流式；攒完再发≈0）；
4. 首个 token 到达时模型**还没跑完**；
5. `first_token < 50% × total`。

**这个用例对旧实现是红的**（变异验证）：把 `stream_answer` 换回"等全量再吐两个事件"后，
实测 `events=['evidence','answer'] tokens=0 gap=0.000s` —— 第 2、3 条断言立刻失败。

## 6. 降级路径（不许静默）

上游拒绝流式（实测本机 magpie 的 qwen-cn 偶发 `HTTP 400 Required body invalid`）时：

- `LLMClient.chat_stream` 抛 `LLMError`（不 mock、不假装在流）；
- `AnswerStream` 退回一次非流式 `chat`，事件顺序仍是 `evidence → token → answer`；
- `answer.debug.streaming.degraded` 写明原因（截断 300 字），`/query/stream` 的调用方能看到；
- 用例：`test_stream_falls_back_to_single_shot_and_says_so`。

## 7. 已知缺口（本文不修）

- **检索 21 s** → **已修**：`BAAI/bge-reranker-base` 对 30 条长块打分在 CPU 上 ~22 s，占端到端一半以上。
  候选数 30→10 + 按问题挑窗口截断后，重排 P50 2.7 s、首字节 1.6 s，Recall@5/Context Recall 未降；
  完整拆解与前后数字见 `docs/RERANK-PERF.md`。剩下的 TTFT 占比由**模型思考时间**决定（本文 §4 的老结论）。
- **推理模型的思考时间**不透明：正文前 9~100 s。要真正压低"用户看到字"的时间，
  要么换非推理模型，要么把进度（"已思考 N 秒"）当独立事件暴露，而不是泄露 CoT。
- `/query` 的 60 s 超时会把慢上游变成"LLM 不可用"（4b 实测），需要单独的超时/重试策略。