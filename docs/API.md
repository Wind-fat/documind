# DocuMind API 契约（前端据此开发，v0.1）

身份：`POST /auth/login {"name":"alice"}` 拿 `token`，之后所有请求带 `Authorization: Bearer <token>`。
生产（`DOCUMIND_ENV=prod`）下裸 `X-User` 一律 401；开发环境允许裸 `X-User` 但用户必须已存在（不自动建号）。
演示用户：alice=employee / bob=finance / admin=admin。
写操作（上传、建用户、跑评测）带 `X-Admin-Token: dev-admin-token`。
后端默认 `http://127.0.0.1:8080`。

## GET /health
```json
{"status":"ok","documents":3,"chunks":912,"embedder":"BAAI/bge-small-zh-v1.5","embed_dim":512,
 "embed_space":"BAAI/bge-small-zh-v1.5@512","embedder_backend":"bge",
 "degraded":[],"degraded_reason":"","index_version":1,
 "llm":{"base_url":"http://127.0.0.1:3425/v1","model":"qwen-cn/qwen3.8-flash"},
 "refusal_threshold":{"reranker":"BAAI/bge-reranker-base","value":-4.25,"source":"calibrated",
   "calibrated":-4.25,"calibrated_run":"a1b2c3","calibrated_at":"2026-11-07T12:00:00+00:00",
   "mismatch":false,"warning":""},
 "streaming":{"endpoint":"/query/stream","events":["evidence","token","answer"]}}
```
`degraded` 非空表示发生了自动降级（目前只有 `embedder`：bge 拉不下来退到 hash），原因在 `degraded_reason`。
**别忽略这一项**：降级后检索质量会掉，而接口仍返回 200。

`refusal_threshold` 是**当前实际生效**的拒答阈值与它的来源（P1-03）：

| `source` | 含义 |
|---|---|
| `env` | `DOCUMIND_REFUSAL_THRESHOLD` 显式覆盖（优先）。与最近标定值不一致时 `mismatch=true`、`warning` 写明两者与 eval_run id |
| `calibrated` | 该 reranker 最近一次评测标定写回的 `eval_runs.recommended_threshold` |
| `builtin` | 从未标定过，用的是内置占位（`warning` 会提示"跑一次 documind eval"） |

阈值按 `reranker.name` 分桶（`lexical` / `BAAI/bge-reranker-base` 各一套），换重排后端不会静默沿用另一个域的数值。

## GET /documents
```json
[{"id":"a1b2","title":"公司章程","filename":"zhicheng_suhao.pdf","pages":59,
  "acl_roles":["all"],"visible":true,"created_at":"2026-11-07T08:00:00"}]
```
`visible=false` 的文档（当前角色无权）**不给内容也不给 404 细节**，只显示"无权访问"。

## GET /documents/{doc_id}/pages/{page_no}
```json
{"doc_id":"a1b2","page":7,"text":"第四章 股份\n第一条 ..."}
```
页文本是后端解析后的规范化文本，citation 的 char_start/char_end 就是这段文本里的偏移。

## GET /documents/{doc_id}/outline
```json
{"doc_id":"a1b2","sections":[{"title":"第一章 总则","page":2}]}
```

## POST /documents  (multipart, 需 admin token)
fields: `file`, `acl_roles`（JSON 字符串，如 `["all"]` 或 `["finance","admin"]`）, `title?`
```json
{"document_id":"a1b2","title":"公司章程","pages":59,"paragraphs":523,
 "strategy_stats":{"structure":{"chunks":47,"parents":0,"avg_chars":614.3,"max_chars":903}},
 "elapsed_ms":84859,"embedder":"BAAI/bge-small-zh-v1.5","embed_space":"BAAI/bge-small-zh-v1.5@512"}
```
失败：

| 状态 | `detail.code` | 场景 |
|---|---|---|
| 422 | `empty_document` | 扫描件/图片型 PDF 没有文本层、空文件。`detail.message` 是中文可读原因（"未提取到文本，需要 OCR"）。不会建出空文档。 |

## 409 检索侧：向量空间不一致
库里的块由别的 embedder（或降级的 hash）生成时，`POST /query`、`GET /query/stream` 返回 409
`{"detail":{"code":"embedding_space_mismatch","message":"..."}}`，而不是拿不可比的向量硬算排名。
处理：用当前 embedder 重新入库，或把 `DOCUMIND_EMBED_BACKEND` / `DOCUMIND_EMBED_MODEL` 改回建库时的配置。

## POST /query
```json
{"question":"公司注册资本是多少？","conversation_id":null,"strategy":"structure","top_k":5,"selfcheck":false}
```
响应（Answer）：
```json
{
  "text":"公司注册资本为人民币438,847,974元 [1]。",
  "citations":[{"ref":1,"doc_id":"a1b2","doc_title":"公司章程","chunk_id":"c9",
    "page_start":2,"char_start":96,"page_end":2,"char_end":140,
    "quote":"第六条 公司注册资本为人民币438,847,974元。","section_path":"第一章 总则"}],
  "refused":false,"reason":"","rewritten_query":"",
  "latency_ms":1840,"cost_usd":0.000021,"prompt_tokens":1200,"completion_tokens":80,
  "retrieval_ms":210,"generation_ms":1500,"top_score":2.31,
  "selfcheck":{"supported":true},
  "debug":{"intent":"fact","strategy":"structure","hits":[ ...同 Hit 结构... ],
           "candidates":58,"conversation_id":"conv123"}
}
```
拒答时：`refused=true`、`citations=[]`、`text` 为"文档中未找到相关条款。"，`reason` 说明原因（阈值/无引用/资料不足）。

Hit 结构（evidence 卡片的数据源）：
```json
{"chunk_id":"c9","doc_id":"a1b2","doc_title":"公司章程","text":"...","page_start":2,
 "char_start":96,"page_end":2,"char_end":140,"section_path":"第一章 总则",
 "score":2.31,"bm25_rank":0,"vec_rank":2,"rrf":0.032}
```

## GET /query/stream?question=...&strategy=structure&conversation_id=...&selfcheck=false

SSE，**真流式**（P1-05）：检索完成即发证据，答案正文逐块转发。事件顺序是契约：

```
event: evidence
data: [Hit, ...]                 ← 检索完成即发（之前没有任何模型调用）

event: token
data: {"text":"增量正文"}          ← 0..n 个，模型每吐一块转发一块

event: answer
data: {Answer}                   ← 权威结果，字段与 POST /query 完全一致
```

- `token` 是**草稿**：引用尚未逐字校验、自检未跑；`answer` 到达后前端应以它为准（可能拒答或被重试改写）。
- 向后兼容：`evidence` 仍是 Hit 数组、`answer` 仍是完整 Answer；`token` 是新增事件，旧客户端忽略未知事件即可。
- 响应头带 `Cache-Control: no-cache, no-transform`、`X-Accel-Buffering: no`，避免代理层把它重新攒成一次性响应。
- 上游不支持流式时服务端自动退回一次非流式调用，事件顺序不变，原因写在
  `answer.debug.streaming.degraded`（非空即代表降级）。
- 延迟口径与实测数字（含真网关的前后对比）见 `docs/STREAM.md`。

`answer.debug.streaming`：
```json
{"mode":"sse","chunks":18,"ttft_ms":120,"token_events":17,"degraded":"","degraded_reason":""}
```

## GET /evals/runs
```json
[{"id":"r1","strategy":"structure","dataset":"default","n_cases":100,
  "metrics":{"recall_at_5":0.79,"faithfulness":0.92,...},"created_at":"...","report_path":"...",
  "reranker":"BAAI/bge-reranker-base","recommended_threshold":-4.25}]
```
`reranker` + `recommended_threshold` 是标定回写（P1-03）：阈值按 `reranker.name` 分桶，
`POST /evals/run` 结束时把推荐值落库，服务启动/检索时按当前 reranker 取最近一次。
环境变量 `DOCUMIND_REFUSAL_THRESHOLD` 可显式覆盖（不一致会在 `/health.refusal_threshold.mismatch` 亮出）。

## POST /evals/run  (需 admin token)
`{"dataset":"default","strategies":["fixed","semantic","structure","parent_child"],"limit":null}`

## GET /evals/report
返回 `evals/reports/ablation.html`（完整评测看板，可直接 iframe）。

## GET /stats
```json
{"answers":12,"avg_latency_ms":1750.2,"total_cost_usd":0.00021,"refusal_rate":0.16}
```

## 错误约定
- 401：缺少/错误 admin token
- 404：文档不存在**或无权访问**（不区分，避免探测）
- 400：acl_roles 不是合法 JSON / 不支持的文档类型