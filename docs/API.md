# DocuMind API 契约（前端据此开发，v0.1）

所有请求默认带 `X-User: alice`（alice=employee / bob=finance / admin=admin）。
写操作（上传、建用户、跑评测）带 `X-Admin-Token: dev-admin-token`。
后端默认 `http://127.0.0.1:8080`。

## GET /health
```json
{"status":"ok","documents":3,"chunks":912,"embedder":"bge-small-zh-v1.5","index_version":1,
 "llm":{"base_url":"http://127.0.0.1:3425/v1","model":"qwen-cn/qwen3.8-flash"}}
```

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
 "elapsed_ms":84859,"embedder":"bge-small-zh-v1.5"}
```

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

## GET /query/stream?question=...&strategy=structure&conversation_id=...
SSE，两个事件：
```
event: evidence
data: [Hit, ...]

event: answer
data: {Answer}
```

## GET /evals/runs
```json
[{"id":"r1","strategy":"structure","dataset":"default","n_cases":100,
  "metrics":{"recall_at_5":0.79,"faithfulness":0.92,...},"created_at":"...","report_path":"..."}]
```

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