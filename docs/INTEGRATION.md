# 与项目三（AIGate 网关）的对接：证明"底层模型可替换"

DocuMind 不直接持有任何供应商 key，它只认一个 OpenAI 兼容的 `base_url` + `api_key` + `model`。
默认指向本机 magpie 网关；把这三行换成项目三的 AIGate，同一个问答链路就跑在自建控制面后面：

```bash
# 1) 起 AIGate（项目三）
cd D:/code/aigate && uv run uvicorn aigate.main:app --port 8090

# 2) 在 AIGate 上签一把 dev key（明文只落文件，不打印）
cd D:/code/documind && ./.venv/Scripts/python.exe scripts/issue_aigate_key.py

# 3) 让 DocuMind 走 AIGate
export DOCUMIND_LLM_BASE_URL=http://127.0.0.1:8090/v1
export DOCUMIND_LLM_API_KEY=$(cat data/aigate.key)
export DOCUMIND_LLM_MODEL=<aigate 路由到的模型名>
uv run documind ask "公司注册资本是多少？"
```

## 本机验证状态（2026-11-07）

| 断言 | 状态 | 证据 |
|---|---|---|
| 网关鉴权生效 | 已验证 | 无 key 调 `/v1/models` → 401；带签发的 key → 200 并列出路由 |
| 可由应用侧签发 dev key | 已验证 | `POST /admin/v1/keys` → 201，明文只落 `data/aigate.key`（49 字符，未打印） |
| 路由可指向自建 OpenAI 兼容端点 | 已配置，待复跑 | `scripts/use_provider.py vllm qwen-cn/qwen3.8-flash --route chat-default` 写库成功；当时上游返回失败，网关按设计**降级到 mock-alt 并返回 200 + `degraded=true`**（降级路径本身是项目三的核心能力，这里算是被现场证明了） |
| DocuMind 全链路走 AIGate | 未完成 | AIGate 仓库此刻正被另一处改动（`src/aigate/config.py` 有未提交的缩进错误），等服务恢复后重跑本节第 3 步 |

> 这一行必须留着：**没跑完的就不写"已验证"**。恢复后重跑，成功再把表格改绿。

## 这条链路设计上证明了什么

| 断言 | 证据 |
|---|---|
| 换供应商不改代码 | 只改 3 个环境变量；`generate.py` / `pipeline.py` 一行没动 |
| 成本可归因 | AIGate 侧按 key 记账，DocuMind 侧自己也算 token 成本，两边能对账 |
| 可私有化 | AIGate 支持把后端指到 vLLM；DocuMind 只看到 OpenAI 协议 |
| 可限流/熔断 | 限流与熔断在网关做，应用层不用内建这些逻辑 |

## 为什么不在 DocuMind 里自己实现限流/熔断

一个系统只该有一个地方管"调用谁、花多少、失败了换谁"。这两件事放在应用里，
第二个应用就要重写一遍，而且对不上账——这正是项目三存在的理由。

## 反向依赖说明

DocuMind **不依赖** AIGate 才能运行（默认直连 magpie / 任意 OpenAI 兼容端点），
AIGate 也不依赖 DocuMind。两者是"应用层 → 平台层"的单向关系，可独立演示。