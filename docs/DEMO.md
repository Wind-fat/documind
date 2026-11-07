# 3 分钟演示脚本（录屏用）

准备：`uv run documind seed && uv run documind serve`，前端 `cd web && npm run dev`，浏览器开 `http://localhost:3000`。

| 时间 | 操作 | 想让人看到什么 |
|---|---|---|
| 0:00-0:15 | 打开问答页，身份选 `alice`（employee） | 顶部显示当前身份与可见文档数 |
| 0:15-0:40 | 问"公司注册资本是多少？" | 答案里带 `[1]`；右侧证据卡显示《公司章程》第 2 页 · 第一章 总则 |
| 0:40-0:55 | 点证据卡 → 原文弹层 | 弹层自动滚动并高亮到"第六条 公司注册资本为人民币…"那一行 |
| 0:55-1:15 | 问"年假可以折现吗？" | 醒目的拒答卡片："文档中未找到相关条款"，旁边给出 reason（最高重排分 x.xx 低于阈值 y.yy） |
| 1:15-1:45 | 切到 `alice` 问"备用金额度是多少？" → 拒答；切 `bob`（finance）再问 → 正常回答 | 同一句话、不同角色、结果不同：ACL 在检索阶段生效，`alice` 的 trace 里根本没有那份文档 |
| 1:45-2:10 | 打开 `/evals` | 四种切块策略的对比表（Recall@5 / Faithfulness / 拒答准确率 / P95 / 单问成本）+ 拒答阈值 P-R 曲线 |
| 2:10-2:35 | 终端 `uv run documind eval --dataset default --limit 20` | 现场跑一次回归，展示数字可复现 |
| 2:35-3:00 | 打开 `data/traces/<今天>.jsonl` | 每次检索命中数、token、延迟、成本，字段与 Langfuse observation 对齐 |

## 现场可能被追问的三个点

1. **"为什么不用 Dify/FastGPT？"** —— 先用它们做过 PoC：ACL 只能到"应用级"、拒答只能靠 prompt、切块策略改不动、
   评测要自己外挂。这三件事正好是本项目的核心，所以下沉到代码层。
2. **"0.79 是怎么来的？"** —— `evals/reports/details-*.json` 里每道题都有检索名次与引用，
   现场可以抽任意一条回放：`uv run documind ask "<那条问题>" --role admin`。
3. **"如果我换成 GPT/Claude 呢？"** —— 改 `DOCUMIND_LLM_BASE_URL` 与 `DOCUMIND_LLM_MODEL` 两行；
   本项目默认就是打本机 magpie 网关，也可以打项目三的 aigate 网关（`docs/INTEGRATION.md`）。