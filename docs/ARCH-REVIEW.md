# 架构复核：模块边界与对外契约（documind × bidpilot）

- 复核人：@架构（arch）
- 日期：2026-11-07
- 被复核快照：documind `087b101`、bidpilot `c634140`（工作区有未提交改动，见 §1 口径）
- 复核范围：`docs/adr/0001..0005` 五篇 ADR、`src/documind/` 的模块边界、bidpilot 对 documind 的复用方式
- 结论：**P0 5 条 / P1 11 条**；最危险的一条是 **P0-01 身份信任边界缺失（ACL 判定全对，但主体可伪造，端到端归零）**
- 本文只写边界与契约，不改代码。凡"必须改代码才能解决"的，列在 §7 阻塞项，交 @hermes 排期。

---

## 0. 一句话结论

ADR 里描述的三条强保证（span 可回映射、ACL 检索期生效、阈值由评测标定）在**模块内部**都是真的，但每一条都在**边界上**漏了一段：
span 契约只被 documind 自己的 `chunking` 遵守（bidpilot 绕过）；ACL 只保证"给定 role 不泄露"（role 来自可伪造的 HTTP 头）；阈值标定只产出建议（没有任何代码把它写回配置）。
同时存在一类"同名不实"的抽象：`Retriever` Protocol 零实现零引用、`DoclingParser` 声称可切换但不可达、`stream_answer` 不含流式、`outline()` 只被下游一个消费者使用而上游零测试。

---

## 1. 证据口径（可复现）

| 项 | 值 |
|---|---|
| documind HEAD | `087b101 chore: 同步工作区（2026-11-07）`，`git status` 有 `M web/next.config.ts` 等未提交改动 |
| bidpilot HEAD | `c634140 init: bidpilot`，`git status` 有 `M src/bidpilot/{config,extract,pipeline}.py`、`M scripts/run_pipeline.py` 未提交 |
| 测试基线 | documind `pytest --collect-only -q` → **40 tests collected**；bidpilot → **26 tests collected** |
| 行号基准 | **工作区文件**，不是 HEAD。两仓库都有未提交改动，所以"某次提交可复现"当前不成立（见 R1） |
| 探针输出 | 本文标 `[探针]` 的结论来自 `.venv/Scripts/python.exe -c` 实跑，命令与输出见附录 A |

---

## 2. 模块边界现状（谁依赖谁）

```
                 ┌──────────── app.py (FastAPI) ────────────┐
                 │  /documents /query /query/stream /evals  │
                 └───────────────┬─────────────────────────┘
                                 │ ingest_document / ask / stream_answer
        cli.py ──────────────────┤
                                 ▼
                          pipeline.py  ────────────────┐
                    (入库编排 / 问答编排)                │
      ┌──────────────┬───────────────┬────────────────┤
      ▼              ▼               ▼                ▼
  parsing.py     chunking.py     retrieve.py      generate.py
  parse_file     chunk_document  retrieve()       answer_question
  ParsedDocument      │          Hit/Result           │
      │               │              │                │
      │          slice_text ◄────────┘                │
      │               │              │                │
      ▼               ▼              ▼                ▼
   pypdf/docx      numpy        index.py          llm.py / query.py
                                BM25Index         (OpenAI 兼容)
                                VectorIndex
                                   │
                                   ▼
                              embed.py (bge / hash)
                              rerank.py (CE / lexical)
                                   │
                                   ▼
                                 db.py (SQLAlchemy, chunks.embedding BLOB)

  下游：bidpilot ──► documind.parsing (parse_file / Paragraph / ParsedDocument)
                 └─► documind.index   (tokenize / IndexItem / BM25Index)
```

模块依赖是**单向、无环**的（`parsing → chunking → retrieve → pipeline → app`），这一点是干净的：`parsing.py` 不 import 任何同包模块，`index.py` 只向下依赖 `embed.cosine_search`。问题不在环，在**没有一处是"端口"**——除了 `embed.Embedder` / `rerank.Reranker` 两个基类，其余"接口"都是具体类加约定。

### 2.1 每个模块是否是真接口（判定标准：≥2 个实现且被选择 / 有下游消费者且不依赖实现细节 / 有测试钉住）

| 模块 | 对外符号 | 多实现 | 下游消费者 | 有测试钉住契约 | 判定 |
|---|---|---|---|---|---|
| `parsing` | `parse_file`, `ParsedDocument`, `Paragraph`, `Parser`/`get_parser` | 是（pypdf/docx/md，第 4 个不可达） | chunking、pipeline、**bidpilot** | `test_parsing.py`、`test_chunking.py:43` | **真**（选择机制是硬编码后缀表，换实现要改代码，见 P1-02） |
| `parsing` | `ParsedDocument.slice_text` | —（单一实现） | chunking（唯一） | `test_chunking.py:43` 全策略断言 `sliced == c.text` | **真，且是全仓最强的契约**（唯一被构造保证 + 测试双重钉住的接口） |
| `parsing` | `ParsedDocument.outline` | 否 | **仅 bidpilot**（sections.py:100） | 上游无测试 | **半假**（见 P1-08） |
| `parsing` | `iter_paragraph_text` | 否 | **无** | 无 | 死代码 |
| `index` | `tokenize`, `BM25Index`, `IndexItem` | BM25 单实现 | retrieve、**bidpilot** | `test_retrieval.py:21,27` | **真**（但 `IndexItem` 混入了入库模型，见 P1-10） |
| `index` | `VectorIndex` | 单实现（numpy 暴力） | retrieve | 经由 retrieve 间接覆盖 | **假接口**：`matrix` 是构造入参，换 pgvector 必须重写（见 P1-01） |
| `index` | `Retriever` Protocol | **零实现** | **零引用** | 无 | **纯假**（`# pragma: no cover - 接口文档` 自我承认了这一点） |
| `chunking` | `chunk_document(doc, strategy, embed)` | 是（4 策略） | pipeline | `test_chunking.py` 四策略 | **真** |
| `embed` / `rerank` | `Embedder` / `Reranker` | 是（bge/hash、CE/lexical） | retrieve、pipeline | 部分 | **真接口，假降级**（降级静默且不可观测，见 P0-02） |
| `retrieve` | `retrieve()`, `RetrievalResult`, `Hit` | — | pipeline、generate、evals | `test_retrieval.py` | **真**（但 `IndexStore` 的缓存失效是进程内的，见 P1-04） |
| `pipeline` | `stream_answer` | — | app（SSE） | `test_api.py:133` 只断言事件名 | **假**（无流式，见 P1-05） |
| `pipeline` | `_page_texts` | — | **evals 跨包调用** | — | 私有被外部消费（见 P1-09） |

---

## 3. 同名不实的抽象：逐条对账

| # | 声称（出处） | 实际 | 后果 |
|---|---|---|---|
| A1 | `Retriever` Protocol 是检索接口（index.py:137-138） | 全仓零 import、零实现；`VectorIndex.search` 收 `query_vec: np.ndarray`，与 Protocol 的 `search(query: str, ...)` 签名不兼容 | 读代码的人以为"换检索器实现这个 Protocol 即可"，实际改了也不生效；ADR-0002 的"接口不变"由此被虚构 |
| A2 | "生产环境这两个索引分别换 OpenSearch / pgvector，接口不变"（index.py:6、db.py:3-6） | 无端口。`VectorIndex` 构造要吃 numpy 矩阵；`IndexStore.get` 全表 `vstack`（retrieve.py:110）；`chunks.embedding` 是 BLOB（db.py:92） | 换后端 = 改 4 个文件 6 处（见 §4），不是"接口不变" |
| A3 | "docling 作为可选适配器保留，装上就换，不改上层"；docstring 说 `DOCUMIND_PARSER=docling` 启用（ADR-0001:8、parsing.py:304-306） | `get_parser` 只按文件后缀查表（parsing.py:336-340），`Settings` 里**没有** parser 字段 `[探针]`，`DoclingParser` 不可达 | 想换解析器的人会先信 ADR，再花半天发现是死代码；且该类用长度伪造分页（parsing.py:321-323），真接上会静默错位高亮 |
| A4 | `stream_answer` / `GET /query/stream`（pipeline.py:222-227、app.py:197-216） | 内部 `await ask(...)` 拿完整答案后，才 yield 两个事件 | 前端"先画证据卡"的时序假设不成立；首字节延迟 = 全量延迟，SSE 只增加了复杂度 |
| A5 | "评测报告里会明写用的是哪个 embedder"（embed.py:64） | `write_reports(dataset, summary, details, calibration)` 无此参数（evals/report.py:26），报告里搜不到 embedder/bge | 降级到 hash 后，历史报告与当前报告无法区分——正是"静默降级"里最难发现的一类 |
| A6 | `ParsedDocument.outline()` "给 BidPilot 的章节树复用，也给 /outline 用"（parsing.py:75-76） | bidpilot 用了（sections.py:100）；documind 自己的 `/documents/{id}/outline` 另用 chunk 表算一份（app.py:168-179）；上游零测试 | 同一概念两份实现，且 API 那份在非 `structure` 策略入库时为空 |

---

## 4. 换后端要改哪里（真实切换成本）

| 换什么 | 现在写死在哪 | 必改（文件:行） | ADR 声称 | 判定 |
|---|---|---|---|---|
| 关键词索引 → OpenSearch | `BM25Index` 全类 | index.py:44-85（含 `allowed` 掩码语义 78-81）、retrieve.py:74-113（组装）、retrieve.py:157-163（rank 透传） | "接口不变"（index.py:6） | **假**：3 处 + 权限语义要重写（OpenSearch 侧要用 filter 而非负分掩码） |
| 向量索引 → pgvector / Milvus | numpy 矩阵 + BLOB | index.py:88-104、embed.py:101-110、retrieve.py:110、db.py:92,94-96 | 同上 | **假**：4 处，且列类型迁移 |
| 解析器 → docling | 后缀表 | parsing.py:327-333、336-340、303-324，config.py 需新增字段 | "装上就换"（ADR-0001:8） | **假**（不可达） |
| Embedder → 远程 API | `get_embedder` 工厂 | embed.py:86-98、config.py:46；**外加历史向量迁移**（无指纹可判定，见 P0-02） | 只说了本地/哈希两档 | 半真：工厂换了能跑，但旧向量会静默混用 |
| Reranker → 别的 CE | `get_reranker` + 阈值 | rerank.py:84-93 + config.py:40（阈值域） | ADR-0004 说阈值随评测标定 | 半真：换模型即换分数域，阈值不会跟着变 |
| LLM → 任意 OpenAI 兼容端点 | 3 个环境变量 | **不需要改代码**（llm.py:58-101） | INTEGRATION.md 有验证表 | **真**：全仓唯一一处"换后端零代码改动"，可作为其它后端的模板 |

结论：六个"可换后端"的说法里只有一个是真的。ADR-0002 应改成"dev 单进程内存索引；换后端需改 N 处，清单见 ARCH-REVIEW §4"，而不是"接口不变"。

---

## 5. bidpilot 复用审计（是否只用了公开接口）

### 5.1 实际 import 面（`grep -rn 'from documind' src tests`）

| 位置 | 导入 | 性质 |
|---|---|---|
| bidpilot/src/bidpilot/sections.py:16 | `Paragraph, ParsedDocument, parse_file` | 数据类 + 入口函数 |
| bidpilot/src/bidpilot/capabilities.py:15 | `BM25Index, IndexItem, tokenize` | 检索原语 |
| bidpilot/tests/conftest.py:13 | `Paragraph, ParsedDocument` | 手搓夹具 |
| bidpilot/.probe/*.py（5 个） | `parse_file` / `tokenize` | 一次性探针，非产品路径 |

### 5.2 私有函数检查结论

**没有偷偷 import 任何下划线函数**：全仓 grep `_sentences` / `_explode` / `_assemble` / `_group_page` / `_PARSERS` 在 bidpilot 侧**零命中**（唯一 `.` 开头访问是 `.probe/cap_probe.py:15` 的 `idx._index`，那是 bidpilot 自己的 `CapabilityIndex`，与 documind 无关）。
`chunk_document` / `chunk_fixed` 等切块 API 也**未被** bidpilot 使用——bidpilot 自己按页切块（extract.py:153-190），这是对的，切块策略不该跨项目复用。

### 5.3 但"只用了公开接口"这个结论**不成立**——原因不是越界，是边界不存在

| 事实 | 证据 | 性质 |
|---|---|---|
| `documind/__init__.py` 全长 **0 行**，无 `__all__`、无导出 | `[探针]`：`__all__: MISSING`，`dir(documind)` 只剩子模块 | 没有"公开 API 面"这回事 |
| bidpilot 直接 import **子模块** `documind.parsing` / `documind.index` | sections.py:16、capabilities.py:15 | 子模块即 API，等于把 3.2k 行全当契约 |
| bidpilot 手搓 `ParsedDocument(...)` / `Paragraph(...)` 位置参数 | tests/conftest.py:38-63 | 依赖 dataclass **字段顺序**，字段一动测试静默变味 |
| bidpilot 运行时绕过 `slice_text` 直接读 `parsed.pages[i]` | sections.py:86-89、107 | 放弃唯一被测试钉住的强契约（P1-06） |
| 依赖声明是 `path = "../documind", editable = true`，无版本约束 | bidpilot/pyproject.toml:20,27-28 | 任何 documind 改动即时生效，无版本闸门（P0-03） |

**判定**：bidpilot 用的是"公开的名字"，但 documind 从未定义"什么是公开"。当前状态等价于**两个仓库共用一个源码树**——这不是复用，是把耦合藏进了 `pyproject.toml` 的一行 path。
替代方案（不新增包、不引第三方）：见 `bidpilot/docs/DEPENDENCY.md` §1「五个能力」与 §3「何时抽第三个包」。

---

## 6. 发现清单

### P0（阻塞：数据丢失 / 静默错误结果 / 兼容性破坏）

#### P0-01 身份信任边界缺失：ACL 判定全对，主体可伪造，端到端归零
- **证据**
  - app.py:58-66 `current_user` 只从 `X-User` 头取名字，无签名、无校验；用户不存在时**自动建号**（app.py:61-65）
  - app.py:219-222 `GET /users` 无鉴权，直接返回全部 `{name, role}` ——先枚举，再冒充
  - config.py:57 `session_secret` 定义了但全仓零引用（grep 只有定义处）——说明鉴权层是规划过、没落地
  - docs/API.md:3 把"默认带 `X-User: alice`"写进了对外契约
  - index.py:78-81、97-103 的 ACL 过滤与 `test_retrieval_respects_document_acl` 都是对的——问题不在判定
- **后果**：`curl -H 'X-User: admin' /query` 即穿透文档级 ACL。ADR-0003 承诺的"财务制度不会被 employee 看到"在 HTTP 入口处不成立，且日志里没有任何异常（冒充是合法请求）。这条使 ADR-0003 的安全性归属落空：它保证的是"给定 role 不泄露"，产品对外说的是"不泄露"。
- **建议动作**
  1. `current_user` 改为从 `Authorization: Bearer <token>` 解 `sub/role`，token 用已有的 `session_secret` 签（HS256，无新依赖可用 `hmac`+`json` 自签）；
  2. `X-User` 只在 `DOCUMIND_ENV=dev` 下接受，且启动时打印 `INSECURE_IDENTITY=1` 到 `/health`；
  3. `GET /users` 加 `Depends(require_admin)`；
  4. 补契约测试：带合法 token 但 `X-User: admin` 时 role 仍取 token 里的值。

#### P0-02 向量空间无指纹 + 静默降级：换模型/模型拉不下来时新旧向量混用
- **证据**
  - embed.py:86-98 `get_embedder` 捕获**所有**异常降级 `HashEmbedder`，只写一条 `logging.warning`
  - embed.py:69 `HashEmbedder.dim = 512`，与 bge-small-zh-v1.5 同维 → 降级后矩阵运算**不报错**，只是语义全错
  - db.py:92 `chunks.embedding` 只有 BLOB，**没有** `embed_model`/`dim`/`space_id` 列
  - retrieve.py:86-92 只对 `embedding IS NULL` 的行补算，已存在的旧空间向量原样沿用
  - embed.py:64 声称"评测报告里会明写用的是哪个 embedder"，实际 evals/report.py:26 无此字段（A5）
- **后果**：三个真实场景会静默产出错误检索——① 首次 `download` 失败降级入库，之后模型可用；② CI 用 `DOCUMIND_EMBED_BACKEND=hash` 建库后，本地用 bge 查询；③ 换 `embed_model` 后未重算历史块。三种情况都不报错、不掉测试，表现为"检索质量变差"，而评测报告无法佐证用了哪个后端。
- **建议动作**
  1. `chunks` 增 `embed_model: str`、`embed_dim: int`；`documents` 增 `embed_space`（`model@dim`）；
  2. `IndexStore.get` 发现空间不一致直接 fail-fast（`raise`），不要"尽力而为"；
  3. `get_embedder` 的降级必须进 `/health`（`degraded: ["embedder"]`）与 `EvalRun.metrics`；
  4. 报告写入 `embedder/dim/backend`，与 ADR 的承诺对齐（或改掉 embed.py:64 那句话）。

#### P0-03 没有"公开 API 面"，却有一个无版本约束的 path 依赖
- **证据**
  - documind/src/documind/__init__.py：**0 行**，无 `__all__` `[探针]`
  - bidpilot/pyproject.toml:20 `"documind"` + 27-28 `documind = { path = "../documind", editable = true }`，无 `>=`/`<`、无 marker
  - bidpilot 的 import 落在子模块（sections.py:16、capabilities.py:15）
  - bidpilot 无 `.github/workflows`，只有本地 26 项测试
- **后果**：documind 任何模块重命名、字段调整、函数签名变化，都**即时**作用于 bidpilot，且没有版本闸门能拦住；bidpilot 在 GitHub 上无法独立 `uv sync`（要 `../documind` 存在），也没有 CI 能在 PR 上发现断裂。团队章程 P0 定义里的"兼容性破坏"在此处是**默认状态**，不是事故。
- **建议动作**
  1. `documind/__init__.py` 显式导出 5 个符号并写 `__all__`（清单以 `DEPENDENCY.md` §1 为准）；
  2. bidpilot 全部改为 `from documind import parse_file, ParsedDocument, ...`（一处集中 import，便于审计）；
  3. 版本闸门：documind `0.1.0` + bidpilot 依赖写 `documind>=0.1,<0.2`（path 源保留，语义仍生效）；
  4. 两侧各加一个 `tests/test_documind_contract.py`：断言 5 个符号存在、`Paragraph` 字段集、`slice_text` 行为、`BM25Index.search` 返回形状。**这是唯一能在 documind 侧发现下游被破坏的位置**。

#### P0-04 空文本静默成功：ADR-0001 承诺的护栏不存在，空交付物一路绿灯
- **证据**
  - ADR-0001:21 明确写"无文本层时解析结果为空，**会显式报'未提取到文本'**，而不是静默产出空知识库"
  - 全仓 grep `未提取到文本` → **零命中**；parsing.py:343-345 `parse_file` 无任何空结果校验
  - `[探针]` 空文本 md：`pages=1 paragraphs=0 page_count=1 pages=['']`，四种策略 `chunks: 0`
  - pipeline.py:45,65-70,120-130：照常建 Document、照常 commit、照常返回 `strategy_stats` 全 0 的成功响应
  - bidpilot 端 `[探针]`：`sections: 0` / `chunks: 0` → **仍生成 35,726 字节的 .docx**，正文只有封面 + "目录"
- **后果**：扫描件/图片型 PDF（招标文件里很常见）整条链路"成功"：API 200、`page_count=1`、`chunks=0`，之后任何提问都回"文档中未找到相关条款"。用户看到的是"入库成功 + 系统很笨"，不是"这份文件没有文本层"。bidpilot 侧更糟：产出一份只有封面和目录的《技术应答初稿》并报成功。
- **建议动作**
  1. `parsing.parse_file` 出口加守卫：`pages` 全空或 `paragraphs` 为空 → `raise EmptyDocumentError("未提取到文本，需要 OCR")`；
  2. `ingest_document` 在解析后、切块前再断言一次（防自定义 Parser 绕过）；
  3. bidpilot `load_tender` 捕获该异常，返回可读错误（"该 PDF 无文本层，请先 OCR"），**不要**继续 build_sections；
  4. 补测试：空 md / 空页 PDF 各一条，断言抛错而非 200。

#### P0-05 bidpilot：mock 降级污染交付物，且标记在导出层被丢弃
- **证据**
  - config.py:38 `allow_mock` 默认 **True**（`BIDPILOT_ALLOW_MOCK != "0"`）
  - workflow.py:284-286 任一章节 LLM 失败 → `_mock_paragraphs`（模板句），状态仍是 `CITATION_VERIFIED`（workflow.py:312）
  - workflow.py:218,226 模板段落的 `kind = "技术响应（模板，LLM 不可用）"`
  - export.py:96 只渲染 `p["text"]`，**`kind` 从不落文档**；export.py:62-65 只有一句通用免责"须人工复核后使用"
  - pipeline.py:98-102 直接把 aggregated 交给 `build_docx`
- **后果**：LLM 网关挂掉时（本机 magpie 常态），产出的 .docx 结构完整、每段都带来源 id，**与真实生成不可区分**。投标场景下这就是"看起来能交、实际是模板句"的交付物——比报错危险得多，且没有任何一处 UI/文件能让评审员发现。
- **建议动作**
  1. `allow_mock` 默认改 `False`，Mock 只能显式开启；
  2. 任一章 `mocked=True` → docx 首页插入加粗横幅（"本节为模板应答，未调用模型"），文件名加 `_MOCK` 后缀；
  3. `build_docx` 渲染 `kind`（至少在被降级的段落尾注）；
  4. `PipelineResult.stats()["workflow"]["mocked"]` 提升为 CLI 的**非零退出码**。

### P1（严重：可解释性、承诺不实、可维护性）

#### P1-01 ADR-0002 的"接口不变"没有端口支撑，`Retriever` 是死代码
- **证据**：index.py:137-138 `Retriever` 零引用零实现；`VectorIndex.search(query_vec, ...)` 与 Protocol 签名不兼容（index.py:94）；retrieve.py:110 `np.vstack` 全量矩阵；db.py:92 BLOB 列；切换成本表见 §4
- **后果**：任何人都无法"实现一个协议就换后端"；ADR 给出的是结论而非路径，后续接手者会重走一遍发现过程
- **动作**：二选一——① 在 index.py 定义 `KeywordStore`/`VectorStore` 两个 Protocol，`BM25Index`/`VectorIndex` 实现之，`IndexStore` 只依赖 Protocol；② 删掉 `Retriever`，并在 ADR-0002 补"切换需改 4 处，清单见 ARCH-REVIEW §4"。**不允许保留现状**（假装有接口比没有更贵）

#### P1-02 DoclingParser 不可达，且实现与 ADR 的 span 要求自相矛盾
- **证据**：parsing.py:303-324（用 `len(blocks)>3000` 伪造分页，321-323）、336-340（只按后缀选）、config.py 无 parser 字段 `[探针]`、ADR-0001:8 与 25
- **后果**：ADR 的"留了适配器、装上就换"不成立；真接上会得到伪造页码，citation 的高亮会指向错误页（且不报错）
- **动作**：`Settings.parser: str = "auto"` + `_PARSERS` 改成注册表；`DoclingParser` 拿到真实 provenance 前，类 docstring 首行写"未接线，禁止使用"；ADR-0001 补一行"当前不可切换"

#### P1-03 拒答阈值：标定只产出建议，从不回写；网格与双后端共用同一常量
- **证据**：config.py:40 `refusal_rerank_threshold = -6.0` 硬编码；evals/runner.py:108 调 `calibrate_threshold`，146-150 只 `return {"recommended": ...}`；全仓只有 generate.py:135 读该配置，**没有任何写入路径**；runner.py:135 网格写死 `-14 + 0.5*i (i<29)` 即 `[-14, 0]`，贴合 LexicalReranker 的 `[-12,0]`（rerank.py:79），cross-encoder logits 可能 >0 扫不到；两个 reranker 共用一个阈值
- **后果**：ADR-0004 的核心论点"阈值由评测集标定，不是拍脑袋的 0.6"在代码里不成立——实际是拍脑袋的 -6.0。切到 lexical 兜底时（模型下载失败即自动切），分数域不同而阈值不变，拒答率静默漂移，且评测报告不会提示
- **动作**：① `EvalRun` 记录 `recommended_threshold`，启动时读 `DOCUMIND_REFUSAL_THRESHOLD`（未设则用最近一次标定值并对不上就打警告）；② 阈值按 `reranker.name` 分桶（两套常量），改 reranker 必须换阈值；③ 网格按实际分数分位数生成，范围覆盖 `[min-1, max+1]`

#### P1-04 `IndexStore` 缓存只在进程内失效，跨进程永不刷新
- **证据**：retrieve.py:66-76 按 `strategy` 缓存；pipeline.py:121 `store.invalidate()` 只影响当前进程；retrieve.py:63 的 docstring 自认"多副本会短暂不一致"（实际是**永久**不一致，不是短暂）
- **后果**：CLI `documind ingest` + 已在跑的 `uvicorn`（或 `--workers >1`）时，新文档查不到、已删文档仍命中；日志无异常，只有用户觉得"系统没更新"
- **动作**：`documents`/`chunks` 增 `index_version`（或用 `max(chunks.rowid)`），`get()` 每次比对版本号，不一致即重建；或直接提供 `POST /admin/reindex` 并在 `/health` 暴露 `index_version`

#### P1-05 `/query/stream` 名不副实：全量等待后一次吐两个事件
- **证据**：pipeline.py:222-227 `ans = await ask(...)` 之后才 `yield`；app.py:197-216 以 `text/event-stream` 返回
- **后果**：前端按 SSE 写的"先证据卡、后答案"时序拿不到预期收益；P95 延迟等于非流式，还多了一层包装
- **动作**：要么真实流式（llm.py 加 `stream=True`，generate 逐 token yield，证据事件在检索后立即发），要么改名 `/query/events` 并在 docs/API.md 标注"非流式"

#### P1-06 bidpilot 绕过 `slice_text`，并手搓 dataclass 夹具 —— 唯一强契约被丢弃
- **证据**：sections.py:86-89 `self.parsed.pages[page-1]`、107 `len(self.parsed.paragraphs)`；对照 chunking.py:54 是 `slice_text` 的唯一消费者；`test_chunking.py:43` 全策略断言 `sliced.strip() == c.text.strip()`；tests/conftest.py:38-63 手工构造 `Paragraph/ParsedDocument`（位置参数）
- **后果**：documind 一旦调整 `pages` 的构造（加页眉清洗、改分页策略、slice 语义），bidpilot 的 quote 校验页码会静默漂移，而 documind 的 40 项测试全绿——两边测的不是同一个东西
- **动作**：`TenderDoc.page_text` 改为 `self.parsed.slice_text(p, 0, p, len(self.parsed.pages[p-1]))`（或明确写入 DEPENDENCY.md：`pages` 是规范化页文本，禁止再加工）；夹具改成 `parse_file(tmp_path/"x.md")` 构造，禁止直接实例化数据类

#### P1-07 章节边界只到"页"粒度，丢掉了 documind 已经给到段落的字符偏移
- **证据**：sections.py:144-145 `start <= p.page <= end` 按页取段；182-187 `end = max(start, nxt-1)`（同页起两章时区间塌缩/重叠）；对照 parsing.py:29-38 `Paragraph.start/end` 是页内精确偏移
- **后果**：同一页上有两章时，后一章吞掉前一章的开头；`extract.py:244` 的 page 范围校验比真实章节宽，模型把邻章条目塞进来不会被拦，只会在 quote 校验里碰运气
- **动作**：`Section` 保存 `(page_start, char_start, page_end, char_end)`；`_paragraphs_in` 用 `(page, start)` 元组比较；`extract._validate` 的边界校验同步升级

#### P1-08 `outline()` 双实现，且上游零测试
- **证据**：parsing.py:75-86 定义；唯一消费者是 bidpilot（sections.py:100-101）；documind 自己的 `/documents/{id}/outline` 另起一份（app.py:168-179，从 `strategy == "structure"` 的 chunk 反推），并且只在用 structure 入库时非空
- **后果**：两个"章节树"定义会漂移；API 返回的层级信息比 `outline()` 少（只有 `section_path` 字符串去重），前端按哪个都说不清
- **动作**：`/outline` 改为解析期落库的 `documents.outline`（入库时写 `parsed.outline()`），app 直接读；documind 补 `outline()` 单测；或删掉 `outline()`，让 bidpilot 自己建（但那要放弃 `ParsedDocument` 的层级语义）

#### P1-09 evals 跨包调用 `pipeline._page_texts`
- **证据**：evals/runner.py:39 `from ..pipeline import _page_texts`，43 处调用
- **后果**：pipeline 内部改名即破坏评测链路，而评测是"阈值/指标"的唯一来源——私有边界在最重要的地方失守
- **动作**：提升为 `pipeline.page_texts()`（或移到 `db.py` 成 `load_page_texts(session, doc_ids)`），runner 走公开名

#### P1-10 `IndexItem` 把入库模型塞进通用检索类型
- **证据**：index.py:29-41 必填 `doc_id`/`strategy`；bidpilot capabilities.py:110-113 用假值 `doc_id="capabilities"`, `strategy="capability"` 填充
- **后果**：下游被迫编造无意义字段；将来 `doc_id` 若承担 ACL/租户语义（ADR-0003 的 P1 计划正是这个方向），bidpilot 会继承一个它不需要也不理解的维度
- **动作**：`doc_id`/`strategy` 给默认值，或拆出 `BM25Doc(id, text)` 供通用场景；`BM25Index` 构造只要求 `id/text`

#### P1-11 示例能力库是默认值，且"空库"与"无命中"不可区分，警告不进交付物
- **证据**：config.py:47-51 默认指向 `src/bidpilot/data/capabilities.json`；该文件 `_meta` 自述"示例数据…不具备商业意义"；pipeline.py:43 只把它塞进 `stats()`；export.py 不渲染；capabilities.py:121-126 "无命中"直接判 `MISSING` 并写"BM25 分数为 0"（空库与无命中同一分支）
- **后果**：默认配置下，一份真实招标文件会得到"全部不满足"的能力矩阵，并据此生成全篇"偏离说明 + 补齐措施"——技术分归零，且没有任何一处提示"你还没配能力库"
- **动作**：`load_capabilities` 返回 `is_example: bool`；示例库生效时 docx 首页横幅 + CLI 非零退出码；`CapabilityIndex` 区分 `empty_library` 与 `no_match`（两种 evidence 文案、两种 status：`MISSING` vs `NEEDS_REVIEW`）

---

## 7. 阻塞项（必须改代码，本文不改）

| # | 阻塞项 | 关联 | 需要谁 |
|---|---|---|---|
| B1 | `ParsedDocument` 增 `is_empty`/`require_text()`，`parse_file` 抛 `EmptyDocumentError` | P0-04 | @dev（documind） |
| B2 | `chunks` 增 `embed_model`/`embed_dim` 列 + 迁移脚本 | P0-02 | @dev + @data（要重算历史向量，有成本） |
| B3 | `documind/__init__.py` 导出 5 符号 + 两侧契约测试 | P0-03 | @dev（两个仓库各一 PR） |
| B4 | 身份层：token 签发/校验 + `/users` 加 admin | P0-01 | @dev（涉及对外接口，**须 @hermes 报批**） |
| B5 | bidpilot mock 横幅 + 默认关 mock | P0-05 | @dev（bidpilot） |
| B6 | `Section` 升级到字符偏移 | P1-07 | @dev（bidpilot），会动 `extract` 的边界校验 |

---

## 8. 三条最可能翻车的假设

1. **"ADR 里写了就等于实现了"**。五篇 ADR 里，`0003` 的 ACL 判定真、`0005` 的四策略并存真；`0001` 的空文档护栏、`0002` 的接口不变、`0004` 的阈值标定三条**只有文档没有代码**。任何人按 ADR 做技术尽调，会在三处踩空。翻车信号：有人问"你们的 docling 适配器怎么开"。
2. **"40 项 + 26 项测试全绿 = 边界是安全的"**。测试全在**进程内**：documind 的 40 项不 import 下游，bidpilot 的 26 项不校验 documind 版本。真正会断的地方（`pyproject` 的 path 依赖、`__init__` 无导出、无 CI）恰好没有任何测试覆盖。翻车信号：documind 某次重构后 bidpilot 的 5 个夹具测试红了但没人知道为什么。
3. **"降级是安全的，因为降级路径有写"**。hash embedder、lexical reranker、模板应答三条降级路径都写了，但三条都**不可观测**：不体现在 `/health`、不体现在评测报告、不体现在交付物。降级本身不是问题，"无人知道降级了"才是。翻车信号：一份报告说 Recall@5 = 0.82，没人知道那份是用 hash embedder 跑的。

## R1 风险（复核本身的不确定）

- 两仓库工作区都有未提交改动，行号只对当前工作区成立；`c634140` 里的 bidpilot 与我现在读到的不是同一份代码。
- P0-01 的"可伪造"由代码路径推定（无签名校验），未做线上请求验证——启动服务会写 `documind.db`，本次未执行。
- 性能类结论（全表 `vstack`、全表取标题 retrieve.py:176）只做了复杂度判断，未压测。

---

## 附录 A：探针命令与输出

```bash
# 1) documind 的公开面 / parser 配置 / 空文档行为
cd D:/code/documind && ./.venv/Scripts/python.exe -c '...'
# init exports: ['parsing']
# __all__: MISSING
# settings has parser attr: False | env prefix fields: []
# pdf parser -> pypdf | docling reachable via get_parser: False
# empty doc: pages=1 paragraphs=0 page_count=1 repr_pages=['']
# fixed chunks: 0 parents: 0 / structure chunks: 0 / parent_child chunks: 0

# 2) bidpilot 在空解析结果上的行为
cd D:/code/bidpilot && ./.venv/Scripts/python.exe -c '...'
# sections: 0 disq: 0
# chunks: 0
# docx bytes: 35726
# paragraphs in docx: ['空扫描件', 'BidPilot 自动生成 · ...须人工复核后使用。', '目录', '（在 Word/WPS 中右键"更新域"生成目录页码）']

# 3) 测试基线
cd D:/code/documind && ./.venv/Scripts/python.exe -m pytest --collect-only -q   # 40 tests collected
cd D:/code/bidpilot && ./.venv/Scripts/python.exe -m pytest --collect-only -q   # 26 tests collected
```