# ADR-0002 混合检索：BM25 ∥ 向量 → RRF → cross-encoder 重排

- 状态：已采纳（2026-11）

## 背景

制度/条款类问题有两类：一类是"人名、编号、金额、条款号"这种字面精确匹配（"第六条第 2 款"），
另一类是"年假能不能攒到明年"这种说法与原文完全不同、只能靠语义。任何单一检索都会在另一半上翻车。

## 决策

1. **两路召回**：BM25（中文按字符 bigram 切分）+ 向量（bge-small-zh-v1.5, 512 维），各取 Top-30。
2. **RRF 融合**（k=60）：只用名次不用分数——BM25 分数与 cosine 相似度量纲不同，归一化会引入新的偏差。
3. **cross-encoder 重排**（bge-reranker-base）：从融合后的前 10 条（`rerank_candidates=10`，不低于 top_k）里取 Top-5 喂模型。

## 端口在哪、切换要改哪几处（2026-11 复核后修订）

"换 OpenSearch / pgvector 接口不变"这句话在 2026-11 的架构复核里被判定为不成立
（docs/ARCH-REVIEW.md A2 / P1-01），现在改成两句话：

1. **检索口径有端口**：`index.KeywordStore`（`search(query, top_n, allowed)`）与
   `index.VectorStore`（`search(query_vec, top_n, allowed)`）。`BM25Index` / `VectorIndex` 是
   dev 实现，`SearchIndex.bm25/vector` 的注解就是这两个 Protocol，`IndexStore` 只依赖端口，
   所以"换实现"这一层确实不需要动编排代码。
   （旧的 `Retriever` Protocol 已删除：它声明 `search(query: str, ...)`，与向量检索收
   `np.ndarray` 的签名不兼容，且全仓零实现零引用。）
2. **切换成本不是"零"**：换后端仍需改 **4 处**，清单见 docs/ARCH-REVIEW.md §4 ——
   ① `chunks.embedding` 的 BLOB 列类型（db.py:92 一带）；② `IndexStore` 的全表 `np.vstack`
   组装（retrieve.py）；③ 实现 `VectorStore` 的 `<=>`/kNN 查询本身；
   ④ 权限语义：现在用"负分掩码"，OpenSearch 侧必须改成 query filter（否则无权文档会
   以低分形式上榜）。任何一处不改就上线，等于换了个会静默错排率的后端。

## 为什么不用 SQLite FTS5

FTS5 的 unicode61 分词器把连续中文视为一个 token，`"年假可以折现吗"` 匹配不到 `"员工享有年假"`，
中文场景直接失效。字符 bigram 是无词典中文检索的标准做法，且不引入分词依赖。生产环境换 OpenSearch（有 IK 插件）。

## 为什么必须有重排

RRF 保序但不判相关性：融合后的第 7-12 位经常是"字面对得上但答非所问"的块。cross-encoder 逐对打分，
在评测集上把 Recall@5 从 RRF-only 的水平抬上去（见 `evals/reports/ablation.md` 的对照）。
代价是延迟：focus@256×10 实测重排 P50 2 711 ms、`retrieve()` P50 2 821 ms（`docs/RERANK-PERF.md` §4/§5）。

## 放弃了什么

- 学习型融合（LTR）：需要标注数据，当前量级不值得。
- 单一向量库（如 Milvus）：万级 chunk 用进程内 numpy 暴力搜索 <10ms，多一个组件多一次不一致。
- 外部索引服务的强一致缓存：`IndexStore` 的失效靠库里的 `index_state.version` + 行数指纹
  每请求比对（P1-04），多副本之间仍有"一次请求"窗口的不一致；这在单机 dev 是可接受的，
  在 prod 是必须被外部索引替换的部分。