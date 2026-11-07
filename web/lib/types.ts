/** 与 docs/API.md 逐一对应的类型定义，字段名以后端契约为主。 */

export type Strategy = "fixed" | "semantic" | "structure" | "parent_child";

export const STRATEGIES: { value: Strategy; label: string; hint: string }[] = [
  { value: "fixed", label: "fixed · 定长切块", hint: "按固定字符窗口切分，召回稳定但易断句" },
  { value: "semantic", label: "semantic · 语义切块", hint: "按句向量相似度断句，边界更贴合语义" },
  { value: "structure", label: "structure · 结构切块", hint: "按文档章节层级切分，保留 section_path" },
  { value: "parent_child", label: "parent_child · 父子块", hint: "子块召回、父块补全上下文" },
];

export type UserName = "alice" | "bob" | "admin";

export const USERS: { name: UserName; role: string; label: string }[] = [
  { name: "alice", role: "employee", label: "alice · 员工" },
  { name: "bob", role: "finance", label: "bob · 财务" },
  { name: "admin", role: "admin", label: "admin · 管理员" },
];

export interface Health {
  status: string;
  documents: number;
  chunks: number;
  embedder: string;
  /** 库内索引版本（P1-04）：别用进程内计数，跨进程失效靠它 */
  index_version: number;
  /** 版本之外的兜底指纹：文档数/块数/token 总数 */
  index_fingerprint?: { documents: number; chunks: number; tokens: number };
  llm: { base_url: string; model: string };
}

/** 检索命中，也是证据卡的数据源。 */
export interface Hit {
  chunk_id: string;
  doc_id: string;
  doc_title: string;
  text: string;
  page_start: number;
  char_start: number;
  page_end: number;
  char_end: number;
  section_path: string;
  score: number;
  bm25_rank: number | null;
  vec_rank: number | null;
  rrf: number;
}

export interface Citation {
  ref: number;
  doc_id: string;
  doc_title: string;
  chunk_id: string;
  page_start: number;
  char_start: number;
  page_end: number;
  char_end: number;
  quote: string;
  section_path: string;
}

export interface AnswerDebug {
  intent?: string;
  strategy?: Strategy | string;
  hits?: Hit[];
  candidates?: number;
  conversation_id?: string;
  total_ms?: number;
  settings_topk?: number;
  contexts?: number;
  [key: string]: unknown;
}

export interface Answer {
  text: string;
  citations: Citation[];
  refused: boolean;
  reason: string;
  rewritten_query: string;
  latency_ms: number;
  cost_usd: number;
  prompt_tokens: number;
  completion_tokens: number;
  retrieval_ms: number;
  generation_ms: number;
  top_score: number;
  selfcheck?: { supported?: boolean | null; retried?: boolean; unparsed?: boolean; error?: string; [key: string]: unknown };
  debug: AnswerDebug;
}

export interface DocumentMeta {
  id: string;
  title: string;
  filename: string;
  pages: number;
  acl_roles: string[];
  visible: boolean;
  created_at: string;
}

export interface PageText {
  doc_id: string;
  page: number;
  text: string;
}

export interface OutlineSection {
  /** 章节路径（`第一章 总则 > 第一节 目的`），与老接口同义 */
  title: string;
  page: number;
  /** 1=章/节；解析层的层级 */
  level: number;
  /** 标题本身（不含父路径） */
  heading: string;
}

export interface OutlineNode {
  title: string;
  page: number;
  level: number;
  children: OutlineNode[];
}

export interface Outline {
  doc_id: string;
  sections: OutlineSection[];
  /** 层级形式；要父子关系读这个，不要去切 sections[].title 的 ` > ` */
  tree: OutlineNode[];
  /** parsed=解析期落库的章节树；missing=迁移前历史文档；invalid=库里 JSON 损坏 */
  outline_source: "parsed" | "missing" | "invalid";
}

export interface UploadResult {
  document_id: string;
  title: string;
  pages: number;
  paragraphs: number;
  strategy_stats: Record<string, Record<string, number>>;
  elapsed_ms: number;
  embedder: string;
}

export type EvalMetricValue = number | string | null;

export interface EvalRun {
  id: string;
  strategy: Strategy | string;
  dataset: string;
  n_cases: number;
  metrics: Record<string, EvalMetricValue>;
  created_at: string;
  report_path?: string | null;
}

export interface Stats {
  answers: number;
  avg_latency_ms: number;
  total_cost_usd: number;
  refusal_rate: number;
}

/** 原文弹层的定位目标：来自证据卡（chunk 跨度）或引用（精确定位后的跨度）。 */
export interface ViewerTarget {
  doc_id: string;
  doc_title: string;
  page_start: number;
  page_end: number;
  char_start: number;
  char_end: number;
  quote?: string;
  section_path?: string;
  chunk_id?: string;
}

export interface QueryInput {
  question: string;
  strategy: Strategy;
  conversation_id?: string | null;
  top_k?: number;
  selfcheck?: boolean;
}