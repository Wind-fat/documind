import type { EvalMetricValue } from "./types";

export function pageLabel(pageStart: number, pageEnd: number): string {
  if (!pageStart && !pageEnd) return "页码未知";
  const start = pageStart || pageEnd;
  const end = pageEnd || pageStart;
  if (start === end) return `第 ${start} 页`;
  return `第 ${start}–${end} 页`;
}

export function isMultiPage(pageStart: number, pageEnd: number): boolean {
  return !!pageStart && !!pageEnd && pageStart !== pageEnd;
}

export function rankLabel(rank: number | null | undefined): string {
  if (rank === null || rank === undefined) return "未召回";
  return `#${rank + 1}`;
}

export function score(value: number | null | undefined, digits = 3): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  return value.toFixed(digits);
}

export function percent(value: number | null | undefined, digits = 1): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  return `${(value * 100).toFixed(digits)}%`;
}

export function ms(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  return value >= 1000 ? `${(value / 1000).toFixed(2)} s` : `${Math.round(value)} ms`;
}

export function usd(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  if (value === 0) return "$0";
  if (value < 0.01) return `$${value.toFixed(6)}`;
  return `$${value.toFixed(4)}`;
}

export function dateTime(value: string | null | undefined): string {
  if (!value) return "—";
  const d = new Date(value);
  if (Number.isNaN(d.getTime())) return value;
  const pad = (n: number): string => String(n).padStart(2, "0");
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}`;
}

export const METRIC_LABELS: Record<string, string> = {
  recall_at_5: "Recall@5",
  context_recall: "Context Recall",
  mrr: "MRR",
  faithfulness: "Faithfulness",
  answer_relevancy: "Answer Relevancy",
  citation_validity: "引用有效率",
  refusal_accuracy: "拒答准确率",
  false_refusal_rate: "误拒率",
  latency_p50_ms: "P50 延迟",
  latency_p95_ms: "P95 延迟",
  cost_per_query_usd: "单问成本",
  total_cost_usd: "总成本",
  n_cases: "用例数",
  n_answerable: "可答用例",
  n_unanswerable: "不可答用例",
};

/** 评测表格列顺序；后端多出来的指标会追加在后面。 */
export const METRIC_ORDER = [
  "recall_at_5",
  "context_recall",
  "mrr",
  "faithfulness",
  "answer_relevancy",
  "citation_validity",
  "refusal_accuracy",
  "false_refusal_rate",
  "latency_p50_ms",
  "latency_p95_ms",
  "cost_per_query_usd",
];

export const PCT_METRICS = new Set([
  "recall_at_5",
  "context_recall",
  "mrr",
  "faithfulness",
  "answer_relevancy",
  "citation_validity",
  "refusal_accuracy",
  "false_refusal_rate",
]);

export const MS_METRICS = new Set(["latency_p50_ms", "latency_p95_ms", "latency_ms"]);
export const USD_METRICS = new Set(["cost_per_query_usd", "total_cost_usd", "cost_usd"]);

export function formatMetric(key: string, value: EvalMetricValue): string {
  if (value === null || value === undefined) return "—";
  if (typeof value === "string") return value;
  if (PCT_METRICS.has(key)) return percent(value);
  if (MS_METRICS.has(key)) return ms(value);
  if (USD_METRICS.has(key)) return usd(value);
  if (key.startsWith("n_")) return String(value);
  if (Math.abs(value) < 10) return value.toFixed(3);
  return String(Math.round(value));
}

export const STRATEGY_LABELS: Record<string, string> = {
  fixed: "fixed",
  semantic: "semantic",
  structure: "structure",
  parent_child: "parent_child",
};

export function strategyLabel(value: string): string {
  return STRATEGY_LABELS[value] ?? value;
}