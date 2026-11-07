"use client";

import type { ApiError } from "@/lib/api";
import { ms, percent, usd } from "@/lib/format";
import type { Stats } from "@/lib/types";

export default function StatsStrip({
  stats,
  loading,
  error,
  onRetry,
}: {
  stats: Stats | null;
  loading: boolean;
  error: ApiError | null;
  onRetry: () => void;
}) {
  const cells = [
    { label: "答案数", value: stats ? String(stats.answers) : "—", tone: "" },
    { label: "平均延迟", value: stats ? ms(stats.avg_latency_ms) : "—", tone: "" },
    { label: "累计成本", value: stats ? usd(stats.total_cost_usd) : "—", tone: "" },
    {
      label: "拒答率",
      value: stats ? percent(stats.refusal_rate) : "—",
      tone: stats && stats.refusal_rate > 0.3 ? "warn" : "ok",
    },
  ];

  return (
    <div>
      <div className="section-title">
        <span>运行指标 / stats</span>
        <span style={{ textTransform: "none", letterSpacing: 0 }}>
          {loading ? (
            <span className="stream-hint">
              <span className="spinner" /> 刷新中
            </span>
          ) : error ? (
            <button type="button" className="btn btn-ghost btn-sm" onClick={onRetry}>
              指标加载失败 · 重试
            </button>
          ) : null}
        </span>
      </div>
      <div className="stat-strip">
        {cells.map((cell) => (
          <div className="stat-cell" key={cell.label}>
            <span className="stat-label">{cell.label}</span>
            <span className={`stat-value ${cell.tone}`}>{cell.value}</span>
          </div>
        ))}
      </div>
    </div>
  );
}