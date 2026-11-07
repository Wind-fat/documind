"use client";

import { useCallback, useEffect, useState } from "react";
import PageHeader from "@/components/PageHeader";
import { EmptyState, ErrorState, LoadingState } from "@/components/States";
import { getHealth, getStats, toApiError, type ApiError } from "@/lib/api";
import { ms, percent, usd } from "@/lib/format";
import { useCurrentUser } from "@/lib/session";
import type { Health, Stats } from "@/lib/types";

export default function StatsPage() {
  const user = useCurrentUser();
  const [stats, setStats] = useState<Stats | null>(null);
  const [health, setHealth] = useState<Health | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<ApiError | null>(null);
  const [healthError, setHealthError] = useState<ApiError | null>(null);

  const load = useCallback(() => {
    setLoading(true);
    setError(null);
    setHealthError(null);
    getStats()
      .then((data) => {
        setStats(data);
        setLoading(false);
      })
      .catch((err: unknown) => {
        setError(toApiError(err));
        setLoading(false);
      });
    getHealth()
      .then(setHealth)
      .catch((err: unknown) => setHealthError(toApiError(err)));
  }, []);

  useEffect(() => {
    load();
  }, [load, user]);

  const kpis = [
    { label: "答案数", key: "answers", value: stats ? String(stats.answers) : "—", hint: "assistant 消息总数" },
    { label: "平均延迟", key: "latency", value: stats ? ms(stats.avg_latency_ms) : "—", hint: "端到端 latency_ms 均值" },
    { label: "累计成本", key: "cost", value: stats ? usd(stats.total_cost_usd) : "—", hint: "sum(cost_usd)" },
    {
      label: "拒答率",
      key: "refusal",
      value: stats ? percent(stats.refusal_rate) : "—",
      hint: "refused=true 占比",
      tone: stats && stats.refusal_rate > 0.3 ? "warn" : "ok",
    },
  ];

  return (
    <>
      <PageHeader
        title="运行指标"
        subtitle="/stats · /health"
        right={
          <>
            <span>Bearer · {user}</span>
            <button type="button" className="btn btn-sm" onClick={load} disabled={loading}>
              刷新
            </button>
          </>
        }
      />

      <div className="page">
        <div className="page-narrow" style={{ display: "flex", flexDirection: "column", gap: 16 }}>
          {error ? <ErrorState message={error.detail} status={error.status} onRetry={load} /> : null}

          {loading && !stats ? <LoadingState label="读取 /stats…" rows={2} /> : null}

          {stats ? (
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(190px, 1fr))", gap: 12 }}>
              {kpis.map((kpi) => (
                <div className="card" key={kpi.key} style={{ padding: "14px 16px", display: "flex", flexDirection: "column", gap: 4 }}>
                  <span className="stat-label">{kpi.label}</span>
                  <span className={`stat-value ${kpi.tone ?? ""}`} style={{ fontSize: 24 }}>
                    {kpi.value}
                  </span>
                  <span className="card-sub mono" style={{ fontSize: 10.5 }}>
                    {kpi.hint}
                  </span>
                </div>
              ))}
            </div>
          ) : null}

          {!loading && !stats && !error ? (
            <EmptyState icon="◎" title="暂无统计数据" desc="还没有问答记录，去问答台提一个问题再回来。" />
          ) : null}

          <div className="card">
            <div className="card-head">
              <span className="card-title">
                服务健康
                {health ? <span className="tag ok">online</span> : healthError ? <span className="tag danger">offline</span> : null}
              </span>
              <span className="card-sub">GET /health</span>
            </div>

            {healthError ? (
              <div style={{ padding: 14 }}>
                <ErrorState title="后端不可达" message={healthError.detail} status={healthError.status} onRetry={load} />
              </div>
            ) : null}

            {health ? (
              <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))", gap: 1, background: "var(--border)" }}>
                {[
                  { label: "文档数", value: String(health.documents) },
                  { label: "chunk 数", value: String(health.chunks) },
                  { label: "embedder", value: health.embedder },
                  { label: "索引版本", value: `v${health.index_version}` },
                  { label: "生成模型", value: health.llm.model },
                  { label: "LLM 网关", value: health.llm.base_url },
                ].map((row) => (
                  <div key={row.label} className="stat-cell" style={{ background: "var(--panel)", padding: "12px 16px" }}>
                    <span className="stat-label">{row.label}</span>
                    <span className="mono" style={{ fontSize: 12.5, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }} title={row.value}>
                      {row.value}
                    </span>
                  </div>
                ))}
              </div>
            ) : null}

            {!health && !healthError ? <LoadingState label="探测后端…" rows={1} /> : null}
          </div>

          <div className="card" style={{ padding: "14px 16px", display: "flex", flexDirection: "column", gap: 8 }}>
            <span className="section-title" style={{ margin: 0 }}>
              指标口径
            </span>
            <div className="card-sub" style={{ lineHeight: 1.9 }}>
              答案数来自 messages 表中 role=assistant 的行；平均延迟为这些行的 latency_ms 均值；累计成本为 cost_usd 求和；
              拒答率为 refused=true 的占比。指标随问答实时变化（含被 ACL 过滤后的拒答）。
            </div>
          </div>
        </div>
      </div>
    </>
  );
}