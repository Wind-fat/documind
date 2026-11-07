"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import PageHeader from "@/components/PageHeader";
import { EmptyState, ErrorState, LoadingState } from "@/components/States";
import { getEvalReportHtml, listEvalRuns, toApiError, type ApiError } from "@/lib/api";
import { METRIC_LABELS, METRIC_ORDER, dateTime, formatMetric, strategyLabel } from "@/lib/format";
import type { EvalRun } from "@/lib/types";

const DOWN_METRICS = new Set([
  "false_refusal_rate",
  "latency_p50_ms",
  "latency_p95_ms",
  "cost_per_query_usd",
  "total_cost_usd",
]);

const STRATEGY_ORDER = ["fixed", "semantic", "structure", "parent_child"];

function sortStrategies(values: string[]): string[] {
  return [...values].sort((a, b) => {
    const ai = STRATEGY_ORDER.indexOf(a);
    const bi = STRATEGY_ORDER.indexOf(b);
    if (ai === -1 && bi === -1) return a.localeCompare(b);
    if (ai === -1) return 1;
    if (bi === -1) return -1;
    return ai - bi;
  });
}

export default function EvalsPage() {
  const [runs, setRuns] = useState<EvalRun[] | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<ApiError | null>(null);

  const [reportHtml, setReportHtml] = useState<string | null>(null);
  const [reportError, setReportError] = useState<ApiError | null>(null);
  const [reportLoading, setReportLoading] = useState(true);

  const loadRuns = useCallback(() => {
    setLoading(true);
    setError(null);
    listEvalRuns()
      .then((data) => {
        setRuns(data);
        setLoading(false);
      })
      .catch((err: unknown) => {
        setError(toApiError(err));
        setLoading(false);
      });
  }, []);

  const loadReport = useCallback(() => {
    setReportLoading(true);
    setReportError(null);
    getEvalReportHtml()
      .then((html) => {
        setReportHtml(html);
        setReportLoading(false);
      })
      .catch((err: unknown) => {
        setReportError(toApiError(err));
        setReportHtml(null);
        setReportLoading(false);
      });
  }, []);

  useEffect(() => {
    loadRuns();
    loadReport();
  }, [loadRuns, loadReport]);

  // 每个策略取最新一次运行（后端按 created_at desc 返回）
  const latest = useMemo(() => {
    const map = new Map<string, EvalRun>();
    for (const run of runs ?? []) {
      if (!map.has(run.strategy)) map.set(run.strategy, run);
    }
    return map;
  }, [runs]);

  const strategies = useMemo(() => sortStrategies(Array.from(latest.keys())), [latest]);

  const metricKeys = useMemo(() => {
    const present = new Set<string>();
    latest.forEach((run) => {
      Object.keys(run.metrics ?? {}).forEach((key) => present.add(key));
    });
    const ordered = METRIC_ORDER.filter((key) => present.has(key));
    const extras = Array.from(present)
      .filter((key) => !METRIC_ORDER.includes(key))
      .sort();
    return [...ordered, ...extras];
  }, [latest]);

  const bestByMetric = useMemo(() => {
    const map = new Map<string, number>();
    for (const key of metricKeys) {
      const values: number[] = [];
      for (const strategy of strategies) {
        const value = latest.get(strategy)?.metrics?.[key];
        if (typeof value === "number") values.push(value);
      }
      if (values.length < 2) continue;
      if (new Set(values).size < 2) continue; // 全相同就不标最佳
      map.set(key, DOWN_METRICS.has(key) ? Math.min(...values) : Math.max(...values));
    }
    return map;
  }, [metricKeys, strategies, latest]);

  const totalRuns = runs?.length ?? 0;

  return (
    <>
      <PageHeader
        title="评测看板"
        subtitle={`${totalRuns} 次运行 · ${strategies.length} 种策略`}
        right={
          <>
            <span>/evals/runs</span>
            <button
              type="button"
              className="btn btn-sm"
              onClick={() => {
                loadRuns();
                loadReport();
              }}
              disabled={loading}
            >
              刷新
            </button>
          </>
        }
      />

      <div className="page">
        <div className="page-narrow" style={{ display: "flex", flexDirection: "column", gap: 16 }}>
          <div className="card">
            <div className="card-head">
              <span className="card-title">
                策略对比
                <span className="tag accent">按 strategy 分列</span>
              </span>
              <span className="card-sub">每列取该策略最新一次运行 · 绿=该指标最优</span>
            </div>

            {loading ? <LoadingState label="读取 /evals/runs…" rows={4} /> : null}
            {error ? (
              <div style={{ padding: 14 }}>
                <ErrorState message={error.detail} status={error.status} onRetry={loadRuns} />
              </div>
            ) : null}

            {!loading && !error && totalRuns === 0 ? (
              <EmptyState
                icon="◱"
                title="还没有评测运行"
                desc="先以 admin 身份触发一次评测：POST /evals/run {dataset:'default', strategies:[...]}，随后这里的对比表和 ablation.html 会出现。"
              />
            ) : null}

            {!loading && !error && strategies.length > 0 && metricKeys.length > 0 ? (
              <div className="eval-table-wrap">
                <table className="eval-table">
                  <thead>
                    <tr>
                      <th style={{ minWidth: 150 }}>指标</th>
                      {strategies.map((strategy) => {
                        const run = latest.get(strategy);
                        return (
                          <th key={strategy}>
                            <div className="eval-strategy">{strategyLabel(strategy)}</div>
                            <div style={{ fontWeight: 400, color: "var(--faint)", textTransform: "none", letterSpacing: 0, fontSize: 10 }}>
                              {run ? `${run.dataset} · n=${run.n_cases} · ${dateTime(run.created_at).slice(5)}` : "无运行"}
                            </div>
                          </th>
                        );
                      })}
                    </tr>
                  </thead>
                  <tbody>
                    {metricKeys.map((key) => {
                      const best = bestByMetric.get(key);
                      return (
                        <tr key={key}>
                          <td>
                            {METRIC_LABELS[key] ?? key}
                            <span className="mono" style={{ color: "var(--faint)", marginLeft: 6, fontSize: 10 }}>
                              {key}
                            </span>
                          </td>
                          {strategies.map((strategy) => {
                            const raw = latest.get(strategy)?.metrics?.[key];
                            const isBest = typeof raw === "number" && best !== undefined && raw === best;
                            return (
                              <td key={strategy} className={isBest ? "best" : ""}>
                                {formatMetric(key, raw ?? null)}
                                {isBest ? " ▲" : ""}
                              </td>
                            );
                          })}
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            ) : null}
          </div>

          <div className="card" style={{ overflow: "hidden" }}>
            <div className="card-head">
              <span className="card-title">
                完整报告
                <span className="tag">ablation.html</span>
              </span>
              <a className="report-link" href="/api/evals/report" target="_blank" rel="noreferrer">
                新窗口打开 ↗
              </a>
            </div>

            {reportLoading ? <LoadingState label="读取 /evals/report…" rows={5} /> : null}
            {!reportLoading && reportError ? (
              <EmptyState
                icon="◌"
                title={reportError.status === 404 ? "还没有评测报告" : "报告加载失败"}
                desc={
                  reportError.status === 404
                    ? "后端 /evals/report 返回 404：先跑一次 POST /evals/run，生成 evals/reports/ablation.html。"
                    : reportError.detail
                }
                action={
                  <button type="button" className="btn btn-sm" onClick={loadReport}>
                    重试
                  </button>
                }
              />
            ) : null}
            {!reportLoading && !reportError && reportHtml !== null ? (
              <iframe className="report-frame" title="评测报告 ablation.html" srcDoc={reportHtml} />
            ) : null}
          </div>

          {runs && runs.length > 0 ? (
            <div className="card">
              <div className="card-head">
                <span className="card-title">运行记录</span>
                <span className="card-sub">GET /evals/runs · 最近 {runs.length} 条</span>
              </div>
              <table className="doc-table">
                <thead>
                  <tr>
                    <th>run</th>
                    <th>strategy</th>
                    <th>dataset</th>
                    <th>用例</th>
                    <th>Recall@5</th>
                    <th>Faithfulness</th>
                    <th>时间</th>
                  </tr>
                </thead>
                <tbody>
                  {runs.map((run) => (
                    <tr key={run.id} style={{ cursor: "default" }}>
                      <td className="mono" style={{ fontSize: 11 }}>
                        {run.id}
                      </td>
                      <td className="eval-strategy">{strategyLabel(run.strategy)}</td>
                      <td className="mono muted">{run.dataset}</td>
                      <td className="mono">{run.n_cases}</td>
                      <td className="mono">{formatMetric("recall_at_5", run.metrics?.recall_at_5 ?? null)}</td>
                      <td className="mono">{formatMetric("faithfulness", run.metrics?.faithfulness ?? null)}</td>
                      <td className="mono muted" style={{ fontSize: 11 }}>
                        {dateTime(run.created_at)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : null}
        </div>
      </div>
    </>
  );
}