"use client";

import type { ReactNode } from "react";

export function LoadingState({ label = "加载中…", rows = 3 }: { label?: string; rows?: number }) {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 10, padding: 16 }} aria-busy="true" aria-live="polite">
      <span className="stream-hint">
        <span className="spinner" />
        {label}
      </span>
      {Array.from({ length: rows }).map((_, i) => (
        <div key={i} className="skeleton" style={{ height: 44, opacity: 1 - i * 0.18 }} />
      ))}
    </div>
  );
}

export function EmptyState({
  icon = "◌",
  title,
  desc,
  action,
}: {
  icon?: string;
  title: string;
  desc?: string;
  action?: ReactNode;
}) {
  return (
    <div className="state">
      <div className="state-icon">{icon}</div>
      <div className="state-title">{title}</div>
      {desc ? <div className="state-desc">{desc}</div> : null}
      {action}
    </div>
  );
}

export function ErrorState({
  title = "请求失败",
  message,
  status,
  onRetry,
}: {
  title?: string;
  message: string;
  status?: number;
  onRetry?: () => void;
}) {
  return (
    <div className="banner error" role="alert" style={{ alignItems: "center" }}>
      <div style={{ flex: 1 }}>
        <div className="banner-title">
          {title}
          {status !== undefined && status !== 0 ? <span className="tag danger" style={{ marginLeft: 8 }}>HTTP {status}</span> : null}
        </div>
        <div>{message}</div>
      </div>
      {onRetry ? (
        <button type="button" className="btn btn-sm" onClick={onRetry}>
          重试
        </button>
      ) : null}
    </div>
  );
}