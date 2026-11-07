import type { ReactNode } from "react";

export default function PageHeader({
  title,
  subtitle,
  right,
}: {
  title: string;
  subtitle?: string;
  right?: ReactNode;
}) {
  return (
    <header className="topbar">
      <div style={{ display: "flex", alignItems: "baseline", gap: 10, minWidth: 0 }}>
        <span className="topbar-title">{title}</span>
        {subtitle ? <span className="card-sub" style={{ fontFamily: "var(--mono)" }}>{subtitle}</span> : null}
      </div>
      <div className="topbar-meta">{right}</div>
    </header>
  );
}