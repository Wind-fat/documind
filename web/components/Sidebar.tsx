"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";
import { getHealth } from "@/lib/api";
import { setCurrentUser, useCurrentUser } from "@/lib/session";
import { USERS } from "@/lib/types";

const NAV = [
  { href: "/", label: "问答台", icon: "◈" },
  { href: "/documents", label: "文档库", icon: "▤" },
  { href: "/evals", label: "评测看板", icon: "◱" },
  { href: "/stats", label: "运行指标", icon: "◎" },
];

export default function Sidebar() {
  const pathname = usePathname();
  const user = useCurrentUser();
  const [online, setOnline] = useState<boolean | null>(null);
  const [chunks, setChunks] = useState<number | null>(null);

  useEffect(() => {
    let alive = true;
    const probe = (): void => {
      getHealth()
        .then((h) => {
          if (!alive) return;
          setOnline(true);
          setChunks(h.chunks);
        })
        .catch(() => {
          if (alive) setOnline(false);
        });
    };
    probe();
    const timer = window.setInterval(probe, 45000);
    return () => {
      alive = false;
      window.clearInterval(timer);
    };
  }, []);

  const activeRole = USERS.find((u) => u.name === user)?.role ?? "employee";

  return (
    <aside className="sidebar">
      <div className="brand">
        <div className="brand-mark">D</div>
        <div>
          <div className="brand-name">DocuMind</div>
          <div className="brand-sub">retrieval console</div>
        </div>
      </div>

      <nav className="nav">
        <div className="nav-label">工作区</div>
        {NAV.map((item) => {
          const active = item.href === "/" ? pathname === "/" : pathname.startsWith(item.href);
          return (
            <Link key={item.href} href={item.href} className={`nav-item${active ? " active" : ""}`}>
              <span className="nav-icon">{item.icon}</span>
              {item.label}
            </Link>
          );
        })}
      </nav>

      <div className="sidebar-foot">
        <div className="identity">
          <div className="identity-head">
            <span>当前身份</span>
            <span className="identity-role">{activeRole}</span>
          </div>
          <div className="seg">
            {USERS.map((u) => (
              <button
                key={u.name}
                type="button"
                className={u.name === user ? "on" : ""}
                onClick={() => setCurrentUser(u.name)}
                title={`以 ${u.name}（${u.role}）身份请求，影响 ACL 与可见文档`}
              >
                {u.name}
              </button>
            ))}
          </div>
        </div>
        <div className="health-pill">
          <span className={`dot ${online === null ? "" : online ? "ok" : "bad"}`} />
          {online === null ? "连接后端…" : online ? "后端在线" : "后端离线"}
          {online && chunks !== null ? <span className="mono">· {chunks} chunks</span> : null}
        </div>
      </div>
    </aside>
  );
}