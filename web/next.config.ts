import type { NextConfig } from "next";

/**
 * 所有 /api/* 调用都代理到 FastAPI 后端，规避浏览器 CORS，
 * 也让 X-User / X-Admin-Token 这类自定义头直接透传。
 *
 * 代理实现放在 `app/api/[...path]/route.ts`（route handler），不再用 rewrites：
 * rewrites 命中后端不可达时只能由 Next 代理层吐 500 纯文本，
 * 前端拿不到"后端没起"的动作项（审计 P1-11）。
 * 注意 rewrites 的 beforeFiles 阶段在文件系统路由之前生效，两者不能并存——
 * 留着 rewrites 会让 route handler 永远不被调用。
 *
 * 后端地址 `DOCUMIND_BACKEND`（默认 http://127.0.0.1:8080）与首字节超时 180s 都在 route.ts 顶部。
 */
const nextConfig: NextConfig = {
  reactStrictMode: true,
};

export default nextConfig;