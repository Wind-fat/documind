import type { NextConfig } from "next";

/**
 * 所有 /api/* 调用都代理到 FastAPI 后端，规避浏览器 CORS，
 * 也让 X-User / X-Admin-Token 这类自定义头直接透传。
 */
const backend = (process.env.DOCUMIND_BACKEND ?? "http://127.0.0.1:8080").replace(/\/+$/, "");

const nextConfig: NextConfig = {
  reactStrictMode: true,
  experimental: {
    // 本地 LLM 回答一次要 30s+（检索+生成串行），Next 代理默认 30s 就 socket hang up，
    // 这里放宽到 3 分钟，否则生产模式（next start）下 /api/query/* 一定 500。
    proxyTimeout: 180_000,
  },
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: `${backend}/:path*`,
      },
    ];
  },
};

export default nextConfig;