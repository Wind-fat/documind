import type { NextConfig } from "next";

/**
 * 所有 /api/* 调用都代理到 FastAPI 后端，规避浏览器 CORS，
 * 也让 X-User / X-Admin-Token 这类自定义头直接透传。
 */
const backend = (process.env.DOCUMIND_BACKEND ?? "http://127.0.0.1:8080").replace(/\/+$/, "");

const nextConfig: NextConfig = {
  reactStrictMode: true,
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