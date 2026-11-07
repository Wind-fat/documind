import type { NextRequest } from "next/server";

/**
 * `/api/*` 统一代理到 FastAPI 后端（替代 next.config 里的 rewrites，审计 P1-11）。
 *
 * 为什么不用 rewrites：后端没起时 Next 的代理层直接吐 `500` + `text/plain`
 * "Internal Server Error"，前端 `readErrorDetail` 只能原样展示，用户看不到任何动作项。
 * 换成 route handler 后，`ECONNREFUSED` 能被翻译成 `502` + 一条可直接粘进终端的启动命令。
 *
 * 三条约束：
 * 1. 不做重试、不吞状态码——后端返回什么就透传什么（502 只用于"连不上后端"）；
 * 2. 响应体按流透传（SSE `/query/stream`、上传的 FormData 都不能被缓冲）；
 * 3. 不转发 content-encoding / content-length：undici 会自动解压，头留着会让浏览器二次解压失败。
 */

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const BACKEND = (process.env.DOCUMIND_BACKEND ?? "http://127.0.0.1:8080").replace(/\/+$/, "");

/** 首字节超时：本地 LLM 一次回答 30s+，留到 180s（与旧 rewrites 的 proxyTimeout 对齐）。 */
const HEADERS_TIMEOUT_MS = 180_000;

const START_HINT =
  "cd D:/code/documind && ./.venv/Scripts/python.exe -m uvicorn documind.app:app --host 127.0.0.1 --port 8080";

/** 逐跳头不能代理；content-length/content-encoding 由 undici 重新决定。 */
const DROP_REQUEST_HEADERS = new Set(["connection", "keep-alive", "proxy-authorization", "te", "trailer", "transfer-encoding", "upgrade", "host"]);
const DROP_RESPONSE_HEADERS = new Set(["connection", "keep-alive", "transfer-encoding", "content-encoding", "content-length"]);

type UpstreamFailure = { status: number; detail: string; code: string };

function describeFailure(err: unknown): UpstreamFailure {
  const cause = (err as { cause?: { code?: string; message?: string } } | undefined)?.cause;
  const code = cause?.code ?? "";
  const name = (err as { name?: string } | undefined)?.name ?? "";
  const message = cause?.message ?? (err instanceof Error ? err.message : String(err));

  if (name === "TimeoutError" || code === "ETIMEDOUT" || code === "UND_ERR_HEADERS_TIMEOUT") {
    return {
      status: 504,
      code: "upstream_timeout",
      detail: `后端 ${BACKEND} 在 ${HEADERS_TIMEOUT_MS / 1000}s 内没有返回响应。长任务可稍后重试；否则看 uvicorn 日志（命令：${START_HINT}）。`,
    };
  }
  if (code === "ECONNREFUSED") {
    return {
      status: 502,
      code: "upstream_unreachable",
      detail: `后端未启动：${BACKEND} 拒绝连接（ECONNREFUSED）。在另一个终端执行：${START_HINT}`,
    };
  }
  if (code === "ENOTFOUND" || code === "EAI_AGAIN") {
    return {
      status: 502,
      code: "upstream_dns",
      detail: `后端地址无法解析：${BACKEND}。检查 web/.env.local 里的 DOCUMIND_BACKEND 是否写错。`,
    };
  }
  if (code === "ECONNRESET" || code === "UND_ERR_SOCKET") {
    return {
      status: 502,
      code: "upstream_reset",
      detail: `后端 ${BACKEND} 在响应中途断开连接（ECONNRESET）。多半是 uvicorn 崩了或超时被杀，看它那个终端的最后 20 行。`,
    };
  }
  return {
    status: 502,
    code: "upstream_error",
    detail: `连不上后端 ${BACKEND}：${message}。确认它活着：${START_HINT}`,
  };
}

function failureResponse(failure: UpstreamFailure): Response {
  return Response.json(
    { detail: failure.detail, code: failure.code, backend: BACKEND },
    { status: failure.status, headers: { "cache-control": "no-store", "x-documind-proxy": "upstream-down" } },
  );
}

async function proxy(req: NextRequest): Promise<Response> {
  // nextUrl.pathname 已经是百分号编码的，直接拼，别再 encodeURIComponent 一次（会二次编码 %）
  const suffix = req.nextUrl.pathname.replace(/^\/api\/?/, "");
  const target = `${BACKEND}/${suffix}${req.nextUrl.search}`;

  const headers = new Headers();
  req.headers.forEach((value, key) => {
    if (!DROP_REQUEST_HEADERS.has(key.toLowerCase())) headers.set(key, value);
  });

  const controller = new AbortController();
  const abortUpstream = (): void => controller.abort();
  req.signal.addEventListener("abort", abortUpstream);

  // 只对"等首字节"计时；流式响应开始后不再计时（否则长答案会被截断）
  const timer = setTimeout(() => controller.abort(new DOMException("upstream headers timeout", "TimeoutError")), HEADERS_TIMEOUT_MS);

  const hasBody = req.method !== "GET" && req.method !== "HEAD" && req.body !== null;

  let upstream: Response;
  try {
    upstream = await fetch(target, {
      method: req.method,
      headers,
      body: hasBody ? req.body : undefined,
      // Node 的 fetch 读 ReadableStream 必须显式声明半双工
      duplex: "half",
      redirect: "manual",
      cache: "no-store",
      signal: controller.signal,
    } as RequestInit & { duplex: "half" });
  } catch (err) {
    clearTimeout(timer);
    req.signal.removeEventListener("abort", abortUpstream);
    if (req.signal.aborted) {
      // 浏览器主动取消（切换提问 / 关掉弹层）：不是后端故障，别报 502
      return new Response(null, { status: 499 });
    }
    return failureResponse(describeFailure(err));
  } finally {
    clearTimeout(timer);
  }

  const outHeaders = new Headers();
  upstream.headers.forEach((value, key) => {
    if (!DROP_RESPONSE_HEADERS.has(key.toLowerCase())) outHeaders.set(key, value);
  });
  if (!outHeaders.has("cache-control")) outHeaders.set("cache-control", "no-store");
  outHeaders.set("x-documind-proxy", "ok");

  return new Response(upstream.body, { status: upstream.status, statusText: upstream.statusText, headers: outHeaders });
}

export const GET = proxy;
export const POST = proxy;
export const PUT = proxy;
export const PATCH = proxy;
export const DELETE = proxy;
export const HEAD = proxy;
export const OPTIONS = proxy;