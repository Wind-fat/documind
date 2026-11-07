import { getCurrentUser } from "./session";
import type {
  Answer,
  DocumentMeta,
  EvalRun,
  Health,
  Hit,
  Outline,
  PageText,
  QueryInput,
  Stats,
  Strategy,
  UploadResult,
} from "./types";

/** 后端写操作固定 dev token（docs/API.md）。不落任何密钥。 */
const ADMIN_TOKEN = "dev-admin-token";

export class ApiError extends Error {
  readonly status: number;
  readonly detail: string;

  constructor(status: number, detail: string) {
    super(detail);
    this.name = "ApiError";
    this.status = status;
    this.detail = detail;
  }
}

export function toApiError(err: unknown): ApiError {
  if (err instanceof ApiError) return err;
  if (err instanceof Error) {
    if (err.name === "AbortError") return new ApiError(-1, "请求已取消");
    return new ApiError(0, err.message);
  }
  return new ApiError(0, "未知错误");
}

/* -------------------------------- 身份令牌 -------------------------------- */

/**
 * 后端只认签名令牌（auth.py / P0-01）：GET 与 POST 一律带
 * `Authorization: Bearer <token>`，裸 `X-User` 仅开发环境兜底、生产直接 401。
 * 所以这里先用 POST /auth/login 换 token，按用户缓存，过期前 30s 自动重签。
 */
interface TokenEntry {
  user: string;
  token: string;
  exp: number;
}

const TOKEN_PREFIX = "documind.token.";

let cachedToken: TokenEntry | null = null;
let loginInFlight: { user: string; promise: Promise<string> } | null = null;

function tokenExpiry(token: string): number {
  const parts = token.split(":");
  if (parts.length !== 4) return 0;
  const exp = Number(parts[2]);
  return Number.isFinite(exp) ? exp : 0;
}

function tokenValid(entry: TokenEntry | null): entry is TokenEntry {
  return entry !== null && entry.exp - 30 > Math.floor(Date.now() / 1000);
}

function loadStoredToken(user: string): TokenEntry | null {
  if (typeof window === "undefined") return null;
  try {
    const raw = window.sessionStorage.getItem(TOKEN_PREFIX + user);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as Partial<TokenEntry>;
    if (typeof parsed.token !== "string") return null;
    return { user, token: parsed.token, exp: tokenExpiry(parsed.token) };
  } catch {
    return null;
  }
}

function storeToken(entry: TokenEntry): void {
  if (typeof window === "undefined") return;
  try {
    window.sessionStorage.setItem(TOKEN_PREFIX + entry.user, JSON.stringify(entry));
  } catch {
    /* 隐私模式下 sessionStorage 可能不可用，退化为内存缓存 */
  }
}

/** 401 后作废缓存，下一次请求会重新登录。 */
export function forgetToken(user?: string): void {
  if (user) {
    cachedToken = cachedToken?.user === user ? null : cachedToken;
    try {
      window.sessionStorage.removeItem(TOKEN_PREFIX + user);
    } catch {
      /* ignore */
    }
    return;
  }
  if (cachedToken) {
    try {
      window.sessionStorage.removeItem(TOKEN_PREFIX + cachedToken.user);
    } catch {
      /* ignore */
    }
  }
  cachedToken = null;
}

async function login(user: string): Promise<string> {
  if (loginInFlight && loginInFlight.user === user) return loginInFlight.promise;

  const promise = (async (): Promise<string> => {
    let res: Response;
    try {
      res = await fetch("/api/auth/login", {
        method: "POST",
        headers: { "Content-Type": "application/json", Accept: "application/json" },
        body: JSON.stringify({ name: user }),
        cache: "no-store",
      });
    } catch (err) {
      throw new ApiError(0, `无法连接后端：${err instanceof Error ? err.message : String(err)}`);
    }
    if (!res.ok) {
      throw new ApiError(res.status, `登录失败（${user}）：${await readErrorDetail(res)}`);
    }
    const data = (await res.json()) as { token?: string };
    if (typeof data.token !== "string" || !data.token) {
      throw new ApiError(0, "登录响应缺少 token");
    }
    const entry: TokenEntry = { user, token: data.token, exp: tokenExpiry(data.token) };
    cachedToken = entry;
    storeToken(entry);
    return entry.token;
  })();

  loginInFlight = { user, promise };
  try {
    return await promise;
  } finally {
    if (loginInFlight?.promise === promise) loginInFlight = null;
  }
}

/** 取当前身份可用 token：内存 → sessionStorage → 重新登录。 */
export async function authToken(user: string): Promise<string> {
  if (tokenValid(cachedToken) && cachedToken.user === user) return cachedToken.token;
  const stored = loadStoredToken(user);
  if (tokenValid(stored)) {
    cachedToken = stored;
    return stored.token;
  }
  return login(user);
}

function buildHeaders(init: RequestInit, options: RequestOptions, token: string, user: string): Headers {
  const headers = new Headers(init.headers);
  headers.set("Accept", "application/json");
  headers.set("Authorization", `Bearer ${token}`);
  headers.set("X-User", user); // 仅开发环境兜底；生产以 Authorization 为准
  if (options.admin) headers.set("X-Admin-Token", options.adminToken ?? ADMIN_TOKEN);
  if (init.body !== undefined && !(init.body instanceof FormData)) {
    headers.set("Content-Type", "application/json");
  }
  return headers;
}

/** 统一带令牌请求；401 换一次新 token 再试，仍失败才抛给调用方。 */
async function authFetch(
  path: string,
  init: RequestInit = {},
  options: RequestOptions = {},
  accept = "application/json",
): Promise<Response> {
  const user = getCurrentUser();

  const attempt = async (token: string): Promise<Response> => {
    const headers = buildHeaders(init, options, token, user);
    headers.set("Accept", accept);
    try {
      return await fetch(path, { ...init, headers, cache: "no-store", signal: options.signal });
    } catch (err) {
      if (err instanceof Error && err.name === "AbortError") throw err;
      throw new ApiError(0, `无法连接后端：${err instanceof Error ? err.message : String(err)}`);
    }
  };

  let res = await attempt(await authToken(user));
  if (res.status === 401) {
    forgetToken(user);
    res = await attempt(await authToken(user));
  }
  return res;
}

async function readErrorDetail(res: Response): Promise<string> {
  const fallback = `HTTP ${res.status} ${res.statusText}`.trim();
  try {
    const text = await res.text();
    if (!text) return fallback;
    try {
      const parsed: unknown = JSON.parse(text);
      if (parsed && typeof parsed === "object") {
        const obj = parsed as Record<string, unknown>;
        const detail = obj.detail ?? obj.message ?? obj.error;
        if (typeof detail === "string" && detail) return detail;
        if (detail !== undefined) return JSON.stringify(detail);
      }
    } catch {
      /* 非 JSON 错误体，原样截断 */
    }
    return text.slice(0, 400);
  } catch {
    return fallback;
  }
}

interface RequestOptions {
  admin?: boolean;
  adminToken?: string;
  signal?: AbortSignal;
}

async function request<T>(path: string, init: RequestInit = {}, options: RequestOptions = {}): Promise<T> {
  const res = await authFetch(path, init, options);

  if (!res.ok) throw new ApiError(res.status, await readErrorDetail(res));
  if (res.status === 204) return undefined as T;
  return (await res.json()) as T;
}

async function requestText(path: string, options: RequestOptions = {}): Promise<string> {
  const res = await authFetch(path, {}, options, "text/html");

  if (!res.ok) throw new ApiError(res.status, await readErrorDetail(res));
  return res.text();
}

/* ---------------------------------- 读取 ---------------------------------- */

export function getHealth(): Promise<Health> {
  return request<Health>("/api/health");
}

export function listDocuments(): Promise<DocumentMeta[]> {
  return request<DocumentMeta[]>("/api/documents");
}

export function getPageText(docId: string, page: number, signal?: AbortSignal): Promise<PageText> {
  return request<PageText>(
    `/api/documents/${encodeURIComponent(docId)}/pages/${encodeURIComponent(String(page))}`,
    {},
    { signal },
  );
}

export function getOutline(docId: string, signal?: AbortSignal): Promise<Outline> {
  return request<Outline>(`/api/documents/${encodeURIComponent(docId)}/outline`, {}, { signal });
}

export function getStats(signal?: AbortSignal): Promise<Stats> {
  return request<Stats>("/api/stats", {}, { signal });
}

export function listEvalRuns(signal?: AbortSignal): Promise<EvalRun[]> {
  return request<EvalRun[]>("/api/evals/runs", {}, { signal });
}

export function getEvalReportHtml(signal?: AbortSignal): Promise<string> {
  return requestText("/api/evals/report", { signal });
}

/* ---------------------------------- 问答 ---------------------------------- */

export function askQuestion(input: QueryInput, signal?: AbortSignal): Promise<Answer> {
  const body: QueryInput = {
    question: input.question,
    strategy: input.strategy,
    conversation_id: input.conversation_id ?? null,
    top_k: input.top_k ?? 5,
    selfcheck: input.selfcheck ?? false,
  };
  return request<Answer>(
    "/api/query",
    { method: "POST", body: JSON.stringify(body) },
    { signal },
  );
}

export interface StreamHandlers {
  /** 检索完成：先渲染证据卡（后端会在生成答案之前发出）。 */
  onEvidence?: (hits: Hit[]) => void;
  /** 答案到达。 */
  onAnswer?: (answer: Answer) => void;
  /** 其它/未知事件，便于调试。 */
  onEvent?: (event: string) => void;
}

export interface StreamInput {
  question: string;
  strategy: Strategy;
  conversationId?: string | null;
  signal?: AbortSignal;
}

interface SseEvent {
  event: string;
  data: string;
}

function parseEventBlock(block: string): SseEvent | null {
  let event = "message";
  const dataLines: string[] = [];
  for (const rawLine of block.split("\n")) {
    const line = rawLine.replace(/\r$/, "");
    if (!line || line.startsWith(":")) continue;
    const colon = line.indexOf(":");
    const field = colon === -1 ? line : line.slice(0, colon);
    let value = colon === -1 ? "" : line.slice(colon + 1);
    if (value.startsWith(" ")) value = value.slice(1);
    if (field === "event") event = value;
    else if (field === "data") dataLines.push(value);
  }
  if (dataLines.length === 0) return null;
  return { event, data: dataLines.join("\n") };
}

function isHitArray(value: unknown): value is Hit[] {
  return Array.isArray(value);
}

/**
 * SSE 必须用 fetch + reader：EventSource 无法携带 X-User 自定义头。
 * 逐块解码，按空行切事件，逐行解析 event: / data:。
 */
export async function streamQuery(input: StreamInput, handlers: StreamHandlers): Promise<Answer | null> {
  const params = new URLSearchParams({ question: input.question, strategy: input.strategy });
  if (input.conversationId) params.set("conversation_id", input.conversationId);

  let res: Response;
  try {
    const token = await authToken(getCurrentUser());
    res = await fetch(`/api/query/stream?${params.toString()}`, {
      method: "GET",
      headers: { Accept: "text/event-stream", Authorization: `Bearer ${token}`, "X-User": getCurrentUser() },
      cache: "no-store",
      signal: input.signal,
    });
    if (res.status === 401) {
      // 令牌过期：重签一次再开流，别把 401 误报成后端挂了
      forgetToken(getCurrentUser());
      const retryToken = await authToken(getCurrentUser());
      res = await fetch(`/api/query/stream?${params.toString()}`, {
        method: "GET",
        headers: { Accept: "text/event-stream", Authorization: `Bearer ${retryToken}`, "X-User": getCurrentUser() },
        cache: "no-store",
        signal: input.signal,
      });
    }
  } catch (err) {
    if (err instanceof Error && err.name === "AbortError") throw err;
    if (err instanceof ApiError) throw err;
    throw new ApiError(0, `SSE 连接失败：${err instanceof Error ? err.message : String(err)}`);
  }

  if (!res.ok) throw new ApiError(res.status, await readErrorDetail(res));
  if (!res.body) throw new ApiError(0, "浏览器不支持流式响应（response.body 为空）");

  const reader = res.body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";
  let answer: Answer | null = null;

  const dispatch = (block: string): void => {
    const evt = parseEventBlock(block);
    if (!evt) return;
    let parsed: unknown;
    try {
      parsed = JSON.parse(evt.data);
    } catch {
      return;
    }
    if (evt.event === "evidence") {
      if (isHitArray(parsed)) handlers.onEvidence?.(parsed);
      return;
    }
    if (evt.event === "answer") {
      answer = parsed as Answer;
      handlers.onAnswer?.(answer);
      return;
    }
    handlers.onEvent?.(evt.event);
  };

  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      buffer = buffer.replace(/\r\n/g, "\n");
      for (;;) {
        const cut = buffer.indexOf("\n\n");
        if (cut === -1) break;
        const block = buffer.slice(0, cut);
        buffer = buffer.slice(cut + 2);
        if (block.trim()) dispatch(block);
      }
    }
    buffer += decoder.decode();
    buffer = buffer.replace(/\r\n/g, "\n");
    if (buffer.trim()) dispatch(buffer);
  } finally {
    reader.releaseLock();
  }

  return answer;
}

/* ---------------------------------- 写入 ---------------------------------- */

export function uploadDocument(
  file: File,
  aclRoles: string[],
  title?: string,
  adminToken?: string,
): Promise<UploadResult> {
  const form = new FormData();
  form.append("file", file, file.name);
  form.append("acl_roles", JSON.stringify(aclRoles));
  if (title && title.trim()) form.append("title", title.trim());
  return request<UploadResult>(
    "/api/documents",
    { method: "POST", body: form },
    { admin: true, adminToken: adminToken?.trim() || undefined },
  );
}