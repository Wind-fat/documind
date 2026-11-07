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
  const headers = new Headers(init.headers);
  headers.set("Accept", "application/json");
  headers.set("X-User", getCurrentUser());
  if (options.admin) headers.set("X-Admin-Token", options.adminToken ?? ADMIN_TOKEN);
  if (init.body !== undefined && !(init.body instanceof FormData)) {
    headers.set("Content-Type", "application/json");
  }

  let res: Response;
  try {
    res = await fetch(path, { ...init, headers, cache: "no-store", signal: options.signal });
  } catch (err) {
    if (err instanceof Error && err.name === "AbortError") throw err;
    throw new ApiError(0, `无法连接后端：${err instanceof Error ? err.message : String(err)}`);
  }

  if (!res.ok) throw new ApiError(res.status, await readErrorDetail(res));
  if (res.status === 204) return undefined as T;
  return (await res.json()) as T;
}

async function requestText(path: string, options: RequestOptions = {}): Promise<string> {
  const headers = new Headers();
  headers.set("Accept", "text/html");
  headers.set("X-User", getCurrentUser());
  if (options.admin) headers.set("X-Admin-Token", ADMIN_TOKEN);

  let res: Response;
  try {
    res = await fetch(path, { headers, cache: "no-store", signal: options.signal });
  } catch (err) {
    throw new ApiError(0, `无法连接后端：${err instanceof Error ? err.message : String(err)}`);
  }
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
    res = await fetch(`/api/query/stream?${params.toString()}`, {
      method: "GET",
      headers: { Accept: "text/event-stream", "X-User": getCurrentUser() },
      cache: "no-store",
      signal: input.signal,
    });
  } catch (err) {
    if (err instanceof Error && err.name === "AbortError") throw err;
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