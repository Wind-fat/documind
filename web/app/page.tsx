"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import AnswerText from "@/components/AnswerText";
import EvidenceCard, { EvidenceSkeleton } from "@/components/EvidenceCard";
import PageHeader from "@/components/PageHeader";
import PageViewer from "@/components/PageViewer";
import StatsStrip from "@/components/StatsStrip";
import { EmptyState, ErrorState } from "@/components/States";
import {
  askQuestion,
  getStats,
  streamQuery,
  toApiError,
  type ApiError,
} from "@/lib/api";
import { ms, pageLabel, score, usd } from "@/lib/format";
import { useCurrentUser } from "@/lib/session";
import {
  STRATEGIES,
  type Answer,
  type Citation,
  type Hit,
  type Stats,
  type Strategy,
  type ViewerTarget,
} from "@/lib/types";

const SAMPLES = [
  "公司注册资本是多少？",
  "董事会会议多久召开一次？",
  "股东转让股权需要经过什么程序？",
  "对外担保的审批权限是怎么规定的？",
  "监事的任期是几年？",
];

type Phase = "idle" | "streaming" | "evidence" | "loading" | "done" | "error";

function hitTarget(hit: Hit): ViewerTarget {
  return {
    doc_id: hit.doc_id,
    doc_title: hit.doc_title,
    page_start: hit.page_start,
    page_end: hit.page_end,
    char_start: hit.char_start,
    char_end: hit.char_end,
    quote: hit.text,
    section_path: hit.section_path,
    chunk_id: hit.chunk_id,
  };
}

function citationTarget(citation: Citation): ViewerTarget {
  return {
    doc_id: citation.doc_id,
    doc_title: citation.doc_title,
    page_start: citation.page_start,
    page_end: citation.page_end,
    char_start: citation.char_start,
    char_end: citation.char_end,
    quote: citation.quote,
    section_path: citation.section_path,
    chunk_id: citation.chunk_id,
  };
}

function citationAsHit(citation: Citation): Hit {
  return {
    chunk_id: citation.chunk_id,
    doc_id: citation.doc_id,
    doc_title: citation.doc_title,
    text: citation.quote,
    page_start: citation.page_start,
    char_start: citation.char_start,
    page_end: citation.page_end,
    char_end: citation.char_end,
    section_path: citation.section_path,
    score: 0,
    bm25_rank: null,
    vec_rank: null,
    rrf: 0,
  };
}

export default function AskPage() {
  const user = useCurrentUser();
  const [question, setQuestion] = useState("公司注册资本是多少？");
  const [strategy, setStrategy] = useState<Strategy>("structure");
  const [streaming, setStreaming] = useState(true);
  const [phase, setPhase] = useState<Phase>("idle");
  const [answer, setAnswer] = useState<Answer | null>(null);
  const [hits, setHits] = useState<Hit[]>([]);
  const [error, setError] = useState<ApiError | null>(null);
  const [flashRef, setFlashRef] = useState<number | null>(null);
  const [viewer, setViewer] = useState<ViewerTarget | null>(null);
  const [elapsed, setElapsed] = useState(0);
  const [stats, setStats] = useState<Stats | null>(null);
  const [statsLoading, setStatsLoading] = useState(true);
  const [statsError, setStatsError] = useState<ApiError | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const rightRef = useRef<HTMLElement | null>(null);

  const busy = phase === "streaming" || phase === "evidence" || phase === "loading";

  const loadStats = useCallback(() => {
    setStatsLoading(true);
    setStatsError(null);
    getStats()
      .then((data) => {
        setStats(data);
        setStatsLoading(false);
      })
      .catch((err: unknown) => {
        setStatsError(toApiError(err));
        setStatsLoading(false);
      });
  }, []);

  useEffect(() => {
    loadStats();
  }, [loadStats, user]);

  // 身份切换后旧答案里的引用可能越权，直接清空。
  useEffect(() => {
    setAnswer(null);
    setHits([]);
    setError(null);
    setPhase("idle");
  }, [user]);

  useEffect(() => {
    if (!busy) {
      setElapsed(0);
      return;
    }
    const started = Date.now();
    const timer = window.setInterval(() => setElapsed(Date.now() - started), 100);
    return () => window.clearInterval(timer);
  }, [busy]);

  useEffect(() => {
    return () => abortRef.current?.abort();
  }, []);

  const run = useCallback(
    async (raw: string) => {
      const text = raw.trim();
      if (!text) return;
      abortRef.current?.abort();
      const controller = new AbortController();
      abortRef.current = controller;

      setAnswer(null);
      setHits([]);
      setError(null);
      setViewer(null);
      setPhase(streaming ? "streaming" : "loading");
      rightRef.current?.scrollTo({ top: 0, behavior: "smooth" });

      try {
        if (streaming) {
          const result = await streamQuery(
            { question: text, strategy, signal: controller.signal },
            {
              onEvidence: (incoming) => {
                setHits(incoming);
                setPhase("evidence");
              },
              onAnswer: (incoming) => {
                setAnswer(incoming);
                if (incoming.debug?.hits?.length) setHits(incoming.debug.hits);
                setPhase("done");
              },
            },
          );
          if (!result && !controller.signal.aborted) {
            setPhase("error");
            setError(toApiError(new Error("SSE 连接结束但未收到 answer 事件，请改用普通模式重试")));
          }
        } else {
          const result = await askQuestion({ question: text, strategy, top_k: 5 }, controller.signal);
          setAnswer(result);
          setHits(result.debug?.hits ?? []);
          setPhase("done");
        }
        loadStats();
      } catch (err) {
        if (controller.signal.aborted) return;
        setError(toApiError(err));
        setPhase("error");
      }
    },
    [loadStats, strategy, streaming],
  );

  const onCite = (ref: number): void => {
    const node = document.getElementById(`evidence-${ref}`);
    if (node) {
      node.scrollIntoView({ behavior: "smooth", block: "center" });
      setFlashRef(ref);
      window.setTimeout(() => setFlashRef((cur) => (cur === ref ? null : cur)), 2000);
    }
  };

  const citations = answer?.citations ?? [];
  const evidenceHits: Hit[] = hits.length > 0 ? hits : citations.map(citationAsHit);
  const knownRefs = new Set<number>(evidenceHits.map((_, index) => index + 1));
  const refused = answer?.refused === true;
  const currentStrategy = STRATEGIES.find((item) => item.value === strategy);
  const hasAnswer = answer !== null;

  return (
    <>
      <PageHeader
        title="问答台"
        subtitle={`${strategy} · top_k=5 · ${streaming ? "SSE" : "POST /query"}`}
        right={
          <>
            <span>Bearer · {user}</span>
            <span className={`tag ${busy ? "warn" : "ok"}`}>{busy ? "推理中" : "就绪"}</span>
          </>
        }
      />

      <div className="ask">
        <section className="ask-left">
          <form
            className="composer"
            onSubmit={(event) => {
              event.preventDefault();
              void run(question);
            }}
          >
            <div className="field">
              <label htmlFor="question">问题</label>
              <textarea
                id="question"
                className="textarea"
                value={question}
                placeholder="就公司文档提问，例如：董事会会议多久召开一次？"
                onChange={(event) => setQuestion(event.target.value)}
                onKeyDown={(event) => {
                  if (event.key === "Enter" && (event.metaKey || event.ctrlKey)) {
                    event.preventDefault();
                    void run(question);
                  }
                }}
              />
            </div>

            <div className="field">
              <label htmlFor="strategy">切块策略</label>
              <select
                id="strategy"
                className="select"
                value={strategy}
                onChange={(event) => setStrategy(event.target.value as Strategy)}
              >
                {STRATEGIES.map((item) => (
                  <option key={item.value} value={item.value}>
                    {item.label}
                  </option>
                ))}
              </select>
              <span className="card-sub">{currentStrategy?.hint ?? ""}</span>
            </div>

            <div className="composer-actions">
              <label className="switch" title="带自定义头，无法用 EventSource，改用 fetch + reader 解析 SSE">
                <input type="checkbox" checked={streaming} onChange={(event) => setStreaming(event.target.checked)} />
                SSE 流式
              </label>
              <button type="submit" className="btn btn-primary" disabled={busy || question.trim().length === 0}>
                {busy ? (
                  <>
                    <span className="spinner" /> {phase === "streaming" ? "检索中" : "生成中"}
                  </>
                ) : (
                  "提问 ⌘↵"
                )}
              </button>
            </div>

            {busy ? (
              <div className="stream-hint">
                <span className="spinner" />
                {phase === "streaming"
                  ? "已建立 SSE 连接，等待 evidence 事件…"
                  : phase === "evidence"
                    ? `已收到 ${hits.length} 条证据，等待 answer 事件…`
                    : "请求 /query 中…"}
                <span className="mono muted">{elapsed > 0 ? `${(elapsed / 1000).toFixed(1)}s` : ""}</span>
              </div>
            ) : null}
          </form>

          <div>
            <div className="section-title">
              <span>示例问题</span>
            </div>
            <div className="samples">
              {SAMPLES.map((sample) => (
                <button
                  key={sample}
                  type="button"
                  className="sample"
                  onClick={() => {
                    setQuestion(sample);
                    void run(sample);
                  }}
                >
                  {sample}
                </button>
              ))}
            </div>
          </div>

          <div>
            <div className="section-title">
              <span>请求上下文</span>
            </div>
            <div style={{ display: "grid", gap: 6, fontSize: 11.5 }}>
              <div style={{ display: "flex", justifyContent: "space-between" }}>
                <span className="muted">identity</span>
                <span className="mono">{user}</span>
              </div>
              <div style={{ display: "flex", justifyContent: "space-between" }}>
                <span className="muted">strategy</span>
                <span className="mono">{strategy}</span>
              </div>
              <div style={{ display: "flex", justifyContent: "space-between" }}>
                <span className="muted">transport</span>
                <span className="mono">{streaming ? "GET /query/stream" : "POST /query"}</span>
              </div>
              <div style={{ display: "flex", justifyContent: "space-between" }}>
                <span className="muted">evidence</span>
                <span className="mono">{evidenceHits.length} 条</span>
              </div>
            </div>
          </div>
        </section>

        <section className="ask-right" ref={rightRef}>
          <div className="answer-wrap">
            {error ? (
              <ErrorState
                title={
                  error.status === 0
                    ? "后端不可达"
                    : error.status === 401
                      ? "身份无效（401）"
                      : "请求失败"
                }
                message={error.detail}
                status={error.status}
                onRetry={() => void run(question)}
              />
            ) : null}

            {!hasAnswer && !busy && !error ? (
              <div className="card">
                <div className="answer-head">
                  <span className="card-title">答案与证据</span>
                  <span className="card-sub">每个结论都可回溯到页级原文</span>
                </div>
                <EmptyState
                  icon="◈"
                  title="输入问题开始检索"
                  desc="左侧选择身份与切块策略后提问。答案中的 [n] 可点击，会滚动定位到对应证据卡；点击证据卡则打开原文弹层，按 char 偏移高亮原句。"
                />
              </div>
            ) : null}

            {busy && !hasAnswer ? (
              <div className="card" aria-busy="true">
                <div className="answer-head">
                  <span className="stream-hint">
                    <span className="spinner" /> 正在检索并生成答案…
                  </span>
                  <span className="card-sub">
                    {phase === "streaming" ? "等待 evidence" : phase === "evidence" ? "evidence 已到，等待 answer" : "POST /query"}
                  </span>
                </div>
                <div style={{ padding: "18px 20px", display: "flex", flexDirection: "column", gap: 9 }}>
                  <div className="skeleton" style={{ height: 13, width: "88%" }} />
                  <div className="skeleton" style={{ height: 13, width: "74%" }} />
                  <div className="skeleton" style={{ height: 13, width: "52%" }} />
                </div>
              </div>
            ) : null}

            {hasAnswer && refused ? (
              <div className="refused-card" role="alert">
                <div className="refused-icon">✕</div>
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div className="refused-title">文档中未找到相关条款</div>
                  <div className="refused-reason">
                    模型判定当前资料不足以回答该问题，已按拒答处理，不生成推测性结论。
                    <br />
                    原因：<code>{answer.reason || "未提供原因"}</code>
                  </div>
                  <div style={{ marginTop: 10, display: "flex", gap: 6, flexWrap: "wrap" }}>
                    <span className="tag warn">refused</span>
                    <span className="tag">strategy {String(answer.debug?.strategy ?? strategy)}</span>
                    <span className="tag">latency {ms(answer.latency_ms)}</span>
                    <span className="tag">
                      候选 {typeof answer.debug?.candidates === "number" ? answer.debug.candidates : "—"} · top_score{" "}
                      {score(answer.top_score)}
                    </span>
                  </div>
                  <div className="card-sub" style={{ marginTop: 8 }}>
                    可以换个问法、切换身份（ACL 可能过滤了文档）或调整切块策略后重试。
                  </div>
                </div>
              </div>
            ) : null}

            {hasAnswer && !refused ? (
              <article className="answer-card">
                <div className="answer-head">
                  <span className="card-title">
                    答案
                    <span className="tag accent">{(answer.citations ?? []).length} 引用</span>
                    {answer.selfcheck?.supported === true ? <span className="tag ok">自检通过</span> : null}
                    {answer.selfcheck?.supported === false ? <span className="tag warn">自检未通过</span> : null}
                  </span>
                  <span className="answer-metric">
                    总 <b>{ms(answer.latency_ms)}</b>
                  </span>
                  <span className="answer-metric">
                    检索 <b>{ms(answer.retrieval_ms)}</b>
                  </span>
                  <span className="answer-metric">
                    生成 <b>{ms(answer.generation_ms)}</b>
                  </span>
                  <span className="answer-metric">
                    成本 <b>{usd(answer.cost_usd)}</b>
                  </span>
                  <span className="answer-metric">
                    token <b>{answer.prompt_tokens}↑ {answer.completion_tokens}↓</b>
                  </span>
                  <span className="answer-metric">
                    top <b>{score(answer.top_score)}</b>
                  </span>
                </div>
                <div className="answer-body">
                  <AnswerText text={answer.text} knownRefs={knownRefs} onCite={onCite} />
                </div>
                {answer.rewritten_query ? (
                  <div className="card-sub" style={{ padding: "0 20px 16px" }}>
                    改写查询：<span className="mono">{answer.rewritten_query}</span>
                  </div>
                ) : null}
              </article>
            ) : null}

            <div className="evidence-section">
              {busy || hasAnswer ? (
                <>
                  <div className="section-title">
                    <span>
                      {refused ? "候选片段（未达回答阈值）" : "证据 / evidence"} · {evidenceHits.length}
                    </span>
                    <span style={{ textTransform: "none", letterSpacing: 0, fontWeight: 400 }}>
                      点击卡片查看原文并定位高亮
                    </span>
                  </div>

                  {phase === "streaming" && evidenceHits.length === 0 ? (
                    <div className="evidence-list">
                      <EvidenceSkeleton />
                      <EvidenceSkeleton />
                      <EvidenceSkeleton />
                    </div>
                  ) : null}

                  {evidenceHits.length > 0 ? (
                    <div className="evidence-list">
                      {evidenceHits.map((hit, index) => (
                        <EvidenceCard
                          key={`${hit.chunk_id}-${index}`}
                          hit={hit}
                          refNo={index + 1}
                          flash={flashRef === index + 1}
                          onOpen={(target) => setViewer(hitTarget(target))}
                        />
                      ))}
                    </div>
                  ) : null}

                  {!busy && evidenceHits.length === 0 ? (
                    <EmptyState icon="▤" title="暂无证据" desc="提问后这里会展示命中片段、页码与 BM25/向量名次、重排分。" />
                  ) : null}
                </>
              ) : null}
            </div>

            {hasAnswer ? (
              <div className="card-sub" style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
                <span className="tag">doc {evidenceHits[0]?.doc_id ?? "—"}</span>
                <span className="tag">contexts {String(answer.debug?.contexts ?? evidenceHits.length)}</span>
                <span className="tag">candidates {String(answer.debug?.candidates ?? "—")}</span>
                <span className="tag">intent {String(answer.debug?.intent ?? "—")}</span>
                <span className="tag">first page {evidenceHits[0] ? pageLabel(evidenceHits[0].page_start, evidenceHits[0].page_end) : "—"}</span>
              </div>
            ) : null}

            <StatsStrip stats={stats} loading={statsLoading} error={statsError} onRetry={loadStats} />
          </div>
        </section>
      </div>

      {viewer ? (
        <PageViewer
          key={`${viewer.doc_id}-${viewer.page_start}-${viewer.char_start}-${viewer.page_end}`}
          target={viewer}
          onClose={() => setViewer(null)}
        />
      ) : null}
    </>
  );
}