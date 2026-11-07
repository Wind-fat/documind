"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { getPageText, toApiError, type ApiError } from "@/lib/api";
import { pageLabel } from "@/lib/format";
import type { PageText, ViewerTarget } from "@/lib/types";
import { LoadingState } from "./States";

interface ViewSlice {
  before: string;
  mark: string;
  after: string;
  note: string;
}

export default function PageViewer({ target, onClose }: { target: ViewerTarget; onClose: () => void }) {
  const [page, setPage] = useState<number>(target.page_start);
  const [pageText, setPageText] = useState<PageText | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<ApiError | null>(null);
  const markRef = useRef<HTMLElement | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    setLoading(true);
    setError(null);
    getPageText(target.doc_id, page, controller.signal)
      .then((data) => {
        if (controller.signal.aborted) return;
        setPageText(data);
        setLoading(false);
      })
      .catch((err: unknown) => {
        if (controller.signal.aborted) return;
        setError(toApiError(err));
        setPageText(null);
        setLoading(false);
      });
    return () => controller.abort();
  }, [target.doc_id, page]);

  useEffect(() => {
    const onKey = (event: KeyboardEvent): void => {
      if (event.key === "Escape") onClose();
    };
    document.addEventListener("keydown", onKey);
    const previous = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      document.removeEventListener("keydown", onKey);
      document.body.style.overflow = previous;
    };
  }, [onClose]);

  /** 字符偏移的精确高亮：多页 span 只在首页高亮到页尾，其余页靠翻页阅读。 */
  const view = useMemo<ViewSlice>(() => {
    const text = pageText?.text ?? "";
    if (!text) return { before: "", mark: "", after: "", note: "" };
    const multiPage = target.page_end > target.page_start;
    const hasSpan = target.char_end > target.char_start;
    if (!hasSpan || page !== target.page_start) {
      return { before: text, mark: "", after: "", note: "" };
    }
    const length = text.length;
    let start = Math.max(0, Math.min(target.char_start, length));
    let end = multiPage ? length : Math.max(start, Math.min(target.char_end, length));
    let note = "";
    if (end <= start) {
      const quote = target.quote?.trim() ?? "";
      const found = quote ? text.indexOf(quote) : -1;
      if (found >= 0) {
        start = found;
        end = found + quote.length;
      }
    }
    if (end <= start) {
      return {
        before: text,
        mark: "",
        after: "",
        note: `未能按 char 偏移 ${target.char_start}–${target.char_end} 定位（超出本页文本长度 ${length}），已展示整页。`,
      };
    }
    if (!multiPage && target.char_end > length) {
      note = `引用偏移 ${target.char_end} 超出本页文本长度 ${length}，高亮已截断到页尾。`;
    }
    return {
      before: text.slice(0, start),
      mark: text.slice(start, end),
      after: text.slice(end),
      note,
    };
  }, [pageText, page, target]);

  useEffect(() => {
    const node = markRef.current;
    if (!node || !pageText) return;
    const timer = window.setTimeout(() => {
      node.scrollIntoView({ block: "center", behavior: "smooth" });
    }, 80);
    return () => window.clearTimeout(timer);
  }, [pageText, page, view.mark]);

  const multiPage = target.page_end > target.page_start;
  const canPrev = page > 1;
  const canNext = page < Math.max(target.page_end, page);

  return (
    <div
      className="modal-backdrop"
      role="presentation"
      onMouseDown={(event) => {
        if (event.target === event.currentTarget) onClose();
      }}
    >
      <div className="modal" role="dialog" aria-modal="true" aria-label="原文定位">
        <div className="modal-head">
          <div style={{ flex: 1, minWidth: 0 }}>
            <div className="modal-title">
              《{target.doc_title}》
              <span className={`tag${multiPage ? " warn" : ""}`}>{pageLabel(target.page_start, target.page_end)}</span>
              {target.section_path ? <span className="tag accent">{target.section_path}</span> : null}
            </div>
            <div className="modal-sub">
              doc {target.doc_id} · chars {target.char_start}–{target.char_end}
              {target.chunk_id ? ` · chunk ${target.chunk_id}` : ""}
            </div>
          </div>
          <button type="button" className="btn btn-sm" onClick={onClose} aria-label="关闭">
            Esc ✕
          </button>
        </div>

        <div className="modal-body">
          {multiPage ? (
            <div className="span-note">
              <span>⚠</span>
              <span>
                该引用跨第 {target.page_start}–{target.page_end} 页，共 {target.page_end - target.page_start + 1} 页。
                当前显示第 {page} 页：
                {page === target.page_start
                  ? "高亮为引用在首页的起始片段（至页尾），后续页请翻页查看。"
                  : "本页为引用续页，不重复高亮。"}
              </span>
            </div>
          ) : null}

          <div className="page-nav">
            <button
              type="button"
              className="btn btn-sm"
              disabled={!canPrev || loading}
              onClick={() => setPage((p) => Math.max(1, p - 1))}
            >
              ← 上一页
            </button>
            <span>
              第 {page} 页 / {Math.max(target.page_end, page)}
            </span>
            <button
              type="button"
              className="btn btn-sm"
              disabled={!canNext || loading}
              onClick={() => setPage((p) => p + 1)}
            >
              下一页 →
            </button>
            {multiPage && page !== target.page_end ? (
              <button type="button" className="btn btn-sm btn-ghost" onClick={() => setPage(target.page_end)}>
                跳到第 {target.page_end} 页
              </button>
            ) : null}
          </div>

          {loading ? <LoadingState label={`读取第 ${page} 页文本…`} rows={4} /> : null}

          {error ? (
            <div className="banner error" role="alert">
              <div>
                <div className="banner-title">
                  {error.status === 404 ? "文档不存在或无权访问" : error.status === 401 ? "缺少管理员令牌" : "页面加载失败"}
                  <span className="tag danger" style={{ marginLeft: 8 }}>
                    HTTP {error.status}
                  </span>
                </div>
                <div>
                  {error.detail}
                  {error.status === 404 ? "（后端不区分“不存在”与“越权”，请切换身份后重试）" : ""}
                </div>
              </div>
            </div>
          ) : null}

          {!loading && !error ? (
            <>
              {view.note ? <div className="span-note">{view.note}</div> : null}
              <pre className="page-text">
                {view.before}
                {view.mark ? (
                  <mark ref={markRef} title={`char ${target.char_start}–${target.char_end}`}>
                    {view.mark}
                  </mark>
                ) : null}
                {view.after}
              </pre>
              <div className="modal-sub" style={{ marginTop: 10 }}>
                {pageText?.text ? `本页 ${pageText.text.length} 字符` : ""}
                {view.mark ? ` · 已高亮 ${view.mark.length} 字符` : " · 本页无高亮"}
              </div>
            </>
          ) : null}
        </div>
      </div>
    </div>
  );
}