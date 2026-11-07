"use client";

import { pageLabel, rankLabel, score } from "@/lib/format";
import type { Hit } from "@/lib/types";

export default function EvidenceCard({
  hit,
  refNo,
  flash,
  onOpen,
}: {
  hit: Hit;
  refNo: number;
  flash: boolean;
  onOpen: (hit: Hit) => void;
}) {
  return (
    <article
      id={`evidence-${refNo}`}
      className={`evidence-card${flash ? " flash" : ""}`}
      role="button"
      tabIndex={0}
      onClick={() => onOpen(hit)}
      onKeyDown={(event) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          onOpen(hit);
        }
      }}
      aria-label={`证据 ${refNo}：${hit.doc_title} ${pageLabel(hit.page_start, hit.page_end)}`}
    >
      <div className="evidence-top">
        <span className="evidence-ref">{refNo}</span>
        <span className="evidence-title" title={hit.doc_title}>
          《{hit.doc_title}》
        </span>
        <span className={`tag${hit.page_start === hit.page_end ? "" : " warn"}`}>
          {pageLabel(hit.page_start, hit.page_end)}
        </span>
        {hit.section_path ? <span className="tag accent">{hit.section_path}</span> : null}
      </div>

      <div className="evidence-quote">{hit.text.trim() || "（空片段）"}</div>

      <div className="evidence-foot">
        <span title="BM25 召回名次（0 基）">BM25 {rankLabel(hit.bm25_rank)}</span>
        <span className="sep">/</span>
        <span title="向量召回名次（0 基）">VEC {rankLabel(hit.vec_rank)}</span>
        <span className="sep">/</span>
        <span title="cross-encoder 重排分">RERANK {score(hit.score)}</span>
        <span className="sep">/</span>
        <span title="RRF 融合分">RRF {score(hit.rrf, 4)}</span>
        <span className="sep">/</span>
        <span className="mono" title="在页文本中的字符偏移">
          {hit.page_start}:{hit.char_start}–{hit.char_end}
        </span>
        <span className="evidence-open">查看原文 →</span>
      </div>
    </article>
  );
}

export function EvidenceSkeleton() {
  return (
    <div className="evidence-skel" aria-hidden="true">
      <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
        <div className="skeleton" style={{ width: 22, height: 14 }} />
        <div className="skeleton" style={{ width: 130, height: 12 }} />
        <div className="skeleton" style={{ width: 52, height: 12 }} />
      </div>
      <div className="skeleton" style={{ width: "94%", height: 11 }} />
      <div className="skeleton" style={{ width: "82%", height: 11 }} />
      <div className="skeleton" style={{ width: "46%", height: 9 }} />
    </div>
  );
}