"""引用 span 真机校验：`/query` 返回的 citations 能不能在 pages 表原文里逐字回查。

为什么要单独做一次：eval 报告里的 `citation_validity` 只检查 quote 的**前缀**是否出现在
那一页里（宽松，能容忍 span 偏移错位），而"引用定位有没有变差"问的是更硬的东西——
`page_text[char_start:char_end]` 这一刀切下去，是不是正好是模型引用的那段原文。

用法：
    python scripts/verify_citation_spans.py --base-url http://127.0.0.1:8911 \
        --user admin --limit 8 --out evals/reports/span_check_focus.json
    # 对照（改前的重排参数）：
    DOCUMIND_RERANK_CANDIDATES=30 DOCUMIND_RERANK_MAX_TOKENS=512 \
    DOCUMIND_RERANK_TRUNCATE=head uvicorn documind.app:app --port 8912
    python scripts/verify_citation_spans.py --base-url http://127.0.0.1:8912 ...
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from documind.evals.metrics import citation_validity  # noqa: E402
from documind.generate import Answer, Citation  # noqa: E402

DATASET = ROOT / "evals" / "datasets" / "qa_default.jsonl"


def _load_questions(path: Path, limit: int | None) -> list[str]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    # 只挑"文档里有答案"的题：拒答题没有 citation，混进来只会把命中率稀释成 100%
    questions = [r["question"] for r in rows if r.get("answerable")]
    return questions[:limit] if limit else questions


def _check_citation(c: dict, page: str) -> dict:
    cs, ce = int(c["char_start"]), int(c["char_end"])
    quote = (c.get("quote") or "").strip()
    sliced = page[cs:ce] if 0 <= cs <= ce <= len(page) else ""
    # locate_quote 找的是 quote 的前 40 字，所以切片应当等于 quote 的开头
    span_ok = bool(sliced) and quote.startswith(sliced)
    return {
        "ref": c["ref"],
        "page_start": c["page_start"],
        "char_start": cs,
        "char_end": ce,
        "slice_len": len(sliced),
        "span_ok": span_ok,
        "slice_head": sliced[:40],
        "quote_head": quote[:40],
        "quote_verbatim": bool(quote) and quote in page,
        "quote_prefix_ok": bool(quote) and page.find(quote[:40]) >= 0,
        "page_len": len(page),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8911")
    ap.add_argument("--user", default="admin")
    ap.add_argument("--dataset", default=str(DATASET))
    ap.add_argument("--limit", type=int, default=8)
    ap.add_argument("--strategy", default="structure")
    ap.add_argument("--timeout", type=float, default=180.0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    questions = _load_questions(Path(args.dataset), args.limit)
    headers = {"X-User": args.user}
    rows: list[dict] = []
    page_cache: dict[tuple[str, int], str] = {}

    with httpx.Client(base_url=args.base_url, timeout=args.timeout, headers=headers, trust_env=False) as c:
        for q in questions:
            r = c.post("/query", json={"question": q, "strategy": args.strategy})
            r.raise_for_status()
            body = r.json()
            cites = body.get("citations") or []
            checks = []
            for cite in cites:
                key = (cite["doc_id"], cite["page_start"])
                if key not in page_cache:
                    pr = c.get(f"/documents/{key[0]}/pages/{key[1]}")
                    page_cache[key] = pr.json()["text"] if pr.status_code == 200 else ""
                checks.append(_check_citation(cite, page_cache[key]))
            ans = Answer(text=body.get("text") or "", citations=[Citation(**x) for x in cites])
            page_texts = {k: v for k, v in page_cache.items()}
            rows.append(
                {
                    "question": q,
                    "refused": body.get("refused"),
                    "reason": body.get("reason", ""),
                    "n_citations": len(cites),
                    "span_ok": sum(1 for x in checks if x["span_ok"]),
                    "quote_verbatim": sum(1 for x in checks if x["quote_verbatim"]),
                    "citation_validity": citation_validity(ans, page_texts),
                    "checks": checks,
                }
            )
            print(
                f"[{len(rows)}/{len(questions)}] cites={len(cites)} "
                f"span_ok={sum(1 for x in checks if x['span_ok'])}/{len(checks)} "
                f"refused={body.get('refused')} q={q[:28]}"
            )

    total = sum(r["n_citations"] for r in rows)
    span_ok = sum(r["span_ok"] for r in rows)
    verbatim = sum(r["quote_verbatim"] for r in rows)
    answered = [r for r in rows if not r["refused"]]
    summary = {
        "base_url": args.base_url,
        "questions": len(rows),
        "answered": len(answered),
        "refused": len(rows) - len(answered),
        "citations": total,
        "span_ok": span_ok,
        "span_hit_rate": round(span_ok / total, 4) if total else None,
        "quote_verbatim": verbatim,
        "quote_verbatim_rate": round(verbatim / total, 4) if total else None,
        "citation_validity_mean": round(sum(r["citation_validity"] for r in rows) / len(rows), 4) if rows else None,
        "rows": rows,
    }
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, ensure_ascii=False, indent=2))
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print("written:", out)
    return 0 if total else 1


if __name__ == "__main__":
    raise SystemExit(main())