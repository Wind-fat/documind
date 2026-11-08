"""对同一批问题两次跑 `/query`（两个服务端，思考开/关），报 token、延迟、span 命中率、拒答判定。

为什么需要它：`enable_thinking` 是**成本开关**，改它的时候必须能同时看到"省了多少 token"和
"引用/拒答有没有变差"。只报 token 的对照等于没做对照。

用法（两个终端各起一个 uvicorn，只有思考档不同）：
    DOCUMIND_LLM_ANSWER_THINKING=true  ... --port 8913
    DOCUMIND_LLM_ANSWER_THINKING=false ... --port 8914
    python scripts/ab_thinking_e2e.py --base-url http://127.0.0.1:8913 --tag thinking_on
    python scripts/ab_thinking_e2e.py --base-url http://127.0.0.1:8914 --tag thinking_off

数据集是 jsonl（默认 `evals/datasets/real_provider_qa.jsonl`）：`{"question": "...", "answerable": true|false}`；`answerable=false` 的题用来验拒答，
它的答案应该带 `refused=true`。span 校验复用 `scripts/verify_citation_spans.py` 的判定函数，
口径与评测报告里的 `citation_validity` 一致。
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]


def _load_verifier():
    spec = importlib.util.spec_from_file_location(
        "verify_citation_spans", ROOT / "scripts" / "verify_citation_spans.py"
    )
    if spec is None or spec.loader is None:  # pragma: no cover - 文件被挪走
        raise SystemExit("找不到 scripts/verify_citation_spans.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8913")
    ap.add_argument("--dataset", default=str(ROOT / "evals" / "datasets" / "real_provider_qa.jsonl"))
    ap.add_argument("--user", default="admin")
    ap.add_argument("--strategy", default="structure")
    ap.add_argument("--tag", default="run")
    ap.add_argument("--out", default="")
    ap.add_argument("--timeout", type=float, default=180.0)
    args = ap.parse_args()

    vcs = _load_verifier()
    rows = [json.loads(line) for line in Path(args.dataset).read_text(encoding="utf-8").splitlines() if line.strip()]
    page_cache: dict[tuple[str, int], str] = {}
    out_rows: list[dict] = []

    with httpx.Client(
        base_url=args.base_url, timeout=args.timeout, trust_env=False, headers={"X-User": args.user}
    ) as c:
        for i, row in enumerate(rows, start=1):
            t0 = time.perf_counter()
            r = c.post("/query", json={"question": row["question"], "strategy": args.strategy})
            wall_ms = int((time.perf_counter() - t0) * 1000)
            r.raise_for_status()
            body = r.json()
            cites = body.get("citations") or []
            for cite in cites:
                lo, hi = sorted((int(cite["page_start"]), int(cite["page_end"])))
                for page_no in range(lo, hi + 1):
                    key = (cite["doc_id"], page_no)
                    if key not in page_cache:
                        pr = c.get(f"/documents/{key[0]}/pages/{key[1]}")
                        page_cache[key] = pr.json()["text"] if pr.status_code == 200 else ""
            checks = [vcs._check_citation(cite, page_cache) for cite in cites]
            out_rows.append(
                {
                    "question": row["question"],
                    "expected_answerable": bool(row.get("answerable", True)),
                    "refused": bool(body.get("refused")),
                    "reason": body.get("reason", ""),
                    "answer": body.get("text", ""),
                    "selfcheck_supported": (body.get("selfcheck") or {}).get("supported"),
                    "prompt_tokens": body.get("prompt_tokens"),
                    "completion_tokens": body.get("completion_tokens"),
                    "cost_usd": body.get("cost_usd"),
                    "latency_ms": body.get("latency_ms"),
                    "wall_ms": wall_ms,
                    "retrieval_ms": body.get("retrieval_ms"),
                    "generation_ms": body.get("generation_ms"),
                    "top_score": body.get("top_score"),
                    "n_citations": len(cites),
                    "span_ok": sum(1 for x in checks if x["span_ok"]),
                    "span_valid": sum(1 for x in checks if x["span_valid"]),
                    "span_sources": sorted({x["span_source"] for x in checks}),
                }
            )
            print(
                f"[{i}/{len(rows)}] refused={out_rows[-1]['refused']} cites={len(cites)} "
                f"tok={out_rows[-1]['prompt_tokens']}+{out_rows[-1]['completion_tokens']} "
                f"ms={out_rows[-1]['latency_ms']} q={row['question'][:26]}"
            )

    answered = [r for r in out_rows if not r["refused"]]
    unans = [r for r in out_rows if not r["expected_answerable"]]
    cites = sum(r["n_citations"] for r in out_rows)
    summary = {
        "tag": args.tag,
        "base_url": args.base_url,
        "questions": len(out_rows),
        "answered": len(answered),
        "refused": len(out_rows) - len(answered),
        "citations": cites,
        "span_ok": sum(r["span_ok"] for r in out_rows),
        "span_valid": sum(r["span_valid"] for r in out_rows),
        "span_hit_rate": round(sum(r["span_ok"] for r in out_rows) / cites, 4) if cites else None,
        "span_valid_rate": round(sum(r["span_valid"] for r in out_rows) / cites, 4) if cites else None,
        "prompt_tokens": sum(r["prompt_tokens"] or 0 for r in out_rows),
        "completion_tokens": sum(r["completion_tokens"] or 0 for r in out_rows),
        "cost_usd": round(sum(r["cost_usd"] or 0 for r in out_rows), 8),
        "latency_ms_mean": round(sum(r["latency_ms"] or 0 for r in out_rows) / len(out_rows)),
        "latency_ms_max": max(r["latency_ms"] or 0 for r in out_rows),
        "refusals_on_unanswerable": sum(1 for r in unans if r["refused"]),
        "unanswerable": len(unans),
        "false_refusals": sum(1 for r in out_rows if r["expected_answerable"] and r["refused"]),
        "rows": out_rows,
    }
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, ensure_ascii=False, indent=2))
    out = Path(args.out) if args.out else ROOT / "data" / "real_run" / f"ab_{args.tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("written:", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())