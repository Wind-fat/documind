"""同一批真问题，在两套拒答阈值下各跑一遍，对比拒答判定 / 引用 span 可回查率 / 误拒。

为什么要单独一个脚本：`verify_citation_spans.py` 只挑 `answerable=true` 的题（它的口径是
"引用命中率"，混进拒答题会把分母搞乱）。而阈值验收要看的恰恰是**两侧**：
可答题不能被误拒、难例必须被拒。所以这里对每题都发 `/query`，把 `refused` / `reason` /
citation span 回查结果并排列出来，并按题上的 `expect_refuse` 算误拒与漏拒。

判定口径与 `verify_citation_spans.py` 完全一致（直接 import 它的 `_check_citation`）：
- `span_ok`：`page[char_start:char_end]` 切出来是 quote 的开头
- `span_valid`：整句 quote 逐字落在这段切片里（= `citation_validity` 的判定）

成本：每题一次 `/query`（`selfcheck=false`、无历史 → 不做 query 改写），一次上游生成本。
被 pre-LLM 阈值拦下的题**不产生**生成调用。

用法：
    ./.venv/Scripts/python.exe scripts/compare_thresholds_e2e.py \
        --base-url http://127.0.0.1:8914 --label before --strategy structure \
        --dataset evals/datasets/threshold_probe_suhao.jsonl --out data/real_run/thr_before.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from verify_citation_spans import _as_citation, _check_citation  # noqa: E402

from documind.evals.metrics import citation_validity  # noqa: E402
from documind.generate import Answer  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8913")
    ap.add_argument("--user", default="admin")
    ap.add_argument("--dataset", default=str(ROOT / "evals" / "datasets" / "threshold_probe_suhao.jsonl"))
    ap.add_argument("--strategy", default="structure")
    ap.add_argument("--label", default="run")
    ap.add_argument("--timeout", type=float, default=180.0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    cases = [
        json.loads(line)
        for line in Path(args.dataset).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    headers = {"X-User": args.user}
    rows: list[dict] = []
    page_cache: dict[tuple[str, int], str] = {}

    with httpx.Client(base_url=args.base_url, timeout=args.timeout, headers=headers, trust_env=False) as c:
        thr = c.get("/health").json().get("refusal_threshold", {})
        for case in cases:
            r = c.post(
                "/query",
                json={"question": case["question"], "strategy": args.strategy, "selfcheck": False},
            )
            r.raise_for_status()
            body = r.json()
            cites = body.get("citations") or []
            checks = []
            for cite in cites:
                lo, hi = sorted((int(cite["page_start"]), int(cite["page_end"])))
                for page_no in range(lo, hi + 1):
                    key = (cite["doc_id"], page_no)
                    if key not in page_cache:
                        pr = c.get(f"/documents/{key[0]}/pages/{key[1]}")
                        page_cache[key] = pr.json()["text"] if pr.status_code == 200 else ""
                checks.append(_check_citation(cite, page_cache))
            ans = Answer(text=body.get("text") or "", citations=[_as_citation(x) for x in cites])
            expect = bool(case.get("expect_refuse"))
            refused = bool(body.get("refused"))
            rows.append(
                {
                    "id": case["id"],
                    "question": case["question"],
                    "expect_refuse": expect,
                    "refused": refused,
                    "reason": body.get("reason", ""),
                    "top_score": body.get("top_score"),
                    "n_citations": len(cites),
                    "span_ok": sum(1 for x in checks if x["span_ok"]),
                    "span_valid": sum(1 for x in checks if x["span_valid"]),
                    "citation_validity": citation_validity(ans, page_cache),
                    "false_refusal": refused and not expect,
                    "missed_refusal": (not refused) and expect,
                    "latency_ms": body.get("latency_ms"),
                    "answer_head": (body.get("text") or "")[:120],
                    "checks": checks,
                }
            )
            flag = "误拒" if rows[-1]["false_refusal"] else ("漏拒" if rows[-1]["missed_refusal"] else "ok")
            print(
                f"[{len(rows)}/{len(cases)}] {case['id']:10s} refused={str(refused):5s} "
                f"expect_refuse={str(expect):5s} top={body.get('top_score')} "
                f"cites={len(cites)} span_ok={rows[-1]['span_ok']}/{len(checks)} "
                f"span_valid={rows[-1]['span_valid']}/{len(checks)} {flag}",
                flush=True,
            )

    total = sum(r["n_citations"] for r in rows)
    span_ok = sum(r["span_ok"] for r in rows)
    span_valid = sum(r["span_valid"] for r in rows)
    ans_rows = [r for r in rows if not r["expect_refuse"]]
    neg_rows = [r for r in rows if r["expect_refuse"]]
    summary = {
        "label": args.label,
        "base_url": args.base_url,
        "threshold": thr,
        "questions": len(rows),
        "n_citations": total,
        "span_ok": span_ok,
        "span_ok_rate": round(span_ok / total, 4) if total else None,
        "span_valid": span_valid,
        "span_valid_rate": round(span_valid / total, 4) if total else None,
        "refused_should_answer": sum(1 for r in ans_rows if r["refused"]),
        "n_should_answer": len(ans_rows),
        "false_refusal_rate": round(sum(1 for r in ans_rows if r["refused"]) / len(ans_rows), 4) if ans_rows else None,
        "refused_should_refuse": sum(1 for r in neg_rows if r["refused"]),
        "n_should_refuse": len(neg_rows),
        "refusal_accuracy": round(sum(1 for r in neg_rows if r["refused"]) / len(neg_rows), 4) if neg_rows else None,
        "missed_refusal_ids": [r["id"] for r in neg_rows if not r["refused"]],
        "rows": rows,
    }
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, ensure_ascii=False, indent=2))
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print("written:", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())