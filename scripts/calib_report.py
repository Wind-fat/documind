"""把 `eval_retrieval.py` 的 json 摊开算「间隔」与分域拒答率（纯算术，不调模型）。

为什么要单独一步：`eval_retrieval.py` 只报一个总拒答准确率，而总池里 20 条是 HR 域
固定题（与章程距离远、分数低），28 条是同域难例（分数高）。混在一起会把
"阈值的分域能力"平均掉。这里按 `qa_*.jsonl` 里的 `category` 字段分组重算：

- `interval = 可答下界(min top_score of answerable) - 不可答上界(max top_score of unanswerable)`
  上一轮口径就是这样（1.5759 - 1.3999 = 0.176）。interval > 0 才存在一个能完全分开
  两类的阈值；interval <= 0 表示两类分数重叠，任何单阈值都必须牺牲一边。
- 同一批题上并列比较多个候选阈值（旧标定值 vs 新推荐值），报各自的
  拒答准确率（分域）与误拒率。

用法：
    ./.venv/Scripts/python.exe scripts/calib_report.py \
        --json evals/reports/calib_suhao_hardneg.json \
        --dataset evals/datasets/qa_suhao.jsonl --threshold 1.5055
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_labels(path: Path) -> tuple[dict[str, str], set[str]]:
    out: dict[str, str] = {}
    ambiguous: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        c = json.loads(line)
        out[c["id"]] = c.get("category") or "hr_fixed"
        if c.get("ambiguous"):
            ambiguous.add(c["id"])
    return out, ambiguous


def rate(rows: list[dict], th: float) -> tuple[int, int]:
    return sum(1 for r in rows if r["top_score"] < th), len(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--dataset", default=str(ROOT / "evals" / "datasets" / "qa_suhao.jsonl"))
    ap.add_argument("--threshold", type=float, action="append", default=None,
                    help="候选阈值，可重复；默认只比 旧标定值 1.5055 与 本次推荐值")
    args = ap.parse_args()

    data = json.loads(Path(args.json).read_text(encoding="utf-8"))
    labels, ambiguous = load_labels(Path(args.dataset))
    strategy = next(iter(data["strategies"]))
    block = data["strategies"][strategy]
    rows, s = block["rows"], block["summary"]

    ans = [r for r in rows if r["answerable"]]
    unans = [r for r in rows if not r["answerable"]]
    lo = min(r["top_score"] for r in ans)
    hi = max(r["top_score"] for r in unans)
    rec = s["recommended_threshold"]

    print(f"# {Path(args.json).name} · strategy={strategy} · reranker={data['reranker']} · n={s['n_cases']}")
    print(f"可答 {len(ans)} / 不可答 {len(unans)}")
    print(f"Recall@5={s['recall_at_5']} ({s['recall_at_5_hits']})  ContextRecall={s['context_recall']}"
          f" ({s['context_recall_hits']})  MRR={s['mrr']}")
    print(f"\n可答下界(min)={lo:.4f}  不可答上界(max)={hi:.4f}  间隔={lo - hi:+.4f}")
    print(f"运行阈值={s['threshold']} (source={s['threshold_source']})  推荐阈值={rec}")

    print("\n## 分组（按 top_score 分布）")
    groups: dict[str, list[dict]] = {}
    for r in rows:
        groups.setdefault("answerable" if r["answerable"] else labels.get(r["id"], "?"), []).append(r)
    for name, sub in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        vals = sorted(x["top_score"] for x in sub)
        print(f"  {name:<10} n={len(sub):>2}  min={vals[0]:>7.4f}  max={vals[-1]:>7.4f}  median={vals[len(vals)//2]:>7.4f}")

    cands = args.threshold or sorted({1.5055, rec})
    print("\n## 同一批题上并列比较阈值（refused = top_score < 阈值）")
    header = f"{'阈值':>8} | " + " | ".join(f"{g:<10}" for g in ["可答误拒", "同域难例", "HR 固定", "全部不可答"])
    print(header)
    print("-" * len(header))
    for th in cands:
        fr, fn = rate(ans, th)
        cells = [f"{fr}/{fn}={fr / max(fn, 1):.4f}"]
        for g in ("hardneg", "hr_fixed"):
            sub = groups.get(g, [])
            if g == "hardneg":
                sub = [r for r in unans if labels.get(r["id"]) not in (None, "hr_fixed")]
            rr, rn = rate(sub, th)
            cells.append(f"{rr}/{rn}={rr / max(rn, 1):.4f}")
        rr, rn = rate(unans, th)
        cells.append(f"{rr}/{rn}={rr / max(rn, 1):.4f}")
        print(f"{th:>8} | " + " | ".join(f"{c:<10}" for c in cells))

    print("\n## 每条不可答题的 top_score（升序）")
    for r in sorted(unans, key=lambda x: x["top_score"]):
        tag = "  ← 标注有争议" if r["id"] in ambiguous else ""
        print(f"  {r['id']:<12} {r['top_score']:>7.4f}  {labels.get(r['id'], '?')}{tag}")


if __name__ == "__main__":
    main()