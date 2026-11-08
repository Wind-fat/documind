"""把「缺项信号」和「分数阈值」放在同一批题上并列比较（纯算术，不调模型）。

输入是 `eval_retrieval.py --dump-hits N` 产出的 json（每题带前 N 条上下文的文本），
所以这个脚本自己不检索、不联网、不调生成模型。它只回答一个问题：

    在不误拒可答题的前提下，缺项信号能不能把同域难例的拒答率抬起来？

四种组合都算（`refused = top_score < 阈值`；`signal = missing_specifics(...).should_refuse`）：
- `threshold`：现行行为（1.5055）；
- `signal`：只用缺项信号；
- `OR`：任一判拒即拒 —— 覆盖优先，代价是误拒；
- `AND`：两者都判拒才拒 —— 精度优先，但会漏掉「分高且缺项」的难例。

两种检查范围（`documind.refusal.missing_specifics(scope=...)`）：
- `block`：形态出现在前 N 条块里任意位置（需求原始口径）；
- `sentence`：形态必须出现在「和问题词面最重合的那一条」里（章程数字密度高时才有区分力）。

用法：
    ./.venv/Scripts/python.exe scripts/refusal_signal_report.py \
        --json evals/reports/calib_suhao_hardneg_hits.json --threshold 1.5055 --top-n 5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from documind.refusal import missing_specifics  # noqa: E402

CATEGORIES = ("rate", "amount", "deadline", "subject")


def load_cases(path: Path) -> dict[str, dict]:
    cases: dict[str, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            c = json.loads(line)
            cases[c["id"]] = c
    return cases


def pct(hit: int, n: int) -> str:
    return f"{hit}/{n}={hit / n:.4f}" if n else f"{hit}/0=-"


def render(rows: list[dict], scope: str, threshold: float) -> list[str]:
    for r in rows:
        check = missing_specifics(r["question"], r["hits"], scope=scope)
        r["thr"] = r["top_score"] < threshold
        r["sig"] = check.should_refuse
        r["intents"] = check.intents
        r["missing"] = check.missing

    ans = [r for r in rows if r["answerable"]]
    hard = [r for r in rows if not r["answerable"] and r["category"] != "hr_fixed"]
    hr = [r for r in rows if not r["answerable"] and r["category"] == "hr_fixed"]
    unans = [r for r in rows if not r["answerable"]]

    combos = [
        ("threshold", lambda r: r["thr"]),
        ("signal", lambda r: r["sig"]),
        ("OR (thr|sig)", lambda r: r["thr"] or r["sig"]),
        ("AND (thr&sig)", lambda r: r["thr"] and r["sig"]),
    ]

    out: list[str] = []
    add = out.append
    add(f"## scope={scope}（每题前 {rows[0]['top_n']} 条）")
    add("")
    add("| 组合 | 可答误拒 ↓ | 同域难例拒答 ↑ | HR 拒答 | 全部不可答拒答 | 拒答率−误拒率 |")
    add("|---|---|---|---|---|---|")
    for name, fn in combos:
        for r in rows:
            r["combo"] = fn(r)
        fr = sum(1 for r in ans if r["combo"])
        hh = sum(1 for r in hard if r["combo"])
        hrh = sum(1 for r in hr if r["combo"])
        uu = sum(1 for r in unans if r["combo"])
        add(
            f"| {name} | {pct(fr, len(ans))} | {pct(hh, len(hard))} | {pct(hrh, len(hr))} | "
            f"{pct(uu, len(unans))} | {hh / max(len(hard), 1) - fr / max(len(ans), 1):+.4f} |"
        )

    add("")
    add("| 类别 | n | 阈值判拒 | 信号判拒 | 两类都拒(OR) | 信号漏掉 |")
    add("|---|---|---|---|---|---|")
    for cat in CATEGORIES:
        sub = [r for r in hard if r["category"] == cat]
        if not sub:
            continue
        missed = [r["id"].replace("hardneg-", "") for r in sub if not r["sig"]]
        add(
            f"| {cat} | {len(sub)} | {sum(1 for r in sub if r['thr'])} | {sum(1 for r in sub if r['sig'])} "
            f"| {sum(1 for r in sub if r['thr'] or r['sig'])} | {','.join(missed) or '-'} |"
        )

    add("")
    add("### 可答题上被判拒的条目（误拒来源）")
    add("")
    refused_ans = [r for r in ans if r["sig"]]
    if refused_ans:
        add("| id | top_score | 意图 | 缺项 | 问题 |")
        add("|---|---|---|---|---|")
        for r in refused_ans:
            add(
                f"| {r['id'][-3:]} | {r['top_score']} | {','.join(r['intents'])} "
                f"| {','.join(r['missing'])} | {r['question']} |"
            )
    else:
        add("（无）")

    add("")
    add("### 同域难例逐条")
    add("")
    add("| id | 类别 | top_score | 阈值拒 | 意图 | 缺项 | 信号拒 | 问题 |")
    add("|---|---|---|---|---|---|---|---|")
    for r in hard:
        add(
            f"| {r['id']} | {r['category']} | {r['top_score']} | {r['thr']} "
            f"| {','.join(r['intents']) or '-'} | {','.join(r['missing']) or '-'} | {r['sig']} | {r['question']} |"
        )
    add("")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--dataset", default=str(ROOT / "evals" / "datasets" / "qa_suhao.jsonl"))
    ap.add_argument("--threshold", type=float, default=1.5055)
    ap.add_argument("--top-n", type=int, default=5)
    ap.add_argument("--scope", action="append", default=None, choices=["block", "sentence"])
    ap.add_argument("--md", default=None, help="把同一份报告另存为 markdown")
    args = ap.parse_args()

    data = json.loads(Path(args.json).read_text(encoding="utf-8"))
    cases = load_cases(Path(args.dataset))
    strategy = next(iter(data["strategies"]))
    raw = data["strategies"][strategy]["rows"]

    if raw and "hits" not in raw[0]:
        sys.exit("这份 json 没有 hits 字段：先用 eval_retrieval.py --dump-hits 5 重跑")

    rows = []
    for r in raw:
        case = cases[r["id"]]
        hits = r["hits"][: args.top_n]
        rows.append(
            {
                **r,
                "question": case["question"],
                "category": case.get("category") or ("answerable" if r["answerable"] else "hr_fixed"),
                "hits": hits,
                "top_n": len(hits),
            }
        )

    ans_n = sum(1 for r in rows if r["answerable"])
    hard_n = sum(1 for r in rows if not r["answerable"] and r["category"] != "hr_fixed")
    hr_n = sum(1 for r in rows if not r["answerable"] and r["category"] == "hr_fixed")

    lines = [
        f"# 缺项信号 vs 分数阈值 · n={len(rows)} · strategy={strategy} · reranker={data['reranker']}",
        f"阈值={args.threshold}  top_n={args.top_n}  可答/{ans_n}  同域难例/{hard_n}  HR/{hr_n}",
        "",
    ]
    for scope in args.scope or ["block", "sentence"]:
        lines = lines + render(rows, scope, args.threshold)

    text = "\n".join(lines)
    print(text)
    if args.md:
        Path(args.md).write_text(text + "\n", encoding="utf-8")
        print(f"\n[md] {args.md}")


if __name__ == "__main__":
    main()