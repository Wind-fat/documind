"""校验「同域拒答难例」的出处与缺失证据（不调模型、不联网）。

为什么需要它：难例的价值全在标注可信度上。上一轮只有 2 条同域难例，其中一条的
`abs_check` 是散文（"万元/元 金额表述在该章无对应数值"），复核时无法判断是真缺
还是没找。这里把"有出处"和"答案不在"都变成可执行的断言：

1. `source_quote` 必须**逐字**出现在 `source_page` 那一页（归一化空白后比较）——
   证明那条"相似条款"真的在文里，不是编的；
2. `anchor` 必须是 `source_quote` 的子串，且在默认口径下，包含 anchor 的那一句
   **不含任何数量词**（阿拉伯数字，或"三十日/500 万元/25%"这类中文数量）——
   证明条款确实"没给数值"；确实给了解释不了的数值时，条目里显式写
   `no_digits=false` + `digits_note`，不静默放过；
3. `abs_terms` 每一项在**全文**里必须 0 次命中——这就是 `abs_check` 的机器版，
   散文只作为人读的补充说明。

用法：
    ./.venv/Scripts/python.exe scripts/verify_hard_neg.py \
        --dataset evals/datasets/hard_neg_suhao.jsonl \
        --db data/real_dashscope.db
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# 数量词：阿拉伯数字，或「二十五%」「500 万元」「30 日」「十二个月」这类中文数量。
# 中文数字后必须跟单位才算数量，避免把「第一条第（一）项」「统一」误判。
_QUANTITY = re.compile(
    r"[0-9０-９]+"
    r"|(?<![下上每同本该次当])[零一二三四五六七八九十百千万亿两]{1,6}\s*(?:日|天|个月|月|年|元|万元|亿元|%|％|倍|份|名|人|次)"
)


def norm(s: str) -> str:
    return re.sub(r"\s+", "", s or "")


def sentences(text: str) -> list[str]:
    # 页码是版面页脚/页眉（"12\n\n第四条 …" 或残缺行首的 "6合决策程序…"），
    # 不剥掉会被当成数量词，把"条款没给数值"的断错判成 FAIL。
    return [s for s in re.split(r"[。；;\n]", re.sub(r"^\s*\d+\s*", "", text)) if s.strip()]


def load_pages(db: Path) -> dict[int, str]:
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        return {int(no): (txt or "") for no, txt in con.execute("select page_no, text from pages order by page_no")}
    finally:
        con.close()


def check_case(case: dict, pages: dict[int, str], corpus: str) -> list[str]:
    errs: list[str] = []
    cid = case.get("id", "?")
    for key in ("category", "source_page", "source_quote", "anchor", "abs_terms", "source", "abs_check"):
        if not case.get(key):
            errs.append(f"{cid}: 缺字段 {key}")

    page = pages.get(int(case.get("source_page", -1)))
    if page is None:
        return errs + [f"{cid}: source_page={case.get('source_page')} 不在库里（共 {len(pages)} 页）"]

    quote, anchor = norm(case["source_quote"]), norm(case["anchor"])
    if quote not in norm(page):
        errs.append(f"{cid}: source_quote 不在第 {case['source_page']} 页原文里（出处不可核）")
    if anchor and anchor not in quote:
        errs.append(f"{cid}: anchor 不是 source_quote 的子串")

    if case.get("no_digits", True) and anchor:
        hits = [s for s in sentences(page) if anchor in norm(s)]
        if not hits:
            errs.append(f"{cid}: 在第 {case['source_page']} 页找不到含 anchor 的句子")
        for s in hits:
            m = _QUANTITY.search(norm(s))
            if m:
                errs.append(f"{cid}: 声称没给数值，但该句含数量词 {m.group(0)!r}（若属他义，请写 no_digits=false + digits_note）")

    for term in case["abs_terms"]:
        n = corpus.count(norm(term))
        if n:
            errs.append(f"{cid}: abs_terms 里的 {term!r} 在全文中出现 {n} 次，不能作为「答案不在」的证据")
    return errs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=str(ROOT / "evals" / "datasets" / "hard_neg_suhao.jsonl"))
    ap.add_argument("--db", default=str(ROOT / "data" / "real_dashscope.db"))
    args = ap.parse_args()

    cases = [json.loads(x) for x in Path(args.dataset).read_text(encoding="utf-8").splitlines() if x.strip()]
    pages = load_pages(Path(args.db))
    corpus = norm("".join(pages[k] for k in sorted(pages)))

    errors: list[str] = []
    for c in cases:
        errors += check_case(c, pages, corpus)

    cat = Counter(c["category"] for c in cases)
    amb = [c["id"] for c in cases if c.get("ambiguous")]
    print(f"[hard-neg] n={len(cases)} 页数={len(pages)} 全文={len(corpus)} 字")
    print("[分类] " + " · ".join(f"{k}={cat[k]}" for k in ("rate", "amount", "deadline", "subject")))
    print(f"[有争议] {len(amb)} 条: {', '.join(amb) if amb else '无'}")
    for c in cases:
        print(f"  {c['id']} p{c['source_page']:>2} {c['category']:<8} amb={str(bool(c.get('ambiguous'))):<5} {c['question'][:42]}")
    if errors:
        print(f"\n[FAIL] {len(errors)} 处问题：")
        for e in errors:
            print("  - " + e)
        return 1
    print(f"\n[OK] {len(cases)}/{len(cases)} 条通过：出处逐字可核 · 缺失词全文零命中 · 条款无数量词（除显式豁免）")
    return 0


if __name__ == "__main__":
    sys.exit(main())