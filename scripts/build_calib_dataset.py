"""按 gold_doc 把 87 题的全量集裁成"与当前库匹配"的标定集。

为什么需要它：`data/real_dashscope.db` 里只有 1 份 PDF（zhicheng_suhao，59 页 / 47 块），
而 `qa_default.jsonl` 的 67 条可答题是从**3 份**章程的块里抽的。直接拿全量跑，
另外两份文档的 gold_pages / gold_quote 在这份库里永远召不回来，Recall@5 与 Context Recall
会被系统性拉低 —— 那不是模型的问题，是问题和语料不匹配。

所以这里只做**减法**，不动任何一条题的标注：
1. 保留 `gold_doc == --doc` 的可答题（原 id / gold_pages / gold_quote 原样保留）；
2. 保留全部不可答题（它们本来就没有 gold，且是"文档里确实没有"的题）；
3. 可选追加手写难例（`--hard`），每条的 `source` / `abs_check` 字段写清来源与"缺失验证"，
   便于复核"是不是为了让数字好看而加题"。

用法：
    ./.venv/Scripts/python.exe scripts/build_calib_dataset.py \
        --doc "江苏苏豪时尚集团股份有限公司章 程" \
        --hard evals/datasets/hard_neg_suhao.jsonl \
        -o evals/datasets/qa_suhao.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=str(ROOT / "evals" / "datasets" / "qa_default.jsonl"))
    ap.add_argument("--doc", required=True, help="只保留 gold_doc 等于该标题的可答题")
    ap.add_argument("--hard", default=None, help="手写难例 jsonl（可多份，逗号分隔）")
    ap.add_argument("-o", "--out", default=str(ROOT / "evals" / "datasets" / "qa_suhao.jsonl"))
    args = ap.parse_args()

    base = read_jsonl(Path(args.base))
    ans = [c for c in base if c["answerable"] and c.get("gold_doc") == args.doc]
    unans = [c for c in base if not c["answerable"]]
    hard: list[dict] = []
    for part in [p for p in (args.hard or "").split(",") if p]:
        hard += read_jsonl(Path(part))

    out = ans + unans + hard
    Path(args.out).write_text(
        "".join(json.dumps(c, ensure_ascii=False) + "\n" for c in out), encoding="utf-8"
    )
    print(
        f"[build] base={len(base)} answerable(全量)={sum(1 for c in base if c['answerable'])} "
        f"→ 匹配 {args.doc!r}: {len(ans)} · 不可答 {len(unans)} · 手写难例 {len(hard)} · 合计 {len(out)}"
    )
    print(f"[out] {args.out}")


if __name__ == "__main__":
    main()