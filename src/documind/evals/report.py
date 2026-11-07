"""报告渲染：markdown（贴 README）+ HTML（评测看板，给前端 iframe 用）。"""

from __future__ import annotations

import html
import json
import time

from ..config import ROOT

REPORTS = ROOT / "evals" / "reports"

COLS = [
    ("recall_at_5", "Recall@5", "up"),
    ("context_recall", "Context Recall", "up"),
    ("faithfulness", "Faithfulness", "up"),
    ("answer_relevancy", "Answer Relevancy", "up"),
    ("citation_validity", "引用有效率", "up"),
    ("refusal_accuracy", "拒答准确率", "up"),
    ("false_refusal_rate", "误拒率", "down"),
    ("latency_p95_ms", "P95 延迟(ms)", "down"),
    ("cost_per_query_usd", "单问成本($)", "down"),
]


def write_reports(dataset: str, summary: dict, details: dict, calibration: dict) -> dict:
    REPORTS.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    md = _markdown(dataset, summary, calibration)
    md_path = REPORTS / "ablation.md"
    md_path.write_text(md, encoding="utf-8")
    snap = REPORTS / f"ablation-{stamp}.md"
    snap.write_text(md, encoding="utf-8")
    html_path = REPORTS / "ablation.html"
    html_path.write_text(_html(dataset, summary, calibration), encoding="utf-8")
    json_path = REPORTS / f"details-{stamp}.json"
    json_path.write_text(json.dumps({"summary": summary, "calibration": calibration, "details": details}, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"markdown": str(md_path), "html": str(html_path), "json": str(json_path), "snapshot": str(snap)}


def _fmt(v, kind: str) -> str:
    if v is None:
        return "—"
    if kind == "down":
        return f"{v:.0f}" if isinstance(v, int) or float(v) >= 100 else f"{float(v):.3f}"
    return f"{float(v):.3f}" if float(v) <= 1 else f"{float(v):.0f}"


def _markdown(dataset: str, summary: dict, calibration: dict) -> str:
    strategies = list(summary.keys())
    lines = [
        f"# 切块策略消融实验（dataset: {dataset}）",
        "",
        f"生成时间：{time.strftime('%Y-%m-%d %H:%M')}　"
        f"样本：{summary[strategies[0]]['n_cases']} 条"
        f"（可答 {summary[strategies[0]]['n_answerable']} / 不可答 {summary[strategies[0]]['n_unanswerable']}）",
        "",
        "| 指标 | " + " | ".join(strategies) + " |",
        "|---|" + "---|" * len(strategies),
    ]
    for key, label, kind in COLS:
        row = " | ".join(_fmt(summary[s].get(key), kind) for s in strategies)
        lines.append(f"| {label} | {row} |")
    rec = calibration["recommended"]
    lines += [
        "",
        "## 拒答阈值标定",
        "",
        calibration["rule"],
        "",
        f"推荐阈值 **top_score ≥ {rec['threshold']}**：此时无答案问题拒答率 {rec['refusal_accuracy']:.1%}、"
        f"有答案问题误拒率 {rec['false_refusal']:.1%}。",
        "",
        "| 阈值 | 拒答准确率 | 误拒率 |",
        "|---|---|---|",
    ]
    for c in calibration["curve"][:: max(1, len(calibration["curve"]) // 12)]:
        mark = " ← 推荐" if c["threshold"] == rec["threshold"] else ""
        lines.append(f"| {c['threshold']} | {c['refusal_accuracy']:.1%} | {c['false_refusal']:.1%}{mark} |")
    lines += [
        "",
        "## 口径说明",
        "",
        "- 指标定义见 `src/documind/evals/metrics.py` 顶部注释；judge 用同一个模型，prompt 在代码里可审计。",
        "- 数字可复现：`uv run documind eval --dataset default` 重跑即可（语料与评测集都在仓库里）。",
        "- 评测集是**合成**的（LLM 从原文出题），不是真实用户日志——这一点不做美化。",
    ]
    return "\n".join(lines) + "\n"


def _html(dataset: str, summary: dict, calibration: dict) -> str:
    strategies = list(summary.keys())
    rec = calibration["recommended"]

    def bar(v: float, maxv: float, good: bool) -> str:
        w = min(100, 0 if maxv == 0 else v / maxv * 100)
        color = "var(--accent)" if good else "#e06c75"
        return (
            f"<div style='background:var(--border);border-radius:4px;height:6px;width:120px'>"
            f"<div style='background:{color};height:6px;border-radius:4px;width:{w:.0f}%'></div></div>"
        )

    rows = ""
    for key, label, kind in COLS:
        vals = [float(summary[s].get(key) or 0) for s in strategies]
        maxv = max(vals) if vals else 1
        cells = "".join(
            f"<td style='padding:10px 14px;text-align:right;font-variant-numeric:tabular-nums'>"
            f"{_fmt(v, kind)}<div style='margin-top:6px;display:flex;justify-content:flex-end'>"
            f"{bar(v, maxv, kind == 'up')}</div></td>"
            for v in vals
        )
        rows += f"<tr style='border-top:1px solid var(--border)'><td style='padding:10px 14px'>{html.escape(label)}</td>{cells}</tr>"

    curve = calibration["curve"]
    pts = " ".join(
        f"{20 + i * (560 / max(len(curve) - 1, 1)):.1f},"
        f"{200 - c['refusal_accuracy'] * 180:.1f}"
        for i, c in enumerate(curve)
    )
    pts2 = " ".join(
        f"{20 + i * (560 / max(len(curve) - 1, 1)):.1f},"
        f"{200 - (1 - c['false_refusal']) * 180:.1f}"
        for i, c in enumerate(curve)
    )
    return f"""<!doctype html><html lang="zh"><head><meta charset="utf-8"><title>DocuMind 评测看板</title>
<style>
  body{{background:#0b0e14;color:#e6edf3;font-family:-apple-system,'Segoe UI',sans-serif;margin:0;padding:28px 32px}}
  h1{{font-size:20px;margin:0 0 4px}} .sub{{color:#8b949e;font-size:13px;margin-bottom:22px}}
  table{{border-collapse:collapse;background:#11151d;border:1px solid #21262d;border-radius:8px;overflow:hidden}}
  th{{padding:10px 14px;font-size:12px;color:#8b949e;font-weight:600;text-align:right}}
  th:first-child,td:first-child{{text-align:left}}
  .card{{background:#11151d;border:1px solid #21262d;border-radius:8px;padding:16px 18px;margin-top:18px}}
  .kpi{{display:flex;gap:28px;margin-top:8px}} .kpi div{{font-size:24px;font-weight:600}}
  .kpi span{{display:block;font-size:12px;color:#8b949e;font-weight:400;margin-top:2px}}
</style></head><body>
<h1>切块策略消融实验</h1>
<div class="sub">dataset: {html.escape(dataset)} · {summary[strategies[0]]['n_cases']} 条样本 · 同一语料、同一套检索与生成链路，只换切块</div>
<table><tr><th>指标</th>{''.join(f"<th>{html.escape(s)}</th>" for s in strategies)}</tr>{rows}</table>
<div class="card"><b>拒答阈值标定</b>
  <div class="sub" style="margin:6px 0 10px">{html.escape(calibration['rule'])}</div>
  <svg width="600" height="215" style="overflow:visible">
    <polyline points="{pts}" fill="none" stroke="#58a6ff" stroke-width="2"/>
    <polyline points="{pts2}" fill="none" stroke="#e06c75" stroke-width="2" stroke-dasharray="4 3"/>
    <text x="0" y="215" fill="#8b949e" font-size="11">阈值低 →</text>
    <text x="480" y="215" fill="#8b949e" font-size="11">→ 阈值高</text>
    <text x="330" y="26" fill="#58a6ff" font-size="11">拒答准确率</text>
    <text x="330" y="42" fill="#e06c75" font-size="11">不误拒率（1-误拒）</text>
  </svg>
  <div class="kpi">
    <div>{rec['threshold']}<span>推荐阈值</span></div>
    <div>{rec['refusal_accuracy']:.0%}<span>无答案问题上拒答率</span></div>
    <div>{rec['false_refusal']:.0%}<span>有答案问题上误拒率</span></div>
  </div>
</div>
</body></html>"""