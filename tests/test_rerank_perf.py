"""重排性能的回归钉子：截断长度、候选数收敛、以及一条宽松的耗时上限。

耗时断言刻意不写成"绝对秒数小于 N"：这条测试在并行跑标定（12 核被打满）时
误报过一次。现在分两层，见 `test_rerank_default_config_latency` 的 docstring——
相对基线（10 条 / 2 条的倍数）不受机器忙闲影响，绝对上限只在机器没被占满时才判定。
没有本地 bge-reranker-base 时整组跳过（CI 用 lexical 后端，不下载模型）。
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from documind.config import ROOT, get_settings
from documind.rerank import CrossEncoderReranker, Reranker
from documind.retrieve import retrieve

MODEL_SNAPSHOTS = Path(ROOT) / ".models" / "models--BAAI--bge-reranker-base" / "snapshots"
HAS_MODEL = any(MODEL_SNAPSHOTS.glob("*/onnx/model.onnx"))
needs_model = pytest.mark.skipif(not HAS_MODEL, reason="本地无 bge-reranker-base 权重，跳过（CI 走 lexical）")


class RecordingReranker(Reranker):
    """只记录收到多少条候选，不真打分。"""

    name = "recording"

    def __init__(self) -> None:
        self.seen: list[str] = []

    def rerank(self, query, items, top_k):
        self.seen = [i for i, _ in items]
        return [(i, -float(n)) for n, (i, _) in enumerate(items)][:top_k]


async def test_rerank_pool_capped_by_setting(db, tmp_path: Path, monkeypatch):
    """候选数收敛：retrieve 只把前 `rerank_candidates` 条喂给重排器（但不低于 top_k）。"""
    from documind.pipeline import ingest_document

    # 语料太小会被切块按 900 字合并成一两个块，测不出"收敛"，所以现造一份。
    # 用 fixed 切块：md 解析器会把这一串"条"并成一段，structure 下只剩 1 块。
    big = tmp_path / "big.md"
    lines = ["# 员工制度汇编", "", "## 第一章 休假", ""]
    body = ("员工每年享有带薪年假五天，工作满十年者享有十天，公司不得以任何名义折现年假，"
            "也不得以调休或补贴替代；未休完的年假可以顺延至次年第一季度，逾期作废，")
    for i in range(40):
        lines += [f"第{i + 1}条 {body}条款编号{i}。", ""]
    big.write_text("\n".join(lines), encoding="utf-8")
    await ingest_document(db, big, acl_roles=["all"])

    monkeypatch.setattr(get_settings(), "rerank_candidates", 3)
    monkeypatch.setattr(get_settings(), "recall_n", 30)

    rec = RecordingReranker()
    await retrieve(db, "年假有几天", role="employee", strategy="fixed", top_k=2, reranker=rec)
    assert len(rec.seen) == 3  # 3 > top_k=2，收敛生效

    rec2 = RecordingReranker()
    await retrieve(db, "年假有几天", role="employee", strategy="fixed", top_k=5, reranker=rec2)
    assert len(rec2.seen) >= 5  # 收敛不能把 top_k 饿死（max(top_k, candidates)）


@needs_model
def test_focus_window_keeps_relevant_sentence():
    """focus 窗口要把与问题重合的那句留在窗口里，而不是按位置截前 N 字。"""
    filler = "第一条 本制度由人力资源部负责解释并在全公司范围内公布实施。" * 4
    gold = "第十二条 员工每年享有带薪年假五天，未休完的年假不得折现。"
    text = filler + gold + filler
    focus = CrossEncoderReranker(max_tokens=120, batch_size=4, truncate="focus")._window("年假可以折现吗", text)
    assert len(focus) <= 120
    assert "不得折现" in focus
    # 对照：head 模式下按 token 截到 120，gold 句（在 ~120 字之后）根本进不了模型
    assert "不得折现" not in text[:120]


@needs_model
def test_cross_encoder_truncates_to_max_tokens():
    """截断钉在 max_tokens 上：3 千字的块进模型前会被截到上限，而不是照 512 全长跑。"""
    r = CrossEncoderReranker(max_tokens=64, batch_size=4)
    inner = r._prepare()
    assert inner.tokenizer.truncation["max_length"] == 64
    enc = inner.tokenizer.encode_batch([("问：年假几天", "长" * 3000)])
    assert len(enc[0].ids) <= 64


@needs_model
def test_cross_encoder_scores_follow_input_order():
    """按长度排序组批后结果要按入参顺序还回去，否则分数会错配到别的块上。"""
    r = CrossEncoderReranker(max_tokens=64, batch_size=2)
    docs = ["第一条 员工每年享有带薪年假五天" * (i + 1) for i in range(6)]
    a = r.score("年假有几天", docs)
    b = r.score("年假有几天", list(reversed(docs)))
    assert a == pytest.approx(list(reversed(b)), rel=1e-3, abs=1e-3)


def _best_of(fn, repeats: int = 3) -> tuple[float, float]:
    """跑 `repeats` 次取最快的一次，返回 (wall, cpu)。

    取最小值而不是平均：偶发的调度抖动只会把它推慢，不会把它推快；
    平均值会让一次误报污染整条断言（这正是原来的 flaky 来源）。
    """
    best = (float("inf"), 0.0)
    for _ in range(repeats):
        cpu0, wall0 = time.process_time(), time.perf_counter()
        fn()
        wall, cpu = time.perf_counter() - wall0, time.process_time() - cpu0
        if wall < best[0]:
            best = (wall, cpu)
    return best


@needs_model
def test_rerank_default_config_latency():
    """默认配置下重排 production 的候选条数，耗时要停在一秒级。

    为什么不只是 `< 5s`：并行跑标定时这台机器 12 个核被打满，同一条断言偶发失败。
    绝对秒数在共享机器上测的是"别人占了多少"，不是"我们的实现有没有退化"。所以拆两层：

    1. **相对基线**（主断言，任何负载下都成立）：同样两条文本的耗时 vs 默认候选数
       的耗时，比值上限 10。本机基线 4.8–5.1（10 条 × 256 token / 2 条 × 256 token，
       批大小 8 带来的固定开销已包含在内）；把候选数改回 30、截断改回 512 token 时
       实测比值 14.5（16.4s vs 1.1s，见 commit 里的验证）。两次测量在同一次调用里
       前后脚做，机器忙就一起忙，比值基本不动。
    2. **绝对上限 12s**（基线 2.9s，4 倍余量，回归目标是 21s）：只在机器没被占满时判定。
       判据是本次测量的 CPU 并行度 `cpu/wall` —— 被抢占时它会明显掉下来；
       掉到逻辑核数的一半以下就 `skip` 这一条而不是红灯（断言本身没被删，机器空闲时照跑）。
    """
    s = get_settings()
    r = CrossEncoderReranker()
    docs = [("第三条 员工每年享有带薪年假五天，工作满十年者享有十天。" * 40) for _ in range(s.rerank_candidates)]
    small = docs[:2]
    r.score("年假有几天", small)  # 预热：首次 forward 要把权重页进来

    t_small, _ = _best_of(lambda: r.score("年假有几天", small))
    out: list[list[float]] = []

    def _score_full() -> None:
        out.append(r.score("年假有几天", docs))

    t_full, cpu_full = _best_of(_score_full)
    assert len(out[-1]) == len(docs)

    ratio = t_full / max(t_small, 1e-6)
    assert ratio < 10.0, (
        f"重排 {len(docs)} 条 / 2 条的耗时比 {ratio:.1f}（基线 ~5，回归到 30 条 × 512 token 会到 ~30）："
        f"{t_full:.2f}s vs {t_small:.2f}s"
    )

    achieved = cpu_full / max(t_full, 1e-6)
    logical = max(1, os.cpu_count() or 1)
    if achieved < 0.5 * logical:
        pytest.skip(
            f"机器被占（CPU 并行度 {achieved:.1f}/{logical}），绝对耗时不可比；"
            f"相对基线已校验：{t_full:.2f}s / {t_small:.2f}s = {ratio:.1f}"
        )
    assert t_full < 12.0, f"重排 {len(docs)} 条用了 {t_full:.2f}s（基线 2.9s），回退到全量 512 token 打分了？"