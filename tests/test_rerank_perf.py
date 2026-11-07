"""重排性能的回归钉子：截断长度、候选数收敛、以及一条宽松的耗时上限。

耗时上限刻意放宽（默认配置下 <5s，实测约 2.7s）——CI 机器负载不可控，
钉住的是"别退回 21s 那种按 512 token × 30 条全量打分"的量级，不是微基准。
没有本地 bge-reranker-base 时整组跳过（CI 用 lexical 后端，不下载模型）。
"""

from __future__ import annotations

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


@needs_model
def test_rerank_default_config_latency():
    """默认配置下重排 production 的候选条数，耗时要在一秒级（宽松上限 5s 防 CI 抽风）。"""
    s = get_settings()
    r = CrossEncoderReranker()
    docs = [("第三条 员工每年享有带薪年假五天，工作满十年者享有十天。" * 40) for _ in range(s.rerank_candidates)]
    r.score("年假有几天", docs[:2])  # 预热：首次 forward 要把权重页进来
    t0 = time.perf_counter()
    scores = r.score("年假有几天", docs)
    elapsed = time.perf_counter() - t0
    assert len(scores) == len(docs)
    assert elapsed < 5.0, f"重排 {len(docs)} 条用了 {elapsed:.2f}s，回退到全量 512 token 打分了？"