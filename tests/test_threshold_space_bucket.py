"""拒答阈值按 `(reranker, embed_space, corpus_fp)` 三元桶隔离（P1-10 / P1-12）。

上一轮交叉验证的**硬结论**（`evals/README.md` §"第二份文档交叉验证"、
`evals/reports/refusal_signal_meinuohua.md`）：同一份语料上标定的 1.5055，换到第二份文档后
可答题最低分从 1.5759 掉到 0.0706，同域难例只拒 10/24。**阈值不跨库**，而当时桶键只有 `reranker`，
后来加了 `embed_space`——但 suhao 与美诺华用的是**同一个** embedder，所以空间也相同。

这份测试钉住四件事，任何一件回退就说明"跨库静默继承"又回来了：
1. 三元桶内有标定 → 生效（source=calibrated），/health 报出空间、语料与来源；
2. 换空间 → **不继承**，退回 builtin，/health 带专门字段 + 文案（"未生效"）；
3. 同空间换语料 → **不继承**（P1-12 修掉的已知缺陷），/health 说"因为语料变了"，
   并把已知有标定的语料（fp 前 8 位 + 数据集 + 值）列出来——**只列不用**；
4. `embed_space=''` / `corpus_fp=''` 的老库行 → 按"历史通用值"处理：只告警、不生效，并给出回填命令。
"""

from __future__ import annotations

import httpx
import pytest
from conftest import OTHER_CORPUS_SHA, TEST_CORPUS_FP, seed_corpus

from documind.app import app
from documind.db import EvalRun
from documind.thresholds import (
    LEGACY_CORPUS,
    LEGACY_SPACE,
    current_embed_space,
    resolve_refusal_threshold,
)

# conftest 固定 DOCUMIND_EMBED_BACKEND=hash / DOCUMIND_RERANK_BACKEND=lexical
SPACE = "hash-bigram@512"
SUHAO_SPACE = "text-embedding-v3@1024"  # 真实交叉验证用的空间（suhao 库）
SUHAO_THRESHOLD = 1.5055  # 真实标定行：2026-11-08 00:30，dataset=suhao


@pytest.fixture()
def clean_settings():
    """环境变量改过之后必须清 lru_cache，否则读到的是上一个用例的 Settings。"""
    from documind.config import get_settings

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


async def _health() -> dict:
    """真打 /health（ASGI 进程内，同一库）：断言的就是运维看到的那个 JSON。"""
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test", trust_env=False) as c:
        r = await c.get("/health")
    assert r.status_code == 200, r.text
    return r.json()


def _cal(
    *,
    reranker: str = "lexical",
    embed_space: str = SPACE,
    corpus_fp: str = TEST_CORPUS_FP,
    value: float = -4.2,
    dataset: str = "default",
) -> EvalRun:
    return EvalRun(
        strategy="retrieval-calib:structure",
        dataset=dataset,
        n_cases=63,
        reranker=reranker,
        embed_space=embed_space,
        corpus_fp=corpus_fp,
        recommended_threshold=value,
    )


# ---------------------------------------------------------------- ① 三元桶全中


async def test_same_space_calibration_takes_effect(db, clean_settings, monkeypatch):
    monkeypatch.delenv("DOCUMIND_REFUSAL_THRESHOLD", raising=False)
    monkeypatch.delenv("DOCUMIND_REFUSAL_RERANK_THRESHOLD", raising=False)
    from documind.config import get_settings

    get_settings.cache_clear()

    assert current_embed_space() == SPACE, "测试环境的空间变了，下面所有常量都要跟着改"
    fp = await seed_corpus(db)
    db.add(_cal(value=-4.2))
    await db.commit()

    st = await resolve_refusal_threshold(db, "lexical")
    assert st.source == "calibrated"
    assert st.value == pytest.approx(-4.2)
    assert st.embed_space == SPACE
    assert st.corpus_fp == fp == TEST_CORPUS_FP
    assert st.calibrated_spaces == [SPACE]
    assert st.warning == ""
    assert "hash-bigram@512" in st.reason and fp in st.reason

    body = await _health()
    thr = body["refusal_threshold"]
    assert thr["source"] == "calibrated" and thr["value"] == pytest.approx(-4.2)
    assert thr["embed_space"] == SPACE == body["embed_space"]
    assert thr["corpus_fp"] == fp == body["corpus_fp"], "顶层与阈值里的指纹必须是同一个"
    assert thr["calibrated_spaces"] == [SPACE]
    assert thr["other_space_calibration"] is None
    assert thr["legacy_calibration"] is None
    assert thr["other_corpus_calibration"] is None
    # 已知有标定的语料清单：fp 前 8 位 + 数据集名 + 值
    assert thr["calibrated_corpora"][0]["corpus_fp"] == fp[:8]
    assert thr["calibrated_corpora"][0]["dataset"] == "default"
    assert thr["calibrated_corpora"][0]["value"] == pytest.approx(-4.2)
    # 老字段语义不变：calibrated 仍是生效的标定值
    assert thr["calibrated"] == pytest.approx(-4.2)
    assert thr["mismatch"] is False


# ---------------------------------------------------------------- ② 换空间


async def test_other_space_is_not_inherited_and_health_says_why(db, clean_settings, monkeypatch):
    """suhao 空间标了 1.5055，当前库是另一个空间 → 1.5055 **不生效**，/health 说清原因。"""
    monkeypatch.delenv("DOCUMIND_REFUSAL_THRESHOLD", raising=False)
    monkeypatch.delenv("DOCUMIND_REFUSAL_RERANK_THRESHOLD", raising=False)
    from documind.config import get_settings

    get_settings.cache_clear()

    await seed_corpus(db)
    db.add(_cal(embed_space=SUHAO_SPACE, value=SUHAO_THRESHOLD, dataset="suhao"))
    await db.commit()

    st = await resolve_refusal_threshold(db, "lexical")
    assert st.source == "builtin"
    assert st.value == pytest.approx(-6.0), "换了空间还沿用上一个空间的分数就是本单要堵的洞"
    assert st.value != pytest.approx(SUHAO_THRESHOLD)
    assert st.embed_space == SPACE
    assert st.calibrated_spaces == [SUHAO_SPACE]
    assert st.other_space_calibration is not None
    assert st.other_space_calibration["value"] == pytest.approx(SUHAO_THRESHOLD)
    assert st.other_space_calibration["embed_space"] == SUHAO_SPACE
    assert "还没有标定值" in st.warning and "未生效" in st.warning
    assert SUHAO_SPACE in st.warning and "1.5055" in st.warning

    body = await _health()
    thr = body["refusal_threshold"]
    assert thr["source"] == "builtin" and thr["value"] == pytest.approx(-6.0)
    assert thr["embed_space"] == SPACE
    assert thr["calibrated_spaces"] == [SUHAO_SPACE]
    assert thr["other_space_calibration"]["value"] == pytest.approx(SUHAO_THRESHOLD)
    assert "未生效" in thr["warning"]
    # 已知有标定的语料仍然列出来（只列不用）
    assert thr["calibrated_corpora"][0]["dataset"] == "suhao"
    assert thr["calibrated_corpora"][0]["value"] == pytest.approx(SUHAO_THRESHOLD)
    # 老字段：calibrated=None（没有可以生效的标定值），不能把别的空间的值塞进来
    assert thr["calibrated"] is None
    assert thr["calibrated_run"] == ""


# ---------------------------------------------------------------- ③ 老库空空间行


async def test_legacy_row_without_space_is_advisory_only(db, clean_settings, monkeypatch):
    """`embed_space=''`（迁移前的行）：只告警、不生效，并指出回填命令。

    选的处理方式：**不自动归属任何空间**。理由写在 thresholds.py 顶部第 4 条：
    无标注的行无法证明属于哪个空间，猜错就等于跨库继承——而这正是本单要消灭的行为。
    """
    monkeypatch.delenv("DOCUMIND_REFUSAL_THRESHOLD", raising=False)
    monkeypatch.delenv("DOCUMIND_REFUSAL_RERANK_THRESHOLD", raising=False)
    from documind.config import get_settings

    get_settings.cache_clear()

    await seed_corpus(db)
    db.add(_cal(embed_space=LEGACY_SPACE, value=SUHAO_THRESHOLD, dataset="suhao"))
    await db.commit()

    st = await resolve_refusal_threshold(db, "lexical")
    assert st.source == "builtin", "老行不许被当成当前空间的标定"
    assert st.value == pytest.approx(-6.0)
    assert st.calibrated is None
    assert st.calibrated_spaces == [], "空空间不算'已知有标定的空间'"
    assert st.legacy_calibration is not None
    assert st.legacy_calibration["value"] == pytest.approx(SUHAO_THRESHOLD)
    assert "历史通用值" in st.warning and "不生效" in st.warning
    assert "backfill_calibration_space.py" in st.warning

    body = await _health()
    thr = body["refusal_threshold"]
    assert thr["source"] == "builtin"
    assert thr["legacy_calibration"]["value"] == pytest.approx(SUHAO_THRESHOLD)
    assert thr["calibrated"] is None
    assert "backfill_calibration_space.py" in thr["warning"]


async def test_row_without_corpus_fp_is_advisory_only(db, clean_settings, monkeypatch):
    """同空间、但 `corpus_fp=''`（这一列出现之前的行）：同样只告警、不生效。

    这是本单的行为变更点：以前"空间匹配"就够了，现在空间匹配、语料未知也不行——
    不知道它是在哪份语料上标的，自动归属就等于重现跨语料继承（1.5055 → 美诺华 2/26 误拒）。
    """
    monkeypatch.delenv("DOCUMIND_REFUSAL_THRESHOLD", raising=False)
    monkeypatch.delenv("DOCUMIND_REFUSAL_RERANK_THRESHOLD", raising=False)
    from documind.config import get_settings

    get_settings.cache_clear()

    fp = await seed_corpus(db)
    db.add(_cal(corpus_fp=LEGACY_CORPUS, value=-4.2))
    await db.commit()

    st = await resolve_refusal_threshold(db, "lexical")
    assert st.source == "builtin", "没有语料指纹的行不许生效"
    assert st.value == pytest.approx(-6.0)
    assert st.calibrated is None
    assert st.corpus_fp == fp
    assert st.legacy_corpus_calibration is not None
    assert st.legacy_corpus_calibration["corpus_fp"] == LEGACY_CORPUS
    assert st.legacy_corpus_calibration["value"] == pytest.approx(-4.2)
    assert "没有语料指纹" in st.warning and "不生效" in st.warning
    assert "backfill_calibration_space.py" in st.warning
    assert st.calibrated_spaces == [], "缺语料的标定不算'这个空间有可用标定'"

    body = await _health()
    thr = body["refusal_threshold"]
    assert thr["source"] == "builtin"
    assert thr["legacy_corpus_calibration"]["value"] == pytest.approx(-4.2)
    assert "backfill_calibration_space.py" in thr["warning"]


# ------------------------------------------------- ④ 回归：suhao 的 1.5055 不许在美诺华生效


async def test_suhao_threshold_does_not_leak_into_meinuohua_space(db, clean_settings, monkeypatch):
    """回归场景（真实数据，不是构造的）：

    `data/real_dashscope.db`（suhao，59 页）上标出 1.5055；`data/meinuohua_dashscope.db`
    （美诺华，50 页）上可答最低分只有 0.0706、同域难例只拒 10/24。两份文档用**同一个**
    embedder text-embedding-v3@1024，所以只按空间分桶在真实数据上**不构成隔离**。

    这条测试钉住"空间不同 → 不生效"这一半；同空间换语料那一半见
    `test_same_space_different_corpus_is_not_inherited`。
    """
    monkeypatch.delenv("DOCUMIND_REFUSAL_THRESHOLD", raising=False)
    monkeypatch.delenv("DOCUMIND_REFUSAL_RERANK_THRESHOLD", raising=False)
    from documind.config import get_settings

    get_settings.cache_clear()

    await seed_corpus(db)
    db.add(_cal(reranker="lexical", embed_space=LEGACY_SPACE, value=SUHAO_THRESHOLD, dataset="suhao"))
    db.add(_cal(reranker="lexical", embed_space=SUHAO_SPACE, value=SUHAO_THRESHOLD, dataset="suhao"))
    await db.commit()

    # meinuohua 用的是另一套 embedder（例如换成本地 bge 的 512 维）：两条 suhao 标定都不适用
    for space in ("BAAI/bge-small-zh-v1.5@512", SPACE):
        st = await resolve_refusal_threshold(db, "lexical", space)
        assert st.source == "builtin", f"{space} 上不该有可用标定"
        assert st.value == pytest.approx(-6.0)
        assert st.value != pytest.approx(SUHAO_THRESHOLD)
        assert st.calibrated is None
        assert st.other_space_calibration["value"] == pytest.approx(SUHAO_THRESHOLD)
        assert "未生效" in st.warning


async def test_same_space_different_corpus_is_not_inherited(db, clean_settings, monkeypatch):
    """**已知缺陷已修**（P1-12）：同一空间、不同语料 → **不继承**。

    真实数据就是这个形状：suhao 与美诺华都用 `text-embedding-v3@1024`（空间相同），同一份
    1.5055 在后者上会把 2/26 可答题误拒。这条测试以前钉的是"仍然会共享"（提醒别人别把空间
    分桶当成跨库隔离），现在改成反向断言：**同一空间 + 同一份语料才生效，语料一变就不生效**。
    """
    monkeypatch.delenv("DOCUMIND_REFUSAL_THRESHOLD", raising=False)
    monkeypatch.delenv("DOCUMIND_REFUSAL_RERANK_THRESHOLD", raising=False)
    from documind.config import get_settings

    get_settings.cache_clear()

    # 语料 A 上标出 1.5055（空间取当前进程的空间：这样 /health 走的正是"同空间换语料"这条分支）
    fp_a = await seed_corpus(db)
    db.add(_cal(embed_space=SPACE, value=SUHAO_THRESHOLD, dataset="suhao"))
    await db.commit()

    st = await resolve_refusal_threshold(db, "lexical")
    assert st.source == "calibrated", "同一空间 + 同一份语料，标定必须生效"
    assert st.value == pytest.approx(SUHAO_THRESHOLD)
    assert st.corpus_fp == fp_a
    # 报告里能看出这条标定是在哪份语料上标的
    assert st.calibrated_dataset == "suhao"

    # 换一份语料：同一个空间、同一个 reranker，但库里的文档集合变了 → 指纹变了
    fp_b = await seed_corpus(db, OTHER_CORPUS_SHA)
    assert fp_b != fp_a

    st2 = await resolve_refusal_threshold(db, "lexical")
    assert st2.source == "builtin", "同一空间换语料还沿用旧标定，就是本单要堵的洞"
    assert st2.value == pytest.approx(-6.0)
    assert st2.value != pytest.approx(SUHAO_THRESHOLD)
    assert st2.calibrated is None
    assert st2.corpus_fp == fp_b
    assert st2.other_corpus_calibration is not None
    assert st2.other_corpus_calibration["value"] == pytest.approx(SUHAO_THRESHOLD)
    assert st2.other_corpus_calibration["corpus_fp"] == fp_a
    # 警告必须说"因为语料变了"，而不是让人以为阈值被谁改过
    assert "另一份语料" in st2.warning and "未生效" in st2.warning and "1.5055" in st2.warning
    assert fp_a[:8] in st2.warning and fp_b[:8] in st2.warning

    body = await _health()
    thr = body["refusal_threshold"]
    assert thr["source"] == "builtin" and thr["calibrated"] is None
    assert thr["corpus_fp"] == fp_b
    assert thr["other_corpus_calibration"]["corpus_fp"] == fp_a
    assert thr["other_corpus_calibration"]["value"] == pytest.approx(SUHAO_THRESHOLD)
    # 已知语料清单：列出 A（fp 前 8 位 + 数据集 + 值），但**不生效**
    listed = thr["calibrated_corpora"][0]
    assert listed["corpus_fp"] == fp_a[:8]
    assert listed["dataset"] == "suhao"
    assert listed["value"] == pytest.approx(SUHAO_THRESHOLD)


# ------------------------------------------------------------ 其它边界


async def test_unknown_space_never_applies_any_calibration(db, clean_settings):
    """embedder 不可用（空间未知）时，宁可退回 builtin，也不猜一个空间去匹配。"""
    await seed_corpus(db)
    db.add(_cal(embed_space=SPACE, value=-4.2))
    await db.commit()

    st = await resolve_refusal_threshold(db, "lexical", "")
    assert st.source == "builtin" and st.value == pytest.approx(-6.0)
    assert st.embed_space == ""
    assert st.calibrated_spaces == [SPACE]
    assert "无法确定当前向量空间" in st.warning


async def test_empty_library_never_applies_any_calibration(db, clean_settings):
    """空库（语料指纹为空串）：没有任何语料可归属，标定一律不生效。

    空串不是通配符——否则"标定行也没有指纹"的老行会在空库上悄悄生效。
    """
    db.add(_cal(corpus_fp=LEGACY_CORPUS, value=-4.2))
    await db.commit()

    st = await resolve_refusal_threshold(db, "lexical", SPACE)
    assert st.corpus_fp == ""
    assert st.source == "builtin"
    assert st.calibrated is None
    assert st.legacy_corpus_calibration is not None
    assert "没有语料指纹" in st.warning


async def test_env_override_reports_both_space_and_calibration(db, clean_settings, monkeypatch):
    """env 覆盖仍然最高优先，但说明里必须带上空间与语料（否则不知道在覆盖哪个桶的值）。"""
    fp = await seed_corpus(db)
    db.add(_cal(embed_space=SPACE, value=-4.0))
    await db.commit()
    monkeypatch.setenv("DOCUMIND_REFUSAL_THRESHOLD", "2.5")
    from documind.config import get_settings

    get_settings.cache_clear()

    st = await resolve_refusal_threshold(db, "lexical")
    assert st.source == "env" and st.value == pytest.approx(2.5)
    assert st.mismatch is True
    assert st.embed_space == SPACE and SPACE in st.warning
    assert st.corpus_fp == fp and fp in st.warning
    # 别的空间的标定与 env 无关，不能混进 calibrated
    monkeypatch.delenv("DOCUMIND_REFUSAL_THRESHOLD")
    get_settings.cache_clear()
    st2 = await resolve_refusal_threshold(db, "lexical", SUHAO_SPACE)
    assert st2.source == "builtin" and st2.calibrated is None