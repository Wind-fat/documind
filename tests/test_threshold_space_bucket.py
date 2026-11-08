"""阈值按 `(reranker, embed_space)` 分桶（P1-10）。

上一轮交叉验证的**硬结论**（`evals/README.md` §"第二份文档交叉验证"、
`evals/reports/refusal_signal_meinuohua.md`）：同一份语料上标定的 1.5055，换到第二份文档后
可答题最低分从 1.5759 掉到 0.0706，同域难例只拒 10/24。**阈值不跨库**，而当时桶键只有 `reranker`。

这份测试钉住三件事，任何一件回退就说明"跨库静默继承"又回来了：
1. 空间内有标定 → 生效（source=calibrated），/health 报出空间与来源；
2. 换空间 → **不继承**，退回 builtin，/health 带专门字段 + 文案（"未生效"）；
3. `embed_space=''` 的老库行 → 按"历史通用值"处理：只告警、不生效，并给出回填命令。

已知边界（同样被测试钉住，防止有人把它当成"空间分桶解决了跨库问题"）：
两份语料共用同一个 embedder 时空间相同，标定值**仍然会互相生效**——
空间分桶拦的是"换 embedder"，不是"换语料"。见 `test_same_space_different_corpus_still_shares`。
"""

from __future__ import annotations

import httpx
import pytest

from documind.app import app
from documind.db import EvalRun
from documind.thresholds import (
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
    value: float = -4.2,
    dataset: str = "default",
) -> EvalRun:
    return EvalRun(
        strategy="retrieval-calib:structure",
        dataset=dataset,
        n_cases=63,
        reranker=reranker,
        embed_space=embed_space,
        recommended_threshold=value,
    )


# ---------------------------------------------------------------- ① 同空间


async def test_same_space_calibration_takes_effect(db, clean_settings, monkeypatch):
    monkeypatch.delenv("DOCUMIND_REFUSAL_THRESHOLD", raising=False)
    monkeypatch.delenv("DOCUMIND_REFUSAL_RERANK_THRESHOLD", raising=False)
    from documind.config import get_settings

    get_settings.cache_clear()

    assert current_embed_space() == SPACE, "测试环境的空间变了，下面所有常量都要跟着改"
    db.add(_cal(value=-4.2))
    await db.commit()

    st = await resolve_refusal_threshold(db, "lexical")
    assert st.source == "calibrated"
    assert st.value == pytest.approx(-4.2)
    assert st.embed_space == SPACE
    assert st.calibrated_spaces == [SPACE]
    assert st.warning == ""
    assert "hash-bigram@512" in st.reason

    body = await _health()
    thr = body["refusal_threshold"]
    assert thr["source"] == "calibrated" and thr["value"] == pytest.approx(-4.2)
    assert thr["embed_space"] == SPACE == body["embed_space"]
    assert thr["calibrated_spaces"] == [SPACE]
    assert thr["other_space_calibration"] is None
    assert thr["legacy_calibration"] is None
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


# ------------------------------------------------- ④ 回归：suhao 的 1.5055 不许在美诺华生效


async def test_suhao_threshold_does_not_leak_into_meinuohua_space(db, clean_settings, monkeypatch):
    """回归场景（真实数据，不是构造的）：

    `data/real_dashscope.db`（suhao，59 页）上标出 1.5055；`data/meinuohua_dashscope.db`
    （美诺华，50 页）上可答最低分只有 0.0706、同域难例只拒 10/24。两份文档用**同一个**
    embedder text-embedding-v3@1024，所以按空间分桶在这条真实数据上**不构成隔离**——
    能拦住它的是"这份标定属于哪个空间/哪份语料"这件事必须被显式写下。

    这条测试钉住当前实现能做到的部分：当解析空间与标定空间不同（或标定行没有空间）时，
    1.5055 一律不生效；同时下面 `test_same_space_different_corpus_still_shares`
    把"同空间仍会共享"这个边界也钉住，避免把空间分桶当成跨库隔离。
    """
    monkeypatch.delenv("DOCUMIND_REFUSAL_THRESHOLD", raising=False)
    monkeypatch.delenv("DOCUMIND_REFUSAL_RERANK_THRESHOLD", raising=False)
    from documind.config import get_settings

    get_settings.cache_clear()

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


async def test_same_space_different_corpus_still_shares(db, clean_settings, monkeypatch):
    """**已知边界**：同一空间、不同语料 → 标定值仍然生效。

    真实数据就是这个形状：suhao 与美诺华都用 `text-embedding-v3@1024`，同一份 1.5055
    在后者上会把 2/26 可答题误拒。空间分桶拦不住它，拦它需要"语料/库"维度的桶键。
    这条测试存在的意义是：不许有人拿"已经按空间分桶了"当作跨库问题已解决的证据。
    """
    monkeypatch.delenv("DOCUMIND_REFUSAL_THRESHOLD", raising=False)
    monkeypatch.delenv("DOCUMIND_REFUSAL_RERANK_THRESHOLD", raising=False)
    from documind.config import get_settings

    get_settings.cache_clear()

    db.add(_cal(embed_space=SUHAO_SPACE, value=SUHAO_THRESHOLD, dataset="suhao"))
    await db.commit()

    st = await resolve_refusal_threshold(db, "lexical", SUHAO_SPACE)
    assert st.source == "calibrated"
    assert st.value == pytest.approx(SUHAO_THRESHOLD)
    # 报告里能看出这条标定是在哪份语料上标的（运维据此判断能不能用到另一份语料）
    assert st.calibrated_dataset == "suhao"


# ------------------------------------------------------------ 其它边界


async def test_unknown_space_never_applies_any_calibration(db, clean_settings):
    """embedder 不可用（空间未知）时，宁可退回 builtin，也不猜一个空间去匹配。"""
    db.add(_cal(embed_space=SPACE, value=-4.2))
    await db.commit()

    st = await resolve_refusal_threshold(db, "lexical", "")
    assert st.source == "builtin" and st.value == pytest.approx(-6.0)
    assert st.embed_space == ""
    assert st.calibrated_spaces == [SPACE]
    assert "无法确定当前向量空间" in st.warning


async def test_env_override_reports_both_space_and_calibration(db, clean_settings, monkeypatch):
    """env 覆盖仍然最高优先，但说明里必须带上空间（否则不知道在覆盖哪个桶的值）。"""
    db.add(_cal(embed_space=SPACE, value=-4.0))
    await db.commit()
    monkeypatch.setenv("DOCUMIND_REFUSAL_THRESHOLD", "2.5")
    from documind.config import get_settings

    get_settings.cache_clear()

    st = await resolve_refusal_threshold(db, "lexical")
    assert st.source == "env" and st.value == pytest.approx(2.5)
    assert st.mismatch is True
    assert st.embed_space == SPACE and SPACE in st.warning
    # 别的空间的标定与 env 无关，不能混进 calibrated
    monkeypatch.delenv("DOCUMIND_REFUSAL_THRESHOLD")
    get_settings.cache_clear()
    st2 = await resolve_refusal_threshold(db, "lexical", SUHAO_SPACE)
    assert st2.source == "builtin" and st2.calibrated is None