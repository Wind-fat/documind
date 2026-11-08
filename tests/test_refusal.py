"""缺项信号（`documind.refusal`）的契约测试。

信号要回答的只有一件事：**问题在要一个具体值，而检索到的块里没有那个形态**。
所以两个方向都要钉住：
- 该响的时候响（条款在、值不在）；
- 不该响的时候绝不响（问题没要值 / 值以另一种合法形态出现）。

第二种情况决定了它上线后的误拒率，比第一种更重要 —— 一个只会多拒的信号没人敢开。
"""

from __future__ import annotations

import pytest

from documind.refusal import SpecificsCheck, missing_specifics


def check(question: str, *texts: str) -> SpecificsCheck:
    return missing_specifics(question, list(texts))


class _Hit:
    """只有 `.text` 的最小替身：函数不该要求调用方先转成 str。"""

    def __init__(self, text: str) -> None:
        self.text = text


# ---------------------------------------------------------------- rate：要百分数


def test_rate_intent_without_percent_is_missing():
    r = check(
        "公司从税后利润中提取任意公积金的比例是多少？",
        "公司从税后利润中提取法定公积金后，经股东大会决议，还可以从税后利润中提取任意公积金。",
    )
    assert r.intents == ("rate",)
    assert r.missing == ("rate",)
    assert r.should_refuse is True


def test_rate_intent_satisfied_by_percent_sign():
    r = check("任意公积金的比例是多少？", "经股东大会决议，任意公积金提取比例为 5%。")
    assert r.intents == ("rate",)
    assert r.missing == ()
    assert r.should_refuse is False


def test_rate_accepts_chinese_percent_wording():
    assert check("任意公积金的比例是多少？", "提取比例不超过百分之十。").should_refuse is False


def test_rate_not_satisfied_by_plain_count():
    """数字但不是比率（这里 2 个月）不能当税率答案 —— 这正是难例的陷阱。"""
    assert check("公司逾期派发股利的，按什么利率支付利息？", "须在 2 个月内完成股利派发。").should_refuse


# -------------------------------------------------------------- amount：要货币金额


def test_amount_intent_without_currency_is_missing():
    r = check("公司独立董事每年的津贴标准是多少元？", "决定有关董事、监事的报酬事项")
    assert r.intents == ("amount",)
    assert r.should_refuse is True


@pytest.mark.parametrize(
    "text",
    [
        "独立董事津贴为每人每年 12 万元。",
        "独立董事津贴为每人每年 120,000 元。",
        "审计费用共计人民币 80 万元。",
        "年度捐赠额度不超过 500 万元。",
    ],
)
def test_amount_intent_satisfied_by_currency_amount(text: str):
    assert check("独立董事每年的津贴标准是多少元？", text).should_refuse is False


def test_amount_accepts_percent_of_base():
    """\"不超过净资产的 10%\" 也是具体额度，不能因为没写\"元\"就判缺项。"""
    assert check("公司一年最多可以对外捐赠多少钱？", "捐赠金额不超过上一年度净利润的 10%。").should_refuse is False


def test_amount_not_satisfied_by_plain_number():
    assert check("员工持股计划最多可以授予多少股？", "合计持有不超过已发行股份 10% 的回购上限。").missing == ("amount",)


# -------------------------------------------------------------- deadline：要时限


def test_deadline_intent_without_period_is_missing():
    r = check(
        "公司内部审计部门多久向董事会提交一次内部审计报告？",
        "公司实行内部审计制度，配备专职审计人员，对公司财务收支和经济活动进行内部审计监督。",
    )
    assert r.intents == ("deadline",)
    assert r.should_refuse is True


@pytest.mark.parametrize(
    "text",
    [
        "每 6 个月向董事会提交一次内部审计报告。",
        "应当在 10 日内书面答复该股东。",
        "公司将在股东大会结束后 2 个月内实施具体方案。",
        "承担忠实义务的具体期限为六个月。",
        "应当在三十日内作出决定。",
        "每年召开 1 次年度股东大会。",
    ],
)
def test_deadline_intent_satisfied_by_period(text: str):
    assert check("内部审计报告多久提交一次？", text).should_refuse is False


def test_deadline_not_satisfied_by_amount():
    assert check("整改期限是多少天？", "整改保证金为 1000 万元。").should_refuse


# --------------------------------------------------------------- subject：要主体


def test_subject_intent_without_assigning_clause_is_missing():
    r = check("公司内部审计负责人由谁聘任和解聘？", "审计负责人向董事会负责并报告工作。")
    assert r.intents == ("subject",)
    assert r.should_refuse is True


@pytest.mark.parametrize(
    "text",
    [
        "副总经理由总经理提名，由董事会聘任或解聘。",
        "董事长不能履行职务时，由副董事长主持。",
        "公司分立前的债务由分立后的公司承担连带责任。",
        "党组织工作经费预算由公司党委审批。",
        "年度报告由董事会办公室负责编制。",
    ],
)
def test_subject_intent_satisfied_by_assigning_clause(text: str):
    assert check("这个事由谁负责？", text).should_refuse is False


def test_subject_ignores_bare_responsibility_statement():
    """\"向董事会负责并报告\"是职责归属，不是任免主体 —— 别把它当答案。"""
    r = check("内部审计负责人由谁聘任？", "审计负责人向董事会负责并报告工作。")
    assert r.should_refuse is True


# ------------------------------------------------------------- 不该响的情况


@pytest.mark.parametrize(
    "question",
    [
        "董事会开会通知一般有哪些方式？",
        "公司如果要回购股份，一般可以通过什么方式？",
        "股东要查阅章程或者索取资料，需要先向公司提交什么？",
        "董事辞职或者任期结束后，对公司的忠实义务会自动解除吗？",
        "公司是否提供免费午餐或餐补？",
    ],
)
def test_no_value_intent_never_fires(question: str):
    r = check(question, "公司召开董事会的会议通知，以书面、电话、传真、电子邮件等方式通知。")
    assert r.intents == ()
    assert r.should_refuse is False


def test_yes_no_question_about_number_does_not_fire():
    """\"能不能立刻卖掉\"问的是可否，不是要一个数值。"""
    assert check("离职后能不能立刻把股票卖掉？", "离职后半年内不得转让。").should_refuse is False


# --------------------------------------------------------------- 接口细节


def test_accepts_objects_with_text_attribute():
    assert missing_specifics("津贴是多少元？", [_Hit("津贴为每年 12 万元")]).should_refuse is False


def test_accepts_mapping_hits():
    """json 里 dump 出来的 hits 是 dict；取值失败会被静默当成"无文本"，把信号变成"永远该拒"。"""
    assert missing_specifics("津贴是多少元？", [{"text": "津贴为每年 12 万元"}]).should_refuse is False


# --------------------------------------------------- scope：块级 vs 句级（可选）


def test_sentence_scope_ignores_value_from_unrelated_clause():
    """章程里数字/日期到处都是：块级只要"有百分号"就满足，句级要求它和问题问的那句在一起。"""
    text = (
        "公司回购股份的资金总额不超过已发行股份总额的百分之十。"
        "任意公积金的提取比例由股东大会决议确定。"
    )
    q = "任意公积金的提取比例是多少？"
    assert missing_specifics(q, [text], scope="block").should_refuse is False
    assert missing_specifics(q, [text], scope="sentence").should_refuse is True


def test_sentence_scope_finds_value_in_matching_clause():
    text = (
        "公司回购股份的资金总额不超过已发行股份总额的百分之十。"
        "任意公积金的提取比例为百分之五，由股东大会决议确定。"
    )
    assert missing_specifics("任意公积金的提取比例是多少？", [text], scope="sentence").should_refuse is False


def test_sentence_scope_falls_back_to_block_without_matching_sentence():
    """没有一句和问题重合时不做"缺项"结论——那已经不是形态问题，交给分数。"""
    text = "公司回购股份的资金总额不超过已发行股份总额的 10%。"
    assert missing_specifics("企业所得税税率是多少？", [text], scope="sentence").should_refuse is False


def test_unknown_scope_is_rejected():
    with pytest.raises(ValueError):
        missing_specifics("税率是多少？", ["10%"], scope="paragraph")


def test_empty_hits_count_as_missing():
    assert missing_specifics("税率是多少？", []).should_refuse is True


def test_presence_in_any_hit_counts():
    r = check("企业所得税税率是多少？", "无关条款。", "企业所得税税率为 25%。")
    assert r.should_refuse is False


def test_multiple_intents_reports_each():
    r = check("税率是多少？期限是几天？", "公司应当在 10 日内答复。")
    assert set(r.intents) == {"rate", "deadline"}
    assert r.missing == ("rate",)
    assert r.should_refuse is True


def test_report_is_jsonable_and_human_readable():
    r = check("税率是多少？", "无")
    d = r.to_dict()
    assert d["intents"] == ["rate"] and d["missing"] == ["rate"] and d["should_refuse"] is True
    assert "rate" in r.describe()


def test_default_refusal_path_does_not_use_the_signal():
    """不改变现有默认拒答行为：分数够时 `_pre_llm_refusal` 仍返回 None。"""
    from documind.generate import _pre_llm_refusal
    from documind.retrieve import Hit, RetrievalResult

    hit = Hit(
        chunk_id="c1", doc_id="d1", doc_title="章程", text="公司从税后利润中提取任意公积金。",
        page_start=47, char_start=0, page_end=47, char_end=24, section_path="第九章", score=3.0,
    )
    res = RetrievalResult(hits=[hit], contexts=[hit], top_score=3.0)
    assert _pre_llm_refusal(res, 1.5055) is None