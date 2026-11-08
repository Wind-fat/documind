"""缺项信号：不靠分数高低，判「问题要的那个值，检索结果里到底有没有」。

背景（`evals/README.md`「间隔：0.176 → -4.2269」）：同域难例扩到 28 条后，15/28 的
`top_score` 高于最容易的可答题，**任何单阈值都必须牺牲一边**。分数回答的是
「问法和条款像不像」，而这类题的病根是「条款在、值不在」——两件事不是一回事。

所以这里加一个正交信号：先认问题**要什么形态的值**（比率 / 金额 / 时限 / 主体），
再去前 N 条块里找**那个形态**。找不到就判「该拒」。

设计上的三条边界：
1. **纯离线、纯函数**：不调模型、不读库、不碰 settings；`hits` 只要求 `.text` 或 `str`。
   所以它能被评测脚本、单元测试、以及将来的 `/query` 以同一份代码调用。
2. **不改变默认拒答行为**：`generate` 的私有 `_pre_llm_refusal` 不看这个信号（有测试钉住）。
   接入与否是主控的决定，不是这个模块的副作用。
3. **只认形态，不认语义**：`10%` 出现在「法定公积金」句里也算「有百分数」。这是刻意的
   保守选择——它会把 `hardneg-003`（企业所得税率，同页有 10%）漏掉，但不会凭空
   断言「文档里没有」。漏掉的代价是拒答率上不去，不是答错。

已知局限（评测时要一起报，不要藏）：
- 日期里的年份（`2023 年`）会被时限形态当成"有期限"；
- 条款写「及时」「视情节」这类无数字时限时，信号会判缺项——方向正确，但同域难例里
  这类题本来就是要拒的，不算误差；
- 主体形态靠动词表（审批/决定/聘任/…）与「由/经」句式，`向…负责` 这种职责归属
  被显式排除（见 `_RESPONSIBILITY_RE`）；
- 中文数字只覆盖「三十日/六个月」这类常见写法，不解析「两个半月」。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final

# 全角数字/百分号/逗号：PDF 抽出来的正文里三者都出现过
_FULLWIDTH = str.maketrans("０１２３４５６７８９％．，", "0123456789%.,")

# 问题要什么：四类意图。词表刻意宽——漏认意图等于信号不生效，多认一个词的代价
# 只是一次「去找形态」的空转（找不到才判缺项），所以宁可宽。
_RATE_RE: Final = re.compile(r"税率|费率|利率|比率|比例|占比|百分点|折扣|提成")
_AMOUNT_RE: Final = re.compile(
    r"多少钱|多少元|多少万元|多少亿|金额|额度|数额|上限|薪酬|津贴|奖金|年薪|预算"
    r"|费用|价格|收费标准|标准是多少|每股收益|净值"
)
_COUNT_RE: Final = re.compile(r"多少\s*(?:股|份|人|家|名|个)|股数|股份总数")
_DEADLINE_RE: Final = re.compile(
    r"几日内|几日|多少天|多少日|多久|多长时间|多少时间|期限|时限|频次|多久一次"
    r"|提前(?:多久|多少)|什么时间|多久内|几月"
)
# 主体意图：问「谁来做这件事」。「谁负责/找谁/向谁」都算，但「是否是/能不能」不算。
_SUBJECT_RE: Final = re.compile(
    r"由谁|谁来|找谁|向谁|何人|哪个部门|哪一个部门|哪一位|哪位|"
    r"谁(?:负责|审批|批准|决定|聘任|解聘|提名|承担|主持|作出|编制|组织|召集|答复|报告)"
)

# 「有值」的形态。四类各一个正则，互不代偿——这正是单阈值做不到的事。
_PERCENT_RE: Final = re.compile(r"%|‰|百分之|千分之")
_MONEY_RE: Final = re.compile(r"\d[\d,.]*\s*(?:万元|亿元|元|万|亿|块)")
_COUNT_FORM_RE: Final = re.compile(r"\d[\d,.]*\s*(?:股|份|人|家|名|个)")
_PERIOD_RE: Final = re.compile(
    r"\d+\s*(?:个)?\s*(?:工作日|日|天|个月|月|年|小时|分钟|周|届)"
    r"|[一二三四五六七八九十百零两]+\s*(?:个)?\s*(?:工作日|日|天|个月|月|年|小时|周)"
    r"|每\s*(?:\d+|[一二三四五六七八九十]+)\s*(?:个)?\s*(?:日|天|月|年|周|季度)"
    r"|每(?:年|月|周|季度|日|天)"
    r"|当日|次日|届时|即时"
)
# 主体形态：必须是「由/经 X 做 Y」或「机构 + 动作」的指派句。
_ASSIGN_RE: Final = re.compile(
    r"(?:由|经)[^，。；]{0,20}"
    r"(?:审批|批准|决定|聘任|解聘|提名|选举|委派|任命|承担|主持|组织|编制|负责|提出|召集|审议|核准|答复)"
    r"|(?:股东大会|董事会|监事会|总经理|董事长|副董事长|党委|经理层|高级管理人员|董事会秘书"
    r"|审计委员会|薪酬与考核委员会|提名委员会|审计部|财务部|董事会办公室|人力资源部|公司)"
    r"[^，。；]{0,8}"
    r"(?:审批|批准|决定|聘任|解聘|提名|选举|承担|主持|组织|编制|负责|审议|召集)"
)
# 「向董事会负责并报告工作」是职责归属，不是任免主体；先摘掉这个句式再匹配主体形态，
# 否则「审计负责人向董事会负责」会被当成"有主体"——那正是 hardneg-020 的陷阱。
_RESPONSIBILITY_RE: Final = re.compile(r"向[^，。；]{0,15}(?:负责|报告)[^，。；]{0,10}")

_INTENT_ORDER: Final = ("rate", "amount", "deadline", "subject")
_SCOPES: Final = ("block", "sentence")
_SENT_SPLIT: Final = re.compile(r"[。；！？\n]")
# 问句里的疑问词/虚词不参与"哪句最像问题"的打分，否则每句都因为"公司/多少"而平手
_QUESTION_WORDS: Final = re.compile(r"[\s，。？！、；：]|多少|几|什么|怎么|哪个|哪一位|是否|吗|呢")


@dataclass(frozen=True, slots=True)
class SpecificsCheck:
    """一次缺项判定的完整结果——足以回答「它凭什么判该拒」。"""

    intents: tuple[str, ...]
    missing: tuple[str, ...]
    found: tuple[str, ...]

    @property
    def should_refuse(self) -> bool:
        """问题要值、且前 N 条里一个对应形态都没有。"""
        return bool(self.missing)

    def to_dict(self) -> dict:
        return {
            "intents": list(self.intents),
            "missing": list(self.missing),
            "found": list(self.found),
            "should_refuse": self.should_refuse,
        }

    def describe(self) -> str:
        if not self.intents:
            return "问题没有要具体值，信号不适用"
        if not self.missing:
            return f"要 {','.join(self.intents)}；前 N 条里都有对应形态"
        return f"要 {','.join(self.intents)}；前 N 条里缺 {','.join(self.missing)}"


def detect_intents(query: str) -> tuple[str, ...]:
    """问题在要哪几类具体值（顺序固定，便于比对与测试）。"""
    q = query.translate(_FULLWIDTH)
    hits = set()
    if _RATE_RE.search(q):
        hits.add("rate")
    if _AMOUNT_RE.search(q) or _COUNT_RE.search(q):
        hits.add("amount")
    if _DEADLINE_RE.search(q):
        hits.add("deadline")
    if _SUBJECT_RE.search(q):
        hits.add("subject")
    return tuple(i for i in _INTENT_ORDER if i in hits)


def _hit_texts(hits) -> list[str]:
    parts: list[str] = []
    for h in hits:
        if isinstance(h, str):
            text = h
        elif isinstance(h, dict):
            # `eval_retrieval.py --dump-hits` 落盘的就是 dict：取值失败会被静默当成空文本，
            # 那会把「拿不到文本」变成「永远缺项」——这条分支必须显式存在。
            text = h.get("text", "")
        else:
            text = getattr(h, "text", "")
        if text:
            parts.append(text.translate(_FULLWIDTH))
    return parts


def _bigrams(text: str) -> set[str]:
    t = re.sub(r"\s+", "", text)
    return {t[i : i + 2] for i in range(max(len(t) - 1, 0))}


def _narrow_to_matching_clause(query: str, texts: list[str]) -> str:
    """只留「和问题用词最重合的那一条块里，最重合的那几句」。

    动机（这是本模块唯一一处启发式）：章程里每一页都有百分比、日期、金额，
    块级形态检查因此几乎永远成立（实测 28 条难例 0 命中）。把形态检查限制到
    「最像问题的那一条」里，才可能问出「这一条有没有给值」，而不是
    「这五块里有没有数字」——后者在 28674 字的章程上等于恒真。

    打分是朴素的 query 字符二元组重合数，不是语义。并列全留（不按检索顺序取第一条，
    否则结果依赖 rerank 的先后）。所有块都没有任何重合词时退回整块：那时没有任何
    依据说"缺项"，宁可放过（交给分数），也不要凭空白拒。
    """
    grams = _bigrams(_QUESTION_WORDS.sub("", query))
    per: list[tuple[int, list[tuple[int, str]]]] = []
    for t in texts:
        sents = [s for s in _SENT_SPLIT.split(t) if s.strip()]
        if not sents:
            continue
        scored = [(len(grams & _bigrams(s)), s) for s in sents]
        per.append((max(n for n, _ in scored), scored))
    if not per:
        return ""
    best = max(n for n, _ in per)
    if best == 0:
        return "\n".join(texts)
    return "\n".join(s for n, scored in per if n == best for m, s in scored if m == best)


def _has_rate(text: str) -> bool:
    return _PERCENT_RE.search(text) is not None


def _has_amount(text: str, count_question: bool) -> bool:
    if count_question:
        # 「多少股/多少份」要的是计数，不能拿百分数或金额顶替
        return _COUNT_FORM_RE.search(text) is not None
    # 金额类：货币金额，或「按…的 10%」这种相对额度（也是具体值）
    return _MONEY_RE.search(text) is not None or _PERCENT_RE.search(text) is not None


def _has_deadline(text: str) -> bool:
    return _PERIOD_RE.search(text) is not None


def _has_subject(text: str) -> bool:
    return _ASSIGN_RE.search(_RESPONSIBILITY_RE.sub("", text)) is not None


def missing_specifics(query: str, hits, top_n: int | None = None, scope: str = "block") -> SpecificsCheck:
    """问题要具体值、而前 `top_n` 条块里没有对应形态 → `should_refuse`。

    `hits` 可以是 `str`、`Hit`、dict（`--dump-hits` 的落盘格式）、或任何带 `.text` 的对象
    （`retrieve.RetrievalResult.hits` 直接可用）。
    `top_n=None` 表示看传进来的全部；给 5 就是「和模型看到的 top-5 上下文对齐」。

    `scope`：
    - `"block"`（默认，与需求口径一致）：形态出现在**块里任何位置**就算有；
    - `"sentence"`：形态必须出现在**和问题用词最重合的那一条条款**里。章程类文本里数字密度太高，
      块级检查几乎永远成立，收紧到单条才有机会看出"值不在这一条"（详见 `_narrow_to_matching_clause`）。
    """
    if scope not in _SCOPES:
        raise ValueError(f"未知 scope={scope!r}：可选 {_SCOPES}")
    intents = detect_intents(query)
    if not intents:
        return SpecificsCheck(intents=(), missing=(), found=())

    selected = list(hits or [])[:top_n] if top_n is not None else list(hits or [])
    texts = _hit_texts(selected)
    text = "\n".join(texts) if scope == "block" else _narrow_to_matching_clause(query, texts)
    count_question = _COUNT_RE.search(query.translate(_FULLWIDTH)) is not None

    checks = {
        "rate": _has_rate,
        "amount": lambda t: _has_amount(t, count_question),
        "deadline": _has_deadline,
        "subject": _has_subject,
    }
    found = tuple(i for i in intents if checks[i](text))
    missing = tuple(i for i in intents if i not in found)
    return SpecificsCheck(intents=intents, missing=missing, found=found)