"""切块测试：四种策略的块数、字符上限，以及最关键的 span 不变量。"""

from __future__ import annotations

from pathlib import Path

import pytest

from documind.chunking import STRATEGIES, chunk_document
from documind.embed import HashEmbedder
from documind.parsing import parse_file


@pytest.fixture()
def doc(corpus_md: Path):
    return parse_file(corpus_md)


def test_all_strategies_produce_chunks(doc):
    emb = HashEmbedder()
    for strategy in STRATEGIES:
        children, parents = chunk_document(doc, strategy, emb.embed)
        assert children, f"{strategy} 没有产出任何块"
        assert all(c.text.strip() for c in children)
        if strategy == "parent_child":
            assert parents and all(c.parent_key for c in children)


@pytest.mark.parametrize("strategy", ["fixed", "semantic", "structure"])
def test_span_slices_page_text(doc, strategy):
    """块的 (page, char_start, char_end) 必须真的能从页文本里切出这段文字。

    这是"点回原文高亮"的地基：偏移错了，前端就高亮错行。
    """
    emb = HashEmbedder()
    children, _ = chunk_document(doc, strategy, emb.embed)
    checked = 0
    for c in children:
        if c.page_start != c.page_end:
            continue
        page = doc.pages[c.page_start - 1]
        sliced = page[c.char_start : c.char_end]
        assert sliced.strip() == c.text.strip(), f"{strategy} span 不匹配: {c.text[:30]!r} vs {sliced[:30]!r}"
        checked += 1
    assert checked > 0


def test_structure_respects_max_chars(doc):
    children, _ = chunk_document(doc, "structure", None)
    assert all(len(c.text) <= 1000 for c in children)


def test_article_is_atomic(doc):
    """结构切块不能把一个条款劈成两半。"""
    children, _ = chunk_document(doc, "structure", None)
    whole = [c for c in children if "年假五天" in c.text]
    assert whole and "十天" in whole[0].text


def test_semantic_needs_embedder(doc):
    with pytest.raises(ValueError):
        chunk_document(doc, "semantic", None)


def test_unknown_strategy(doc):
    with pytest.raises(ValueError):
        chunk_document(doc, "magic", None)