"""解析层测试：结构识别、页码、字符偏移、目录剔除。"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from documind.config import Settings
from documind.parsing import _PARSERS, DoclingParser, _group_page, get_parser, parse_file


def test_structure_and_offsets(corpus_md: Path):
    doc = parse_file(corpus_md)
    assert doc.page_count >= 1
    chapters = [p.text for p in doc.paragraphs if p.level == 1 and p.text.startswith("第")]
    assert chapters[:2] == ["第一章 总则", "第二章 休假"]

    articles = [p for p in doc.paragraphs if p.text.startswith("第") and "条" in p.text[:6]]
    assert len(articles) >= 6
    # 章节路径要跟着走：第二章下面的条，前缀必须是第二章
    second = [a for a in articles if "年假" in a.text][0]
    assert "第二章" in second.section_path


def test_offsets_match_page_text(corpus_md: Path):
    doc = parse_file(corpus_md)
    for p in doc.paragraphs:
        page = doc.pages[p.page - 1]
        assert page[p.start : p.end] == p.text, f"偏移与页文本不一致: {p.text[:20]}"


def test_toc_lines_dropped():
    lines = [
        "第一章 总则 ......................... 1",
        "第二章 休假 ......................... 5",
        "第一章 总则",
        "第一条 为规范公司管理，制定本制度。",
    ]
    units = _group_page(lines)
    texts = [u[0] for u in units]
    assert not any("....." in t for t in texts)
    assert "第一章 总则" in texts


def test_article_lines_merged_across_pdf_linebreaks():
    """PDF 里一个条款常被拆成多行；解析必须合并，不能一行一个 chunk。"""
    lines = ["第一条 员工每年享有带薪年假五天，", "工作满十年者享有十天。", "第二条 病假需提供证明。"]
    units = _group_page(lines)
    assert units[0][0] == "第一条 员工每年享有带薪年假五天，工作满十年者享有十天。"


def test_unsupported_suffix(tmp_path: Path):
    p = tmp_path / "a.xyz"
    p.write_text("x")
    try:
        parse_file(p)
    except ValueError as exc:
        assert "不支持" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("应当抛出 ValueError")

def test_docling_parser_is_explicitly_unreachable():
    """P1-02：不许再以"装上就换"的面貌存在——类注释首行就写死，注册表里也没有它。"""
    first_line = (DoclingParser.__doc__ or "").strip().splitlines()[0]
    assert first_line == "未接线，禁止使用。"
    assert not any(isinstance(p, DoclingParser) for p in _PARSERS.values())
    for name in ("a.pdf", "a.docx", "a.md", "a.txt"):
        assert not isinstance(get_parser(name), DoclingParser)


def test_parser_setting_rejects_docling(monkeypatch):
    """设了 DOCUMIND_PARSER=docling 不能静默无效，必须在设置校验阶段报错。"""
    monkeypatch.setenv("DOCUMIND_PARSER", "docling")
    with pytest.raises(ValidationError) as ei:
        Settings()
    assert "未接线" in str(ei.value)


def test_parser_setting_is_auto_by_default(monkeypatch):
    monkeypatch.delenv("DOCUMIND_PARSER", raising=False)
    assert Settings().parser == "auto"
    # 大小写不影响判断：DOCLING 同样被拒绝，而不是被静默忽略
    with pytest.raises(ValidationError):
        Settings(parser="DOCLING")


def test_parser_setting_rejects_unknown_values():
    """只有 auto 一档：写 pypdf/docling 之外的值也不能"看起来生效"（真实选择按后缀）。"""
    with pytest.raises(ValidationError) as ei:
        Settings(parser="pypdf")
    assert "只支持 auto" in str(ei.value)
