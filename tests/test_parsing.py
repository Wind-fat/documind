"""解析层测试：结构识别、页码、字符偏移、目录剔除。"""

from __future__ import annotations

from pathlib import Path

from documind.parsing import _group_page, parse_file


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