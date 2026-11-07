"""DocuMind 侧公开 API 契约（ARCH-REVIEW P0-03 / bidpilot docs/DEPENDENCY.md §4）。

这组断言的作用只有一个：**当下游（bidpilot）被破坏时，documind 自己的测试先红**。
所以断言的对象只能是顶层公开符号，不能 import 子模块，也不能用私有名字——否则
测的就不是契约，而是实现。

改动本文件 = 修改对外契约，须 @hermes 报批。
"""

from __future__ import annotations

import inspect
import re
import tomllib
from pathlib import Path

import pytest

import documind

ROOT = Path(__file__).resolve().parents[1]

# DEPENDENCY.md §1 的五个能力，映射到 7 个顶层符号（C3/C4 是方法，C5 三个名字，C1 含失败类型）
EXPECTED_SYMBOLS = {
    "parse_file",
    "ParsedDocument",
    "Paragraph",
    "tokenize",
    "IndexItem",
    "BM25Index",
    # C1 的失败侧：下游要 catch 它（DEPENDENCY.md §2 第 2 行"无文本层 → 硬失败"）
    "EmptyDocumentError",
}

# bidpilot/tests/conftest.py 与 capabilities.py 实际依赖的字段集（P1-06 的"手搓夹具"就是靠它）
PARAGRAPH_FIELDS = {"text", "page", "start", "end", "level", "section_path"}
PARSED_FIELDS = {"title", "filename", "pages", "paragraphs"}


def test_all_is_exactly_the_contract():
    """__all__ 存在、等于预期清单、且每个名字都真的能 import。"""
    assert hasattr(documind, "__all__"), "__init__.py 必须有 __all__，否则'公开'只是口头约定"
    assert set(documind.__all__) == EXPECTED_SYMBOLS
    # 顺序去重：__all__ 里不能有重复名字（重复会让 `from documind import *` 行为不确定）
    assert len(documind.__all__) == len(set(documind.__all__))
    for name in documind.__all__:
        assert getattr(documind, name, None) is not None, f"{name} 在 __all__ 里但不可导入"


def test_no_private_names_leak_into_public_surface():
    """顶层不应出现下划线开头的名字（那是"私有即公开"的老毛病）。"""
    leaked = [n for n in dir(documind) if n.startswith("_") and not n.startswith("__")]
    assert leaked == [], f"顶层暴露了私有名字：{leaked}"


def test_star_import_is_scoped():
    """`from documind import *` 只带出 __all__，不把子模块一股脑塞进来。"""
    ns: dict = {}
    exec("from documind import *", ns)  # noqa: S102 - 被测行为就是 star import 的作用域
    imported = {k for k in ns if not k.startswith("__")}
    assert imported == EXPECTED_SYMBOLS


def test_version_matches_pyproject_and_is_gated():
    """版本闸门：顶层 __version__ 必须与 pyproject 一致，且在 0.1.x 这条 minor 上。

    bidpilot 声明的是 documind>=0.1,<0.2；这里把"我们还在 0.1.x"变成可执行断言，
    避免有人改 pyproject 版本却忘了 __version__（或反过来）导致下游闸门静默失效。
    """
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    declared = pyproject["project"]["version"]
    assert documind.__version__ == declared, f"__version__={documind.__version__} != pyproject {declared}"
    assert re.fullmatch(r"0\.1\.\d+", declared), f"{declared} 不在 bidpilot 声明的 >=0.1,<0.2 区间内"


def test_paragraph_field_set_is_frozen():
    """字段集是契约：增字段必须有默认值，删/改字段即是 breaking（须升 minor）。"""
    fields = set(documind.Paragraph.__dataclass_fields__)
    assert fields == PARAGRAPH_FIELDS, f"Paragraph 字段集漂移：多了 {fields - PARAGRAPH_FIELDS}，少了 {PARAGRAPH_FIELDS - fields}"
    assert set(documind.ParsedDocument.__dataclass_fields__) == PARSED_FIELDS

    # 必填/默认：前四个必填，level 与 section_path 有默认值（下游位置参数构造依赖这个顺序）
    params = list(inspect.signature(documind.Paragraph).parameters.values())
    required = [p.name for p in params if p.default is inspect.Parameter.empty]
    assert required == ["text", "page", "start", "end"], f"必填字段顺序变了：{required}"


def test_slice_text_is_the_single_source_of_truth():
    """C3 与 C4 的一致性：按 (页, 页内偏移) 切出来的必须是 pages[i] 的同一段原文。"""
    pages = ["第一页正文：年假可以折现。", "第二页正文：婚假十天。", "第三页尾部"]
    doc = documind.ParsedDocument(
        title="契约样本",
        filename="contract.md",
        pages=list(pages),
        paragraphs=[
            documind.Paragraph(text="年假可以折现。", page=1, start=6, end=13),
            documind.Paragraph(text="婚假十天。", page=2, start=6, end=11),
        ],
    )

    assert doc.page_count == 3
    assert doc.slice_text(1, 0, 1, len(pages[0])) == pages[0]
    assert doc.slice_text(1, 6, 1, 13) == "年假可以折现。"
    # 跨页：首页从 char_start 到页尾，末页从页首到 char_end（左闭右开），中间页整页，用 \n 拼接
    assert doc.slice_text(1, 6, 2, 5) == "年假可以折现。\n第二页正文"
    # 越界不抛异常（前端高亮依赖这条：偏移超长时展示整页而不是 500）
    assert doc.slice_text(9, 0, 9, 10) == ""
    assert doc.slice_text(1, 0, 1, 9999) == pages[0]

    # 段落偏移必须落在页文本内，否则 quote 高亮会错位
    for p in doc.paragraphs:
        assert doc.slice_text(p.page, p.start, p.page, p.end) == p.text


def test_bm25_search_return_shape():
    """C5：返回 [(str, float)]，按分数降序；空索引返回 []；allowed 在打分前生效。"""
    from documind import BM25Index, IndexItem

    items = [
        IndexItem(id="a", text="员工享有年假，年假可以折现吗", doc_id="doc-1", strategy="structure"),
        IndexItem(id="b", text="婚假十天，产假按国家规定", doc_id="doc-2", strategy="structure"),
    ]
    idx = BM25Index(items)

    hits = idx.search("年假折现", 3)
    assert isinstance(hits, list)
    assert all(isinstance(i, str) and isinstance(s, float) for i, s in hits), hits
    assert hits, "命中了却返回空，检索契约失效"
    assert hits[0][0] == "a"
    assert [s for _, s in hits] == sorted((s for _, s in hits), reverse=True)
    assert len(hits) <= 3

    # top_n 生效
    assert len(idx.search("年假 婚假", 1)) == 1
    # 空索引：不是异常，是空列表（下游 capabilities.py 依赖这条区分"无命中"）
    assert BM25Index([]).search("年假", 3) == []
    # ACL：无权文档的分数直接不存在（不是"生成后再隐藏"）
    assert [i for i, _ in idx.search("婚假", 3, allowed={"doc-2"})] == ["b"]
    assert idx.search("婚假", 3, allowed={"doc-1"}) == []
    assert idx.search("年假", 3, allowed=set()) == []


def test_parse_file_rejects_unsupported_suffix(tmp_path: Path):
    """降级表第 1 行可执行化：不认识的格式必须抛 ValueError，而不是返回空文档。"""
    bad = tmp_path / "x.xyz"
    bad.write_text("任意内容", encoding="utf-8")
    with pytest.raises(ValueError):
        documind.parse_file(bad)


def test_parse_file_returns_public_document_type(tmp_path: Path):
    """parse_file 的返回类型必须是公开的 ParsedDocument，否则下游只能靠鸭子类型。"""
    md = tmp_path / "contract.md"
    md.write_text("# 第一章 总则\n\n第一条 本制度适用于全体员工。\n", encoding="utf-8")
    doc = documind.parse_file(md)
    assert isinstance(doc, documind.ParsedDocument)
    assert all(isinstance(p, documind.Paragraph) for p in doc.paragraphs)
    assert doc.paragraphs, "正常 md 应解析出段落"
    first = doc.paragraphs[0]
    assert doc.slice_text(first.page, first.start, first.page, first.end) == first.text


def test_empty_document_error_is_part_of_the_public_surface(tmp_path: Path):
    """C1 的失败侧也是契约：下游必须能从顶层 catch 到同一个类型（DEPENDENCY.md §2 第 2 行）。

    这条曾经差点漏掉：sections.py 抓 EmptyDocumentError 做"无文本层=硬失败"，
    如果它只存在于 documind.parsing，下游就只能 import 子模块——边界又破了。
    """
    assert issubclass(documind.EmptyDocumentError, ValueError)
    empty = tmp_path / "empty.md"
    empty.write_text("   \n\n  \n", encoding="utf-8")
    with pytest.raises(documind.EmptyDocumentError) as exc:
        documind.parse_file(empty)
    assert "OCR" in str(exc.value)