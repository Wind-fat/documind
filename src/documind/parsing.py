"""文档解析：PDF / DOCX / MD → 带页码与字符偏移的结构化段落。

设计取舍（见 docs/adr/0001-parser.md）：
- 默认解析器用 pypdf，而不是 docling。理由：我们要的是**精确到字符的 span 偏移**做原文高亮，
  pypdf 的 extract_text 与页文本一一对应、偏移可控；docling 输出的是结构化 markdown，
  回映射到页/字符很麻烦，且带 torch 依赖（本机装不下）。
- docling 适配器（DoclingParser）**未接线、禁止使用**：它的分页是按字数伪造的，
  真接上会让 citation 的高亮指向错误页且不报错。见 docs/adr/0001-parser.md。
- 中文制度文本的天然切分单位是"第 X 条"，解析阶段就把 条/章/节 识别成结构，
  切块阶段才能做"结构感知"而不是按字数硬切。
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

CN_NUM = "一二三四五六七八九十百零〇两"
RE_CHAPTER = re.compile(rf"^第([{CN_NUM}]+)章\s*(.*)$")
RE_SECTION = re.compile(rf"^第([{CN_NUM}]+)节\s*(.*)$")
RE_ARTICLE = re.compile(rf"^第([{CN_NUM}]+)条")
RE_CLAUSE = re.compile(rf"^第([{CN_NUM}]+)款")
RE_APPENDIX = re.compile(rf"^第([{CN_NUM}]+)篇\s*(.*)$")
RE_TERMINAL = re.compile(r"[。；：！？!?;:]$")
RE_MD_HEADING = re.compile(r"^(#{1,6})\s*(.+)$")


class EmptyDocumentError(ValueError):
    """解析结果里没有任何可检索文本（多为扫描件 / 图片型 PDF，需要先做 OCR）。

    继承 ValueError 是为了让已有的 `except ValueError`（不支持的后缀）调用方不用改；
    但调用方应当**优先**按本类型分支，给出可读提示而不是当成一般参数错误。
    """


@dataclass(slots=True)
class Paragraph:
    """一段正文，带页内字符偏移。offset 相对于 `ParsedDocument.pages[page-1]`。"""

    text: str
    page: int
    start: int
    end: int
    level: int = 0  # 0=正文/条, 1=章, 2=节
    section_path: str = ""


@dataclass(slots=True)
class ParsedDocument:
    title: str
    filename: str
    pages: list[str] = field(default_factory=list)
    paragraphs: list[Paragraph] = field(default_factory=list)

    @property
    def page_count(self) -> int:
        return len(self.pages)

    @property
    def text(self) -> str:
        return "\n".join(self.pages)

    def slice_text(self, page_start: int, char_start: int, page_end: int, char_end: int) -> str:
        """按 (页, 页内字符偏移) 取出原文。块的 text 一律由它生成，
        这样"块文本 === 原文切片"是构造保证的，前端高亮不会错位。"""
        if page_start == page_end:
            page = self.pages[page_start - 1] if 0 < page_start <= len(self.pages) else ""
            return page[char_start:char_end]
        parts = []
        for page_no in range(page_start, page_end + 1):
            if not (0 < page_no <= len(self.pages)):
                continue
            page = self.pages[page_no - 1]
            if page_no == page_start:
                parts.append(page[char_start:])
            elif page_no == page_end:
                parts.append(page[:char_end])
            else:
                parts.append(page)
        return "\n".join(p for p in parts if p)

    @property
    def is_empty(self) -> bool:
        """页文本全为空 / 没有段落 → 这份文件没有可检索的文本层。"""
        if not self.paragraphs:
            return True
        return not any(p.strip() for p in self.pages)

    def require_text(self) -> ParsedDocument:
        """空文档出口守卫（ADR-0001 承诺的\"显式报未提取到文本\"）。

        放在解析出口而不是切块阶段，是为了让空交付物在**落库之前**就失败：
        一旦 Document/Chunk 被 add 进 session，调用方很容易顺手 commit 出一个空知识库。
        """
        if self.is_empty:
            raise EmptyDocumentError(
                f"未提取到文本，需要 OCR：{self.filename} 共 {len(self.pages)} 页，未识别出任何段落"
                "（若为扫描件/图片型 PDF，请先 OCR 再入库）"
            )
        return self

    def outline(self) -> list[dict]:
        """章节树（给 BidPilot 的"章节树"复用，也给 /documents/{id}/outline 用）。"""
        root: list[dict] = []
        for p in self.paragraphs:
            if p.level <= 0:
                continue
            node = {"title": p.text[:80], "page": p.page, "level": p.level, "children": []}
            if p.level == 1 or not root:
                root.append(node)
            else:
                root[-1]["children"].append(node)
        return root


def _strip_md(line: str) -> tuple[str, int]:
    """Markdown 标题 → (去掉 # 的文本, 层级)。非标题返回 (原文, 0)。"""
    m = RE_MD_HEADING.match(line)
    if m:
        return m.group(2).strip(), min(2, len(m.group(1)))
    return line, 0


def _is_marker(line: str) -> tuple[int, str]:
    """返回 (level, kind)。level 1=章 2=节 0=条。"""
    text, md_level = _strip_md(line)
    if RE_CHAPTER.match(text):
        return 1, "chapter"
    if RE_SECTION.match(text):
        return 2, "section"
    if RE_APPENDIX.match(text):
        return 1, "appendix"
    if RE_ARTICLE.match(text):
        return 0, "article"
    if md_level:
        return md_level, "chapter" if md_level == 1 else "section"
    return -1, ""


def _clean(line: str) -> str:
    line = line.replace("\u3000", " ").replace("\xa0", " ")
    line = re.sub(r"[ \t]+", " ", line)
    return line.strip()


def _is_toc(line: str) -> bool:
    """目录行：`第四章 股份 ......... 7`。它们会污染章节路径，解析阶段就丢掉。"""
    if re.search(r"\.{4,}|…{2,}|·{4,}", line):
        return True
    return False


def _join(a: str, b: str) -> str:
    """拼接两行中文文本：中文之间不加空格，英文/数字之间加。"""
    if not a:
        return b
    if not b:
        return a
    if re.search(r"[A-Za-z0-9,;:]$", a) and re.match(r"[A-Za-z0-9]", b):
        return a + " " + b
    return a + b


def _group_page(lines: list[str]) -> list[tuple[str, int, str]]:
    """把一页的原始行合并成段落，返回 [(文本, level, kind)]。

    章程类文本以"第 X 条"为硬边界；不在边界上的行并入当前条（中文 PDF 换行不代表分段）。
    没有条标记的文本退化为"行末终止符即分段"。
    """
    units: list[tuple[str, int, str]] = []
    buf = ""
    buf_level = 0
    buf_kind = ""
    has_markers = any(_is_marker(_clean(line))[0] >= 0 for line in lines)

    def flush() -> None:
        nonlocal buf, buf_level, buf_kind
        if buf.strip():
            units.append((buf.strip(), buf_level, buf_kind))
        buf, buf_level, buf_kind = "", 0, ""

    i = 0
    while i < len(lines):
        line = _clean(lines[i])
        i += 1
        if not line:
            continue
        if _is_toc(line):
            continue
        level, kind = _is_marker(line)
        if level in (1, 2):  # 章 / 节标题：自成一段，不吸收正文
            flush()
            title, _md = _strip_md(line)
            # 标题可能被 PDF 拆成两行："第三章" + "股份"
            if len(title) <= 8 and i < len(lines):
                nxt = _clean(lines[i])
                if nxt and _is_marker(nxt)[0] < 0 and len(nxt) <= 20:
                    title = _join(title, nxt)
                    i += 1
            units.append((title, level, kind))
            continue
        if level == 0:  # 条：开新段，后续行并入本条直到下一个标记
            flush()
            buf, buf_level, buf_kind = _strip_md(line)[0], 0, kind
            continue
        if has_markers:
            buf = _join(buf, line)
        else:
            if buf and RE_TERMINAL.search(buf) and not RE_TERMINAL.search(line):
                flush()
            buf = _join(buf, line) if buf else line
            if RE_TERMINAL.search(line) and len(buf) > 200:
                flush()
    flush()
    return units


def _assemble(pages: list[list[str]], title: str, filename: str) -> ParsedDocument:
    doc = ParsedDocument(title=title, filename=filename)
    section_path = ""
    chapter = ""
    for page_no, lines in enumerate(pages, start=1):
        page_text_parts: list[str] = []
        cursor = 0
        for text, level, kind in _group_page(lines):
            if kind == "chapter":
                chapter = text
                section_path = text
            elif kind == "section":
                section_path = f"{chapter} > {text}" if chapter else text
            if page_text_parts:
                cursor += 1  # '\n'
            doc.paragraphs.append(
                Paragraph(
                    text=text,
                    page=page_no,
                    start=cursor,
                    end=cursor + len(text),
                    level=level,
                    section_path=section_path,
                )
            )
            page_text_parts.append(text)
            cursor += len(text)
        doc.pages.append("\n".join(page_text_parts))
    doc.title = title or (doc.pages[0][:40] if doc.pages else filename)
    return doc


class Parser:
    name = "base"

    def parse(self, path: Path) -> ParsedDocument:  # pragma: no cover - interface
        raise NotImplementedError


class PdfParser(Parser):
    name = "pypdf"

    def parse(self, path: Path) -> ParsedDocument:
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        pages = []
        for page in reader.pages:
            raw = page.extract_text() or ""
            pages.append(raw.splitlines())
        title = ""
        meta = getattr(reader, "metadata", None)
        if meta and meta.title:
            title = str(meta.title).strip()
        doc = _assemble(pages, title, path.name)
        # PDF 元数据里的标题经常是"公司""章程"这种没有信息量的值，
        # 且它会被写进每一张证据卡，所以宁可回退到首页正文。
        if len(doc.title.strip()) < 8 or doc.title.strip() in {"公司", "章程", "Untitled", "无标题"}:
            cand = ""
            for p in doc.paragraphs[:6]:
                t = p.text.strip()
                if re.search(r"(股份有限公司|有限责任公司|有限公司)", t):
                    cand = re.split(r"[（(]", t)[0].strip()
                    break
            if not cand:
                cand = re.split(r"[（(]", doc.paragraphs[0].text.strip())[0] if doc.paragraphs else ""
            cand = cand.strip()[:40]
            if len(cand) >= 4:
                doc.title = cand
        return doc


class DocxParser(Parser):
    name = "python-docx"

    def parse(self, path: Path) -> ParsedDocument:
        import docx

        d = docx.Document(str(path))
        # DOCX 没有页的概念（分页由渲染器决定），整篇当作 1 页，另用标题层级做结构
        lines: list[str] = []
        for p in d.paragraphs:
            txt = _clean(p.text)
            if not txt:
                continue
            style_obj = p.style
            style = ((style_obj.name if style_obj is not None else "") or "").lower()
            if style.startswith("heading"):
                level = min(2, int(style.replace("heading", "").strip() or 1))
                prefix = "#" * level
                lines.append(f"{prefix} {txt}")
            else:
                lines.append(txt)
        return _assemble([lines], "", path.name)


class MarkdownParser(Parser):
    name = "markdown"

    def parse(self, path: Path) -> ParsedDocument:
        raw = path.read_text(encoding="utf-8", errors="ignore")
        pages: list[list[str]] = [[]]
        for block in raw.split("\n\n"):
            block = block.strip()
            if not block:
                continue
            if len("\n".join(pages[-1])) > 4000:  # 模拟分页，便于演示 span
                pages.append([])
            pages[-1].extend(block.splitlines())
        return _assemble(pages, "", path.name)


class DoclingParser(Parser):
    """未接线，禁止使用。

    它**不在** `_PARSERS` 里，`DOCUMIND_PARSER=docling` 会在 Settings 校验阶段直接报错（P1-02），
    所以没有任何一条运行路径能走到这里。留着这个类只为记录"接 docling 要解决什么":

    - 分页是**伪造**的（按 markdown 每 3000 字切一页），真接上会让 `Hit.page_start/page_end`
      指向错误页，citation 高亮错位且不报错；
    - 正确接法：取 docling 的 provenance（段级 page_no + bbox），映射成
      `Paragraph(page, start, end)`，并补一条 `slice_text` 的等价断言（现有契约见
      tests/test_parsing.py::test_offsets_match_page_text）；
    - 拿到真 provenance 之前，不许把它加进注册表，也不许在文档里写成"装上就换"。
    """

    name = "docling"

    def parse(self, path: Path) -> ParsedDocument:  # pragma: no cover - 需要可选依赖
        from docling.document_converter import DocumentConverter  # type: ignore

        conv = DocumentConverter()
        md = conv.convert(str(path)).document.export_to_markdown()
        pages: list[list[str]] = [[]]
        for block in md.split("\n\n"):
            if not block.strip():
                continue
            if len("\n".join(pages[-1])) > 3000:
                pages.append([])
            pages[-1].extend(block.splitlines())
        return _assemble(pages, "", path.name)


_PARSERS: dict[str, Parser] = {
    ".pdf": PdfParser(),
    ".docx": DocxParser(),
    ".md": MarkdownParser(),
    ".markdown": MarkdownParser(),
    ".txt": MarkdownParser(),
}


def get_parser(filename: str | Path) -> Parser:
    suffix = Path(filename).suffix.lower()
    if suffix not in _PARSERS:
        raise ValueError(f"不支持的文档类型: {suffix}（支持 pdf/docx/md/txt）")
    return _PARSERS[suffix]


def parse_file(path: str | Path) -> ParsedDocument:
    """解析文件；结果为空（扫描件 / 空文件）时抛 EmptyDocumentError，不返回空壳。"""
    p = Path(path)
    return get_parser(p.name).parse(p).require_text()


def iter_paragraph_text(doc: ParsedDocument) -> Iterable[str]:
    for para in doc.paragraphs:
        yield para.text