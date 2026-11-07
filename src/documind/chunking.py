"""四种切块策略，同一份语料上可对比（消融实验的对象）。

- fixed           512 字滑窗 + 50 字重叠（基线，最常见的"拍脑袋"做法）
- semantic        按句子切，相邻句向量相似度低于阈值就断开，再按上限合并
- structure       按"第 X 章 / 第 X 节 / 第 X 条"结构合并（本语料最自然）
- parent_child    小块（~200 字）用于召回，父块（~900 字）喂给模型

所有块的 text 都由 `ParsedDocument.slice_text(span)` 生成——"块文本 == 原文切片"是
构造保证的，前端拿 span 直接高亮不会错位。
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from .parsing import Paragraph, ParsedDocument

EmbedFn = Callable[[Sequence[str]], np.ndarray]


@dataclass(slots=True)
class Chunk:
    text: str
    page_start: int
    char_start: int
    page_end: int
    char_end: int
    section_path: str = ""
    ordinal: int = 0
    parent_key: str | None = None
    is_parent: bool = False
    key: str = ""

    def span(self) -> dict:
        return {
            "page_start": self.page_start,
            "char_start": self.char_start,
            "page_end": self.page_end,
            "char_end": self.char_end,
        }


RE_SENT = re.compile(r"[^。！？!?；;\n]+[。！？!?；;]?")


def _make(doc: ParsedDocument, paras: Sequence[Paragraph], section_path: str | None = None) -> Chunk:
    assert paras
    first, last = paras[0], paras[-1]
    return Chunk(
        text=doc.slice_text(first.page, first.start, last.page, last.end),
        page_start=first.page,
        char_start=first.start,
        page_end=last.page,
        char_end=last.end,
        section_path=section_path if section_path is not None else (first.section_path or ""),
    )


def _sentences(paras: Sequence[Paragraph]) -> list[Paragraph]:
    """把段落拆成句子级 Paragraph（保留偏移，偏移仍是页内偏移）。"""
    out: list[Paragraph] = []
    for p in paras:
        for m in RE_SENT.finditer(p.text):
            s = m.group(0).strip()
            if not s:
                continue
            lead = len(m.group(0)) - len(m.group(0).lstrip())
            out.append(Paragraph(s, p.page, p.start + lead, p.start + lead + len(s), p.level, p.section_path))
    return out


def _explode(paras: Sequence[Paragraph], max_chars: int) -> list[Paragraph]:
    """把超长段落（一个几百字的"条"）按标点切成不超过 max_chars 的片段，偏移保持精确。

    不做这一步，fixed 策略的块长会从 512 飘到 1700 字——"固定 512 字切分"名不副实，
    消融实验就不公平了。
    """
    out: list[Paragraph] = []
    for p in paras:
        if len(p.text) <= max_chars:
            out.append(p)
            continue
        text, start = p.text, 0
        while start < len(text):
            end = min(len(text), start + max_chars)
            if end < len(text):
                cuts = [text.rfind(ch, start + max_chars // 2, end) for ch in "。；，、！？"]
                best = max(cuts)
                if best > start:
                    end = best + 1
            out.append(Paragraph(text[start:end], p.page, p.start + start, p.start + end, p.level, p.section_path))
            start = end
    return out


def chunk_fixed(doc: ParsedDocument, size: int = 512, overlap: int = 50) -> list[Chunk]:
    chunks: list[Chunk] = []
    buf: list[Paragraph] = []
    length = 0
    for p in _explode(doc.paragraphs, size):
        if p.level >= 1:  # 章/节标题不单独成块，挂到下一块开头
            if not buf:
                buf.append(p)
            continue
        buf.append(p)
        length += len(p.text)
        if length >= size:
            chunks.append(_make(doc, buf))
            tail: list[Paragraph] = []
            acc = 0
            for q in reversed(buf):
                tail.insert(0, q)
                acc += len(q.text)
                if acc >= overlap:
                    break
            buf, length = tail, acc
    if buf and sum(len(p.text) for p in buf) > 30:
        chunks.append(_make(doc, buf))
    for i, c in enumerate(chunks):
        c.ordinal = i
    return chunks


def chunk_semantic(
    doc: ParsedDocument,
    embed: EmbedFn,
    threshold: float = 0.62,
    min_chars: int = 160,
    max_chars: int = 700,
) -> list[Chunk]:
    sents = _explode(_sentences([p for p in doc.paragraphs if p.level == 0]), max_chars)
    if not sents:
        return []
    vecs = embed([s.text for s in sents])
    groups: list[list[Paragraph]] = [[sents[0]]]
    for i in range(1, len(sents)):
        sim = float(np.dot(vecs[i - 1], vecs[i]))
        cur_len = sum(len(p.text) for p in groups[-1])
        fits = cur_len + len(sents[i].text) <= max_chars
        if fits and (cur_len < min_chars or sim >= threshold):
            groups[-1].append(sents[i])
        else:
            groups.append([sents[i]])
    chunks = [_make(doc, g) for g in groups]
    for i, c in enumerate(chunks):
        c.ordinal = i
    return chunks


def chunk_structure(doc: ParsedDocument, max_chars: int = 900, min_chars: int = 80) -> list[Chunk]:
    """结构感知：以"章/节"分组、以"条"为最小不可分单位，超长才继续合并。

    条不可分是关键——一个条款被劈成两半，答案就永远拼不完整。
    """
    chunks: list[Chunk] = []
    buf: list[Paragraph] = []
    section = ""

    def flush() -> None:
        nonlocal buf
        if buf and sum(len(p.text) for p in buf) >= min_chars:
            chunks.append(_make(doc, buf, section))
        buf = []

    for p in doc.paragraphs:
        p = _explode([p], max_chars)[0] if len(p.text) > max_chars else p
        if p.level >= 1:
            flush()
            section = p.section_path or p.text
            buf.append(p)
            continue
        if buf and sum(len(q.text) for q in buf) + len(p.text) > max_chars:
            flush()
            section = p.section_path or section
        buf.append(p)
    if buf:
        chunks.append(_make(doc, buf, section))
    for i, c in enumerate(chunks):
        c.ordinal = i
    return chunks


def chunk_parent_child(
    doc: ParsedDocument,
    embed: EmbedFn | None = None,
    parent_chars: int = 900,
    child_chars: int = 200,
) -> tuple[list[Chunk], list[Chunk]]:
    """小块检索、大块喂模型。返回 (children, parents)，children 用 parent_key 指向父块。

    父块按"章/节 + 条"分（同 structure）；子块由父块内的正文段落切到 child_chars，
    偏移一律来自解析层，所以子块 text 与原文切片逐字一致。
    """
    groups: list[tuple[list[Paragraph], str]] = []
    buf: list[Paragraph] = []
    section = ""
    for p in doc.paragraphs:
        if p.level >= 1:
            if buf:
                groups.append((buf, section))
                buf = []
            section = p.section_path or p.text
            buf.append(p)
            continue
        if buf and sum(len(q.text) for q in buf) + len(p.text) > parent_chars:
            groups.append((buf, section))
            buf = []
            section = p.section_path or section
        buf.append(p)
    if buf:
        groups.append((buf, section))

    parents: list[Chunk] = []
    children: list[Chunk] = []
    for gi, (paras, sec) in enumerate(groups):
        parent = _make(doc, paras, sec)
        parent.is_parent = True
        parent.key = f"p{gi}"
        parent.ordinal = gi
        parents.append(parent)

        buf2: list[Paragraph] = []
        length = 0
        for piece in _explode([p for p in paras if p.level == 0], child_chars):
            buf2.append(piece)
            length += len(piece.text)
            if length >= child_chars:
                child = _make(doc, buf2, sec)
                child.parent_key = parent.key
                child.ordinal = len(children)
                children.append(child)
                buf2, length = [], 0
        if buf2:
            child = _make(doc, buf2, sec)
            child.parent_key = parent.key
            child.ordinal = len(children)
            children.append(child)
        if not children or children[-1].parent_key != parent.key:
            # 父块里没有正文（只有标题）时，用父块自身当子块，保证每个父块都可被召回
            child = Chunk(
                text=parent.text,
                page_start=parent.page_start,
                char_start=parent.char_start,
                page_end=parent.page_end,
                char_end=parent.char_end,
                section_path=sec,
                ordinal=len(children),
                parent_key=parent.key,
            )
            children.append(child)
    return children, parents


STRATEGIES = ("fixed", "semantic", "structure", "parent_child")


def chunk_document(
    doc: ParsedDocument, strategy: str, embed: EmbedFn | None = None
) -> tuple[list[Chunk], list[Chunk]]:
    """返回 (检索用块, 父块)。非 parent_child 策略父块为空。"""
    if strategy == "fixed":
        return chunk_fixed(doc), []
    if strategy == "semantic":
        if embed is None:
            raise ValueError("semantic 策略需要 embed 函数")
        return chunk_semantic(doc, embed), []
    if strategy == "structure":
        return chunk_structure(doc), []
    if strategy == "parent_child":
        return chunk_parent_child(doc, embed)
    raise ValueError(f"未知切块策略: {strategy}")