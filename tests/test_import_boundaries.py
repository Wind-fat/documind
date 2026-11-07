"""跨包/跨模块边界测试（P1-09）。

评测链路是"阈值与指标的唯一来源"，它一旦 import 编排层的私有名，pipeline 内部改名就会
静默打断评测——而且打断的是最没人看着的那条链路。这里用 AST 把"下划线名不许跨模块 import"
变成一条可执行的规则，而不是一条口头约定。
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import documind

PKG = Path(documind.__file__).resolve().parent
SIBLING_MODULES = (
    "pipeline",
    "db",
    "retrieve",
    "generate",
    "chunking",
    "parsing",
    "index",
    "rerank",
    "embed",
    "query",
    "llm",
    "config",
    "thresholds",
    "app",
    "cli",
)


def _modules() -> list[Path]:
    return sorted(PKG.rglob("*.py"))


def test_no_private_name_crosses_a_module_boundary():
    """`from ..pipeline import _page_texts` 这类写法在包内一处都不许有。"""
    offenders: list[str] = []
    for path in _modules():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or node.level < 1:
                continue
            for alias in node.names:
                if alias.name.startswith("_"):
                    rel = path.relative_to(PKG).as_posix()
                    offenders.append(f"{rel}:{node.lineno} from {'.' * node.level}{node.module or ''} import {alias.name}")
    assert offenders == [], "包内不许 import 兄弟模块的私有名：\n" + "\n".join(offenders)


def test_no_private_attribute_access_on_sibling_modules():
    """同样拦 `pipeline._page_texts(...)` 这种"先 import 模块再点私有名"的绕法。"""
    pattern = re.compile(r"\b(?:" + "|".join(SIBLING_MODULES) + r")\._")
    offenders: list[str] = []
    for path in _modules():
        for no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if line.lstrip().startswith("#"):
                continue
            if pattern.search(line):
                offenders.append(f"{path.relative_to(PKG).as_posix()}:{no} {line.strip()}")
    assert offenders == [], "包内不许访问兄弟模块的私有属性：\n" + "\n".join(offenders)


def test_evals_runner_uses_the_public_page_texts():
    """P1-09 的具体落点：评测链路走 `db.load_page_texts`，不再有 `pipeline._page_texts`。"""
    src = (PKG / "evals" / "runner.py").read_text(encoding="utf-8")
    assert "load_page_texts" in src
    # 注意 `load_page_texts` 本身含有 `_page_texts` 子串，必须用词边界判断
    assert re.search(r"(?<!load)\b_page_texts\b", src) is None

    from documind.db import load_page_texts

    assert not load_page_texts.__name__.startswith("_")
    # 它必须能脱离 pipeline 单独导入：评测不该顺带拉起编排层（LLM / reranker）
    assert load_page_texts.__module__ == "documind.db"


async def test_load_page_texts_is_usable_without_pipeline(db, corpus_md: Path):
    """公开后的行为不变：仍然返回 `(doc_id, 页码) → 页文本`，空集合不查库。"""
    from documind.db import load_page_texts
    from documind.pipeline import ingest_document

    res = await ingest_document(db, corpus_md, acl_roles=["all"])
    texts = await load_page_texts(db, {res["document_id"]})
    assert texts
    assert all(isinstance(k, tuple) and len(k) == 2 and k[1] >= 1 for k in texts)
    assert all(isinstance(v, str) for v in texts.values())
    assert await load_page_texts(db, set()) == {}
    # 单个 doc_id 也接受（pipeline 里 `_page_texts(session, doc_id)` 的老调用形态）
    assert await load_page_texts(db, res["document_id"]) == texts