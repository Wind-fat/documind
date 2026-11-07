"""重排：bge-reranker-base（cross-encoder），无模型时降级为词面重排。

重排的作用在消融里看得见：RRF 融合后的前 30 条里，正确答案常常排在第 7-12 位，
cross-encoder 能把它提到前 3 —— 这是 Recall@5 与 faithfulness 的主要来源。

本机（i5-12500H，12 核/16 线程）实测 fp32 bge-reranker-base 约 700 token/s，
即 **延迟 ≈ 送进模型的 token 数 / 700**：30 条候选 × 512 token = 21.3s，其中 99% 是 forward。
所以这里的取舍全在"送几条"和"每条送多少 token"上，拆解与前后数字见 docs/RERANK-PERF.md。
"""

from __future__ import annotations

import logging
import math
import os
import re
import threading
from collections.abc import Sequence
from functools import lru_cache

import numpy as np

from .config import get_settings

# 句界：章程正文里条款以"。"或换行收尾；分号也算可切点（一条里并列多项时）。
_SENT_RE = re.compile(r"[^。；！？\n]*[。；！？\n]?")


def _bigrams(text: str) -> set[str]:
    t = re.sub(r"\s+", "", text.lower())
    return {t[i : i + 2] for i in range(max(len(t) - 1, 0))}


class Reranker:
    name = "base"

    def score(self, query: str, docs: Sequence[str]) -> list[float]:  # pragma: no cover
        raise NotImplementedError

    def rerank(self, query: str, items: Sequence[tuple[str, str]], top_k: int) -> list[tuple[str, float]]:
        """items = [(id, text)] → [(id, score)]，按分数降序取前 top_k。"""
        if not items:
            return []
        scores = self.score(query, [t for _, t in items])
        ranked = sorted(zip([i for i, _ in items], scores, strict=False), key=lambda x: -x[1])
        return ranked[:top_k]


class CrossEncoderReranker(Reranker):
    name = "bge-reranker-base"

    def __init__(
        self,
        model_name: str | None = None,
        cache_dir: str | None = None,
        max_tokens: int | None = None,
        batch_size: int | None = None,
        threads: int | None = None,
        truncate: str | None = None,
    ) -> None:
        s = get_settings()
        os.environ.setdefault("HF_ENDPOINT", s.hf_endpoint)
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        self.max_tokens = int(max_tokens or s.rerank_max_tokens)
        self.truncate = (truncate or s.rerank_truncate).lower()
        self.batch_size = max(1, int(batch_size or s.rerank_batch_size))
        th = s.rerank_threads if threads is None else threads
        self._model = TextCrossEncoder(
            model_name or s.rerank_model,
            cache_dir=str(cache_dir or s.model_cache_dir),
            threads=th or None,
        )
        self.name = model_name or s.rerank_model
        self._lock = threading.Lock()
        self._input_names: list[str] = []
        self._ready = False

    def _prepare(self):
        """拿到 fastembed 内部的 tokenizer 与 onnx session，并把截断收紧到 `max_tokens`。

        为什么不用 `self._model.rerank()`：它只有 `batch_size` 一个旋钮，按 tokenizer.json 里的
        `truncation.max_length=512` 截断，而且把整批 padding 到批内最长。30 条候选一起跑时，
        每条都要算满 512 token —— 这就是 21.9s 的来源。这里复用同一份模型（不重复加载），
        只自己控制截断长度和组批方式。

        onnxruntime 没有 autograd，`torch.inference_mode()` 在这里没有对应物；一次 `session.run`
        就是一次纯前向，不存在需要关闭的梯度开关。
        """
        inner = self._model.model
        if getattr(inner, "model", None) is None:
            inner.load_onnx_model()  # type: ignore[union-attr]
        inner._ensure_tokenizer()  # type: ignore[union-attr]
        if not self._ready:
            inner.tokenizer.enable_truncation(max_length=self.max_tokens)  # type: ignore[union-attr]
            self._input_names = [i.name for i in inner.model.get_inputs()]  # type: ignore[union-attr]
            self._ready = True
        return inner

    @property
    def _native(self) -> bool:
        """fastembed 的内部结构是不是我们认识的那个 ONNX cross-encoder。

        将来 fastembed 改了内部字段时不至于炸：退回它自己的 `rerank()`（慢，但正确）。
        """
        try:
            inner = self._model.model
        except AttributeError:  # pragma: no cover
            return False
        return hasattr(inner, "tokenizer") and hasattr(inner, "model")

    def _feed(self, encodings) -> dict:
        feed = {"input_ids": [e.ids for e in encodings]}
        if "token_type_ids" in self._input_names:
            feed["token_type_ids"] = [e.type_ids for e in encodings]
        if "attention_mask" in self._input_names:
            feed["attention_mask"] = [e.attention_mask for e in encodings]
        return {k: np.array(v, dtype=np.int64) for k, v in feed.items()}

    def _window(self, query: str, text: str) -> str:
        """按 `rerank_truncate` 把一条候选压到 `max_tokens` 个 token 以内。

        head：直接截前 N 个 token（老行为，便宜，但会切掉句尾）。
        focus：按句切，把与问题字面最重合的那句连同邻句凑到预算。中文 1 字 ≈ 1 token，
               所以预算按"字符数 ≈ token 数"折算，最后由 tokenizer 截断兜底。

        为什么需要 focus：块平均 ~710 字，直接 head 截到 256 时 Context Recall 从 0.90
        掉到 0.65（20 题里丢 5 题）——gold 句常常在第 256 字之后。focus 用 1/3 的窗口
        把召回拉回 0.95，见 docs/RERANK-PERF.md §4。
        """
        if self.truncate != "focus" or len(text) <= self.max_tokens:
            return text
        sents = [s for s in _SENT_RE.findall(text) if s.strip()]
        if not sents:
            return text[: self.max_tokens]
        q = _bigrams(query)
        overlap = [len(q & _bigrams(s)) for s in sents]
        if not max(overlap, default=0):
            return text[: self.max_tokens]
        best = max(range(len(sents)), key=lambda i: overlap[i])
        lo = hi = best
        size = len(sents[best])
        while True:
            cands = []
            if lo > 0:
                cands.append((len(sents[lo - 1]), 0))
            if hi + 1 < len(sents):
                cands.append((len(sents[hi + 1]), 1))
            if not cands:
                break
            size_next, side = min(cands)
            if size + size_next > self.max_tokens:
                break
            size += size_next
            if side == 0:
                lo -= 1
            else:
                hi += 1
        return "".join(sents[lo : hi + 1])

    def score(self, query: str, docs: Sequence[str]) -> list[float]:
        if not docs:
            return []
        if not self._native:  # pragma: no cover - 只有 fastembed 内部结构变了才走到
            with self._lock:
                return [float(x) for x in self._model.rerank(query, list(docs), batch_size=self.batch_size)]
        with self._lock:
            inner = self._prepare()
            tok, sess = inner.tokenizer, inner.model
            texts = [self._window(query, d) for d in docs]
            n = len(texts)
            out = [0.0] * n
            # 按字符数排序再组小批：中文 1 字 ≈ 1 token，长度相近的放一起能把 padding 降到最低。
            order = sorted(range(n), key=lambda i: len(texts[i]))
            for start in range(0, n, self.batch_size):
                idxs = order[start : start + self.batch_size]
                enc = tok.encode_batch([(query, texts[i]) for i in idxs])
                raw = sess.run(None, self._feed(enc))[0][:, 0]
                for i, v in zip(idxs, raw, strict=True):
                    out[i] = float(v)
            return out


class LexicalReranker(Reranker):
    """词面重排兜底：BM25-ish 覆盖率打分，映射到与 cross-encoder 可比的对数域。

    分数域刻意贴着 cross-encoder 的 logits 区间，否则拒答阈值没法共用一套标定。
    """

    name = "lexical"

    @staticmethod
    def _grams(text: str) -> set[str]:
        t = re.sub(r"\s+", "", text.lower())
        return {t[i : i + 2] for i in range(max(len(t) - 1, 0))}

    def score(self, query: str, docs: Sequence[str]) -> list[float]:
        q = self._grams(query)
        out = []
        for d in docs:
            g = self._grams(d)
            if not g:
                out.append(-12.0)
                continue
            cover = len(q & g) / max(len(q), 1)
            out.append(round(-12.0 + 12.0 * (cover**0.5), 4))
        _ = math
        return out


@lru_cache
def get_reranker() -> Reranker:
    s = get_settings()
    if s.rerank_backend == "lexical":
        return LexicalReranker()
    try:
        return CrossEncoderReranker(cache_dir=str(s.model_cache_dir))
    except Exception as exc:  # pragma: no cover
        logging.getLogger(__name__).warning("cross-encoder 不可用，降级为词面重排: %s", exc)
        return LexicalReranker()


def reranker_name() -> str:
    """当前 reranker 的名字（阈值分桶的键），**不触发模型加载**。

    /health 与启动日志要在模型还没预热时就能说清"阈值属于哪个后端"，
    所以未实例化时按配置推断；一旦实例化（含降级）就以真实对象为准。
    """
    if get_reranker.cache_info().currsize:
        return get_reranker().name
    s = get_settings()
    return "lexical" if s.rerank_backend == "lexical" else s.rerank_model