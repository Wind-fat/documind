"""运行配置。

dev 默认值保证 `git clone && uv sync && uv run documind ingest ...` 直接能跑：
没有外部服务、没有 API key 也能用本地 bge 模型 + 本地 mock LLM 走完整个链路。
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="DOCUMIND_", env_file=".env", extra="ignore")

    env: str = "dev"
    data_dir: Path = ROOT / "data"
    # 解析器：只有 "auto"（按后缀选 pypdf / python-docx / markdown）。
    # 没有 docling 档：DoclingParser 未接线且分页是伪造的，显式设置会直接报错（P1-02）。
    parser: str = "auto"
    database_url: str = "sqlite+aiosqlite:///./documind.db"

    # ---- LLM（OpenAI 兼容）。默认指向本机 magpie 网关，可一行切到 aigate / 任意供应商 ----
    llm_base_url: str = "http://127.0.0.1:3425/v1"
    llm_api_key: str = "magpie"
    llm_model: str = "qwen-cn/qwen3.8-flash"
    llm_timeout_s: float = 60.0
    llm_temperature: float = 0.0
    # 每百万 token 单价（用于成本看板，默认取 qwen3.8-flash 公开价，单位 USD）
    price_prompt_per_mtok: float = 0.15
    price_completion_per_mtok: float = 0.47

    # ---- 检索 ----
    top_k: int = 5
    recall_n: int = 30            # 粗排召回数，精排从这里挑 top_k
    rrf_k: int = 60
    # 重排（cross-encoder）的延迟杠杆，实测见 docs/RERANK-PERF.md：
    # 本机 i5-12500H（12 核/16 线程）上 fp32 bge-reranker-base 约 700 token/s
    # （= 1.4ms/token，与批大小基本无关），30 候选×512 token = 15 360 token ≈ 21.3s，
    # 其中 99% 是 forward 而不是分词。
    # candidates：真正送进 CE 的候选数（RRF 顺序截断，但不低于 top_k）
    # max_tokens：每条候选的 token 上限（模型自身 512）；focus 模式下按问题挑句窗口
    # batch_size：按长度排序后每批条数（批内 padding 到最长，长度聚合能少浪费 padding）
    rerank_candidates: int = 10
    rerank_max_tokens: int = 256
    rerank_truncate: str = "focus"  # head | focus（按问题挑句窗口，见 rerank.py:_window）
    rerank_batch_size: int = 8
    rerank_threads: int = 0       # 0 = 交给 onnxruntime 默认（= 物理核数）
    # 拒答阈值：由 evals 的 P-R 曲线标定，不是拍脑袋的 0.6。见 docs/adr/0004-refusal-threshold.md
    # 优先级（P1-03，见 thresholds.py）：DOCUMIND_REFUSAL_THRESHOLD > 最近一次标定值 > 内置占位。
    # 两个字段都是 None 表示"没显式覆盖"——这里**不能**给默认数值，
    # 否则代码永远分不清"用户设了"与"默认值"，标定回写就没有生效的机会。
    refusal_threshold: float | None = None          # DOCUMIND_REFUSAL_THRESHOLD（推荐）
    refusal_rerank_threshold: float | None = None   # 旧名，兼容既有 .env；新名优先
    selfcheck_retry: bool = True

    # ---- 模型 ----
    embed_backend: str = "auto"  # auto | hash（CI / 无模型环境用 hash，报告里会标注）
    rerank_backend: str = "auto"
    embed_model: str = "BAAI/bge-small-zh-v1.5"
    rerank_model: str = "BAAI/bge-reranker-base"
    model_cache_dir: Path = ROOT / ".models"
    hf_endpoint: str = "https://hf-mirror.com"

    # ---- 观测 ----
    langfuse_public_key: str = ""
    trace_dir: Path = ROOT / "data" / "traces"

    # ---- 安全 ----
    admin_token: str = "dev-admin-token"
    session_secret: str = "dev-session-secret-change-me"

    @field_validator("parser")
    @classmethod
    def _parser_is_auto_or_nothing(cls, v: str) -> str:
        """把"设了 DOCUMIND_PARSER=docling 却毫无效果"变成启动期的可读错误（P1-02）。"""
        value = (v or "auto").strip().lower()
        if value == "auto":
            return value
        if value == "docling":
            raise ValueError(
                "parser=docling 不可用：DoclingParser 未接线，且它的分页是按字数伪造的，"
                "会让 citation 高亮指向错误页。见 docs/adr/0001-parser.md"
            )
        raise ValueError(f"未知 parser={value!r}：当前只支持 auto（按文件后缀自动选择）")

    @field_validator(
        "llm_timeout_s", "refusal_threshold", "refusal_rerank_threshold", mode="before"
    )
    @classmethod
    def _blank_to_default(cls, v):
        # 空 env 值会被解析成 ""，数值字段直接 ValidationError（aigate 上踩过）
        return None if v in ("", None) else v

    @property
    def llm_configured(self) -> bool:
        return bool(self.llm_base_url and self.llm_model)

    def ensure_dirs(self) -> None:
        for p in (self.data_dir, self.trace_dir, self.model_cache_dir):
            p.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    s = Settings()
    s.ensure_dirs()
    return s
