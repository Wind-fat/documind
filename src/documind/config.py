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
    # 拒答阈值：由 evals 的 P-R 曲线标定，不是拍脑袋的 0.6。见 docs/adr/0004-refusal-threshold.md
    refusal_rerank_threshold: float = -6.0
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

    @field_validator("llm_timeout_s", "refusal_rerank_threshold", mode="before")
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