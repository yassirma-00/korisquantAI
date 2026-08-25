"""Central application configuration.

All settings are overridable through environment variables or a `.env` file.
The platform is designed to run in three modes:

* ``hybrid``  (default) -> try live providers, fall back to the synthetic engine
* ``live``               -> only live providers, raise when unavailable
* ``offline``            -> never hit the network (deterministic demo / CI)
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BACKEND_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(PROJECT_ROOT / ".env", BACKEND_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ------------------------------------------------------------------ app
    APP_NAME: str = "KorisQuant AI - Intelligent Financial Analysis & Portfolio Management"
    APP_VERSION: str = "0.1.0"
    API_V1_PREFIX: str = "/api/v1"
    ENVIRONMENT: Literal["development", "staging", "production"] = "development"
    DEBUG: bool = True
    LOG_LEVEL: str = "INFO"
    SECRET_KEY: str = "change-me-in-production-please-use-a-long-random-string"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24

    # ------------------------------------------------------------- data mode
    DATA_MODE: Literal["hybrid", "live", "offline"] = "hybrid"
    NETWORK_TIMEOUT: float = 8.0
    CACHE_TTL_SECONDS: int = 300
    SYNTHETIC_SEED: int = 42

    # ------------------------------------------------------------ providers
    ALPHA_VANTAGE_API_KEY: str | None = None
    FINNHUB_API_KEY: str | None = None
    POLYGON_API_KEY: str | None = None
    NEWSAPI_KEY: str | None = None

    # ------------------------------------------------------ authentication
    # Every page and business endpoint requires a session. Only the landing
    # page, the auth screens and /health stay public.
    REQUIRE_AUTH: bool = True
    # "Remember me" trades a longer window for convenience; the short session
    # is the default so an unattended browser expires quickly.
    SESSION_MINUTES: int = 60 * 12          # 12 hours
    SESSION_REMEMBER_MINUTES: int = 60 * 24 * 30   # 30 days

    # ---- email delivery ----------------------------------------------
    # With no SMTP host configured, verification and reset links are written to
    # the server log and returned to the caller instead of being emailed. That
    # keeps the whole flow testable locally; set SMTP_HOST to send for real.
    SMTP_HOST: str | None = None
    SMTP_PORT: int = 587
    SMTP_USER: str | None = None
    SMTP_PASSWORD: str | None = None
    SMTP_FROM: str = "KorisQuant AI <no-reply@korisquant.ai>"
    SMTP_USE_TLS: bool = True
    PUBLIC_BASE_URL: str = "http://localhost:8000"

    @property
    def email_enabled(self) -> bool:
        return bool(self.SMTP_HOST)

    # ------------------------------------------------- conversational agent
    # The assistant runs on Ollama, which exposes an OpenAI-compatible
    # /v1/chat/completions endpoint. Two deployments are supported with the same
    # code path:
    #   * Ollama Cloud  -> https://ollama.com/v1        (needs OLLAMA_API_KEY)
    #   * local Ollama  -> http://localhost:11434/v1    (no key required)
    # Point OLLAMA_BASE_URL at whichever you run; the key is simply omitted when
    # empty, which is exactly what a local daemon expects.
    #
    # The key is read from the environment and is NEVER shipped to the browser:
    # every call is proxied by this backend (see app/api/v1/endpoints/chat.py).
    OLLAMA_API_KEY: str | None = None
    OLLAMA_BASE_URL: str = "https://ollama.com/v1"
    # gpt-oss:20b - tool calling verified against the live catalogue, ~1.5 s per
    # turn, and available without an Ollama subscription.
    OLLAMA_MODEL: str = "gpt-oss:20b"
    # Tried in order when the primary model errors out or is saturated. Every
    # entry MUST support function calling, otherwise the assistant loses its only
    # access to real platform data. Same NoDecode treatment as CORS_ORIGINS:
    # this is a comma-separated env var, not JSON.
    OLLAMA_FALLBACK_MODELS: Annotated[list[str], NoDecode] = Field(default_factory=lambda: [
        "gpt-oss:120b",
        "nemotron-3-super",
        "gemma4:31b",
    ])
    CHAT_ENABLED: bool = True
    CHAT_TIMEOUT: float = 90.0
    # Connecting is fast or impossible; only generation is slow. Kept separate
    # so an unreachable service fails in seconds instead of burning the full
    # generation budget on every model in the chain.
    CHAT_CONNECT_TIMEOUT: float = 5.0
    CHAT_MAX_TOOL_ROUNDS: int = 4      # tool -> model -> tool ... before answering
    CHAT_MAX_HISTORY: int = 12         # turns kept from the client transcript
    CHAT_MAX_TOKENS: int = 1400
    CHAT_TEMPERATURE: float = 0.3      # low: this assistant reports figures
    CHAT_RATE_LIMIT_PER_MIN: int = 20  # per client IP

    @field_validator("OLLAMA_FALLBACK_MODELS", mode="before")
    @classmethod
    def _split_models(cls, v):
        if isinstance(v, str):
            return [m.strip() for m in v.split(",") if m.strip()]
        return v

    @property
    def ollama_is_local(self) -> bool:
        """A local daemon needs no credential; the cloud does."""
        return any(h in self.OLLAMA_BASE_URL for h in ("localhost", "127.0.0.1", "0.0.0.0"))

    @property
    def chat_available(self) -> bool:
        if not self.CHAT_ENABLED:
            return False
        # Local Ollama is usable without a key, so requiring one would wrongly
        # report the assistant as unconfigured on a perfectly working setup.
        return bool(self.OLLAMA_API_KEY) or self.ollama_is_local

    # ------------------------------------------------------------- storage
    # The database filename intentionally keeps its original name: renaming it
    # would orphan every portfolio, transaction and alert already stored on an
    # existing installation. A cosmetic rebrand must not cost real data.
    DATABASE_URL: str = f"sqlite+aiosqlite:///{DATA_DIR / 'finai.db'}"
    SYNC_DATABASE_URL: str = f"sqlite:///{DATA_DIR / 'finai.db'}"
    REDIS_URL: str | None = None            # optional, in-memory cache otherwise
    MONGO_URL: str | None = None            # optional, for raw news documents

    # ----------------------------------------------------------------- ml
    MODEL_DIR: Path = DATA_DIR / "models"
    ARTIFACT_DIR: Path = DATA_DIR / "artifacts"
    CACHE_DIR: Path = DATA_DIR / "cache"
    TORCH_DEVICE: str = "cpu"
    DEFAULT_LOOKBACK: int = 60              # sequence length for DL models
    DEFAULT_HORIZON: int = 5                # forecast horizon (business days)
    TRAIN_EPOCHS: int = 25
    BATCH_SIZE: int = 32

    # ----------------------------------------------------------------- rl
    RL_INITIAL_BALANCE: float = 100_000.0
    RL_TRANSACTION_COST: float = 0.001      # 10 bps per trade
    # Feed the detected market regime into the RL observation and reward.
    # Default False: enabling it widens the observation vector, and the 11
    # agents already on disk were trained on the narrower one. New agents can
    # opt in per training request; the choice is stored in their metadata so
    # inference always rebuilds the environment they were trained in.
    RL_REGIME_AWARE: bool = False
    RL_MAX_STEPS: int = 1_000
    RL_TRAIN_EPISODES: int = 30

    # -------------------------------------------------------------- alerts
    ALERT_PRICE_MOVE_PCT: float = 3.0
    ALERT_VOL_SPIKE_Z: float = 2.5
    ALERT_DRAWDOWN_PCT: float = 10.0
    # The API reference stays reachable at /docs for developers; it is simply not
    # advertised anywhere in the dashboard UI. Set to False to disable the routes
    # entirely (e.g. a hardened public deployment).
    EXPOSE_API_DOCS: bool = True
    SCHEDULER_ENABLED: bool = False
    SCHEDULER_INTERVAL_MINUTES: int = 15

    # ---------------------------------------------------------------- cors
    # Annotated[..., NoDecode] is required: without it pydantic-settings tries to
    # json.loads() the raw env value *before* the validator runs, so the documented
    # `CORS_ORIGINS=*` in .env.example crashed the app at import with a
    # SettingsError. NoDecode hands the plain string to _split_origins instead.
    CORS_ORIGINS: Annotated[list[str], NoDecode] = Field(default_factory=lambda: ["*"])

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def _split_origins(cls, v):
        if isinstance(v, str):
            return [o.strip() for o in v.split(",") if o.strip()]
        return v

    @property
    def offline(self) -> bool:
        return self.DATA_MODE == "offline"

    @property
    def allow_network(self) -> bool:
        return self.DATA_MODE in ("hybrid", "live")

    def ensure_dirs(self) -> None:
        for path in (DATA_DIR, self.MODEL_DIR, self.ARTIFACT_DIR, self.CACHE_DIR):
            Path(path).mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_dirs()
    return settings


settings = get_settings()
