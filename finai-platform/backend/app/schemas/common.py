"""Pydantic request/response schemas for the public API."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.services.rl.catalogue import CONTINUOUS_KEYS, DISCRETE_KEYS, get_algorithm


def _validate_algo(value: str, allowed: set[str], context: str) -> str:
    """Validate against the live catalogue instead of a hard-coded Literal.

    A frozen Literal silently drifts out of sync every time an algorithm is
    added - the request is then rejected with an opaque schema error before the
    service can explain what is actually supported.
    """
    key = (value or "").lower().strip()
    if key not in allowed:
        spec = get_algorithm(key)
        # The most common mistake is sending a continuous-action algorithm to the
        # single-asset endpoint (or vice versa). Say so explicitly and name the
        # endpoint that does support it, instead of just listing valid keys.
        if spec is not None:
            other = ("/api/v1/rl/portfolio/train (multi-asset allocation)"
                     if context == "discrete" else "/api/v1/rl/train (single asset)")
            raise ValueError(
                f"'{spec.name}' uses a {spec.action_space} action space and cannot be trained "
                f"on the {context} endpoint. Use {other} instead.")
        raise ValueError(
            f"'{value}' is not a recognised algorithm. Available for {context}: {sorted(allowed)}")
    spec = get_algorithm(key)
    if spec and not spec.available:
        raise ValueError(
            f"'{spec.name}' requires the optional '{spec.backend}' backend, which is not installed.")
    return key


class APIResponse(BaseModel):
    """Generic envelope used by simple endpoints."""
    success: bool = True
    message: str | None = None
    data: Any = None


class HealthResponse(BaseModel):
    status: str
    version: str
    environment: str
    data_mode: str
    live_providers: list[str]
    sb3_available: bool
    torch_available: bool
    universe_size: int
    timestamp: str


# ------------------------------------------------------------------ market
class Candle(BaseModel):
    date: str
    open: float | None = None
    high: float | None = None
    low: float | None = None
    close: float | None = None
    volume: float | None = None


class HistoryResponse(BaseModel):
    symbol: str
    name: str
    asset_class: str
    currency: str
    period: str
    interval: str
    source: str
    is_live: bool
    bars: int
    candles: list[dict]
    indicators: list[str] = Field(default_factory=list)


class QuoteResponse(BaseModel):
    model_config = ConfigDict(extra="allow")
    symbol: str
    price: float | None = None
    change: float | None = None
    change_percent: float | None = None
    source: str | None = None


# --------------------------------------------------------------- forecast
class TrainForecastRequest(BaseModel):
    symbol: str = Field(..., examples=["AAPL"])
    model: Literal["lstm", "gru", "tcn", "transformer", "cnn_lstm"] = "lstm"
    period: str = "5y"
    horizon: int = Field(5, ge=1, le=60)
    lookback: int = Field(60, ge=10, le=250)
    epochs: int = Field(25, ge=1, le=300)
    batch_size: int = Field(32, ge=8, le=256)
    learning_rate: float = Field(1e-3, gt=0, le=0.1)
    target: Literal["target_return", "target_price", "target_volatility"] = "target_return"


class PredictRequest(BaseModel):
    symbol: str
    model: Literal["lstm", "gru", "tcn", "transformer", "cnn_lstm"] = "lstm"
    horizon: int = Field(5, ge=1, le=60)
    period: str = "2y"


class CompareModelsRequest(BaseModel):
    symbol: str
    models: list[Literal["lstm", "gru", "tcn", "transformer", "cnn_lstm"]] = ["lstm", "gru", "tcn"]
    period: str = "3y"
    horizon: int = Field(5, ge=1, le=60)
    epochs: int = Field(12, ge=1, le=100)


# --------------------------------------------------------------------- RL
class TrainRLRequest(BaseModel):
    symbol: str
    algo: str = "dueling_dqn"

    @field_validator("algo")
    @classmethod
    def _check_algo(cls, v: str) -> str:
        # Continuous algorithms are trained as a 1-asset allocation problem.
        return _validate_algo(v, DISCRETE_KEYS | CONTINUOUS_KEYS, "single-asset")
    period: str = "3y"
    episodes: int | None = Field(None, ge=1, le=500)
    total_timesteps: int | None = Field(None, ge=1000, le=1_000_000)
    # These default to None, not to a number. A literal default here is
    # indistinguishable from a value the user actually typed, so it was sent on
    # every request and silently overrode the selected profile: picking
    # "Conservative" (risk_penalty 0.30) still trained at 0.15. None means
    # "not specified — take it from the profile".
    initial_balance: float | None = Field(None, gt=0)
    transaction_cost: float | None = Field(None, ge=0, le=0.05)
    risk_penalty: float | None = Field(None, ge=0, le=2.0)
    test_fraction: float | None = Field(None, ge=0.05, le=0.5)
    # Feed the detected market regime into the observation and scale the risk
    # penalties by it. Defaults to the server setting so existing clients keep
    # producing agents of the shape they already expect.
    regime_aware: bool | None = None
    cvar_penalty: float | None = Field(None, ge=0, le=2.0)
    regime_reward_weight: float | None = Field(None, ge=0, le=3.0)
    # Save under a suffixed name instead of replacing the existing agent, so a
    # regime-aware twin can sit beside its baseline for comparison. Empty (the
    # default) keeps the historical filename exactly as it is.
    variant: str = ""
    # Which saved profile to train with. The service already accepted this;
    # the API never forwarded it, so every run silently used "default" no
    # matter what the user had selected in the dashboard.
    profile: str = "default"


class TrainPortfolioRLRequest(BaseModel):
    symbols: list[str] = Field(..., min_length=2, max_length=20)
    algo: str = "ppo"

    @field_validator("algo")
    @classmethod
    def _check_algo(cls, v: str) -> str:
        return _validate_algo(v, CONTINUOUS_KEYS, "continuous")
    period: str = "3y"
    total_timesteps: int | None = Field(None, ge=1000, le=1_000_000)
    initial_balance: float | None = Field(None, gt=0)
    transaction_cost: float | None = Field(None, ge=0, le=0.05)
    # One regime track per asset, and risk aversion weighted by the allocation.
    regime_aware: bool | None = None
    cvar_penalty: float | None = Field(None, ge=0, le=2.0)
    regime_reward_weight: float | None = Field(None, ge=0, le=3.0)
    variant: str = ""
    profile: str = "default"


# ---------------------------------------------------------------- portfolio
class CreatePortfolioRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=128)
    description: str | None = None
    initial_capital: float = Field(100_000.0, gt=0)
    base_currency: str = "USD"
    strategy: str = "manual"


class TradeRequest(BaseModel):
    symbol: str
    side: Literal["BUY", "SELL"]
    quantity: float | None = Field(None, gt=0)
    notional: float | None = Field(None, gt=0)
    price: float | None = Field(None, gt=0)
    notes: str | None = None


class RebalanceRequest(BaseModel):
    target_weights: dict[str, float] | None = None
    objective: Literal["max_sharpe", "min_volatility", "max_return", "risk_parity"] = "max_sharpe"
    period: str = "1y"
    execute: bool = False
    tolerance: float = Field(0.02, ge=0.001, le=0.5)


class OptimiseRequest(BaseModel):
    symbols: list[str] = Field(..., min_length=2, max_length=30)
    objective: Literal["max_sharpe", "min_volatility", "max_return", "risk_parity"] = "max_sharpe"
    period: str = "2y"
    risk_free_rate: float = Field(0.02, ge=-0.05, le=0.25)
    allow_short: bool = False


# ------------------------------------------------------------ recommendation
class RecommendRequest(BaseModel):
    symbol: str
    period: str = "2y"
    forecast_model: Literal["lstm", "gru", "tcn", "transformer", "cnn_lstm"] = "lstm"
    horizon: int = Field(5, ge=1, le=60)
    rl_algo: str = "dueling_dqn"

    @field_validator("rl_algo")
    @classmethod
    def _check_rl_algo(cls, v: str) -> str:
        # Every algorithm is usable here: continuous ones run as a single-asset
        # allocation and their target weight is mapped to BUY/HOLD/SELL.
        return _validate_algo(v, DISCRETE_KEYS | CONTINUOUS_KEYS, "recommendation")
    include_xai: bool = True


class ScreenRequest(BaseModel):
    symbols: list[str] = Field(..., min_length=1, max_length=30)
    period: str = "1y"
    horizon: int = Field(5, ge=1, le=60)


# --------------------------------------------------------------------- NLP
class SentimentRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=8000)


class BatchSentimentRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=100)


# ------------------------------------------------------------------ alerts
class AlertCondition(BaseModel):
    metric: str = Field(..., min_length=1, max_length=48)
    operator: str = Field("above", max_length=24)
    # Categorical conditions (regime, AI action) carry a string; numeric ones a
    # number. Accepting both here keeps one condition shape for the whole UI.
    value: float | str


class CreateAlertRuleRequest(BaseModel):
    """A rule.

    `rule_type` + `threshold` remain for the single-condition form that older
    clients and stored rules use. When `conditions` is supplied it takes
    precedence, and rule_type is recorded as "custom".
    """

    symbol: str
    rule_type: str = "custom"
    threshold: float = 0.0
    cooldown_minutes: int = Field(60, ge=1, le=10_080)

    name: str | None = Field(None, max_length=120)
    conditions: list[AlertCondition] = Field(default_factory=list, max_length=8)
    logic: Literal["AND", "OR"] = "AND"
    priority: Literal["low", "medium", "high", "critical"] = "medium"
    period: str = "6mo"
    notify_in_app: bool = True
    notify_email: bool = False
    notify_push: bool = False
    expires_at: datetime | None = None
    recurring: bool = True
    template: str | None = Field(None, max_length=48)


class UpdateAlertRuleRequest(BaseModel):
    """Every field optional: an edit changes only what it names."""

    symbol: str | None = None
    name: str | None = Field(None, max_length=120)
    conditions: list[AlertCondition] | None = Field(None, max_length=8)
    logic: Literal["AND", "OR"] | None = None
    priority: Literal["low", "medium", "high", "critical"] | None = None
    period: str | None = None
    cooldown_minutes: int | None = Field(None, ge=1, le=10_080)
    notify_in_app: bool | None = None
    notify_email: bool | None = None
    notify_push: bool | None = None
    expires_at: datetime | None = None
    recurring: bool | None = None
    is_active: bool | None = None


class BulkRuleActionRequest(BaseModel):
    rule_ids: list[int] = Field(..., min_length=1, max_length=200)
    action: Literal["enable", "disable", "delete", "duplicate"]


class ScanRequest(BaseModel):
    symbols: list[str] = Field(..., min_length=1, max_length=50)
    checks: list[Literal["price", "volatility", "signals", "risk", "news"]] | None = None
    persist: bool = False


# --------------------------------------------------------------- assistant
class ChatTurn(BaseModel):
    """One previous message replayed from the client transcript."""
    role: Literal["user", "assistant"]
    content: str = Field(..., max_length=8000)


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)
    # History lives in the browser: the assistant is stateless server-side, so a
    # restart never loses a conversation and no transcript is persisted here.
    history: list[ChatTurn] | None = Field(default=None, max_length=40)
    # Page + symbol let the assistant resolve "this stock" without asking.
    page: str | None = Field(default=None, max_length=40)
    symbol: str | None = Field(default=None, max_length=24)

    @field_validator("symbol")
    @classmethod
    def _clean_symbol(cls, v: str | None) -> str | None:
        return v.strip().upper() if v and v.strip() else None


# ------------------------------------------------------------------- auth
class RegisterRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=32,
                          pattern=r"^[A-Za-z0-9_.-]+$")
    email: str = Field(..., max_length=255)
    password: str = Field(..., min_length=8, max_length=256)
    full_name: str | None = Field(default=None, max_length=128)
    remember_me: bool = False

    @field_validator("email")
    @classmethod
    def _valid_email(cls, v: str) -> str:
        """A dependency-free sanity check that says *why* it refused.

        Not full RFC 5321 — that needs email-validator, which is overkill for a
        paper-trading research tool. The goal is catching the typos people
        actually make ("@gmail" with no TLD, "@2021" with no dot) while giving
        a message that points at the mistake instead of a flat "invalid".
        """
        value = (v or "").strip()
        if "@" not in value:
            raise ValueError("Enter a valid email address, including the @ sign")
        local, _, domain = value.rpartition("@")
        if not local:
            raise ValueError("Enter the part of your address before the @ sign")
        if not domain:
            raise ValueError("Enter the domain after the @ sign, for example gmail.com")
        if "." not in domain:
            # The single most common signup typo, and the one that silently
            # produces an address no verification email can ever reach.
            raise ValueError(
                f"'{domain}' is not a complete domain — it needs a suffix such "
                f"as {domain}.com")
        if domain.startswith(".") or domain.endswith(".") or ".." in domain:
            raise ValueError("The domain in that address is not valid")
        if len(value) < 6:
            raise ValueError("Enter a valid email address")
        return value.lower()


class LoginRequest(BaseModel):
    # Either the username or the email works; people remember one or the other.
    identifier: str = Field(..., min_length=1, max_length=255)
    password: str = Field(..., min_length=1, max_length=256)
    remember_me: bool = False


