"""SQLAlchemy ORM models (SQLite by default, PostgreSQL-ready)."""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utcnow() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    username: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    # Nullable rather than an empty string: "no password set" must stay
    # distinguishable from a password that hashes to nothing, which would
    # invite an accidental match.
    hashed_password: Mapped[str | None] = mapped_column(String(255), nullable=True)
    full_name: Mapped[str | None] = mapped_column(String(128), nullable=True)
    risk_profile: Mapped[str] = mapped_column(String(32), default="balanced")
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    # ---- identity provider -------------------------------------------
    # Only "local" today. Kept as a column rather than assumed, so adding a
    # second provider later does not need a migration on a populated table.
    #
    # Installations that once ran the Google build still carry `google_id` and
    # `avatar_url` in their `users` table. They are deliberately left in place:
    # dropping a column rewrites the table, and no amount of tidiness is worth
    # risking real accounts to delete two columns that are already empty.
    auth_provider: Mapped[str] = mapped_column(String(32), default="local")

    # ---- email verification ------------------------------------------
    email_verified: Mapped[bool] = mapped_column(Boolean, default=False)
    verified_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True),
                                                         nullable=True)
    last_login: Mapped[datetime | None] = mapped_column(DateTime(timezone=True),
                                                        nullable=True)

    portfolios: Mapped[list[Portfolio]] = relationship(back_populates="owner", cascade="all, delete-orphan")
    alerts: Mapped[list[Alert]] = relationship(back_populates="user", cascade="all, delete-orphan")


class Portfolio(Base):
    __tablename__ = "portfolios"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True, index=True)
    name: Mapped[str] = mapped_column(String(128))
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    base_currency: Mapped[str] = mapped_column(String(8), default="USD")
    initial_capital: Mapped[float] = mapped_column(Float, default=100_000.0)
    cash: Mapped[float] = mapped_column(Float, default=100_000.0)
    strategy: Mapped[str] = mapped_column(String(64), default="manual")
    is_paper: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)

    owner: Mapped[User | None] = relationship(back_populates="portfolios")
    positions: Mapped[list[Position]] = relationship(back_populates="portfolio", cascade="all, delete-orphan")
    transactions: Mapped[list[Transaction]] = relationship(back_populates="portfolio", cascade="all, delete-orphan")
    snapshots: Mapped[list[PortfolioSnapshot]] = relationship(back_populates="portfolio", cascade="all, delete-orphan")


class Position(Base):
    __tablename__ = "positions"
    __table_args__ = (UniqueConstraint("portfolio_id", "symbol", name="uq_portfolio_symbol"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    portfolio_id: Mapped[int] = mapped_column(ForeignKey("portfolios.id"), index=True)
    symbol: Mapped[str] = mapped_column(String(32), index=True)
    quantity: Mapped[float] = mapped_column(Float, default=0.0)
    average_price: Mapped[float] = mapped_column(Float, default=0.0)
    asset_class: Mapped[str] = mapped_column(String(32), default="equity")
    opened_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)

    portfolio: Mapped[Portfolio] = relationship(back_populates="positions")


class Transaction(Base):
    __tablename__ = "transactions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    portfolio_id: Mapped[int] = mapped_column(ForeignKey("portfolios.id"), index=True)
    symbol: Mapped[str] = mapped_column(String(32), index=True)
    side: Mapped[str] = mapped_column(String(8))            # BUY / SELL
    quantity: Mapped[float] = mapped_column(Float)
    price: Mapped[float] = mapped_column(Float)
    fees: Mapped[float] = mapped_column(Float, default=0.0)
    source: Mapped[str] = mapped_column(String(32), default="manual")   # manual | rl_agent | rebalance
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    executed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)

    portfolio: Mapped[Portfolio] = relationship(back_populates="transactions")


class PortfolioSnapshot(Base):
    __tablename__ = "portfolio_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    portfolio_id: Mapped[int] = mapped_column(ForeignKey("portfolios.id"), index=True)
    total_value: Mapped[float] = mapped_column(Float)
    cash: Mapped[float] = mapped_column(Float)
    invested_value: Mapped[float] = mapped_column(Float)
    pnl: Mapped[float] = mapped_column(Float, default=0.0)
    pnl_pct: Mapped[float] = mapped_column(Float, default=0.0)
    metrics: Mapped[dict] = mapped_column(JSON, default=dict)
    taken_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)

    portfolio: Mapped[Portfolio] = relationship(back_populates="snapshots")


class Alert(Base):
    __tablename__ = "alerts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True, index=True)
    symbol: Mapped[str] = mapped_column(String(32), index=True)
    alert_type: Mapped[str] = mapped_column(String(48))     # price_move | volatility | signal | news | risk
    severity: Mapped[str] = mapped_column(String(16), default="info")
    title: Mapped[str] = mapped_column(String(255))
    message: Mapped[str] = mapped_column(Text)
    payload: Mapped[dict] = mapped_column(JSON, default=dict)
    is_read: Mapped[bool] = mapped_column(Boolean, default=False)
    triggered_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)

    user: Mapped[User | None] = relationship(back_populates="alerts")


class AlertRule(Base):
    """A user-defined alert.

    The original shape was one metric, one threshold. Everything added since —
    multiple conditions, AND/OR, priority, expiry — is nullable or defaulted so
    that rules created under the old schema keep working untouched:
    ``rule_type`` + ``threshold`` remain the single-condition form, and
    ``conditions`` is only consulted when it holds something.
    """

    __tablename__ = "alert_rules"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True, index=True)
    symbol: Mapped[str] = mapped_column(String(32), index=True)
    rule_type: Mapped[str] = mapped_column(String(48))      # price_above | price_below | pct_move | rsi | risk
    threshold: Mapped[float] = mapped_column(Float)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    cooldown_minutes: Mapped[int] = mapped_column(Integer, default=60)
    last_triggered: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    # ---- multi-condition support -------------------------------------
    # [{"metric": "rsi", "operator": "above", "value": 70}, ...]
    conditions: Mapped[list] = mapped_column(JSON, default=list)
    logic: Mapped[str] = mapped_column(String(8), default="AND")     # AND | OR
    name: Mapped[str | None] = mapped_column(String(120), nullable=True)
    priority: Mapped[str] = mapped_column(String(16), default="medium")
    # Which history window the indicator conditions are evaluated over. An RSI
    # on 1 month and an RSI on 5 years are different questions.
    period: Mapped[str] = mapped_column(String(8), default="6mo")
    notify_in_app: Mapped[bool] = mapped_column(Boolean, default=True)
    notify_email: Mapped[bool] = mapped_column(Boolean, default=False)
    notify_push: Mapped[bool] = mapped_column(Boolean, default=False)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    # False = fire once then deactivate; True = keep firing after each cooldown.
    recurring: Mapped[bool] = mapped_column(Boolean, default=True)
    trigger_count: Mapped[int] = mapped_column(Integer, default=0)
    template: Mapped[str | None] = mapped_column(String(48), nullable=True)


class ModelRegistry(Base):
    __tablename__ = "model_registry"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    symbol: Mapped[str] = mapped_column(String(32), index=True)
    model_type: Mapped[str] = mapped_column(String(32))     # forecast | rl
    algorithm: Mapped[str] = mapped_column(String(48))
    horizon: Mapped[int] = mapped_column(Integer, default=5)
    metrics: Mapped[dict] = mapped_column(JSON, default=dict)
    config: Mapped[dict] = mapped_column(JSON, default=dict)
    artifact_path: Mapped[str] = mapped_column(String(512))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    trained_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)


class RecommendationLog(Base):
    __tablename__ = "recommendation_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    symbol: Mapped[str] = mapped_column(String(32), index=True)
    action: Mapped[str] = mapped_column(String(16))
    composite_score: Mapped[float] = mapped_column(Float)
    confidence: Mapped[float] = mapped_column(Float)
    price_at_reco: Mapped[float] = mapped_column(Float)
    signals: Mapped[dict] = mapped_column(JSON, default=dict)
    explanation: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)

    # ---------------------------------------------- model-governance fields
    # A decision log is only auditable if it records *which* model produced the
    # call, under what market conditions, and with which risk figures. Without
    # these, a reviewer can see that a SELL was issued but cannot reconstruct
    # why, nor tell whether the model that issued it is the one running today.
    source: Mapped[str | None] = mapped_column(String(32), default="engine", index=True)
    model_version: Mapped[str | None] = mapped_column(String(64), default=None)
    algo: Mapped[str | None] = mapped_column(String(32), default=None)
    regime: Mapped[str | None] = mapped_column(String(32), default=None, index=True)
    regime_confidence: Mapped[float | None] = mapped_column(Float, default=None)
    regime_influence: Mapped[str | None] = mapped_column(String(16), default=None)
    risk_metrics: Mapped[dict] = mapped_column(JSON, default=dict)
    regime_explanation: Mapped[dict] = mapped_column(JSON, default=dict)
