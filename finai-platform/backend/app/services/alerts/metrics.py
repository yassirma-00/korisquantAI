"""Metric resolution for custom alert rules.

Every condition a user can write resolves through one registry, so a rule that
says "RSI above 70" and a rule that says "crash risk above 60%" are evaluated by
the same machinery and reported in the same shape.

Two design choices worth stating:

* **One market fetch per (symbol, period), reused across conditions.** A rule
  with five indicator conditions must not pull the same history five times.
* **A metric that cannot be computed returns ``None``, never 0.** Zero is a
  legitimate RSI-adjacent value and a legitimate drawdown; using it as "no
  data" would silently fire — or silently suppress — real alerts. ``None``
  propagates as "condition not met, reason unknown" and is reported as such.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from app.core.logging import get_logger
from app.services.data.market_data import market_data_service

logger = get_logger(__name__)


@dataclass
class MetricSpec:
    """What a metric is, in the units the user sees."""

    key: str
    label: str
    unit: str                 # "%", "$", "", "score"
    group: str                # price | technical | risk | ai
    hint: str
    # Value shown as the default threshold when this metric is picked.
    default: float = 0.0


# The catalogue the UI renders. Order matters: it is the order of the dropdown.
METRIC_SPECS: tuple[MetricSpec, ...] = (
    # ---- price and volume
    MetricSpec("price", "Price", "$", "price", "Last traded price", 200.0),
    MetricSpec("pct_change", "Daily change", "%", "price",
               "Percent move since the previous close", 3.0),
    MetricSpec("volume_ratio", "Volume vs 20d avg", "x", "price",
               "Today's volume divided by its 20-day average", 2.0),
    # ---- technical
    MetricSpec("rsi", "RSI (14)", "", "technical",
               "Relative Strength Index; >70 overbought, <30 oversold", 70.0),
    MetricSpec("macd_hist", "MACD histogram", "", "technical",
               "MACD minus its signal line; sign marks the crossover", 0.0),
    MetricSpec("bb_position", "Bollinger %B", "", "technical",
               "0 = lower band, 1 = upper band", 1.0),
    MetricSpec("sma_distance", "Distance from 50-day MA", "%", "technical",
               "How far price sits above or below its 50-day average", 5.0),
    MetricSpec("ma_cross", "50/200 MA spread", "%", "technical",
               "50-day minus 200-day, as a percent; positive is a golden cross", 0.0),
    # ---- risk
    MetricSpec("crash_risk", "Crash Risk Score", "%", "risk",
               "Composite tail-risk score, 0-100%", 55.0),
    MetricSpec("bubble_score", "Bubble Indicator", "%", "risk",
               "Overheating score, 0-100%", 60.0),
    MetricSpec("var_95", "Daily VaR 95%", "%", "risk",
               "Loss exceeded on 5% of days (negative)", -3.0),
    MetricSpec("cvar_95", "Daily CVaR 95%", "%", "risk",
               "Average loss on the worst 5% of days (negative)", -5.0),
    MetricSpec("drawdown", "Drawdown from peak", "%", "risk",
               "Distance below the highest point in the period (negative)", -15.0),
    MetricSpec("volatility", "Annualised volatility", "%", "risk",
               "21-day realised volatility, annualised", 40.0),
    # ---- AI
    MetricSpec("ai_score", "AI recommendation score", "", "ai",
               "Ensemble conviction, -1 (sell) to +1 (buy)", 0.4),
    MetricSpec("rl_signal", "RL agent signal", "", "ai",
               "Trained agent's action: -1 sell, 0 hold, +1 buy", 1.0),
    MetricSpec("regime_probability", "Market regime probability", "%", "ai",
               "Confidence in the detected regime, 0-100%", 70.0),
)

METRICS_BY_KEY = {m.key: m for m in METRIC_SPECS}

OPERATORS = {
    "above": (">", lambda a, b: a > b),
    "below": ("<", lambda a, b: a < b),
    "equals": ("=", lambda a, b: abs(a - b) < 1e-9),
    "crosses_above": (">", lambda a, b: a > b),   # needs prior state; see below
    "crosses_below": ("<", lambda a, b: a < b),
}

# Regime and RL conditions compare a label, not a number.
CATEGORICAL_METRICS = {"regime", "ai_action"}

REGIME_CHOICES = ("bull_market", "bear_market", "sideways", "high_volatility",
                  "low_volatility", "recovery", "crash_risk")
ACTION_CHOICES = ("STRONG_BUY", "BUY", "HOLD", "SELL", "STRONG_SELL")


def _safe(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


class MetricResolver:
    """Computes metric values for one symbol over one period.

    Instantiated per evaluation pass so the history and the derived reports are
    fetched once and shared by every condition in the rule.
    """

    def __init__(self, symbol: str, period: str = "6mo") -> None:
        self.symbol = symbol.upper()
        self.period = period
        self._df: pd.DataFrame | None = None
        self._quote: dict | None = None
        self._risk: dict | None = None
        self._bubble: dict | None = None
        self._regime: dict | None = None
        self._recommendation: dict | None = None
        self._cache: dict[str, float | None] = {}

    # ------------------------------------------------------------ sources
    @property
    def df(self) -> pd.DataFrame | None:
        if self._df is None:
            try:
                self._df = market_data_service.get_history(
                    self.symbol, period=self.period).df
            except Exception as exc:
                logger.warning("history unavailable for %s: %s", self.symbol, exc)
                self._df = pd.DataFrame()
        return None if self._df is None or self._df.empty else self._df

    @property
    def quote(self) -> dict:
        if self._quote is None:
            try:
                self._quote = market_data_service.get_quote(self.symbol) or {}
            except Exception as exc:
                logger.warning("quote unavailable for %s: %s", self.symbol, exc)
                self._quote = {}
        return self._quote

    def _risk_report(self) -> dict:
        if self._risk is None:
            from app.services.risk.anomaly import anomaly_detector
            frame = self.df
            self._risk = anomaly_detector.crash_risk(frame) if frame is not None else {}
        return self._risk

    def _bubble_report(self) -> dict:
        if self._bubble is None:
            from app.services.risk.anomaly import anomaly_detector
            frame = self.df
            self._bubble = anomaly_detector.bubble_indicator(frame) if frame is not None else {}
        return self._bubble

    def regime_report(self) -> dict:
        if self._regime is None:
            from app.services.risk.regime import market_regime_detector
            frame = self.df
            # timeline_step is large on purpose: an alert needs today's label,
            # not a full history, and the history walk is the slow part.
            self._regime = (market_regime_detector.detect(
                self.symbol, frame, timeline_step=200) if frame is not None else {})
        return self._regime

    def recommendation(self) -> dict:
        if self._recommendation is None:
            try:
                from app.services.recommendation.engine import recommendation_engine
                self._recommendation = recommendation_engine.recommend(
                    self.symbol, period=self.period)
            except Exception as exc:
                logger.warning("recommendation unavailable for %s: %s", self.symbol, exc)
                self._recommendation = {}
        return self._recommendation

    # ------------------------------------------------------------ metrics
    def value(self, metric: str) -> float | None:
        if metric in self._cache:
            return self._cache[metric]
        fn: Callable[[], float | None] | None = getattr(self, f"_m_{metric}", None)
        if fn is None:
            logger.warning("unknown alert metric %r", metric)
            result = None
        else:
            try:
                result = fn()
            except Exception as exc:
                logger.warning("metric %s failed for %s: %s", metric, self.symbol, exc)
                result = None
        self._cache[metric] = result
        return result

    def label(self, metric: str) -> str | None:
        """Categorical counterpart of `value` for regime / action conditions."""
        if metric == "regime":
            return (self.regime_report() or {}).get("regime")
        if metric == "ai_action":
            return (self.recommendation() or {}).get("action")
        return None

    # -- price -----------------------------------------------------------
    def _m_price(self) -> float | None:
        return _safe(self.quote.get("price"))

    def _m_pct_change(self) -> float | None:
        return _safe(self.quote.get("change_percent"))

    def _m_volume_ratio(self) -> float | None:
        frame = self.df
        if frame is None or "volume" not in frame or len(frame) < 21:
            return None
        recent = _safe(frame["volume"].iloc[-1])
        baseline = _safe(frame["volume"].tail(21).mean())
        if not recent or not baseline:
            return None
        return recent / baseline

    # -- technical -------------------------------------------------------
    def _m_rsi(self) -> float | None:
        from app.services.indicators.technical import rsi
        frame = self.df
        if frame is None or len(frame) < 20:
            return None
        return _safe(rsi(frame["close"], 14).iloc[-1])

    def _m_macd_hist(self) -> float | None:
        from app.services.indicators.technical import macd
        frame = self.df
        if frame is None or len(frame) < 40:
            return None
        out = macd(frame["close"])
        col = "macd_hist" if "macd_hist" in out else out.columns[-1]
        return _safe(out[col].iloc[-1])

    def _m_bb_position(self) -> float | None:
        from app.services.indicators.technical import bollinger_bands
        frame = self.df
        if frame is None or len(frame) < 25:
            return None
        bands = bollinger_bands(frame["close"])
        cols = list(bands.columns)
        upper = next((c for c in cols if "upper" in c), None)
        lower = next((c for c in cols if "lower" in c), None)
        if not upper or not lower:
            return None
        hi, lo = _safe(bands[upper].iloc[-1]), _safe(bands[lower].iloc[-1])
        price = _safe(frame["close"].iloc[-1])
        if hi is None or lo is None or price is None or hi <= lo:
            return None
        return (price - lo) / (hi - lo)

    def _m_sma_distance(self) -> float | None:
        from app.services.indicators.technical import sma
        frame = self.df
        if frame is None or len(frame) < 50:
            return None
        ma = _safe(sma(frame["close"], 50).iloc[-1])
        price = _safe(frame["close"].iloc[-1])
        if not ma or price is None:
            return None
        return (price / ma - 1) * 100

    def _m_ma_cross(self) -> float | None:
        from app.services.indicators.technical import sma
        frame = self.df
        if frame is None or len(frame) < 200:
            return None
        fast = _safe(sma(frame["close"], 50).iloc[-1])
        slow = _safe(sma(frame["close"], 200).iloc[-1])
        if not fast or not slow:
            return None
        return (fast / slow - 1) * 100

    # -- risk ------------------------------------------------------------
    def _m_crash_risk(self) -> float | None:
        score = (self._risk_report() or {}).get("crash_risk_score")
        return None if score is None else _safe(score) * 100

    def _m_bubble_score(self) -> float | None:
        score = (self._bubble_report() or {}).get("bubble_score")
        return None if score is None else _safe(score) * 100

    def _m_var_95(self) -> float | None:
        value = (self._risk_report() or {}).get("var_95_daily")
        return None if value is None else _safe(value) * 100

    def _m_cvar_95(self) -> float | None:
        value = (self._risk_report() or {}).get("cvar_95_daily")
        return None if value is None else _safe(value) * 100

    def _m_drawdown(self) -> float | None:
        value = (self._risk_report() or {}).get("current_drawdown")
        return None if value is None else _safe(value) * 100

    def _m_volatility(self) -> float | None:
        frame = self.df
        if frame is None or len(frame) < 22:
            return None
        returns = frame["close"].pct_change().dropna()
        return _safe(returns.tail(21).std() * np.sqrt(252) * 100)

    # -- AI --------------------------------------------------------------
    def _m_ai_score(self) -> float | None:
        # The engine returns `composite_score` (risk-adjusted); `score` does not
        # exist and silently resolved to None, so every AI condition was dead.
        return _safe((self.recommendation() or {}).get("composite_score"))

    def _m_rl_signal(self) -> float | None:
        rec = self.recommendation() or {}
        for contribution in rec.get("signals", []) or []:
            if str(contribution.get("source", "")).lower().startswith("rl"):
                return _safe(contribution.get("score"))
        return None

    def _m_regime_probability(self) -> float | None:
        prob = (self.regime_report() or {}).get("probability")
        return None if prob is None else _safe(prob) * 100
