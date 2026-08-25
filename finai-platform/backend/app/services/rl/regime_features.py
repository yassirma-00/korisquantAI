"""Regime awareness for the RL environment.

What this does
--------------
Turns the existing Market Regime Detection module into a per-bar feature block
the RL agent can observe, and into a per-bar risk context the reward function
can react to. It **reuses** ``app.services.risk.regime.market_regime_detector``
— there is no second classifier here, and no duplicated thresholds.

Three constraints shaped the design.

**1. Cost.** ``_classify`` was measured at ~9.8 ms per call on a 400-bar slice.
Calling it inside ``step()`` would add ~245 s to a 25 000-step training run and
scale linearly with every extra episode. It is therefore evaluated **once per
bar** over the price series (7.4 s for 752 bars, done once and cached), and the
environment reads a precomputed row. Same numbers, ~30x cheaper for a short run
and far more beyond that.

**2. No look-ahead.** Bar *t* is classified from ``df.iloc[:t+1]`` only. This
mirrors ``RegimeDetector.history``, which is already leak-free. An agent that
could see tomorrow's regime would post excellent backtests and fail live — the
exact class of bug that cost this project eight trained agents once already.

**3. Backward compatibility.** Adding columns to the observation changes its
shape, and a trained network refuses a vector of the wrong width:

    RuntimeError: mat1 and mat2 shapes cannot be multiplied (1x42 and 36x128)

There are 11 trained agents on disk. So regime awareness is **opt-in**
(``EnvConfig.regime_aware``) and the flag is persisted in each agent's
metadata; an old agent keeps loading into a 36-dimensional environment, a new
one into a 42-dimensional one.

The feature block (6 values, all bounded)
-----------------------------------------
================  ==========================================================
``regime_risk``   0 = calm, 1 = crash. Ordinal severity of the regime.
``regime_bull``   Signed directional bias: +1 bull, -1 bear.
``confidence``    How firmly the classifier holds its call, in [0, 1].
``vol_ratio``     21-day volatility over its long-run average, rescaled.
``crash_prob``    Probability mass on the crash-risk regime.
``drawdown``      Current drawdown from the running peak, in [-1, 0].
================  ==========================================================

Bounded on purpose: an unbounded input can dominate a network's first layer,
and these sit alongside features already normalised to roughly this range.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from app.core.logging import get_logger
from app.services.risk.regime import REGIMES, market_regime_detector

logger = get_logger(__name__)

# Number of columns appended to the observation. Anything reading the
# observation width must use this rather than a literal.
REGIME_FEATURE_DIM = 6

REGIME_FEATURE_NAMES = (
    "regime_risk",
    "regime_bull",
    "regime_confidence",
    "regime_vol_ratio",
    "regime_crash_prob",
    "regime_drawdown",
)

# Ordinal severity, 0 (calm) to 1 (crash). The agent needs a *scalar* it can
# learn a monotone response to; a one-hot of seven regimes would spend most of
# its capacity on distinctions that do not change the risk decision.
REGIME_RISK = {
    "low_volatility": 0.00,
    "bull_market": 0.15,
    "recovery": 0.30,
    "sideways": 0.40,
    "high_volatility": 0.70,
    "bear_market": 0.80,
    "crash_risk": 1.00,
}

# Directional bias, -1 (bearish) to +1 (bullish). Separate from severity
# because "risky" and "falling" are different questions: a violent rally is
# high-severity and bullish at once.
REGIME_BULL = {
    "bull_market": 1.00,
    "recovery": 0.55,
    "low_volatility": 0.20,
    "sideways": 0.00,
    "high_volatility": -0.25,
    "bear_market": -0.85,
    "crash_risk": -1.00,
}

# How the reward is re-weighted per regime. Risk aversion rises with severity:
# the penalty a policy pays for volatility and drawdown is multiplied by these.
#
# Rationale, and it is deliberately asymmetric: the cost of being defensive in
# a bull market is forgone gain, while the cost of being long into a crash is
# capital that may never come back. Being wrong in the two directions is not
# equally expensive, so the multipliers are not symmetric either.
REGIME_RISK_AVERSION = {
    "low_volatility": 0.70,
    "bull_market": 0.80,
    "recovery": 0.95,
    "sideways": 1.00,
    "high_volatility": 1.60,
    "bear_market": 1.90,
    "crash_risk": 2.50,
}

# Minimum bars the detector needs before it will classify at all.
MIN_BARS = market_regime_detector.MIN_BARS


@dataclass(frozen=True)
class RegimeRow:
    """One bar's regime reading, as consumed by the env and the explainer."""

    regime: str
    label: str
    confidence: float
    probability: float
    risk: float
    bull: float
    vol_ratio: float
    crash_prob: float
    drawdown: float
    risk_aversion: float
    classified: bool          # False -> a neutral placeholder, not a real call

    def to_vector(self) -> np.ndarray:
        return np.array([
            self.risk,
            self.bull,
            self.confidence,
            np.clip(self.vol_ratio / 3.0, 0.0, 1.0),
            self.crash_prob,
            np.clip(self.drawdown, -1.0, 0.0),
        ], dtype=np.float32)


def _neutral_row() -> RegimeRow:
    """What to emit before enough history exists to classify.

    Not zeros-as-calm: ``risk`` sits at the 'sideways' level and confidence at
    0, so the agent is told "unknown, assume ordinary" rather than the much
    stronger and unearned claim "measured, and safe".
    """
    return RegimeRow(
        regime="unknown", label="Unknown", confidence=0.0, probability=0.0,
        risk=REGIME_RISK["sideways"], bull=0.0, vol_ratio=1.0, crash_prob=0.0,
        drawdown=0.0, risk_aversion=1.0, classified=False)


class RegimeFeatureProvider:
    """Per-bar regime features, computed once and reused.

    One instance per price series. ``build`` walks the series forward, so bar
    *t* only ever sees bars up to *t*.
    """

    def __init__(self, step: int = 5, window: int = 252) -> None:
        # `step` trades resolution for time: the regime is re-classified every
        # `step` bars and held constant in between. Regimes persist for weeks,
        # so 5 sessions costs almost nothing in fidelity and cuts the build to
        # a fifth. `window` caps how much history each classification reads,
        # matching RegimeDetector.history.
        self.step = max(1, int(step))
        self.window = int(window)
        self._rows: list[RegimeRow] = []
        self._index: pd.DatetimeIndex | None = None

    # ------------------------------------------------------------- building
    def build(self, df: pd.DataFrame) -> RegimeFeatureProvider:
        """Classify every bar of ``df`` using only past data."""
        n = len(df)
        self._index = df.index
        rows: list[RegimeRow] = []
        neutral = _neutral_row()
        last: RegimeRow = neutral
        next_classify = MIN_BARS

        for t in range(n):
            if t + 1 < MIN_BARS:
                rows.append(neutral)
                continue
            if t + 1 >= next_classify:
                start = max(0, t + 1 - self.window)
                # `t + 1` is exclusive, so this ends at bar t. No look-ahead.
                slice_df = df.iloc[start:t + 1]
                last = self._classify_slice(slice_df) or last
                next_classify = t + 1 + self.step
            rows.append(last)

        self._rows = rows
        logger.debug("regime features built for %d bars (%d classifications)",
                     n, max(0, (n - MIN_BARS) // self.step + 1))
        return self

    def _classify_slice(self, slice_df: pd.DataFrame) -> RegimeRow | None:
        try:
            verdict = market_regime_detector._classify(
                slice_df, sentiment=None, with_history=False)
        except Exception as exc:      # pragma: no cover - defensive
            logger.debug("regime classification failed: %s", exc)
            return None

        regime = verdict.get("regime", "sideways")
        probabilities = verdict.get("probabilities", {}) or {}
        context = verdict.get("context", {}) or {}

        returns = slice_df["close"].pct_change().dropna()
        vol_ratio = float(context.get("volatility_ratio") or 0.0)
        if not np.isfinite(vol_ratio) or vol_ratio <= 0:
            recent = float(returns.tail(21).std() or 0.0)
            longrun = float(returns.std() or 0.0) or 1e-9
            vol_ratio = recent / longrun
        drawdown = float(context.get("drawdown") or 0.0)
        if not np.isfinite(drawdown):
            drawdown = 0.0

        return RegimeRow(
            regime=regime,
            label=verdict.get("label", regime.replace("_", " ").title()),
            confidence=float(np.clip(verdict.get("confidence") or 0.0, 0.0, 1.0)),
            probability=float(np.clip(probabilities.get(regime, 0.0), 0.0, 1.0)),
            risk=REGIME_RISK.get(regime, 0.4),
            bull=REGIME_BULL.get(regime, 0.0),
            vol_ratio=float(max(vol_ratio, 0.0)),
            crash_prob=float(np.clip(probabilities.get("crash_risk", 0.0), 0.0, 1.0)),
            drawdown=float(np.clip(drawdown, -1.0, 0.0)),
            risk_aversion=REGIME_RISK_AVERSION.get(regime, 1.0),
            classified=True,
        )

    # -------------------------------------------------------------- reading
    def at(self, t: int) -> RegimeRow:
        if not self._rows:
            return _neutral_row()
        return self._rows[min(max(t, 0), len(self._rows) - 1)]

    def vector_at(self, t: int) -> np.ndarray:
        return self.at(t).to_vector()

    def __len__(self) -> int:
        return len(self._rows)

    # ------------------------------------------------------------ reporting
    def summary(self) -> dict:
        """Distribution of regimes over the series, for training metadata."""
        classified = [r for r in self._rows if r.classified]
        if not classified:
            return {"classified_bars": 0, "distribution": {}}
        counts: dict[str, int] = {}
        for row in classified:
            counts[row.regime] = counts.get(row.regime, 0) + 1
        total = len(classified)
        return {
            "classified_bars": total,
            "unclassified_bars": len(self._rows) - total,
            "distribution": {k: round(v / total, 4)
                             for k, v in sorted(counts.items(),
                                                key=lambda kv: -kv[1])},
            "regimes_seen": sorted(counts),
            "regimes_never_seen": sorted(set(REGIMES) - set(counts)),
        }


def build_provider(df: pd.DataFrame, step: int = 5,
                   window: int = 252) -> RegimeFeatureProvider:
    return RegimeFeatureProvider(step=step, window=window).build(df)
