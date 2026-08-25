"""Per-asset quantitative risk profile and the Overall Risk Score.

Why this module exists
----------------------
The page already had a Crash Risk Score and a Bubble Indicator, and both are
genuinely per-asset. What it did not have was a defensible *Overall Risk Score*.
The headline was ``max(crash_band, bubble_band, recent_anomaly_band)`` — an
ordinal maximum over three labels. Three consequences, all measured:

1. **It ignored absolute risk.** Every term in the crash score is *relative to
   the asset's own history*: ``21-day vol / period vol``, drawdown as a fraction
   of its own peak, its own skew. An instrument can therefore be calm relative
   to itself while being violent in absolute terms. Measured on real data,
   NVDA at 36.4% annualised volatility scored 0.269 ("low") while GLD at 28.2%
   scored 0.563 ("high"). A ladder of synthetic series identical in every
   respect except volatility produced 0.106 at 5% vol and 0.421 at 120% vol —
   a 24x change in real risk compressed into one and a half bands, and
   non-monotone at the top.

2. **A maximum discards evidence.** An asset that is mildly elevated on every
   single measure scored exactly the same as one that is mildly elevated on one
   and pristine on the rest.

3. **It was not a score.** The UI drew a band name, so "how far into High?" had
   no answer, and nothing could be ranked.

What this module adds is the missing layer: **absolute, annualised, textbook
risk measures for the selected symbol over the selected window**, combined with
the two existing relative scores into a single 0-100 Overall Risk Score whose
every contribution is published.

Design rules
------------
* Every number is derived from the passed DataFrame. No constants stand in for
  data, no value is shared between assets, nothing is cached across symbols.
* Each contributor is normalised by an **absolute** reference level drawn from
  quantitative-finance convention (see ``ANCHORS``), not by the asset's own
  history — otherwise the score cannot compare two assets, which is the entire
  requirement.
* Every contributor is bounded to [0, 1] so no single term can run away, and
  the weights sum to 1, so the composite is a genuine weighted mean.
* A contributor with no data is *dropped and its weight redistributed*, and the
  response says so. It is never silently treated as zero, because zero means
  "measured, and safe".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from app.core.logging import get_logger
from app.services.risk.metrics import (
    conditional_var,
    downside_deviation,
    drawdown_series,
    sharpe_ratio,
    sortino_ratio,
    value_at_risk,
)
from app.utils.timeseries import TRADING_DAYS, annualise_return, annualise_vol

logger = get_logger(__name__)

# Annual risk-free rate used for Sharpe/Sortino. Exposed so a caller can pass
# the prevailing short rate rather than inherit a stale constant.
DEFAULT_RISK_FREE = 0.02

# Minimum observations before a statistic is reported at all. Kurtosis and skew
# are badly biased on tiny samples; a beta on ten overlapping days is noise.
MIN_OBS_BASIC = 20
MIN_OBS_TAIL = 60
MIN_OBS_BETA = 60


@dataclass(frozen=True)
class Anchor:
    """An absolute reference scale for one risk dimension.

    ``low`` is the level at which the contribution is 0, ``high`` the level at
    which it saturates at 1. Both are stated in the metric's own units so the
    UI can show the user exactly what the score is measured against.
    """

    low: float
    high: float
    unit: str
    rationale: str

    def normalise(self, value: float) -> float:
        if self.high == self.low:
            return 0.0
        return float(np.clip((value - self.low) / (self.high - self.low), 0.0, 1.0))


# Absolute scales. These are calibration constants for a *scoring* function,
# not substitutes for measured data: each one is applied to a value computed
# from the asset's own returns. The levels follow common practice — roughly,
# a broad equity index sits near the bottom of each range and a leveraged or
# crypto instrument near the top.
ANCHORS: dict[str, Anchor] = {
    "volatility": Anchor(
        0.10, 0.80, "annualised σ",
        "10% is a broad-index-like level; 80% is crypto/single-name extreme."),
    "var": Anchor(
        0.010, 0.070, "daily VaR₉₅",
        "1% daily VaR is index-like; 7% is a highly speculative instrument."),
    "cvar": Anchor(
        0.015, 0.100, "daily CVaR₉₅",
        "Expected shortfall beyond VaR; saturates at a 10% average tail day."),
    "drawdown": Anchor(
        0.10, 0.60, "max drawdown",
        "A 10% correction is routine; 60% is a capital-impairing collapse."),
    "downside_deviation": Anchor(
        0.08, 0.60, "annualised downside σ",
        "Semi-deviation below the risk-free target."),
    "tail": Anchor(
        0.0, 8.0, "excess kurtosis",
        "0 is Gaussian; 8+ means extreme moves dominate the distribution."),
    "skew": Anchor(
        0.0, 1.5, "negative skew",
        "Only left-skew adds risk: crashes larger than rallies."),
    "beta": Anchor(
        0.5, 2.0, "|beta| to benchmark",
        "Systematic exposure that diversification cannot remove."),
}

# Weights of the Overall Risk Score. They sum to 1.0 (asserted below).
#
# The split is deliberate: two thirds of the weight sits on *absolute*,
# directly-measured quantities, and one third on the two *relative* composite
# scores that were previously the only inputs. That keeps crash and bubble
# dynamics influential without letting a self-referential ratio decide the
# headline on its own.
WEIGHTS: dict[str, float] = {
    "volatility": 0.20,
    "tail_risk": 0.16,          # CVaR₉₅, the coherent tail measure
    "drawdown": 0.14,
    "crash_risk": 0.18,         # existing relative composite
    "bubble": 0.10,             # existing relative composite
    "distribution": 0.10,       # skew + excess kurtosis
    "market_beta": 0.07,
    "recent_anomalies": 0.05,
}

assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9, "Overall Risk weights must sum to 1"

# Band edges for the 0-1 composite. Stated once, used by the API, the UI and
# the tests, so the label can never disagree with the number that produced it.
BANDS: tuple[tuple[str, float], ...] = (
    ("low", 0.30),
    ("moderate", 0.50),
    ("high", 0.72),
    ("critical", 1.01),
)


def classify(score: float) -> str:
    """Map a 0-1 composite onto its band. Single source of truth."""
    for name, upper in BANDS:
        if score < upper:
            return name
    return "critical"


def band_table() -> dict[str, str]:
    """Human-readable band edges, for the UI legend."""
    out, lower = {}, 0.0
    for name, upper in BANDS:
        top = min(upper, 1.0)
        out[name] = f"{lower * 100:.0f}-{top * 100:.0f}%"
        lower = top
    return out


# Which benchmark a given instrument should be measured against. Beta against
# an unrelated index is a meaningless number, so the mapping is by asset class
# rather than one hardcoded index for everything.
BENCHMARK_BY_CLASS: dict[str, str] = {
    "equity": "^GSPC",
    "etf": "^GSPC",
    "index": "^GSPC",
    "crypto": "BTC-USD",
    "commodity": "^GSPC",
    "forex": "^GSPC",
}


def benchmark_for(symbol: str) -> str | None:
    """Pick a benchmark for this instrument, or None when none is sensible."""
    from app.services.data.universe import get_instrument

    try:
        instrument = get_instrument(symbol)
    except Exception:      # pragma: no cover - defensive
        instrument = None
    # `get_instrument` returns None for anything outside the reference universe
    # rather than raising, so the except branch alone did not cover it.
    if instrument is None:
        return "^GSPC"
    if instrument.symbol.upper() == "^GSPC":
        return None        # an index is not its own benchmark
    if instrument.asset_class == "crypto" and instrument.symbol.upper() == "BTC-USD":
        return "^GSPC"
    return BENCHMARK_BY_CLASS.get(instrument.asset_class, "^GSPC")


def _contribution(key: str, name: str, raw: float | None, anchor: Anchor,
                  detail: str, weight: float) -> dict:
    """One row of the score breakdown."""
    # bool(...) is load-bearing: np.isfinite returns np.bool_, which Pydantic
    # cannot serialise ("'numpy.bool' object is not iterable") and which turns
    # the whole endpoint into a 500.
    available = bool(raw is not None and np.isfinite(raw))
    normalised = anchor.normalise(float(raw)) if available else None
    return {
        "key": key,
        "name": name,
        "weight": round(weight, 4),
        "raw": None if not available else round(float(raw), 6),
        "unit": anchor.unit,
        "value": None if normalised is None else round(normalised, 4),
        "points": None if normalised is None else round(normalised * weight * 100, 2),
        "max_points": round(weight * 100, 2),
        "scale_low": anchor.low,
        "scale_high": anchor.high,
        "detail": detail,
        "basis": anchor.rationale,
        "available": available,
    }


class RiskProfiler:
    """Computes every metric for one symbol over one window, from its own data."""

    def profile(self, symbol: str, df: pd.DataFrame,
                benchmark_df: pd.DataFrame | None = None,
                benchmark_symbol: str | None = None,
                crash: dict | None = None,
                bubble: dict | None = None,
                recent_anomaly_pressure: float | None = None,
                risk_free: float = DEFAULT_RISK_FREE) -> dict:
        """Full risk profile.

        ``df`` is the price history for *this* symbol over *this* window; every
        statistic below is derived from it, so two different symbols cannot
        produce the same output unless their returns are genuinely identical.
        """
        returns = df["close"].pct_change().dropna()
        n = len(returns)

        if n < MIN_OBS_BASIC:
            return {
                "symbol": symbol.upper(),
                "available": False,
                "observations": int(n),
                "observations_required": MIN_OBS_BASIC,
                "reason": (f"Needs {MIN_OBS_BASIC} returns to measure risk; this window "
                           f"has {n}."),
                "metrics": {}, "overall": None,
            }

        # ---------------------------------------------------------- measures
        ann_vol = annualise_vol(returns)
        ann_ret = annualise_return(returns)
        dd = drawdown_series(returns)
        max_dd = float(dd.min())
        cur_dd = float(dd.iloc[-1])
        dsd = downside_deviation(returns, risk_free / TRADING_DAYS)

        var95 = value_at_risk(returns, 0.95)
        var99 = value_at_risk(returns, 0.99)
        cvar95 = conditional_var(returns, 0.95)
        cvar99 = conditional_var(returns, 0.99)

        # Skew and kurtosis are unstable below ~60 points; report them as
        # unknown rather than as a confident number computed from noise.
        has_tail = n >= MIN_OBS_TAIL
        skew = float(returns.skew()) if has_tail else None
        kurt = float(returns.kurtosis()) if has_tail else None

        sharpe = sharpe_ratio(returns, risk_free)
        sortino = sortino_ratio(returns, risk_free)

        beta = alpha = corr = None
        overlap = 0
        if benchmark_df is not None and len(benchmark_df):
            beta, alpha, corr, overlap = self._beta_block(returns, benchmark_df, risk_free)

        # Realised vol of the most recent month, for context (not scored here:
        # the crash score already owns the "vol vs its own regime" dimension).
        vol_21 = (float(returns.tail(21).std() * np.sqrt(TRADING_DAYS))
                  if n >= 21 else None)

        metrics = {
            "observations": int(n),
            "period_start": str(df.index[0].date()),
            "period_end": str(df.index[-1].date()),
            "annualised_return": round(ann_ret, 4),
            "annualised_volatility": round(ann_vol, 4),
            "volatility_21d": None if vol_21 is None else round(vol_21, 4),
            "downside_deviation": round(dsd, 4),
            "max_drawdown": round(max_dd, 4),
            "current_drawdown": round(cur_dd, 4),
            "var_95_daily": None if var95 is None else round(var95, 5),
            "var_99_daily": None if var99 is None else round(var99, 5),
            "cvar_95_daily": None if cvar95 is None else round(cvar95, 5),
            "cvar_99_daily": None if cvar99 is None else round(cvar99, 5),
            "skewness": None if skew is None else round(skew, 3),
            "excess_kurtosis": None if kurt is None else round(kurt, 3),
            "sharpe_ratio": round(sharpe, 3),
            "sortino_ratio": round(sortino, 3),
            "beta": None if beta is None else round(beta, 3),
            "alpha": None if alpha is None else round(alpha, 4),
            "correlation_to_benchmark": None if corr is None else round(corr, 3),
            "benchmark": benchmark_symbol,
            "benchmark_overlap_days": int(overlap) if overlap else 0,
            "risk_free_rate": risk_free,
        }

        overall = self._overall(
            metrics=metrics, crash=crash, bubble=bubble,
            recent_anomaly_pressure=recent_anomaly_pressure)

        return {
            "symbol": symbol.upper(),
            "available": True,
            "metrics": metrics,
            "overall": overall,
            # Scaling a daily VaR to a 10-day horizon is the one place where a
            # convention silently misleads, so state it rather than imply it.
            "var_note": ("VaR and CVaR are daily, one-sided, and measured on this "
                         "window's own returns. √t scaling to longer horizons "
                         "assumes independence and understates clustered risk."),
        }

    # ------------------------------------------------------------------ beta
    @staticmethod
    def _beta_block(returns: pd.Series, benchmark_df: pd.DataFrame,
                    risk_free: float) -> tuple[float | None, float | None,
                                               float | None, int]:
        """Beta / alpha / correlation on the overlapping dates only."""
        from app.services.risk.metrics import beta_alpha

        bench = benchmark_df["close"].pct_change().dropna()

        def _norm(s: pd.Series) -> pd.Series:
            if isinstance(s.index, pd.DatetimeIndex):
                idx = s.index.tz_localize(None) if s.index.tz is not None else s.index
                return pd.Series(s.to_numpy(), index=idx.normalize())
            return s

        joined = pd.concat([_norm(returns), _norm(bench)], axis=1,
                           join="inner").dropna()
        if len(joined) < MIN_OBS_BETA:
            return None, None, None, len(joined)
        r, b = joined.iloc[:, 0], joined.iloc[:, 1]
        beta, alpha = beta_alpha(r, b, risk_free)
        corr = float(r.corr(b))
        return beta, alpha, (None if pd.isna(corr) else corr), len(joined)

    # --------------------------------------------------------------- scoring
    def _overall(self, metrics: dict, crash: dict | None, bubble: dict | None,
                 recent_anomaly_pressure: float | None) -> dict:
        """Weighted composite of eight bounded contributors.

        Unavailable contributors are removed and their weight is spread over
        the rest, so the composite stays on a 0-1 scale instead of being
        silently dragged toward zero by a missing input.
        """
        rows: list[dict] = []

        vol = metrics["annualised_volatility"]
        rows.append(_contribution(
            "volatility", "Volatility", vol, ANCHORS["volatility"],
            f"{vol * 100:.1f}% annualised (scored from 10% to 80%)",
            WEIGHTS["volatility"]))

        cvar = metrics["cvar_95_daily"]
        rows.append(_contribution(
            "tail_risk", "Tail risk (CVaR₉₅)",
            None if cvar is None else abs(cvar), ANCHORS["cvar"],
            "not enough returns to estimate the tail" if cvar is None
            else f"average loss on the worst 5% of days is {abs(cvar) * 100:.2f}%",
            WEIGHTS["tail_risk"]))

        mdd = abs(metrics["max_drawdown"])
        rows.append(_contribution(
            "drawdown", "Maximum drawdown", mdd, ANCHORS["drawdown"],
            f"peak-to-trough loss of {mdd * 100:.1f}% inside this window",
            WEIGHTS["drawdown"]))

        crash_score = (crash or {}).get("crash_risk_score")
        rows.append(_contribution(
            "crash_risk", "Crash Risk Score", crash_score,
            Anchor(0.0, 1.0, "score", "Composite of vol regime, drawdown, skew, "
                                      "tails, expected shortfall and losing streak."),
            "insufficient history" if crash_score is None
            else f"{crash_score * 100:.0f}% — {(crash or {}).get('level', '?')}",
            WEIGHTS["crash_risk"]))

        bubble_score = (bubble or {}).get("bubble_score")
        rows.append(_contribution(
            "bubble", "Bubble Indicator", bubble_score,
            Anchor(0.0, 1.0, "score", "Deviation from long-run trend plus momentum "
                                      "and volatility-regime confirmation."),
            "insufficient history" if bubble_score is None
            else f"{bubble_score * 100:.0f}% — {(bubble or {}).get('level', '?')}",
            WEIGHTS["bubble"]))

        # Distribution shape: left-skew and fat tails together, since either
        # alone understates how asymmetric a crash-prone series is.
        skew, kurt = metrics["skewness"], metrics["excess_kurtosis"]
        if skew is None or kurt is None:
            dist_raw, dist_detail = None, "needs 60 returns for a stable shape estimate"
        else:
            skew_part = ANCHORS["skew"].normalise(max(-skew, 0.0))
            kurt_part = ANCHORS["tail"].normalise(max(kurt, 0.0))
            dist_raw = 0.5 * skew_part + 0.5 * kurt_part
            dist_detail = (f"skew {skew:+.2f}"
                           f"{' (left-tailed)' if skew < 0 else ' (no left tail)'}, "
                           f"excess kurtosis {kurt:.1f}")
        rows.append(_contribution(
            "distribution", "Return distribution", dist_raw,
            Anchor(0.0, 1.0, "shape", "Half negative skew, half excess kurtosis."),
            dist_detail, WEIGHTS["distribution"]))

        beta = metrics["beta"]
        bench = metrics.get("benchmark") or "benchmark"
        rows.append(_contribution(
            "market_beta", "Market beta",
            None if beta is None else abs(beta), ANCHORS["beta"],
            (f"no overlapping history with {bench}" if beta is None
             else f"β {beta:+.2f} vs {bench} over "
                  f"{metrics.get('benchmark_overlap_days', 0)} shared days"),
            WEIGHTS["market_beta"]))

        rows.append(_contribution(
            "recent_anomalies", "Recent anomalies", recent_anomaly_pressure,
            Anchor(0.0, 1.0, "pressure",
                   "High-severity detector hits in the last 31 sessions."),
            "no anomaly scan for this window" if recent_anomaly_pressure is None
            else f"{recent_anomaly_pressure * 100:.0f}% of the 31-day alert budget used",
            WEIGHTS["recent_anomalies"]))

        # ---- redistribute the weight of anything unmeasurable
        available = [r for r in rows if r["available"]]
        missing = [r for r in rows if not r["available"]]
        total_weight = sum(r["weight"] for r in available)
        if total_weight <= 0:
            return {
                "score": None, "level": "unknown", "scale": band_table(),
                "contributions": rows,
                "explanation": "No risk contributor could be measured on this window.",
            }

        score = sum(r["value"] * r["weight"] for r in available) / total_weight
        score = float(np.clip(score, 0.0, 1.0))
        level = classify(score)

        # Effective points after redistribution, so the published contributions
        # actually add up to the published score.
        scale_up = 1.0 / total_weight
        for r in rows:
            if r["available"]:
                r["effective_weight"] = round(r["weight"] * scale_up, 4)
                r["points"] = round(r["value"] * r["weight"] * scale_up * 100, 2)
                r["max_points"] = round(r["weight"] * scale_up * 100, 2)
            else:
                r["effective_weight"] = 0.0
                r["points"] = None
                r["max_points"] = 0.0

        ranked = sorted((r for r in rows if r["available"]),
                        key=lambda r: r["points"], reverse=True)
        top = ranked[:3]

        return {
            "score": round(score, 4),
            "level": level,
            "scale": band_table(),
            "method": ("Weighted mean of bounded contributors, each normalised "
                       "against an absolute reference range rather than the "
                       "asset's own history, so scores are comparable between "
                       "assets."),
            "contributions": rows,
            "top_drivers": [{"name": r["name"], "points": r["points"],
                             "detail": r["detail"]} for r in top],
            "unmeasured": [{"name": r["name"], "reason": r["detail"]} for r in missing],
            "weight_redistributed": bool(missing),
            "explanation": self._explain(level, top, missing),
        }

    @staticmethod
    def _explain(level: str, top: list[dict], missing: list[dict]) -> str:
        if not top:
            return "Nothing could be measured on this window."
        drivers = ", ".join(f"{r['name'].lower()} ({r['points']:.0f} pts)" for r in top)
        text = f"Overall risk is {level}, driven mainly by {drivers}."
        if missing:
            names = ", ".join(r["name"].lower() for r in missing)
            text += (f" {len(missing)} contributor"
                     f"{'s' if len(missing) > 1 else ''} could not be measured "
                     f"({names}); the remaining weights were rescaled to keep the "
                     f"score on a 0-100 scale.")
        return text


risk_profiler = RiskProfiler()
