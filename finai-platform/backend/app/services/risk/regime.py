"""Market-regime detection: which environment are we actually in?

Why this exists
---------------
Every other number on the platform is conditional on a regime. A momentum
signal that works in a trending market whipsaws in a range; a VaR estimate
fitted on calm data understates a crisis. Naming the regime makes that
conditionality explicit instead of leaving it implicit in every other panel.

How the classification works
----------------------------
Two layers, deliberately kept separate:

1. **Evidence layer.** Eight transparent factors (trend, moving-average
   structure, momentum, RSI, MACD, volatility level, volatility direction,
   volume and drawdown, plus optional news sentiment). Each is scored to a
   bounded [-1, +1] and votes for the regimes it supports. Every vote is
   returned, so a classification can be audited rather than trusted.

2. **Statistical layer.** A Gaussian mixture over (rolling return, rolling
   volatility) provides an unsupervised second opinion. It never overrides the
   evidence layer; it either corroborates it or flags the disagreement, which
   is itself information — the two disagree precisely at turning points.

Probabilities come from a softmax over the regime scores. They are *relative
plausibilities across the seven labels*, not calibrated frequencies: nothing
here has been backtested against realised regime labels, because no such
ground truth exists. The API says so in `confidence_basis` rather than
implying a rigour the method does not have.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.core.logging import get_logger
from app.services.indicators.technical import adx, macd, rsi, sma

logger = get_logger(__name__)

# The seven labels the product asks for. Ordered from most bearish to most
# bullish where that ordering is meaningful, with the two volatility states and
# crash risk as cross-cutting conditions.
REGIMES = (
    "crash_risk",
    "bear_market",
    "high_volatility",
    "sideways",
    "low_volatility",
    "recovery",
    "bull_market",
)

REGIME_LABELS = {
    "crash_risk": "Crash Risk",
    "bear_market": "Bear Market",
    "high_volatility": "High Volatility",
    "sideways": "Sideways",
    "low_volatility": "Low Volatility",
    "recovery": "Recovery",
    "bull_market": "Bull Market",
}

# Action per regime. Deliberately conservative: the cost of being defensive in
# a bull market is opportunity, the cost of being long into a crash is capital.
REGIME_ACTIONS = {
    "crash_risk": ("HEDGE", "Hedge tail risk and cut gross exposure. Correlations "
                            "converge toward 1 in a crash, so diversification "
                            "stops helping exactly when it is needed."),
    "bear_market": ("REDUCE", "Reduce exposure and favour quality. Rallies inside "
                              "a downtrend are frequent and usually short-lived."),
    "high_volatility": ("REDUCE", "Cut position sizes rather than direction. The "
                                  "same conviction needs a smaller position when "
                                  "daily ranges double."),
    "sideways": ("HOLD", "Range strategies over trend-following. Breakout signals "
                         "whipsaw here; keep sizes modest and wait for direction."),
    "low_volatility": ("HOLD", "Calm conditions favour carry and patience. Note "
                               "that suppressed volatility precedes expansion more "
                               "often than it persists."),
    "recovery": ("BUY", "Early-trend conditions: momentum is turning up from a "
                        "drawdown. Scale in rather than commit at once — failed "
                        "recoveries look identical at the start."),
    "bull_market": ("BUY", "Trend-following is favoured. Maintain trailing stops; "
                           "the risk in a bull market is giving back the gain."),
}

MODEL_RELIABILITY = {
    "crash_risk": "low — models trained on normal conditions extrapolate badly",
    "bear_market": "moderate — trend signals hold, mean-reversion often fails",
    "high_volatility": "low — parameter estimates are unstable at this vol level",
    "sideways": "moderate — mean-reversion favoured, trend signals whipsaw",
    "low_volatility": "good — but calm regimes end abruptly",
    "recovery": "moderate — the turn is only confirmed in hindsight",
    "bull_market": "good — most historical patterns remain applicable",
}


def _safe(value: float, default: float = 0.0) -> float:
    """NaN and inf reach the JSON encoder as invalid tokens; stop them here."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    return v if np.isfinite(v) else default


def _clip(value: float, low: float = -1.0, high: float = 1.0) -> float:
    return float(np.clip(_safe(value), low, high))


class MarketRegimeDetector:
    """Classify the current regime from price, volume and (optionally) news."""

    MIN_BARS = 120

    # ------------------------------------------------------------- factors
    def _factors(self, df: pd.DataFrame) -> list[dict]:
        """Score each piece of evidence to [-1, +1], with a readable detail.

        Positive means "supports a constructive regime", negative means
        "supports a defensive one". Volatility factors are signed so that
        *higher* volatility is negative.
        """
        close = df["close"]
        returns = close.pct_change().dropna()
        factors: list[dict] = []

        # ---- trend: 63-day price change, saturating at +/-20%
        trend = _safe(close.iloc[-1] / close.iloc[-63] - 1) if len(close) > 63 else 0.0
        factors.append({
            "name": "Trend (63d)",
            "score": _clip(trend / 0.20),
            "value": round(trend * 100, 2),
            "unit": "%",
            "detail": f"Price is {trend * 100:+.1f}% over the last 63 sessions",
        })

        # ---- moving-average structure: the classic golden/death cross
        sma50 = sma(close, 50)
        sma200 = sma(close, 200) if len(close) >= 200 else None
        last_price = _safe(close.iloc[-1])
        ma_score, ma_detail = 0.0, "Not enough history for a 200-day average"
        if sma200 is not None and np.isfinite(_safe(sma200.iloc[-1], np.nan)):
            s50, s200 = _safe(sma50.iloc[-1]), _safe(sma200.iloc[-1])
            spread = (s50 / s200 - 1) if s200 else 0.0
            above = last_price > s200
            # Spread of +/-5% between the averages is a decisive structure.
            ma_score = _clip(spread / 0.05) * 0.7 + (0.3 if above else -0.3)
            ma_detail = (f"50-day is {spread * 100:+.1f}% vs the 200-day; price is "
                         f"{'above' if above else 'below'} the 200-day")
        elif np.isfinite(_safe(sma50.iloc[-1], np.nan)):
            s50 = _safe(sma50.iloc[-1])
            ma_score = _clip((last_price / s50 - 1) / 0.05) if s50 else 0.0
            ma_detail = f"Price is {(last_price / s50 - 1) * 100:+.1f}% vs its 50-day average"
        factors.append({
            "name": "Moving averages", "score": _clip(ma_score),
            "value": round(_safe(sma50.iloc[-1]), 2), "unit": "",
            "detail": ma_detail,
        })

        # ---- momentum: 21-day rate of change
        mom = _safe(close.iloc[-1] / close.iloc[-21] - 1) if len(close) > 21 else 0.0
        factors.append({
            "name": "Momentum (21d)", "score": _clip(mom / 0.10),
            "value": round(mom * 100, 2), "unit": "%",
            "detail": f"{mom * 100:+.1f}% over the last month of trading",
        })

        # ---- RSI(14): centred on 50, saturating at the 30/70 bands
        rsi_now = _safe(rsi(close, 14).iloc[-1], 50.0)
        factors.append({
            "name": "RSI (14)", "score": _clip((rsi_now - 50) / 20),
            "value": round(rsi_now, 1), "unit": "",
            "detail": (f"{rsi_now:.0f} — "
                       + ("overbought" if rsi_now > 70 else
                          "oversold" if rsi_now < 30 else "neutral range")),
        })

        # ---- MACD histogram, normalised by price so it compares across assets
        macd_df = macd(close)
        hist = _safe(macd_df["macd_hist"].iloc[-1]) if "macd_hist" in macd_df else 0.0
        hist_pct = (hist / last_price * 100) if last_price else 0.0
        factors.append({
            "name": "MACD", "score": _clip(hist_pct / 1.0),
            "value": round(hist_pct, 3), "unit": "% of price",
            "detail": (f"Histogram {hist_pct:+.2f}% of price — "
                       + ("bullish" if hist > 0 else "bearish") + " crossover state"),
        })

        # ---- volatility level: 21-day realised vs the full-sample average
        vol_recent = _safe(returns.tail(21).std() * np.sqrt(252))
        vol_long = _safe(returns.std() * np.sqrt(252)) or 1e-9
        vol_ratio = vol_recent / vol_long
        factors.append({
            "name": "Volatility regime", "score": _clip(-(vol_ratio - 1) / 0.8),
            "value": round(vol_ratio, 2), "unit": "x",
            "detail": (f"21-day realised {vol_recent * 100:.1f}% vs period average "
                       f"{vol_long * 100:.1f}% ({vol_ratio:.2f}x)"),
        })

        # ---- volatility direction: is it expanding or contracting?
        vol_prev = _safe(returns.tail(63).head(42).std() * np.sqrt(252)) or 1e-9
        vol_change = vol_recent / vol_prev - 1
        factors.append({
            "name": "Volatility trend", "score": _clip(-vol_change / 0.5),
            "value": round(vol_change * 100, 1), "unit": "%",
            "detail": (f"Realised volatility is {'expanding' if vol_change > 0.05 else 'contracting' if vol_change < -0.05 else 'stable'}"
                       f" ({vol_change * 100:+.0f}% vs the prior window)"),
        })

        # ---- volume: is the move backed by participation?
        vol_score, vol_detail = 0.0, "No volume data for this instrument"
        if "volume" in df and _safe(df["volume"].sum()) > 0:
            v_recent = _safe(df["volume"].tail(21).mean())
            v_long = _safe(df["volume"].tail(252).mean()) or 1e-9
            v_ratio = v_recent / v_long
            # Volume confirms the move it accompanies, so the sign follows the
            # trend. Note the asymmetry: *heavy* volume amplifies whatever
            # direction is in force, *light* volume undercuts it. Multiplying a
            # negative (below-average) reading by the trend sign produced
            # "confirming the up move" next to a negative score, which is the
            # opposite of what thin participation means.
            participation = _clip((v_ratio - 1) / 0.5)
            vol_score = participation * (1 if trend >= 0 else -1)
            if abs(v_ratio - 1) <= 0.15:
                strength = "neither confirming nor contradicting"
            elif v_ratio > 1:
                strength = "confirming"
            else:
                strength = "failing to confirm"
            vol_detail = (f"21-day volume is {v_ratio:.2f}x the yearly average, "
                          f"{strength} the {'up' if trend >= 0 else 'down'} move")
            factors.append({"name": "Volume", "score": vol_score,
                            "value": round(v_ratio, 2), "unit": "x",
                            "detail": vol_detail})
        else:
            factors.append({"name": "Volume", "score": 0.0, "value": None,
                            "unit": "", "detail": vol_detail})

        # ---- drawdown from the period peak
        equity = (1 + returns).cumprod()
        dd = _safe(equity.iloc[-1] / equity.cummax().iloc[-1] - 1)
        factors.append({
            "name": "Drawdown", "score": _clip(dd / 0.20),
            "value": round(dd * 100, 2), "unit": "%",
            "detail": f"{dd * 100:.1f}% below the highest point in this period",
        })

        # ---- ADX: is there a trend at all, or just noise?
        adx_val = 0.0
        try:
            adx_df = adx(df, 14)
            col = "adx" if "adx" in adx_df else adx_df.columns[0]
            adx_val = _safe(adx_df[col].iloc[-1])
        except Exception as exc:      # pragma: no cover - indicator edge cases
            logger.debug("ADX unavailable: %s", exc)
        factors.append({
            "name": "Trend strength (ADX)",
            # ADX is direction-agnostic: it scores "is there a trend", so it is
            # reported but contributes through the sideways test, not as a
            # bullish or bearish vote.
            "score": 0.0,
            "value": round(adx_val, 1), "unit": "",
            "detail": (f"ADX {adx_val:.0f} — "
                       + ("strong trend" if adx_val >= 25 else
                          "weak or absent trend" if adx_val < 20 else "developing trend")),
            "directional": False,
        })
        return factors

    # ------------------------------------------------------------- scoring
    def _regime_scores(self, factors: dict[str, float], context: dict) -> dict[str, float]:
        """Turn factor scores into one score per regime.

        Written as explicit sums rather than a trained model on purpose: with no
        labelled regime data to learn from, a fitted classifier would encode the
        author's assumptions while hiding them behind weights.
        """
        trend = factors["Trend (63d)"]
        ma = factors["Moving averages"]
        mom = factors["Momentum (21d)"]
        rsi_f = factors["RSI (14)"]
        macd_f = factors["MACD"]
        vol_dir = factors["Volatility trend"]         # negative = expanding
        volume = factors["Volume"]

        vol_ratio = context["volatility_ratio"]
        vol_abs = context["annualised_volatility"]
        drawdown = context["drawdown"]
        adx_val = context["adx"]
        slope_r2 = context["trend_r2"]
        range_pos = context["range_position"]
        sentiment = context.get("sentiment_score", 0.0)

        bullish = float(np.mean([trend, ma, mom, macd_f, max(rsi_f, -0.5)]))
        bearish = -bullish

        # Volatility has to be judged in absolute terms as well as relative.
        # `vol_ratio` compares the last 21 days with the period average, so a
        # market that is uniformly violent scores ~1.0 and looks calm. A 50%
        # annualised random walk was being called Crash Risk purely on the
        # drawdown it wandered into.
        vol_elevated = max(vol_abs - 0.28, 0) / 0.22        # 28% -> 50% ramps 0..1
        vol_suppressed = max(0.16 - vol_abs, 0) / 0.10      # below 16% ramps up

        # A crash is a volatility *event*: a change, not a level. Allowing a
        # high absolute level to satisfy the shock test on its own meant a
        # permanently wild instrument (a small-cap, a crypto pair) read as
        # crashing every single day. Expansion against its own norm is the
        # necessary condition; a high absolute level only intensifies it.
        expansion = max(vol_ratio - 1.35, 0) / 0.55
        crash_shock = expansion * (1.0 + 0.5 * min(vol_elevated, 1.0))
        scores = {
            "crash_risk": (
                1.8 * crash_shock
                # Damage only counts toward a crash when it arrived violently.
                + 1.4 * max(-drawdown - 0.15, 0) / 0.15 * min(crash_shock, 1.0)
                + 0.6 * max(-vol_dir, 0) * min(crash_shock + 0.3, 1.0)
                + 0.4 * max(bearish, 0)
            ),
            "bear_market": (
                1.6 * max(bearish, 0)
                + 1.0 * max(-drawdown - 0.10, 0) / 0.15
                + 0.6 * max(-trend, 0)
                + 0.3 * max(-sentiment, 0)
            ),
            "high_volatility": (
                1.5 * max(max(vol_ratio - 1.25, 0) / 0.6, vol_elevated)
                + 0.7 * max(-vol_dir, 0)
                + 0.3 * abs(mom)
            ),
            # Sideways means "goes nowhere", which is about the *path*, not just
            # a small net change. Rewarding a small |trend| alone made a steady
            # compounding uptrend — which never moves far in any 63-day window —
            # score higher on sideways than on bull. The R² of a linear fit and
            # the position inside the recent range measure directionlessness
            # properly: a trend has high R², a range does not.
            "sideways": (
                1.4 * max(1 - slope_r2 / 0.35, 0)
                + 0.9 * max(1 - adx_val / 22, 0)
                + 0.7 * max(1 - abs(trend) / 0.12, 0)
                + 0.5 * (1 - abs(range_pos - 0.5) * 2)
            ),
            "low_volatility": (
                1.5 * max(vol_suppressed, max(1.0 - vol_ratio, 0) / 0.35)
                + 0.7 * max(vol_dir, 0)
                # Calm is not the same as directionless; do not reward a trend.
                + 0.3 * max(1 - abs(mom), 0)
            ),
            "recovery": (
                1.6 * max(mom, 0) * float(drawdown < -0.07)
                + 1.0 * max(trend, 0) * float(drawdown < -0.05)
                + 0.6 * max(vol_dir, 0)
                + 0.5 * max(macd_f, 0) * float(drawdown < -0.05)
            ),
            "bull_market": (
                1.8 * max(bullish, 0)
                + 1.0 * max(trend, 0)
                + 0.8 * max(ma, 0)
                # A persistent, well-fitted uptrend is the defining feature.
                + 0.9 * slope_r2 * float(trend > 0)
                + 0.3 * max(volume, 0)
                + 0.3 * max(sentiment, 0)
                # A deep drawdown disqualifies "bull"; that is recovery at best.
                - 1.2 * float(drawdown < -0.12)
            ),
        }
        # Suppress the directional labels while dispersion dominates: calling a
        # crash "a bear market" is a materially different instruction.
        if vol_ratio > 1.6 or vol_abs > 0.55:
            scores["bull_market"] *= 0.5
            scores["low_volatility"] = 0.0
        return {k: float(max(v, 0.0)) for k, v in scores.items()}

    @staticmethod
    def _probabilities(scores: dict[str, float]) -> dict[str, float]:
        """Softmax over regime scores.

        These are relative plausibilities across the seven labels, not
        backtested frequencies — there is no ground-truth regime series to
        calibrate against, and the API says so rather than implying otherwise.
        """
        keys = list(scores)
        values = np.array([scores[k] for k in keys], dtype=float)
        if not np.any(values > 0):
            # No evidence for anything: report an honest uniform prior.
            return {k: round(1.0 / len(keys), 4) for k in keys}
        # Temperature 0.55 sharpens the winner without collapsing to 100%.
        exp = np.exp((values - values.max()) / 0.55)
        probs = exp / exp.sum()
        return {k: round(float(p), 4) for k, p in zip(keys, probs, strict=False)}

    # --------------------------------------------------------- statistical
    @staticmethod
    def _gaussian_mixture_view(returns: pd.Series) -> dict | None:
        """Unsupervised second opinion over (rolling return, rolling vol)."""
        try:
            from sklearn.mixture import GaussianMixture

            feats = pd.DataFrame({
                "ret": returns.rolling(21).mean(),
                "vol": returns.rolling(21).std(),
            }).dropna()
            if len(feats) < 120:
                return None
            gmm = GaussianMixture(n_components=3, random_state=42, n_init=3)
            labels = gmm.fit_predict(feats.values)
            by_vol = np.argsort(gmm.means_[:, 1])
            names = {int(by_vol[0]): "calm", int(by_vol[1]): "normal",
                     int(by_vol[2]): "turbulent"}
            current = names[int(labels[-1])]
            posterior = gmm.predict_proba(feats.values[-1:])[0]
            return {
                "cluster": current,
                "confidence": round(float(posterior.max()), 3),
                "note": (f"An unsupervised mixture model places today in its "
                         f"'{current}' cluster, fitted without seeing any labels."),
            }
        except Exception as exc:      # pragma: no cover
            logger.debug("GMM regime view failed: %s", exc)
            return None

    # -------------------------------------------------------------- history
    def history(self, df: pd.DataFrame, step: int = 5, window: int = 252) -> list[dict]:
        """Walk the series and classify each step, for the timeline.

        Each point is classified using only the bars up to that date — no
        look-ahead. Recomputing the full factor set every 5 sessions is the
        expensive part, which is why `step` exists.
        """
        points: list[dict] = []
        if len(df) < self.MIN_BARS + step:
            return points
        for end in range(self.MIN_BARS, len(df) + 1, step):
            slice_df = df.iloc[max(0, end - window):end]
            if len(slice_df) < self.MIN_BARS:
                continue
            try:
                verdict = self._classify(slice_df, sentiment=None, with_history=False)
            except Exception as exc:      # pragma: no cover
                logger.debug("history point failed at %s: %s", end, exc)
                continue
            points.append({
                "date": str(slice_df.index[-1].date()),
                "regime": verdict["regime"],
                "label": REGIME_LABELS[verdict["regime"]],
                "probability": verdict["probability"],
                "close": round(_safe(slice_df["close"].iloc[-1]), 2),
            })
        return points

    @staticmethod
    def transitions(timeline: list[dict]) -> list[dict]:
        """Collapse the timeline into contiguous regime spells."""
        spells: list[dict] = []
        for point in timeline:
            if spells and spells[-1]["regime"] == point["regime"]:
                spells[-1]["to"] = point["date"]
                spells[-1]["points"] += 1
            else:
                spells.append({
                    "regime": point["regime"], "label": point["label"],
                    "from": point["date"], "to": point["date"], "points": 1,
                })
        return spells

    # ------------------------------------------------------------- classify
    def _classify(self, df: pd.DataFrame, sentiment: dict | None,
                  with_history: bool) -> dict:
        close = df["close"]
        returns = close.pct_change().dropna()
        factor_list = self._factors(df)
        factors = {f["name"]: f["score"] for f in factor_list}

        vol_recent = _safe(returns.tail(21).std() * np.sqrt(252))
        vol_long = _safe(returns.std() * np.sqrt(252)) or 1e-9
        equity = (1 + returns).cumprod()
        adx_val = next((f["value"] for f in factor_list
                        if f["name"] == "Trend strength (ADX)"), 0.0) or 0.0

        # How well a straight line explains the last quarter of log-prices.
        # This is what separates "drifted upward relentlessly" from "wandered
        # and ended up in the same place": both can show a small net change,
        # only the first has a high R².
        window = close.tail(63)
        slope_r2 = 0.0
        if len(window) >= 20:
            y = np.log(window.to_numpy(dtype=float))
            x = np.arange(len(y), dtype=float)
            fit = np.polyfit(x, y, 1)
            resid = y - (fit[0] * x + fit[1])
            var = float(np.var(y))
            slope_r2 = float(1 - np.var(resid) / var) if var > 1e-12 else 0.0
            slope_r2 = float(np.clip(slope_r2, 0.0, 1.0))

        # Where the last close sits inside the recent range: mid-range is the
        # signature of a market going nowhere.
        lo, hi = float(window.min()), float(window.max())
        range_pos = float((close.iloc[-1] - lo) / (hi - lo)) if hi > lo else 0.5

        context = {
            "volatility_ratio": vol_recent / vol_long,
            "annualised_volatility": vol_recent,
            "drawdown": _safe(equity.iloc[-1] / equity.cummax().iloc[-1] - 1),
            "adx": adx_val,
            "trend_r2": slope_r2,
            "range_position": range_pos,
            "sentiment_score": (sentiment or {}).get("score", 0.0),
        }

        scores = self._regime_scores(factors, context)
        probabilities = self._probabilities(scores)
        regime = max(probabilities, key=probabilities.get)
        top_p = probabilities[regime]
        runner_up = sorted(probabilities.values(), reverse=True)[1]

        # Confidence is the margin over the next-best label, not the raw
        # probability: a 30% winner in a seven-way tie is not a confident call.
        margin = top_p - runner_up
        confidence = float(np.clip(0.35 + margin * 2.2, 0.0, 0.95))

        out = {
            "regime": regime,
            "label": REGIME_LABELS[regime],
            "probability": round(top_p, 4),
            "confidence": round(confidence, 3),
            "probabilities": probabilities,
            "factors": factor_list,
            "context": {k: round(_safe(v), 4) for k, v in context.items()},
        }
        if with_history:
            out["statistical_view"] = self._gaussian_mixture_view(returns)
        return out

    # ------------------------------------------------------------- public
    def detect(self, symbol: str, df: pd.DataFrame, sentiment: dict | None = None,
               timeline_step: int = 5) -> dict:
        """Full regime report for one instrument."""
        if df is None or len(df) < self.MIN_BARS:
            return {
                "symbol": symbol.upper(),
                "regime": "unknown",
                "label": "Unknown",
                "probability": None,
                "confidence": None,
                "reason": (f"Needs {self.MIN_BARS} daily bars to classify a regime; "
                           f"this window has {0 if df is None else len(df)}."),
                "factors": [], "probabilities": {}, "timeline": [], "transitions": [],
            }

        verdict = self._classify(df, sentiment, with_history=True)
        regime = verdict["regime"]
        action, rationale = REGIME_ACTIONS[regime]

        timeline = self.history(df, step=timeline_step)
        transitions = self.transitions(timeline)
        # How long the current spell has lasted, in classified steps.
        current_spell = transitions[-1] if transitions else None

        supporting = sorted(
            [f for f in verdict["factors"] if f.get("directional", True)],
            key=lambda f: abs(f["score"]), reverse=True)[:4]

        out = {
            "symbol": symbol.upper(),
            "as_of": str(df.index[-1].date()),
            "period_start": str(df.index[0].date()),
            "bars_analysed": int(len(df)),
            **verdict,
            "action": action,
            "action_rationale": rationale,
            "model_reliability": MODEL_RELIABILITY[regime],
            "key_factors": supporting,
            "timeline": timeline,
            "transitions": transitions,
            "current_spell": current_spell,
            "sentiment": sentiment,
            "confidence_basis": (
                "Confidence is the margin between the leading regime and the "
                "runner-up, not a backtested hit rate. Probabilities are relative "
                "plausibilities across the seven labels — there is no ground-truth "
                "regime series to calibrate them against."
            ),
        }
        out["insight"] = self._insight(out)
        return out

    # -------------------------------------------------------------- insight
    @staticmethod
    def _insight(report: dict) -> str:
        """A short written read of the evidence, generated from the factors.

        Composed from the computed values rather than a language model: the
        text can then never claim something the numbers do not support.
        """
        regime = report["regime"]
        label = report["label"]
        pieces: list[str] = []
        prob = report["probability"] or 0
        conf = report["confidence"] or 0

        strength = ("a clear reading" if conf > 0.7
                    else "a tentative reading" if conf > 0.5
                    else "a genuinely ambiguous reading")
        pieces.append(f"{label} at {prob * 100:.0f}% probability — {strength}.")

        top = report.get("key_factors", [])[:3]
        if top:
            named = "; ".join(f"{f['name'].lower()} ({f['detail'].rstrip('.')})"
                              for f in top)
            pieces.append(f"The classification is driven mainly by {named}.")

        stat = report.get("statistical_view")
        if stat:
            agrees = (
                (stat["cluster"] == "turbulent" and regime in ("crash_risk", "high_volatility", "bear_market"))
                or (stat["cluster"] == "calm" and regime in ("low_volatility", "sideways", "bull_market"))
                or stat["cluster"] == "normal"
            )
            pieces.append(
                f"{stat['note']} That {'corroborates' if agrees else 'disagrees with'} "
                f"the factor-based call"
                + ("." if agrees else " — disagreement is common at turning points.")
            )

        runner = sorted(report["probabilities"].items(), key=lambda kv: -kv[1])
        if len(runner) > 1 and runner[1][1] > 0.20:
            pieces.append(
                f"{REGIME_LABELS[runner[1][0]]} remains a live alternative at "
                f"{runner[1][1] * 100:.0f}%.")
        return " ".join(pieces)


market_regime_detector = MarketRegimeDetector()
