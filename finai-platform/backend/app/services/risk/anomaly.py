"""Anomaly detection and market-regime / crash-risk analytics.

Detectors implemented
---------------------
* Volatility spikes             - rolling z-score of realised volatility
* Return outliers               - modified z-score (MAD) on daily returns
* Volume anomalies              - z-score on log volume
* Structural breaks             - CUSUM on standardised returns
* Multivariate outliers         - Isolation Forest on the engineered features
* Bubble detection              - price vs. long-run trend + momentum + vol regime
* Crash risk                    - composite score (VaR, drawdown, tail, correlation)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from app.core.logging import get_logger
from app.services.indicators.features import build_features
from app.services.indicators.technical import atr, rsi

logger = get_logger(__name__)

SEVERITY_ORDER = {"low": 0, "medium": 1, "high": 2, "critical": 3}


def _severity(z: float) -> str:
    a = abs(z)
    if a >= 5:
        return "critical"
    if a >= 4:
        return "high"
    if a >= 3:
        return "medium"
    return "low"


class AnomalyDetector:
    # ------------------------------------------------------- single signals
    def volatility_spikes(self, df: pd.DataFrame, window: int = 21, threshold: float = 2.5) -> list[dict]:
        returns = df["close"].pct_change()
        vol = returns.rolling(window).std() * np.sqrt(252)
        z = (vol - vol.rolling(window * 3, min_periods=window).mean()) / \
            vol.rolling(window * 3, min_periods=window).std().replace(0, np.nan)
        hits = z[z > threshold].dropna()
        return [
            {"date": str(idx.date()), "type": "volatility_spike", "z_score": round(float(v), 2),
             "volatility": round(float(vol.loc[idx]) * 100, 2), "severity": _severity(v),
             "description": f"Realised volatility {vol.loc[idx]*100:.1f}% is {v:.1f}σ above its regime average"}
            for idx, v in hits.items()
        ]

    def return_outliers(self, df: pd.DataFrame, threshold: float = 3.5) -> list[dict]:
        returns = df["close"].pct_change().dropna()
        median = returns.median()
        mad = (returns - median).abs().median()
        if mad == 0:
            return []
        modified_z = 0.6745 * (returns - median) / mad
        hits = modified_z[modified_z.abs() > threshold]
        return [
            {"date": str(idx.date()), "type": "return_outlier", "z_score": round(float(v), 2),
             "return_pct": round(float(returns.loc[idx]) * 100, 2), "severity": _severity(v),
             "direction": "up" if v > 0 else "down",
             "description": f"Daily move of {returns.loc[idx]*100:+.2f}% ({v:+.1f} modified σ)"}
            for idx, v in hits.items()
        ]

    def volume_anomalies(self, df: pd.DataFrame, window: int = 20, threshold: float = 3.0) -> list[dict]:
        if "volume" not in df or df["volume"].sum() == 0:
            return []
        log_vol = np.log1p(df["volume"])
        z = (log_vol - log_vol.rolling(window).mean()) / log_vol.rolling(window).std().replace(0, np.nan)
        hits = z[z > threshold].dropna()
        return [
            {"date": str(idx.date()), "type": "volume_anomaly", "z_score": round(float(v), 2),
             "volume": float(df["volume"].loc[idx]), "severity": _severity(v),
             "description": f"Trading volume {v:.1f}σ above its 20-day norm"}
            for idx, v in hits.items()
        ]

    def structural_breaks(self, df: pd.DataFrame, threshold: float = 4.0) -> list[dict]:
        """CUSUM test on standardised returns -> regime-change candidates."""
        returns = df["close"].pct_change().dropna()
        if len(returns) < 60:
            return []
        standardised = (returns - returns.mean()) / (returns.std() or 1)
        cusum_pos, cusum_neg = 0.0, 0.0
        breaks = []
        for idx, value in standardised.items():
            cusum_pos = max(0.0, cusum_pos + value - 0.5)
            cusum_neg = min(0.0, cusum_neg + value + 0.5)
            if cusum_pos > threshold or cusum_neg < -threshold:
                breaks.append({
                    "date": str(idx.date()), "type": "structural_break",
                    "direction": "bullish" if cusum_pos > threshold else "bearish",
                    "statistic": round(float(max(cusum_pos, abs(cusum_neg))), 2),
                    "severity": "high",
                    "description": "Persistent drift detected by CUSUM - possible regime change",
                })
                cusum_pos, cusum_neg = 0.0, 0.0
        return breaks

    def isolation_forest(self, df: pd.DataFrame, contamination: float = 0.02) -> list[dict]:
        features = build_features(df, dropna=True)
        if len(features) < 60:
            return []
        model = IsolationForest(n_estimators=150, contamination=contamination,
                                random_state=42, n_jobs=1)
        preds = model.fit_predict(features.values)
        scores = model.score_samples(features.values)
        out = []
        for i, (idx, pred) in enumerate(zip(features.index, preds, strict=False)):
            if pred == -1:
                contributors = features.loc[idx].abs().sort_values(ascending=False).head(3)
                out.append({
                    "date": str(idx.date()), "type": "multivariate_anomaly",
                    "anomaly_score": round(float(-scores[i]), 4),
                    "severity": "high" if -scores[i] > 0.65 else "medium",
                    "top_features": {k: round(float(v), 4) for k, v in contributors.items()},
                    "description": "Unusual joint behaviour across technical features (Isolation Forest)",
                })
        return out

    # ------------------------------------------------------------ regimes
    def bubble_indicator(self, df: pd.DataFrame) -> dict:
        close = df["close"]
        if len(close) < 200:
            # `None`, not 0.0. A score of zero renders as a green "no bubble"
            # bar, which states the opposite of what is true: the answer is
            # unknown, not reassuring. The trend fit needs a long-run baseline
            # to measure a deviation against.
            return {
                "bubble_score": None,
                "level": "insufficient_data",
                "bars_available": int(len(close)),
                "bars_required": 200,
                "interpretation": (
                    f"Needs 200 daily bars to fit a long-run trend; this window has "
                    f"{len(close)}. Select a longer period to compute it."
                ),
            }
        log_price = np.log(close)
        x = np.arange(len(log_price))
        slope, intercept = np.polyfit(x, log_price.values, 1)
        trend = slope * x + intercept
        deviation = float(log_price.values[-1] - trend[-1])
        resid_std = float(np.std(log_price.values - trend))
        # `or 1e-9` only caught an exact zero. On a series that hugs its trend
        # (a synthetic ramp, a pegged rate, a stale quote) resid_std collapses
        # to ~1e-16 of floating-point residue, and dividing by it turned pure
        # rounding noise into a 4-sigma "stretch": a flat line scored as a
        # moderate bubble. Below a floor of 0.5% residual dispersion there is no
        # meaningful trend channel to be stretched away from.
        MIN_RESID_STD = 0.005
        stretch = deviation / resid_std if resid_std >= MIN_RESID_STD else 0.0

        mom_3m = float(close.iloc[-1] / close.iloc[-63] - 1) if len(close) > 63 else 0.0
        # When there is no year of history, this used to silently reuse the
        # 3-month figure and report it as 12-month momentum — the same number
        # then entered the score twice, under two different weights, and the UI
        # displayed a "12-month" value measured over three. Annualise the
        # longest window actually available and say what was used instead.
        has_12m = len(close) > 252
        if has_12m:
            mom_12m = float(close.iloc[-1] / close.iloc[-252] - 1)
            mom_12m_basis = "252 bars"
        else:
            span = len(close) - 1
            raw = float(close.iloc[-1] / close.iloc[0] - 1)
            # Scale to a 252-day equivalent so the weighting stays comparable.
            mom_12m = float((1 + raw) ** (252 / span) - 1) if span > 0 else 0.0
            mom_12m_basis = f"{span} bars, annualised"
        rsi_now = float(rsi(close, 14).iloc[-1])
        vol_recent = float(close.pct_change().tail(21).std() * np.sqrt(252))
        vol_long = float(close.pct_change().std() * np.sqrt(252)) or 1e-9
        vol_ratio = vol_recent / vol_long

        score = float(np.clip(
            0.35 * np.clip(stretch / 2.5, 0, 1)
            + 0.25 * np.clip(mom_12m / 1.0, 0, 1)
            + 0.20 * np.clip(mom_3m / 0.4, 0, 1)
            + 0.10 * np.clip((rsi_now - 60) / 25, 0, 1)
            + 0.10 * np.clip((vol_ratio - 1) / 1.2, 0, 1), 0, 1))
        level = ("extreme" if score > 0.8 else "elevated" if score > 0.6
                 else "moderate" if score > 0.4 else "low")
        return {
            "bubble_score": round(score, 3), "level": level,
            # The score is a weighted sum of five bounded terms. Publishing the
            # breakdown turns an unexplained 0.41 into something a user can
            # check: it shows which input drove it and how far each can go.
            "scale": {
                "min": 0.0, "max": 1.0,
                "bands": {"low": "0-40%", "moderate": "40-60%",
                          "elevated": "60-80%", "extreme": "80-100%"},
            },
            "components": [
                {"name": "Trend deviation", "weight": 0.35,
                 "value": round(float(np.clip(stretch / 2.5, 0, 1)), 3),
                 "detail": f"{stretch:.2f}σ above the fitted log-price trend (caps at 2.5σ)"},
                {"name": "12-month momentum", "weight": 0.25,
                 "value": round(float(np.clip(mom_12m / 1.0, 0, 1)), 3),
                 "detail": f"{mom_12m * 100:+.1f}% over {mom_12m_basis} (caps at +100%)"},
                {"name": "3-month momentum", "weight": 0.20,
                 "value": round(float(np.clip(mom_3m / 0.4, 0, 1)), 3),
                 "detail": f"{mom_3m * 100:+.1f}% over 63 bars (caps at +40%)"},
                {"name": "RSI(14)", "weight": 0.10,
                 "value": round(float(np.clip((rsi_now - 60) / 25, 0, 1)), 3),
                 "detail": f"{rsi_now:.1f} (contributes above 60, caps at 85)"},
                {"name": "Volatility regime", "weight": 0.10,
                 "value": round(float(np.clip((vol_ratio - 1) / 1.2, 0, 1)), 3),
                 "detail": f"21-day vol is {vol_ratio:.2f}x the period average"},
            ],
            "trend_deviation_sigma": round(stretch, 2),
            "momentum_3m": round(mom_3m, 4), "momentum_12m": round(mom_12m, 4),
            "momentum_12m_basis": mom_12m_basis,
            "rsi": round(rsi_now, 1), "volatility_ratio": round(vol_ratio, 2),
            "interpretation": {
                "extreme": "Price far above long-run trend with euphoric momentum - high reversal risk",
                "elevated": "Signs of speculative overheating - tighten risk controls",
                "moderate": "Somewhat stretched but within historical norms",
                "low": "No evidence of bubble dynamics",
            }[level],
        }

    def crash_risk(self, df: pd.DataFrame) -> dict:
        returns = df["close"].pct_change().dropna()
        if len(returns) < 60:
            # None rather than 0.0: see bubble_indicator. A zero here reads as
            # "no crash risk" when the truth is "not enough data to say".
            return {
                "crash_risk_score": None,
                "level": "insufficient_data",
                "bars_available": int(len(returns)),
                "bars_required": 60,
                "recommendation": (
                    f"Needs 60 daily returns to estimate tail risk; this window has "
                    f"{len(returns)}. Select a longer period to compute it."
                ),
            }

        var_95 = float(np.percentile(returns, 5))
        cvar_95 = float(returns[returns <= var_95].mean())
        skew = float(returns.skew())
        kurt = float(returns.kurtosis())
        equity = (1 + returns).cumprod()
        current_dd = float(equity.iloc[-1] / equity.cummax().iloc[-1] - 1)
        vol_recent = float(returns.tail(21).std() * np.sqrt(252))
        vol_long = float(returns.std() * np.sqrt(252)) or 1e-9
        atr_pct = float((atr(df, 14).iloc[-1] / df["close"].iloc[-1]) * 100)
        downside_streak = int((returns.tail(10) < 0).sum())

        # `kurt` is already EXCESS kurtosis: pandas .kurtosis() is Fisher's
        # definition, which reports 0 for a normal distribution rather than 3.
        # Subtracting 3 again here removed the entire tail-risk contribution for
        # anything below excess kurtosis 3 — a t(5)-like return series, decidedly
        # fat-tailed, scored 0.12 on this term instead of 0.50. Dividing by 8
        # keeps the original saturation point: excess kurtosis >= 8 maxes it out.
        score = float(np.clip(
            0.22 * np.clip(vol_recent / vol_long - 1, 0, 1.5) / 1.5
            + 0.20 * np.clip(-current_dd / 0.25, 0, 1)
            + 0.18 * np.clip(max(-skew, 0) / 1.5, 0, 1)
            + 0.15 * np.clip(kurt / 8, 0, 1)
            + 0.15 * np.clip(-cvar_95 / 0.06, 0, 1)
            + 0.10 * (downside_streak / 10), 0, 1))
        level = ("critical" if score > 0.75 else "high" if score > 0.55
                 else "moderate" if score > 0.35 else "low")
        return {
            "crash_risk_score": round(score, 3), "level": level,
            "scale": {
                "min": 0.0, "max": 1.0,
                "bands": {"low": "0-35%", "moderate": "35-55%",
                          "high": "55-75%", "critical": "75-100%"},
            },
            "components": [
                {"name": "Volatility regime", "weight": 0.22,
                 "value": round(float(np.clip(vol_recent / vol_long - 1, 0, 1.5) / 1.5), 3),
                 "detail": (f"21-day vol {vol_recent * 100:.1f}% vs period "
                            f"{vol_long * 100:.1f}% (caps at 2.5x)")},
                {"name": "Current drawdown", "weight": 0.20,
                 "value": round(float(np.clip(-current_dd / 0.25, 0, 1)), 3),
                 "detail": f"{current_dd * 100:.1f}% below the period peak (caps at -25%)"},
                {"name": "Negative skew", "weight": 0.18,
                 "value": round(float(np.clip(max(-skew, 0) / 1.5, 0, 1)), 3),
                 "detail": (f"skew {skew:+.2f}"
                            f"{' — only negative skew adds risk' if skew >= 0 else ''}")},
                {"name": "Fat tails", "weight": 0.15,
                 "value": round(float(np.clip(kurt / 8, 0, 1)), 3),
                 "detail": f"excess kurtosis {kurt:.1f} (0 = normal, caps at 8)"},
                {"name": "Expected shortfall", "weight": 0.15,
                 "value": round(float(np.clip(-cvar_95 / 0.06, 0, 1)), 3),
                 "detail": f"CVaR95 {cvar_95 * 100:.2f}% per day (caps at -6%)"},
                {"name": "Recent down days", "weight": 0.10,
                 "value": round(downside_streak / 10, 3),
                 "detail": f"{downside_streak} of the last 10 sessions closed lower"},
            ],
            "var_95_daily": round(var_95, 4), "cvar_95_daily": round(cvar_95, 4),
            "skewness": round(skew, 3), "excess_kurtosis": round(kurt, 3),
            "current_drawdown": round(current_dd, 4),
            "volatility_regime": round(vol_recent / vol_long, 2),
            "atr_pct": round(atr_pct, 2), "down_days_last_10": downside_streak,
            "recommendation": {
                "critical": "Reduce exposure materially and hedge tail risk",
                "high": "Trim positions, tighten stops, raise cash",
                "moderate": "Maintain exposure with active risk monitoring",
                "low": "Normal risk conditions",
            }[level],
        }

    # ----------------------------------------------------------- aggregate
    def scan(self, symbol: str, df: pd.DataFrame, lookback_days: int = 180,
             crash_bars: int | None = None, bubble_bars: int | None = None) -> dict:
        # `lookback_days` is calendar days; `.tail(n)` counts trading bars. The
        # two were used interchangeably, so `max(lookback_days, 220)` meant the
        # 220-bar floor governed every request up to lookback_days=220, and the
        # detectors saw an identical 220-bar slice whether the user asked for
        # 1 year or 10. Selecting a longer period changed nothing on screen.
        #
        # Convert properly (~252 trading days per 365 calendar days) and keep a
        # warm-up margin, because the rolling windows below need history before
        # the reporting window to have anything to compare against.
        WARMUP_BARS = 220
        wanted_bars = int(round(lookback_days * 252 / 365)) + WARMUP_BARS
        recent = df.tail(wanted_bars) if len(df) > wanted_bars else df
        # Every detector below derives its baseline from `recent` — rolling
        # mean/std, the MAD, the CUSUM origin, the Isolation Forest fit. When a
        # short period could not supply the full warm-up, that baseline was
        # computed on fewer bars, so the *same* 180-day question returned 23
        # anomalies at period=1y and 21 at period=5y. Report the shortfall
        # rather than let it silently change the answer.
        warmup_available = max(0, len(recent) - int(round(lookback_days * 252 / 365)))
        warmup_complete = warmup_available >= WARMUP_BARS
        anomalies: list[dict] = []
        anomalies += self.volatility_spikes(recent)
        anomalies += self.return_outliers(recent)
        anomalies += self.volume_anomalies(recent)
        anomalies += self.structural_breaks(recent)
        try:
            anomalies += self.isolation_forest(recent)
        except Exception as exc:
            logger.warning("isolation forest failed for %s: %s", symbol, exc)

        # Clamp to what the period actually contains. Asking for a 180-day
        # window on three months of data reported "anomalies since 2026-02-04"
        # when nothing before 2026-05-04 had been examined — a window three
        # months wider than the evidence behind it.
        requested_cutoff = recent.index[-1] - pd.Timedelta(days=lookback_days)
        effective_cutoff = max(requested_cutoff, recent.index[0])
        cutoff = str(effective_cutoff.date())
        window_truncated = bool(requested_cutoff < recent.index[0])
        anomalies = [a for a in anomalies if a["date"] >= cutoff]
        anomalies.sort(key=lambda a: (a["date"], SEVERITY_ORDER.get(a.get("severity", "low"), 0)), reverse=True)

        # Each score reads its own window: the number of bars the user selected,
        # floored at what that specific model needs. Passing the whole fetched
        # frame instead made every range from 1D to 1Y produce one identical
        # answer, because all seven over-fetch to the same 2y of history.
        crash_df = df.tail(crash_bars) if crash_bars else df
        bubble_df = df.tail(bubble_bars) if bubble_bars else df
        bubble = self.bubble_indicator(bubble_df)
        crash = self.crash_risk(crash_df)
        counts: dict[str, int] = {}
        for a in anomalies:
            counts[a["type"]] = counts.get(a["type"], 0) + 1

        # ---- overall level, and why it is what it is
        #
        # Two faults were fixed here. The anomaly term keyed off "the five most
        # recent items" — but the list is sorted by date *and severity*, so a
        # months-old structural break could sit in that slice and pin the
        # headline to HIGH while every current measure said otherwise (GLD:
        # driven by an anomaly 136 days stale). It is now restricted to the last
        # 21 sessions, so "elevated risk" means elevated *now*.
        #
        # Second, insufficient data was silently reported as "low" — absence of
        # evidence shown as evidence of safety. It now surfaces as "unknown".
        RANK = {"unknown": -1, "low": 0, "moderate": 1, "high": 2, "critical": 3}
        as_of = recent.index[-1]
        recent_cutoff = str((as_of - pd.Timedelta(days=31)).date())
        fresh = [a for a in anomalies
                 if a["date"] >= recent_cutoff
                 and a.get("severity") in ("high", "critical")]

        drivers: list[dict] = []
        if crash["level"] == "insufficient_data":
            components = ["unknown"]
            drivers.append({"source": "crash_risk", "level": "unknown",
                            "detail": "not enough history to estimate tail risk"})
        else:
            components = [crash["level"]]
            drivers.append({"source": "crash_risk", "level": crash["level"],
                            "detail": f"score {crash['crash_risk_score']}"})

        if bubble.get("level") in ("extreme", "elevated"):
            components.append("high")
            drivers.append({"source": "bubble", "level": "high",
                            "detail": f"{bubble['level']} (score {bubble['bubble_score']})"})

        if fresh:
            components.append("high")
            drivers.append({
                "source": "recent_anomalies", "level": "high",
                "detail": (f"{len(fresh)} high-severity anomal"
                           f"{'y' if len(fresh) == 1 else 'ies'} in the last 31 days "
                           f"(most recent {fresh[0]['date']})")})

        risk_level = max(components, key=lambda x: RANK.get(x, 0))

        # Continuous version of the "fresh anomalies" signal, for the weighted
        # Overall Risk Score. Four or more high-severity hits in a month
        # saturates it; the band-maximum above cannot express "one" vs "nine".
        ANOMALY_BUDGET = 4
        anomaly_pressure = min(len(fresh) / ANOMALY_BUDGET, 1.0)

        return {
            "symbol": symbol.upper(),
            "as_of": str(recent.index[-1].date()),
            "overall_risk_level": risk_level,
            # What produced the headline, so the badge can be explained rather
            # than taken on faith when it disagrees with the two scores below.
            "risk_drivers": drivers,
            "window_start": str(recent.index[0].date()),
            "bars_analysed": int(len(recent)),
            "anomaly_lookback_days": lookback_days,
            "n_anomalies": len(anomalies),
            "counts_by_type": counts,
            "recent_high_severity": len(fresh),
            "anomaly_pressure": round(anomaly_pressure, 3),
            "anomalies": anomalies[:60],
            "bubble": bubble,
            "crash_risk": crash,
            # Which sample produced which number. Crash risk and the bubble read
            # the whole selected period; the anomaly list reads only the lookback
            # window. Without this the two disagree on screen for no visible
            # reason, and a drawdown measured from a 10-year peak looks like the
            # same quantity as one measured from a 1-year peak.
            "basis": {
                "scores_from": {
                    "bars": int(len(crash_df)),
                    "start": str(crash_df.index[0].date()),
                    "end": str(crash_df.index[-1].date()),
                    "note": ("Drawdown is measured from the highest point inside the "
                             "window, so a longer period can show a deeper drawdown "
                             "for the same price today."),
                },
                # The two scores have different data floors (60 vs 200 bars), so
                # on a short selection they genuinely read different windows.
                # Reporting one figure for both was a claim the page could not
                # support.
                "crash_window": {
                    "bars": int(len(crash_df)),
                    "start": str(crash_df.index[0].date()),
                    "end": str(crash_df.index[-1].date()),
                },
                "bubble_window": {
                    "bars": int(len(bubble_df)),
                    "start": str(bubble_df.index[0].date()),
                    "end": str(bubble_df.index[-1].date()),
                },
                "anomalies_from": {
                    "bars": int(len(recent)),
                    "reported_since": cutoff,
                    "requested_lookback_days": lookback_days,
                    "window_truncated": window_truncated,
                    "warmup_bars": int(warmup_available),
                    "warmup_complete": bool(warmup_complete),
                    "note": (
                        f"The {lookback_days}-day window was shortened to the start of "
                        f"the selected period ({cutoff}); choose a longer period to "
                        f"look further back."
                        if window_truncated else
                        "Detectors need history before the window to have a baseline "
                        "to compare against."
                        if warmup_complete else
                        f"Only {warmup_available} baseline bars were available before "
                        f"this window (wants {WARMUP_BARS}); counts may differ from a "
                        f"longer period covering the same dates."),
                },
            },
        }


anomaly_detector = AnomalyDetector()
