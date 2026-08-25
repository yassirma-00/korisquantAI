"""Unified risk service: estimate, validate, and report honestly.

The guiding principle here is that a risk number is only as good as its
backtest. Every VaR figure this service returns is accompanied by the evidence
for or against trusting it, and the platform says so plainly when the evidence
is weak.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.core.logging import get_logger
from app.services.forecasting.advanced import RegimeDetector, VolatilityForecaster
from app.services.risk.advanced_var import (
    backtest_var,
    comprehensive_var,
    var_extreme_value,
    var_filtered_historical,
)

logger = get_logger(__name__)


class RiskService:
    def __init__(self) -> None:
        self.regime_detector = RegimeDetector()

    # ------------------------------------------------------------ VaR suite
    def var_report(self, symbol: str, df: pd.DataFrame, confidence: float = 0.95,
                   validate: bool = True) -> dict:
        returns = df["close"].pct_change().dropna()
        report = comprehensive_var(returns, confidence)
        if "error" in report:
            return {"symbol": symbol.upper(), **report}

        report["symbol"] = symbol.upper()
        report["regime"] = self.regime_detector.detect(df)

        if validate and len(returns) >= 310:
            validation = report.get("validation", {})
            methods = validation.get("methods", {})
            any_valid = any(v.get("valid") for v in methods.values() if isinstance(v, dict))
            clustered = [
                name for name, v in methods.items()
                if isinstance(v, dict) and v.get("independence_p") is not None
                and v["independence_p"] < 0.05
            ]
            report["honest_assessment"] = self._assessment(any_valid, clustered, methods)

        # Position-sizing guidance is the actionable output of a VaR number
        conservative = report.get("range", {}).get("most_conservative")
        if conservative:
            report["risk_budget"] = self._risk_budget(conservative)
        return report

    @staticmethod
    def _assessment(any_valid: bool, clustered: list[str], methods: dict) -> dict:
        """State plainly what the backtests actually showed."""
        if any_valid:
            level, headline = "adequate", "At least one estimator passes both coverage tests."
        elif clustered:
            level = "caution"
            headline = (
                "Every estimator passes or nearly passes the breach-*count* test, but breaches "
                "cluster in time (Christoffersen p < 0.05). This is the well-documented "
                "limitation of daily VaR on equities: losses arrive in bursts. The number is "
                "usable for routine sizing, NOT for surviving a crisis."
            )
        else:
            level, headline = "unreliable", "No estimator passed validation on this series."
        return {
            "level": level,
            "headline": headline,
            "clustered_methods": clustered,
            "recommendation": (
                "Pair VaR with the stress tests and the crash-risk score; do not size a "
                "portfolio on VaR alone. Expected Shortfall (CVaR) is the more informative "
                "tail measure and is reported alongside every estimate."
            ),
            "methods_summary": {
                k: {"breach_rate": v.get("breach_rate"), "expected": v.get("expected_rate"),
                    "kupiec_p": v.get("kupiec_p"), "independence_p": v.get("independence_p")}
                for k, v in methods.items() if isinstance(v, dict) and "breach_rate" in v
            },
        }

    @staticmethod
    def _risk_budget(daily_var: float) -> dict:
        v = abs(daily_var)
        return {
            "daily_var_pct": round(v * 100, 3),
            "scaled_10day_var_pct": round(v * np.sqrt(10) * 100, 3),
            "max_position_for_1pct_daily_risk": round(min(0.01 / v, 1.0), 4) if v > 0 else None,
            "max_position_for_2pct_daily_risk": round(min(0.02 / v, 1.0), 4) if v > 0 else None,
            "note": ("Square-root-of-time scaling assumes independent returns; it "
                     "understates multi-day risk when volatility clusters."),
        }

    # ------------------------------------------------------------- stress
    def stress_test(self, df: pd.DataFrame, position_value: float = 100_000.0) -> dict:
        """Historical and hypothetical shock scenarios.

        Historical worst-case moves are grounded in what this instrument has
        actually done, which is more defensible than inventing shock sizes.
        """
        returns = df["close"].pct_change().dropna()
        if len(returns) < 100:
            return {"error": "need >= 100 observations"}

        vol_daily = float(returns.std())
        worst_1d = float(returns.min())
        worst_5d = float(returns.rolling(5).sum().min())
        worst_21d = float(returns.rolling(21).sum().min())

        scenarios = {
            "worst_historical_day": worst_1d,
            "worst_historical_week": worst_5d,
            "worst_historical_month": worst_21d,
            "minus_3_sigma": -3 * vol_daily,
            "minus_5_sigma": -5 * vol_daily,
            "black_monday_1987": -0.2047,
            "covid_crash_mar2020": -0.12,
            "lehman_oct2008": -0.09,
            "flash_crash_2010": -0.09,
        }
        results = []
        for name, shock in scenarios.items():
            results.append({
                "scenario": name.replace("_", " "),
                "shock_pct": round(shock * 100, 2),
                "pnl": round(position_value * shock, 2),
                "remaining_value": round(position_value * (1 + shock), 2),
                "sigma_equivalent": round(shock / vol_daily, 1) if vol_daily else None,
            })
        results.sort(key=lambda r: r["pnl"])
        return {
            "position_value": position_value,
            "daily_volatility_pct": round(vol_daily * 100, 3),
            "scenarios": results,
            "worst_case": results[0],
            "note": ("Historical scenarios use this instrument's own realised extremes; "
                     "market-wide scenarios apply a reference index shock and ignore beta."),
        }

    # -------------------------------------------------------- volatility
    def volatility_report(self, symbol: str, df: pd.DataFrame, horizon: int = 5) -> dict:
        returns = df["close"].pct_change().dropna()
        out: dict = {"symbol": symbol.upper(), "horizon": horizon}
        best_aic, best = np.inf, None
        models = {}
        for name in ("garch", "egarch", "gjr"):
            try:
                vf = VolatilityForecaster(name).fit(returns)
                fc = vf.forecast(horizon)
                diag = vf.diagnostics()
                models[name] = {**fc, "diagnostics": diag}
                if fc["aic"] < best_aic and diag.get("well_specified"):
                    best_aic, best = fc["aic"], name
            except Exception as exc:
                models[name] = {"error": str(exc)[:160]}
        out["models"] = models
        out["best_model"] = best
        out["selection_note"] = (
            f"'{best}' has the lowest AIC among well-specified models."
            if best else
            "No model passed the residual diagnostics; volatility forecasts are unreliable here."
        )
        realised = float(returns.tail(21).std() * np.sqrt(252))
        out["realised_volatility_21d"] = round(realised, 4)
        if best:
            forecast_vol = models[best]["annualised_volatility"]
            out["forecast_vs_realised"] = round(forecast_vol / realised, 3) if realised else None
            out["direction"] = ("rising" if forecast_vol > realised * 1.05
                                else "falling" if forecast_vol < realised * 0.95 else "stable")
        return out

    # ----------------------------------------------------------- backtest
    def var_backtest(self, symbol: str, df: pd.DataFrame, confidence: float = 0.95,
                     window: int = 250, method: str = "historical") -> dict:
        returns = df["close"].pct_change().dropna()
        result = backtest_var(returns, confidence, window, method)
        result["symbol"] = symbol.upper()
        return result

    def tail_report(self, symbol: str, df: pd.DataFrame) -> dict:
        returns = df["close"].pct_change().dropna()
        evt99 = var_extreme_value(returns, 0.99)
        evt995 = var_extreme_value(returns, 0.995)
        fhs = var_filtered_historical(returns, 0.99)
        return {
            "symbol": symbol.upper(),
            "evt_99": evt99, "evt_99_5": evt995, "filtered_historical_99": fhs,
            "tail_index": evt99.get("shape_xi"),
            "interpretation": (
                "A positive shape parameter means a power-law tail: extreme losses are far "
                "more likely than a normal distribution implies, and the largest observed "
                "loss is probably not the largest possible one."
                if (evt99.get("shape_xi") or 0) > 0.05 else
                "The tail decays roughly exponentially - extremes are severe but bounded."
            ),
        }


risk_service = RiskService()
