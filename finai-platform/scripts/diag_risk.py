"""Diagnostic harness for the Risk Engine.

Not a test — a measurement. It exercises exactly the code path
``GET /api/v1/risk/scan`` uses, and prints the three properties the engine is
supposed to have:

A. different assets get different, correctly-ordered scores;
B. the score is monotone in real volatility;
C. changing the selected period changes the numbers.

    python3 scripts/diag_risk.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

import numpy as np
import pandas as pd
from app.services.data.market_data import market_data_service
from app.services.risk.anomaly import anomaly_detector
from app.services.risk.profile import benchmark_for, risk_profiler
from app.utils.periods import analysis_window, model_bars

SYMBOLS = ["EURUSD=X", "SPY", "JNJ", "AAPL", "GLD", "BTC-USD", "NVDA", "TSLA",
           "ETH-USD", "^VIX"]
PERIODS = ["1mo", "3mo", "6mo", "ytd", "1y", "3y", "5y", "10y"]


def _scan(symbol: str, period: str) -> tuple[dict, dict]:
    """Reproduce the endpoint: fetch, slice per model, score, profile."""
    fit = analysis_window(period, "bubble")
    crash_bars = model_bars(period, "crash_risk")
    bubble_bars = model_bars(period, "bubble")
    df = market_data_service.get_history(symbol, period=fit).df
    scan = anomaly_detector.scan(symbol, df, lookback_days=180,
                                 crash_bars=crash_bars, bubble_bars=bubble_bars)

    bench_symbol = benchmark_for(symbol)
    bench_df = None
    if bench_symbol and bench_symbol.upper() != symbol.upper():
        try:
            bench_df = market_data_service.get_history(bench_symbol, period=fit).df
        except Exception:      # noqa: BLE001
            bench_symbol = None

    profile = risk_profiler.profile(
        symbol, df.tail(max(crash_bars, bubble_bars)),
        benchmark_df=bench_df, benchmark_symbol=bench_symbol,
        crash=scan.get("crash_risk"), bubble=scan.get("bubble"),
        recent_anomaly_pressure=scan.get("anomaly_pressure"))
    return scan, profile


def separation() -> None:
    print("=" * 92)
    print("A. CROSS-ASSET SEPARATION  (period = 1y)")
    print("=" * 92)
    rows = []
    for sym in SYMBOLS:
        scan, profile = _scan(sym, "1y")
        m, o = profile["metrics"], profile["overall"]
        rows.append({
            "symbol": sym,
            "ann_vol_%": round(m["annualised_volatility"] * 100, 1),
            "VaR95_%": round((m["var_95_daily"] or 0) * 100, 2),
            "CVaR95_%": round((m["cvar_95_daily"] or 0) * 100, 2),
            "maxDD_%": round(m["max_drawdown"] * 100, 1),
            "beta": m["beta"],
            "sharpe": m["sharpe_ratio"],
            "sortino": m["sortino_ratio"],
            "crash": scan["crash_risk"].get("crash_risk_score"),
            "OVERALL": o["score"],
            "level": o["level"],
        })
    out = pd.DataFrame(rows).sort_values("ann_vol_%")
    pd.set_option("display.width", 220)
    print(out.to_string(index=False))
    rho_overall = out["ann_vol_%"].corr(out["OVERALL"], method="spearman")
    rho_crash = out["ann_vol_%"].corr(out["crash"], method="spearman")
    print(f"\nSpearman(annualised vol, Overall Risk) = {rho_overall:.3f}")
    print(f"Spearman(annualised vol, crash only)   = {rho_crash:.3f}"
          "   <- the old headline; relative to each asset's own history")
    print(f"distinct Overall scores: {out['OVERALL'].nunique()} / {len(out)}")


def controlled_volatility() -> None:
    """Identical return path, only sigma scaled. Isolates volatility exactly.

    Using a fresh random draw per rung instead would vary the *path* as well as
    the scale, and the resulting wobble looks like non-monotonicity when it is
    only sampling noise. That false positive is the reason this uses one fixed
    innovation vector.
    """
    print()
    print("=" * 92)
    print("B. CONTROLLED VOLATILITY LADDER  (one fixed path, only sigma scaled)")
    print("=" * 92)
    z = np.random.default_rng(7).standard_normal(400)
    index = pd.date_range("2023-01-02", periods=400, freq="B")

    def frame(vol: float) -> pd.DataFrame:
        r = 0.0003 + z * (vol / np.sqrt(252))
        close = 100 * np.exp(np.cumsum(r))
        return pd.DataFrame({"open": close, "high": close * 1.005,
                             "low": close * 0.995, "close": close,
                             "volume": 1e6}, index=index)

    print(f"{'ann vol':>9s} {'overall':>9s} {'level':>10s}")
    scores = []
    for vol in (0.05, 0.10, 0.20, 0.40, 0.80, 1.20):
        df = frame(vol)
        profile = risk_profiler.profile(
            "T", df, crash=anomaly_detector.crash_risk(df),
            bubble=anomaly_detector.bubble_indicator(df))
        o = profile["overall"]
        scores.append(o["score"])
        print(f"{vol * 100:8.0f}% {o['score']:9.3f} {o['level']:>10s}")
    print(f"\nmonotone in volatility: {scores == sorted(scores)}"
          f"   spread: {scores[0]:.3f} -> {scores[-1]:.3f}")


def period_sensitivity() -> None:
    print()
    print("=" * 92)
    print("C. PERIOD SENSITIVITY  (AAPL, exactly what /risk/scan does)")
    print("=" * 92)
    seen: dict[tuple, list[str]] = {}
    for per in PERIODS:
        scan, profile = _scan("AAPL", per)
        crash = scan["crash_risk"].get("crash_risk_score")
        bubble = scan["bubble"].get("bubble_score")
        overall = profile["overall"]["score"]
        seen.setdefault((crash, bubble, overall), []).append(per)
        print(f"  select {per:4s} -> crash bars {model_bars(per, 'crash_risk'):5d}"
              f" | bubble bars {model_bars(per, 'bubble'):5d}"
              f" -> crash={crash} bubble={bubble} OVERALL={overall}")
    print(f"\n  {len(seen)} distinct answers for {len(PERIODS)} selectable ranges")


if __name__ == "__main__":
    separation()
    controlled_volatility()
    period_sensitivity()
