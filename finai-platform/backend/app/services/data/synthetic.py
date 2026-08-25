"""Deterministic synthetic market generator.

Used as the offline fallback of the hybrid data layer. It produces realistic
OHLCV series featuring:

* geometric brownian motion with instrument-specific drift/vol
* volatility clustering (GARCH-like persistence)
* fat tails (Student-t innovations) and occasional jump/crash regimes
* weekly seasonality and intraday-consistent OHLC relationships
* deterministic output for a given (symbol, seed) pair
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from app.core.config import settings
from app.services.data.universe import Instrument, infer_instrument
from app.utils.timeseries import clean_frame, interval_to_timedelta


def _seed_for(symbol: str, extra: int = 0) -> int:
    digest = hashlib.sha256(f"{symbol}|{settings.SYNTHETIC_SEED}|{extra}".encode()).hexdigest()
    return int(digest[:8], 16)


def _garch_vol(n: int, base_vol: float, rng: np.random.Generator) -> np.ndarray:
    """Simulate a persistent conditional volatility path (GARCH(1,1)-like)."""
    omega, alpha, beta = 0.05, 0.10, 0.85
    long_run = base_vol ** 2
    var = np.empty(n)
    var[0] = long_run
    shock = rng.standard_normal(n)
    for t in range(1, n):
        var[t] = omega * long_run + alpha * (shock[t - 1] ** 2) * var[t - 1] + beta * var[t - 1]
    return np.sqrt(np.clip(var, 1e-10, None))


def generate_ohlcv(
    symbol: str,
    periods: int = 750,
    interval: str = "1d",
    end: pd.Timestamp | None = None,
    instrument: Instrument | None = None,
    seed_offset: int = 0,
) -> pd.DataFrame:
    """Generate a synthetic OHLCV frame ending at ``end`` (defaults to today)."""
    inst = instrument or infer_instrument(symbol)
    rng = np.random.default_rng(_seed_for(inst.symbol, seed_offset))
    periods = max(int(periods), 30)

    step = interval_to_timedelta(interval)
    end_ts = pd.Timestamp(end or pd.Timestamp.utcnow().normalize())
    if step >= pd.Timedelta(days=1):
        index = pd.bdate_range(end=end_ts, periods=periods)
        bars_per_year = 252.0
    else:
        index = pd.date_range(end=end_ts, periods=periods, freq=step)
        bars_per_year = 252.0 * (pd.Timedelta(days=1) / step)

    dt = 1.0 / bars_per_year
    mu = inst.annual_drift
    vol_path = _garch_vol(periods, inst.annual_vol, rng)

    # Student-t innovations -> fat tails
    dof = 4.5
    raw = rng.standard_t(dof, size=periods)
    raw /= np.sqrt(dof / (dof - 2))

    log_ret = (mu - 0.5 * vol_path ** 2) * dt + vol_path * np.sqrt(dt) * raw

    # Regime shifts: rare crashes and melt-ups
    n_jumps = rng.poisson(periods / 260.0)
    for _ in range(int(n_jumps)):
        idx = rng.integers(0, periods)
        magnitude = rng.normal(0.0, 0.06) * (2.5 if inst.asset_class == "crypto" else 1.0)
        length = int(rng.integers(1, 6))
        log_ret[idx: idx + length] += magnitude / max(length, 1)

    # Mild mean reversion keeps the series anchored around a plausible level
    log_ret -= 0.002 * np.cumsum(log_ret) / np.arange(1, periods + 1)

    close = inst.base_price * np.exp(np.cumsum(log_ret))
    close = close * (inst.base_price / close[-1])  # anchor last price to reference

    bar_vol = vol_path * np.sqrt(dt)
    open_ = np.empty(periods)
    open_[0] = close[0] * (1 + rng.normal(0, bar_vol[0] * 0.3))
    open_[1:] = close[:-1] * (1 + rng.normal(0, 1, periods - 1) * bar_vol[1:] * 0.25)

    wick_up = np.abs(rng.normal(0, 1, periods)) * bar_vol * close * 0.8
    wick_dn = np.abs(rng.normal(0, 1, periods)) * bar_vol * close * 0.8
    high = np.maximum(open_, close) + wick_up
    low = np.minimum(open_, close) - wick_dn
    low = np.clip(low, close * 0.5, None)

    # Volume: log-normal, correlated with |return| and with weekly seasonality
    base_volume = {
        "equity": 3.5e7, "etf": 6.0e7, "crypto": 2.2e10,
        "commodity": 2.0e5, "forex": 0.0, "index": 0.0,
    }.get(inst.asset_class, 1e7)
    if base_volume > 0:
        seasonal = 1.0 + 0.12 * np.sin(2 * np.pi * np.arange(periods) / 5.0)
        shock_v = np.exp(rng.normal(0, 0.35, periods))
        activity = 1.0 + 4.0 * np.abs(log_ret) / max(np.abs(log_ret).mean(), 1e-9) * 0.25
        volume = base_volume * seasonal * shock_v * activity
    else:
        volume = np.zeros(periods)

    df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=index,
    )
    df.index.name = "date"
    return clean_frame(df)


def generate_quote(symbol: str, instrument: Instrument | None = None) -> dict:
    """Latest snapshot quote derived from the synthetic history."""
    inst = instrument or infer_instrument(symbol)
    df = generate_ohlcv(inst.symbol, periods=60, instrument=inst)
    last, prev = float(df["close"].iloc[-1]), float(df["close"].iloc[-2])
    change = last - prev
    return {
        "symbol": inst.symbol,
        "name": inst.name,
        "asset_class": inst.asset_class,
        "currency": inst.currency,
        "price": round(last, 6),
        "previous_close": round(prev, 6),
        "change": round(change, 6),
        "change_percent": round(change / prev * 100 if prev else 0.0, 4),
        "day_high": round(float(df["high"].iloc[-1]), 6),
        "day_low": round(float(df["low"].iloc[-1]), 6),
        "volume": float(df["volume"].iloc[-1]),
        "source": "synthetic",
    }
