"""Feature engineering pipeline feeding the DL forecasters, the RL agent and XAI."""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.services.indicators.technical import (
    adx,
    atr,
    bollinger_bands,
    cci,
    ema,
    macd,
    money_flow_index,
    obv,
    rsi,
    sma,
    stochastic,
)

FEATURE_COLUMNS: list[str] = [
    "return_1d", "return_5d", "return_10d", "return_21d",
    "log_return", "volatility_10d", "volatility_21d", "volatility_ratio",
    "rsi_14", "macd", "macd_hist", "bb_pct_b", "bb_width",
    "atr_pct", "adx", "plus_di", "minus_di", "stoch_k", "cci_20",
    "price_to_sma20", "price_to_sma50", "price_to_ema12", "sma20_to_sma50",
    "volume_ratio", "obv_slope", "mfi_14",
    "high_low_range", "close_to_high", "gap", "day_of_week", "month",
]


def build_features(df: pd.DataFrame, dropna: bool = True) -> pd.DataFrame:
    """Compute the full engineered feature matrix from an OHLCV frame."""
    if df is None or len(df) < 30:
        return pd.DataFrame()

    out = pd.DataFrame(index=df.index)
    close, high, low = df["close"], df["high"], df["low"]
    volume = df["volume"] if "volume" in df else pd.Series(0.0, index=df.index)

    # --- returns ---------------------------------------------------------
    out["return_1d"] = close.pct_change()
    out["return_5d"] = close.pct_change(5)
    out["return_10d"] = close.pct_change(10)
    out["return_21d"] = close.pct_change(21)
    out["log_return"] = np.log(close / close.shift(1))

    # --- volatility ------------------------------------------------------
    out["volatility_10d"] = out["return_1d"].rolling(10).std() * np.sqrt(252)
    out["volatility_21d"] = out["return_1d"].rolling(21).std() * np.sqrt(252)
    out["volatility_ratio"] = out["volatility_10d"] / out["volatility_21d"].replace(0, np.nan)

    # --- momentum / oscillators -----------------------------------------
    out["rsi_14"] = rsi(close, 14)
    macd_df = macd(close)
    out["macd"] = macd_df["macd"]
    out["macd_hist"] = macd_df["macd_hist"]
    bb = bollinger_bands(close, 20, 2.0)
    out["bb_pct_b"] = bb["bb_pct_b"]
    out["bb_width"] = bb["bb_width"]
    out["atr_pct"] = atr(df, 14) / close * 100
    adx_df = adx(df, 14)
    out["adx"] = adx_df["adx"]
    out["plus_di"] = adx_df["plus_di"]
    out["minus_di"] = adx_df["minus_di"]
    out["stoch_k"] = stochastic(df)["stoch_k"]
    out["cci_20"] = cci(df, 20)

    # --- trend position --------------------------------------------------
    sma20, sma50, ema12 = sma(close, 20), sma(close, 50), ema(close, 12)
    out["price_to_sma20"] = close / sma20 - 1
    out["price_to_sma50"] = close / sma50 - 1
    out["price_to_ema12"] = close / ema12 - 1
    out["sma20_to_sma50"] = sma20 / sma50 - 1

    # --- volume ----------------------------------------------------------
    # Forex, indices and some futures report no volume at all. Volume-derived
    # features are then undefined, not "missing": they must collapse to a
    # neutral constant rather than NaN, otherwise the downstream dropna() wipes
    # out the entire feature matrix and silently disables ML for those assets.
    has_volume = bool(volume.abs().sum() > 0)
    if has_volume:
        vol_ma = volume.rolling(20).mean().replace(0, np.nan)
        out["volume_ratio"] = (volume / vol_ma).fillna(1.0)
        obv_series = obv(df)
        obv_scale = obv_series.abs().rolling(20).mean().replace(0, np.nan)
        out["obv_slope"] = (obv_series.diff(5) / obv_scale).fillna(0.0)
        out["mfi_14"] = money_flow_index(df, 14)
    else:
        out["volume_ratio"] = 1.0     # neutral: "average" activity
        out["obv_slope"] = 0.0        # no flow information available
        out["mfi_14"] = 50.0          # neutral oscillator reading

    # --- candle geometry / calendar --------------------------------------
    out["high_low_range"] = (high - low) / close
    out["close_to_high"] = (high - close) / (high - low).replace(0, np.nan)
    out["gap"] = (df["open"] - close.shift(1)) / close.shift(1)
    out["day_of_week"] = df.index.dayofweek.astype(float)
    out["month"] = df.index.month.astype(float)

    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.dropna() if dropna else out.ffill().bfill()
    return out


def build_targets(df: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    """Regression + classification targets aligned with ``build_features``."""
    close = df["close"]
    fwd_return = close.shift(-horizon) / close - 1
    return pd.DataFrame({
        "target_price": close.shift(-horizon),
        "target_return": fwd_return,
        "target_direction": (fwd_return > 0).astype(int),
        "target_volatility": close.pct_change().rolling(horizon).std().shift(-horizon) * np.sqrt(252),
    }, index=df.index)


def build_supervised(df: pd.DataFrame, horizon: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aligned (X, y) pair with no leakage and no NaN."""
    features = build_features(df, dropna=False)
    targets = build_targets(df, horizon=horizon)
    joined = features.join(targets).dropna()
    x = joined[[c for c in features.columns if c in joined.columns]]
    y = joined[list(targets.columns)]
    return x, y


def make_sequences(
    x: np.ndarray,
    y: np.ndarray,
    lookback: int = 60,
) -> tuple[np.ndarray, np.ndarray]:
    """Sliding-window tensors for LSTM/GRU/TCN/Transformer models."""
    if len(x) <= lookback:
        return np.empty((0, lookback, x.shape[1])), np.empty((0,))
    xs, ys = [], []
    for i in range(lookback, len(x)):
        xs.append(x[i - lookback: i])
        ys.append(y[i])
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)
