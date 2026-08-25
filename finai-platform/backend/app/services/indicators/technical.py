"""Pure-pandas technical analysis library (no TA-Lib / C dependency).

Every function takes an OHLCV DataFrame (lower-case columns) and returns either
a Series or a DataFrame aligned on the same index, so indicators compose freely.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# --------------------------------------------------------------- moving avg
def sma(series: pd.Series, window: int = 20) -> pd.Series:
    return series.rolling(window=window, min_periods=max(2, window // 2)).mean()


def ema(series: pd.Series, window: int = 20) -> pd.Series:
    return series.ewm(span=window, adjust=False, min_periods=max(2, window // 2)).mean()


def wma(series: pd.Series, window: int = 20) -> pd.Series:
    weights = np.arange(1, window + 1)
    return series.rolling(window).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)


# ---------------------------------------------------------------- momentum
def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """Wilder's RSI.

    Edge cases are handled explicitly rather than collapsed to the neutral 50:
    a window with no losses is a genuine 100 (pure uptrend), no gains is a 0,
    and a perfectly flat window is 50. Only the un-warmed leading window stays
    neutral.
    """
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()

    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100 - (100 / (1 + rs))

    warmed = avg_gain.notna() & avg_loss.notna()
    no_loss = warmed & (avg_loss <= 1e-12) & (avg_gain > 1e-12)
    no_gain = warmed & (avg_gain <= 1e-12) & (avg_loss > 1e-12)
    flat = warmed & (avg_gain <= 1e-12) & (avg_loss <= 1e-12)

    out = out.mask(no_loss, 100.0).mask(no_gain, 0.0).mask(flat, 50.0)
    return out.fillna(50.0).clip(0, 100)


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    macd_line = ema(series, fast) - ema(series, slow)
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return pd.DataFrame({
        "macd": macd_line,
        "macd_signal": signal_line,
        "macd_hist": macd_line - signal_line,
    })


def stochastic(df: pd.DataFrame, k: int = 14, d: int = 3) -> pd.DataFrame:
    low_k = df["low"].rolling(k).min()
    high_k = df["high"].rolling(k).max()
    percent_k = 100 * (df["close"] - low_k) / (high_k - low_k).replace(0, np.nan)
    return pd.DataFrame({
        "stoch_k": percent_k.fillna(50),
        "stoch_d": percent_k.rolling(d).mean().fillna(50),
    })


def roc(series: pd.Series, window: int = 12) -> pd.Series:
    return series.pct_change(window) * 100


def williams_r(df: pd.DataFrame, window: int = 14) -> pd.Series:
    high = df["high"].rolling(window).max()
    low = df["low"].rolling(window).min()
    return (-100 * (high - df["close"]) / (high - low).replace(0, np.nan)).fillna(-50)


def cci(df: pd.DataFrame, window: int = 20) -> pd.Series:
    tp = (df["high"] + df["low"] + df["close"]) / 3
    ma = tp.rolling(window).mean()
    md = (tp - ma).abs().rolling(window).mean()
    return ((tp - ma) / (0.015 * md.replace(0, np.nan))).fillna(0)


# -------------------------------------------------------------- volatility
def bollinger_bands(series: pd.Series, window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    mid = sma(series, window)
    std = series.rolling(window, min_periods=max(2, window // 2)).std(ddof=0)
    upper, lower = mid + num_std * std, mid - num_std * std
    width = (upper - lower) / mid.replace(0, np.nan)
    pct_b = (series - lower) / (upper - lower).replace(0, np.nan)
    return pd.DataFrame({
        "bb_upper": upper, "bb_middle": mid, "bb_lower": lower,
        "bb_width": width, "bb_pct_b": pct_b.clip(-0.5, 1.5),
    })


def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    return pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)


def atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    return true_range(df).ewm(alpha=1 / window, adjust=False, min_periods=window).mean()


def keltner_channels(df: pd.DataFrame, window: int = 20, mult: float = 2.0) -> pd.DataFrame:
    mid = ema(df["close"], window)
    rng = atr(df, window) * mult
    return pd.DataFrame({"kc_upper": mid + rng, "kc_middle": mid, "kc_lower": mid - rng})


def historical_volatility(series: pd.Series, window: int = 21, periods: int = 252) -> pd.Series:
    log_ret = np.log(series / series.shift(1))
    return log_ret.rolling(window).std(ddof=1) * np.sqrt(periods) * 100


# ------------------------------------------------------------------ trend
def adx(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    up = df["high"].diff()
    down = -df["low"].diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    tr = true_range(df).ewm(alpha=1 / window, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1 / window, adjust=False).mean() / tr.replace(0, np.nan)
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1 / window, adjust=False).mean() / tr.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return pd.DataFrame({
        "adx": dx.ewm(alpha=1 / window, adjust=False).mean().fillna(0),
        "plus_di": plus_di.fillna(0), "minus_di": minus_di.fillna(0),
    })


def ichimoku(df: pd.DataFrame) -> pd.DataFrame:
    high9, low9 = df["high"].rolling(9).max(), df["low"].rolling(9).min()
    high26, low26 = df["high"].rolling(26).max(), df["low"].rolling(26).min()
    high52, low52 = df["high"].rolling(52).max(), df["low"].rolling(52).min()
    tenkan = (high9 + low9) / 2
    kijun = (high26 + low26) / 2
    return pd.DataFrame({
        "tenkan_sen": tenkan, "kijun_sen": kijun,
        "senkou_a": ((tenkan + kijun) / 2).shift(26),
        "senkou_b": ((high52 + low52) / 2).shift(26),
        "chikou": df["close"].shift(-26),
    })


# ----------------------------------------------------------------- volume
def obv(df: pd.DataFrame) -> pd.Series:
    direction = np.sign(df["close"].diff().fillna(0.0))
    return (direction * df.get("volume", pd.Series(0, index=df.index))).cumsum()


def vwap(df: pd.DataFrame, window: int = 20) -> pd.Series:
    tp = (df["high"] + df["low"] + df["close"]) / 3
    vol = df.get("volume", pd.Series(0.0, index=df.index)).replace(0, np.nan)
    return (tp * vol).rolling(window).sum() / vol.rolling(window).sum()


def money_flow_index(df: pd.DataFrame, window: int = 14) -> pd.Series:
    tp = (df["high"] + df["low"] + df["close"]) / 3
    mf = tp * df.get("volume", pd.Series(0.0, index=df.index))
    pos = mf.where(tp > tp.shift(1), 0.0).rolling(window).sum()
    neg = mf.where(tp < tp.shift(1), 0.0).rolling(window).sum()
    ratio = pos / neg.replace(0, np.nan)
    return (100 - 100 / (1 + ratio)).fillna(50)


# -------------------------------------------------------------- aggregate
INDICATOR_REGISTRY = {
    "sma", "ema", "rsi", "macd", "bbands", "atr", "adx", "stoch",
    "obv", "vwap", "mfi", "cci", "williams_r", "roc", "keltner", "ichimoku", "hv",
}


def compute_indicators(
    df: pd.DataFrame,
    indicators: list[str] | None = None,
    params: dict | None = None,
) -> pd.DataFrame:
    """Return the OHLCV frame enriched with the requested indicator columns."""
    if df is None or df.empty:
        return pd.DataFrame()
    p = params or {}
    wanted = set(indicators or ["sma", "ema", "rsi", "macd", "bbands", "atr", "adx"])
    out = df.copy()
    close = out["close"]

    if "sma" in wanted:
        for w in p.get("sma_windows", [20, 50, 200]):
            out[f"sma_{w}"] = sma(close, w)
    if "ema" in wanted:
        for w in p.get("ema_windows", [12, 26, 50]):
            out[f"ema_{w}"] = ema(close, w)
    if "rsi" in wanted:
        out["rsi"] = rsi(close, p.get("rsi_window", 14))
    if "macd" in wanted:
        out = out.join(macd(close, p.get("macd_fast", 12), p.get("macd_slow", 26), p.get("macd_signal", 9)))
    if "bbands" in wanted:
        out = out.join(bollinger_bands(close, p.get("bb_window", 20), p.get("bb_std", 2.0)))
    if "atr" in wanted:
        out["atr"] = atr(out, p.get("atr_window", 14))
        out["atr_pct"] = out["atr"] / close * 100
    if "adx" in wanted:
        out = out.join(adx(out, p.get("adx_window", 14)))
    if "stoch" in wanted:
        out = out.join(stochastic(out))
    if "obv" in wanted and "volume" in out:
        out["obv"] = obv(out)
    if "vwap" in wanted and "volume" in out:
        out["vwap"] = vwap(out)
    if "mfi" in wanted and "volume" in out:
        out["mfi"] = money_flow_index(out)
    if "cci" in wanted:
        out["cci"] = cci(out)
    if "williams_r" in wanted:
        out["williams_r"] = williams_r(out)
    if "roc" in wanted:
        out["roc"] = roc(close)
    if "keltner" in wanted:
        out = out.join(keltner_channels(out))
    if "ichimoku" in wanted:
        out = out.join(ichimoku(out))
    if "hv" in wanted:
        out["hist_vol"] = historical_volatility(close)

    return out.replace([np.inf, -np.inf], np.nan)


def signal_summary(enriched: pd.DataFrame) -> dict:
    """Rule-based read of the latest indicator values -> per-indicator signals."""
    if enriched is None or enriched.empty:
        return {}
    last = enriched.iloc[-1]
    signals: dict[str, dict] = {}

    def add(name: str, signal: str, value, note: str) -> None:
        signals[name] = {
            "signal": signal,
            "value": None if value is None or (isinstance(value, float) and not np.isfinite(value)) else round(float(value), 4),
            "note": note,
        }

    if "rsi" in last:
        v = last["rsi"]
        add("RSI", "buy" if v < 30 else "sell" if v > 70 else "neutral", v,
            "Oversold" if v < 30 else "Overbought" if v > 70 else "Neutral momentum")
    if "macd_hist" in last:
        v = last["macd_hist"]
        add("MACD", "buy" if v > 0 else "sell", v,
            "Bullish crossover" if v > 0 else "Bearish crossover")
    if {"bb_pct_b"} <= set(last.index):
        v = last["bb_pct_b"]
        add("Bollinger", "buy" if v < 0.05 else "sell" if v > 0.95 else "neutral", v,
            "Price at lower band" if v < 0.05 else "Price at upper band" if v > 0.95 else "Inside bands")
    if {"sma_50", "sma_200"} <= set(last.index) and np.isfinite(last.get("sma_200", np.nan)):
        diff = last["sma_50"] - last["sma_200"]
        add("Golden/Death Cross", "buy" if diff > 0 else "sell", diff,
            "SMA50 above SMA200 (golden cross)" if diff > 0 else "SMA50 below SMA200 (death cross)")
    if "adx" in last:
        v = last["adx"]
        trend = "strong" if v > 25 else "weak"
        direction = "buy" if last.get("plus_di", 0) > last.get("minus_di", 0) else "sell"
        add("ADX", direction if v > 25 else "neutral", v, f"{trend.capitalize()} trend (ADX={v:.1f})")
    if "stoch_k" in last:
        v = last["stoch_k"]
        add("Stochastic", "buy" if v < 20 else "sell" if v > 80 else "neutral", v,
            "Oversold" if v < 20 else "Overbought" if v > 80 else "Mid-range")
    if "mfi" in last:
        v = last["mfi"]
        add("MFI", "buy" if v < 20 else "sell" if v > 80 else "neutral", v, "Money-flow extreme" if (v < 20 or v > 80) else "Balanced flows")
    if "atr_pct" in last:
        add("ATR", "neutral", last["atr_pct"], f"Volatility {last['atr_pct']:.2f}% of price")

    votes = [s["signal"] for s in signals.values() if s["signal"] in ("buy", "sell")]
    buy, sell = votes.count("buy"), votes.count("sell")
    total = max(buy + sell, 1)
    if buy > sell:
        consensus, strength = "bullish", buy / total
    elif sell > buy:
        consensus, strength = "bearish", sell / total
    else:
        consensus, strength = "neutral", 0.5

    return {
        "indicators": signals,
        "consensus": consensus,
        "strength": round(strength, 3),
        "buy_votes": buy,
        "sell_votes": sell,
        "neutral_votes": len(signals) - buy - sell,
    }
