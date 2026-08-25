"""Small helpers shared across the analytics services."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def utcnow() -> datetime:
    return datetime.now(UTC)


VALID_PERIODS = ("1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "3y", "5y", "10y", "max", "ytd")
VALID_INTERVALS = ("1m", "5m", "15m", "30m", "1h", "1d", "1wk", "1mo")


def period_to_days(period: str) -> int:
    """Convert a yfinance-like period string into a number of calendar days."""
    period = period.strip().lower()
    table = {
        "1d": 1, "5d": 5, "1mo": 31, "3mo": 92, "6mo": 183,
        "1y": 365, "2y": 730, "3y": 1095, "5y": 1825, "10y": 3650, "max": 3650,
        "ytd": (utcnow() - datetime(utcnow().year, 1, 1, tzinfo=UTC)).days or 1,
    }
    if period in table:
        return table[period]
    raise ValueError(f"Unsupported period: {period}")


def validate_period(period: str) -> str:
    """Normalise and validate a period, raising a 422-mapped domain error."""
    from app.core.exceptions import InvalidRequestError

    normalised = (period or "").strip().lower()
    if normalised not in VALID_PERIODS:
        raise InvalidRequestError(
            f"Unsupported period '{period}'",
            details={"valid_periods": list(VALID_PERIODS)},
        )
    return normalised


def validate_interval(interval: str) -> str:
    """Normalise and validate an interval, raising a 422-mapped domain error."""
    from app.core.exceptions import InvalidRequestError

    normalised = (interval or "").strip().lower()
    if normalised not in VALID_INTERVALS:
        raise InvalidRequestError(
            f"Unsupported interval '{interval}'",
            details={"valid_intervals": list(VALID_INTERVALS)},
        )
    return normalised


def interval_to_timedelta(interval: str) -> timedelta:
    interval = interval.strip().lower()
    table = {
        "1m": timedelta(minutes=1), "5m": timedelta(minutes=5),
        "15m": timedelta(minutes=15), "30m": timedelta(minutes=30),
        "1h": timedelta(hours=1), "1d": timedelta(days=1),
        "1wk": timedelta(weeks=1), "1mo": timedelta(days=30),
    }
    return table.get(interval, timedelta(days=1))


def to_returns(prices: pd.Series, log: bool = False) -> pd.Series:
    prices = pd.Series(prices).astype(float)
    if log:
        return np.log(prices / prices.shift(1)).dropna()
    return prices.pct_change().dropna()


def annualise_return(returns: pd.Series, periods: int = TRADING_DAYS) -> float:
    returns = pd.Series(returns).dropna()
    if returns.empty:
        return 0.0
    cumulative = float((1 + returns).prod())
    if cumulative <= 0:
        return -1.0
    return cumulative ** (periods / len(returns)) - 1


def annualise_vol(returns: pd.Series, periods: int = TRADING_DAYS) -> float:
    returns = pd.Series(returns).dropna()
    if len(returns) < 2:
        return 0.0
    return float(returns.std(ddof=1) * np.sqrt(periods))


def max_drawdown(equity: pd.Series) -> tuple[float, int]:
    """Return ``(max_drawdown_fraction, duration_in_periods)``."""
    equity = pd.Series(equity).astype(float).dropna()
    if equity.empty:
        return 0.0, 0
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    mdd = float(drawdown.min())
    trough = int(drawdown.idxmin()) if not isinstance(drawdown.index, pd.DatetimeIndex) else int(
        drawdown.reset_index(drop=True).idxmin()
    )
    series = drawdown.reset_index(drop=True)
    peak = int(series.iloc[: trough + 1][series.iloc[: trough + 1] == 0].index.max()) if (
        (series.iloc[: trough + 1] == 0).any()
    ) else 0
    return mdd, max(trough - peak, 0)


def safe_div(a: float, b: float, default: float = 0.0) -> float:
    try:
        if b == 0 or not np.isfinite(b):
            return default
        value = a / b
        return float(value) if np.isfinite(value) else default
    except Exception:  # pragma: no cover - defensive
        return default


def clean_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise an OHLCV frame: sorted DatetimeIndex, float columns, no gaps."""
    if df is None or df.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    out = df.copy()
    out.columns = [str(c).lower().replace(" ", "_") for c in out.columns]
    if "adj_close" in out.columns and "close" not in out.columns:
        out["close"] = out["adj_close"]
    keep = [c for c in ("open", "high", "low", "close", "volume") if c in out.columns]
    out = out[keep]
    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index, utc=True, errors="coerce")
    out.index = pd.DatetimeIndex(out.index).tz_localize(None) if out.index.tz is not None else out.index
    out = out[~out.index.duplicated(keep="last")].sort_index()
    out = out.astype(float).ffill().dropna(how="all")
    out.index.name = "date"
    return out


def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    agg = {k: v for k, v in agg.items() if k in df.columns}
    return df.resample(rule).agg(agg).dropna(how="all")


def frame_to_records(df: pd.DataFrame) -> list[dict]:
    """JSON-serialisable list of candles."""
    if df is None or df.empty:
        return []
    out = df.reset_index()
    date_col = out.columns[0]
    out[date_col] = pd.to_datetime(out[date_col]).dt.strftime("%Y-%m-%dT%H:%M:%S")
    out = out.rename(columns={date_col: "date"})
    out = out.replace([np.inf, -np.inf], np.nan).where(pd.notnull(out), None)
    return out.to_dict(orient="records")


def business_days_ahead(last: datetime | pd.Timestamp, n: int) -> list[pd.Timestamp]:
    start = pd.Timestamp(last) + pd.Timedelta(days=1)
    return list(pd.bdate_range(start=start, periods=n))
