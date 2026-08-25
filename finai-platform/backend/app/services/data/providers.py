"""Live market-data providers.

Each provider implements the same tiny interface so the aggregator in
``market_data.py`` can try them in order and fall back cleanly:

    fetch_history(symbol, period, interval) -> pd.DataFrame | None
    fetch_quote(symbol)                     -> dict | None

Every call is defensive: a provider that raises, times out or returns garbage
simply yields ``None`` and the next one is tried.
"""

from __future__ import annotations

from typing import Protocol

import pandas as pd
import requests

from app.core.config import settings
from app.core.logging import get_logger
from app.utils.timeseries import clean_frame, period_to_days

logger = get_logger(__name__)


class MarketDataProvider(Protocol):
    name: str

    def available(self) -> bool: ...
    def fetch_history(self, symbol: str, period: str, interval: str) -> pd.DataFrame | None: ...
    def fetch_quote(self, symbol: str) -> dict | None: ...


# ------------------------------------------------------------------ yfinance
class YahooFinanceProvider:
    name = "yahoo"

    def available(self) -> bool:
        if not settings.allow_network:
            return False
        try:
            import yfinance  # noqa: F401
            return True
        except Exception:
            return False

    def fetch_history(self, symbol: str, period: str, interval: str) -> pd.DataFrame | None:
        if not self.available():
            return None
        try:
            import yfinance as yf

            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval, auto_adjust=True, timeout=settings.NETWORK_TIMEOUT)
            if df is None or df.empty:
                return None
            return clean_frame(df)
        except Exception as exc:
            logger.debug("[yahoo] history failed for %s: %s", symbol, exc)
            return None

    def fetch_quote(self, symbol: str) -> dict | None:
        if not self.available():
            return None
        try:
            import yfinance as yf

            info = yf.Ticker(symbol).fast_info
            price = float(info.get("last_price") or info.get("lastPrice"))
            prev = float(info.get("previous_close") or info.get("previousClose") or price)
            return {
                "symbol": symbol,
                "price": price,
                "previous_close": prev,
                "change": price - prev,
                "change_percent": (price - prev) / prev * 100 if prev else 0.0,
                "day_high": float(info.get("day_high") or price),
                "day_low": float(info.get("day_low") or price),
                "volume": float(info.get("last_volume") or 0.0),
                "currency": info.get("currency") or "USD",
                "source": self.name,
            }
        except Exception as exc:
            logger.debug("[yahoo] quote failed for %s: %s", symbol, exc)
            return None


# ------------------------------------------------------------- Alpha Vantage
class AlphaVantageProvider:
    name = "alpha_vantage"
    BASE = "https://www.alphavantage.co/query"

    def available(self) -> bool:
        return bool(settings.allow_network and settings.ALPHA_VANTAGE_API_KEY)

    def fetch_history(self, symbol: str, period: str, interval: str) -> pd.DataFrame | None:
        if not self.available() or interval != "1d":
            return None
        try:
            params = {
                "function": "TIME_SERIES_DAILY_ADJUSTED",
                "symbol": symbol,
                "outputsize": "full" if period_to_days(period) > 100 else "compact",
                "apikey": settings.ALPHA_VANTAGE_API_KEY,
            }
            resp = requests.get(self.BASE, params=params, timeout=settings.NETWORK_TIMEOUT)
            payload = resp.json()
            series = payload.get("Time Series (Daily)")
            if not series:
                return None
            df = pd.DataFrame(series).T
            df.index = pd.to_datetime(df.index)
            df = df.rename(columns={
                "1. open": "open", "2. high": "high", "3. low": "low",
                "5. adjusted close": "close", "6. volume": "volume",
            })
            df = clean_frame(df)
            cutoff = pd.Timestamp.utcnow().tz_localize(None) - pd.Timedelta(days=period_to_days(period))
            return df[df.index >= cutoff]
        except Exception as exc:
            logger.debug("[alpha_vantage] history failed for %s: %s", symbol, exc)
            return None

    def fetch_quote(self, symbol: str) -> dict | None:
        if not self.available():
            return None
        try:
            params = {"function": "GLOBAL_QUOTE", "symbol": symbol, "apikey": settings.ALPHA_VANTAGE_API_KEY}
            data = requests.get(self.BASE, params=params, timeout=settings.NETWORK_TIMEOUT).json()
            q = data.get("Global Quote") or {}
            if not q:
                return None
            price = float(q["05. price"])
            prev = float(q["08. previous close"])
            return {
                "symbol": symbol, "price": price, "previous_close": prev,
                "change": price - prev,
                "change_percent": (price - prev) / prev * 100 if prev else 0.0,
                "day_high": float(q.get("03. high", price)),
                "day_low": float(q.get("04. low", price)),
                "volume": float(q.get("06. volume", 0)),
                "currency": "USD", "source": self.name,
            }
        except Exception as exc:
            logger.debug("[alpha_vantage] quote failed for %s: %s", symbol, exc)
            return None


# -------------------------------------------------------------------- Finnhub
class FinnhubProvider:
    name = "finnhub"
    BASE = "https://finnhub.io/api/v1"

    def available(self) -> bool:
        return bool(settings.allow_network and settings.FINNHUB_API_KEY)

    def fetch_history(self, symbol: str, period: str, interval: str) -> pd.DataFrame | None:
        if not self.available():
            return None
        try:
            resolution = {"1m": "1", "5m": "5", "15m": "15", "30m": "30",
                          "1h": "60", "1d": "D", "1wk": "W", "1mo": "M"}.get(interval, "D")
            now = int(pd.Timestamp.utcnow().timestamp())
            start = now - period_to_days(period) * 86400
            params = {"symbol": symbol, "resolution": resolution, "from": start,
                      "to": now, "token": settings.FINNHUB_API_KEY}
            data = requests.get(f"{self.BASE}/stock/candle", params=params,
                                timeout=settings.NETWORK_TIMEOUT).json()
            if data.get("s") != "ok":
                return None
            df = pd.DataFrame({
                "open": data["o"], "high": data["h"], "low": data["l"],
                "close": data["c"], "volume": data["v"],
            }, index=pd.to_datetime(data["t"], unit="s"))
            return clean_frame(df)
        except Exception as exc:
            logger.debug("[finnhub] history failed for %s: %s", symbol, exc)
            return None

    def fetch_quote(self, symbol: str) -> dict | None:
        if not self.available():
            return None
        try:
            data = requests.get(f"{self.BASE}/quote",
                                params={"symbol": symbol, "token": settings.FINNHUB_API_KEY},
                                timeout=settings.NETWORK_TIMEOUT).json()
            price, prev = float(data.get("c", 0)), float(data.get("pc", 0))
            if not price:
                return None
            return {
                "symbol": symbol, "price": price, "previous_close": prev,
                "change": price - prev,
                "change_percent": (price - prev) / prev * 100 if prev else 0.0,
                "day_high": float(data.get("h", price)), "day_low": float(data.get("l", price)),
                "volume": 0.0, "currency": "USD", "source": self.name,
            }
        except Exception as exc:
            logger.debug("[finnhub] quote failed for %s: %s", symbol, exc)
            return None


# ------------------------------------------------------------------ Polygon
class PolygonProvider:
    name = "polygon"
    BASE = "https://api.polygon.io"

    def available(self) -> bool:
        return bool(settings.allow_network and settings.POLYGON_API_KEY)

    def fetch_history(self, symbol: str, period: str, interval: str) -> pd.DataFrame | None:
        if not self.available():
            return None
        try:
            mult, span = {"1m": (1, "minute"), "5m": (5, "minute"), "15m": (15, "minute"),
                          "1h": (1, "hour"), "1d": (1, "day"), "1wk": (1, "week")}.get(interval, (1, "day"))
            end = pd.Timestamp.utcnow().normalize()
            start = end - pd.Timedelta(days=period_to_days(period))
            url = (f"{self.BASE}/v2/aggs/ticker/{symbol}/range/{mult}/{span}/"
                   f"{start:%Y-%m-%d}/{end:%Y-%m-%d}")
            data = requests.get(url, params={"adjusted": "true", "limit": 50000,
                                             "apiKey": settings.POLYGON_API_KEY},
                                timeout=settings.NETWORK_TIMEOUT).json()
            results = data.get("results")
            if not results:
                return None
            df = pd.DataFrame(results)
            df.index = pd.to_datetime(df["t"], unit="ms")
            df = df.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
            return clean_frame(df)
        except Exception as exc:
            logger.debug("[polygon] history failed for %s: %s", symbol, exc)
            return None

    def fetch_quote(self, symbol: str) -> dict | None:
        if not self.available():
            return None
        try:
            url = f"{self.BASE}/v2/aggs/ticker/{symbol}/prev"
            data = requests.get(url, params={"adjusted": "true", "apiKey": settings.POLYGON_API_KEY},
                                timeout=settings.NETWORK_TIMEOUT).json()
            res = (data.get("results") or [{}])[0]
            price, prev = float(res.get("c", 0)), float(res.get("o", 0))
            if not price:
                return None
            return {"symbol": symbol, "price": price, "previous_close": prev,
                    "change": price - prev,
                    "change_percent": (price - prev) / prev * 100 if prev else 0.0,
                    "day_high": float(res.get("h", price)), "day_low": float(res.get("l", price)),
                    "volume": float(res.get("v", 0)), "currency": "USD", "source": self.name}
        except Exception as exc:
            logger.debug("[polygon] quote failed for %s: %s", symbol, exc)
            return None


PROVIDER_CHAIN: list[MarketDataProvider] = [
    YahooFinanceProvider(),
    FinnhubProvider(),
    AlphaVantageProvider(),
    PolygonProvider(),
]


def available_providers() -> list[str]:
    return [p.name for p in PROVIDER_CHAIN if p.available()]
