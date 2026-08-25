"""Hybrid market-data aggregator.

Resolution order for every request:

1. fresh cache (memory -> disk)
2. live provider chain (yahoo -> finnhub -> alpha vantage -> polygon)
3. stale cache (better a slightly old real price than none)
4. deterministic synthetic engine (never fails)

The returned payload always advertises which tier answered through ``source``
so the UI can display an honest "live / cached / simulated" badge.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
import pandas as pd

from app.core.config import settings
from app.core.exceptions import DataUnavailableError
from app.core.logging import get_logger
from app.services.data import cache
from app.services.data.providers import PROVIDER_CHAIN, available_providers
from app.services.data.synthetic import generate_ohlcv, generate_quote
from app.services.data.universe import Instrument, get_instrument, infer_instrument, list_instruments
from app.utils.timeseries import clean_frame, period_to_days, validate_interval, validate_period

logger = get_logger(__name__)

_PERIOD_BARS = {
    "1d": 2, "5d": 6, "1mo": 23, "3mo": 65, "6mo": 128,
    "1y": 253, "2y": 505, "3y": 756, "5y": 1260, "10y": 2520, "max": 2520, "ytd": 150,
}


@dataclass
class MarketSeries:
    symbol: str
    instrument: Instrument
    df: pd.DataFrame
    source: str
    interval: str
    period: str

    @property
    def is_live(self) -> bool:
        return self.source not in ("synthetic", "cache:stale")


class MarketDataService:
    """Single entry point used by every other service of the platform."""

    def __init__(self) -> None:
        self.providers = PROVIDER_CHAIN

    # ------------------------------------------------------------- history
    def get_history(
        self,
        symbol: str,
        period: str = "1y",
        interval: str = "1d",
        force_refresh: bool = False,
    ) -> MarketSeries:
        symbol = symbol.upper().strip()
        # Validate here rather than in each endpoint: every caller (API, workers,
        # scripts, notebooks) gets the same 422-mapped error instead of a 500.
        period = validate_period(period)
        interval = validate_interval(interval)
        inst = infer_instrument(symbol)
        key = f"hist::{symbol}::{period}::{interval}"

        if not force_refresh:
            cached = cache.frame_get(key, max_age=settings.CACHE_TTL_SECONDS)
            if cached is not None and not cached.empty:
                return MarketSeries(symbol, inst, cached, "cache", interval, period)

        if settings.allow_network:
            for provider in self.providers:
                if not provider.available():
                    continue
                df = provider.fetch_history(symbol, period, interval)
                # A short window legitimately returns few rows: one trading day
                # at 5-minute bars is ~78, but `1d` at daily interval is a
                # single bar. The old `>= 5` floor rejected real data and fell
                # through to the synthetic engine, so asking for "1D" silently
                # produced 120 invented bars. Trust any non-empty provider
                # response; the caller decides whether the length is usable.
                if df is not None and len(df) >= 1:
                    df = clean_frame(df)
                    cache.frame_set(key, df)
                    logger.info("history %s from %s (%d bars)", symbol, provider.name, len(df))
                    return MarketSeries(symbol, inst, df, provider.name, interval, period)

        stale = cache.stale_frame(key)
        if stale is not None and not stale.empty:
            logger.warning("history %s served from stale cache", symbol)
            return MarketSeries(symbol, inst, stale, "cache:stale", interval, period)

        if settings.DATA_MODE == "live":
            raise DataUnavailableError(
                f"No live provider could serve {symbol}",
                details={"providers": available_providers()},
            )

        bars = _PERIOD_BARS.get(period, max(period_to_days(period) * 5 // 7, 60))
        df = generate_ohlcv(symbol, periods=max(bars, 120), interval=interval, instrument=inst)
        cache.frame_set(key, df)
        logger.info("history %s served by synthetic engine (%d bars)", symbol, len(df))
        return MarketSeries(symbol, inst, df, "synthetic", interval, period)

    # --------------------------------------------------------------- quote
    def get_quote(self, symbol: str) -> dict:
        symbol = symbol.upper().strip()
        inst = infer_instrument(symbol)
        key = f"quote::{symbol}"

        cached = cache.cache_get(key)
        if cached:
            return cached

        if settings.allow_network:
            for provider in self.providers:
                if not provider.available():
                    continue
                quote = provider.fetch_quote(symbol)
                if quote and quote.get("price"):
                    quote.setdefault("name", inst.name)
                    quote.setdefault("asset_class", inst.asset_class)
                    quote["symbol"] = symbol
                    self._enrich_quote(quote, symbol)
                    cache.cache_set(key, quote, ttl=60)
                    return quote

        if settings.DATA_MODE == "live":
            raise DataUnavailableError(f"No live quote available for {symbol}")

        quote = generate_quote(symbol, instrument=inst)
        cache.cache_set(key, quote, ttl=60)
        return quote

    def _enrich_quote(self, quote: dict, symbol: str) -> None:
        """Backfill volume / day range from cached history when a provider omits them."""
        needs_volume = not quote.get("volume")
        needs_range = quote.get("day_high") in (None, quote.get("price")) or \
            quote.get("day_low") in (None, quote.get("price"))
        if not (needs_volume or needs_range):
            return
        try:
            cached = cache.frame_get(f"hist::{symbol}::1y::1d", max_age=None)
            if cached is None or cached.empty:
                cached = self.get_history(symbol, period="1mo").df
            if cached is None or cached.empty:
                return
            last = cached.iloc[-1]
            if needs_volume and "volume" in cached.columns:
                quote["volume"] = float(last["volume"])
            if needs_range:
                quote["day_high"] = float(max(last.get("high", quote["price"]), quote["price"]))
                quote["day_low"] = float(min(last.get("low", quote["price"]), quote["price"]))
        except Exception as exc:  # pragma: no cover - purely cosmetic enrichment
            logger.debug("quote enrichment skipped for %s: %s", symbol, exc)

    def get_quotes(self, symbols: list[str], max_workers: int = 8) -> list[dict]:
        symbols = [s.upper().strip() for s in symbols if s.strip()]
        if not symbols:
            return []
        with ThreadPoolExecutor(max_workers=min(max_workers, len(symbols))) as pool:
            return list(pool.map(self._safe_quote, symbols))

    def _safe_quote(self, symbol: str) -> dict:
        try:
            return self.get_quote(symbol)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("quote failed for %s: %s", symbol, exc)
            return {"symbol": symbol, "price": None, "error": str(exc), "source": "error"}

    # ----------------------------------------------------------- multi-asset
    def get_price_matrix(
        self,
        symbols: list[str],
        period: str = "1y",
        interval: str = "1d",
        column: str = "close",
    ) -> pd.DataFrame:
        """Aligned close-price matrix used by portfolio / RL / correlation modules."""
        frames: dict[str, pd.Series] = {}
        for sym in symbols:
            try:
                series = self.get_history(sym, period=period, interval=interval)
                if not series.df.empty:
                    frames[sym.upper()] = series.df[column]
            except Exception as exc:
                logger.warning("price matrix skipped %s: %s", sym, exc)
        if not frames:
            raise DataUnavailableError("No price data for the requested symbols")
        matrix = pd.DataFrame(frames).sort_index()
        return matrix.ffill().dropna(how="all").dropna()

    def get_returns_matrix(self, symbols: list[str], period: str = "1y") -> pd.DataFrame:
        return self.get_price_matrix(symbols, period=period).pct_change().dropna()

    # ------------------------------------------------------------ discovery
    def search(self, query: str | None = None, asset_class: str | None = None) -> list[dict]:
        return [i.to_dict() for i in list_instruments(asset_class=asset_class, query=query)]

    def get_profile(self, symbol: str) -> dict:
        inst = get_instrument(symbol) or infer_instrument(symbol)
        series = self.get_history(symbol, period="1y")
        df = series.df
        returns = df["close"].pct_change().dropna()
        return {
            **inst.to_dict(),
            "data_source": series.source,
            "bars": int(len(df)),
            "first_date": df.index[0].strftime("%Y-%m-%d") if len(df) else None,
            "last_date": df.index[-1].strftime("%Y-%m-%d") if len(df) else None,
            "last_price": float(df["close"].iloc[-1]) if len(df) else None,
            "annualised_vol": float(returns.std() * np.sqrt(252)) if len(returns) > 2 else None,
            "ytd_return": float(df["close"].iloc[-1] / df["close"].iloc[0] - 1) if len(df) else None,
        }

    def health(self) -> dict:
        return {
            "data_mode": settings.DATA_MODE,
            "network_allowed": settings.allow_network,
            "live_providers": available_providers(),
            "universe_size": len(list_instruments()),
        }


market_data_service = MarketDataService()
