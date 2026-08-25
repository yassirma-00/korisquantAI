"""Market data, technical analysis and screening endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Query

from app.core.exceptions import InvalidRequestError
from app.schemas.common import HistoryResponse
from app.services.data.market_data import market_data_service
from app.services.indicators.technical import INDICATOR_REGISTRY, compute_indicators, signal_summary
from app.services.risk.metrics import correlation_matrix, full_metrics
from app.utils.timeseries import frame_to_records

router = APIRouter(prefix="/market", tags=["Market Data"])


@router.get("/instruments", summary="List / search the tradable universe")
async def list_instruments(
    q: str | None = Query(None, description="Free-text search on symbol or name"),
    asset_class: str | None = Query(None, description="equity | crypto | etf | commodity | forex | index"),
):
    instruments = market_data_service.search(q, asset_class)
    return {"count": len(instruments), "instruments": instruments}


@router.get("/quote/{symbol}", summary="Latest quote for one instrument")
async def get_quote(symbol: str):
    return market_data_service.get_quote(symbol)


@router.get("/quotes", summary="Batch quotes")
async def get_quotes(symbols: str = Query(..., description="Comma-separated symbols")):
    parsed = [s.strip() for s in symbols.split(",") if s.strip()]
    if not parsed:
        raise InvalidRequestError("No symbols provided")
    return {"quotes": market_data_service.get_quotes(parsed)}


@router.get("/history/{symbol}", response_model=HistoryResponse, summary="OHLCV history + indicators")
async def get_history(
    symbol: str,
    period: str = Query("1y", description="1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,max,ytd"),
    interval: str = Query("1d", description="1m,5m,15m,30m,1h,1d,1wk,1mo"),
    indicators: str | None = Query(None, description=f"Comma-separated. Available: {sorted(INDICATOR_REGISTRY)}"),
    refresh: bool = Query(False, description="Bypass the cache"),
):
    series = market_data_service.get_history(symbol, period=period, interval=interval, force_refresh=refresh)
    df = series.df
    requested: list[str] = []
    if indicators:
        requested = [i.strip().lower() for i in indicators.split(",") if i.strip()]
        unknown = [i for i in requested if i not in INDICATOR_REGISTRY]
        if unknown:
            raise InvalidRequestError(f"Unknown indicators: {unknown}",
                                      details={"available": sorted(INDICATOR_REGISTRY)})
        df = compute_indicators(df, requested)

    return HistoryResponse(
        symbol=series.symbol, name=series.instrument.name,
        asset_class=series.instrument.asset_class, currency=series.instrument.currency,
        period=period, interval=interval, source=series.source, is_live=series.is_live,
        bars=len(df), candles=frame_to_records(df), indicators=requested,
    )


@router.get("/indicators/{symbol}", summary="Technical indicator values + rule-based signals")
async def get_indicators(
    symbol: str,
    period: str = Query("1y"),
    indicators: str = Query("sma,ema,rsi,macd,bbands,atr,adx,stoch,mfi"),
):
    series = market_data_service.get_history(symbol, period=period)
    requested = [i.strip().lower() for i in indicators.split(",") if i.strip()]
    enriched = compute_indicators(series.df, requested)
    summary = signal_summary(enriched)
    latest = enriched.iloc[-1].to_dict()
    return {
        "symbol": series.symbol, "as_of": str(enriched.index[-1].date()),
        "source": series.source,
        "latest_values": {k: (None if v != v else round(float(v), 6)) for k, v in latest.items()},
        "signals": summary,
    }


@router.get("/profile/{symbol}", summary="Instrument profile & summary statistics")
async def get_profile(symbol: str):
    return market_data_service.get_profile(symbol)


@router.get("/time-ranges", summary="The shared time-range catalogue")
async def time_ranges():
    """One definition of the period control, served to every page.

    The frontend renders exactly what this returns, so a range can never exist
    in the UI that the backend does not understand — the failure mode when each
    page hard-coded its own list.
    """
    from app.utils.periods import catalogue

    return {"ranges": catalogue(), "default": "1y"}


@router.get("/statistics/{symbol}", summary="Return distribution & risk statistics")
async def get_statistics(symbol: str, period: str = "2y", benchmark: str | None = "SPY"):
    series = market_data_service.get_history(symbol, period=period)
    returns = series.df["close"].pct_change().dropna()
    bench_returns = None
    if benchmark and benchmark.upper() != symbol.upper():
        try:
            bench = market_data_service.get_history(benchmark, period=period).df["close"].pct_change().dropna()
            bench_returns = bench.reindex(returns.index).dropna()
            returns = returns.reindex(bench_returns.index)
        except Exception:
            bench_returns = None
    return {
        "symbol": series.symbol, "period": period, "benchmark": benchmark,
        "source": series.source, "metrics": full_metrics(returns, bench_returns),
    }


@router.get("/correlation", summary="Correlation matrix across instruments")
async def get_correlation(
    symbols: str = Query(..., description="Comma-separated symbols"),
    period: str = Query("1y"),
):
    parsed = [s.strip() for s in symbols.split(",") if s.strip()]
    if len(parsed) < 2:
        raise InvalidRequestError("Provide at least 2 symbols")
    matrix = market_data_service.get_returns_matrix(parsed, period=period)
    return {"period": period, **correlation_matrix(matrix)}


@router.get("/movers", summary="Top gainers / losers across a watchlist")
async def get_movers(symbols: str | None = Query(None), limit: int = Query(5, ge=1, le=20)):
    from app.services.data.universe import DEFAULT_WATCHLIST

    parsed = [s.strip() for s in symbols.split(",")] if symbols else DEFAULT_WATCHLIST
    quotes = [q for q in market_data_service.get_quotes(parsed) if q.get("change_percent") is not None]
    ranked = sorted(quotes, key=lambda q: q["change_percent"], reverse=True)
    return {"gainers": ranked[:limit], "losers": ranked[-limit:][::-1], "count": len(ranked)}


@router.get("/health", summary="Data-layer health")
async def data_health():
    return market_data_service.health()
