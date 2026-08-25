"""Aggregated dashboard endpoints - one call powers the whole home screen."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime

from fastapi import APIRouter, Query

from app.core.logging import get_logger
from app.services.data.market_data import market_data_service
from app.services.data.universe import DEFAULT_WATCHLIST
from app.services.indicators.technical import compute_indicators, signal_summary
from app.services.nlp.news import news_service
from app.services.risk.anomaly import anomaly_detector
from app.services.risk.metrics import full_metrics
from app.utils.timeseries import frame_to_records

logger = get_logger(__name__)
router = APIRouter(prefix="/dashboard", tags=["Dashboard"])


@router.get("/overview", summary="Market overview: indices, movers, sentiment pulse")
async def overview(watchlist: str | None = Query(None)):
    symbols = [s.strip() for s in watchlist.split(",")] if watchlist else DEFAULT_WATCHLIST
    indices = ["^GSPC", "^IXIC", "^FCHI", "^VIX"]

    with ThreadPoolExecutor(max_workers=4) as pool:
        f_watch = pool.submit(market_data_service.get_quotes, symbols)
        f_idx = pool.submit(market_data_service.get_quotes, indices)
        f_pulse = pool.submit(news_service.market_pulse, symbols[:5])
        quotes = f_watch.result()
        index_quotes = f_idx.result()
        try:
            pulse = f_pulse.result()
        except Exception as exc:
            logger.warning("pulse failed: %s", exc)
            pulse = {"mood": "unknown", "score": 0.0}

    valid = [q for q in quotes if q.get("change_percent") is not None]
    ranked = sorted(valid, key=lambda q: q["change_percent"], reverse=True)
    advancing = sum(1 for q in valid if q["change_percent"] > 0)

    return {
        "as_of": datetime.now(UTC).isoformat(),
        "data_health": market_data_service.health(),
        "indices": index_quotes,
        "watchlist": quotes,
        "top_gainers": ranked[:3],
        "top_losers": ranked[-3:][::-1],
        "breadth": {
            "advancing": advancing, "declining": len(valid) - advancing,
            "advance_decline_ratio": round(advancing / max(len(valid) - advancing, 1), 2),
            "average_change_pct": round(sum(q["change_percent"] for q in valid) / max(len(valid), 1), 3),
        },
        "sentiment_pulse": pulse,
    }


@router.get("/symbol/{symbol}", summary="Everything needed to render one instrument page")
async def symbol_dashboard(
    symbol: str,
    period: str = Query("1y"),
    indicators: str = Query("sma,ema,rsi,macd,bbands,atr"),
):
    series = market_data_service.get_history(symbol, period=period)
    requested = [i.strip() for i in indicators.split(",") if i.strip()]
    enriched = compute_indicators(series.df, requested)
    returns = series.df["close"].pct_change().dropna()

    with ThreadPoolExecutor(max_workers=3) as pool:
        f_news = pool.submit(news_service.sentiment_summary, symbol, 10)
        f_risk = pool.submit(anomaly_detector.scan, symbol, series.df, 120)
        try:
            sentiment = f_news.result()
        except Exception as exc:
            logger.warning("sentiment failed: %s", exc)
            sentiment = {}
        try:
            risk = f_risk.result()
        except Exception as exc:
            logger.warning("risk scan failed: %s", exc)
            risk = {}

    return {
        "symbol": series.symbol,
        "profile": {
            **series.instrument.to_dict(),
            "data_source": series.source, "is_live": series.is_live,
        },
        "quote": market_data_service.get_quote(symbol),
        "candles": frame_to_records(enriched),
        "signals": signal_summary(enriched),
        "statistics": full_metrics(returns),
        "sentiment": sentiment,
        "risk": {
            "overall_risk_level": risk.get("overall_risk_level"),
            "crash_risk": risk.get("crash_risk", {}),
            "bubble": risk.get("bubble", {}),
            "anomalies": risk.get("anomalies", [])[:12],
            "n_anomalies": risk.get("n_anomalies", 0),
        },
    }


@router.get("/heatmap", summary="Performance heatmap across the universe")
async def heatmap(
    asset_class: str | None = Query(None),
    period: str = Query("1mo"),
    limit: int = Query(24, ge=4, le=60),
):
    instruments = market_data_service.search(asset_class=asset_class)[:limit]

    def _perf(inst: dict) -> dict:
        try:
            series = market_data_service.get_history(inst["symbol"], period=period)
            close = series.df["close"]
            change = float(close.iloc[-1] / close.iloc[0] - 1) * 100
            vol = float(close.pct_change().std() * (252 ** 0.5) * 100)
            return {"symbol": inst["symbol"], "name": inst["name"],
                    "asset_class": inst["asset_class"], "sector": inst.get("sector"),
                    "change_pct": round(change, 2), "volatility_pct": round(vol, 2),
                    "last_price": round(float(close.iloc[-1]), 4), "source": series.source}
        except Exception as exc:
            logger.debug("heatmap failed for %s: %s", inst["symbol"], exc)
            return {}

    with ThreadPoolExecutor(max_workers=8) as pool:
        cells = [c for c in pool.map(_perf, instruments) if c]

    return {"period": period, "count": len(cells),
            "cells": sorted(cells, key=lambda c: c["change_pct"], reverse=True)}
