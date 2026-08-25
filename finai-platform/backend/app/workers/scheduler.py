"""Background scheduler: periodic alert scans, snapshots and cache warming."""

from __future__ import annotations

import asyncio

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger

from app.core.config import settings
from app.core.logging import get_logger
from app.db.session import db_context
from app.services.alerts.engine import alert_engine
from app.services.data.market_data import market_data_service
from app.services.data.universe import DEFAULT_WATCHLIST
from app.services.recommendation.portfolio import portfolio_service

logger = get_logger(__name__)


def warm_cache() -> None:
    """Pre-fetch quotes and history for the default watchlist."""
    try:
        market_data_service.get_quotes(DEFAULT_WATCHLIST)
        for symbol in DEFAULT_WATCHLIST[:4]:
            market_data_service.get_history(symbol, period="1y")
        logger.info("cache warmed for %d symbols", len(DEFAULT_WATCHLIST))
    except Exception as exc:
        logger.warning("cache warming failed: %s", exc)


def scan_alerts() -> None:
    async def _run() -> None:
        try:
            result = alert_engine.scan_watchlist(DEFAULT_WATCHLIST[:6],
                                                 checks=["price", "volatility", "risk"])
            flat = [a for alerts in result["alerts"].values() for a in alerts]
            if flat:
                async with db_context() as db:
                    await alert_engine.persist(db, flat)
                logger.info("scheduler persisted %d alerts", len(flat))
        except Exception as exc:
            logger.warning("scheduled alert scan failed: %s", exc)

    asyncio.run(_run())


def evaluate_rules() -> None:
    async def _run() -> None:
        try:
            async with db_context() as db:
                triggered = await alert_engine.evaluate_rules(db)
            if triggered:
                logger.info("scheduler triggered %d custom rules", len(triggered))
        except Exception as exc:
            logger.warning("rule evaluation failed: %s", exc)

    asyncio.run(_run())


def snapshot_portfolios() -> None:
    async def _run() -> None:
        try:
            async with db_context() as db:
                for portfolio in await portfolio_service.list_all(db):
                    await portfolio_service.take_snapshot(db, portfolio.id)
        except Exception as exc:
            logger.warning("portfolio snapshot failed: %s", exc)

    asyncio.run(_run())


def start_scheduler() -> BackgroundScheduler:
    scheduler = BackgroundScheduler(timezone="UTC")
    interval = settings.SCHEDULER_INTERVAL_MINUTES
    scheduler.add_job(warm_cache, IntervalTrigger(minutes=max(interval // 3, 5)),
                      id="warm_cache", replace_existing=True)
    scheduler.add_job(scan_alerts, IntervalTrigger(minutes=interval),
                      id="scan_alerts", replace_existing=True)
    scheduler.add_job(evaluate_rules, IntervalTrigger(minutes=max(interval // 2, 5)),
                      id="evaluate_rules", replace_existing=True)
    scheduler.add_job(snapshot_portfolios, IntervalTrigger(hours=6),
                      id="snapshot_portfolios", replace_existing=True)
    scheduler.start()
    return scheduler
