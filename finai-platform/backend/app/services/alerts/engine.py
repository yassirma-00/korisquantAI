"""Intelligent alert engine.

Generates alerts from five families of triggers:

* price moves beyond a configurable threshold
* volatility spikes (z-score based)
* technical signal flips (RSI extremes, MACD crossovers, band breaks)
* elevated risk (crash-risk score, drawdown, bubble)
* high-impact news
* user-defined rules stored in the database
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.logging import get_logger
from app.db.models import Alert, AlertRule
from app.services.data.market_data import market_data_service
from app.services.indicators.technical import compute_indicators
from app.services.nlp.news import news_service
from app.services.risk.anomaly import anomaly_detector

logger = get_logger(__name__)

SEVERITIES = ("info", "warning", "critical")


class AlertEngine:
    # ---------------------------------------------------------- detectors
    def check_price_moves(self, symbol: str, threshold_pct: float | None = None) -> list[dict]:
        threshold = threshold_pct if threshold_pct is not None else settings.ALERT_PRICE_MOVE_PCT
        quote = market_data_service.get_quote(symbol)
        change = float(quote.get("change_percent") or 0.0)
        if abs(change) < threshold:
            return []
        severity = "critical" if abs(change) > threshold * 2.5 else "warning"
        direction = "surged" if change > 0 else "dropped"
        return [{
            "symbol": symbol.upper(), "alert_type": "price_move", "severity": severity,
            "title": f"{symbol.upper()} {direction} {abs(change):.2f}%",
            "message": (f"{symbol.upper()} {direction} {abs(change):.2f}% to "
                        f"{quote.get('price'):,.4f} (previous close {quote.get('previous_close'):,.4f})."),
            "payload": {"change_percent": change, "price": quote.get("price"),
                        "threshold": threshold, "source": quote.get("source")},
        }]

    def check_volatility(self, symbol: str, z_threshold: float | None = None) -> list[dict]:
        z_threshold = z_threshold if z_threshold is not None else settings.ALERT_VOL_SPIKE_Z
        series = market_data_service.get_history(symbol, period="1y")
        returns = series.df["close"].pct_change().dropna()
        if len(returns) < 60:
            return []
        vol = returns.rolling(21).std() * np.sqrt(252)
        z = float((vol.iloc[-1] - vol.mean()) / (vol.std() or 1))
        if z < z_threshold:
            return []
        return [{
            "symbol": symbol.upper(), "alert_type": "volatility", "severity":
                "critical" if z > z_threshold * 1.6 else "warning",
            "title": f"Volatility spike on {symbol.upper()} ({z:.1f}σ)",
            "message": (f"21-day realised volatility is {vol.iloc[-1]*100:.1f}%, "
                        f"{z:.1f} standard deviations above its one-year average. "
                        "Consider reducing position size or widening stops."),
            "payload": {"z_score": round(z, 2), "volatility_pct": round(float(vol.iloc[-1]) * 100, 2)},
        }]

    def check_technical_signals(self, symbol: str) -> list[dict]:
        series = market_data_service.get_history(symbol, period="1y")
        enriched = compute_indicators(series.df, ["rsi", "macd", "bbands", "sma"])
        if len(enriched) < 60:
            return []
        last, prev = enriched.iloc[-1], enriched.iloc[-2]
        alerts = []

        rsi_now = last.get("rsi", 50)
        if rsi_now < 30:
            alerts.append({
                "symbol": symbol.upper(), "alert_type": "signal", "severity": "info",
                "title": f"{symbol.upper()} oversold (RSI {rsi_now:.0f})",
                "message": f"RSI(14) at {rsi_now:.1f} indicates oversold conditions - potential buying opportunity.",
                "payload": {"indicator": "RSI", "value": round(float(rsi_now), 2), "signal": "buy"}})
        elif rsi_now > 70:
            alerts.append({
                "symbol": symbol.upper(), "alert_type": "signal", "severity": "warning",
                "title": f"{symbol.upper()} overbought (RSI {rsi_now:.0f})",
                "message": f"RSI(14) at {rsi_now:.1f} indicates overbought conditions - consider taking profits.",
                "payload": {"indicator": "RSI", "value": round(float(rsi_now), 2), "signal": "sell"}})

        if "macd_hist" in last and np.isfinite(last["macd_hist"]) and np.isfinite(prev.get("macd_hist", np.nan)):
            if prev["macd_hist"] <= 0 < last["macd_hist"]:
                alerts.append({
                    "symbol": symbol.upper(), "alert_type": "signal", "severity": "info",
                    "title": f"Bullish MACD crossover on {symbol.upper()}",
                    "message": "The MACD line crossed above its signal line - momentum is turning positive.",
                    "payload": {"indicator": "MACD", "signal": "buy",
                                "value": round(float(last["macd_hist"]), 5)}})
            elif prev["macd_hist"] >= 0 > last["macd_hist"]:
                alerts.append({
                    "symbol": symbol.upper(), "alert_type": "signal", "severity": "warning",
                    "title": f"Bearish MACD crossover on {symbol.upper()}",
                    "message": "The MACD line crossed below its signal line - momentum is turning negative.",
                    "payload": {"indicator": "MACD", "signal": "sell",
                                "value": round(float(last["macd_hist"]), 5)}})

        if "bb_pct_b" in last and np.isfinite(last["bb_pct_b"]):
            if last["bb_pct_b"] > 1.0:
                alerts.append({
                    "symbol": symbol.upper(), "alert_type": "signal", "severity": "warning",
                    "title": f"{symbol.upper()} broke above the upper Bollinger band",
                    "message": "Price closed above the upper band - stretched conditions, mean reversion possible.",
                    "payload": {"indicator": "Bollinger", "pct_b": round(float(last["bb_pct_b"]), 3)}})
            elif last["bb_pct_b"] < 0.0:
                alerts.append({
                    "symbol": symbol.upper(), "alert_type": "signal", "severity": "info",
                    "title": f"{symbol.upper()} broke below the lower Bollinger band",
                    "message": "Price closed below the lower band - potential oversold bounce.",
                    "payload": {"indicator": "Bollinger", "pct_b": round(float(last["bb_pct_b"]), 3)}})
        return alerts

    def check_risk(self, symbol: str) -> list[dict]:
        series = market_data_service.get_history(symbol, period="2y")
        scan = anomaly_detector.scan(symbol, series.df, lookback_days=30)
        alerts = []
        crash = scan["crash_risk"]
        if crash.get("crash_risk_score", 0) > 0.55:
            alerts.append({
                "symbol": symbol.upper(), "alert_type": "risk",
                "severity": "critical" if crash["crash_risk_score"] > 0.75 else "warning",
                "title": f"Elevated crash risk on {symbol.upper()} ({crash['level']})",
                "message": (f"Crash-risk score {crash['crash_risk_score']:.2f}. "
                            f"Daily VaR₉₅ {crash['var_95_daily']:.2%}, current drawdown "
                            f"{crash['current_drawdown']:.1%}. {crash['recommendation']}"),
                "payload": crash})

        bubble = scan["bubble"]
        if bubble.get("bubble_score", 0) > 0.6:
            alerts.append({
                "symbol": symbol.upper(), "alert_type": "risk",
                "severity": "critical" if bubble["bubble_score"] > 0.8 else "warning",
                "title": f"Speculative overheating on {symbol.upper()} ({bubble['level']})",
                "message": f"Bubble score {bubble['bubble_score']:.2f}. {bubble['interpretation']}",
                "payload": bubble})

        for anomaly in scan["anomalies"][:3]:
            if anomaly.get("severity") in ("high", "critical"):
                alerts.append({
                    "symbol": symbol.upper(), "alert_type": "anomaly", "severity": "warning",
                    "title": f"Anomaly detected on {symbol.upper()}: {anomaly['type'].replace('_', ' ')}",
                    "message": anomaly.get("description", ""), "payload": anomaly})
        return alerts

    def check_news(self, symbol: str, impact_threshold: float = 0.25) -> list[dict]:
        news = news_service.get_news(symbol, limit=10, analyze=True)
        alerts = []
        for item in news:
            if item.get("impact_score", 0) < impact_threshold:
                continue
            sentiment = item["sentiment"]["label"]
            alerts.append({
                "symbol": symbol.upper(), "alert_type": "news",
                "severity": "warning" if sentiment == "negative" else "info",
                "title": f"High-impact {sentiment} news on {symbol.upper()}",
                "message": item["title"],
                "payload": {"category": item["category"], "source": item["source"],
                            "url": item["url"], "impact_score": item["impact_score"],
                            "sentiment": item["sentiment"]},
            })
        return alerts[:3]

    # ------------------------------------------------------------- scan
    def scan_symbol(self, symbol: str, checks: list[str] | None = None) -> list[dict]:
        checks = checks or ["price", "volatility", "signals", "risk", "news"]
        runners = {
            "price": self.check_price_moves, "volatility": self.check_volatility,
            "signals": self.check_technical_signals, "risk": self.check_risk,
            "news": self.check_news,
        }
        alerts: list[dict] = []
        for check in checks:
            runner = runners.get(check)
            if not runner:
                continue
            try:
                alerts += runner(symbol)
            except Exception as exc:
                logger.warning("alert check '%s' failed for %s: %s", check, symbol, exc)
        order = {"critical": 0, "warning": 1, "info": 2}
        alerts.sort(key=lambda a: order.get(a["severity"], 3))
        for alert in alerts:
            alert["triggered_at"] = datetime.now(UTC).isoformat()
        return alerts

    def scan_watchlist(self, symbols: list[str], checks: list[str] | None = None) -> dict:
        results, total = {}, 0
        for symbol in symbols:
            alerts = self.scan_symbol(symbol, checks)
            if alerts:
                results[symbol.upper()] = alerts
                total += len(alerts)
        return {
            "scanned": len(symbols), "symbols_with_alerts": len(results),
            "total_alerts": total, "alerts": results,
            "critical": sum(1 for a in sum(results.values(), []) if a["severity"] == "critical"),
            "scanned_at": datetime.now(UTC).isoformat(),
        }

    # ----------------------------------------------------------- database
    async def persist(self, db: AsyncSession, alerts: list[dict], user_id: int | None = None) -> list[Alert]:
        stored = []
        for alert in alerts:
            row = Alert(
                user_id=user_id, symbol=alert["symbol"], alert_type=alert["alert_type"],
                severity=alert["severity"], title=alert["title"], message=alert["message"],
                payload=alert.get("payload", {}))
            db.add(row)
            stored.append(row)
        await db.flush()
        return stored

    async def list_alerts(self, db: AsyncSession, symbol: str | None = None,
                          unread_only: bool = False, limit: int = 100,
                          severity: str | None = None,
                          search: str | None = None) -> list[Alert]:
        """Stored alerts, newest first.

        Severity and text filtering happen in SQL, not afterwards in Python.
        Filtering a already-LIMITed page silently searched only the newest N
        rows: with one critical alert among 586, "show critical" returned
        nothing and looked like an empty history rather than a truncated one.
        """
        query = select(Alert)
        if symbol:
            query = query.where(Alert.symbol == symbol.upper())
        if unread_only:
            query = query.where(Alert.is_read.is_(False))
        if severity:
            query = query.where(Alert.severity == severity)
        if search:
            needle = f"%{search.lower()}%"
            query = query.where(
                func.lower(Alert.title).like(needle) | func.lower(Alert.message).like(needle))
        query = query.order_by(Alert.triggered_at.desc()).limit(limit)
        result = await db.execute(query)
        return list(result.scalars().all())

    async def mark_read(self, db: AsyncSession, alert_ids: list[int]) -> int:
        result = await db.execute(select(Alert).where(Alert.id.in_(alert_ids)))
        rows = list(result.scalars().all())
        for row in rows:
            row.is_read = True
        await db.flush()
        return len(rows)

    # -------------------------------------------------------- custom rules
    async def evaluate_rules(self, db: AsyncSession) -> list[dict]:
        """Evaluate every active rule and persist whatever fired.

        Rewritten to run through the shared condition evaluator so that legacy
        single-threshold rules and new multi-condition rules take exactly one
        code path. The previous version inlined four `if rule.rule_type ==`
        branches, which is why adding a metric meant editing this method.
        """
        from app.services.alerts.rules import (
            PRIORITY_SEVERITY,
            build_message,
            evaluate_rule,
            is_expired,
        )

        result = await db.execute(select(AlertRule).where(AlertRule.is_active.is_(True)))
        rules = list(result.scalars().all())
        triggered: list[dict] = []
        now = datetime.now(UTC)

        for rule in rules:
            # An expired rule is switched off rather than skipped, so the list
            # shows why it stopped firing instead of looking mysteriously idle.
            if is_expired(rule, now):
                rule.is_active = False
                logger.info("rule %s expired and was deactivated", rule.id)
                continue

            if rule.last_triggered:
                last = rule.last_triggered
                if last.tzinfo is None:
                    last = last.replace(tzinfo=UTC)
                if now - last < timedelta(minutes=rule.cooldown_minutes):
                    continue

            try:
                verdict = evaluate_rule(rule)
            except Exception as exc:
                logger.warning("rule %s evaluation failed: %s", rule.id, exc)
                continue

            if not verdict["fired"]:
                continue

            priority = getattr(rule, "priority", None) or "medium"
            alert = {
                "symbol": rule.symbol,
                "alert_type": "custom_rule",
                "severity": PRIORITY_SEVERITY.get(priority, "info"),
                "title": (getattr(rule, "name", None)
                          or f"Rule triggered on {rule.symbol}"),
                "message": build_message(rule, verdict),
                # The evidence travels with the alert: the history page can then
                # show which condition fired and on what value, without having
                # to recompute anything.
                "payload": {
                    "rule_id": rule.id,
                    "priority": priority,
                    "logic": verdict["logic"],
                    "period": verdict["period"],
                    "conditions": verdict["conditions"],
                    "notify": {
                        "in_app": bool(getattr(rule, "notify_in_app", True)),
                        "email": bool(getattr(rule, "notify_email", False)),
                        "push": bool(getattr(rule, "notify_push", False)),
                    },
                },
            }
            await self.persist(db, [alert], user_id=rule.user_id)
            rule.last_triggered = now
            rule.trigger_count = (getattr(rule, "trigger_count", 0) or 0) + 1
            # A one-shot rule retires itself; leaving it active would re-fire
            # after every cooldown and bury the history in duplicates.
            if not getattr(rule, "recurring", True):
                rule.is_active = False
            triggered.append(alert)

        await db.flush()
        return triggered


alert_engine = AlertEngine()
