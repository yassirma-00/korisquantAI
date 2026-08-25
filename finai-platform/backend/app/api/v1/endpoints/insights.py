"""Recommendations, news/NLP, risk analytics, XAI and alerts endpoints."""

from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import SymbolNotFoundError
from app.core.logging import get_logger
from app.db.models import AlertRule
from app.db.session import get_db
from app.schemas.common import (
    BatchSentimentRequest,
    BulkRuleActionRequest,
    CreateAlertRuleRequest,
    RecommendRequest,
    ScanRequest,
    SentimentRequest,
    UpdateAlertRuleRequest,
)
from app.services.alerts.engine import alert_engine
from app.services.data.market_data import market_data_service
from app.services.data.universe import DEFAULT_WATCHLIST
from app.services.nlp.news import news_service
from app.services.nlp.sentiment import sentiment_analyzer
from app.services.recommendation.engine import recommendation_engine
from app.services.risk.anomaly import anomaly_detector
from app.services.risk.profile import benchmark_for, risk_profiler
from app.services.risk.regime import market_regime_detector
from app.services.xai.explainer import explainer

logger = get_logger(__name__)

router = APIRouter()

# ---------------------------------------------------------------- signals
signals_router = APIRouter(prefix="/signals", tags=["Recommendations"])


@signals_router.post("/recommend", summary="Full multi-signal recommendation")
async def recommend(request: RecommendRequest):
    return recommendation_engine.recommend(
        symbol=request.symbol, period=request.period,
        forecast_model=request.forecast_model, horizon=request.horizon,
        rl_algo=request.rl_algo, include_xai=request.include_xai)


@signals_router.get("/recommend/{symbol}", summary="Recommendation (GET convenience)")
async def recommend_get(
    symbol: str,
    period: str = Query("2y"),
    forecast_model: str = Query("lstm"),
    horizon: int = Query(5, ge=1, le=60),
    rl_algo: str = Query("dueling_dqn"),
    include_xai: bool = Query(False),
):
    return recommendation_engine.recommend(
        symbol=symbol, period=period, forecast_model=forecast_model,
        horizon=horizon, rl_algo=rl_algo, include_xai=include_xai)


@signals_router.get("/confidence/{symbol}", summary="AI Confidence Score")
async def ai_confidence(symbol: str, period: str = Query("2y"),
                        forecast_model: str = Query("lstm"),
                        rl_algo: str = Query("dueling_dqn")):
    """How much to trust the Buy / Hold / Sell call for this symbol.

    Returned alongside the recommendation itself so the dashboard can render
    the verdict and its reliability from a single request — two round trips
    would let the two panels disagree while one was still loading.
    """
    from app.services.recommendation.confidence import confidence_report

    recommendation = recommendation_engine.recommend(
        symbol=symbol, period=period, forecast_model=forecast_model,
        rl_algo=rl_algo, include_xai=False)
    report = confidence_report(recommendation)
    report["symbol"] = recommendation.get("symbol", symbol.upper())
    report["as_of"] = recommendation.get("as_of")
    report["last_price"] = recommendation.get("last_price")
    report["composite_score"] = recommendation.get("composite_score")
    report["recommendation"] = {
        "action": recommendation.get("action"),
        "composite_score": recommendation.get("composite_score"),
        "signals": recommendation.get("signals"),
    }
    return report


@signals_router.get("/direction/{symbol}", summary="AI Direction Prediction")
async def direction_prediction(
    symbol: str,
    period: str = Query("1y"),
    horizon: int = Query(5, ge=1, le=60),
    forecast_model: str = Query("lstm"),
    rl_algo: str = Query("dueling_dqn"),
):
    """Will this instrument rise, fall, or go nowhere over the horizon?

    A fusion of the four signals the Recommendations page already computes —
    deep-learning forecast, RL agent, technical consensus and news sentiment —
    reduced to a single directional call with a confidence level.

    Nothing here is random or hardcoded. `expected_move_pct` has exactly one
    source: a trained forecaster's own `predicted_return` for this symbol *and*
    this horizon. When no such checkpoint exists the field is `null`,
    `magnitude_basis` is `no_trained_forecaster` and the verdict is `NEUTRAL` —
    the platform does not manufacture a movement it did not predict.

    Realised volatility is reported separately as `market_volatility_pct`, is
    always unsigned, and is never used as an expected move: it measures how far
    a price tends to travel, not which way it goes.

    `period` controls how much history is loaded for the analysis; `horizon` is
    how far ahead the call looks.
    """
    from app.services.recommendation.direction import direction_predictor
    from app.utils.periods import analysis_window

    window = analysis_window(period, model="forecast")
    series = market_data_service.get_history(symbol, period=window)
    df = getattr(series, "df", series)
    if df is None or df.empty:
        raise SymbolNotFoundError(f"No market data for {symbol}")

    result = direction_predictor.predict(
        symbol, df, horizon=horizon,
        forecast_model=forecast_model, rl_algo=rl_algo)
    # Echo the window actually used, so a caller can tell a fresh answer from a
    # stale one and can see that the period selector really reached the models.
    result["period"] = period
    result["analysis_window"] = window
    result["data_source"] = getattr(series, "source", None)
    # The date of the last bar, matching `recommend`: a wall-clock stamp would
    # imply the analysis is fresher than the data behind it.
    result["as_of"] = str(df.index[-1].date())
    return result


@signals_router.post("/screen", summary="Rank a watchlist by composite score")
async def screen(request: ScanRequest):
    return {"results": recommendation_engine.screen(request.symbols)}


@signals_router.get("/screen", summary="Screen the default watchlist")
async def screen_get(symbols: str | None = Query(None)):
    parsed = [s.strip() for s in symbols.split(",")] if symbols else DEFAULT_WATCHLIST
    return {"results": recommendation_engine.screen(parsed)}


# -------------------------------------------------------------------- news
news_router = APIRouter(prefix="/news", tags=["News & NLP"])


@news_router.get("/{symbol}", summary="Latest news with sentiment & impact scoring")
async def get_news(symbol: str, limit: int = Query(12, ge=1, le=50),
                   analyze: bool = Query(True)):
    items = news_service.get_news(symbol, limit=limit, analyze=analyze)
    return {"symbol": symbol.upper(), "count": len(items), "news": items}


@news_router.get("/{symbol}/sentiment", summary="Aggregated sentiment for one instrument")
async def get_sentiment(symbol: str, limit: int = Query(20, ge=1, le=50)):
    return news_service.sentiment_summary(symbol, limit=limit)


@news_router.get("/market/pulse", summary="Market-wide sentiment pulse")
async def market_pulse(symbols: str | None = Query(None)):
    parsed = [s.strip() for s in symbols.split(",")] if symbols else DEFAULT_WATCHLIST[:6]
    return news_service.market_pulse(parsed)


@news_router.post("/analyze", summary="Analyse arbitrary text")
async def analyze_text(request: SentimentRequest):
    from app.services.nlp.news import classify

    result = sentiment_analyzer.analyze(request.text)
    return {**result.to_dict(), "category": classify(request.text)}


@news_router.post("/analyze/batch", summary="Batch sentiment analysis")
async def analyze_batch(request: BatchSentimentRequest):
    results = sentiment_analyzer.analyze_batch(request.texts)
    return {"count": len(results),
            "results": [r.to_dict() for r in results],
            "aggregate": sentiment_analyzer.aggregate(results)}


# -------------------------------------------------------------------- risk
risk_router = APIRouter(prefix="/risk", tags=["Risk & Anomalies"])


@risk_router.get("/scan/{symbol}", summary="Full anomaly & risk scan")
async def scan_risk(symbol: str, period: str = Query("1y"),
                    lookback_days: int = Query(180, ge=7, le=1000)):
    """Risk analytics for the selected window.

    The display period and the computation period are deliberately separate.
    Selecting "1M" must not fit the crash-risk model on 22 bars — it cannot,
    and the honest answer would be a dash on every panel. Instead the model
    reads whatever history it needs and the presentation is trimmed afterwards,
    which is how a terminal behaves.
    """
    from app.utils.periods import analysis_window, model_bars, resolve

    selected = resolve(period)
    # Fetch on the longest floor (bubble, 200 bars) so nothing is ever short of
    # data; each score then reads only the bars it should. `analysis_window`
    # deliberately differs from `compute_period`: over-fetching is right for
    # training a network and wrong for measuring the selected window, and it
    # used to collapse seven of the eleven ranges onto one identical answer.
    fit_period = analysis_window(period, "bubble")
    crash_bars = model_bars(period, "crash_risk")
    bubble_bars = model_bars(period, "bubble")

    series = market_data_service.get_history(symbol, period=fit_period)
    report = anomaly_detector.scan(
        symbol, series.df, lookback_days=lookback_days,
        crash_bars=crash_bars, bubble_bars=bubble_bars)

    # ---- absolute, per-asset risk measures + the weighted Overall Risk Score
    risk_df = series.df.tail(max(crash_bars, bubble_bars))
    bench_symbol = benchmark_for(symbol)
    bench_df = None
    if bench_symbol and bench_symbol.upper() != symbol.upper():
        try:
            bench_df = market_data_service.get_history(
                bench_symbol, period=fit_period).df
        except Exception as exc:      # pragma: no cover - provider dependent
            logger.info("benchmark %s unavailable for %s: %s", bench_symbol, symbol, exc)
            bench_symbol = None

    profile = risk_profiler.profile(
        symbol, risk_df,
        benchmark_df=bench_df, benchmark_symbol=bench_symbol,
        crash=report.get("crash_risk"), bubble=report.get("bubble"),
        recent_anomaly_pressure=report.get("anomaly_pressure"))
    report["risk_profile"] = profile

    overall = profile.get("overall") or {}
    if overall.get("score") is not None:
        # The headline is now a measured score, not the maximum of three band
        # labels. Keep `overall_risk_level` as the single name the whole UI
        # reads, so the badge and the score can never disagree.
        report["overall_risk_level"] = overall["level"]
        report["overall_risk_score"] = overall["score"]
        report["risk_drivers"] = [
            {"source": d["name"], "level": overall["level"], "detail": d["detail"]}
            for d in overall.get("top_drivers", [])
        ]

    report["display_period"] = selected.key
    report["display_label"] = selected.label
    report["computed_over"] = fit_period
    report["crash_bars"] = crash_bars
    report["bubble_bars"] = bubble_bars
    report["computation_note"] = (
        f"Charts show {selected.label}. Crash Risk reads {crash_bars} bars and the "
        f"Bubble Indicator {bubble_bars} — each the longer of your selection and "
        "that model's minimum, so a short window still produces a real score.")
    return report


@risk_router.get("/crash/{symbol}", summary="Crash-risk assessment")
async def crash_risk(symbol: str, period: str = Query("1y")):
    """Crash risk over the selected window, floored at the model's 60-bar minimum."""
    from app.utils.periods import analysis_window, model_bars

    series = market_data_service.get_history(
        symbol, period=analysis_window(period, "crash_risk"))
    bars = model_bars(period, "crash_risk")
    report = anomaly_detector.crash_risk(series.df.tail(bars))
    return {"symbol": symbol.upper(), "bars_used": min(bars, len(series.df)), **report}


@risk_router.get("/bubble/{symbol}", summary="Bubble / overheating indicator")
async def bubble(symbol: str, period: str = Query("1y")):
    """Bubble reading over the selected window, floored at its 200-bar minimum."""
    from app.utils.periods import analysis_window, model_bars

    series = market_data_service.get_history(
        symbol, period=analysis_window(period, "bubble"))
    bars = model_bars(period, "bubble")
    report = anomaly_detector.bubble_indicator(series.df.tail(bars))
    return {"symbol": symbol.upper(), "bars_used": min(bars, len(series.df)), **report}


@risk_router.get("/profile/{symbol}", summary="Full quantitative risk profile")
async def risk_profile(symbol: str, period: str = Query("1y"),
                       benchmark: str | None = Query(None)):
    """Volatility, VaR, CVaR, drawdown, beta, Sharpe, Sortino and the weighted
    Overall Risk Score — every figure computed from this symbol's own history
    over the selected window."""
    from app.utils.periods import analysis_window, model_bars, resolve

    selected = resolve(period)
    fit_period = analysis_window(period, "bubble")
    series = market_data_service.get_history(symbol, period=fit_period)
    bars = max(model_bars(period, "crash_risk"), model_bars(period, "bubble"))
    window = series.df.tail(bars)

    bench_symbol = benchmark or benchmark_for(symbol)
    bench_df = None
    if bench_symbol and bench_symbol.upper() != symbol.upper():
        try:
            bench_df = market_data_service.get_history(bench_symbol, period=fit_period).df
        except Exception as exc:      # pragma: no cover - provider dependent
            logger.info("benchmark %s unavailable: %s", bench_symbol, exc)
            bench_symbol = None

    report = risk_profiler.profile(
        symbol, window, benchmark_df=bench_df, benchmark_symbol=bench_symbol,
        crash=anomaly_detector.crash_risk(series.df.tail(model_bars(period, "crash_risk"))),
        bubble=anomaly_detector.bubble_indicator(series.df.tail(model_bars(period, "bubble"))))
    report["display_period"] = selected.key
    report["display_label"] = selected.label
    report["data_source"] = series.source
    return report


@risk_router.get("/regime/{symbol}", summary="Market-regime detection")
async def market_regime(symbol: str, period: str = Query("2y"),
                        include_sentiment: bool = Query(True),
                        timeline_step: int = Query(5, ge=1, le=21)):
    """Classify the current market regime, with the evidence behind the call.

    News sentiment is optional and fails soft: it is one weighted factor among
    nine, and an unreachable news provider must not take the whole panel down
    with it.
    """
    series = market_data_service.get_history(symbol, period=period)

    sentiment = None
    if include_sentiment:
        try:
            articles = news_service.get_news(symbol, limit=10, analyze=True)
            results = [a["sentiment"] for a in articles if a.get("sentiment")]
            if results:
                sentiment = {
                    "score": round(
                        sum(r.get("score", 0.0) for r in results) / len(results), 4),
                    "articles": len(results),
                    "source": "headline lexicon",
                }
        except Exception as exc:      # pragma: no cover - provider dependent
            logger.info("regime sentiment unavailable for %s: %s", symbol, exc)

    report = market_regime_detector.detect(
        symbol, series.df, sentiment=sentiment, timeline_step=timeline_step)
    report["data_source"] = series.source
    # Where this reading is consumed elsewhere, so the panel can link out
    # instead of leaving the integration implicit.
    report["related"] = {
        "risk_scan": f"/api/v1/risk/scan/{symbol.upper()}?period={period}",
        "recommendation": f"/api/v1/signals/recommend/{symbol.upper()}",
        "rl_agents": f"/api/v1/rl/agents?symbol={symbol.upper()}",
        "portfolio": "/api/v1/portfolio",
    }
    return report


# --------------------------------------------------------------------- XAI
xai_router = APIRouter(prefix="/xai", tags=["Explainable AI"])


@xai_router.get("/explain/{symbol}", summary="SHAP + LIME + global importance")
async def explain(
    symbol: str,
    period: str = Query("2y"),
    horizon: int = Query(5, ge=1, le=60),
    methods: str = Query("shap,lime,global", description="shap, lime, global, counterfactual"),
):
    series = market_data_service.get_history(symbol, period=period)
    parsed = [m.strip().lower() for m in methods.split(",") if m.strip()]
    return explainer.explain(symbol, series.df, horizon=horizon, methods=parsed)


@xai_router.get("/importance/{symbol}", summary="Global feature importance")
async def importance(symbol: str, period: str = Query("2y"),
                     horizon: int = Query(5, ge=1, le=60), top_k: int = Query(12, ge=3, le=30)):
    series = market_data_service.get_history(symbol, period=period)
    return explainer.global_importance(symbol, series.df, horizon=horizon, top_k=top_k)


@xai_router.get("/counterfactual/{symbol}", summary="What would flip the forecast?")
async def counterfactual(symbol: str, period: str = Query("2y"), horizon: int = Query(5, ge=1, le=60)):
    series = market_data_service.get_history(symbol, period=period)
    return explainer.counterfactual(symbol, series.df, horizon=horizon)


# ------------------------------------------------------------------ alerts
alerts_router = APIRouter(prefix="/alerts", tags=["Alerts"])


@alerts_router.get("/scan/{symbol}", summary="Scan one instrument for alerts")
async def scan_symbol(symbol: str, checks: str | None = Query(None)):
    parsed = [c.strip() for c in checks.split(",")] if checks else None
    alerts = alert_engine.scan_symbol(symbol, parsed)
    return {"symbol": symbol.upper(), "count": len(alerts), "alerts": alerts}


@alerts_router.post("/scan", summary="Scan a watchlist (optionally persist)")
async def scan_watchlist(request: ScanRequest, db: AsyncSession = Depends(get_db)):
    result = alert_engine.scan_watchlist(request.symbols, request.checks)
    if request.persist:
        flat = [a for alerts in result["alerts"].values() for a in alerts]
        await alert_engine.persist(db, flat)
        result["persisted"] = len(flat)
    return result


@alerts_router.get("", summary="List stored alerts")
async def list_alerts(symbol: str | None = Query(None), unread_only: bool = Query(False),
                      severity: str | None = Query(None),
                      rule_id: int | None = Query(None),
                      q: str | None = Query(None, description="search title or message"),
                      limit: int = Query(50, ge=1, le=500),
                      db: AsyncSession = Depends(get_db)):
    """Alert history, filterable.

    With hundreds of stored alerts an unfiltered list is an archive, not a
    history: the filters are what make it answerable.
    """
    rows = await alert_engine.list_alerts(db, symbol, unread_only, limit,
                                          severity=severity, search=q)

    if rule_id is not None:
        rows = [a for a in rows if (a.payload or {}).get("rule_id") == rule_id]
    def _entry(alert) -> dict:
        payload = alert.payload or {}
        conditions = payload.get("conditions") or []
        # The trigger values, spelled out. A history that says only "rule fired"
        # cannot be audited after the fact.
        triggers = [
            {"metric": c.get("metric"), "observed": c.get("observed"),
             "target": c.get("target"), "description": c.get("description"),
             "passed": c.get("passed")}
            for c in conditions
        ]
        return {
            "id": alert.id, "symbol": alert.symbol, "alert_type": alert.alert_type,
            "severity": alert.severity, "title": alert.title, "message": alert.message,
            "payload": payload, "is_read": alert.is_read,
            "priority": payload.get("priority"),
            "rule_id": payload.get("rule_id"),
            "logic": payload.get("logic"),
            "period": payload.get("period"),
            "triggers": triggers,
            "reason": alert.message,
            "triggered_at": alert.triggered_at.isoformat(),
        }

    return {"count": len(rows), "alerts": [_entry(a) for a in rows]}


@alerts_router.post("/read", summary="Mark alerts as read")
async def mark_read(alert_ids: list[int], db: AsyncSession = Depends(get_db)):
    return {"updated": await alert_engine.mark_read(db, alert_ids)}


def _rule_payload(rule: AlertRule) -> dict:
    """One serialisation used by every rule endpoint, so create, list and edit
    can never drift into returning different shapes."""
    from app.services.alerts.rules import describe_condition, normalise_conditions

    conditions = normalise_conditions(rule)
    return {
        "id": rule.id,
        "symbol": rule.symbol,
        "name": rule.name or f"{rule.symbol} alert",
        "rule_type": rule.rule_type,
        "threshold": rule.threshold,
        "conditions": conditions,
        "summary": [describe_condition(c) for c in conditions],
        "logic": rule.logic or "AND",
        "priority": rule.priority or "medium",
        "period": rule.period or "6mo",
        "is_active": rule.is_active,
        "cooldown_minutes": rule.cooldown_minutes,
        "notify": {"in_app": bool(rule.notify_in_app), "email": bool(rule.notify_email),
                   "push": bool(rule.notify_push)},
        "expires_at": rule.expires_at.isoformat() if rule.expires_at else None,
        "recurring": bool(rule.recurring),
        "trigger_count": rule.trigger_count or 0,
        "template": rule.template,
        "last_triggered": rule.last_triggered.isoformat() if rule.last_triggered else None,
        "created_at": rule.created_at.isoformat() if rule.created_at else None,
    }


def _apply_conditions(rule: AlertRule, conditions: list) -> None:
    """Store conditions, rejecting a rule that could never evaluate."""
    from app.core.exceptions import InvalidRequestError
    from app.services.alerts.rules import validate_conditions

    raw = [c.model_dump() if hasattr(c, "model_dump") else dict(c) for c in conditions]
    problems = validate_conditions(raw)
    if problems:
        # Saving a rule that silently never fires is worse than refusing it:
        # the user believes they are covered when nothing is watching.
        raise InvalidRequestError("This rule cannot be evaluated",
                                  details={"problems": problems})
    rule.conditions = raw


@alerts_router.get("/metrics", summary="Metrics available to alert conditions")
async def alert_metrics():
    """Powers the condition builder, so the UI can never offer a metric the
    evaluator does not implement."""
    from app.services.alerts.metrics import (
        ACTION_CHOICES,
        METRIC_SPECS,
        OPERATORS,
        REGIME_CHOICES,
    )
    from app.utils.timeseries import VALID_PERIODS

    return {
        "metrics": [
            {"key": m.key, "label": m.label, "unit": m.unit, "group": m.group,
             "hint": m.hint, "default": m.default} for m in METRIC_SPECS
        ],
        "categorical": [
            {"key": "regime", "label": "Market regime", "group": "ai",
             "choices": list(REGIME_CHOICES),
             "hint": "The regime detected by Market Regime Detection"},
            {"key": "ai_action", "label": "AI recommendation", "group": "ai",
             "choices": list(ACTION_CHOICES),
             "hint": "The ensemble's BUY / HOLD / SELL call"},
        ],
        "operators": [{"key": k, "symbol": v[0]} for k, v in OPERATORS.items()],
        "priorities": ["low", "medium", "high", "critical"],
        # Monthly and yearly windows, so an RSI condition can be asked over a
        # month or a decade rather than one hard-coded default.
        "periods": [p for p in VALID_PERIODS if p not in ("1d", "5d")],
    }


@alerts_router.get("/templates", summary="One-click alert templates")
async def alert_templates():
    from app.services.alerts.rules import TEMPLATES, describe_condition

    return {"templates": [
        {**t, "summary": [describe_condition(c) for c in t["conditions"]]}
        for t in TEMPLATES
    ]}


@alerts_router.post("/rules", summary="Create a custom alert rule")
async def create_rule(request: CreateAlertRuleRequest, db: AsyncSession = Depends(get_db)):
    rule = AlertRule(
        symbol=request.symbol.upper(),
        rule_type=request.rule_type,
        threshold=request.threshold,
        cooldown_minutes=request.cooldown_minutes,
        name=request.name,
        logic=request.logic,
        priority=request.priority,
        period=request.period,
        notify_in_app=request.notify_in_app,
        notify_email=request.notify_email,
        notify_push=request.notify_push,
        expires_at=request.expires_at,
        recurring=request.recurring,
        template=request.template,
    )
    if request.conditions:
        _apply_conditions(rule, request.conditions)
    else:
        # Single-condition form: normalise it into the same shape so the rest of
        # the system only ever sees one representation.
        from app.services.alerts.rules import LEGACY_RULE_MAP
        metric, operator = LEGACY_RULE_MAP.get(
            request.rule_type, (request.rule_type, "above"))
        rule.conditions = [{"metric": metric, "operator": operator,
                            "value": request.threshold}]
    db.add(rule)
    await db.flush()
    return _rule_payload(rule)


@alerts_router.post("/rules/from-template/{key}", summary="Create a rule from a template")
async def create_from_template(key: str, symbol: str = Query(...),
                               db: AsyncSession = Depends(get_db)):
    from app.core.exceptions import InvalidRequestError
    from app.services.alerts.rules import TEMPLATES_BY_KEY

    template = TEMPLATES_BY_KEY.get(key)
    if template is None:
        raise InvalidRequestError(f"Unknown template '{key}'",
                                  details={"available": list(TEMPLATES_BY_KEY)})
    rule = AlertRule(
        symbol=symbol.upper(), rule_type="custom", threshold=0.0,
        name=template["name"], conditions=list(template["conditions"]),
        logic=template["logic"], priority=template["priority"],
        period=template["period"], template=key,
    )
    db.add(rule)
    await db.flush()
    return _rule_payload(rule)


@alerts_router.get("/rules", summary="List custom alert rules")
async def list_rules(
    symbol: str | None = Query(None),
    priority: str | None = Query(None),
    status: str | None = Query(None, description="active | paused"),
    q: str | None = Query(None, description="search name or symbol"),
    sort: str = Query("created", description="created | symbol | priority | triggers"),
    db: AsyncSession = Depends(get_db),
):
    """The rules a user created, with the filtering the list needs once it
    grows past a handful."""
    result = await db.execute(select(AlertRule))
    rules = list(result.scalars().all())

    if symbol:
        rules = [r for r in rules if r.symbol.upper() == symbol.upper()]
    if priority:
        rules = [r for r in rules if (r.priority or "medium") == priority]
    if status == "active":
        rules = [r for r in rules if r.is_active]
    elif status == "paused":
        rules = [r for r in rules if not r.is_active]
    if q:
        needle = q.lower()
        rules = [r for r in rules
                 if needle in r.symbol.lower() or needle in (r.name or "").lower()]

    order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    if sort == "symbol":
        rules.sort(key=lambda r: r.symbol)
    elif sort == "priority":
        rules.sort(key=lambda r: order.get(r.priority or "medium", 9))
    elif sort == "triggers":
        rules.sort(key=lambda r: r.trigger_count or 0, reverse=True)
    else:
        rules.sort(key=lambda r: r.created_at or datetime.min, reverse=True)

    return {"count": len(rules), "rules": [_rule_payload(r) for r in rules]}


@alerts_router.patch("/rules/{rule_id}", summary="Edit an alert rule")
async def update_rule(rule_id: int, request: UpdateAlertRuleRequest,
                      db: AsyncSession = Depends(get_db)):
    rule = await db.get(AlertRule, rule_id)
    if rule is None:
        raise SymbolNotFoundError(f"No alert rule with id {rule_id}")

    data = request.model_dump(exclude_unset=True)
    if "conditions" in data and data["conditions"] is not None:
        _apply_conditions(rule, request.conditions)
        data.pop("conditions")
    if data.get("symbol"):
        rule.symbol = data.pop("symbol").upper()
    for field, value in data.items():
        if value is not None:
            setattr(rule, field, value)
    await db.flush()
    return _rule_payload(rule)


@alerts_router.post("/rules/{rule_id}/duplicate", summary="Duplicate a rule")
async def duplicate_rule(rule_id: int, db: AsyncSession = Depends(get_db)):
    """Copying a working rule onto another symbol is the most common way a
    second rule gets made; retyping every field invites a mistake."""
    source = await db.get(AlertRule, rule_id)
    if source is None:
        raise SymbolNotFoundError(f"No alert rule with id {rule_id}")
    clone = AlertRule(
        symbol=source.symbol, rule_type=source.rule_type, threshold=source.threshold,
        cooldown_minutes=source.cooldown_minutes,
        name=f"{source.name or source.symbol} (copy)",
        conditions=list(source.conditions or []), logic=source.logic,
        priority=source.priority, period=source.period,
        notify_in_app=source.notify_in_app, notify_email=source.notify_email,
        notify_push=source.notify_push, recurring=source.recurring,
        template=source.template,
    )
    db.add(clone)
    await db.flush()
    return _rule_payload(clone)


@alerts_router.post("/rules/bulk", summary="Act on several rules at once")
async def bulk_rules(request: BulkRuleActionRequest, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(AlertRule).where(AlertRule.id.in_(request.rule_ids)))
    rules = list(result.scalars().all())
    affected = 0
    created: list[dict] = []
    for rule in rules:
        if request.action == "enable":
            rule.is_active = True
        elif request.action == "disable":
            rule.is_active = False
        elif request.action == "delete":
            await db.delete(rule)
        elif request.action == "duplicate":
            clone = AlertRule(
                symbol=rule.symbol, rule_type=rule.rule_type, threshold=rule.threshold,
                cooldown_minutes=rule.cooldown_minutes,
                name=f"{rule.name or rule.symbol} (copy)",
                conditions=list(rule.conditions or []), logic=rule.logic,
                priority=rule.priority, period=rule.period, template=rule.template,
            )
            db.add(clone)
            created.append({"from": rule.id})
        affected += 1
    await db.flush()
    return {"action": request.action, "affected": affected, "created": len(created)}


@alerts_router.delete("/rules/{rule_id}", summary="Delete a custom alert rule")
async def delete_rule(rule_id: int, db: AsyncSession = Depends(get_db)):
    rule = await db.get(AlertRule, rule_id)
    if rule is None:
        raise SymbolNotFoundError(f"No alert rule with id {rule_id}")
    await db.delete(rule)
    return {"deleted": rule_id}


@alerts_router.post("/rules/{rule_id}/toggle", summary="Enable or disable a rule")
async def toggle_rule(rule_id: int, db: AsyncSession = Depends(get_db)):
    """Pausing a rule is not the same as deleting it: a noisy threshold is worth
    silencing for a while without losing how it was configured."""
    rule = await db.get(AlertRule, rule_id)
    if rule is None:
        raise SymbolNotFoundError(f"No alert rule with id {rule_id}")
    rule.is_active = not rule.is_active
    return {"id": rule.id, "is_active": rule.is_active}


@alerts_router.post("/rules/evaluate", summary="Evaluate all active rules now")
async def evaluate_rules(db: AsyncSession = Depends(get_db)):
    triggered = await alert_engine.evaluate_rules(db)
    return {"triggered": len(triggered), "alerts": triggered}


for sub in (signals_router, news_router, risk_router, xai_router, alerts_router):
    router.include_router(sub)
