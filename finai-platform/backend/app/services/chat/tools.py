"""Tools the assistant can call to read real platform data.

Why tools instead of a prompt full of numbers
---------------------------------------------
A language model asked about "NVDA's RSI" will happily invent a plausible
number. That is the single worst failure mode for a finance product, because a
fabricated figure is indistinguishable from a real one at a glance.

So the assistant is given **no** market figures in its prompt. It has to call a
function, and every function here reads from the very same services that render
the dashboard — ``market_data_service``, ``recommendation_engine``,
``rl_service`` and friends. If a number appears in the chat, it came out of the
platform, and the user can verify it on the matching page.

Payload discipline
------------------
Each tool returns a *trimmed* dict. The raw ``/signals/recommend`` response is
~6 KB of nested JSON; feeding that to a free-tier model wastes context and
invites the model to fixate on irrelevant fields. We keep what answers a
question and drop the rest.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

from app.core.logging import get_logger

logger = get_logger(__name__)

# How long any single tool may run before we give up on it. The RL agent
# decision measures ~1.5 s and a cold forecast a little more; 45 s is generous
# while still bounding a hung provider call.
TOOL_TIMEOUT = 45.0


def _round(value: Any, digits: int = 4) -> Any:
    """Round floats for the model: 12 decimal places are noise in a chat."""
    if isinstance(value, float):
        return round(value, digits)
    if isinstance(value, dict):
        return {k: _round(v, digits) for k, v in value.items()}
    if isinstance(value, list):
        return [_round(v, digits) for v in value]
    return value


# ============================================================ tool functions
def tool_get_quote(symbol: str) -> dict:
    from app.services.data.market_data import market_data_service
    from app.services.data.universe import get_instrument

    quote = market_data_service.get_quote(symbol)
    synthetic = quote.get("source") == "synthetic"
    known = get_instrument(symbol) is not None

    payload = _round({
        "symbol": quote.get("symbol"),
        "name": quote.get("name"),
        "asset_class": quote.get("asset_class"),
        "price": quote.get("price"),
        "change": quote.get("change"),
        "change_percent": quote.get("change_percent"),
        "day_high": quote.get("day_high"),
        "day_low": quote.get("day_low"),
        "volume": quote.get("volume"),
        # `source` matters: "synthetic" means no live provider answered and these
        # are simulated prices. The model is instructed to disclose that.
        "data_source": quote.get("source"),
        "is_simulated": synthetic,
        "in_known_universe": known,
    })

    # An unknown ticker that only the synthetic engine could price is almost
    # certainly a typo or an instrument this platform does not cover. The
    # generator never fails, so without this flag a fabricated $100.00 reads
    # exactly like a real quote — the single most misleading thing the
    # assistant could relay.
    if synthetic and not known:
        payload["warning"] = (
            f"'{quote.get('symbol')}' is not in the platform's instrument universe and no "
            "live provider returned a price. The figures above are SIMULATED by the "
            "built-in synthetic engine, not market data. Tell the user the ticker could "
            "not be verified and do not present these numbers as a real quote.")
        payload["ticker_verified"] = False
    else:
        payload["ticker_verified"] = True
    return payload


def tool_get_technical_analysis(symbol: str, period: str = "1y") -> dict:
    from app.services.data.market_data import market_data_service
    from app.services.indicators.technical import compute_indicators, signal_summary

    series = market_data_service.get_history(symbol, period=period)
    enriched = compute_indicators(
        series.df, ["sma", "ema", "rsi", "macd", "bbands", "atr", "adx", "stoch", "mfi"])
    summary = signal_summary(enriched)
    last = enriched.iloc[-1]

    def val(column: str) -> float | None:
        raw = last.get(column)
        return None if raw is None or raw != raw else round(float(raw), 4)

    return _round({
        "symbol": symbol.upper(),
        "as_of": str(enriched.index[-1].date()),
        "bars_analysed": int(len(enriched)),
        "data_source": series.source,
        "last_close": val("close"),
        "indicators": {
            "rsi_14": val("rsi"), "macd": val("macd"), "macd_signal": val("macd_signal"),
            "sma_20": val("sma_20"), "sma_50": val("sma_50"), "sma_200": val("sma_200"),
            "ema_12": val("ema_12"), "ema_26": val("ema_26"),
            "bb_upper": val("bb_upper"), "bb_lower": val("bb_lower"),
            "atr_14": val("atr"), "adx_14": val("adx"),
            "stoch_k": val("stoch_k"), "mfi_14": val("mfi"),
        },
        "consensus": summary.get("consensus"),
        "buy_votes": summary.get("buy_votes"),
        "sell_votes": summary.get("sell_votes"),
        "neutral_votes": summary.get("neutral_votes"),
        "strength": summary.get("strength"),
        "per_indicator_signals": {
            name: {"signal": info.get("signal"), "note": info.get("note")}
            for name, info in (summary.get("indicators") or {}).items()
        },
    })


def tool_get_recommendation(symbol: str, forecast_model: str = "lstm",
                            rl_algo: str = "dueling_dqn", horizon: int = 5) -> dict:
    from app.services.recommendation.engine import recommendation_engine

    reco = recommendation_engine.recommend(
        symbol=symbol, forecast_model=forecast_model, rl_algo=rl_algo,
        horizon=horizon, include_xai=False)

    # Signals that could not run are as informative as the ones that did: a
    # recommendation built on 2 of 4 signals deserves to be read differently.
    signals = []
    for sig in reco.get("signals", []):
        entry = {
            "source": sig.get("source"),
            "available": sig.get("available"),
            "score": sig.get("score"),
            "reliability": sig.get("reliability"),
        }
        if not sig.get("available"):
            entry["unavailable_reason"] = (sig.get("detail") or {}).get("reason")
        signals.append(entry)

    return _round({
        "symbol": reco["symbol"],
        "name": reco.get("name"),
        "as_of": reco.get("as_of"),
        "last_price": reco.get("last_price"),
        "action": reco["action"],
        "composite_score": reco.get("composite_score"),
        "confidence": reco.get("confidence"),
        "signal_agreement": reco.get("signal_agreement"),
        "risk_adjustment": reco.get("risk_adjustment"),
        "signals": signals,
        "risk_level": (reco.get("risk") or {}).get("overall_level"),
        "position_sizing": reco.get("position_sizing"),
        "explanation": reco.get("explanation"),
        "data_source": reco.get("data_source"),
    })


def tool_get_forecast(symbol: str, model: str = "lstm", horizon: int = 5) -> dict:
    from app.services.data.market_data import market_data_service
    from app.services.forecasting.trainer import forecast_trainer

    if not forecast_trainer.is_trained(symbol, model, horizon):
        # Training takes minutes; silently kicking one off would hang the chat.
        # Say so and point at the page that does it.
        return {
            "error": "model_not_trained",
            "symbol": symbol.upper(), "model": model, "horizon": horizon,
            "message": (f"No trained {model.upper()} checkpoint exists for "
                        f"{symbol.upper()} at horizon {horizon}. Train it on the "
                        f"AI Forecasting page — it takes a couple of minutes."),
        }

    series = market_data_service.get_history(symbol, period="2y")
    pred = forecast_trainer.predict(symbol, series.df, model_name=model, horizon=horizon)
    metrics = pred.get("test_metrics") or {}
    coverage = pred.get("coverage_validation") or {}
    return _round({
        "symbol": symbol.upper(),
        "model": model,
        "horizon_days": horizon,
        "last_price": pred.get("last_price"),
        "predicted_price": pred.get("predicted_price"),
        "predicted_return_pct": (pred.get("predicted_return") or 0) * 100,
        "direction": pred.get("direction"),
        "confidence": pred.get("confidence"),
        "interval_method": pred.get("interval_method"),
        "confidence_level": pred.get("confidence_level"),
        "conformal_half_width": pred.get("conformal_half_width"),
        # Empirical coverage is the honest counterweight to a confident-looking
        # band: it says how often the interval actually contained the outcome.
        "interval_coverage_check": coverage,
        "out_of_sample_metrics": {
            "directional_accuracy_pct": metrics.get("directional_accuracy"),
            "rmse": metrics.get("rmse"), "mae": metrics.get("mae"),
            "r2": metrics.get("r2"),
        },
        "trained_at": pred.get("trained_at"),
        "data_source": series.source,
    })


def tool_get_agent_decision(symbol: str, algo: str = "dueling_dqn") -> dict:
    from app.services.rl.service import rl_service

    try:
        decision = rl_service.recommend_action(symbol, algo=algo)
    except Exception as exc:
        from app.services.rl.catalogue import get_algorithm
        spec = get_algorithm(algo)
        return {
            "error": "agent_unavailable",
            "symbol": symbol.upper(), "algo": algo,
            "algorithm_name": spec.name if spec else algo,
            "message": (f"No trained {algo} agent for {symbol.upper()}: {exc}. "
                        "Train one on the RL Agent page."),
        }

    perf = decision.get("agent_test_performance") or {}
    explanation = decision.get("explanation") or {}
    return _round({
        "symbol": decision["symbol"],
        "algorithm": decision.get("algorithm_name"),
        "algorithm_family": decision.get("algorithm_family"),
        "action": decision["action"],
        "confidence": decision.get("confidence"),
        "last_price": decision.get("last_price"),
        "as_of": decision.get("as_of"),
        "q_values": decision.get("q_values"),
        "return_distribution": decision.get("return_distribution"),
        "risk": decision.get("risk"),
        "trade_plan": decision.get("trade_plan"),
        "investment_horizon": decision.get("investment_horizon"),
        "rationale": explanation.get("summary") or explanation,
        # The honest part: how the agent actually did out of sample, and how
        # that compares with simply holding the asset.
        "out_of_sample_performance": perf,
        "baselines": decision.get("baselines"),
        "trained_at": decision.get("trained_at"),
    })


def tool_get_risk_assessment(symbol: str, period: str = "2y") -> dict:
    from app.services.data.market_data import market_data_service
    from app.services.risk.anomaly import anomaly_detector

    series = market_data_service.get_history(symbol, period=period)
    scan = anomaly_detector.scan(symbol, series.df)
    crash = scan.get("crash_risk") or {}
    bubble = scan.get("bubble") or {}
    return _round({
        "symbol": symbol.upper(),
        "as_of": scan.get("as_of"),
        "overall_risk_level": scan.get("overall_risk_level"),
        "crash_risk": {
            "score": crash.get("crash_risk_score"), "level": crash.get("level"),
            "var_95_daily": crash.get("var_95_daily"),
            "cvar_95_daily": crash.get("cvar_95_daily"),
            "current_drawdown": crash.get("current_drawdown"),
            "volatility_regime": crash.get("volatility_regime"),
            "down_days_last_10": crash.get("down_days_last_10"),
            "recommendation": crash.get("recommendation"),
        },
        "bubble": {
            "score": bubble.get("bubble_score"), "level": bubble.get("level"),
            "trend_deviation_sigma": bubble.get("trend_deviation_sigma"),
            "momentum_3m": bubble.get("momentum_3m"),
            "momentum_12m": bubble.get("momentum_12m"),
            "interpretation": bubble.get("interpretation"),
        },
        "anomaly_count": scan.get("n_anomalies"),
        "anomaly_counts_by_type": scan.get("counts_by_type"),
        "recent_anomalies": (scan.get("anomalies") or [])[:5],
        "data_source": series.source,
    })


def tool_get_news_sentiment(symbol: str, limit: int = 12) -> dict:
    from app.services.nlp.news import news_service

    summary = news_service.sentiment_summary(symbol, limit=limit)
    if not summary.get("n"):
        return {"symbol": symbol.upper(), "articles_analysed": 0,
                "message": "No news articles were available for this instrument."}
    return _round({
        "symbol": symbol.upper(),
        "label": summary.get("label"),
        "score": summary.get("score"),
        "confidence": summary.get("confidence"),
        "articles_analysed": summary.get("n"),
        "bullish_ratio": summary.get("bullish_ratio"),
        "bearish_ratio": summary.get("bearish_ratio"),
        "distribution": summary.get("distribution"),
        "by_category": summary.get("by_category"),
        # "lexicon" vs "finbert" changes how much weight this deserves.
        "nlp_backend": summary.get("backend"),
        "top_headlines": [
            # `sentiment` is already a flat label string in this payload.
            {"title": n.get("title"), "sentiment": n.get("sentiment"),
             "impact": n.get("impact_score"), "source": n.get("source"),
             "category": n.get("category"), "published_at": n.get("published_at")}
            for n in (summary.get("top_impact_news") or [])[:5]
        ],
    })


def tool_compare_strategies(symbol: str, period: str = "2y") -> dict:
    from app.services.data.market_data import market_data_service
    from app.services.recommendation.intelligence import strategy_benchmarks

    series = market_data_service.get_history(symbol, period=period)
    comparison = strategy_benchmarks.compare_all(series.df["close"])
    # `strategies` is a list of rows, each carrying a full equity curve. Strip
    # the curves: hundreds of points would swamp the model's context.
    return _round({
        "symbol": symbol.upper(),
        "period": period,
        "strategies": [
            {k: v for k, v in row.items() if k != "equity_curve"}
            for row in (comparison.get("strategies") or [])
        ],
        "ranking_by_sharpe": comparison.get("ranking"),
        "best_by_sharpe": comparison.get("best_by_sharpe"),
        "verdict": comparison.get("verdict"),
        "cost_model": comparison.get("cost_model"),
    })


def tool_get_market_regime(symbol: str, period: str = "2y") -> dict:
    from app.services.data.market_data import market_data_service
    from app.services.forecasting.advanced import regime_detector

    series = market_data_service.get_history(symbol, period=period)
    return _round({"symbol": series.symbol, **regime_detector.detect(series.df)})


def tool_get_performance_metrics(symbol: str, period: str = "2y",
                                 benchmark: str = "SPY") -> dict:
    from app.services.recommendation.intelligence import portfolio_intelligence

    dossier = portfolio_intelligence.instrument_dossier(
        symbol, period=period, benchmark=benchmark)
    metrics = dossier.get("metrics") or {}
    return _round({
        "symbol": dossier.get("symbol", symbol.upper()),
        "name": dossier.get("name"),
        "period": period,
        "benchmark": benchmark,
        "last_price": dossier.get("last_price"),
        # Scalars only: the dossier also carries multi-hundred-point curves.
        "metrics": {k: v for k, v in metrics.items() if not isinstance(v, (list, dict))},
        "risk_exposure": dossier.get("risk_exposure"),
        "worst_drawdowns": (dossier.get("drawdown_episodes") or [])[:3],
        "data_source": dossier.get("data_source"),
    })


def tool_explain_prediction(symbol: str, top_k: int = 8) -> dict:
    from app.services.data.market_data import market_data_service
    from app.services.xai.explainer import explainer

    series = market_data_service.get_history(symbol, period="2y")
    explanation = explainer.shap_values(symbol, series.df, horizon=5, top_k=top_k)
    payload = explanation.to_dict()
    return _round({
        "symbol": symbol.upper(),
        "method": payload.get("method"),
        "prediction": payload.get("prediction"),
        "base_value": payload.get("base_value"),
        "top_drivers": [
            {"feature": c.get("label"), "value": c.get("value"),
             "contribution": c.get("contribution"), "direction": c.get("direction")}
            for c in (payload.get("feature_importance") or [])[:top_k]
        ],
        "narrative": payload.get("narrative"),
        "details": payload.get("details"),
    })


def tool_list_rl_algorithms() -> dict:
    from app.services.rl.catalogue import CATALOGUE

    return {
        "count": len(CATALOGUE),
        "algorithms": [
            {"key": a.key, "name": a.name, "family": a.family,
             "action_space": a.action_space, "available": a.available,
             "best_for": a.best_for}
            for a in CATALOGUE
        ],
        "note": ("Continuous-action algorithms (SAC, TD3, DDPG) output a target "
                 "exposure; on a single asset the platform maps that to BUY/HOLD/SELL."),
    }


def tool_list_trained_models() -> dict:
    from app.services.forecasting.trainer import forecast_trainer
    from app.services.rl.service import rl_service

    agents = rl_service.list_agents()
    checkpoints = []
    model_dir = forecast_trainer.model_dir
    for path in sorted(model_dir.glob("forecast_*.json")):
        parts = path.stem.replace("forecast_", "").rsplit("_", 2)
        if len(parts) == 3:
            checkpoints.append({"symbol": parts[0], "model": parts[1], "horizon": parts[2]})

    return {
        "forecast_checkpoints": checkpoints,
        "forecast_count": len(checkpoints),
        "rl_agents": [
            {"symbol": a.get("symbol"), "algo": a.get("algo"),
             "trained_at": a.get("trained_at"), "stale": a.get("stale", False)}
            for a in agents
        ],
        "rl_agent_count": len(agents),
        "note": ("Only these symbol/model pairs can be forecast or acted on without "
                 "training first. Anything else needs a training run."),
    }


def tool_search_instruments(query: str = "", asset_class: str = "") -> dict:
    from app.services.data.market_data import market_data_service

    results = market_data_service.search(query or None, asset_class or None)
    return {
        "count": len(results),
        "instruments": [
            {"symbol": i.get("symbol"), "name": i.get("name"),
             "asset_class": i.get("asset_class")}
            for i in results[:25]
        ],
    }


def tool_get_platform_status() -> dict:
    from app.core.config import settings
    from app.services.data.market_data import market_data_service
    from app.services.rl.agents.policy_gradient import SB3_AVAILABLE

    health = market_data_service.health()
    return {
        "version": settings.APP_VERSION,
        "data_mode": health["data_mode"],
        "live_providers": health["live_providers"],
        "universe_size": health["universe_size"],
        "sb3_available": SB3_AVAILABLE,
        "pages": ["Market Overview", "Technical Analysis", "AI Forecasting",
                  "RL Agent", "Recommendations", "Explainability", "Portfolio",
                  "Risk & Alerts"],
    }


# ============================================================ registry
# Each entry: the callable + the JSON schema advertised to the model.
REGISTRY: dict[str, tuple[Callable[..., dict], dict]] = {
    "get_quote": (tool_get_quote, {
        "description": "Latest price, daily change and volume for one instrument. "
                       "Use for any 'what is X trading at' question.",
        "parameters": {
            "type": "object",
            "properties": {"symbol": {"type": "string",
                                      "description": "Ticker, e.g. AAPL, BTC-USD, EURUSD=X"}},
            "required": ["symbol"],
        },
    }),
    "get_technical_analysis": (tool_get_technical_analysis, {
        "description": "Technical indicators (RSI, MACD, moving averages, Bollinger, "
                       "ATR, ADX, Stochastic, MFI) plus the rule-based buy/sell "
                       "consensus for one instrument.",
        "parameters": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string"},
                "period": {"type": "string", "enum": ["1mo", "3mo", "6mo", "1y", "2y", "5y"],
                           "description": "History window. Default 1y."},
            },
            "required": ["symbol"],
        },
    }),
    "get_recommendation": (tool_get_recommendation, {
        "description": "The platform's full multi-signal recommendation: fuses the "
                       "deep-learning forecast, the RL agent, technical indicators and "
                       "news sentiment into one action with confidence and position sizing.",
        "parameters": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string"},
                "forecast_model": {"type": "string",
                                   "enum": ["lstm", "gru", "tcn", "transformer", "cnn_lstm"]},
                "rl_algo": {"type": "string",
                            "description": "RL agent to consult, e.g. dueling_dqn, ppo, c51."},
                "horizon": {"type": "integer", "description": "Forecast horizon in days (1-60)."},
            },
            "required": ["symbol"],
        },
    }),
    "get_forecast": (tool_get_forecast, {
        "description": "Deep-learning price forecast with its conformal prediction "
                       "interval and out-of-sample accuracy. Returns model_not_trained "
                       "if no checkpoint exists yet.",
        "parameters": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string"},
                "model": {"type": "string",
                          "enum": ["lstm", "gru", "tcn", "transformer", "cnn_lstm"]},
                "horizon": {"type": "integer", "description": "Days ahead. Default 5."},
            },
            "required": ["symbol"],
        },
    }),
    "get_agent_decision": (tool_get_agent_decision, {
        "description": "Trading decision from a trained reinforcement-learning agent, "
                       "with its trade plan (stop-loss, take-profit, sizing) and its "
                       "real out-of-sample performance versus buy & hold.",
        "parameters": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string"},
                "algo": {"type": "string",
                         "description": "dqn, double_dqn, dueling_dqn, c51, qr_dqn, iqn, "
                                        "rainbow, ppo, a2c, trpo, sac, td3 or ddpg."},
            },
            "required": ["symbol"],
        },
    }),
    "get_risk_assessment": (tool_get_risk_assessment, {
        "description": "Risk scan: crash-risk score, bubble/overheating indicator and "
                       "recent statistical anomalies for one instrument.",
        "parameters": {
            "type": "object",
            "properties": {"symbol": {"type": "string"},
                           "period": {"type": "string"}},
            "required": ["symbol"],
        },
    }),
    "get_news_sentiment": (tool_get_news_sentiment, {
        "description": "Aggregated news sentiment with the highest-impact headlines "
                       "for one instrument.",
        "parameters": {
            "type": "object",
            "properties": {"symbol": {"type": "string"},
                           "limit": {"type": "integer", "description": "Articles, 1-50."}},
            "required": ["symbol"],
        },
    }),
    "compare_strategies": (tool_compare_strategies, {
        "description": "Backtest Buy & Hold, MA Crossover, Momentum and RSI mean-reversion "
                       "on the same instrument with identical costs, and rank them.",
        "parameters": {
            "type": "object",
            "properties": {"symbol": {"type": "string"}, "period": {"type": "string"}},
            "required": ["symbol"],
        },
    }),
    "get_market_regime": (tool_get_market_regime, {
        "description": "Current market regime (bull/bear/sideways, calm/volatile) with "
                       "trend, volatility ratio and drawdown context.",
        "parameters": {
            "type": "object",
            "properties": {"symbol": {"type": "string"}, "period": {"type": "string"}},
            "required": ["symbol"],
        },
    }),
    "get_performance_metrics": (tool_get_performance_metrics, {
        "description": "Risk/return dossier: Sharpe, Sortino, Calmar, volatility, max "
                       "drawdown, VaR, beta and alpha versus a benchmark.",
        "parameters": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string"}, "period": {"type": "string"},
                "benchmark": {"type": "string", "description": "Default SPY."},
            },
            "required": ["symbol"],
        },
    }),
    "explain_prediction": (tool_explain_prediction, {
        "description": "Explainable-AI attribution: which features drive the model's "
                       "prediction for this instrument, and by how much (SHAP).",
        "parameters": {
            "type": "object",
            "properties": {"symbol": {"type": "string"},
                           "top_k": {"type": "integer", "description": "Features to return."}},
            "required": ["symbol"],
        },
    }),
    "list_rl_algorithms": (tool_list_rl_algorithms, {
        "description": "The 13 reinforcement-learning algorithms available on this "
                       "platform, their families and what each is best suited to.",
        "parameters": {"type": "object", "properties": {}},
    }),
    "list_trained_models": (tool_list_trained_models, {
        "description": "Which forecast checkpoints and RL agents are actually trained "
                       "and ready. Call this before promising a forecast or a decision.",
        "parameters": {"type": "object", "properties": {}},
    }),
    "search_instruments": (tool_search_instruments, {
        "description": "Search the tradable universe by name or ticker, optionally "
                       "filtered by asset class.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "asset_class": {"type": "string",
                                "enum": ["equity", "crypto", "etf", "commodity",
                                         "forex", "index"]},
            },
        },
    }),
    "get_platform_status": (tool_get_platform_status, {
        "description": "Platform health: data mode, live providers, universe size and "
                       "which pages exist. Use for 'what can this app do' questions.",
        "parameters": {"type": "object", "properties": {}},
    }),
}


def openai_tool_schemas() -> list[dict]:
    """Registry rendered as the OpenAI-style function-calling schema."""
    return [
        {"type": "function", "function": {"name": name, **schema}}
        for name, (_, schema) in REGISTRY.items()
    ]


async def execute(name: str, arguments: dict) -> dict:
    """Run one tool off the event loop, with a timeout and no raised exception.

    Failures come back as ``{"error": ...}`` on purpose: the model can then tell
    the user *what* went wrong and suggest the page that fixes it, which beats a
    500 that kills the whole conversation.
    """
    entry = REGISTRY.get(name)
    if entry is None:
        return {"error": "unknown_tool", "message": f"No tool named '{name}'."}

    func, schema = entry
    allowed = set((schema.get("parameters") or {}).get("properties", {}))
    clean = {k: v for k, v in (arguments or {}).items() if k in allowed and v not in (None, "")}

    missing = [p for p in (schema.get("parameters") or {}).get("required", [])
               if p not in clean]
    if missing:
        return {"error": "missing_arguments", "message": f"{name} needs: {', '.join(missing)}"}

    try:
        logger.info("chat tool %s(%s)", name, clean)
        return await asyncio.wait_for(asyncio.to_thread(func, **clean), timeout=TOOL_TIMEOUT)
    except TimeoutError:
        return {"error": "timeout",
                "message": f"{name} took longer than {TOOL_TIMEOUT:.0f}s and was cancelled."}
    except TypeError as exc:
        return {"error": "bad_arguments", "message": str(exc)[:200]}
    except Exception as exc:
        logger.warning("chat tool %s failed: %s", name, exc)
        return {"error": type(exc).__name__, "message": str(exc)[:300]}
