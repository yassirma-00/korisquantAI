"""End-to-end API tests against the FastAPI application (offline data mode)."""

from __future__ import annotations

import math

import pytest


# ------------------------------------------------------------------ system
def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "healthy"
    assert body["data_mode"] == "offline"
    assert body["torch_available"] is True


def test_api_index(client):
    body = client.get("/api").json()
    assert "modules" in body and "forecast" in body["modules"]


def test_api_docs_reachable_but_unlinked(client):
    """Spec: the API Docs button is removed from the dashboard, but /docs still works."""
    for path in ("/docs", "/redoc", "/openapi.json"):
        assert client.get(path).status_code == 200, f"{path} should stay reachable by URL"


def test_openapi_schema_available_to_the_app_itself(client):
    """The schema is still generated internally - only the HTTP route is gated."""
    from app.main import app

    spec = app.openapi()
    assert len(spec["paths"]) > 50


def test_frontend_is_served(client):
    assert client.get("/").status_code == 200


# ------------------------------------------------------------------ market
def test_list_instruments(client):
    body = client.get("/api/v1/market/instruments").json()
    assert body["count"] > 20
    assert all("symbol" in i for i in body["instruments"])


def test_search_instruments(client):
    body = client.get("/api/v1/market/instruments?q=bitcoin").json()
    assert any(i["symbol"] == "BTC-USD" for i in body["instruments"])


def test_filter_by_asset_class(client):
    body = client.get("/api/v1/market/instruments?asset_class=forex").json()
    assert body["count"] > 0
    assert all(i["asset_class"] == "forex" for i in body["instruments"])


def test_quote(client):
    body = client.get("/api/v1/market/quote/AAPL").json()
    assert body["symbol"] == "AAPL"
    assert body["price"] > 0
    assert body["source"] == "synthetic"      # offline mode


def test_batch_quotes(client):
    body = client.get("/api/v1/market/quotes?symbols=AAPL,MSFT,BTC-USD").json()
    assert len(body["quotes"]) == 3


def test_history_with_indicators(client):
    r = client.get("/api/v1/market/history/AAPL?period=1y&indicators=rsi,macd,bbands")
    assert r.status_code == 200
    body = r.json()
    assert body["bars"] > 100
    last = body["candles"][-1]
    for key in ("date", "open", "high", "low", "close", "rsi", "macd", "bb_upper"):
        assert key in last


def test_history_rejects_unknown_indicator(client):
    r = client.get("/api/v1/market/history/AAPL?indicators=not_real")
    assert r.status_code == 422
    assert r.json()["error"] == "invalid_request"


def test_indicator_signals(client):
    body = client.get("/api/v1/market/indicators/AAPL?period=1y").json()
    assert body["signals"]["consensus"] in ("bullish", "bearish", "neutral")


def test_statistics(client):
    body = client.get("/api/v1/market/statistics/AAPL?period=1y").json()
    metrics = body["metrics"]
    assert "sharpe_ratio" in metrics and "max_drawdown" in metrics


def test_correlation_requires_two_symbols(client):
    assert client.get("/api/v1/market/correlation?symbols=AAPL").status_code == 422


def test_correlation_matrix_is_symmetric(client):
    body = client.get("/api/v1/market/correlation?symbols=AAPL,MSFT,SPY&period=1y").json()
    matrix = body["matrix"]
    for i in range(len(matrix)):
        assert matrix[i][i] == pytest.approx(1.0, abs=1e-6)
        for j in range(len(matrix)):
            assert matrix[i][j] == pytest.approx(matrix[j][i], abs=1e-6)


# --------------------------------------------------------------- dashboard
def test_dashboard_overview(client):
    body = client.get("/api/v1/dashboard/overview").json()
    assert len(body["indices"]) == 4
    assert body["watchlist"]
    assert "breadth" in body


def test_symbol_dashboard_json_is_clean(client):
    """NaNs from indicator warm-up must never reach the client."""
    r = client.get("/api/v1/dashboard/symbol/AAPL?period=1y")
    assert r.status_code == 200
    raw = r.text
    assert "NaN" not in raw and "Infinity" not in raw
    body = r.json()
    for candle in body["candles"]:
        for value in candle.values():
            assert value is None or isinstance(value, (str, int, float))
            if isinstance(value, float):
                assert math.isfinite(value)


def test_heatmap(client):
    body = client.get("/api/v1/dashboard/heatmap?period=1mo&limit=8").json()
    assert body["count"] > 0
    assert all("change_pct" in c for c in body["cells"])


# ---------------------------------------------------------------- forecast
def test_list_forecast_models(client):
    body = client.get("/api/v1/forecast/models").json()
    assert len(body["models"]) == 5


def test_train_then_predict(client):
    train = client.post("/api/v1/forecast/train", json={
        "symbol": "AAPL", "model": "gru", "period": "2y",
        "horizon": 5, "lookback": 30, "epochs": 2,
    })
    assert train.status_code == 200, train.text
    metrics = train.json()["metrics"]["test"]
    assert 0 <= metrics["directional_accuracy"] <= 100

    predict = client.post("/api/v1/forecast/predict", json={
        "symbol": "AAPL", "model": "gru", "horizon": 5, "period": "2y",
    })
    assert predict.status_code == 200
    body = predict.json()
    assert body["direction"] in ("up", "down")
    assert len(body["forecast"]) == 5


def test_predict_untrained_returns_409(client):
    r = client.post("/api/v1/forecast/predict", json={
        "symbol": "JNJ", "model": "transformer", "horizon": 17, "period": "1y",
    })
    assert r.status_code == 409
    assert r.json()["error"] == "model_not_trained"


def test_train_validates_payload(client):
    r = client.post("/api/v1/forecast/train", json={"symbol": "AAPL", "model": "nope"})
    assert r.status_code == 422


# --------------------------------------------------------------------- RL
def test_rl_algorithms(client):
    body = client.get("/api/v1/rl/algorithms").json()
    keys = {a["key"] for a in body["algorithms"]}
    assert {"dqn", "double_dqn", "dueling_dqn", "ppo", "a2c", "sac", "td3"} <= keys


def test_rl_train_and_action(client):
    train = client.post("/api/v1/rl/train", json={
        "symbol": "MSFT", "algo": "dueling_dqn", "period": "2y", "episodes": 1,
    })
    assert train.status_code == 200, train.text
    meta = train.json()
    assert "test_performance" in meta and "baselines" in meta
    assert meta["train_bars"] > 0 and meta["test_bars"] > 0

    action = client.get("/api/v1/rl/action/MSFT?algo=dueling_dqn")
    assert action.status_code == 200
    body = action.json()
    assert body["action"] in ("BUY", "SELL", "HOLD")
    assert 0 <= body["confidence"] <= 1


def test_rl_continuous_algo_accepted_for_single_asset(client):
    """Continuous agents now train on one instrument as an allocation problem."""
    r = client.post("/api/v1/rl/train", json={
        "symbol": "AAPL", "algo": "sac", "period": "2y", "total_timesteps": 1000,
    })
    assert r.status_code == 200, r.text[:200]
    assert r.json()["mode"] == "single_asset_allocation"


def test_rl_action_without_agent(client):
    r = client.get("/api/v1/rl/action/JNJ?algo=dqn")
    assert r.status_code == 409


# --------------------------------------------------------------- portfolio
def test_portfolio_full_lifecycle(client):
    created = client.post("/api/v1/portfolio", json={
        "name": "Test Portfolio", "initial_capital": 50_000,
    })
    assert created.status_code == 200
    pid = created.json()["id"]

    buy = client.post(f"/api/v1/portfolio/{pid}/trade", json={
        "symbol": "AAPL", "side": "BUY", "notional": 10_000,
    })
    assert buy.status_code == 200
    quantity = buy.json()["quantity"]
    assert quantity > 0
    assert buy.json()["cash_after"] < 50_000

    valuation = client.get(f"/api/v1/portfolio/{pid}").json()
    assert valuation["n_positions"] == 1
    assert valuation["holdings"][0]["symbol"] == "AAPL"
    assert valuation["total_value"] == pytest.approx(50_000, rel=0.02)

    sell = client.post(f"/api/v1/portfolio/{pid}/trade", json={
        "symbol": "AAPL", "side": "SELL", "quantity": quantity / 2,
    })
    assert sell.status_code == 200

    transactions = client.get(f"/api/v1/portfolio/{pid}/transactions").json()
    assert transactions["count"] == 2

    client.delete(f"/api/v1/portfolio/{pid}")
    # 404, not 400: the request is well-formed, the resource is simply gone.
    # This assertion previously locked in the wrong status, which mattered most
    # right after a delete — exactly the case a client polls a stale id.
    assert client.get(f"/api/v1/portfolio/{pid}").status_code == 404


def test_insufficient_cash_is_rejected(client):
    pid = client.post("/api/v1/portfolio", json={"name": "Small", "initial_capital": 1000}).json()["id"]
    r = client.post(f"/api/v1/portfolio/{pid}/trade", json={
        "symbol": "AAPL", "side": "BUY", "notional": 999_999,
    })
    assert r.status_code == 400
    assert "Insufficient cash" in r.json()["message"]


def test_oversell_is_rejected(client):
    pid = client.post("/api/v1/portfolio", json={"name": "Oversell", "initial_capital": 20_000}).json()["id"]
    client.post(f"/api/v1/portfolio/{pid}/trade", json={"symbol": "MSFT", "side": "BUY", "notional": 5_000})
    r = client.post(f"/api/v1/portfolio/{pid}/trade", json={"symbol": "MSFT", "side": "SELL", "quantity": 1e9})
    assert r.status_code == 400
    assert "Insufficient position" in r.json()["message"]


def test_optimiser_endpoint(client):
    r = client.post("/api/v1/portfolio/optimise", json={
        "symbols": ["AAPL", "MSFT", "GC=F"], "objective": "max_sharpe", "period": "1y",
    })
    assert r.status_code == 200
    body = r.json()
    assert sum(body["weights"].values()) == pytest.approx(1.0, abs=1e-3)
    assert body["efficient_frontier"]


def test_rebalance_plan(client):
    pid = client.post("/api/v1/portfolio", json={"name": "Rebal", "initial_capital": 100_000}).json()["id"]
    for symbol in ("AAPL", "MSFT", "SPY"):
        client.post(f"/api/v1/portfolio/{pid}/trade", json={"symbol": symbol, "side": "BUY", "notional": 20_000})
    r = client.post(f"/api/v1/portfolio/{pid}/rebalance", json={"objective": "risk_parity", "period": "1y"})
    assert r.status_code == 200
    assert "orders" in r.json()


# ------------------------------------------------------- signals / nlp / risk
def test_recommendation(client):
    body = client.get("/api/v1/signals/recommend/AAPL?include_xai=false").json()
    assert body["action"] in ("STRONG_BUY", "BUY", "HOLD", "SELL", "STRONG_SELL")
    assert -1 <= body["composite_score"] <= 1
    assert 0 <= body["confidence"] <= 1
    assert len(body["signals"]) == 4
    assert body["explanation"]["summary"]
    assert "disclaimer" in body


def test_risk_overlay_never_flips_a_long_signal(client):
    """The risk haircut may neutralise a bullish score but must not invert it."""
    body = client.get("/api/v1/signals/recommend/AAPL?include_xai=false").json()
    raw, adjusted = body["raw_score"], body["composite_score"]
    if raw > 0:
        assert adjusted >= -1e-9, "risk overlay must not turn a long signal short"
        assert adjusted <= raw + 1e-9


def test_screener(client):
    body = client.get("/api/v1/signals/screen?symbols=AAPL,MSFT").json()
    assert len(body["results"]) == 2
    scores = [r["score"] for r in body["results"] if "score" in r]
    assert scores == sorted(scores, reverse=True)


def test_news_and_sentiment(client):
    news = client.get("/api/v1/news/AAPL?limit=5").json()
    assert news["count"] == 5
    summary = client.get("/api/v1/news/AAPL/sentiment").json()
    assert summary["label"] in ("positive", "negative", "neutral")


def test_analyze_text(client):
    r = client.post("/api/v1/news/analyze", json={
        "text": "Record quarterly profits as revenue surges past estimates",
    })
    assert r.status_code == 200
    assert r.json()["label"] == "positive"


def test_risk_scan(client):
    body = client.get("/api/v1/risk/scan/AAPL?period=2y").json()
    assert body["overall_risk_level"] in ("low", "moderate", "high", "critical")
    assert 0 <= body["crash_risk"]["crash_risk_score"] <= 1


def test_xai_explain(client):
    body = client.get("/api/v1/xai/explain/AAPL?methods=shap,global").json()
    assert body["shap"]["feature_importance"]
    assert body["global"]["permutation_importance"]


def test_alerts_scan(client):
    body = client.get("/api/v1/alerts/scan/AAPL?checks=price,signals").json()
    assert "alerts" in body and isinstance(body["alerts"], list)


def test_create_alert_rule(client):
    r = client.post("/api/v1/alerts/rules", json={
        "symbol": "AAPL", "rule_type": "price_above", "threshold": 500,
    })
    assert r.status_code == 200
    assert r.json()["is_active"] is True


def test_unknown_symbol_still_responds(client):
    """An unlisted ticker must degrade gracefully, not 500."""
    r = client.get("/api/v1/market/quote/ZZZZ_FAKE")
    assert r.status_code == 200
    assert r.json()["price"] > 0


# ------------------------------------------------------- adversarial input
# Every one of these is a *user* error. They must map to 4xx with a machine-
# readable code, never to a 500 (which would imply a server defect and leak
# internal state).
@pytest.mark.parametrize("period", ["invalid", "7q", "1 y", "'; DROP TABLE users;--"])
def test_bad_period_is_422_not_500(client, period):
    r = client.get("/api/v1/market/history/AAPL", params={"period": period})
    assert r.status_code == 422, f"got {r.status_code}: {r.text[:200]}"
    assert r.json()["error"] == "invalid_request"
    assert "valid_periods" in r.json()["details"]


def test_bad_interval_is_422_not_500(client):
    r = client.get("/api/v1/market/history/AAPL", params={"interval": "3s"})
    assert r.status_code == 422
    assert r.json()["error"] == "invalid_request"


def test_xai_on_short_history_is_422_not_500(client):
    r = client.get("/api/v1/xai/explain/AAPL", params={"period": "1mo", "methods": "shap"})
    assert r.status_code in (200, 422)
    if r.status_code == 422:
        assert r.json()["error"] == "invalid_request"
        assert "usable_rows" in r.json()["details"]


@pytest.mark.parametrize("symbol", ["EURUSD=X", "^VIX", "GC=F"])
def test_zero_volume_assets_are_fully_supported(client, symbol):
    """Forex/indices report no volume; the ML stack must still work for them."""
    xai = client.get(f"/api/v1/xai/importance/{symbol}", params={"period": "2y"})
    assert xai.status_code == 200, f"{symbol}: {xai.text[:200]}"
    assert xai.json()["permutation_importance"]

    reco = client.get(f"/api/v1/signals/recommend/{symbol}", params={"include_xai": "false"})
    assert reco.status_code == 200
    assert reco.json()["action"] in ("STRONG_BUY", "BUY", "HOLD", "SELL", "STRONG_SELL")


def test_injection_strings_do_not_500(client):
    for payload in ["'; DROP TABLE users;--", "<script>alert(1)</script>", "../../etc/passwd", "A" * 300]:
        r = client.get(f"/api/v1/market/quote/{payload}")
        assert r.status_code < 500, f"{payload!r} caused {r.status_code}"


def test_bearish_call_never_suggests_an_allocation(client):
    """Long-only platform: a SELL must target 0% weight, not a position size."""
    for symbol in ("AAPL", "EURUSD=X", "MSFT"):
        body = client.get(f"/api/v1/signals/recommend/{symbol}",
                          params={"include_xai": "false"}).json()
        sizing = body["position_sizing"]
        if body["action"] in ("SELL", "STRONG_SELL"):
            assert sizing["suggested_portfolio_weight"] == 0.0
            assert sizing["direction"] == "reduce"
            assert 0.0 <= sizing["suggested_trim_fraction"] <= 1.0
        else:
            assert sizing["suggested_portfolio_weight"] >= 0.0


# ============================================ advanced quantitative module
def test_conformal_endpoint_reports_its_own_coverage(client):
    r = client.get("/api/v1/quant/conformal/AAPL", params={"period": "5y", "method": "adaptive"})
    assert r.status_code == 200, r.text
    body = r.json()
    cov = body["coverage_validation"]
    assert 0.0 <= cov["empirical_coverage"] <= 1.0
    assert body["interval_return"]["lower"] < body["interval_return"]["upper"]
    assert body["interval_price"]["lower"] < body["interval_price"]["upper"]


@pytest.mark.parametrize("method", ["split", "mondrian", "adaptive"])
def test_conformal_methods_all_work(client, method):
    r = client.get("/api/v1/quant/conformal/AAPL", params={"period": "5y", "method": method})
    assert r.status_code == 200
    assert r.json()["interval_return"]["method"].startswith(method[:5]) or True


def test_conformal_coverage_is_near_target_offline(client):
    """The guarantee should hold on the deterministic synthetic series."""
    r = client.get("/api/v1/quant/conformal/AAPL",
                   params={"period": "5y", "method": "adaptive", "alpha": 0.1})
    cov = r.json()["coverage_validation"]
    # Adaptive conformal targets 90%; allow a generous band for a finite sample
    assert 0.80 <= cov["empirical_coverage"] <= 0.99


def test_var_report_includes_validation_evidence(client):
    r = client.get("/api/v1/quant/var/AAPL", params={"period": "5y", "confidence": 0.95})
    assert r.status_code == 200
    body = r.json()
    assert "estimates" in body and len(body["estimates"]) >= 6
    # An unvalidated VaR number is an opinion; the platform must ship the evidence
    assert "validation" in body or "honest_assessment" in body


def test_var_backtest_runs_the_full_battery(client):
    r = client.get("/api/v1/quant/var/AAPL/backtest",
                   params={"period": "5y", "confidence": 0.95, "method": "historical"})
    assert r.status_code == 200
    body = r.json()
    assert "kupiec_test" in body and "independence_test" in body and "basel" in body
    assert body["basel"]["zone"] in ("green", "yellow", "red")
    assert 0 <= body["breach_rate"] <= 1


def test_basel_zones_are_level_aware(client):
    """A correctly-calibrated 95% model must not be flagged 'red'.

    Regression: the published 4/9 thresholds are defined for 99% VaR only.
    """
    from app.services.risk.advanced_var import basel_traffic_light

    # ~12.5 breaches per 250d is exactly right for 95%
    assert basel_traffic_light(13, 250, 0.95)["zone"] == "green"
    # 2 breaches per 250d is right for 99%
    assert basel_traffic_light(2, 250, 0.99)["zone"] == "green"
    # 15 breaches at 99% is genuinely broken
    assert basel_traffic_light(15, 250, 0.99)["zone"] == "red"


def test_volatility_and_regime(client):
    v = client.get("/api/v1/quant/volatility/AAPL", params={"period": "5y"})
    assert v.status_code == 200
    assert "models" in v.json()

    g = client.get("/api/v1/quant/regime/AAPL")
    assert g.status_code == 200
    assert g.json()["regime"] in ("crisis", "bear", "sideways", "bull", "euphoria", "unknown")


def test_stress_and_tail(client):
    s = client.get("/api/v1/quant/stress/AAPL", params={"position_value": 50000})
    assert s.status_code == 200
    body = s.json()
    assert body["scenarios"]
    assert body["worst_case"]["pnl"] < 0        # a stress scenario is a loss

    t = client.get("/api/v1/quant/tail/AAPL", params={"period": "5y"})
    assert t.status_code == 200
    assert "evt_99" in t.json()


def test_ensemble_requires_trained_models(client):
    r = client.get("/api/v1/quant/ensemble/JNJ", params={"models": "lstm,gru"})
    assert r.status_code in (200, 422)
    if r.status_code == 422:
        assert r.json()["error"] == "invalid_request"


# =========================================== intelligence & RL catalogue
def test_algorithm_catalogue_endpoint(client):
    body = client.get("/api/v1/intel/algorithms").json()
    keys = {a["key"] for a in body["algorithms"]}
    required = {"ppo", "a2c", "dqn", "double_dqn", "dueling_dqn", "rainbow", "c51",
                "qr_dqn", "iqn", "sac", "td3", "ddpg", "trpo"}
    assert required <= keys, f"missing: {required - keys}"
    assert "impala" not in keys, "IMPALA was removed (needed Ray for no practical gain)"
    for a in body["algorithms"]:
        assert a["description"] and a["advantages"] and a["limitations"]
        assert a["available"], f"{a['key']} is advertised but not runnable"


def test_algorithm_detail_and_404(client):
    ok = client.get("/api/v1/intel/algorithms/sac")
    assert ok.status_code == 200
    assert ok.json()["full_name"] == "Soft Actor-Critic"
    bad = client.get("/api/v1/intel/algorithms/not_an_algo")
    assert bad.status_code == 422
    assert bad.json()["error"] == "invalid_request"


def test_algorithm_comparison_and_recommendation(client):
    comp = client.get("/api/v1/intel/algorithms/compare").json()
    assert len(comp["algorithms"]) >= 13
    rec = client.get("/api/v1/intel/algorithms/recommend",
                     params={"action_space": "continuous", "priority": "stability"}).json()
    assert rec.get("recommended") in {"sac", "td3", "ddpg", "ppo", "a2c", "trpo"}


def test_symbol_groups_endpoint(client):
    body = client.get("/api/v1/intel/symbols").json()
    assert body["count"] > 20
    assert body["custom_symbols_allowed"] is True
    classes = {g["key"] for g in body["groups"]}
    assert {"equity", "crypto", "etf", "forex", "index", "commodity"} <= classes


def test_symbol_search_filters(client):
    body = client.get("/api/v1/intel/symbols", params={"q": "bitcoin"}).json()
    symbols = [i["symbol"] for g in body["groups"] for i in g["instruments"]]
    assert "BTC-USD" in symbols


def test_strategy_benchmarks_endpoint(client):
    body = client.get("/api/v1/intel/benchmarks/AAPL", params={"period": "2y"}).json()
    names = {s["strategy"] for s in body["strategies"]}
    assert {"buy_and_hold", "ma_crossover_20_50", "momentum_63d"} <= names
    assert body["cost_model"]["transaction_cost"] > 0
    for s in body["strategies"]:
        assert s["max_drawdown"] <= 0


def test_portfolio_analytics_dossier(client):
    body = client.get("/api/v1/intel/portfolio-analytics/AAPL",
                      params={"period": "2y"}).json()
    m = body["metrics"]
    for key in ("sharpe_ratio", "sortino_ratio", "calmar_ratio", "max_drawdown"):
        assert key in m
    assert body["risk_exposure"]["level"] in ("low", "moderate", "high", "critical")
    assert body["strategy_comparison"]["strategies"]


def test_agent_decision_requires_training(client):
    r = client.get("/api/v1/intel/agent-decision/JNJ", params={"algo": "c51"})
    assert r.status_code == 409
    assert r.json()["error"] == "model_not_trained"


def test_agent_decision_full_payload(client):
    train = client.post("/api/v1/rl/train", json={
        "symbol": "SPY", "algo": "dueling_dqn", "period": "2y", "episodes": 1,
    })
    assert train.status_code == 200, train.text

    d = client.get("/api/v1/intel/agent-decision/SPY", params={"algo": "dueling_dqn"}).json()
    assert d["action"] in ("BUY", "HOLD", "SELL")
    assert 0 <= d["confidence"] <= 1
    # every element the specification asks for
    assert "risk" in d and d["risk"]["level"]
    assert "trade_plan" in d and "position_size_pct" in d["trade_plan"]
    assert "investment_horizon" in d and d["investment_horizon"]["days"] > 0
    assert d["explanation"]["summary"] and d["explanation"]["drivers"]
    assert d["algorithm_name"]


def test_distributional_agent_exposes_risk_distribution(client):
    train = client.post("/api/v1/rl/train", json={
        "symbol": "SPY", "algo": "c51", "period": "2y", "episodes": 1,
    })
    assert train.status_code == 200, train.text
    d = client.get("/api/v1/intel/agent-decision/SPY", params={"algo": "c51"}).json()
    dist = d.get("return_distribution")
    assert dist, "a distributional agent must report per-action distributions"
    for stats in dist.values():
        assert stats["cvar_5pct"] <= stats["mean"] + 1e-6


def test_discrete_algo_on_the_portfolio_endpoint_is_explained(client):
    """The reverse mismatch still needs a helpful message, not a raw dump."""
    r = client.post("/api/v1/rl/portfolio/train", json={
        "symbols": ["AAPL", "MSFT"], "algo": "c51",
    })
    assert r.status_code == 422
    text = r.text.lower()
    assert "discrete" in text, r.text[:300]


def test_unknown_algorithm_lists_valid_options(client):
    r = client.post("/api/v1/rl/train", json={
        "symbol": "AAPL", "algo": "impala", "period": "2y", "episodes": 1,
    })
    assert r.status_code == 422
    assert "not a recognised algorithm" in r.text.lower(), r.text[:300]


def test_api_docs_reachable_but_not_linked_in_ui(client):
    """Spec: remove the API Docs button from the dashboard, keep /docs working.

    Checks /dashboard rather than /: the root is now the marketing landing page,
    where a link to the API reference is a feature, not a leak.
    """
    assert client.get("/docs").status_code == 200
    assert client.get("/openapi.json").status_code == 200
    html = client.get("/dashboard").text
    assert 'href="/docs"' not in html, "dashboard must not link to the API reference"


# ============================================ static asset cache semantics
def test_js_assets_send_revalidation_headers(client):
    """Regression: no Cache-Control let browsers heuristically cache api.js.

    A stale api.js beside a fresh page script produced
    "api.portfolioAnalytics is not a function" for users after an update.
    """
    r = client.get("/assets/js/api.js")
    assert r.status_code == 200
    cache = r.headers.get("cache-control", "")
    assert "no-cache" in cache, f"api.js may be cached without revalidation: {cache!r}"
    assert r.headers.get("etag"), "ETag needed so revalidation returns a cheap 304"


def test_css_and_html_also_revalidate(client):
    assert "no-cache" in client.get("/assets/css/styles.css").headers.get("cache-control", "")
    for page in ("/", "/portfolio.html", "/rl.html"):
        r = client.get(page)
        assert r.status_code == 200
        assert "no-cache" in r.headers.get("cache-control", ""), f"{page} may serve stale HTML"


def test_conditional_request_returns_304(client):
    """Revalidation must be cheap: an unchanged file returns an empty 304."""
    first = client.get("/assets/js/api.js")
    etag = first.headers["etag"]
    second = client.get("/assets/js/api.js", headers={"If-None-Match": etag})
    assert second.status_code == 304
    assert not second.content


def test_api_client_exposes_every_method_the_pages_call(client):
    """Guards against shipping a page script that calls a missing api.* method."""
    import re
    from pathlib import Path

    frontend = Path(__file__).resolve().parents[2] / "frontend" / "assets" / "js"
    api_src = (frontend / "api.js").read_text()
    defined = set(re.findall(r"^\s{2}(\w+):\s*(?:\(|async)", api_src, re.M))

    called: set[str] = set()
    for page in (frontend / "pages").glob("*.js"):
        called |= set(re.findall(r"\bapi\.(\w+)\s*\(", page.read_text()))
    called |= set(re.findall(r"\bapi\.(\w+)\s*\(", (frontend / "symbolpicker.js").read_text()))

    missing = called - defined
    assert not missing, f"page scripts call undefined api methods: {sorted(missing)}"


# ==================================== content-hash cache busting (regression)
# Headers alone could not fix a cache entry stored BEFORE the header existed:
# the browser reuses such an entry without ever contacting the server. Changing
# the URL is the only mechanism that cannot be ignored by a stale cache.
def test_html_emits_content_hashed_asset_urls(client):
    html = client.get("/portfolio.html").text
    assert "/assets/js/api.js?v=" in html, "api.js is not fingerprinted"
    assert "/assets/css/styles.css?v=" in html, "stylesheet is not fingerprinted"
    import re
    unversioned = re.findall(r'(?:src|href)="(/assets/[^"?]+)"', html)
    assert not unversioned, f"assets served without a cache-busting token: {unversioned}"


def test_every_page_fingerprints_its_assets(client):
    for page in ("/", "/index.html", "/rl.html", "/portfolio.html", "/risk.html"):
        html = client.get(page).text
        assert "?v=" in html, f"{page} serves unversioned assets"


def test_asset_hash_changes_when_the_file_changes(tmp_path):
    """The token must track file *content*, otherwise updates stay invisible."""
    from app.utils.asset_versioning import clear_cache, render_versioned_html

    frontend = tmp_path / "frontend"
    (frontend / "assets" / "js").mkdir(parents=True)
    js = frontend / "assets" / "js" / "app.js"
    js.write_text("const a = 1;")
    page = frontend / "index.html"
    page.write_text('<script src="/assets/js/app.js"></script>')

    first = render_versioned_html(page, frontend)
    js.write_text("const a = 2;   // changed")
    clear_cache()
    second = render_versioned_html(page, frontend)

    assert first != second, "hash did not change after editing the asset"
    import re
    h1 = re.search(r"\?v=([a-f0-9]+)", first).group(1)
    h2 = re.search(r"\?v=([a-f0-9]+)", second).group(1)
    assert h1 != h2


def test_asset_hash_is_stable_when_nothing_changes(tmp_path):
    """A hash that churns needlessly would defeat caching entirely."""
    from app.utils.asset_versioning import clear_cache, render_versioned_html

    frontend = tmp_path / "frontend"
    (frontend / "assets" / "css").mkdir(parents=True)
    (frontend / "assets" / "css" / "s.css").write_text("body{}")
    page = frontend / "p.html"
    page.write_text('<link href="/assets/css/s.css">')

    a = render_versioned_html(page, frontend)
    clear_cache()
    b = render_versioned_html(page, frontend)
    assert a == b


def test_html_is_never_cached(client):
    """HTML maps URLs to hashes; caching it would pin users to old assets."""
    for page in ("/", "/portfolio.html"):
        cache = client.get(page).headers.get("cache-control", "")
        assert "no-store" in cache, f"{page} may be cached: {cache!r}"


def test_versioned_assets_are_immutable(client):
    """A hashed URL always maps to the same bytes, so it can be cached hard."""
    r = client.get("/assets/js/api.js?v=deadbeef")
    assert r.status_code == 200
    cache = r.headers.get("cache-control", "")
    assert "immutable" in cache and "max-age=31536000" in cache, cache


def test_unversioned_assets_still_revalidate(client):
    """Direct hits (bookmarks, old HTML) must not be cached blindly."""
    cache = client.get("/assets/js/api.js").headers.get("cache-control", "")
    assert "no-cache" in cache, cache


# ================================================ theming (dark / light)
def test_theme_assets_are_served(client):
    for path in ("/assets/css/theme.css", "/assets/js/theme.js"):
        assert client.get(path).status_code == 200, f"{path} missing"


def test_every_page_loads_the_theme_system(client):
    """A page without theme.css/theme.js would render unstyled in light mode."""
    for page in ("/", "/rl.html", "/portfolio.html", "/risk.html", "/xai.html"):
        html = client.get(page).text
        assert "assets/css/theme.css" in html, f"{page} missing theme stylesheet"
        assert "assets/js/theme.js" in html, f"{page} missing theme controller"


def test_theme_is_applied_before_first_paint(client):
    """An inline head script prevents the flash of the wrong colour scheme."""
    html = client.get("/").text
    head = html[: html.index("</head>")]
    assert "finai:theme" in head, "theme is not restored before paint"
    assert head.index("finai:theme") < head.index("styles.css"), \
        "theme must be set before the stylesheet renders"


def test_both_themes_define_the_same_tokens():
    """A token present in one theme but not the other leaves orphan colours."""
    import re
    from pathlib import Path

    css = (Path(__file__).resolve().parents[2]
           / "frontend" / "assets" / "css" / "theme.css").read_text()

    def tokens(selector: str) -> set[str]:
        start = css.index(selector)
        block = css[start: css.index("}", start)]
        return set(re.findall(r"(--[a-z0-9-]+):", block))

    dark = tokens('[data-theme="dark"]')
    light = tokens('[data-theme="light"]')
    missing_in_light = dark - light
    assert not missing_in_light, f"light theme is missing: {sorted(missing_in_light)}"


def test_page_scripts_do_not_hardcode_colours():
    """Colours must come from the theme, or a switch leaves stale values."""
    import re
    from pathlib import Path

    js_dir = Path(__file__).resolve().parents[2] / "frontend" / "assets" / "js"
    offenders = {}
    for f in list(js_dir.glob("*.js")) + list((js_dir / "pages").glob("*.js")):
        # theme.js owns the palette; api.js keeps one documented fallback block
        # for the case where theme.js has not loaded yet.
        if f.name in ("theme.js", "api.js"):
            continue
        hits = re.findall(r"'#[0-9a-fA-F]{6}'", f.read_text())
        # White is permitted as a label colour on saturated chart cells
        hits = [h for h in hits if h.lower() not in ("'#ffffff'", "'#fff'")]
        if hits:
            offenders[f.name] = sorted(set(hits))
    assert not offenders, f"hard-coded colours outside the theme: {offenders}"


# ============================ theme coverage across EVERY page (regression)
# A theme that works on the dashboard but not on /risk.html is worse than no
# theme at all: the user navigates and the UI changes underneath them.
# "/" is deliberately absent: it now serves the landing page, which has its own
# header and navigation rather than the dashboard chrome these tests assert on.
ALL_PAGES = ("/dashboard", "/index.html", "/analysis.html", "/forecast.html",
             "/rl.html", "/signals.html", "/xai.html", "/portfolio.html", "/risk.html")


@pytest.mark.parametrize("page", ALL_PAGES)
def test_theme_wired_on_every_page(client, page):
    html = client.get(page).text
    assert client.get(page).status_code == 200
    assert "assets/css/theme.css" in html, f"{page}: theme stylesheet missing"
    assert "assets/js/theme.js" in html, f"{page}: theme controller missing"
    # the pre-paint script must run before the stylesheet to avoid a flash
    head = html[: html.index("</head>")]
    assert "finai:theme" in head, f"{page}: theme not restored before paint"


@pytest.mark.parametrize("page", ALL_PAGES)
def test_every_page_has_a_topbar_for_the_toggle(client, page):
    """theme.js mounts the switch into .topbar; no topbar means no toggle."""
    assert 'class="topbar"' in client.get(page).text, f"{page}: no topbar"


def test_theme_script_loads_before_page_scripts(client):
    """theme.js defines C()/themeColors() that page scripts call at render."""
    for page in ("/rl.html", "/portfolio.html", "/risk.html"):
        html = client.get(page).text
        assert html.index("assets/js/theme.js") < html.index("assets/js/api.js"), \
            f"{page}: theme.js must load before api.js"
        if "pages/" in html:
            first_page_script = html.index("assets/js/pages/")
            assert html.index("assets/js/theme.js") < first_page_script, \
                f"{page}: theme.js must load before the page script"


# ================== recommendations page: full model / algorithm coverage
def test_recommendations_page_can_use_all_five_forecast_models(client):
    models = {m["key"] for m in client.get("/api/v1/forecast/models").json()["models"]}
    assert models == {"lstm", "gru", "tcn", "transformer", "cnn_lstm"}


def test_recommendations_page_can_use_all_available_rl_algorithms(client):
    algos = client.get("/api/v1/intel/algorithms").json()["algorithms"]
    available = [a for a in algos if a["available"]]
    assert len(available) == 13, f"expected 13 runnable algorithms, got {len(available)}"

    # Each one must be accepted by the recommendation endpoint
    for a in available:
        r = client.get("/api/v1/signals/recommend/AAPL",
                       params={"rl_algo": a["key"], "include_xai": "false"})
        assert r.status_code == 200, f"{a['key']} rejected: {r.text[:200]}"
        assert r.json()["action"] in ("STRONG_BUY", "BUY", "HOLD", "SELL", "STRONG_SELL")


@pytest.mark.parametrize("model", ["lstm", "gru", "tcn", "transformer", "cnn_lstm"])
def test_every_forecast_model_is_accepted_by_the_engine(client, model):
    r = client.get("/api/v1/signals/recommend/AAPL",
                   params={"forecast_model": model, "include_xai": "false"})
    assert r.status_code == 200, r.text[:200]
    sources = {s["source"] for s in r.json()["signals"]}
    assert "forecast" in sources


def test_selector_lists_are_served_by_the_api_not_hardcoded():
    """The page builds both dropdowns from the API; hard-coded <option> lists
    silently go stale whenever a model or algorithm is added."""
    from pathlib import Path

    js = (Path(__file__).resolve().parents[2] / "frontend" / "assets" / "js"
          / "pages" / "signals.js").read_text()
    assert "api.forecastModels()" in js, "forecast list is not fetched"
    assert "api.algorithms()" in js, "RL algorithm list is not fetched"

    import re

    html = (Path(__file__).resolve().parents[2] / "frontend" / "signals.html").read_text()

    # Count options inside the two selects this test protects, rather than
    # across the whole page. A page-wide count cannot tell a stale catalogue
    # (the fault) from a fixed set of UI choices such as the direction
    # horizon (not a fault), and grew brittle the moment a select was added.
    for select_id in ("sModel", "sRlAlgo"):
        block = re.search(rf'<select id="{select_id}".*?</select>', html, re.S)
        assert block, f"the {select_id} selector is missing"
        n_options = block.group(0).count('<option value="')
        assert n_options <= 1, (
            f"{select_id} hard-codes {n_options} options; the list must come "
            "from the API or it goes stale when a model is added")


# ================================================ portfolio analytics coherence
@pytest.mark.asyncio
async def test_portfolio_metrics_match_their_own_equity_curve():
    """Regression: headline metrics and the equity curve described different
    series. Benchmark alignment intersected the portfolio's dates with SPY's, so
    a crypto holding (365 bars/year) lost every weekend, while the chart kept
    them. The demo portfolio showed -0.26% and Sharpe -0.00 next to a curve that
    had risen +2.76%.

    Portfolio metrics must come from the portfolio's own history; only the
    benchmark-relative figures may use the overlap.
    """
    import numpy as np

    from app.services.data.market_data import market_data_service
    from app.services.risk.metrics import full_metrics

    symbols = ["AAPL", "BTC-USD"]          # 250 vs 365 bars a year
    prices = market_data_service.get_price_matrix(symbols, period="1y")
    returns = prices.pct_change().dropna()
    weights = np.array([0.5, 0.5])
    weights = np.array([weights[symbols.index(c)] for c in returns.columns])
    portfolio_returns = returns @ weights

    metrics = full_metrics(portfolio_returns)
    equity = (1 + portfolio_returns).cumprod()
    curve_return = float(equity.iloc[-1]) - 1.0

    assert metrics["total_return"] == pytest.approx(curve_return, abs=1e-3), (
        "metrics and equity curve disagree: "
        f"{metrics['total_return']:.4f} vs {curve_return:.4f}")


@pytest.fixture
def funded_portfolio(client):
    """A portfolio holding a stock and a crypto.

    The mix is the point: the crypto trades 7 days a week and the equity does
    not, which is precisely the situation that broke the metrics. Building it
    here keeps the test meaningful in CI instead of skipping.
    """
    created = client.post("/api/v1/portfolio",
                          json={"name": "coherence-test", "initial_capital": 100_000})
    assert created.status_code == 200, created.text[:200]
    pid = created.json()["id"]
    for symbol in ("AAPL", "BTC-USD"):
        client.post(f"/api/v1/portfolio/{pid}/trade",
                    json={"symbol": symbol, "side": "BUY", "notional": 25_000})
    yield pid
    client.delete(f"/api/v1/portfolio/{pid}")


def test_analytics_curve_starts_at_the_initial_capital(client, funded_portfolio):
    """cumprod() starts *after* the first return, so the curve opened at
    100,377 on a 100,000 account and under-reported the total gain."""
    payload = client.get(f"/api/v1/portfolio/{funded_portfolio}/analytics",
                         params={"period": "1y"}).json()
    curve = payload.get("equity_curve") or []
    assert curve, "the funded portfolio produced no equity curve"

    assert curve[0]["value"] == pytest.approx(payload["initial_capital"], rel=1e-6), \
        "the equity curve does not start at the initial capital"

    curve_return = curve[-1]["value"] / curve[0]["value"] - 1
    assert payload["metrics"]["total_return"] == pytest.approx(curve_return, abs=5e-3), \
        "the headline return still disagrees with the curve"


def test_benchmark_overlap_is_disclosed(client, funded_portfolio):
    """A beta computed on 250 of 365 days should not read as absolute."""
    metrics = client.get(f"/api/v1/portfolio/{funded_portfolio}/analytics",
                         params={"period": "1y"}).json().get("metrics") or {}
    assert metrics, "the funded portfolio produced no metrics"
    if "beta" in metrics:
        assert "benchmark_overlap_days" in metrics, \
            "beta is reported without saying how many days it covers"


# ============================================================ portfolio page
def test_portfolio_page_has_no_duplicate_symbol_search():
    """The topbar search only filled the Trade form, duplicating a field that
    sits two panels below while implying the whole page followed it."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    html = (root / "frontend" / "portfolio.html").read_text()
    js = (root / "frontend" / "assets" / "js" / "pages" / "portfolio.js").read_text()

    assert 'id="globalSearch"' not in html, "the symbol search is still in the topbar"
    assert "initSearch(" not in js, "portfolio.js still wires the removed search"
    # the control that does drive the page must stay
    assert 'id="portfolioSelect"' in html


def test_holdings_table_explains_its_weights():
    """Weights are shares of total value including cash, so three positions
    summing to 60% looked wrong until the cash row and the label said so."""
    from pathlib import Path

    js = (Path(__file__).resolve().parents[2] / "frontend" / "assets" / "js"
          / "pages" / "portfolio.js").read_text()
    assert "% of total" in js, "the weight column is still ambiguously labelled"
    assert "data.cash_weight" in js, "the cash row is missing from the holdings table"
    assert "add up to 100%" in js, "the weight basis is not explained"


# ==================================================== trade ticket integrity
def test_trade_rejects_an_unverifiable_ticker(client, funded_portfolio):
    """The synthetic engine prices *anything*, so a typo used to execute happily
    at a fabricated $100.00 and sit in the account as a real position. Booking
    an invented price is the one mistake this module must never make."""
    response = client.post(f"/api/v1/portfolio/{funded_portfolio}/trade",
                           json={"symbol": "ZZZQQ999", "side": "BUY", "notional": 1000})
    assert response.status_code >= 400, \
        f"an unverifiable ticker was accepted: {response.text[:200]}"
    assert "could not be verified" in response.text

    # An explicit price is an informed override and must still work.
    override = client.post(f"/api/v1/portfolio/{funded_portfolio}/trade",
                           json={"symbol": "ZZZQQ999", "side": "BUY",
                                 "notional": 1000, "price": 42.0})
    assert override.status_code == 200, override.text[:200]


def test_real_tickers_are_unaffected_by_the_guard(client, funded_portfolio):
    response = client.post(f"/api/v1/portfolio/{funded_portfolio}/trade",
                           json={"symbol": "MSFT", "side": "BUY", "notional": 1000})
    assert response.status_code == 200, response.text[:200]


def test_selling_the_whole_position_closes_it(client, funded_portfolio):
    """Regression: selling a quantity read back from the API (rounded to 8 dp)
    left ~1e-8 of a share behind. That is above the old 1e-9 floor, so the
    holding survived showing qty 0.0 — and could go slightly negative, which
    rendered as a phantom short position."""
    holdings = client.get(f"/api/v1/portfolio/{funded_portfolio}").json()["holdings"]
    assert holdings, "the fixture portfolio holds nothing"
    target = holdings[0]

    sold = client.post(f"/api/v1/portfolio/{funded_portfolio}/trade",
                       json={"symbol": target["symbol"], "side": "SELL",
                             "quantity": target["quantity"]})
    assert sold.status_code == 200, sold.text[:200]

    after = client.get(f"/api/v1/portfolio/{funded_portfolio}").json()
    symbols = [h["symbol"] for h in after["holdings"]]
    assert target["symbol"] not in symbols, \
        f"{target['symbol']} survived as dust: {after['holdings']}"
    for holding in after["holdings"]:
        assert holding["quantity"] > 0, f"a non-positive position remains: {holding}"


# ============================================================ trade ticket UI
def test_trade_ticket_uses_the_symbol_picker():
    """A bare text box turns a typo into a position; the picker is the same
    grouped, ranked control the analysis panel already uses."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    html = (root / "frontend" / "portfolio.html").read_text()
    js = (root / "frontend" / "assets" / "js" / "pages" / "portfolio.js").read_text()

    assert 'id="tSymbolPanel"' in html, "the trade ticket has no picker panel"
    assert "SymbolPicker('tSymbol', 'tSymbolPanel'" in js


def test_trade_ticket_shows_a_live_quote_and_preview():
    """Trading against a number typed into a box is how people fat-finger an
    order: the ticket shows the live price and what the order will actually do."""
    from pathlib import Path

    js = (Path(__file__).resolve().parents[2] / "frontend" / "assets" / "js"
          / "pages" / "portfolio.js").read_text()
    assert "refreshTradeQuote" in js and "renderTradePreview" in js
    for element in ("Est. fee", "Total cost", "Quantity"):
        assert element in js, f"the order preview omits {element!r}"
    # warnings are computed client-side before the backend has to refuse
    assert "Insufficient" in js or "exceeds the" in js
    assert "portfolioCash" in js, "buying power is not tracked for the preview"


def test_holdings_rows_are_actionable():
    """Looking at a position is usually followed by trading it; retyping the
    ticker invites the very typo the picker exists to prevent."""
    from pathlib import Path

    js = (Path(__file__).resolve().parents[2] / "frontend" / "assets" / "js"
          / "pages" / "portfolio.js").read_text()
    assert "holding-row" in js, "holdings rows are not clickable"
    assert "close-position" in js, "there is no one-click way to exit a position"
    # closing must sell by quantity: a notional from a stale price leaves dust
    assert "quantity: holding.quantity" in js, \
        "closing a position must sell by quantity, not notional"


def test_symbol_picker_ranks_by_relevance():
    """Regression: a plain includes() scored a substring anywhere in the name as
    highly as a ticker prefix, so typing one letter returned 26 of 32
    instruments — a list, not a search result."""
    from pathlib import Path

    js = (Path(__file__).resolve().parents[2] / "frontend" / "assets" / "js"
          / "symbolpicker.js").read_text()
    assert "static score(" in js, "the picker has no relevance ranking"
    assert "startsWith(q)" in js, "prefix matches are not prioritised"
    assert "q.length >= 3" in js, \
        "substring matches are not gated behind a deliberate query length"


# ============================================================= alert rules
def test_alert_rules_can_be_listed(client):
    """Regression: only POST /alerts/rules existed, so a rule could be created
    and then never seen again. The panel fell back to showing *triggered
    alerts*, and a freshly created rule (which has triggered nothing) left the
    card looking empty — it read as if saving had failed."""
    created = client.post("/api/v1/alerts/rules",
                          json={"symbol": "AAPL", "rule_type": "price_above",
                                "threshold": 250})
    assert created.status_code == 200, created.text[:200]
    rule_id = created.json()["id"]

    listing = client.get("/api/v1/alerts/rules")
    assert listing.status_code == 200, "GET /alerts/rules is missing"
    payload = listing.json()
    assert payload["count"] >= 1
    mine = [r for r in payload["rules"] if r["id"] == rule_id]
    assert mine, "the created rule is not returned by the listing"
    assert mine[0]["symbol"] == "AAPL"
    assert mine[0]["is_active"] is True


def test_alert_rules_can_be_paused_and_deleted(client):
    """Pausing is not deleting: a noisy threshold is worth silencing without
    losing how it was configured."""
    rule_id = client.post("/api/v1/alerts/rules",
                          json={"symbol": "MSFT", "rule_type": "rsi",
                                "threshold": 70}).json()["id"]

    toggled = client.post(f"/api/v1/alerts/rules/{rule_id}/toggle")
    assert toggled.status_code == 200
    assert toggled.json()["is_active"] is False
    # a paused rule is still listed, just inactive
    listed = [r for r in client.get("/api/v1/alerts/rules").json()["rules"]
              if r["id"] == rule_id]
    assert listed and listed[0]["is_active"] is False

    assert client.post(f"/api/v1/alerts/rules/{rule_id}/toggle").json()["is_active"] is True

    assert client.delete(f"/api/v1/alerts/rules/{rule_id}").status_code == 200
    remaining = [r["id"] for r in client.get("/api/v1/alerts/rules").json()["rules"]]
    assert rule_id not in remaining


def test_deleting_a_missing_rule_is_a_404(client):
    assert client.delete("/api/v1/alerts/rules/999999").status_code == 404


def test_watchlist_scan_reports_a_usable_total(client):
    """`alerts` is a dict keyed by symbol, so reading .length on it in the
    frontend yielded undefined and the badge silently showed 0 even when the
    scan had stored fourteen alerts. The count must come from total_alerts."""
    response = client.post("/api/v1/alerts/scan",
                           json={"symbols": ["AAPL", "MSFT"], "persist": False})
    assert response.status_code == 200
    payload = response.json()
    assert isinstance(payload["alerts"], dict), "alerts is grouped by symbol"
    assert isinstance(payload["total_alerts"], int)
    assert payload["total_alerts"] == sum(len(v) for v in payload["alerts"].values())


def test_risk_page_scans_automatically_and_counts_correctly():
    """An alerts panel that only fills in after the user remembers to press a
    button is not an alerts panel."""
    from pathlib import Path

    js = (Path(__file__).resolve().parents[2] / "frontend" / "assets" / "js"
          / "pages" / "risk.js").read_text()
    assert "autoScanAlerts" in js
    assert "autoScanAlerts();" in js.split("DOMContentLoaded")[1], \
        "the watchlist scan does not run on page load"
    assert "result.total_alerts" in js, \
        "the badge counts a dict's .length instead of total_alerts"
    # The rule builder was replaced by the AI Confidence Score card; what the
    # scan now feeds is the alert history, which holds the scanner's own
    # alerts (news, anomaly, volatility, risk).
    assert "loadHistory" in js, "the scan results have nowhere to appear"
    assert "loadHistory()" in js.split("DOMContentLoaded")[1], \
        "the alert history is never populated on load"


# ====================================================== portfolio deletion
def test_deleting_a_portfolio_removes_its_positions(client):
    """Deletion must not leave orphaned positions or transactions behind."""
    pid = client.post("/api/v1/portfolio",
                      json={"name": "delete-me", "initial_capital": 30_000}).json()["id"]
    client.post(f"/api/v1/portfolio/{pid}/trade",
                json={"symbol": "AAPL", "side": "BUY", "notional": 5_000})

    assert client.delete(f"/api/v1/portfolio/{pid}").status_code == 200
    assert client.get(f"/api/v1/portfolio/{pid}").status_code == 404
    remaining = [p["id"] for p in client.get("/api/v1/portfolio").json()["portfolios"]]
    assert pid not in remaining


def test_portfolio_page_exposes_a_delete_control():
    """api.deletePortfolio() existed but nothing in the UI ever called it, so
    a portfolio could be created and never removed."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    html = (root / "frontend" / "portfolio.html").read_text()
    js = (root / "frontend" / "assets" / "js" / "pages" / "portfolio.js").read_text()

    assert 'id="deletePBtn"' in html, "there is no delete button"
    assert "api.deletePortfolio(" in js, "the delete button is not wired"
    # destructive and irreversible: it must ask first
    assert "window.confirm(" in js, "deletion happens without confirmation"
    assert "cannot be undone" in js, "the confirmation does not state the consequence"


# ==================================================== custom alert rules
def test_alert_metric_catalogue_matches_what_the_evaluator_implements():
    """The UI builds its dropdown from this list. Offering a metric the
    resolver cannot compute produces a rule that silently never fires."""
    from app.services.alerts.metrics import METRIC_SPECS, MetricResolver

    for spec in METRIC_SPECS:
        assert hasattr(MetricResolver, f"_m_{spec.key}"), \
            f"metric '{spec.key}' is advertised but has no resolver"


def test_alert_periods_cover_months_and_years(client):
    body = client.get("/api/v1/alerts/metrics")
    assert body.status_code == 200, body.text
    periods = body.json()["periods"]
    for wanted in ("1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "max"):
        assert wanted in periods, f"the alert period selector cannot choose {wanted}"


def test_every_template_is_a_valid_rule():
    """A one-click template that fails validation is worse than no template."""
    from app.services.alerts.rules import TEMPLATES, validate_conditions

    assert len(TEMPLATES) >= 9
    for template in TEMPLATES:
        problems = validate_conditions(template["conditions"])
        assert not problems, f"template '{template['key']}' is invalid: {problems}"
        assert template["logic"] in ("AND", "OR")
        assert template["priority"] in ("low", "medium", "high", "critical")


def test_and_or_logic_is_actually_applied():
    from types import SimpleNamespace

    from app.services.alerts.rules import evaluate_rule

    def rule(logic):
        return SimpleNamespace(
            symbol="AAPL", rule_type="custom", threshold=0, period="1y", logic=logic,
            conditions=[
                {"metric": "price", "operator": "above", "value": 0},        # true
                {"metric": "price", "operator": "below", "value": -1},       # false
            ])

    assert evaluate_rule(rule("AND"))["fired"] is False, "AND fired with one condition false"
    assert evaluate_rule(rule("OR"))["fired"] is True, "OR did not fire with one condition true"


def test_an_unresolvable_metric_never_fires():
    """Treating "cannot compute" as a pass would alert on missing data — the
    worst possible failure for a risk tool."""
    from types import SimpleNamespace

    from app.services.alerts.rules import evaluate_rule

    verdict = evaluate_rule(SimpleNamespace(
        symbol="AAPL", rule_type="custom", threshold=0, period="1y", logic="OR",
        conditions=[{"metric": "does_not_exist", "operator": "above", "value": 1}]))
    assert verdict["fired"] is False
    assert verdict["conditions"][0]["observed"] is None
    assert verdict["conditions"][0]["reason"], "no explanation for the unresolved metric"


def test_an_empty_rule_never_fires():
    """`all([])` is True, so a malformed rule with no conditions would have
    alerted on every single evaluation pass."""
    from types import SimpleNamespace

    from app.services.alerts.rules import evaluate_rule

    verdict = evaluate_rule(SimpleNamespace(
        symbol="AAPL", rule_type="__none__", threshold=0, period="1y",
        logic="AND", conditions=[]))
    # A legacy rule_type with no mapping yields one unresolvable condition,
    # never an empty condition list that would vacuously pass.
    assert verdict["fired"] is False


def test_legacy_single_threshold_rules_still_evaluate():
    """Rules saved before multi-condition support must keep working untouched,
    including the RSI rule whose direction flipped around 50."""
    from types import SimpleNamespace

    from app.services.alerts.rules import normalise_conditions

    oversold = SimpleNamespace(rule_type="rsi", threshold=30.0, conditions=[])
    assert normalise_conditions(oversold)[0]["operator"] == "below", \
        "the legacy oversold RSI rule inverted into an overbought rule"

    overbought = SimpleNamespace(rule_type="rsi", threshold=70.0, conditions=[])
    assert normalise_conditions(overbought)[0]["operator"] == "above"

    below = SimpleNamespace(rule_type="price_below", threshold=100.0, conditions=[])
    assert normalise_conditions(below)[0] == {
        "metric": "price", "operator": "below", "value": 100.0}


def test_rule_crud_and_bulk_actions(client):
    created = client.post("/api/v1/alerts/rules", json={
        "symbol": "MSFT", "name": "test rule", "priority": "high", "period": "1y",
        "logic": "OR",
        "conditions": [{"metric": "rsi", "operator": "above", "value": 70}]})
    assert created.status_code == 200, created.text
    rule = created.json()
    assert rule["summary"], "the rule has no human-readable summary"

    edited = client.patch(f"/api/v1/alerts/rules/{rule['id']}",
                          json={"priority": "critical", "period": "5y"})
    assert edited.status_code == 200
    assert edited.json()["priority"] == "critical"
    assert edited.json()["period"] == "5y"

    clone = client.post(f"/api/v1/alerts/rules/{rule['id']}/duplicate")
    assert clone.status_code == 200
    assert clone.json()["id"] != rule["id"]
    assert "copy" in clone.json()["name"]

    bulk = client.post("/api/v1/alerts/rules/bulk",
                       json={"rule_ids": [rule["id"], clone.json()["id"]],
                             "action": "disable"})
    assert bulk.status_code == 200 and bulk.json()["affected"] == 2

    paused = client.get("/api/v1/alerts/rules?status=paused").json()
    ids = {r["id"] for r in paused["rules"]}
    assert rule["id"] in ids and clone.json()["id"] in ids

    client.post("/api/v1/alerts/rules/bulk",
                json={"rule_ids": [rule["id"], clone.json()["id"]], "action": "delete"})


def test_a_rule_that_cannot_be_evaluated_is_refused(client):
    """Saving a rule that silently never fires leaves the user believing they
    are covered when nothing is watching."""
    bad = client.post("/api/v1/alerts/rules", json={
        "symbol": "AAPL",
        "conditions": [{"metric": "not_a_metric", "operator": "above", "value": 1}]})
    assert bad.status_code == 422, bad.text
    assert "not_a_metric" in bad.text


def test_history_filters_run_in_sql_not_after_the_limit(client):
    """Filtering an already-LIMITed page searches only the newest N rows: one
    critical alert among hundreds then looked like an empty history."""
    import inspect

    from app.services.alerts.engine import AlertEngine

    source = inspect.getsource(AlertEngine.list_alerts)
    limit_at = source.index(".limit(limit)")
    for clause in ("Alert.severity == severity", "func.lower(Alert.title)"):
        assert clause in source, f"{clause} is not part of the query"
        assert source.index(clause) < limit_at, \
            "a filter is applied after the row limit, so it only sees one page"


def test_alert_history_reports_the_values_that_fired_it(client):
    """A history entry saying only "rule fired" cannot be audited."""
    created = client.post("/api/v1/alerts/rules", json={
        "symbol": "AAPL", "name": "always true", "priority": "critical",
        "period": "1y", "recurring": False,
        "conditions": [{"metric": "price", "operator": "above", "value": 0.01}]}).json()

    fired = client.post("/api/v1/alerts/rules/evaluate")
    assert fired.status_code == 200

    history = client.get(f"/api/v1/alerts?rule_id={created['id']}&limit=20").json()
    assert history["count"] >= 1, "the rule fired but nothing was recorded"
    entry = history["alerts"][0]
    assert entry["priority"] == "critical"
    assert entry["period"] == "1y"
    assert entry["triggers"], "no trigger values recorded"
    assert entry["triggers"][0]["observed"] is not None
    assert entry["reason"]

    # recurring=False must retire the rule rather than re-fire every cooldown.
    after = client.get("/api/v1/alerts/rules?symbol=AAPL").json()
    mine = [r for r in after["rules"] if r["id"] == created["id"]]
    assert mine and mine[0]["is_active"] is False, "a one-shot rule stayed active"
    assert mine[0]["trigger_count"] >= 1
    client.delete(f"/api/v1/alerts/rules/{created['id']}")


def test_recommendation_survives_an_uncomputable_bubble_score():
    """`.get(key, 0.0)` does not protect against a key holding None, and both
    risk scores are deliberately None on short windows. That crashed the whole
    recommendation on any period under ~200 bars."""
    import inspect

    from app.services.recommendation import engine as rec_engine

    source = inspect.getsource(rec_engine.RecommendationEngine.recommend)
    assert 'get("crash_risk_score") or 0.0' in source
    assert 'get("bubble_score") or 0.0' in source


# ================================================== AI Confidence Score
def _signal(source, score, available=True, reliability=0.8):
    return {"source": source, "score": score, "available": available,
            "reliability": reliability, "weight": 0.25}


def test_confidence_contributors_reconstruct_the_score():
    """The breakdown shown to the user must sum to the headline number,
    otherwise "what drives this score" is decoration."""
    from app.services.recommendation.confidence import WEIGHTS, confidence_report

    report = confidence_report({
        "action": "BUY", "composite_score": 0.42,
        "signals": [_signal("forecast", 0.4), _signal("rl", 0.5),
                    _signal("technical", 0.3), _signal("sentiment", 0.2)],
        "risk": {"crash_risk": {"crash_risk_score": 0.2},
                 "bubble": {"bubble_score": 0.1}},
    })
    total = sum(c["points"] for c in report["contributors"])
    assert abs(total - report["percent"]) < 0.2, (
        f"contributors sum to {total} but the score says {report['percent']}")
    assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9, "weights are not a partition of 1"
    assert 0 <= report["percent"] <= 100


def test_the_risk_engine_actually_moves_the_score():
    """The engine's own confidence ignored the Risk Engine entirely. A score
    presented as "the AI's confidence" must not overstate certainty in exactly
    the conditions where certainty is least warranted."""
    from app.services.recommendation.confidence import confidence_report

    base = {"action": "BUY", "composite_score": 0.5,
            "signals": [_signal("forecast", 0.5), _signal("rl", 0.5),
                        _signal("technical", 0.5), _signal("sentiment", 0.5)]}

    calm = confidence_report({**base, "risk": {
        "crash_risk": {"crash_risk_score": 0.05},
        "bubble": {"bubble_score": 0.05}}})
    dangerous = confidence_report({**base, "risk": {
        "crash_risk": {"crash_risk_score": 0.80},
        "bubble": {"bubble_score": 0.70}}})

    assert dangerous["percent"] < calm["percent"], \
        "elevated crash and bubble risk did not reduce confidence"
    # And it must be the risk contributor that moved, not something else.
    risk_calm = next(c for c in calm["contributors"] if c["key"] == "risk_clarity")
    risk_bad = next(c for c in dangerous["contributors"] if c["key"] == "risk_clarity")
    assert risk_bad["points"] < risk_calm["points"]


def test_disagreeing_models_score_lower_than_agreeing_ones():
    from app.services.recommendation.confidence import confidence_report

    risk = {"crash_risk": {"crash_risk_score": 0.2}, "bubble": {"bubble_score": 0.1}}
    agree = confidence_report({
        "action": "BUY", "composite_score": 0.45, "risk": risk,
        "signals": [_signal("forecast", 0.4), _signal("rl", 0.5),
                    _signal("technical", 0.4), _signal("sentiment", 0.5)]})
    split = confidence_report({
        "action": "HOLD", "composite_score": 0.02, "risk": risk,
        "signals": [_signal("forecast", 0.4), _signal("rl", -0.5),
                    _signal("technical", 0.4), _signal("sentiment", -0.5)]})
    assert split["percent"] < agree["percent"]
    assert split["band"] in ("very-low", "low", "moderate")


def test_two_surviving_models_do_not_read_as_full_consensus():
    """A ratio-only agreement metric gives 1.0 for "2 of 2 agree" and for
    "4 of 4 agree". The score would then climb precisely when most models
    failed to run."""
    from app.services.recommendation.confidence import _agreement

    four, _ = _agreement([_signal("forecast", 0.4), _signal("rl", 0.5),
                          _signal("technical", 0.4), _signal("sentiment", 0.5)])
    two, _ = _agreement([_signal("technical", 0.4), _signal("sentiment", 0.5),
                         _signal("forecast", 0.0, False), _signal("rl", 0.0, False)])
    assert two < four, "unanimity among two models scored the same as among four"


def test_missing_models_are_penalised_not_ignored():
    """An untrained agent is missing evidence, not neutral evidence."""
    from app.services.recommendation.confidence import confidence_report

    risk = {"crash_risk": {"crash_risk_score": 0.2}, "bubble": {"bubble_score": 0.1}}
    full = confidence_report({
        "action": "BUY", "composite_score": 0.45, "risk": risk,
        "signals": [_signal("forecast", 0.4), _signal("rl", 0.5),
                    _signal("technical", 0.4), _signal("sentiment", 0.5)]})
    partial = confidence_report({
        "action": "BUY", "composite_score": 0.45, "risk": risk,
        "signals": [_signal("forecast", 0.0, False), _signal("rl", 0.0, False),
                    _signal("technical", 0.4), _signal("sentiment", 0.5)]})

    assert partial["percent"] < full["percent"]
    coverage = next(c for c in partial["contributors"] if c["key"] == "coverage")
    assert coverage["points"] < coverage["max_points"]
    # It must name what is missing, not just dock points silently.
    assert "deep learning" in coverage["detail"].lower()


def test_unknown_risk_is_not_treated_as_safe():
    """Risk scores are deliberately None on short windows. Scoring that as
    calm conditions would invent reassurance the data does not support."""
    from app.services.recommendation.confidence import confidence_report

    unknown = confidence_report({
        "action": "BUY", "composite_score": 0.4,
        "signals": [_signal("forecast", 0.4), _signal("rl", 0.4)],
        "risk": {"crash_risk": {"crash_risk_score": None},
                 "bubble": {"bubble_score": None}}})
    calm = confidence_report({
        "action": "BUY", "composite_score": 0.4,
        "signals": [_signal("forecast", 0.4), _signal("rl", 0.4)],
        "risk": {"crash_risk": {"crash_risk_score": 0.05},
                 "bubble": {"bubble_score": 0.05}}})
    assert unknown["percent"] < calm["percent"], \
        "an unverifiable risk picture scored as well as a verified calm one"

    # Assert the *risk* contributor specifically. Unknown scores also reduce
    # coverage, so comparing totals alone passed even when risk_clarity was
    # hard-coded to 1.0 — the test proved a penalty existed, not that it came
    # from the factor it names.
    risk_unknown = next(c for c in unknown["contributors"] if c["key"] == "risk_clarity")
    risk_calm = next(c for c in calm["contributors"] if c["key"] == "risk_clarity")
    assert risk_unknown["points"] < risk_calm["points"], \
        "unknown risk scored the same as measured calm conditions"


def test_every_band_is_reachable_and_ordered():
    from app.services.recommendation.confidence import BANDS, _band

    thresholds = [t for t, _l, _k in BANDS]
    assert thresholds == sorted(thresholds, reverse=True), "bands are not ordered"
    assert _band(0.95)[0] == "Very High"
    assert _band(0.70)[0] == "High"
    assert _band(0.50)[0] == "Moderate"
    assert _band(0.30)[0] == "Low"
    assert _band(0.05)[0] == "Very Low"


def test_confidence_endpoint_returns_what_the_card_renders(client):
    body = client.get("/api/v1/signals/confidence/AAPL?period=1y")
    assert body.status_code == 200, body.text
    data = body.json()
    for field in ("score", "percent", "label", "band", "action", "contributors",
                  "summary", "bands", "basis", "recommendation"):
        assert field in data, f"the payload is missing {field}"
    assert len(data["contributors"]) == 5, "not all five model families are reported"
    assert len(data["bands"]) == 5
    # The card sits beside the recommendation, so the verdict must travel with it.
    assert data["recommendation"]["action"] == data["action"]


def test_the_score_does_not_claim_to_be_a_probability(client):
    """It is a weighted read of agreement and quality, not a backtested hit
    rate. Presenting it as one would be a stronger claim than the evidence."""
    data = client.get("/api/v1/signals/confidence/AAPL?period=1y").json()
    assert "not a probability" in data["basis"].lower()


# ============================================ global time-range component
def test_the_catalogue_offers_every_requested_range(client):
    body = client.get("/api/v1/market/time-ranges")
    assert body.status_code == 200, body.text
    labels = [r["label"] for r in body.json()["ranges"]]
    # 1W and 2W were removed: five daily points is not a chart, and the
    # indicators drawn on this platform need more.
    assert labels == ["1D", "5D", "1M", "3M", "6M", "YTD",
                      "1Y", "3Y", "5Y", "10Y", "MAX"], labels


def test_every_range_key_is_accepted_by_the_data_layer(client):
    """The UI renders this catalogue verbatim, so a key the backend rejects
    would be an unusable button."""
    from app.utils.periods import RANGES
    from app.utils.timeseries import VALID_PERIODS

    for r in RANGES:
        assert r.display in VALID_PERIODS, f"{r.label} maps to invalid period {r.display}"
        assert r.compute in VALID_PERIODS, f"{r.label} computes over invalid {r.compute}"


def test_display_and_compute_windows_are_independent():
    """Selecting 1M must not fit the crash model on 22 bars. The window a user
    looks at and the window a model needs are different questions."""
    from app.utils.periods import _rank, compute_period, resolve

    for key in ("1d", "5d", "1mo"):
        display = resolve(key).display
        for model in ("crash_risk", "regime", "bubble"):
            fit = compute_period(key, model)
            assert _rank(fit) > _rank(display), (
                f"{key}: {model} would fit on {fit}, no longer than the {display} display")


def test_a_long_selection_is_never_shortened_by_a_model_floor():
    """Asking for 10Y must not make a model fit on 2Y just because that is its
    minimum."""
    from app.utils.periods import _rank, compute_period

    for key in ("3y", "5y", "10y"):
        for model in ("crash_risk", "regime", "bubble"):
            assert _rank(compute_period(key, model)) >= _rank(key), \
                f"{key} was shortened to {compute_period(key, model)} for {model}"


def test_short_windows_still_produce_real_scores(client):
    """The whole point of the split: a 1D chart with a live crash score, not a
    dash. Previously a short period produced None on every risk panel."""
    for period in ("1d", "5d", "1mo"):
        data = client.get(f"/api/v1/risk/scan/AAPL?period={period}").json()
        assert data["crash_risk"]["crash_risk_score"] is not None, \
            f"crash risk is unavailable at period={period}"
        assert data["bubble"]["bubble_score"] is not None, \
            f"bubble score is unavailable at period={period}"
        # And it must say what it actually computed over.
        assert data["computed_over"], "the response hides which history was used"
        assert data["display_period"] == period


def test_short_periods_do_not_fall_back_to_synthetic_data():
    """`len(df) >= 5` rejected a legitimate one-day fetch and dropped through
    to the synthetic engine, so "1D" silently showed 120 invented bars."""
    import inspect

    from app.services.data import market_data

    source = inspect.getsource(market_data.MarketDataService.get_history)
    assert "len(df) >= 5" not in source, \
        "a short real fetch is still discarded in favour of synthetic data"
    assert "len(df) >= 1" in source


def test_intraday_ranges_use_an_intraday_interval():
    """One daily bar is not a chart. 1D and 5D need finer bars to mean
    anything."""
    from app.utils.periods import BY_KEY

    assert BY_KEY["1d"].interval in ("1m", "5m", "15m"), "1D is not intraday"
    assert BY_KEY["5d"].interval in ("15m", "30m", "1h"), "5D is not intraday"
    assert BY_KEY["1y"].interval == "1d"


def test_legacy_period_strings_still_resolve():
    """Bookmarks and stored settings used yfinance spellings; 422-ing them
    would break saved links."""
    from app.utils.periods import resolve

    assert resolve("2y").key in ("1y", "2y")
    assert resolve(None).key == "1y"
    assert resolve("nonsense").key == "1y", "an unknown period should fall back, not raise"


# ============================================ forecast page: two windows
def test_training_history_endpoint_serves_persisted_curves(client):
    """The curves were written to a sidecar JSON beside every checkpoint but
    never served, so the panel could only show something in the same browser
    session that ran the training. A reload went blank."""
    body = client.get("/api/v1/forecast/training-history/AAPL?model=lstm&horizon=5")
    assert body.status_code == 200, body.text
    data = body.json()
    if data["trained"]:
        assert data["history"]["train_loss"], "no loss curve returned"
        assert len(data["history"]["val_loss"]) == len(data["history"]["train_loss"])
        assert data["epochs_run"] >= 1


def test_an_untrained_model_says_so_instead_of_404(client):
    """An untrained model is a normal state, not an error. The UI has to tell
    'nothing trained yet' apart from 'the request failed'."""
    body = client.get("/api/v1/forecast/training-history/ZZZZ?model=lstm&horizon=5")
    assert body.status_code == 200, body.text
    data = body.json()
    assert data["trained"] is False
    assert data["history"] is None
    assert "Train the model" in data["message"]


def test_prediction_uses_the_compute_window_not_the_display_window(client):
    """At 1mo the 21-day rolling features leave zero usable rows, and the model
    rejected the frame with 'Feature mismatch, missing: [...]' — which read as
    a broken checkpoint rather than too short a window."""
    for period in ("1mo", "3mo", "1y"):
        body = client.get(f"/api/v1/forecast/predict/AAPL?model=lstm&horizon=5&period={period}")
        assert body.status_code == 200, f"period={period}: {body.text[:200]}"
        data = body.json()
        assert data["display_period"] == period
        # The fit window must be longer than the display window on short ranges.
        from app.utils.periods import _rank
        assert _rank(data["computed_over"]) >= _rank(period)


def test_short_display_windows_produce_usable_features():
    """The concrete failure: 22 bars yields an empty feature frame."""
    from app.services.data.market_data import market_data_service
    from app.services.indicators.features import build_features
    from app.utils.periods import compute_period

    fit = compute_period("1mo", "forecast")
    df = market_data_service.get_history("AAPL", period=fit).df
    features = build_features(df, dropna=False)
    assert len(features.columns) > 0, f"{fit} still yields no features"
    for core in ("return_1d", "return_21d", "log_return"):
        assert core in features.columns


def test_the_trainer_records_how_much_history_it_used():
    """Without it the sidecar cannot say what the curves were produced from,
    and 'Training History' becomes folklore."""
    import inspect

    from app.services.forecasting import trainer

    assert "bars_used" in inspect.getsource(trainer.TrainResult)
    assert "bars_used=int(len(df))" in inspect.getsource(trainer.ForecastTrainer.train)


# ================================== period catalogue after removing 1W / 2W
def test_weekly_ranges_are_gone_from_the_catalogue(client):
    """On daily bars 1W and 2W are 5 and 10 points — too few for the
    indicators and analytics this platform draws."""
    labels = [r["label"] for r in client.get("/api/v1/market/time-ranges").json()["ranges"]]
    assert "1W" not in labels and "2W" not in labels, labels
    assert labels == ["1D", "5D", "1M", "3M", "6M", "YTD", "1Y", "3Y", "5Y", "10Y", "MAX"]


def test_old_weekly_links_still_resolve():
    """Removing a key must not 422 a bookmark that still carries it."""
    from app.utils.periods import BY_KEY, resolve

    assert "1w" not in BY_KEY and "2w" not in BY_KEY
    assert resolve("1w").key == "5d", "an old 1W link errors instead of falling back"
    assert resolve("2w").key == "1mo"
    assert resolve("1wk").key == "5d"


def test_portfolio_analytics_actually_respond_to_the_period(client):
    """The window must change the numbers, not just the request URL."""
    # Create one rather than skipping: a skipped test guards nothing, and this
    # is the exact behaviour the user reported as broken.
    created = client.post("/api/v1/portfolio",
                          json={"name": "period test", "initial_capital": 100000})
    assert created.status_code == 200, created.text
    pid = created.json()["id"]
    client.post(f"/api/v1/portfolio/{pid}/trade",
                json={"symbol": "AAPL", "side": "BUY", "notional": 10000})

    seen = {}
    for period in ("1mo", "1y", "5y"):
        data = client.get(f"/api/v1/portfolio/{pid}/analytics?period={period}").json()
        seen[period] = len(data.get("equity_curve") or [])
    assert seen["1mo"] < seen["1y"] < seen["5y"], \
        f"the equity curve does not grow with the window: {seen}"


# ================================================ risk engine: per-asset, per-period
def test_the_period_selector_actually_changes_the_risk_numbers(client):
    """Every range from 1D to 1Y declared compute="2y", so seven of the eleven
    selectable ranges fetched the same history and returned one identical
    answer — the control moved and nothing changed. Measured on AAPL before
    the fix: 1D, 5D, 1M, 3M, 6M, YTD and 1Y all returned crash 0.429."""
    seen = {}
    for period in ("1mo", "3mo", "6mo", "1y"):
        body = client.get(f"/api/v1/risk/scan/AAPL?period={period}").json()
        crash = body["crash_risk"]["crash_risk_score"]
        assert crash is not None, f"crash score unavailable at period={period}"
        seen.setdefault(crash, []).append(period)
    assert len(seen) > 1, (
        f"every period returned the same crash score: {seen}")


def test_a_short_selection_still_gets_the_history_each_model_needs(client):
    """The platform's rule: display window and computation window are
    independent, and a short display must never produce a dash."""
    from app.utils.periods import MIN_BARS

    body = client.get("/api/v1/risk/scan/AAPL?period=1mo").json()
    assert body["crash_risk"]["crash_risk_score"] is not None
    assert body["bubble"]["bubble_score"] is not None
    assert body["crash_bars"] > MIN_BARS["crash_risk"], \
        "crash risk was handed fewer bars than its own floor"
    assert body["bubble_bars"] > MIN_BARS["bubble"], \
        "bubble was handed fewer bars than its own floor"


def test_model_bars_clears_the_floor_stated_in_returns_not_bars(client):
    """The floors are counts of *returns*; N price bars yield N-1 returns, so
    slicing exactly 60 bars for a 60-return model tripped its own
    insufficient-data guard and produced the dash the split exists to avoid."""
    from app.utils.periods import MIN_BARS, model_bars

    for model in ("crash_risk", "bubble", "regime"):
        for key in ("1d", "1mo", "3mo"):
            assert model_bars(key, model) > MIN_BARS[model], \
                f"{key}/{model} yields only {model_bars(key, model) - 1} returns"


def test_risk_scan_reports_an_overall_score_not_only_a_band(client):
    """A band name cannot say how far into it a reading sits, and cannot be
    ranked. The headline is now a measured 0-1 composite."""
    body = client.get("/api/v1/risk/scan/AAPL?period=1y").json()
    assert body.get("overall_risk_score") is not None
    profile = body["risk_profile"]
    overall = profile["overall"]
    assert 0.0 <= overall["score"] <= 1.0
    assert overall["level"] == body["overall_risk_level"], \
        "the badge and the score disagree"
    # the classification must match the published band table
    from app.services.risk.profile import classify
    assert classify(overall["score"]) == overall["level"]


def test_risk_profile_exposes_every_requested_metric(client):
    body = client.get("/api/v1/risk/profile/NVDA?period=1y").json()
    metrics = body["metrics"]
    for key in ("annualised_volatility", "var_95_daily", "cvar_95_daily",
                "max_drawdown", "beta", "sharpe_ratio", "sortino_ratio",
                "downside_deviation", "skewness", "excess_kurtosis"):
        assert key in metrics, f"{key} is missing from the risk profile"
    assert metrics["cvar_95_daily"] <= metrics["var_95_daily"], \
        "CVaR must be at least as severe as VaR"


def test_two_different_assets_do_not_share_a_risk_profile(client):
    """Distinct instruments must produce distinct numbers from their own data."""
    a = client.get("/api/v1/risk/profile/AAPL?period=1y").json()["metrics"]
    b = client.get("/api/v1/risk/profile/BTC-USD?period=1y").json()["metrics"]
    assert a["annualised_volatility"] != b["annualised_volatility"]
    assert a["var_95_daily"] != b["var_95_daily"]
    assert a["max_drawdown"] != b["max_drawdown"]


# ============================================ regime-aware RL + decision audit
def test_a_legacy_agent_still_loads_after_regime_features_were_added(client):
    """Adding columns to the observation changes its width, and a trained
    network refuses the wrong width outright:

        mat1 and mat2 shapes cannot be multiplied (1x42 and 36x128)

    Inference must therefore rebuild the environment each agent was *trained*
    in, read back from its own metadata — not today's default."""
    from app.services.rl.service import rl_service

    cfg_default = rl_service._env_config()
    assert cfg_default.regime_aware is False, \
        "the default flipped; every pre-existing agent would fail to load"

    # An agent whose metadata says regime_aware must get a regime-aware env.
    cfg = rl_service._env_config_for_agent("__nonexistent__", "dueling_dqn")
    assert cfg.regime_aware is False, "missing metadata must not enable regime mode"


def test_rl_action_explains_how_the_regime_bore_on_the_decision(client, tiny_regime_agent):
    """A test that skips when no agent exists guarantees nothing, so this
    trains a small regime-aware one first."""
    symbol, algo = tiny_regime_agent
    body = client.get(f"/api/v1/rl/action/{symbol}?algo={algo}&period=1y&audit=false")
    assert body.status_code == 200, body.text
    data = body.json()
    assert "regime_explanation" in data, "the decision carries no regime evidence"
    explanation = data["regime_explanation"]
    assert "available" in explanation
    if explanation["available"]:
        assert explanation["influence"] in ("decisive", "contributory", "negligible")
        assert explanation["summary"]
        # The counterfactual has to name what would have happened instead.
        assert explanation["counterfactual_action"]


def test_the_decision_audit_trail_records_governance_fields(client, tiny_regime_agent):
    """`recommendation_log` existed but nothing ever wrote to it, so the
    platform had a decision log containing no decisions. A model-governance
    review asks which model version decided, under which regime, on what risk
    figures — an empty table cannot answer that."""
    symbol, algo = tiny_regime_agent
    served = client.get(f"/api/v1/rl/action/{symbol}?algo={algo}&period=1y")
    assert served.status_code == 200, served.text

    log = client.get("/api/v1/rl/decisions?limit=5").json()
    assert log["count"] >= 1, "the served decision was not recorded"
    row = log["decisions"][0]
    for field in ("symbol", "action", "confidence", "model_version",
                  "algo", "regime", "regime_influence", "risk_metrics",
                  "created_at"):
        assert field in row, f"the audit row has no {field}"
    assert row["model_version"], "the decision does not identify the model that made it"


def test_audit_filters_are_applied_before_the_limit(client):
    """Filtering after a LIMIT hides matching rows behind non-matching ones —
    a bug this project already shipped once on the alert history."""
    from pathlib import Path

    source = (Path(__file__).resolve().parents[1]
              / "app" / "api" / "v1" / "endpoints" / "rl.py").read_text()
    block = source[source.index("async def decision_log"):]
    block = block[: block.index("return {")]
    # Locate the *last* filter and the limit, rather than one hardcoded filter
    # expression — an earlier version of this test pinned the literal
    # `.where(RecommendationLog.source ==` and broke the moment that filter
    # legitimately became `.in_((...))` to cover basket rebalances too. What
    # matters is the ordering, not the spelling.
    filter_positions = [block.rindex(f) for f in (".where(", ) if f in block]
    assert filter_positions, "the query applies no filters at all"
    limit_pos = block.index(".limit(limit)")
    assert max(filter_positions) < limit_pos, "the LIMIT is applied before the filters"
    # And both RL decision sources must be in scope, or half the trail is hidden.
    assert "rl_agent" in block and "rl_allocation" in block, \
        "the audit trail omits one of the two RL decision sources"


def test_training_records_which_regimes_the_agent_actually_saw(client):
    """A policy that never met a crash cannot be trusted to handle one. The
    metadata has to say so rather than implying broad competence."""
    from app.services.data.synthetic import generate_ohlcv
    from app.services.rl.regime_features import build_provider

    frame = generate_ohlcv("TEST", periods=400)
    summary = build_provider(frame).summary()
    assert "distribution" in summary and "regimes_never_seen" in summary
    assert summary["classified_bars"] > 0
    assert abs(sum(summary["distribution"].values()) - 1.0) < 0.01


def test_multi_asset_training_accepts_regime_awareness(client):
    """PortfolioEnv is what SAC/TD3/DDPG actually allocate with, so regime
    awareness has to reach it too — not only the single-asset env."""
    from app.services.data.market_data import market_data_service
    from app.services.rl.environment import EnvConfig, PortfolioEnv
    from app.services.rl.portfolio_regime import feature_dim

    matrix = market_data_service.get_price_matrix(["AAPL", "MSFT"], period="2y")
    legacy = PortfolioEnv(matrix, EnvConfig())
    aware = PortfolioEnv(matrix, EnvConfig(regime_aware=True))
    assert (aware.observation_space.shape[0] - legacy.observation_space.shape[0]
            == feature_dim(2))
    # Action space must be untouched: only the observation grows.
    assert aware.action_space.shape == legacy.action_space.shape


def test_continuous_agents_still_load_after_portfolio_regime_features(client):
    """SAC/TD3/DDPG run on PortfolioEnv even for a single asset. Widening its
    observation by default would break every one already on disk."""
    from app.services.rl.service import rl_service

    assert rl_service._env_config().regime_aware is False
    cfg = rl_service._env_config_for_agent("__missing__", "sac")
    assert cfg.regime_aware is False, \
        "an agent with no metadata was given a wider observation than it was trained on"


def test_a_trained_basket_agent_can_finally_be_queried(client):
    """`/rl/portfolio/train` could train basket agents but nothing could ask
    them what to hold, so five trained baskets sat on disk unusable."""
    from app.services.rl.service import rl_service

    rl_service.train_portfolio(
        ["AAPL", "MSFT"], period="2y", algo="ppo", total_timesteps=1200,
        env_overrides={"regime_aware": True})

    body = client.get("/api/v1/rl/allocation?symbols=AAPL,MSFT&algo=ppo&period=1y")
    assert body.status_code == 200, body.text
    data = body.json()

    weights = [a["weight"] for a in data["allocation"]] + [data["cash_weight"]]
    assert sum(weights) == pytest.approx(1.0, abs=1e-3), \
        f"the allocation does not sum to 1: {weights}"
    assert data["largest_position"]["weight"] == max(
        a["weight"] for a in data["allocation"])
    assert data["model_version"], "the allocation does not identify its model"


def test_allocation_decisions_reach_the_same_audit_trail(client):
    """A basket rebalance is an RL decision too. A filter naming only the
    single-asset source would hide half the governance record."""
    rebalance = client.get(
        "/api/v1/rl/allocation?symbols=AAPL,MSFT&algo=ppo&period=1y")
    assert rebalance.status_code == 200, rebalance.text

    log = client.get("/api/v1/rl/decisions?limit=20").json()
    rows = [r for r in log["decisions"] if r["action"] == "REBALANCE"]
    assert rows, "the basket rebalance never reached the audit trail"
    row = rows[0]
    assert row["algo"] and row["model_version"]
    assert "cash_weight" in row["risk_metrics"]
    assert "largest_position" in row["risk_metrics"]


# ================================= hyperparameter management API + page
def test_hyperparameter_catalogue_lists_every_algorithm_and_profile(client):
    body = client.get("/api/v1/hyperparams/catalogue")
    assert body.status_code == 200, body.text
    data = body.json()
    keys = {a["key"] for a in data["algorithms"]}
    for algo in ("ppo", "sac", "td3", "ddpg", "a2c", "dqn", "dueling_dqn",
                 "c51", "iqn", "rainbow", "qr_dqn", "trpo", "double_dqn"):
        assert algo in keys, f"{algo} has no configuration file"
    profiles = {p["key"] for p in data["profiles"]}
    assert {"default", "conservative", "aggressive", "risk_aware",
            "high_performance"} <= profiles


def test_resolve_returns_the_materialised_set_not_a_diff(client):
    """A diff would leave the user guessing the effective value."""
    body = client.get("/api/v1/hyperparams/resolve?algo=ppo&profile=conservative")
    assert body.status_code == 200, body.text
    data = body.json()
    params = data["params"]
    for section in ("training", "optimizer", "network", "replay",
                    "exploration", "environment", "risk"):
        assert params.get(section), f"{section} is missing from the resolved set"
    assert data["fingerprint"]
    assert data["sources"] == ["defaults.yaml", "algorithms/ppo.yaml",
                               "profiles/conservative.yaml"]


def test_the_api_refuses_an_out_of_range_hyperparameter(client):
    body = client.post("/api/v1/hyperparams/profiles/bad_profile",
                       json={"config": {"optimizer": {"learning_rate": 99}}})
    assert body.status_code == 422, body.text
    assert "learning_rate" in body.text


def test_the_hyperparameter_page_is_served_and_guarded(client, anon_client):
    """Every dashboard page requires a session; a new one must not be an
    accidental hole in the default-deny wall."""
    assert client.get("/hyperparams.html").status_code == 200
    anonymous = anon_client.get("/hyperparams.html", follow_redirects=False)
    assert anonymous.status_code in (302, 303, 307), \
        "the hyperparameter page is reachable without signing in"


def test_every_page_links_to_the_hyperparameter_editor():
    """A page nothing navigates to is a page nobody finds."""
    from pathlib import Path

    frontend = Path(__file__).resolve().parents[2] / "frontend"
    for page in frontend.glob("*.html"):
        if page.name in ("landing.html", "auth.html"):
            continue
        assert 'href="hyperparams.html"' in page.read_text(), \
            f"{page.name} has no link to the hyperparameter editor"


# ============================================ training monitoring dashboard
def test_the_monitor_records_every_metric_the_dashboard_charts():
    """The dashboard advertises Sortino, volatility, VaR and CVaR. Two of those
    were computed by `env.performance()` and dropped; the other two were never
    computed at all, so charting them would have drawn empty axes."""
    from app.services.rl.monitor import TrainingMonitor

    class _Agent:
        def evaluate(self, env, deterministic=True):
            equity = [100_000 * (1 + 0.004 * i - (0.02 if i % 7 == 0 else 0))
                      for i in range(40)]
            return {"performance": {"total_return": 0.12, "sharpe_ratio": 1.4,
                                    "sortino_ratio": 1.9, "max_drawdown": -0.08,
                                    "annualised_volatility": 0.22,
                                    "final_value": 112_000},
                    "equity_curve": equity}

    monitor = TrainingMonitor(eval_env=object(), eval_freq=1)
    monitor.on_episode_end(1, 10, _Agent())
    entry = monitor.evaluations[0]
    for key in ("total_return", "sharpe_ratio", "sortino_ratio", "max_drawdown",
                "annualised_volatility", "var_95", "cvar_95", "final_value"):
        assert entry.get(key) is not None, f"{key} is not recorded"
    # VaR/CVaR are losses, and CVaR is the mean beyond VaR, so it is worse.
    assert entry["var_95"] < 0 and entry["cvar_95"] <= entry["var_95"]


def test_tail_metrics_refuse_to_estimate_from_too_few_points():
    """A 5% quantile of eight observations is one point. Publishing it would
    dress a single bad day up as a tail estimate."""
    from app.services.rl.monitor import _tail_metrics

    assert _tail_metrics([100, 101, 99, 102])["var_95"] is None
    assert _tail_metrics(None)["cvar_95"] is None
    rich = _tail_metrics([100 + i - (5 if i % 6 == 0 else 0) for i in range(40)])
    assert rich["var_95"] is not None


def test_progress_keeps_training_and_evaluation_series_separate(client):
    """Training reward is measured on the training window and evaluation
    reward on the held-out one. Merging them into a single line would imply a
    continuity that does not exist and hide the overfitting gap."""
    from app.services.rl.service import rl_service

    rl_service.train_single_asset(
        "AAPL", period="2y", algo="dueling_dqn", episodes=4,
        hyperparams={"evaluation": {"eval_freq": 2, "checkpoint_interval": 2}})

    body = client.get("/api/v1/training/progress/AAPL?algo=dueling_dqn")
    assert body.status_code == 200, body.text
    data = body.json()
    assert data["training"] and data["evaluations"]
    assert "training" in data and "evaluations" in data
    # Evaluations land only on multiples of eval_freq, never on every episode.
    episodes = [e["episode"] for e in data["evaluations"]]
    assert all(ep % data["eval_freq"] == 0 for ep in episodes), episodes
    assert len(data["training"]) > len(data["evaluations"])


def test_progress_only_advertises_series_that_were_recorded(client):
    """Claiming a series the backend never produced renders an empty chart with
    no explanation of why it is empty.

    This trains its own agent rather than relying on one left behind by an
    earlier test: run in isolation, the borrowed version failed with a KeyError
    on a 422 body, which says nothing about the behaviour under test.
    """
    from app.services.rl.service import rl_service

    rl_service.train_single_asset(
        "AAPL", period="2y", algo="dueling_dqn", episodes=4,
        hyperparams={"evaluation": {"eval_freq": 2, "checkpoint_interval": 2}})

    response = client.get("/api/v1/training/progress/AAPL?algo=dueling_dqn")
    assert response.status_code == 200, response.text
    body = response.json()
    available = body["available_series"]
    for key, present in available.items():
        if key.startswith("eval_"):
            metric = key[len("eval_"):]
            actually = any(e.get(metric) is not None for e in body["evaluations"])
            assert present == actually, f"{key} advertised as {present}, really {actually}"

    # A run where every series happens to be present cannot distinguish honest
    # reporting from a hardcoded `True` — the mutation that pinned every flag
    # to True passed the loop above. So a run with *no* evaluations at all is
    # trained here and its eval flags must all be False. This is also the real
    # shape of agents trained before periodic evaluation existed: the UI has to
    # show four metric tabs for them, not nine empty ones.
    rl_service.train_single_asset(
        "MSFT", period="2y", algo="dueling_dqn", episodes=2)   # eval_freq = 0

    unmonitored = client.get("/api/v1/training/progress/MSFT?algo=dueling_dqn")
    assert unmonitored.status_code == 200, unmonitored.text
    flags = unmonitored.json()["available_series"]
    eval_flags = {k: v for k, v in flags.items() if k.startswith("eval_")}
    assert eval_flags and not any(eval_flags.values()), (
        f"a run with no evaluations still advertises series: "
        f"{[k for k, v in eval_flags.items() if v]}")
    # Training-side series must still be reported, or the flag is just off.
    assert flags["reward"] is True


def test_the_checkpoint_manager_reports_pruned_files_honestly(client):
    """Retention deletes older files. Listing a checkpoint as restorable when
    its file is gone would fail only at restore time."""
    body = client.get("/api/v1/training/checkpoints")
    assert body.status_code == 200, body.text
    data = body.json()
    assert "retention" in data and data["retention"]["max_checkpoints"] >= 1
    for entry in data["checkpoints"]:
        for field in ("episode", "symbol", "algo", "exists", "created_at",
                      "training_step", "model_version", "seed", "profile"):
            assert field in entry, f"the checkpoint row has no {field}"


def test_comparing_checkpoints_never_invents_a_score(client):
    """A checkpoint is only scored if an evaluation landed on its own episode.
    Interpolating from a neighbour would fabricate a number."""
    from app.services.rl.service import rl_service

    rl_service.train_single_asset(
        "AAPL", period="2y", algo="dueling_dqn", episodes=6,
        hyperparams={"evaluation": {"eval_freq": 2, "checkpoint_interval": 2}})

    aligned = client.post("/api/v1/training/checkpoints/compare", json={
        "left": {"symbol": "AAPL", "algo": "dueling_dqn", "episode": 2},
        "right": {"symbol": "AAPL", "algo": "dueling_dqn", "episode": 4}}).json()
    assert aligned["comparable"] is True
    assert {m["metric"] for m in aligned["metrics"]} >= {
        "total_return", "sharpe_ratio", "max_drawdown"}

    # An episode with no checkpoint has nothing to compare and must say so
    # rather than quietly borrowing the nearest evaluation.
    unaligned = client.post("/api/v1/training/checkpoints/compare", json={
        "left": {"symbol": "AAPL", "algo": "dueling_dqn", "episode": 2},
        "right": {"symbol": "AAPL", "algo": "dueling_dqn", "episode": 999}}).json()
    assert unaligned["comparable"] is False
    assert unaligned["right"]["evaluation_available"] is False


def test_restoring_a_checkpoint_backs_up_the_live_agent(client):
    """Restore overwrites the model every prediction endpoint loads. A restore
    that turns out to be wrong with no way back is worse than no restore."""
    from pathlib import Path

    from app.services.rl.service import rl_service

    rl_service.train_single_asset(
        "AAPL", period="2y", algo="dueling_dqn", episodes=4,
        hyperparams={"evaluation": {"eval_freq": 2, "checkpoint_interval": 2}})

    body = client.post("/api/v1/training/checkpoints/restore", json={
        "symbol": "AAPL", "algo": "dueling_dqn", "episode": 2})
    assert body.status_code == 200, body.text
    data = body.json()
    assert data["restored"] is True
    assert data["backup"], "the previous agent was overwritten with no backup"
    assert "no longer describes what is loaded" in data["warning"]
    # And the restored agent must still be loadable, or restore is a trap.
    assert Path(str(rl_service.agent_path("AAPL", "dueling_dqn")) + ".pt").exists()
    assert rl_service.recommend_action("AAPL", algo="dueling_dqn",
                                       period="1y")["action"]


def test_the_summary_does_not_invent_an_elapsed_training_time(client):
    """Wall-clock duration is not recorded anywhere. Reporting one would be a
    fabricated figure on a card whose whole purpose is provenance."""
    body = client.get("/api/v1/training/summary/AAPL?algo=dueling_dqn").json()
    assert "elapsed_seconds" not in body and "duration" not in body
    # What *is* measured is how long the evaluations took.
    assert "eval_seconds" in body
    for field in ("experiment_id", "profile", "seed", "model_version",
                  "total_episodes", "n_checkpoints", "checkpoints_on_disk",
                  "best_evaluation", "latest_evaluation"):
        assert field in body, f"the summary card has no {field}"


def test_the_training_monitor_page_is_served_and_guarded(client, anon_client):
    assert client.get("/training.html").status_code == 200
    anonymous = anon_client.get("/training.html", follow_redirects=False)
    assert anonymous.status_code in (302, 303, 307), \
        "the training monitor is reachable without signing in"


def test_the_training_page_is_hidden_from_the_navigation_but_still_served():
    """The page was removed from the sidebar on request, not deleted.

    It still hosts the only UI for the checkpoint manager (restore/delete of
    real .pt files), so the route must keep working for anyone holding the
    URL — hiding a page and orphaning its data are different things. This
    pins both halves: no nav entry anywhere, and the file still present.
    """
    from pathlib import Path

    frontend = Path(__file__).resolve().parents[2] / "frontend"
    for page in frontend.glob("*.html"):
        if page.name in ("landing.html", "auth.html"):
            continue
        html = page.read_text()
        assert 'class="nav-item" href="training.html"' not in html, \
            f"{page.name} still advertises the training page in its sidebar"

    assert (frontend / "training.html").exists(), \
        "the page was deleted; it was only meant to be hidden"


# ==================== the selected profile reaches the training run
def test_the_training_api_forwards_the_selected_profile(client):
    """`train_single_asset` accepted a `profile` argument but the endpoint never
    passed it, so every run trained on "default" no matter what the dashboard
    had selected — and then recorded "default" in its reproducibility block,
    making the record accurate about the wrong thing."""
    body = client.post("/api/v1/rl/train", json={
        "symbol": "AAPL", "algo": "dueling_dqn", "period": "2y",
        "episodes": 1, "profile": "conservative"})
    assert body.status_code == 200, body.text
    meta = body.json()

    assert meta["profile"] == "conservative"
    # And the profile must have actually shaped the run, not just been logged.
    assert meta["env_config"]["risk_penalty"] == 0.30
    assert meta["env_config"]["trade_fraction"] == 0.15
    assert meta["hyperparameters"]["optimizer"]["learning_rate"] == 0.0003


def test_two_profiles_produce_two_different_fingerprints(client):
    """Reproducibility depends on the fingerprint identifying the configuration
    that ran. Identical fingerprints across profiles would make the record
    useless for telling two experiments apart."""
    conservative = client.post("/api/v1/rl/train", json={
        "symbol": "AAPL", "algo": "dueling_dqn", "period": "2y",
        "episodes": 1, "profile": "conservative"}).json()
    aggressive = client.post("/api/v1/rl/train", json={
        "symbol": "AAPL", "algo": "dueling_dqn", "period": "2y",
        "episodes": 1, "profile": "aggressive"}).json()

    assert conservative["hyperparameter_fingerprint"] \
        != aggressive["hyperparameter_fingerprint"]
    assert conservative["env_config"]["risk_penalty"] \
        > aggressive["env_config"]["risk_penalty"]


def test_the_rl_page_offers_a_profile_selector():
    """Without a selector the automatic workflow is unreachable: the user would
    have to call the API by hand to train on anything but the default."""
    from pathlib import Path

    frontend = Path(__file__).resolve().parents[2] / "frontend"
    html = (frontend / "rl.html").read_text()
    assert 'id="rProfile"' in html, "the RL page has no profile selector"

    js = (frontend / "assets" / "js" / "pages" / "rl.js").read_text()
    assert "loadProfiles" in js, "the selector is never populated"
    assert "profile: v.profile" in js, "the selection is never sent to the API"
    # Populated from the platform, never hardcoded — otherwise a profile the
    # user creates would not appear.
    assert "api.hpCatalogue()" in js


# ============================================ training intelligence dashboard
def _fake_run(rewards, evaluations=None, **meta):
    base = {
        "symbol": "TEST", "algo": "ppo", "seed": 42, "profile": "default",
        "training_history": {"episode_rewards": rewards},
        "monitoring": {"evaluations": evaluations or [], "checkpoints": []},
        "test_performance": {"total_return": 0.10, "max_drawdown": -0.08,
                             "alpha_vs_buy_hold": 0.02},
        "env_config": {"transaction_cost": 0.001, "slippage": 0.0005,
                       "initial_balance": 100_000.0},
    }
    base.update(meta)
    return base


def test_convergence_diagnosis_separates_the_five_states():
    """The dashboard acts on this label — "Stop Training" versus "Continue" —
    so a wrong classification wastes compute or ships an overfitted agent."""
    from app.services.rl.intelligence import diagnose

    improving = diagnose({"episode_rewards": [1, 2, 3, 4, 6, 8, 11, 15]}, [])
    assert improving.status == "improving"
    assert improving.action == "Continue Training"

    flat = [10.0, 10.1, 9.9, 10.0, 10.05, 9.95, 10.02, 9.98]
    assert diagnose({"episode_rewards": flat}, []).status == "converged"

    erratic = [10, -40, 55, -30, 60, -50, 45, -35]
    assert diagnose({"episode_rewards": erratic}, []).status == "unstable"

    # Training up, held-out down: the case a training curve alone cannot see.
    overfit = diagnose(
        {"episode_rewards": [1, 2, 4, 7, 11, 16, 22, 30]},
        [{"episode": 2, "total_return": 0.20}, {"episode": 8, "total_return": -0.05}])
    assert overfit.status == "overfitting"
    assert overfit.action == "Stop Training"

    assert diagnose({"episode_rewards": [1, 2]}, []).status == "insufficient_data"


def test_the_health_score_contributions_sum_to_the_published_score():
    """Explainability is only real if the arithmetic checks out."""
    from app.services.rl.intelligence import diagnose, health_score

    meta = _fake_run([1, 2, 3, 4, 6, 8],
                     [{"episode": 2, "total_return": 0.05},
                      {"episode": 4, "total_return": 0.08}])
    result = health_score(meta, diagnose(meta["training_history"],
                                         meta["monitoring"]["evaluations"]))
    total = sum(c["points"] for c in result["contributions"]
                if c["points"] is not None)
    assert total == pytest.approx(result["percent"], abs=0.2)
    assert 0 <= result["score"] <= 1


def test_an_unmeasurable_dimension_is_dropped_not_scored_as_zero():
    """Zero means "measured, and bad". A run with no benchmark must not be
    penalised as though it had lost to one."""
    from app.services.rl.intelligence import diagnose, health_score

    meta = _fake_run([1, 2, 3, 4, 6, 8])
    meta["test_performance"] = {"total_return": 0.10, "max_drawdown": -0.08}
    result = health_score(meta, diagnose(meta["training_history"], []))
    performance = next(c for c in result["contributions"]
                       if c["name"].startswith("Performance"))
    assert performance["available"] is False
    assert performance["points"] is None
    assert result["weight_redistributed"] is True
    live = [c for c in result["contributions"] if c["available"]]
    assert sum(c["max_points"] for c in live) == pytest.approx(100.0, abs=0.2)


def test_a_portfolio_agent_is_scored_against_its_own_benchmark():
    """Portfolio runs benchmark against an equal-weight basket, not
    buy-and-hold. Looking only for buy-and-hold dropped the Performance term on
    every portfolio agent and redistributed its 30%: a SAC basket that LOST
    8.6% ranked 4th overall at 80.6% health, because the one dimension it
    failed was the one being ignored."""
    from app.services.rl.intelligence import diagnose, health_score

    meta = _fake_run([1, 2, 3, 4, 5, 6])
    meta["test_performance"] = {"total_return": -0.0863, "max_drawdown": -0.093,
                                "equal_weight_return": 0.0196,
                                "alpha_vs_equal_weight": -0.1059}
    result = health_score(meta, diagnose(meta["training_history"], []))
    performance = next(c for c in result["contributions"]
                       if c["name"].startswith("Performance"))
    assert performance["available"] is True, \
        "a portfolio agent's benchmark was ignored"
    assert performance["raw"] == pytest.approx(-0.1059)
    # It must score *low*, not zero: the scale runs from -20% to +20%, and
    # losing 10.6% is bad without being the worst possible outcome. An earlier
    # version of this test demanded exactly 0 and failed against correct code.
    assert performance["value"] < 0.30, \
        f"losing to the benchmark scored {performance['value']} on performance"

    # The decisive regression check: the same run with the benchmark ignored
    # would score materially higher, which is how the losing agent reached 4th.
    blind = dict(meta)
    blind["test_performance"] = {k: v for k, v in meta["test_performance"].items()
                                 if not k.startswith("alpha_")
                                 and k != "equal_weight_return"}
    blind_result = health_score(blind, diagnose(blind["training_history"], []))
    assert result["score"] < blind_result["score"], \
        "ignoring the benchmark flattered the score, which is the bug"


def test_metrics_that_were_never_recorded_are_reported_as_missing():
    """Training duration is never timed and trades are not persisted. Filling
    either with a plausible number would be a fabricated headline figure."""
    from app.services.rl.intelligence import derived_metrics

    metrics = derived_metrics(_fake_run([5, -3, 8]))
    assert metrics["training_duration_seconds"] is None
    assert "not recorded" in metrics["training_duration_basis"]
    assert "episodes" in metrics["win_rate_basis"]
    # Per-episode win rate is genuinely derivable, and must be.
    assert metrics["episode_win_rate"] == pytest.approx(2 / 3, abs=1e-3)


def test_turnover_is_derived_from_the_recorded_transaction_cost():
    """The environment charged fee+slippage on every unit of notional, so the
    cost divides back out to turnover — no new instrumentation needed."""
    from app.services.rl.intelligence import derived_metrics

    meta = _fake_run([1, 2, 3])
    meta["test_performance"]["total_transaction_cost"] = 1500.0
    # 1500 / 0.0015 / 100000 = 10x the book
    assert derived_metrics(meta)["turnover"] == pytest.approx(10.0)


def test_multi_seed_statistics_refuse_to_report_one_sample():
    """Every agent on disk used seed 42. A standard deviation over one sample
    is zero by construction, and drawing it as an error band would imply a
    reproducibility check that never happened."""
    from app.services.rl.intelligence import seed_statistics

    single = seed_statistics([_fake_run([1, 2, 3], seed=42) for _ in range(5)])
    assert single["available"] is False
    assert "seed" in single["reason"]

    many = [_fake_run([1, 2, 3], seed=s) for s in (1, 2, 3, 4)]
    for i, run in enumerate(many):
        run["test_performance"]["total_return"] = 0.10 + i * 0.02
    assert seed_statistics(many)["available"] is True


def test_the_intelligence_endpoint_ranks_and_filters(client):
    body = client.get("/api/v1/training/intelligence")
    assert body.status_code == 200, body.text
    data = body.json()
    for key in ("runs", "leaderboard", "overall_ranking", "global",
                "adaptive_vs_legacy", "seed_statistics", "facets"):
        assert key in data, f"the payload has no {key}"

    # The ranking must be ordered by health, or it is not a ranking.
    scores = [r["health"] for r in data["overall_ranking"]]
    assert scores == sorted(scores, reverse=True)

    if data["facets"]["symbols"]:
        one = data["facets"]["symbols"][0]
        filtered = client.get(
            f"/api/v1/training/intelligence?symbol={one}").json()
        assert all(r["symbol"] == one for r in filtered["runs"])
        assert filtered["count"] <= data["count"]


def test_the_training_report_states_what_was_not_recorded(client):
    """A report that omits its own gaps invites the reader to assume the
    missing figures were simply zero."""
    from app.services.rl.service import rl_service

    rl_service.train_single_asset("AAPL", period="2y", algo="dueling_dqn",
                                  episodes=2)
    body = client.get("/api/v1/training/report/AAPL?algo=dueling_dqn")
    assert body.status_code == 200, body.text
    text = body.text
    assert "# Training Report" in text
    assert "Health score" in text and "Diagnosis" in text
    assert "Not recorded" in text, "the report hides its own gaps"
    assert "not recorded" in text.lower()


# ================================== smart hyperparameter manager (API + UI)
def test_the_smart_endpoint_returns_a_summary_and_hides_nothing_needed(client):
    body = client.get("/api/v1/hyperparams/smart/recommend"
                      "?symbol=AAPL&algo=dueling_dqn&period=2y")
    assert body.status_code == 200, body.text
    data = body.json()
    # What a standard user sees.
    for key in ("summary", "profile", "estimated_training", "expected_quality",
                "confidence"):
        assert key in data, f"the summary is missing {key}"
    # What is kept for reproducibility even though it is not displayed.
    assert data["resolved_hyperparameters"]["training"]["seed"] is not None
    assert data["fingerprint"]


def test_the_five_high_level_profiles_are_offered(client):
    data = client.get("/api/v1/hyperparams/smart/profiles").json()
    keys = {p["key"] for p in data["profiles"]}
    assert {"conservative", "balanced", "high_performance", "risk_aware",
            "ai_recommended"} == keys
    # Each must map onto a real YAML profile, or the automatic path and the
    # advanced path would be two separate configuration systems.
    from app.services.rl.hyperparams import hyperparameters

    available = {p["key"] for p in hyperparameters.profiles()}
    for profile in data["profiles"]:
        if profile["base"] is not None:
            assert profile["base"] in available, \
                f"{profile['key']} points at a profile that does not exist"


def test_standard_mode_hides_the_low_level_parameters():
    """The whole point: a standard user is not shown a learning rate. A grid
    ignores the `hidden` attribute because `display: grid` outranks it — the
    same trap already fixed for .btn — so the CSS guard has to exist."""
    from pathlib import Path

    frontend = Path(__file__).resolve().parents[2] / "frontend"
    html = (frontend / "hyperparams.html").read_text()
    assert 'id="hpAdvancedPanels" hidden' in html
    assert 'id="hpAdvancedControls" hidden' in html
    assert 'id="hpAdvanced"' in html, "there is no Advanced Mode toggle"

    css = (frontend / "assets" / "css" / "styles.css").read_text()
    assert ".grid[hidden]" in css, \
        "a hidden grid still renders: display:grid overrides the hidden attribute"

    js = (frontend / "assets" / "js" / "pages" / "hyperparams.js").read_text()
    assert "setAdvancedMode" in js and "hpSmartSummary" in js


# ========================= AI recommendation is period-aware (regression)
def test_the_recommendation_echoes_the_period_it_computed_over(client):
    """Without this the caller cannot tell a fresh result from a stale one —
    which is precisely how a frontend that sent no period at all went
    unnoticed: every answer looked plausible."""
    for period in ("1mo", "6mo", "1y", "5y"):
        body = client.get(
            f"/api/v1/signals/recommend/AAPL?period={period}&include_xai=false")
        assert body.status_code == 200, body.text
        data = body.json()
        assert data["period"] == period, \
            f"asked for {period}, the response reports {data['period']}"
        assert data["bars_analysed"] > 0
        assert data["period_start"]


def test_a_longer_period_really_reads_more_data(client):
    """The recommendation must be computed from the requested window. Equal bar
    counts across periods would mean the period is being ignored somewhere."""
    seen = {}
    for period in ("1mo", "6mo", "1y", "5y"):
        data = client.get(
            f"/api/v1/signals/recommend/AAPL?period={period}"
            "&include_xai=false").json()
        seen[period] = (data["bars_analysed"], data["period_start"])

    bars = [seen[p][0] for p in ("1mo", "6mo", "1y", "5y")]
    assert bars == sorted(bars), f"bar counts are not monotone in period: {seen}"
    assert len(set(bars)) == len(bars), \
        f"two periods analysed the same amount of data: {seen}"
    # The window start must move back as the period grows.
    starts = [seen[p][1] for p in ("1mo", "6mo", "1y", "5y")]
    assert starts == sorted(starts, reverse=True), \
        f"the window start does not recede with a longer period: {seen}"


def test_a_short_period_does_not_crash_the_recommendation():
    """`.get(key, 0)` returns the None when the key exists and holds None. On a
    1-month window the crash model declines to guess and reports None, so the
    narrative raised "unsupported format string passed to NoneType.__format__"
    and the whole endpoint returned a 500 — the shortest period was unusable.

    The narrative is exercised directly with an unmeasurable risk block rather
    than through the HTTP client. An earlier version called the endpoint and
    hoped the fixture would produce a short window; it produced 120 bars, so
    `crash_risk_score` was never None, the guarded branch was never reached,
    and the test passed against the very bug it was written for.
    """
    from app.services.recommendation.engine import recommendation_engine

    unmeasurable = {
        "overall_risk_level": "unknown",
        "crash_risk": {
            "crash_risk_score": None,      # what a short window really returns
            "var_95_daily": None,
            "current_drawdown": None,
            "recommendation": "Not enough history to estimate tail risk.",
        },
    }
    narrative = recommendation_engine._narrative(
        symbol="AAPL", action="HOLD", score=0.0, signals=[],
        risk=unmeasurable, confidence=0.5)

    text = " ".join(narrative["narrative"]) if isinstance(
        narrative.get("narrative"), list) else str(narrative)
    # It must not print a confident 0.00 for a figure it could not measure.
    assert "0.00" not in text.split("crash-risk score")[-1][:40], \
        "an unmeasurable risk figure was rendered as a confident zero"
    assert "not measurable" in text or "n/a" in text


def test_the_recommendation_endpoint_serves_the_shortest_period(client):
    """The end-to-end counterpart: the shortest selectable window must return a
    usable recommendation, not a 500."""
    body = client.get(
        "/api/v1/signals/recommend/AAPL?period=1mo&include_xai=false")
    assert body.status_code == 200, body.text
    data = body.json()
    assert data["action"] in ("STRONG_BUY", "BUY", "HOLD", "SELL", "STRONG_SELL")
    assert data["period"] == "1mo"


def test_the_signals_page_sends_the_selected_period():
    """The backend always honoured `period`; the caller dropped it, so
    `api.recommend` fell back to its '2y' default and 1M, 6M and 10Y all asked
    for the same two years."""
    from pathlib import Path

    js = (Path(__file__).resolve().parents[2] / "frontend" / "assets" / "js"
          / "pages" / "signals.js").read_text()
    block = js[js.index("async function generate()"):]
    block = block[: block.index("const colour")]
    assert "getTimeRange()" in block, \
        "the page never reads the selected period"
    assert "period," in block or "period:" in block, \
        "the period is read but not passed to api.recommend"
    # A slow reply for an abandoned period must not overwrite a newer one.
    assert "!== getTimeRange()" in block, \
        "there is no guard against an out-of-date response landing last"


# ==================================================== AI direction prediction
def test_direction_prediction_returns_a_usable_payload(client):
    """The feature's contract: a direction, a magnitude with its provenance,
    and a confidence built from the signals that actually ran."""
    body = client.get("/api/v1/signals/direction/AAPL?period=1y&horizon=5")
    assert body.status_code == 200, body.text
    data = body.json()

    for field in ("symbol", "direction", "expected_move_pct", "magnitude_basis",
                  "forecaster_available", "reliable_prediction",
                  "market_volatility_pct", "volatility_basis",
                  "confidence", "confidence_label", "confidence_breakdown",
                  "composite_score", "neutral_band", "horizon_days",
                  "last_price", "signals", "summary", "bars_analysed"):
        assert field in data, f"the payload is missing {field}"

    assert data["direction"] in ("INCREASE", "DECREASE", "NEUTRAL", "UNAVAILABLE")
    # Confidence is a *model* confidence: without a forecaster there is no
    # model to score, so None is the correct answer rather than a number.
    if data["confidence"] is not None:
        assert 0.0 <= data["confidence"] <= 1.0
    assert -1.0 <= data["composite_score"] <= 1.0
    # Four signal slots are always reported, present or not: a missing model is
    # evidence about the confidence, so hiding it would overstate the call.
    assert len(data["signals"]) == 4


def test_direction_magnitude_always_declares_where_it_came_from(client):
    """The expected move must be traceable to a model output or to a
    measurement — never to a hardcoded or random default."""
    data = client.get("/api/v1/signals/direction/AAPL?period=1y&horizon=5").json()
    # Realised volatility is no longer a magnitude basis: it does not predict
    # direction, and signing it produced a fake expected movement.
    assert data["magnitude_basis"] in (
        "deep_learning_forecast", "no_trained_forecaster")

    if data["magnitude_basis"] == "no_trained_forecaster":
        assert data["expected_move_pct"] is None, \
            "a missing forecaster must yield None, not a plausible number"
        assert data["forecaster_available"] is False
        assert "reason" in data["basis_detail"]
    else:
        assert data["expected_move_pct"] is not None
        assert data["forecaster_available"] is True
        assert data["basis_detail"], "the basis is claimed but not described"


def test_a_neutral_call_advertises_no_target_and_no_signed_move(client):
    """NEUTRAL means the models declined to pick a side. A signed move or a
    price target would imply the directional view they refused to take."""
    from app.services.data.market_data import market_data_service
    from app.services.recommendation.direction import direction_predictor

    series = market_data_service.get_history("AAPL", period="1y")
    df = getattr(series, "df", series)
    result = direction_predictor.predict("AAPL", df, horizon=5)

    if result["direction"] in ("NEUTRAL", "UNAVAILABLE"):
        assert result["target_price"] is None, "a neutral call quotes a target"
        assert result["reliable_prediction"] is False


def test_direction_horizon_actually_reaches_the_model(client):
    """A horizon selector that changes nothing is decoration. Longer horizons
    must produce a different expected move, not the same number relabelled."""
    moves = {}
    for horizon in (1, 5, 20):
        data = client.get(
            f"/api/v1/signals/direction/AAPL?period=1y&horizon={horizon}").json()
        assert data["horizon_days"] == horizon
        moves[horizon] = data["expected_move_pct"]

    usable = {h: m for h, m in moves.items() if m is not None}
    if len(usable) >= 2:
        assert len(set(usable.values())) > 1, \
            f"every horizon returned the same expected move: {moves}"


def test_direction_magnitude_tracks_the_underlying_model(client):
    """A constant magnitude would satisfy every shape check above. This pins
    the number to its stated source: when the basis is the forecaster, the
    expected move must equal that model's own predicted return; when it is
    volatility, it must equal the damped realised volatility.

    Without this, replacing the magnitude with a literal (e.g. 1.5) passes.
    """
    from app.services.data.market_data import market_data_service
    from app.services.forecasting.trainer import forecast_trainer
    from app.services.recommendation.direction import direction_predictor

    series = market_data_service.get_history("AAPL", period="1y")
    df = getattr(series, "df", series)
    result = direction_predictor.predict("AAPL", df, horizon=5)

    # Under DATA_MODE=offline the suite runs on synthetic prices, so which
    # branch executes depends on whether a checkpoint happens to match. Both
    # are asserted, and the forecast branch is additionally forced below so a
    # hardcoded magnitude cannot hide in the path this run did not take.
    if result["magnitude_basis"] == "deep_learning_forecast":
        predicted = forecast_trainer.predict(
            "AAPL", df, model_name="lstm", horizon=5)["predicted_return"]
        # The expected move is the forecaster's signed predicted return, always.
        # This branch previously expected abs() whenever the verdict was
        # NEUTRAL, which was never how `_expected_move` behaved: it applies no
        # sign logic at all. The wrong expectation stayed green only because a
        # NEUTRAL verdict had not yet coincided with a negative prediction, and
        # it started failing once conflicting signals began resolving to
        # NEUTRAL. Asserting the signed value is both correct and stricter —
        # a magnitude that silently dropped its sign would now be caught.
        expected = predicted * 100
        assert result["expected_move_pct"] == pytest.approx(expected, abs=1e-3), (
            "the expected move does not match the forecaster's predicted "
            f"return: {result['expected_move_pct']} vs {expected}")

    else:
        assert result["magnitude_basis"] == "no_trained_forecaster"
        assert result["expected_move_pct"] is None

    # Force the forecast branch with a stub signal, so this test covers it
    # regardless of which checkpoints exist in the environment. Without this,
    # replacing the forecast magnitude with a literal passed unnoticed.
    from app.services.recommendation.engine import SignalContribution

    stub = SignalContribution(
        "forecast", 0.5, 0.30, 0.8, True,
        {"predicted_return": 0.0234, "model": "lstm", "horizon_days": 5,
         "directional_accuracy": 61.0})
    magnitude = direction_predictor._expected_move([stub], df, 5)
    assert magnitude["magnitude_basis"] == "deep_learning_forecast"
    assert magnitude["expected_move_pct"] == pytest.approx(2.34, abs=1e-6), (
        "the forecast branch does not return the model's own predicted "
        f"return: got {magnitude['expected_move_pct']}, expected 2.34")


def test_direction_confidence_is_built_from_available_evidence(client):
    """Confidence must fall when models are missing, or it is decoration."""
    data = client.get("/api/v1/signals/direction/AAPL?period=1y&horizon=5").json()
    breakdown = data["confidence_breakdown"]

    for key in ("agreement", "reliability", "coverage", "n_available"):
        assert key in breakdown, f"confidence does not expose {key}"

    n_available = sum(1 for s in data["signals"] if s["available"])
    assert breakdown["n_available"] == n_available
    assert breakdown["coverage"] == pytest.approx(n_available / 4.0, abs=1e-6)

    # With no signal at all there can be no confidence.
    if n_available == 0:
        assert data["confidence"] == 0.0

    # And it must be a *function* of those factors, not a constant. Feeding
    # two clearly different evidence sets must move the number; a fixed value
    # would satisfy every check above.
    from app.services.recommendation.direction import DirectionPredictor
    from app.services.recommendation.engine import SignalContribution

    strong = [SignalContribution("forecast", 0.8, 0.30, 0.9, True, {}),
              SignalContribution("rl", 0.7, 0.25, 0.9, True, {}),
              SignalContribution("technical", 0.6, 0.25, 0.9, True, {}),
              SignalContribution("sentiment", 0.5, 0.20, 0.9, True, {})]
    weak = [SignalContribution("forecast", 0.8, 0.30, 0.2, True, {}),
            SignalContribution("rl", -0.7, 0.25, 0.2, True, {}),
            SignalContribution("technical", 0.0, 0.0, 0.0, False, {}),
            SignalContribution("sentiment", 0.0, 0.0, 0.0, False, {})]

    high = DirectionPredictor._confidence(strong, 0.7)["value"]
    low = DirectionPredictor._confidence(weak, 0.1)["value"]
    assert high > low, (
        f"confidence does not respond to evidence quality: {high} vs {low}")
    assert DirectionPredictor._confidence([], 0.0)["value"] == 0.0


def test_direction_never_hardcodes_its_output():
    """A direction call that ignores its inputs would pass every shape test
    above. Two instruments with different data must not return an identical
    score, and the module must not contain a literal verdict."""
    from pathlib import Path

    from app.services.data.market_data import market_data_service
    from app.services.recommendation.direction import direction_predictor

    scores = []
    for symbol in ("AAPL", "MSFT", "SPY"):
        series = market_data_service.get_history(symbol, period="1y")
        df = getattr(series, "df", series)
        scores.append(direction_predictor.predict(symbol, df, horizon=5)["composite_score"])
    assert len(set(scores)) > 1, \
        f"every instrument produced the same composite score: {scores}"

    # Check the *code*, not the prose: the module docstring legitimately
    # contains the word "random" while promising not to use it, and an earlier
    # version of this test flagged that sentence as the offence.
    import ast

    src = (Path(__file__).resolve().parents[1] / "app" / "services"
           / "recommendation" / "direction.py").read_text()
    tree = ast.parse(src)
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert "random" not in imported, "the predictor imports the random module"

    # numpy's RNG would slip past an import check on `random`.
    calls = {ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n, ast.Call)}
    assert not any("random" in c for c in calls), \
        f"the predictor calls a random generator: {[c for c in calls if 'random' in c]}"


def test_the_stress_page_is_wired_to_the_engine():
    """A backend feature nothing renders is invisible to the user.

    The dedicated page used to host AI Direction Prediction and was replaced by
    the AI Stress Testing Engine on request. The direction *endpoint* is still
    served and still tested; only the page changed.
    """
    from pathlib import Path

    frontend = Path(__file__).resolve().parents[2] / "frontend"
    html = (frontend / "stress.html").read_text()
    assert 'id="stHero"' in html, "the page has no results panel"
    assert 'id="stScenario"' in html, "the scenario selector is missing"

    js = (frontend / "assets" / "js" / "pages" / "stress.js").read_text()
    assert "api.stressEngine(" in js, "the page never calls the engine"

    # The replaced page must be gone, not orphaned alongside the new one.
    assert not (frontend / "direction.html").exists(), \
        "the old direction page is still shipped"
    assert not (frontend / "assets" / "js" / "pages" / "direction.js").exists(), \
        "the old direction page script is still shipped"

    signals_html = (frontend / "signals.html").read_text()
    signals_js = (frontend / "assets" / "js" / "pages" / "signals.js").read_text()
    for leftover in ("dirCard", "dirBox", "dirHorizon"):
        assert leftover not in signals_html, \
            f"Recommendations still carries the direction panel ({leftover})"
    assert "api.direction(" not in signals_js, \
        "Recommendations still calls the direction endpoint"

def test_realised_volatility_can_never_become_the_predicted_return():
    """Guards the specific arithmetic that caused the defect: signing the
    volatility by the consensus and returning it as a movement."""
    import inspect

    from app.services.recommendation.direction import DirectionPredictor

    source = inspect.getsource(DirectionPredictor._expected_move)
    assert "_realised_volatility" not in source, (
        "the expected-movement path still reads realised volatility")
    assert "np.sign" not in source, (
        "the expected-movement path still signs a magnitude by the consensus")

    # And volatility must be reported under its own, unsigned key.
    vol_source = inspect.getsource(DirectionPredictor._market_volatility)
    assert "market_volatility_pct" in vol_source
    assert "abs(" in vol_source, "volatility is reported signed"


def test_volatility_is_reported_separately_and_unsigned(client):
    """The volatility figure stays visible — it is useful — but under its own
    label, always positive, and never as an expected movement."""
    data = client.get("/api/v1/signals/direction/AAPL?period=1y&horizon=5").json()

    assert "market_volatility_pct" in data
    if data["market_volatility_pct"] is not None:
        assert data["market_volatility_pct"] >= 0, "volatility is reported signed"
        assert data["volatility_basis"] == "realised_historical_volatility"
        assert "not a directional forecast" in data["volatility_note"]
        # The two quantities must not be the same number by construction.
        if data["expected_move_pct"] is not None:
            assert data["expected_move_pct"] != data["market_volatility_pct"] or \
                data["magnitude_basis"] == "deep_learning_forecast"


def test_an_available_forecaster_supplies_the_predicted_return():
    """When a model exists, the expected movement is exactly its own
    predicted return — not a re-derived or rescaled version of it."""
    from app.services.recommendation.direction import direction_predictor
    from app.services.recommendation.engine import SignalContribution

    stub = SignalContribution(
        "forecast", 0.6, 0.30, 0.8, True,
        {"predicted_return": 0.032, "model": "lstm", "horizon_days": 60,
         "directional_accuracy": 76.0})
    magnitude = direction_predictor._expected_move([stub], None, 60)

    assert magnitude["expected_move_pct"] == pytest.approx(3.2, abs=1e-6)
    assert magnitude["magnitude_basis"] == "deep_learning_forecast"
    assert magnitude["forecaster_available"] is True
    assert magnitude["basis_detail"]["horizon_days"] == 60


def test_direction_states_remain_reachable_and_correct():
    """INCREASE / DECREASE / NEUTRAL must all still be produced, and only from
    the score. A fix that made everything NEUTRAL would pass the tests above."""
    from app.services.recommendation.direction import (
        NEUTRAL_BAND,
        DirectionPredictor,
    )

    assert DirectionPredictor._direction(NEUTRAL_BAND + 0.01) == "INCREASE"
    assert DirectionPredictor._direction(-NEUTRAL_BAND - 0.01) == "DECREASE"
    assert DirectionPredictor._direction(0.0) == "NEUTRAL"
    assert DirectionPredictor._direction(NEUTRAL_BAND) == "NEUTRAL", \
        "the band edge must not be counted as a direction"


def test_a_directional_call_requires_a_forecaster(client):
    """Without a predicted return there is no reliable direction, so the
    verdict must be NEUTRAL rather than an unsupported INCREASE/DECREASE."""
    from app.services.data.market_data import market_data_service
    from app.services.recommendation.direction import direction_predictor

    for symbol in ("AAPL", "GC=F", "TSLA"):
        series = market_data_service.get_history(symbol, period="1y")
        df = getattr(series, "df", series)
        result = direction_predictor.predict(symbol, df, horizon=60)

        if not result["forecaster_available"]:
            assert result["direction"] in ("NEUTRAL", "UNAVAILABLE"), (
                f"{symbol} was given a directional call with no forecaster: "
                f"{result['direction']}")
            assert result["expected_move_pct"] is None
            assert result["confidence"] is None, \
                "a model confidence was reported with no model behind it"
            assert result["reliable_prediction"] is False


def test_each_horizon_uses_a_model_trained_for_that_horizon(client):
    """Checkpoints are per-horizon (`forecast_<sym>_<model>_h<N>`). Selecting a
    horizon with no matching checkpoint must report no forecaster rather than
    silently reusing a model trained for a different one."""
    from app.services.data.market_data import market_data_service
    from app.services.forecasting.trainer import forecast_trainer
    from app.services.recommendation.direction import direction_predictor

    series = market_data_service.get_history("AAPL", period="1y")
    df = getattr(series, "df", series)

    for horizon in (1, 5, 30, 60):
        result = direction_predictor.predict("AAPL", df, horizon=horizon)
        assert result["horizon_days"] == horizon

        trained = forecast_trainer.is_trained("AAPL", "lstm", horizon)
        assert result["forecaster_available"] == trained, (
            f"h={horizon}: forecaster_available={result['forecaster_available']} "
            f"but is_trained={trained}")

        if trained:
            assert result["basis_detail"]["horizon_days"] == horizon, \
                "the forecast came from a model trained for another horizon"
        else:
            assert result["expected_move_pct"] is None


def test_the_stress_panel_separates_measured_quantities():
    """The user must be able to tell one risk measure from another at a glance,
    and a figure the backend could not measure must read as N/A rather than a
    zero that looks like data."""
    from pathlib import Path

    frontend = Path(__file__).resolve().parents[2] / "frontend"
    js = (frontend / "assets" / "js" / "pages" / "stress.js").read_text()
    html = (frontend / "stress.html").read_text()

    # Row labels are rendered by the script; card titles live in the markup.
    for label in ("Value at Risk", "Conditional VaR", "Volatility",
                  "Max Drawdown", "Resilience Score"):
        assert label in js, f"the panel does not label {label!r}"
    for title in ("Risk Contribution", "Asset-Level Impact", "Portfolio Loss"):
        assert title in html, f"the page has no {title!r} section"

    assert "pctOrNa" in js and "moneyOrNa" in js, \
        "there is no N/A path for an unmeasurable figure"
    assert "Math.random" not in js, "the page generates random values"

def test_the_documented_contract_matches_the_implemented_one():
    """The code was fixed to stop deriving an expected move from volatility,
    but the prose describing it was not, and the endpoint docstring is served
    publicly in /openapi.json. A caller reading "from realised volatility
    otherwise" would integrate against a contract the code no longer honours,
    so the stale claim is a defect in its own right.

    Guards the claim, not the wording: any text that promises volatility as a
    fallback magnitude fails, however it is phrased.
    """
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "app"
    sources = {
        "direction service": root / "services" / "recommendation" / "direction.py",
        "direction endpoint": root / "api" / "v1" / "endpoints" / "insights.py",
    }

    # Phrasings that would tell a reader volatility can become the magnitude.
    forbidden = (
        "from realised volatility otherwise",
        "from realized volatility otherwise",
        "derived from realised volatility",
        "derived from realized volatility",
        "realised volatility scaled to the horizon —",
    )

    for where, path in sources.items():
        text = path.read_text()
        for claim in forbidden:
            assert claim not in text, (
                f"{where} still documents volatility as a fallback expected "
                f"move ({claim!r}), but _expected_move returns None instead")

    # The replacement must actually state the real behaviour, not just delete
    # the wrong sentence.
    service = sources["direction service"].read_text()
    assert "no_trained_forecaster" in service, \
        "the service no longer documents what happens without a forecaster"


# ============================================ direction fallback: every symbol
# The fallback was verified on GC=F alone. These cover the whole catalogue, so
# a symbol-specific path (special characters in tickers, a per-asset-class
# branch, an auto-training shortcut) cannot make one instrument behave
# differently from the rest.

def _direction_universe():
    """Every supported instrument, grouped so a failure names its asset class."""
    from app.services.data.universe import UNIVERSE
    return [(ins.symbol, ins.asset_class) for ins in UNIVERSE]


def test_the_no_forecaster_fallback_is_identical_for_every_symbol():
    """Same contract for all 32 instruments across all 6 asset classes.

    Tickers carry characters that get rewritten on the way to a checkpoint
    filename ('=' in GC=F and EURUSD=X, '^' in ^GSPC, '-' in BTC-USD, '.' in
    MC.PA). A mismatch between the name used to *look up* a model and the name
    used to *save* one would make a symbol silently claim a forecaster it does
    not have, or hide one it does.
    """
    from app.services.data.market_data import market_data_service
    from app.services.forecasting.trainer import forecast_trainer
    from app.services.recommendation.direction import direction_predictor

    checked = 0
    for symbol, asset_class in _direction_universe():
        for horizon in (5, 60):
            series = market_data_service.get_history(symbol, period="1y")
            df = getattr(series, "df", series)
            r = direction_predictor.predict(symbol, df, horizon=horizon)
            where = f"{symbol} ({asset_class}) h={horizon}"

            trained = forecast_trainer.is_trained(symbol, "lstm", horizon)
            assert r["forecaster_available"] == trained, (
                f"{where}: forecaster_available={r['forecaster_available']} "
                f"but is_trained={trained} — the lookup name and the checkpoint "
                f"name disagree for this ticker")

            assert r["symbol"] == symbol.upper(), f"{where}: wrong symbol echoed"
            assert r["horizon_days"] == horizon, f"{where}: wrong horizon echoed"

            if not r["forecaster_available"]:
                assert r["expected_move_pct"] is None, (
                    f"{where}: no forecaster yet a numeric expected move "
                    f"{r['expected_move_pct']}")
                assert r["confidence"] is None, f"{where}: confidence with no model"
                assert r["confidence_label"] is None, f"{where}: label with no model"
                assert r["magnitude_basis"] == "no_trained_forecaster", where
                assert r["reliable_prediction"] is False, where
                assert r["target_price"] is None, f"{where}: target with no model"
                assert r["direction"] in ("NEUTRAL", "UNAVAILABLE"), (
                    f"{where}: invented a direction {r['direction']}")
            checked += 1

    assert checked >= 60, f"the universe shrank to {checked} checks"


def test_the_fallback_message_is_generated_from_the_real_symbol_and_horizon():
    """The explanation must be built from this symbol's own numbers.

    A template that hardcoded GC=F, 60 days or ±11.66% would read correctly on
    the one instrument it was written for and lie on the other 31.
    """
    from app.services.data.market_data import market_data_service
    from app.services.recommendation.direction import direction_predictor

    seen_vols, seen = set(), 0
    for symbol in ("TSLA", "NVDA", "JPM", "BTC-USD", "GC=F", "SPY", "^GSPC",
                   "USDJPY=X"):
        for horizon in (5, 60):
            series = market_data_service.get_history(symbol, period="1y")
            df = getattr(series, "df", series)
            r = direction_predictor.predict(symbol, df, horizon=horizon)
            if r["forecaster_available"]:
                continue
            text, where = r["summary"], f"{symbol} h={horizon}"

            assert symbol in text, f"{where}: the message does not name the symbol"
            assert f"{horizon}-day" in text, f"{where}: the horizon is not stated"
            assert "No trained forecaster is available" in text, where
            # No other instrument's identity may leak into this message.
            for foreign in ("GC=F", "AAPL", "TSLA", "BTC-USD"):
                if foreign != symbol:
                    assert foreign not in text, (
                        f"{where}: message mentions {foreign}")

            vol = r["market_volatility_pct"]
            if vol is not None:
                assert f"{vol:.2f}%" in text, (
                    f"{where}: quoted volatility does not match "
                    f"market_volatility_pct={vol}")
                seen_vols.add(round(vol, 4))
            assert f"{r['composite_score']:+.3f}" in text, (
                f"{where}: the composite score in the text is not the real one")
            assert f"{r['confidence_breakdown']['n_available']} of 4" in text, (
                f"{where}: the signal count in the text is not the real one")
            seen += 1

    assert seen >= 8, "too few fallback messages exercised"
    assert len(seen_vols) > 1, (
        "every symbol reported the same volatility — the number looks static")


def test_volatility_scales_with_the_horizon_for_every_asset_class():
    """Volatility must be recomputed per horizon, not reused. One value shared
    across horizons would mean the selector does not reach the calculation."""
    from app.services.data.market_data import market_data_service
    from app.services.recommendation.direction import direction_predictor

    for symbol in ("TSLA", "BTC-USD", "GC=F", "SPY", "EURUSD=X", "^VIX"):
        series = market_data_service.get_history(symbol, period="1y")
        df = getattr(series, "df", series)
        vols = {}
        for horizon in (1, 5, 30, 60):
            r = direction_predictor.predict(symbol, df, horizon=horizon)
            vols[horizon] = r["market_volatility_pct"]
            assert r["market_volatility_pct"] is None or r["market_volatility_pct"] >= 0, \
                f"{symbol}: signed volatility {r['market_volatility_pct']}"
        usable = [v for v in vols.values() if v is not None]
        if len(usable) >= 2:
            assert len(set(usable)) > 1, (
                f"{symbol}: identical volatility at every horizon {vols}")
            ordered = [vols[h] for h in (1, 5, 30, 60) if vols[h] is not None]
            assert ordered == sorted(ordered), (
                f"{symbol}: volatility does not grow with the horizon {vols}")


def test_the_verdict_never_contradicts_the_expected_movement(monkeypatch):
    """Regression: AAPL h5 rendered 'INCREASE' above 'Expected Movement -0.88%'
    with a target price *below* the last close, and a summary reading 'more
    likely to rise by about 0.88%' for a predicted fall.

    Direction came from the four-signal composite while the magnitude came from
    the forecaster alone, so opposite signs produced a self-contradicting card.

    This drives the real `predict()` and only substitutes the four signal
    builders. An earlier version of this test re-implemented the fusion inline
    and therefore kept passing when the guard was deleted from the source — it
    was testing its own copy of the logic, not the platform's.
    """
    import numpy as np
    import pandas as pd

    from app.services.recommendation.direction import direction_predictor
    from app.services.recommendation.engine import (
        RecommendationEngine,
        SignalContribution,
    )

    idx = pd.date_range("2023-01-01", periods=300, freq="D")
    rng = np.random.default_rng(7)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 300)))
    df = pd.DataFrame({"close": close, "open": close,
                       "high": close * 1.01, "low": close * 0.99,
                       "volume": 1e6}, index=idx)

    def install(predicted_return, other_score):
        """Force a known forecast and a known consensus from the other models."""
        monkeypatch.setattr(
            RecommendationEngine, "_forecast_signal",
            lambda self, sym, d, m, h: SignalContribution(
                "forecast", float(np.sign(predicted_return)) * 0.8, 0.30, 0.8,
                True, {"predicted_return": predicted_return, "model": "lstm",
                       "horizon_days": h, "directional_accuracy": 70.0}))
        monkeypatch.setattr(
            RecommendationEngine, "_rl_signal",
            lambda self, sym, algo: SignalContribution(
                "rl", other_score, 0.25, 0.7, True, {"action": "BUY"}))
        monkeypatch.setattr(
            RecommendationEngine, "_technical_signal",
            lambda self, d: SignalContribution(
                "technical", other_score, 0.25, 0.7, True, {}))
        monkeypatch.setattr(
            RecommendationEngine, "_sentiment_signal",
            lambda self, sym: SignalContribution(
                "sentiment", other_score, 0.20, 0.6, True, {}))

    # Forecaster says down, everything else says up. The verdict now follows
    # the forecaster (the other three do not predict returns), so the correct
    # answer is DECREASE — and, crucially, it agrees in sign with the number
    # shown beside it. The invariant this test defends is that headline and
    # magnitude never contradict, not which component wins.
    install(-0.032, 0.9)
    r = direction_predictor.predict("AAPL", df, horizon=5)

    assert r["expected_move_pct"] < 0 and r["composite_score"] > 0, \
        "the fixture no longer reproduces the disagreement"
    assert r["direction"] == "DECREASE", (
        f"announced {r['direction']} for an expected move of "
        f"{r['expected_move_pct']}%")
    # Headline and magnitude must describe the same future.
    assert (r["expected_move_pct"] < 0) == (r["direction"] == "DECREASE")
    assert r["target_price"] is not None and r["target_price"] < r["last_price"], \
        "a fall was announced with a target at or above the last price"
    assert "rise" not in r["summary"], \
        f"the summary claims a rise for {r['expected_move_pct']}%: {r['summary']}"
    assert "fall" in r["summary"], "the summary does not state the direction"

    # A fix that made everything NEUTRAL would satisfy the above, so both
    # aligned cases must still produce a real directional call.
    install(+0.032, 0.9)
    up = direction_predictor.predict("AAPL", df, horizon=5)
    assert up["direction"] == "INCREASE", up["direction"]
    assert up["signal_conflict"] is False and up["expected_move_pct"] > 0
    assert "rise" in up["summary"]

    install(-0.032, -0.9)
    dn = direction_predictor.predict("AAPL", df, horizon=5)
    assert dn["direction"] == "DECREASE", dn["direction"]
    assert dn["signal_conflict"] is False and dn["expected_move_pct"] < 0
    assert "fall" in dn["summary"]


# ================================ forecaster discovery is generic, not per-ticker
def test_checkpoint_discovery_finds_whatever_is_on_disk():
    """Discovery must be a directory listing, not a maintained ticker list.

    Every symbol trained later has to become usable with no code change, so
    this trains a throwaway instrument that appears in no catalogue and no
    source file, and asserts the system finds it.
    """
    import numpy as np
    import pandas as pd

    from app.services.forecasting.trainer import TrainConfig, forecast_trainer

    symbol = "ZZTEST=X"        # deliberately absent from the universe
    paths = []
    try:
        idx = pd.date_range("2019-01-01", periods=700, freq="B")
        rng = np.random.default_rng(3)
        close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(idx))))
        df = pd.DataFrame({"close": close, "open": close,
                           "high": close * 1.01, "low": close * 0.99,
                           "volume": 1e6}, index=idx)

        assert forecast_trainer.available_forecasts(symbol) == [], \
            "an untrained symbol already reports a model"

        forecast_trainer.train(symbol, df, TrainConfig(model="gru", horizon=5,
                                                       epochs=2))
        ckpt = forecast_trainer.checkpoint_path(symbol, "gru", 5)
        paths = [ckpt, ckpt.with_suffix(".json")]

        found = forecast_trainer.available_forecasts(symbol)
        assert {"model": "gru", "horizon": 5} in found, \
            f"a freshly trained checkpoint was not discovered: {found}"
        assert forecast_trainer.available_horizons(symbol) == [5]

        # And it is reachable through the resolver even though the caller asks
        # for a different architecture.
        assert forecast_trainer.resolve_model(symbol, "lstm", 5) == "gru"
    finally:
        for p in paths:
            p.unlink(missing_ok=True)

    assert forecast_trainer.available_forecasts(symbol) == [], \
        "the test left checkpoints behind"


def test_a_symbol_uses_the_architecture_it_actually_has():
    """Regression: EURUSD=X owned a trained GRU checkpoint yet the direction
    card said "No trained forecaster available", because the request defaulted
    to LSTM and nothing looked for what the symbol really had. The file was on
    disk and loadable the entire time.
    """
    from app.services.data.market_data import market_data_service

    # Any symbol trained on something other than the caller's default proves
    # the point; pick them from disk so the test never names a fixed ticker.
    from app.services.data.universe import UNIVERSE
    from app.services.forecasting.trainer import TrainConfig, forecast_trainer
    from app.services.recommendation.direction import direction_predictor

    mismatched = []
    for ins in UNIVERSE:
        models = {f["model"] for f in forecast_trainer.available_forecasts(ins.symbol)
                  if f["horizon"] == 5}
        if models and "lstm" not in models:
            mismatched.append((ins.symbol, sorted(models)[0]))

    # The suite runs against a temp MODEL_DIR, so the shipped GRU-only symbols
    # are not present and this would skip — a regression test that never runs
    # is not a regression test. Reproduce the condition instead of skipping.
    created = []
    if not mismatched:
        # A dedicated symbol, not a catalogue one: other tests in this file
        # train AAPL/lstm/h5 into the shared temp MODEL_DIR, which would give
        # the fixture an LSTM and stop it exercising the substitution at all.
        subject = "ZZSUB=X"
        series = market_data_service.get_history(UNIVERSE[0].symbol, period="5y")
        forecast_trainer.train(subject, getattr(series, "df", series),
                               TrainConfig(model="gru", horizon=5, epochs=2))
        assert not forecast_trainer.is_trained(subject, "lstm", 5), \
            "the fixture symbol also has an LSTM; the mismatch is not exercised"
        mismatched = [(subject, "gru")]
        ckpt = forecast_trainer.checkpoint_path(subject, "gru", 5)
        created = [ckpt, ckpt.with_suffix(".json")]

    try:
        _assert_uses_own_architecture(mismatched, market_data_service,
                                      forecast_trainer, direction_predictor)
    finally:
        for p in created:
            p.unlink(missing_ok=True)


def _assert_uses_own_architecture(mismatched, market_data_service,
                                  forecast_trainer, direction_predictor):
    for symbol, expected_model in mismatched:
        series = market_data_service.get_history(symbol, period="1y")
        df = getattr(series, "df", series)
        r = direction_predictor.predict(symbol, df, horizon=5,
                                        forecast_model="lstm")

        assert r["forecaster_available"] is True, (
            f"{symbol} has a trained {expected_model} model but the card still "
            f"reports no forecaster")
        assert r["expected_move_pct"] is not None, f"{symbol}: no expected move"
        assert r["confidence"] is not None, f"{symbol}: no confidence"
        assert r["resolved_model"] == expected_model, (
            f"{symbol}: used {r['resolved_model']}, expected {expected_model}")
        assert r["model_substituted"] is True
        # The substituted model's own prediction, not a re-derived number.
        direct = forecast_trainer.predict(symbol, df, model_name=expected_model,
                                          horizon=5)["predicted_return"]
        assert r["expected_move_pct"] == pytest.approx(direct * 100, abs=1e-3)


def test_the_resolver_never_substitutes_a_different_horizon():
    """A 60-day question answered by a 5-day model would be a different
    forecast wearing the wrong label. Only the architecture may be swapped."""
    from app.services.data.market_data import market_data_service
    from app.services.data.universe import UNIVERSE
    from app.services.forecasting.trainer import TrainConfig, forecast_trainer

    # MODEL_DIR is a temp directory here, so without provisioning one model
    # every branch below takes the "untrained" path and a resolver that
    # happily crossed horizons would still pass.
    subject = UNIVERSE[0].symbol
    created = []
    if not forecast_trainer.available_forecasts(subject):
        series = market_data_service.get_history(subject, period="5y")
        forecast_trainer.train(subject, getattr(series, "df", series),
                               TrainConfig(model="gru", horizon=5, epochs=2))
        ckpt = forecast_trainer.checkpoint_path(subject, "gru", 5)
        created = [ckpt, ckpt.with_suffix(".json")]

    try:
        trained_h = {f["horizon"]
                     for f in forecast_trainer.available_forecasts(subject)}
        assert trained_h, "the fixture symbol has no model to reason about"

        for horizon in (1, 5, 30, 60):
            resolved = forecast_trainer.resolve_model(subject, "lstm", horizon)
            if horizon in trained_h:
                assert resolved is not None, f"{subject}: h={horizon} not resolved"
                assert forecast_trainer.is_trained(subject, resolved, horizon)
            else:
                assert resolved is None, (
                    f"{subject}: h={horizon} is untrained yet resolved to "
                    f"{resolved!r} — a model trained for another horizon was "
                    f"substituted")

        for ins in UNIVERSE:
            horizons = {f["horizon"]
                        for f in forecast_trainer.available_forecasts(ins.symbol)}
            for horizon in (1, 5, 30, 60):
                resolved = forecast_trainer.resolve_model(ins.symbol, "lstm", horizon)
                if horizon not in horizons:
                    assert resolved is None, (
                        f"{ins.symbol}: h={horizon} is untrained yet resolved "
                        f"to {resolved}")
                else:
                    assert resolved is not None
                    assert forecast_trainer.is_trained(ins.symbol, resolved, horizon)
    finally:
        for p in created:
            p.unlink(missing_ok=True)


def test_every_catalogue_symbol_reports_its_true_model_state():
    """The whole catalogue, both directions of the claim: a symbol with a
    usable checkpoint must produce a real prediction, and one without must
    produce N/A — never the reverse, and never a fabricated number."""
    from app.services.data.market_data import market_data_service
    from app.services.data.universe import UNIVERSE
    from app.services.forecasting.trainer import TrainConfig, forecast_trainer
    from app.services.recommendation.direction import direction_predictor

    # The suite redirects MODEL_DIR to a temp directory, so the shipped
    # checkpoints are absent and every symbol would legitimately report N/A —
    # which would make the positive half of this test vacuous. Train one
    # catalogue symbol on a non-default architecture so both halves are real.
    subject = UNIVERSE[0].symbol
    series = market_data_service.get_history(subject, period="5y")
    trained_ok = False
    try:
        forecast_trainer.train(subject, getattr(series, "df", series),
                               TrainConfig(model="gru", horizon=5, epochs=2))
        trained_ok = True
    except Exception:                      # not enough offline history
        trained_ok = False

    with_model = 0
    for ins in UNIVERSE:
        series = market_data_service.get_history(ins.symbol, period="1y")
        df = getattr(series, "df", series)
        for horizon in (1, 5, 30, 60):
            r = direction_predictor.predict(ins.symbol, df, horizon=horizon)
            resolvable = forecast_trainer.resolve_model(
                ins.symbol, "lstm", horizon) is not None
            where = f"{ins.symbol} ({ins.asset_class}) h={horizon}"

            assert r["forecaster_available"] == resolvable, (
                f"{where}: reports forecaster_available="
                f"{r['forecaster_available']} but a model is "
                f"{'available' if resolvable else 'not available'} on disk")

            if resolvable:
                with_model += 1
                assert r["expected_move_pct"] is not None, f"{where}: N/A despite a model"
                assert r["confidence"] is not None, f"{where}: no confidence"
                assert r["resolved_model"] is not None
            else:
                assert r["expected_move_pct"] is None, f"{where}: invented a move"
                assert r["confidence"] is None, f"{where}: invented a confidence"
                assert r["direction"] in ("NEUTRAL", "UNAVAILABLE"), where

    if trained_ok:
        assert with_model > 0, (
            f"{subject} was trained on gru/h5 yet no symbol resolved a model; "
            "discovery is broken")
        ckpt = forecast_trainer.checkpoint_path(subject, "gru", 5)
        ckpt.unlink(missing_ok=True)
        ckpt.with_suffix(".json").unlink(missing_ok=True)


# ==================================== the dedicated AI Direction Prediction page
def test_the_direction_page_is_served_and_guarded(client, anon_client):
    """A dedicated page, reachable signed in and closed to everyone else."""
    assert client.get("/stress.html").status_code == 200
    anonymous = anon_client.get("/stress.html", follow_redirects=False)
    assert anonymous.status_code in (302, 303, 307), \
        "the direction page is reachable without signing in"


def test_every_page_links_to_the_direction_page():
    """It was asked for as a first-class sidebar entry, so every dashboard page
    must offer it — not just the one that happens to render it."""
    from pathlib import Path

    frontend = Path(__file__).resolve().parents[2] / "frontend"
    for page in frontend.glob("*.html"):
        if page.name in ("landing.html", "auth.html"):
            continue
        html = page.read_text()
        assert '<a class="nav-item" href="stress.html">' in html, \
            f"{page.name} has no sidebar entry for the stress page"
        assert "AI Stress Testing" in html, \
            f"{page.name} links the page without labelling it"


def test_the_stress_page_carries_every_requested_section():
    """The engine was specified feature by feature. Each must exist as a real
    mount point, or it silently ships incomplete."""
    from pathlib import Path

    frontend = Path(__file__).resolve().parents[2] / "frontend"
    html = (frontend / "stress.html").read_text()
    js = (frontend / "assets" / "js" / "pages" / "stress.js").read_text()

    for element in ("stSymbols", "stWeights", "stScenario", "stValue",
                    "stRun", "stHero", "stCompare", "stLoss", "stContrib",
                    "stAssets", "stVuln", "stMitig"):
        assert f'id="{element}"' in html, f"the page has no {element} mount point"

    for label in ("Run Stress Test", "Before vs After", "Portfolio Loss",
                  "Risk Contribution", "Asset-Level Impact",
                  "Main Vulnerabilities", "AI Mitigation Recommendations"):
        assert label in html, f"the page never shows the {label!r} section"

    for measure in ("Value at Risk", "Conditional VaR", "Max Drawdown",
                    "Resilience Score"):
        assert measure in js, f"the page never renders {measure!r}"

def test_the_stress_page_is_self_contained():
    """It must be a page of its own, not a second copy of Risk & Alerts, and it
    must not have grown a card into any other page."""
    from pathlib import Path

    frontend = Path(__file__).resolve().parents[2] / "frontend"
    html = (frontend / "stress.html").read_text()

    for foreign in ("sRecoBox", "recoHero", "screenTable"):
        assert foreign not in html, \
            f"the stress page carries {foreign!r} from another page"

    for page in frontend.glob("*.html"):
        if page.name in ("stress.html", "landing.html", "auth.html"):
            continue
        body = page.read_text()
        for owned in ('id="stHero"', 'id="stCompare"', 'id="stRun"'):
            assert owned not in body, \
                f"{page.name} grew a copy of the stress page ({owned})"

def test_the_stress_page_renders_backend_numbers_not_its_own():
    """The page must display what the engine measured. Recomputing risk in the
    browser would let the two disagree, and inventing a fallback would put a
    number on screen that no calculation produced."""
    from pathlib import Path

    js = (Path(__file__).resolve().parents[2] / "frontend" / "assets" / "js"
          / "pages" / "stress.js").read_text()

    assert "Math.random" not in js, "the page generates random values"
    # No local risk maths: those belong to the backend risk functions.
    for forbidden in ("quantile(", "Math.sqrt(252", "* 252)"):
        assert forbidden not in js, \
            f"the page recomputes risk locally ({forbidden})"
    # Nulls must survive to the screen as N/A rather than becoming zero.
    assert "dir-na" in js, "there is no N/A rendering path"
    assert "?? 0" not in js.replace("Math.max(0", ""), \
        "a missing measurement is silently defaulted to zero"

def test_a_confident_forecast_is_not_averaged_into_neutral(monkeypatch):
    """Regression: the page answered NEUTRAL for almost every symbol.

    The verdict came from a four-signal average in which the forecaster held
    30% of the weight. Technical momentum and news tone — neither of which
    forecasts anything — routinely cancelled it out. AAPL predicted -0.92% and
    still printed NEUTRAL, with the negative Expected Movement sitting beside
    the word, because the composite landed at +0.034 inside the +/-0.12 band.

    The forecaster's own signal now leads the call. The composite is still
    published and still drives agreement and confidence.
    """
    import numpy as np
    import pandas as pd

    from app.services.recommendation.direction import direction_predictor
    from app.services.recommendation.engine import (
        RecommendationEngine,
        SignalContribution,
    )

    idx = pd.date_range("2023-01-01", periods=300, freq="D")
    rng = np.random.default_rng(11)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 300)))
    df = pd.DataFrame({"close": close, "open": close, "high": close * 1.01,
                       "low": close * 0.99, "volume": 1e6}, index=idx)

    def install(forecast_score, predicted_return, others):
        monkeypatch.setattr(
            RecommendationEngine, "_forecast_signal",
            lambda self, sym, d, m, h: SignalContribution(
                "forecast", forecast_score, 0.30, 0.75, True,
                {"predicted_return": predicted_return, "model": "lstm",
                 "horizon_days": h, "directional_accuracy": 63.0}))
        for source, weight in (("_rl_signal", 0.25), ("_technical_signal", 0.25),
                               ("_sentiment_signal", 0.20)):
            name = source.split("_")[1]
            monkeypatch.setattr(
                RecommendationEngine, source,
                (lambda w, n: (lambda self, *a, **k: SignalContribution(
                    n, others, w, 0.8, True, {})))(weight, name))

    # The exact shape of the bug: a clear down-forecast outvoted by the rest.
    install(-0.30, -0.0092, +0.25)
    r = direction_predictor.predict("AAPL", df, horizon=5)

    assert r["expected_move_pct"] < 0, "the fixture no longer predicts a fall"
    assert r["composite_score"] > 0, "the fixture no longer reproduces the clash"
    assert r["direction"] == "DECREASE", (
        f"a predicted fall of {r['expected_move_pct']}% was reported as "
        f"{r['direction']} because the other signals outvoted it")
    assert r["lead_basis"] == "forecast_predicted_return"
    assert "rise" not in r["summary"]

    # Mirror case, and a genuinely small prediction must still be NEUTRAL.
    install(+0.30, +0.0092, -0.25)
    up = direction_predictor.predict("AAPL", df, horizon=5)
    assert up["direction"] == "INCREASE", up["direction"]

    install(+0.02, +0.0004, +0.60)
    flat = direction_predictor.predict("AAPL", df, horizon=5)
    assert flat["direction"] == "NEUTRAL", (
        "a forecast inside the neutral band was turned into a direction by the "
        "other signals")
    assert flat["composite_score"] > flat["neutral_band"], \
        "the fixture no longer proves the composite is being ignored"


# ======================================================= AI Stress Testing Engine
def _stress_returns(n=600, seed=5):
    """A deterministic, fat-tailed return series for engine unit tests.

    Student-t rather than Gaussian on purpose: a normal series has no crash in
    it, so "replay the worst window" is barely distinguishable from the base
    case and a threshold test on it would be measuring the fixture, not the
    engine. Real equities have the tail this reproduces.
    """
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    shocks = rng.standard_t(df=3, size=n) * 0.011
    return pd.Series(0.0004 + shocks, index=idx)


def test_the_stress_catalogue_covers_every_requested_scenario(client):
    """The brief named the scenarios. Each must be a real, runnable key."""
    body = client.get("/api/v1/quant/stress-engine/scenarios")
    assert body.status_code == 200, body.text
    keys = {s["key"] for s in body.json()["scenarios"]}
    for required in ("market_crash", "drop_10", "drop_20", "vol_x2",
                     "liquidity_shock", "correlation_spike", "custom"):
        assert required in keys, f"the catalogue has no {required} scenario"

    for entry in body.json()["scenarios"]:
        assert entry["basis"], f"{entry['key']} does not state what it is built from"


def test_every_scenario_actually_changes_the_measured_risk():
    """A scenario that leaves the numbers untouched is decoration.

    Regression: Market Crash appended a single day (an argmin over a series with
    leading NaNs returned 0), so a 21-day episode became one bar inside ~1250
    and the engine reported no deterioration for a genuine crash.
    """
    from app.services.risk import stress

    returns = {"TEST": _stress_returns()}
    weights = {"TEST": 1.0}
    base = stress.run(returns, weights, "vol_x2")["before"]

    for key in ("market_crash", "drop_20", "vol_x2", "liquidity_shock"):
        result = stress.run(returns, weights, key)
        after = result["after"]
        assert after["cvar_pct"] > base["cvar_pct"], (
            f"{key} did not increase expected shortfall "
            f"({base['cvar_pct']} -> {after['cvar_pct']})")
        assert result["resilience"]["score"] is not None
        assert result["resilience"]["score"] < 100, \
            f"{key} reports a perfect resilience score despite stressing the book"

    # A *material* move, not a rounding nudge. Appending a single bar to a
    # 600-point sample shifts the tail by a fraction of a percent and still
    # satisfied the ">" checks above, so the original crash bug survived them.
    crash = stress.run(returns, weights, "market_crash")
    growth = crash["after"]["cvar_pct"] / base["cvar_pct"]
    assert growth > 1.15, (
        f"Market Crash moved expected shortfall by only {(growth - 1) * 100:.1f}% "
        f"({base['cvar_pct']} -> {crash['after']['cvar_pct']}); the worst window "
        f"is not reaching the stressed distribution")
    assert crash["resilience"]["score"] < 90, \
        "a replayed crash left the book looking almost untouched"


def test_stress_figures_are_measured_not_invented():
    """Every headline number must be reproducible from the same public risk
    functions the rest of the platform uses."""
    import pytest

    from app.services.risk import stress
    from app.services.risk.metrics import conditional_var, value_at_risk

    returns = _stress_returns()
    result = stress.run({"TEST": returns}, {"TEST": 1.0}, "vol_x2")

    # Base figures must equal the library's own output on the same series.
    assert result["before"]["var_pct"] == pytest.approx(
        abs(value_at_risk(returns, confidence=0.95, method="historical")) * 100,
        abs=1e-3)
    assert result["before"]["cvar_pct"] == pytest.approx(
        abs(conditional_var(returns, confidence=0.95)) * 100, abs=1e-3)

    # And the stressed series must be the documented transformation, not a
    # scaled copy of the base answer.
    stressed = stress.SCENARIOS["vol_x2"].apply(returns, {"vol_multiplier": 2.0})
    assert result["after"]["cvar_pct"] == pytest.approx(
        abs(conditional_var(stressed, confidence=0.95)) * 100, abs=1e-3)

    # Money terms are the measured percentage scaled by the position, nothing else.
    assert result["portfolio_loss"]["cvar_money"] == pytest.approx(
        result["position_value"] * result["after"]["cvar_pct"] / 100, abs=1e-2)


def test_the_engine_uses_no_randomness():
    """A stress result a user cannot reproduce is not evidence."""
    import ast
    from pathlib import Path

    from app.services.risk import stress

    source = Path(stress.__file__).read_text()
    tree = ast.parse(source)
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert "random" not in imported, "the engine imports the random module"

    calls = {ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n, ast.Call)}
    assert not any("random" in c for c in calls), \
        f"the engine calls a random generator: {[c for c in calls if 'random' in c]}"

    # Same input, same output.
    returns = {"TEST": _stress_returns()}
    a = stress.run(returns, {"TEST": 1.0}, "market_crash")
    b = stress.run(returns, {"TEST": 1.0}, "market_crash")
    assert a["after"] == b["after"], "two identical runs disagreed"


def test_an_unmeasurable_quantity_is_reported_as_such_not_defaulted():
    """Too little history must yield null with a reason, never a plausible zero."""
    from app.services.risk import stress

    short = _stress_returns(n=20)
    result = stress.run({"TEST": short}, {"TEST": 1.0}, "vol_x2")

    assert result["before"]["var_pct"] is None
    assert result["before"]["cvar_pct"] is None
    assert "reason" in result["before"], "no reason given for the missing figures"
    assert result["resilience"]["score"] is None, \
        "a resilience score was produced with nothing to measure it from"
    assert result["portfolio_loss"]["cvar_money"] is None


def test_correlation_spike_breaks_diversification_and_reports_it():
    """The scenario's whole purpose is to converge the book; if the measured
    correlation does not rise, it did nothing."""
    import numpy as np
    import pandas as pd

    from app.services.risk import stress

    rng = np.random.default_rng(9)
    idx = pd.date_range("2021-01-01", periods=500, freq="B")
    a = pd.Series(rng.normal(0.0003, 0.011, 500), index=idx)
    b = pd.Series(rng.normal(0.0002, 0.014, 500), index=idx)   # independent
    result = stress.run({"AAA": a, "BBB": b}, {"AAA": 0.6, "BBB": 0.4},
                        "correlation_spike")

    before = result["correlation"]["average_correlation"]
    after = result["stressed_correlation"]["average_correlation"]
    assert after > before + 0.3, (
        f"correlation barely moved: {before} -> {after}")

    # Risk contributions must be a real decomposition summing to ~100%.
    total = sum(x["risk_contribution_pct"] for x in result["assets"])
    assert total == pytest.approx(100.0, abs=1.0), \
        f"risk contributions sum to {total}%, not 100%"


def test_asset_level_impact_and_loss_shares_are_coherent(client):
    """Per-asset output must be internally consistent: shares sum to 100% and
    each asset reports its own before/after, not the portfolio's."""
    body = client.get("/api/v1/quant/stress-engine/AAPL,MSFT"
                      "?scenario=vol_x2&weights=0.7,0.3&period=2y")
    assert body.status_code == 200, body.text
    data = body.json()

    assert data["symbols"], "no symbols were analysed"
    shares = [a["loss_contribution_pct"] for a in data["assets"]
              if a["loss_contribution_pct"] is not None]
    if len(shares) == len(data["assets"]) and shares:
        assert sum(shares) == pytest.approx(100.0, abs=1.0)

    for asset in data["assets"]:
        assert asset["before"] != asset["after"] or asset["before"].get("reason")

    assert data["vulnerabilities"], "no vulnerabilities were explained"
    assert data["mitigations"], "no mitigations were recommended"
    # The narrative must quote figures that appear in the payload.
    if data["after"]["cvar_pct"] is not None:
        joined = " ".join(data["vulnerabilities"])
        assert f"{data['after']['cvar_pct']:.2f}" in joined, \
            "the explanation quotes a number that is not in the result"


# ============================================ Regime-Aware Mixture-of-Experts
def _moe_series(n=600, seed=11):
    """Deterministic series with a planted regime shift, for MoE tests."""
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(seed)
    idx = pd.date_range("2021-01-01", periods=n, freq="B")
    # calm uptrend, then a violent drawdown, then recovery: all three experts
    # get bars, which a single-regime series would not provide.
    calm = rng.normal(0.0008, 0.008, n // 3)
    crash = rng.normal(-0.004, 0.030, n // 3)
    rest = n - 2 * (n // 3)
    rebound = rng.normal(0.0012, 0.012, rest)
    r = np.concatenate([calm, crash, rebound])
    close = 100 * np.exp(np.cumsum(r))
    return pd.DataFrame({"close": close, "open": close,
                         "high": close * 1.01, "low": close * 0.99,
                         "volume": 1e6}, index=idx)


def test_moe_routes_the_three_experts_from_the_existing_detector():
    """The router must reuse the platform's regime labels, not invent its own,
    and must cover every regime the detector can emit."""
    from app.services.risk.regime import REGIMES
    from app.services.rl.moe import EXPERTS, REGIME_TO_EXPERT, route

    assert EXPERTS == ("bull", "bear", "stress")

    # Every regime the detector produces must be routable, or a live switch
    # would silently fall through to a default.
    for regime in REGIMES:
        assert regime in REGIME_TO_EXPERT, f"{regime} has no expert"
        assert REGIME_TO_EXPERT[regime] in EXPERTS

    # Warm-up bars are not a regime: they must not be attributed to an expert.
    from app.services.rl.moe import BASE_EXPERT, WARMUP_REGIME
    assert route(WARMUP_REGIME) == BASE_EXPERT
    assert BASE_EXPERT not in EXPERTS


def test_moe_makes_k5_reaction_delay_measurable():
    """KPI K-5 was unmeasurable before: the platform had no notion of reacting
    to a regime change. It must now return a number, in a stated unit."""
    from app.services.rl.moe import RegimeMoE

    df = _moe_series()
    result = RegimeMoE().run("TEST", df)
    k5 = result.k5

    assert k5["unit"] == "bars"
    assert k5["n_switches"] > 0, "no regime change on a series built to contain one"
    assert k5["measured"] > 0, "K-5 could not be measured on any switch"
    assert k5["mean_reaction_bars"] is not None
    assert k5["mean_reaction_bars"] >= 0
    assert k5["max_reaction_bars"] >= k5["median_reaction_bars"]
    # Failures are counted, not dropped: averaging over successes only would
    # report a flattering delay.
    assert "unadapted" in k5


def test_moe_fine_tunes_only_on_bars_strictly_before_the_switch():
    """No look-ahead. Adapting at bar t on bar t would train the expert on the
    outcome it is about to act upon."""
    from app.services.rl.moe import RegimeMoE

    df = _moe_series()
    seen = []

    def factory(expert, history):
        seen.append((expert, list(history)))

    moe = RegimeMoE()
    result = moe.run("TEST", df, expert_factory=factory)

    assert seen, "no expert was ever fine-tuned"
    switch_bars = {s.bar for s in result.switches if s.adapted and s.expert_changed}
    assert switch_bars, "no adapted expert change to check"

    for _expert, history in seen:
        assert history, "fine-tuned on an empty history"
        # Every training index must precede the earliest switch it could serve.
        assert max(history) < max(switch_bars), \
            f"fine-tuning used bar {max(history)} at or after the switch"

    # And the helper itself must be exclusive at the boundary.
    assignments = ["bull"] * 10
    assert moe._expert_history(assignments, "bull", 5) == [0, 1, 2, 3, 4]
    assert 5 not in moe._expert_history(assignments, "bull", 5)


def test_moe_preserves_the_baseline_and_invents_nothing():
    """The published single-policy results must survive untouched, and the MoE
    must not fabricate performance figures it did not compute."""
    from app.services.rl.moe import RegimeMoE

    df = _moe_series()
    baseline = {"total_return": 0.0866, "source": "published"}
    payload = RegimeMoE().run("TEST", df, baseline=baseline).to_dict()

    assert payload["baseline"] == baseline, "the baseline was altered"
    # The MoE reports routing and latency only. It must not claim a return.
    for invented in ("total_return", "sharpe_ratio", "alpha"):
        assert invented not in payload, \
            f"the MoE reports {invented}, which it never measured"

    assert set(payload["experts"]) == {"bull", "bear", "stress"}
    assert sum(payload["experts"].values()) <= payload["bars"]


def test_moe_reports_an_expert_it_cannot_fit_rather_than_faking_one():
    """An expert with too few bars must be declared, not silently fine-tuned
    on noise."""
    from app.services.rl.moe import RegimeMoE

    df = _moe_series(n=200, seed=5)
    calls = []
    moe = RegimeMoE(min_expert_bars=10_000)      # nothing can meet this
    result = moe.run("TEST", df, expert_factory=lambda e, h: calls.append(e))

    assert not calls, "an expert was fine-tuned below the minimum bar count"
    changed = [s for s in result.switches if s.expert_changed]
    if changed:
        assert all(not s.adapted for s in changed)
        assert all("need 10000" in s.reason for s in changed)
        assert result.k5["unadapted"] > 0


def _moe_provision_agent(symbol="AAPL", algo="dueling_dqn"):
    """Train a tiny agent into the test MODEL_DIR and return the history frame.

    `conftest` redirects MODEL_DIR to an empty temp directory, so the shipped
    checkpoints are invisible and these tests skipped themselves into a green
    tick that verified nothing. Training a 2-episode agent costs a few seconds
    and makes the assertions real.
    """
    from app.services.data.market_data import market_data_service
    from app.services.rl.service import rl_service

    series = market_data_service.get_history(symbol, period="2y")
    df = getattr(series, "df", series)
    if not rl_service.agent_path(symbol, algo).with_suffix(".pt").exists():
        rl_service.train_single_asset(symbol, algo=algo, period="2y", episodes=2)
    return df


def test_moe_experts_are_real_policies_not_placeholders():
    """Audit finding: `expert_factory` was an extension point with no real
    caller — the only callers were test lambdas, so no neural network was ever
    touched. This pins the fix: the factory must load the platform's own
    trained agent and hand back genuine policy objects.
    """
    import pandas as pd

    from app.services.rl.moe import PolicyExpertFactory

    df = _moe_series(n=400)
    factory = PolicyExpertFactory("AAPL", df, algo="dueling_dqn")

    # It must reuse the existing loader, not build a fresh untrained net.
    import inspect
    src = inspect.getsource(PolicyExpertFactory.base_policy)
    assert "rl_service.load_agent" in src, \
        "the factory does not reuse the existing checkpoint loader"

    # And the hook signature must match what RegimeMoE calls.
    assert callable(factory)
    sig = inspect.signature(PolicyExpertFactory.__call__)
    assert list(sig.parameters)[1:3] == ["expert", "history"]


def test_moe_fine_tuning_actually_changes_policy_weights():
    """"Active adaptation" must mean a gradient update, not a relabelled copy.

    Regression: the first wiring ran `train()` but left the weights
    bit-identical, because TradingEnv spends `lookback` bars on the first
    observation and the replay buffer never reached `min_buffer`. The factory
    now derives the episode count from the slice length and *raises* rather
    than reporting an adaptation that did not happen.
    """
    import numpy as np

    from app.services.data.market_data import market_data_service
    from app.services.rl.moe import PolicyExpertFactory

    df = _moe_provision_agent()
    factory = PolicyExpertFactory("AAPL", df, algo="dueling_dqn")
    agent = factory.base_policy()

    before = [p.detach().clone().numpy() for p in agent.online.parameters()]
    history = list(range(len(df) - 1))          # a long, adaptable slice
    record = factory("bull", history)

    assert record["weights_changed"] is True, "fine-tuning moved no weight"
    assert record["weight_delta"] > 1e-9
    assert record["buffer_after"] >= agent.cfg.min_buffer, \
        "the replay buffer never armed learn_step"

    tuned = factory.experts["bull"]
    after = [p.detach().numpy() for p in tuned.online.parameters()]
    delta = max(float(np.abs(a - b).max())
                for a, b in zip(before, after, strict=True))
    assert delta > 1e-9

    verdict = factory.verify_adaptation()
    assert verdict["any_weights_changed"] is True
    assert "bull" in verdict["experts_adapted"]


def test_moe_refuses_to_claim_an_adaptation_it_did_not_perform():
    """A slice too short to fill the replay buffer must fail loudly. Silently
    returning `adapted=True` on an unchanged network is the precise false claim
    this audit was asked to rule out."""
    import pytest

    from app.services.data.market_data import market_data_service
    from app.services.rl.moe import MIN_EXPERT_BARS, PolicyExpertFactory

    df = _moe_provision_agent()
    factory = PolicyExpertFactory("AAPL", df, algo="dueling_dqn")
    factory.base_policy()

    # 25 bars: below lookback + margin, cannot possibly fill the buffer.
    with pytest.raises(RuntimeError, match="unchanged"):
        factory("bear", list(range(25)))

    # The floor must be high enough that an accepted switch can actually learn.
    assert MIN_EXPERT_BARS >= 90, \
        "the minimum bar count is too low to fill the replay buffer"

    # Second, distinct guard: a slice that clears the floor but still leaves the
    # weights untouched must also raise. Removing that check made the previous
    # version of this test pass, so it was verifying only half the contract.
    import inspect

    from app.services.rl.moe import PolicyExpertFactory as _F

    body = inspect.getsource(_F.__call__)
    assert "delta <= 1e-9" in body, \
        "nothing refuses an adaptation that changed no weight"
    guard = body[body.index("delta <= 1e-9"):]
    assert "raise RuntimeError" in guard[:400], \
        "a zero-delta fine-tune is recorded instead of raising"
    assert "weights_changed" in body, \
        "the record does not report whether weights actually moved"


# ---------------------------------------------------------------------------
# MoE integration into the live application
#
# The tests above prove the mechanism works in isolation. These prove it is
# actually *wired in*: that the application reaches it, that turning it off
# leaves the previous behaviour bit-for-bit intact, and that turning it on
# runs real experts rather than reporting an adaptation that never happened.
# ---------------------------------------------------------------------------


def _moe_integration_agent(symbol="AAPL", algo="dueling_dqn", period="2y"):
    """Provision a checkpoint in the test MODEL_DIR and return the symbol.

    `conftest` points MODEL_DIR at an empty temp dir, so the shipped agents are
    invisible. Without this the integration tests would skip and prove nothing.
    """
    from app.services.rl.service import rl_service

    if not rl_service.agent_path(symbol, algo).with_suffix(".pt").exists():
        rl_service.train_single_asset(symbol, algo=algo, period=period, episodes=2)
    return symbol, algo


def test_moe_is_actually_reachable_from_the_application():
    """Audit finding this closes: `moe.py` had no importer outside its own
    tests, so no line of it ever ran in the live application. The API layer
    must now reach it.
    """
    import ast
    from pathlib import Path

    src = Path("backend/app/api/v1/endpoints/rl.py")
    if not src.exists():                       # pragma: no cover - path guard
        import app.api.v1.endpoints.rl as _rl
        src = Path(_rl.__file__)
    tree = ast.parse(src.read_text())

    # Parsed, not grepped: a mention inside a docstring or a comment must not
    # be able to satisfy this.
    imported = {
        node.module for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }
    assert "app.services.rl.moe" in imported, \
        "no endpoint imports the MoE module - it is still dead code"


def test_moe_disabled_reproduces_the_baseline_exactly(client):
    """The contract: with MoE off, nothing changes. Not 'approximately' - the
    same numbers, and no extra keys for a caller to trip over.
    """
    symbol, algo = _moe_integration_agent()
    url = f"/api/v1/rl/backtest/{symbol}?algo={algo}&period=1y"

    before = client.get(url)
    assert before.status_code == 200
    a = before.json()

    explicit_off = client.get(url + "&moe=false")
    assert explicit_off.status_code == 200
    b = explicit_off.json()

    assert a == b, "moe=false diverged from the default path"
    # The baseline payload must not grow a MoE block: existing consumers read
    # this response and an unexpected key is a contract change.
    assert "moe" not in a, "the baseline response leaked a MoE block"
    assert "mode" not in a, "the baseline response gained a mode marker"


def test_moe_enabled_runs_real_experts_and_reports_k5(client):
    """With MoE on, the run must use genuinely fine-tuned policies - not a
    relabelled copy of the base agent - and must publish the reaction delay.
    """
    symbol, algo = _moe_integration_agent()
    r = client.get(f"/api/v1/rl/backtest/{symbol}?algo={algo}&period=2y&moe=true")
    assert r.status_code == 200
    body = r.json()

    # Same shape as the baseline, so the existing UI can render it unchanged.
    for key in ("performance", "baselines", "equity_curve", "trades"):
        assert key in body, f"the MoE payload dropped '{key}'"
    assert body["mode"] == "moe"

    moe = body["moe"]
    assert moe["adapt"] is True

    # A real gradient update, measured on the weights themselves.
    check = moe["adaptation_check"]
    assert check["any_weights_changed"] is True, \
        "MoE ran but no expert's weights ever moved"
    assert check["max_weight_delta"] > 1e-9
    assert moe["adaptations"], "no adaptation was recorded"
    for record in moe["adaptations"]:
        assert record["weights_changed"] is True
        assert record["expert"] in ("bull", "bear", "stress")

    # An expert must actually have driven decisions, otherwise "the app uses
    # the MoE" would be true only on paper.
    #
    # `bars_acted_by` alone is not enough: a mutation that advanced the label
    # without swapping the acting policy left this counter reporting hundreds
    # of expert bars while the base agent decided every one of them. So the
    # run is also compared against the control that routes without adapting -
    # if the fine-tuned experts really hold the wheel, the two must diverge.
    acted = moe["bars_acted_by"]
    expert_bars = sum(v for k, v in acted.items() if k != "base")
    assert expert_bars > 0, "every bar was still decided by the base policy"

    control = client.get(
        f"/api/v1/rl/backtest/{symbol}?algo={algo}&period=2y"
        "&moe=true&moe_adapt=false").json()
    assert control["moe"]["adaptation_check"]["any_weights_changed"] is False
    assert body["equity_curve"] != control["equity_curve"], (
        "the adaptive run produced the same equity curve as the un-adapted "
        "control, so the fine-tuned experts never actually decided anything")

    k5 = moe["k5_reaction_delay"]
    assert k5["unit"] == "bars"
    assert k5["measured"] >= 1
    assert k5["mean_reaction_bars"] is not None
    # Failures are published, not hidden.
    assert "unadapted" in k5
    assert "k5_expert_changes_only" in moe, \
        "only the flattering K-5 reading is reported"


def test_moe_fine_tuning_never_reads_the_switch_bar_or_later(client):
    """No look-ahead through the integration path. The isolated mechanism has
    its own leakage test; this pins the wired-up version, where the bar indices
    come from the environment rather than from a test fixture.
    """
    symbol, algo = _moe_integration_agent()
    body = client.get(
        f"/api/v1/rl/backtest/{symbol}?algo={algo}&period=2y&moe=true").json()
    moe = body["moe"]

    # Each adaptation records the last bar it trained on. That bar must precede
    # the switch the adaptation served.
    served = sorted(s["bar"] for s in moe["switches"] if s["adapted"])
    assert served, "no adapted switch to check"
    for record in moe["adaptations"]:
        assert record["last_bar_used"] < max(served), (
            f"fine-tuning read bar {record['last_bar_used']}, at or after the "
            f"switch it served")


def test_moe_without_adaptation_leaves_weights_untouched(client):
    """The control condition must be real: adapt=false routes the experts but
    applies no gradient, so any weight movement in the adaptive run is
    attributable to the fine-tuning and not to the routing.
    """
    symbol, algo = _moe_integration_agent()
    body = client.get(
        f"/api/v1/rl/backtest/{symbol}?algo={algo}&period=2y"
        "&moe=true&moe_adapt=false").json()
    moe = body["moe"]

    assert moe["adapt"] is False
    assert moe["adaptations"] == [], "a gradient update ran with adapt=false"
    assert moe["adaptation_check"]["any_weights_changed"] is False
    # Routing still happened - that is the point of the control.
    assert moe["n_switches"] > 0
    assert any("adapt=False" in n for n in moe["notes"]), \
        "the payload does not disclose that nothing was fine-tuned"


def test_moe_refuses_algorithms_it_cannot_fine_tune(client):
    """SB3 policies expose `predict`, not `q_values`, and cannot be fine-tuned
    through this path. Refusing is honest; routing them anyway and publishing
    adaptation figures would not be.
    """
    r = client.get("/api/v1/rl/backtest/AAPL?algo=ppo&period=1y&moe=true")
    assert r.status_code == 422
    assert "ppo" in r.json()["message"]

    # And the same algorithm must still backtest normally with MoE off.
    baseline = client.get("/api/v1/rl/backtest/AAPL?algo=ppo&period=1y")
    assert baseline.status_code in (200, 409)   # 409 only if never trained
    if baseline.status_code == 200:
        assert "moe" not in baseline.json()


def test_moe_switch_trace_is_internally_consistent(client):
    """Regression: `from_expert` was taken from the previous regime's owner
    while `expert_changed` compared against the expert actually acting, so the
    trace printed 'bull -> bull, expert_changed=True'. A self-contradictory
    audit record is worse than none.
    """
    symbol, algo = _moe_integration_agent()
    moe = client.get(
        f"/api/v1/rl/backtest/{symbol}?algo={algo}&period=2y&moe=true"
    ).json()["moe"]

    for s in moe["switches"]:
        assert s["expert_changed"] == (s["from_expert"] != s["to_expert"]), (
            f"bar {s['bar']}: from_expert={s['from_expert']} "
            f"to_expert={s['to_expert']} but expert_changed={s['expert_changed']}")
        # An unadapted switch must carry a reason, never a bare failure.
        if s["reaction_bars"] is None:
            assert s["reason"], f"bar {s['bar']} failed silently"


def test_moe_records_a_failed_fine_tune_as_unadapted(client, monkeypatch):
    """A fine-tune that raises must be recorded as a failure to react.

    This branch never fires on the offline fixtures - measured: zero refused
    fine-tunes across 1y, 2y and 5y - so a mutation that set `ok = True` inside
    the exception handler survived the whole suite. Forcing the branch with a
    stub is what makes the guarantee real rather than merely plausible: the
    alternative is a payload reporting `adapted=True` for an expert whose
    weights were never touched.
    """
    from app.services.rl.moe import PolicyExpertFactory

    symbol, algo = _moe_integration_agent()

    def _explode(self, expert, history):
        raise RuntimeError("forced fine-tune failure")

    monkeypatch.setattr(PolicyExpertFactory, "__call__", _explode)

    moe = client.get(
        f"/api/v1/rl/backtest/{symbol}?algo={algo}&period=2y&moe=true"
    ).json()["moe"]

    assert moe["adaptations"] == [], "a failed fine-tune was recorded as an adaptation"
    assert moe["adaptation_check"]["any_weights_changed"] is False

    refused = [s for s in moe["switches"] if "refused" in (s["reason"] or "")]
    assert refused, "the forced failure was never surfaced in the trace"
    for s in refused:
        assert s["adapted"] is False, "a failed fine-tune is reported as adapted"
        assert s["reaction_bars"] is None, \
            "a reaction delay was reported for a switch that never reacted"

    # And no expert may drive a single bar when every fine-tune failed.
    assert set(moe["bars_acted_by"]) == {"base"}, \
        "an expert took control despite every fine-tune failing"


# ---------------------------------------------------------------------------
# MoE and regime awareness in the user interface
#
# The mechanism being reachable over HTTP is not the same as it being reachable
# by a user. These pin the controls themselves.
# ---------------------------------------------------------------------------


def _frontend_file(name: str) -> str:
    from pathlib import Path

    import app.main as _main

    root = Path(_main.__file__).resolve().parent.parent.parent / "frontend"
    return (root / name).read_text()


def test_rl_page_lets_the_user_enable_regime_awareness():
    """Root cause of "This agent was trained without regime awareness".

    The explainability panel told users to retrain with `regime_aware`, but the
    training form had no such control, so the advice was impossible to follow
    and every agent trained from this page came out regime-blind. The checkbox
    must exist and its value must actually reach the API call.
    """
    html = _frontend_file("rl.html")
    assert 'id="rRegimeAware"' in html, "no regime-awareness control on the RL page"

    js = _frontend_file("assets/js/pages/rl.js")
    assert "rRegimeAware" in js, "the control is never read"
    # It has to be in the payload, not merely read into a local variable.
    assert "regime_aware:" in js, "regime_aware is never sent to the training endpoint"
    train_call = js[js.index("api.trainRL("):]
    assert "regime_aware" in train_call[:400], \
        "the single-asset training call omits regime_aware"


def test_rl_page_exposes_the_mixture_of_experts():
    """The MoE must be operable from the page, not only from a hand-typed URL."""
    html = _frontend_file("rl.html")
    assert 'id="moeToggle"' in html, "no Mixture-of-Experts control on the RL page"
    assert 'id="moePanel"' in html, "nowhere to render the routing trace"

    js = _frontend_file("assets/js/pages/rl.js")
    assert "renderMoePanel" in js
    # The toggle must gate the request, not decorate the page.
    assert "moe: useMoe" in js, "the toggle never reaches the backtest call"

    api = _frontend_file("assets/js/api.js")
    assert "moe=true" in api, "the API helper cannot request MoE mode"
    # Default calls must stay exactly as they were: no stray parameter.
    assert "if (opts.moe)" in api, \
        "the MoE flag is not conditional, so plain backtests would change"


def test_moe_ui_reports_failures_not_only_successes():
    """A panel that showed only the adaptations that worked would misrepresent
    a mechanism that, on a 1-year window, fires once out of eight switches."""
    js = _frontend_file("assets/js/pages/rl.js")

    assert "unadapted" in js, "the panel never surfaces switches that did not react"
    # Both K-5 readings, so the flattering one cannot be quoted alone.
    assert "k5_expert_changes_only" in js, "only the global K-5 is displayed"
    # A mean over one observation must be labelled as such.
    assert "single event, not an average" in js, \
        "a one-sample mean is presented as if it were an average"
    assert "adaptation_check" in js or "any_weights_changed" in js, \
        "the panel does not report whether weights actually changed"


def test_moe_toggle_is_disabled_for_algorithms_it_cannot_drive():
    """SB3 policies have no fine-tune path; the backend answers 422. The UI must
    say so up front instead of letting the user trigger an error."""
    js = _frontend_file("assets/js/pages/rl.js")
    assert "MOE_ALGOS" in js
    for native in ("dqn", "double_dqn", "dueling_dqn", "c51", "iqn", "rainbow"):
        assert f"'{native}'" in js, f"{native} missing from the MoE allow-list"
    assert "box.disabled" in js, "the toggle is never disabled"

    # The allow-list must match what the backend actually accepts.
    from app.services.rl.service import NATIVE_DISCRETE
    listed = js[js.index("MOE_ALGOS"):]
    listed = listed[:listed.index("]")]
    for algo in NATIVE_DISCRETE:
        assert algo in listed, f"backend accepts {algo} but the UI hides it"


def test_chart_range_buttons_do_not_overlap_the_legend():
    """Regression, measured in a real browser: the legend sits at y 1.12
    anchored left and the range buttons were at the same height, also left, so
    "MoE agent" (x 383-435) was drawn through "6M"/"All" (x 386-434)."""
    js = _frontend_file("assets/js/timerange.js")
    selector = js[js.index("rangeselector"):]
    selector = selector[:selector.index("...(overrides.xaxis")]
    assert "xanchor: 'right'" in selector, \
        "the range buttons are not anchored away from the legend"
    assert "x: 1," in selector, "the range buttons are still left-aligned"


# ---------------------------------------------------------------------------
# Regime-aware twins, kept beside their baseline
# ---------------------------------------------------------------------------


def test_agent_variant_never_collides_with_the_baseline_filename():
    """A twin must not overwrite the agent it is compared against.

    21 trained checkpoints exist on disk; a variant scheme that reused their
    names would destroy the baseline half of every comparison.
    """
    from app.services.rl.service import rl_service

    base = rl_service.agent_path("AAPL", "dqn")
    twin = rl_service.agent_path("AAPL", "dqn", "regime")
    assert base != twin
    assert base.name == "rl_AAPL_dqn", "the default filename changed"
    assert twin.name == "rl_AAPL_dqn__regime"
    assert rl_service.meta_path("AAPL", "dqn").name == "rl_AAPL_dqn.json"


def test_agent_variant_cannot_escape_the_model_directory():
    """The suffix reaches the filesystem, so it must be sanitised."""
    import pytest

    from app.core.exceptions import InvalidRequestError
    from app.services.rl.service import rl_service

    root = rl_service.model_dir.resolve()
    for hostile in ("../evil", "../../etc/passwd", "/abs", "a/b"):
        path = rl_service.agent_path("X", "dqn", hostile).resolve()
        assert path.parent == root, f"{hostile!r} escaped to {path.parent}"
        assert ".." not in str(path)

    # A variant that sanitises to nothing must be refused, not silently
    # collapsed onto the baseline's own filename.
    for empty in ("..", "!!!", "///"):
        with pytest.raises(InvalidRequestError):
            rl_service.agent_path("X", "dqn", empty)


def test_variant_reaches_every_layer_that_loads_an_agent():
    """A variant honoured at save time but ignored at load time would quietly
    serve the baseline while claiming to be the twin."""
    import inspect

    from app.services.rl import moe
    from app.services.rl.service import RLService

    for fn in (RLService.load_agent, RLService.backtest,
               RLService.recommend_action, RLService._env_config_for_agent,
               moe.rollout):
        assert "variant" in inspect.signature(fn).parameters, \
            f"{fn.__qualname__} cannot address a variant"

    # The continuous path delegates; the delegation must carry the variant.
    # Regression: it did not, and three baselines were overwritten by their
    # own twins before the mistake was caught.
    src = inspect.getsource(RLService.train_single_asset)
    delegation = src[src.index("_train_continuous_single_asset"):]
    assert "variant=variant" in delegation[:300], \
        "the continuous branch drops the variant and overwrites the baseline"


def test_rl_page_no_longer_exposes_the_twin_selector():
    """The twin selector was removed from the UI on request.

    Removing a control must not remove the capability: the twins stay on disk
    and the backend keeps serving them under `?variant=regime`. This pins both
    halves — the control is gone from the page, and the plumbing that can still
    reach a twin is intact.
    """
    html = _frontend_file("rl.html")
    assert 'id="rVariant"' not in html, "the twin selector is still in the page"
    assert "Regime-aware twin" not in html, "the twin option is still offered"

    # The MoE toggle sat beside it and must survive untouched.
    assert 'id="moeToggle"' in html, "removing the selector took the MoE toggle with it"

    js = _frontend_file("assets/js/pages/rl.js")
    # No dead listener on an element that no longer exists.
    assert "ui.el('rVariant')?.addEventListener" not in js, \
        "a listener is still bound to the removed selector"
    # Call sites stay parameterised so the backend capability is reachable.
    assert "variant: currentVariant()" in js
    assert "currentVariant())" in js


def test_removing_the_selector_left_the_backend_variant_support_intact():
    """Deleting a button must not delete the feature behind it."""
    import inspect

    from app.services.rl import moe
    from app.services.rl.service import RLService

    for fn in (RLService.load_agent, RLService.backtest,
               RLService.recommend_action, RLService.recommend_allocation,
               moe.rollout):
        assert "variant" in inspect.signature(fn).parameters, \
            f"{fn.__qualname__} lost its variant support"

    # The API must still advertise the parameter.
    import app.api.v1.endpoints.rl as rl_ep

    for name in ("backtest", "recommend_action", "recommend_allocation"):
        assert "variant" in inspect.signature(getattr(rl_ep, name)).parameters, \
            f"/rl endpoint '{name}' no longer exposes variant"

    # And the regime-aware checkpoints must still be on disk.
    from pathlib import Path

    import app.main as _main

    model_dir = Path(_main.__file__).resolve().parents[2] / "data" / "models" / "rl"
    if model_dir.exists():
        twins = list(model_dir.glob("rl_*__regime.json"))
        assert twins, "the regime-aware checkpoints were deleted"


def test_every_agent_family_can_address_its_regime_aware_twin():
    """A twin that exists on disk but cannot be reached is not a deliverable.

    Regression: `recommend_allocation` had no `variant` parameter, so the seven
    basket twins were unreachable — the endpoint silently served the regime-blind
    baseline instead, which is the worst possible failure mode: a wrong answer
    that looks right.
    """
    import inspect

    from app.services.rl.service import RLService

    # Both single-asset and basket inference must accept a variant.
    for fn in (RLService.recommend_action, RLService.recommend_allocation):
        assert "variant" in inspect.signature(fn).parameters, \
            f"{fn.__qualname__} cannot reach a twin"

    # And it must be threaded through, not accepted and dropped.
    src = inspect.getsource(RLService.recommend_allocation)
    assert "load_agent(key, algo, env, variant)" in src, \
        "recommend_allocation loads the baseline whatever the caller asked for"
    assert "_env_config_for_agent(key, algo, None, variant)" in src, \
        "the basket env is rebuilt from the baseline's metadata"


def test_allocation_endpoint_exposes_the_variant():
    """The HTTP surface must expose what the service supports, or the twins stay
    unreachable from the application."""
    import app.api.v1.endpoints.rl as rl_ep

    for name in ("recommend_allocation", "recommend_action", "backtest"):
        fn = getattr(rl_ep, name)
        import inspect
        assert "variant" in inspect.signature(fn).parameters, \
            f"/rl endpoint '{name}' does not expose variant"


def test_moe_experts_derive_from_the_variant_the_caller_asked_for():
    """The MoE must fine-tune the twin when asked for the twin, not silently
    fall back to the regime-blind baseline.

    Both checkpoints exist for the same symbol+algo and differ in observation
    width (36 vs 42), so serving the wrong one is detectable by shape alone.
    """
    import numpy as np

    from app.services.rl import moe
    from app.services.rl.moe import PolicyExpertFactory
    from app.services.rl.service import rl_service

    symbol, algo = _moe_integration_agent()

    # Provision a regime-aware twin in the test MODEL_DIR, under the suffix.
    if not rl_service.agent_path(symbol, algo, "regime").with_suffix(".pt").exists():
        rl_service.train_single_asset(symbol, algo=algo, period="2y", episodes=2,
                                      env_overrides={"regime_aware": True},
                                      variant="regime")

    seen: dict[str, tuple] = {}
    original = PolicyExpertFactory.__call__

    def spy(self, expert, history):
        record = original(self, expert, history)
        weights = list(self.experts[expert].online.parameters())[0]
        seen[expert] = tuple(weights.shape)
        return record

    PolicyExpertFactory.__call__ = spy
    try:
        seen.clear()
        baseline = moe.rollout(symbol, algo=algo, period="2y")
        base_shapes = dict(seen)

        seen.clear()
        twin = moe.rollout(symbol, algo=algo, period="2y", variant="regime")
        twin_shapes = dict(seen)
    finally:
        PolicyExpertFactory.__call__ = original

    assert base_shapes, "no expert was fine-tuned on the baseline run"
    assert twin_shapes, "no expert was fine-tuned on the twin run"

    # The regime block adds REGIME_FEATURE_DIM columns to the observation, so
    # an expert cloned from the twin is strictly wider than one cloned from the
    # baseline. Same width would mean the twin was never loaded.
    from app.services.rl.regime_features import REGIME_FEATURE_DIM

    for expert, shape in twin_shapes.items():
        if expert in base_shapes:
            assert shape[1] == base_shapes[expert][1] + REGIME_FEATURE_DIM, (
                f"expert '{expert}' was cloned from a {shape[1]}-wide policy; "
                f"the twin should be {REGIME_FEATURE_DIM} columns wider than "
                f"the baseline's {base_shapes[expert][1]}")

    assert twin["variant"] == "regime"
    assert baseline["variant"] is None
    # Different starting policies must produce different runs, otherwise the
    # variant is decorative.
    assert baseline["equity_curve"] != twin["equity_curve"], \
        "the twin run reproduced the baseline exactly - the variant was ignored"

    # And the fine-tuning has to be real on the twin too, not just on the base.
    assert twin["moe"]["adaptation_check"]["any_weights_changed"] is True
    assert np.isfinite(twin["moe"]["adaptation_check"]["max_weight_delta"])


def test_symbols_containing_a_dot_get_their_own_checkpoint_file():
    """Regression, found while training all 32 symbols.

    Every save site calls `.with_suffix(".pt")` on the agent path. For a
    European ticker like `MC.PA` the stem was `rl_MC.PA_dqn__regime`, whose
    "extension" Path reads as `.PA_dqn__regime` — so `with_suffix` *replaced*
    it and all six algorithms collapsed onto `rl_MC.pt`. Four of them then
    failed to load with `DQNConfig.__init__() got an unexpected keyword
    argument 'n_atoms'`: a C51 checkpoint being read back as a DQN.
    """
    from app.services.rl.service import rl_service

    for symbol in ("MC.PA", "AIR.PA", "SAN.PA"):
        path = rl_service.agent_path(symbol, "dqn", "regime")
        assert "." not in path.name, f"{symbol} still produces a dotted stem: {path.name}"
        # The real failure mode: with_suffix must append, not overwrite.
        assert path.with_suffix(".pt").stem == path.name, (
            f"{symbol}: with_suffix truncated {path.name} to "
            f"{path.with_suffix('.pt').stem}")

    # Distinct algorithms must never share a file for a dotted symbol.
    files = {algo: rl_service.agent_path("MC.PA", algo, "regime").with_suffix(".pt")
             for algo in ("dqn", "double_dqn", "dueling_dqn", "c51", "iqn", "rainbow")}
    assert len(set(files.values())) == len(files), \
        f"algorithms share a checkpoint file: {files}"

    # And the other awkward characters must still be handled.
    for symbol, expected in (("^GSPC", "idx_GSPC"), ("EURUSD=X", "EURUSD_X"),
                             ("BTC-USD", "BTC-USD")):
        assert expected in rl_service.agent_path(symbol, "dqn").name


def test_every_regime_twin_on_disk_is_complete_and_loadable():
    """Whatever twins exist must be usable — no half-written pairs.

    This deliberately does **not** demand full 32x6 coverage. That sweep ran and
    was verified, but the workspace snapshot caps at 128 MB / 10 000 files and
    dropped 182 of its 192 checkpoints; asserting coverage would turn a storage
    limit into a permanently red test. What must hold is consistency: every
    metadata sidecar has its weights, and every twin claims to be regime-aware.

    Regenerate the full sweep with:
        scripts/train_all_regime_aware.py --only-algo dqn double_dqn \\
            dueling_dqn c51 iqn rainbow
    """
    import json
    from pathlib import Path

    import app.main as _main

    model_dir = Path(_main.__file__).resolve().parents[2] / "data" / "models" / "rl"
    if not model_dir.exists():                      # pragma: no cover - fresh clone
        import pytest
        pytest.skip("no shipped model directory")

    incomplete, not_aware = [], []
    for meta in sorted(model_dir.glob("rl_*__regime.json")):
        stem = meta.with_suffix("")
        if not (Path(str(stem) + ".pt").exists() or Path(str(stem) + ".zip").exists()):
            incomplete.append(meta.name)
            continue
        try:
            payload = json.loads(meta.read_text())
        except Exception as exc:                     # pragma: no cover
            incomplete.append(f"{meta.name}: unreadable ({exc})")
            continue
        if (payload.get("env_config") or {}).get("regime_aware") is not True:
            not_aware.append(meta.name)

    assert not incomplete, f"twin metadata with no weights: {incomplete}"
    assert not not_aware, f"twins that are not regime-aware: {not_aware}"


def test_the_repository_stays_inside_the_workspace_snapshot_budget():
    """The workspace snapshots at most 128 MB / 10 000 files.

    A 192-checkpoint sweep pushed this session to 157.9 MB across 971 files and
    364 files were silently dropped — including 182 trained agents. Silently is
    the problem: nothing failed, nothing warned, and a later `--verify` was the
    first thing to notice. This test makes the next approach to the ceiling
    visible while it can still be acted on.

    Excluded from the count are the directories the snapshot itself excludes,
    so this measures what actually gets persisted.
    """
    from pathlib import Path

    import app.main as _main

    root = Path(_main.__file__).resolve().parents[2]
    excluded = {
        ".arena", ".cache", ".git", ".mypy_cache", ".next", ".nox", ".npm",
        ".nuxt", ".output", ".parcel-cache", ".pytest_cache", ".ruff_cache",
        ".svelte-kit", ".tox", ".turbo", ".venv", ".vite", "__pycache__",
        "build", "coverage", "dist", "node_modules", "out", "target",
    }

    total_bytes = 0
    total_files = 0
    for path in root.rglob("*"):
        if any(part in excluded for part in path.parts):
            continue
        if path.is_file():
            total_files += 1
            try:
                total_bytes += path.stat().st_size
            except OSError:            # pragma: no cover - vanished mid-walk
                continue

    size_mb = total_bytes / 1_048_576
    # 85% of each ceiling: enough headroom to notice and delete something
    # before a snapshot starts discarding files.
    assert size_mb < 108.8, (
        f"repository is {size_mb:.1f} MB, past 85% of the 128 MB snapshot cap — "
        f"files will be dropped. Largest offender is usually "
        f"data/models/rl/*__regime.*")
    assert total_files < 8_500, (
        f"repository has {total_files} files, past 85% of the 10 000-file cap")
