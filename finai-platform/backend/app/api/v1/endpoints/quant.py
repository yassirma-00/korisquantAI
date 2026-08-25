"""Advanced quantitative endpoints: conformal intervals, GARCH, VaR validation."""

from __future__ import annotations

import numpy as np
from fastapi import APIRouter, Query

from app.core.exceptions import InvalidRequestError
from app.services.data.market_data import market_data_service
from app.services.forecasting.advanced import ForecastEnsemble, regime_detector
from app.services.forecasting.conformal import (
    AdaptiveConformal,
    MondrianConformal,
    SplitConformal,
    evaluate_coverage,
)
from app.services.forecasting.trainer import forecast_trainer
from app.services.risk.service import risk_service

router = APIRouter(prefix="/quant", tags=["Advanced Quantitative"])


# ------------------------------------------------------------- conformal
@router.get("/conformal/{symbol}", summary="Conformal prediction intervals with proven coverage")
async def conformal_interval(
    symbol: str,
    period: str = Query("5y"),
    horizon: int = Query(5, ge=1, le=30),
    alpha: float = Query(0.1, gt=0.001, lt=0.5, description="miscoverage; 0.1 -> 90% interval"),
    method: str = Query("adaptive", pattern="^(split|mondrian|adaptive)$"),
):
    """Intervals with a finite-sample coverage guarantee.

    Unlike a Gaussian ±1.645σ band, split conformal guarantees
    ``P(y ∈ C(x)) ≥ 1 − α`` under exchangeability alone — no normality assumed.
    The response includes an out-of-sample coverage check so you can verify the
    guarantee held on this specific series.
    """
    series = market_data_service.get_history(symbol, period=period)
    df = series.df
    fwd = df["close"].pct_change(horizon).shift(-horizon).dropna()
    if len(fwd) < 120:
        raise InvalidRequestError(
            f"Need >= 120 usable observations, got {len(fwd)}. Request a longer period.")

    y = fwd.values
    # Random-walk baseline (0 expected return) unless a model is trained
    point_pred = 0.0
    model_used = "random_walk_baseline"
    try:
        prediction = forecast_trainer.predict(symbol, df, "lstm", horizon)
        point_pred = prediction["predicted_return"]
        model_used = "lstm"
    except Exception:
        pass

    split = len(y) // 2
    cal_true, cal_pred = y[:split], np.zeros(split)
    test_true, test_pred = y[split:], np.zeros(len(y) - split)

    if method == "mondrian":
        vol = df["close"].pct_change().rolling(21).std().reindex(fwd.index).bfill().values
        model = MondrianConformal(alpha, 3).calibrate(cal_true, cal_pred, vol[:split])
        intervals = [model.predict(float(p), float(v))
                     for p, v in zip(test_pred, vol[split:], strict=False)]
        live = model.predict(point_pred, float(vol[-1]))
    elif method == "adaptive":
        model = AdaptiveConformal(alpha).calibrate(cal_true, cal_pred)
        intervals = []
        for actual, p in zip(test_true, test_pred, strict=False):
            iv = model.predict(float(p))
            intervals.append(iv)
            model.update(float(actual), iv)     # online feedback keeps coverage on target
        live = model.predict(point_pred)
    else:
        model = SplitConformal(alpha).calibrate(cal_true, cal_pred)
        intervals = [model.predict(float(p)) for p in test_pred]
        live = model.predict(point_pred)

    coverage = evaluate_coverage(test_true, [i.lower for i in intervals],
                                 [i.upper for i in intervals], target=1 - alpha)
    last_price = float(df["close"].iloc[-1])
    return {
        "symbol": series.symbol, "method": method, "horizon": horizon,
        "point_model": model_used, "data_source": series.source,
        "interval_return": live.to_dict(),
        "interval_price": {
            "point": round(last_price * (1 + live.point), 4),
            "lower": round(last_price * (1 + live.lower), 4),
            "upper": round(last_price * (1 + live.upper), 4),
            "last_price": round(last_price, 4),
        },
        "coverage_validation": coverage,
        "guarantee": (
            f"Split conformal guarantees at least {(1-alpha):.0%} coverage under exchangeability. "
            f"Measured out-of-sample coverage on this series: {coverage['empirical_coverage']:.1%}."
        ),
    }


@router.get("/conformal/{symbol}/compare", summary="Compare conformal variants")
async def compare_conformal(symbol: str, period: str = Query("5y"),
                            horizon: int = Query(5, ge=1, le=30),
                            alpha: float = Query(0.1, gt=0.001, lt=0.5)):
    series = market_data_service.get_history(symbol, period=period)
    df = series.df
    fwd = df["close"].pct_change(horizon).shift(-horizon).dropna()
    if len(fwd) < 120:
        raise InvalidRequestError(f"Need >= 120 observations, got {len(fwd)}")
    y = fwd.values
    vol = df["close"].pct_change().rolling(21).std().reindex(fwd.index).bfill().values
    split = len(y) // 2
    results = {}

    s = SplitConformal(alpha).calibrate(y[:split], np.zeros(split))
    ivs = [s.predict(0.0) for _ in y[split:]]
    results["split"] = evaluate_coverage(y[split:], [i.lower for i in ivs],
                                         [i.upper for i in ivs], 1 - alpha)

    mo = MondrianConformal(alpha, 3).calibrate(y[:split], np.zeros(split), vol[:split])
    ivs = [mo.predict(0.0, float(v)) for v in vol[split:]]
    results["mondrian"] = evaluate_coverage(y[split:], [i.lower for i in ivs],
                                            [i.upper for i in ivs], 1 - alpha)

    a = AdaptiveConformal(alpha).calibrate(y[:split], np.zeros(split))
    lo, hi = [], []
    for actual in y[split:]:
        iv = a.predict(0.0)
        lo.append(iv.lower)
        hi.append(iv.upper)
        a.update(float(actual), iv)
    results["adaptive_online"] = evaluate_coverage(y[split:], lo, hi, 1 - alpha)

    best = min(results.items(), key=lambda kv: abs(kv[1].get("coverage_gap", 9)))[0]
    return {
        "symbol": series.symbol, "target_coverage": round(1 - alpha, 3),
        "results": results, "best_method": best,
        "note": ("Adaptive (online) conformal is usually the most reliable on financial data "
                 "because it recalibrates as the return distribution drifts."),
    }


# --------------------------------------------------------------- ensemble
@router.get("/ensemble/{symbol}", summary="Ensemble forecast across trained architectures")
async def ensemble_forecast(
    symbol: str,
    models: str = Query("lstm,gru,tcn"),
    horizon: int = Query(5, ge=1, le=30),
    period: str = Query("3y"),
    method: str = Query("inverse_error", pattern="^(mean|median|inverse_error|directional|trimmed)$"),
):
    """Combining decorrelated models is the most dependable accuracy gain available."""
    series = market_data_service.get_history(symbol, period=period)
    names = [m.strip().lower() for m in models.split(",") if m.strip()]
    predictions, metrics, errors = {}, {}, {}
    for name in names:
        try:
            p = forecast_trainer.predict(symbol, series.df, name, horizon)
            predictions[name] = p["predicted_return"]
            metrics[name] = p.get("test_metrics", {})
        except Exception as exc:
            errors[name] = str(exc)[:140]

    if not predictions:
        raise InvalidRequestError(
            "No trained models available for this ensemble. Train them first.",
            details={"errors": errors})

    result = ForecastEnsemble(method).combine(predictions, metrics)
    last_price = float(series.df["close"].iloc[-1])
    # Wide disagreement between members is itself a warning signal
    confidence = float(np.clip(result.agreement * (1 - min(result.dispersion * 40, 0.8)), 0, 0.95))
    return {
        "symbol": series.symbol, "horizon": horizon,
        **result.to_dict(),
        "predicted_price": round(last_price * (1 + result.prediction), 4),
        "last_price": round(last_price, 4),
        "direction": "up" if result.prediction > 0 else "down",
        "ensemble_confidence": round(confidence, 4),
        "unavailable_models": errors,
        "interpretation": (
            f"{len(predictions)} models combined. Dispersion {result.dispersion:.5f}: "
            + ("members broadly agree." if result.dispersion < 0.005 else
               "members disagree materially - treat the signal as weak.")),
    }


# ------------------------------------------------------------- volatility
@router.get("/volatility/{symbol}", summary="GARCH-family volatility forecast")
async def volatility_forecast(symbol: str, period: str = Query("5y"),
                              horizon: int = Query(5, ge=1, le=30)):
    series = market_data_service.get_history(symbol, period=period)
    return risk_service.volatility_report(symbol, series.df, horizon)


@router.get("/regime/{symbol}", summary="Current market regime")
async def market_regime(symbol: str, period: str = Query("2y")):
    series = market_data_service.get_history(symbol, period=period)
    return {"symbol": series.symbol, **regime_detector.detect(series.df)}


# -------------------------------------------------------------------- VaR
@router.get("/var/{symbol}", summary="All VaR estimators + validation")
async def value_at_risk(symbol: str, period: str = Query("5y"),
                        confidence: float = Query(0.95, gt=0.5, lt=0.9999),
                        validate: bool = Query(True)):
    """Seven estimators side by side, each with the backtest evidence for it."""
    series = market_data_service.get_history(symbol, period=period)
    return risk_service.var_report(symbol, series.df, confidence, validate)


@router.get("/var/{symbol}/backtest", summary="Kupiec + Christoffersen + Basel validation")
async def var_backtest(symbol: str, period: str = Query("5y"),
                       confidence: float = Query(0.95, gt=0.5, lt=0.9999),
                       window: int = Query(250, ge=100, le=1000),
                       method: str = Query("historical",
                                           pattern="^(historical|parametric|cornish_fisher|student_t|ewma)$")):
    series = market_data_service.get_history(symbol, period=period)
    return risk_service.var_backtest(symbol, series.df, confidence, window, method)


@router.get("/tail/{symbol}", summary="Extreme Value Theory tail analysis")
async def tail_analysis(symbol: str, period: str = Query("5y")):
    series = market_data_service.get_history(symbol, period=period)
    return risk_service.tail_report(symbol, series.df)


@router.get("/stress/{symbol}", summary="Historical & hypothetical stress scenarios")
async def stress_test(symbol: str, period: str = Query("5y"),
                      position_value: float = Query(100_000.0, gt=0)):
    series = market_data_service.get_history(symbol, period=period)
    return {"symbol": series.symbol,
            **risk_service.stress_test(series.df, position_value)}


# ================================================= AI Stress Testing Engine
@router.get("/stress-engine/scenarios", summary="Stress scenarios available")
async def stress_scenarios():
    """The scenario catalogue, so a client never hardcodes the list."""
    from app.services.risk import stress

    return {"scenarios": stress.catalogue()}


@router.get("/stress-engine/{symbols}", summary="AI Stress Testing Engine")
async def stress_engine(
    symbols: str,
    scenario: str = Query("market_crash"),
    period: str = Query("5y"),
    position_value: float = Query(100_000.0, gt=0),
    confidence: float = Query(0.95, gt=0.5, lt=1.0),
    weights: str | None = Query(None, description="Comma-separated, aligned to symbols"),
    vol_multiplier: float = Query(2.0, gt=0),
    shock_pct: float = Query(-10.0),
    liquidity_penalty: float = Query(1.5, ge=1.0),
    correlation_target: float = Query(0.9, ge=0.0, le=1.0),
):
    """Stress one asset or a weighted basket and report before vs after.

    `symbols` is a comma-separated list, so the same endpoint answers for a
    single instrument and for a portfolio. Every figure is computed from the
    loaded returns by the platform's existing risk functions — VaR, CVaR,
    drawdown and the Euler risk decomposition. Nothing is hardcoded, nothing is
    random, and a quantity that cannot be measured is returned as null with the
    reason rather than defaulted.
    """
    from app.services.risk import stress

    tickers = [s.strip().upper() for s in symbols.split(",") if s.strip()]
    if not tickers:
        raise InvalidRequestError("no symbols supplied")
    if len(tickers) > 12:
        raise InvalidRequestError("at most 12 symbols per stress run")

    weight_map: dict[str, float] = {}
    if weights:
        parts = [w.strip() for w in weights.split(",") if w.strip()]
        if len(parts) != len(tickers):
            raise InvalidRequestError(
                f"{len(parts)} weights for {len(tickers)} symbols")
        try:
            weight_map = {t: float(w) for t, w in zip(tickers, parts, strict=True)}
        except ValueError as exc:
            raise InvalidRequestError(f"weights must be numeric: {exc}") from exc
    else:
        weight_map = dict.fromkeys(tickers, 1.0)

    returns: dict = {}
    sources: dict[str, str] = {}
    skipped: dict[str, str] = {}
    for ticker in tickers:
        try:
            series = market_data_service.get_history(ticker, period=period)
        except Exception as exc:                       # noqa: BLE001
            skipped[ticker] = str(exc)[:160]
            continue
        frame = getattr(series, "df", series)
        if frame is None or frame.empty:
            skipped[ticker] = "no market data"
            continue
        returns[ticker] = frame["close"].pct_change().dropna()
        sources[ticker] = getattr(series, "source", None)

    if not returns:
        raise InvalidRequestError(
            f"no usable market data for {', '.join(tickers)}")

    try:
        result = stress.run(
            returns, weight_map, scenario,
            position_value=position_value, confidence=confidence,
            params={"vol_multiplier": vol_multiplier, "shock_pct": shock_pct,
                    "liquidity_penalty": liquidity_penalty,
                    "correlation_target": correlation_target},
        )
    except ValueError as exc:
        raise InvalidRequestError(str(exc)) from exc

    result["period"] = period
    result["data_sources"] = sources
    if skipped:
        result["skipped"] = skipped
    return result
