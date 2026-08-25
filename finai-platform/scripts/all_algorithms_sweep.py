#!/usr/bin/env python3
"""Evaluate every *non-RL* algorithm family the project ships.

The RL catalogue (13 algorithms) is covered by `multiseed_study.py`. This
script covers the four remaining families that the platform advertises but the
report did not yet measure:

  * 5 deep-learning forecasters   (LSTM, GRU, TCN, Transformer, CNN-LSTM)
  * 8 value-at-risk estimators    (historical ... extreme-value theory)
  * 3 GARCH-family volatility models (GARCH, EGARCH, GJR)
  * 4 rule-based strategy benchmarks (buy&hold, MA crossover, momentum, RSI)

Design notes
------------
* Nothing is trained or fitted twice: forecaster metrics are read from the
  registry the platform already maintains, so this script reports what the
  product itself would report.
* A family that cannot be evaluated is written with an explicit `reason`
  rather than omitted, so a gap in the report is visible rather than silent.
* Output: data/artifacts/all_algorithms.json
"""
from __future__ import annotations

import json
import pathlib
import sys
import time
import warnings

warnings.filterwarnings("ignore")

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

ART = ROOT / "data" / "artifacts"

SYMBOL = "AAPL"
PERIOD = "5y"


def _prices():
    from app.services.data.market_data import market_data_service as md
    series = md.get_history(SYMBOL, period=PERIOD)
    return getattr(series, "df", series)


# --------------------------------------------------------------- forecasters
def forecasters() -> dict:
    """Held-out metrics for every trained deep-learning architecture.

    Read from the platform's own registry rather than retrained here: the point
    is to report what the product reports.
    """
    from app.core.config import settings
    from app.services.forecasting.models import MODEL_REGISTRY

    known = list(MODEL_REGISTRY)
    rows, missing = [], []

    # Same source the /forecast/trained endpoint reads: the checkpoint metadata
    # written at training time. Going through the files (rather than retraining)
    # is what makes this report the product's own numbers.
    index = {}
    for meta_path in sorted(settings.MODEL_DIR.glob("forecast_*.json")):
        payload = json.loads(meta_path.read_text())
        cfg = payload.get("config", {})
        index[(payload.get("symbol"), cfg.get("model"))] = {
            "symbol": payload.get("symbol"),
            "model": cfg.get("model"),
            "horizon": cfg.get("horizon"),
            # The checkpoint stores held-out metrics under metrics.test, not
            # test_metrics. Reading the wrong key silently produced a table of
            # None values rather than an error, so the fallback chain is
            # explicit and a miss is asserted below.
            "test_metrics": (payload.get("metrics", {}).get("test")
                             or payload.get("test_metrics") or {}),
        }

    for arch in known:
        hit = index.get((SYMBOL, arch))
        if hit is None:
            missing.append(arch)
            continue
        m = hit.get("test_metrics") or {}
        if not m:
            missing.append(f"{arch} (checkpoint present, metrics unreadable)")
            continue
        rows.append({
            "model": arch,
            "symbol": hit["symbol"],
            "horizon": hit.get("horizon"),
            "directional_accuracy": m.get("directional_accuracy"),
            "r2": m.get("r2"),
            "rmse": m.get("rmse"),
            "mae": m.get("mae"),
            "n_samples": m.get("n_samples"),
        })
    return {"family": "deep forecasters", "n_known": len(known),
            "rows": rows,
            "not_trained": missing,
            "note": ("metrics come from the platform's model registry; "
                     "architectures without a trained checkpoint for this "
                     "symbol are listed rather than silently dropped")}


# ----------------------------------------------------------------------- VaR
def var_estimators() -> dict:
    from app.services.risk.advanced_var import comprehensive_var

    df = _prices()
    returns = df["close"].pct_change().dropna()
    out = comprehensive_var(returns, confidence=0.95)

    rows = []
    validation = (out.get("validation") or {}).get("methods", {})
    for name, est in (out.get("estimates") or {}).items():
        v = validation.get(name) or validation.get(name.split("_")[0]) or {}
        rows.append({
            "estimator": name,
            "var": est.get("var"),
            "cvar": est.get("cvar"),
            "breach_rate": v.get("breach_rate"),
            "kupiec_p": v.get("kupiec_p"),
            "independence_p": v.get("independence_p"),
            "basel_zone": v.get("basel_zone"),
        })
    return {"family": "value at risk", "n": len(rows), "rows": rows,
            "n_observations": out.get("n_observations"),
            "recommended": (out.get("validation") or {}).get("recommended"),
            "rationale": (out.get("validation") or {}).get("rationale")}


# --------------------------------------------------------------------- GARCH
def garch_models() -> dict:
    from app.services.forecasting.advanced import VolatilityForecaster

    df = _prices()
    returns = df["close"].pct_change().dropna()   # fit() rescales internally

    rows = []
    for name in VolatilityForecaster.MODELS:
        t0 = time.perf_counter()
        try:
            fitted = VolatilityForecaster(model=name, dist="t").fit(returns).fitted
        except Exception as exc:                      # noqa: BLE001
            rows.append({"model": name, "error": str(exc)})
            continue
        params = dict(fitted.params)
        leverage = params.get("gamma[1]", params.get("gamma"))
        rows.append({
            "model": name,
            "aic": round(float(fitted.aic), 1),
            "bic": round(float(fitted.bic), 1),
            "loglikelihood": round(float(fitted.loglikelihood), 1),
            "nu": round(float(params.get("nu", float("nan"))), 2),
            "leverage": (round(float(leverage), 4) if leverage is not None else None),
            "seconds": round(time.perf_counter() - t0, 2),
        })
    ok = [r for r in rows if "aic" in r]
    best = min(ok, key=lambda r: r["aic"])["model"] if ok else None
    return {"family": "garch volatility", "n": len(rows), "rows": rows,
            "best_by_aic": best}


# ---------------------------------------------------------------- strategies
def strategies() -> dict:
    from app.services.recommendation.intelligence import StrategyBenchmarks

    df = _prices()
    result = StrategyBenchmarks(0.001, 0.0005).compare_all(df["close"], 100_000.0)
    rows = [{
        "strategy": s["strategy"],
        "label": s["label"],
        "total_return": s["total_return"],
        "sharpe_ratio": s["sharpe_ratio"],
        "sortino_ratio": s["sortino_ratio"],
        "max_drawdown": s["max_drawdown"],
        "n_trades": s["n_trades"],
    } for s in result["strategies"]]
    return {"family": "rule-based strategies", "n": len(rows), "rows": rows,
            "best_by_sharpe": result.get("best_by_sharpe"),
            "cost_model": result.get("cost_model")}


def main() -> int:
    payload = {"symbol": SYMBOL, "period": PERIOD, "families": {}}
    for name, fn in (("forecasters", forecasters),
                     ("var", var_estimators),
                     ("garch", garch_models),
                     ("strategies", strategies)):
        print(f"[{name}] …", flush=True)
        try:
            payload["families"][name] = fn()
        except Exception as exc:                      # noqa: BLE001
            payload["families"][name] = {"error": f"{type(exc).__name__}: {exc}"}
            print(f"    FAILED: {exc}", flush=True)
        else:
            block = payload["families"][name]
            print(f"    {block.get('n', len(block.get('rows', [])))} entries", flush=True)

    ART.mkdir(parents=True, exist_ok=True)
    out = ART / "all_algorithms.json"
    out.write_text(json.dumps(payload, indent=2, default=str))
    print(f"\nwritten: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
