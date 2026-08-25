"""Rigorous Value-at-Risk: estimation, *and* statistical validation.

The problem with most VaR implementations
-----------------------------------------
Historical VaR assumes tomorrow resembles a naive average of the past. It reacts
far too slowly to volatility regime changes, so it systematically understates
risk exactly when risk matters. And it is almost never backtested, so nobody
notices.

This module fixes both halves:

**Better estimators**
* Filtered Historical Simulation (FHS) — devolatilise returns with a GARCH fit,
  bootstrap the standardised residuals, then re-volatilise at *today's* level.
  This is the Basel-recommended approach: it keeps the empirical fat tails while
  reacting immediately to the current regime.
* Extreme Value Theory (POT / Generalised Pareto) — models the tail itself
  rather than interpolating between the few observations that exist out there.
* Cornish-Fisher — skew/kurtosis-adjusted parametric VaR.
* Monte Carlo with Student-t innovations.

**Proper validation** (this is what makes the numbers trustworthy)
* Kupiec POF test — is the *number* of breaches consistent with the level?
* Christoffersen independence test — are breaches clustered (a model that fails
  repeatedly in a crisis is useless even if the annual count looks right)?
* Combined conditional-coverage test.
* Basel traffic-light zones (green / yellow / red).

A VaR number without a backtest is an opinion. With one, it is a measurement.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from app.core.logging import get_logger

logger = get_logger(__name__)

try:  # pragma: no cover - optional
    from arch import arch_model
    ARCH_AVAILABLE = True
except Exception:  # pragma: no cover
    ARCH_AVAILABLE = False


# ================================================================ estimators
def var_historical(returns: pd.Series, confidence: float = 0.95) -> float:
    r = pd.Series(returns).dropna()
    return float(np.percentile(r, (1 - confidence) * 100)) if len(r) >= 20 else 0.0


def var_parametric(returns: pd.Series, confidence: float = 0.95) -> float:
    r = pd.Series(returns).dropna()
    if len(r) < 20:
        return 0.0
    return float(r.mean() + stats.norm.ppf(1 - confidence) * r.std(ddof=1))


def var_cornish_fisher(returns: pd.Series, confidence: float = 0.95) -> float:
    """Parametric VaR corrected for skewness and excess kurtosis."""
    r = pd.Series(returns).dropna()
    if len(r) < 30:
        return 0.0
    z = stats.norm.ppf(1 - confidence)
    s, k = float(r.skew()), float(r.kurtosis())
    z_cf = (z + (z ** 2 - 1) * s / 6 + (z ** 3 - 3 * z) * k / 24
            - (2 * z ** 3 - 5 * z) * s ** 2 / 36)
    return float(r.mean() + z_cf * r.std(ddof=1))


def var_student_t(returns: pd.Series, confidence: float = 0.95) -> float:
    """Fit a Student-t and read off the quantile - respects fat tails."""
    r = pd.Series(returns).dropna()
    if len(r) < 30:
        return 0.0
    try:
        nu, loc, scale = stats.t.fit(r.values)
        return float(stats.t.ppf(1 - confidence, nu, loc=loc, scale=scale))
    except Exception:
        return var_parametric(r, confidence)


def var_ewma(returns: pd.Series, confidence: float = 0.95, lam: float = 0.94) -> float:
    """RiskMetrics EWMA volatility + Student-t quantile.

    Exponentially weighted volatility reacts to a regime change within days
    instead of waiting for a 250-day window to roll over. This is the cheapest
    fix for the breach-clustering that defeats static estimators.
    """
    r = pd.Series(returns).dropna()
    if len(r) < 60:
        return var_historical(r, confidence)
    ew_var = r.ewm(alpha=1 - lam).var(bias=True).iloc[-1]
    sigma = float(np.sqrt(max(float(ew_var), 1e-12)))
    try:
        nu = max(float(stats.t.fit(r.values)[0]), 2.5)
        q = float(stats.t.ppf(1 - confidence, nu)) / np.sqrt(nu / (nu - 2))
    except Exception:
        q = float(stats.norm.ppf(1 - confidence))
    return float(r.mean() + q * sigma)


def var_filtered_historical(returns: pd.Series, confidence: float = 0.95,
                            n_simulations: int = 10_000, horizon: int = 1) -> dict:
    """Filtered Historical Simulation — the regime-aware workhorse.

    1. Fit GJR-GARCH to capture volatility clustering *and* the leverage effect.
    2. Standardise: z_t = r_t / sigma_t  (removes the volatility signal, keeps the shape).
    3. Bootstrap z, rescale by the *forecast* sigma for tomorrow.

    The result adapts within days of a regime change, unlike plain historical VaR.
    """
    r = pd.Series(returns).dropna()
    if len(r) < 250 or not ARCH_AVAILABLE:
        return {"var": var_historical(r, confidence), "method": "historical_fallback",
                "reason": "insufficient data" if len(r) < 250 else "arch not installed"}
    try:
        scaled = r * 100
        am = arch_model(scaled, mean="Constant", vol="GARCH", p=1, o=1, q=1, dist="t")
        fitted = am.fit(disp="off", show_warning=False)
        sigma = pd.Series(fitted.conditional_volatility, index=r.index).replace(0, np.nan)
        z = (scaled - float(fitted.params.get("mu", 0.0))) / sigma
        z = z.dropna().values
        if len(z) < 100:
            return {"var": var_historical(r, confidence), "method": "historical_fallback"}

        f = fitted.forecast(horizon=horizon, reindex=False, method="simulation", simulations=500)
        sigma_next = float(np.sqrt(f.variance.values[-1][0]))

        rng = np.random.default_rng(42)
        draws = rng.choice(z, size=(n_simulations, horizon), replace=True)
        sim = (draws * sigma_next).sum(axis=1) / 100.0     # back to decimal returns

        var = float(np.percentile(sim, (1 - confidence) * 100))
        cvar = float(sim[sim <= var].mean()) if (sim <= var).any() else var
        current_vol = float(sigma.iloc[-1]) / 100
        avg_vol = float(sigma.mean()) / 100
        return {
            "var": var, "cvar": cvar, "method": "filtered_historical_simulation",
            "forecast_volatility_daily": round(sigma_next / 100, 6),
            "current_vol_vs_average": round(current_vol / avg_vol, 3) if avg_vol else None,
            "n_simulations": n_simulations, "horizon": horizon,
            "note": ("Volatility-adjusted: reacts to the current regime rather than "
                     "averaging over years of unrelated conditions."),
        }
    except Exception as exc:
        logger.warning("FHS failed: %s", exc)
        return {"var": var_historical(r, confidence), "method": "historical_fallback",
                "error": str(exc)[:160]}


def var_extreme_value(returns: pd.Series, confidence: float = 0.99,
                      threshold_pct: float = 10.0) -> dict:
    """Peaks-Over-Threshold with a Generalised Pareto tail.

    At 99%+ there are too few observations to read a quantile off the empirical
    distribution. EVT fits the *shape* of the tail instead, which is the only
    statistically sound way to extrapolate there.
    """
    r = pd.Series(returns).dropna()
    if len(r) < 250:
        return {"var": var_historical(r, confidence), "method": "historical_fallback",
                "reason": "need >= 250 observations"}
    losses = -r.values
    u = np.percentile(losses, 100 - threshold_pct)
    excesses = losses[losses > u] - u
    if len(excesses) < 25:
        return {"var": var_historical(r, confidence), "method": "historical_fallback",
                "reason": "too few tail exceedances"}
    try:
        xi, _, beta = stats.genpareto.fit(excesses, floc=0)
        n, nu = len(losses), len(excesses)
        p = 1 - confidence
        if abs(xi) < 1e-6:
            var_loss = u + beta * np.log((nu / n) / p)
        else:
            var_loss = u + (beta / xi) * (((n / nu) * p) ** (-xi) - 1)
        cvar_loss = (var_loss + (beta + xi * (var_loss - u)) / (1 - xi)) if xi < 1 else var_loss * 1.4

        # xi > 0 means a heavy (power-law) tail: extreme losses are far more
        # likely than a normal distribution would ever suggest.
        tail_type = ("heavy (power-law)" if xi > 0.05 else
                     "exponential" if abs(xi) <= 0.05 else "bounded")
        return {
            "var": float(-var_loss), "cvar": float(-cvar_loss),
            "method": "extreme_value_theory_pot",
            "shape_xi": round(float(xi), 4), "scale_beta": round(float(beta), 6),
            "threshold": round(float(-u), 6), "n_exceedances": int(nu),
            "tail_type": tail_type,
            "finite_variance": bool(xi < 0.5),
            "note": ("Tail modelled parametrically - appropriate for 99%+ where the "
                     "empirical sample is too thin to interpolate."),
        }
    except Exception as exc:
        logger.warning("EVT failed: %s", exc)
        return {"var": var_historical(r, confidence), "method": "historical_fallback",
                "error": str(exc)[:160]}


def var_monte_carlo(returns: pd.Series, confidence: float = 0.95,
                    n_simulations: int = 50_000, horizon: int = 1) -> dict:
    r = pd.Series(returns).dropna()
    if len(r) < 60:
        return {"var": 0.0, "method": "insufficient_data"}
    try:
        nu, loc, scale = stats.t.fit(r.values)
        nu = max(float(nu), 2.5)
    except Exception:
        nu, loc, scale = 5.0, float(r.mean()), float(r.std())
    rng = np.random.default_rng(42)
    sims = stats.t.rvs(nu, loc=loc, scale=scale, size=(n_simulations, horizon), random_state=rng).sum(axis=1)
    var = float(np.percentile(sims, (1 - confidence) * 100))
    cvar = float(sims[sims <= var].mean()) if (sims <= var).any() else var
    return {"var": var, "cvar": cvar, "method": "monte_carlo_student_t",
            "degrees_of_freedom": round(nu, 2), "n_simulations": n_simulations,
            "horizon": horizon}


# =============================================================== backtesting
def kupiec_pof_test(breaches: np.ndarray, confidence: float = 0.95) -> dict:
    """Unconditional coverage: is the breach *rate* right?"""
    breaches = np.asarray(breaches, dtype=bool)
    n, x = len(breaches), int(breaches.sum())
    p = 1 - confidence
    if n < 30:
        return {"test": "kupiec_pof", "error": "need >= 30 observations"}
    expected = n * p
    if x == 0:
        lr = -2 * n * np.log(1 - p)
    else:
        pi = x / n
        lr = -2 * ((n - x) * np.log(1 - p) + x * np.log(p)
                   - (n - x) * np.log(1 - pi) - x * np.log(pi))
    p_value = float(1 - stats.chi2.cdf(lr, df=1))
    return {
        "test": "kupiec_pof", "n_observations": n, "n_breaches": x,
        "expected_breaches": round(expected, 1),
        "breach_rate": round(x / n, 4), "expected_rate": round(p, 4),
        "lr_statistic": round(float(lr), 4), "p_value": round(p_value, 4),
        "reject_at_5pct": bool(p_value < 0.05),
        "verdict": ("PASS - breach frequency is consistent with the stated level"
                    if p_value >= 0.05 else
                    f"FAIL - {'too many' if x > expected else 'too few'} breaches; "
                    f"the model {'understates' if x > expected else 'overstates'} risk"),
    }


def christoffersen_independence_test(breaches: np.ndarray) -> dict:
    """Are breaches independent, or do they cluster?

    Clustering is the dangerous failure mode: a model can have the right annual
    breach count yet fail every single day of a crash.
    """
    b = np.asarray(breaches, dtype=int)
    if len(b) < 30:
        return {"test": "christoffersen_independence", "error": "need >= 30 observations"}
    n00 = n01 = n10 = n11 = 0
    for prev, cur in zip(b[:-1], b[1:], strict=False):
        if prev == 0 and cur == 0:
            n00 += 1
        elif prev == 0 and cur == 1:
            n01 += 1
        elif prev == 1 and cur == 0:
            n10 += 1
        else:
            n11 += 1

    if (n01 + n11) == 0 or (n00 + n01) == 0 or (n10 + n11) == 0:
        return {"test": "christoffersen_independence", "p_value": 1.0,
                "reject_at_5pct": False, "verdict": "PASS - too few breaches to detect clustering",
                "transitions": {"n00": n00, "n01": n01, "n10": n10, "n11": n11}}

    pi01 = n01 / (n00 + n01)
    pi11 = n11 / (n10 + n11)
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)
    def _safe_log(v: float) -> float:
        return float(np.log(v)) if v > 0 else 0.0
    lr = -2 * (((n00 + n10) * _safe_log(1 - pi) + (n01 + n11) * _safe_log(pi))
               - (n00 * _safe_log(1 - pi01) + n01 * _safe_log(pi01)
                  + n10 * _safe_log(1 - pi11) + n11 * _safe_log(pi11)))
    p_value = float(1 - stats.chi2.cdf(max(lr, 0), df=1))
    return {
        "test": "christoffersen_independence",
        "p_breach_after_calm": round(float(pi01), 4),
        "p_breach_after_breach": round(float(pi11), 4),
        "lr_statistic": round(float(max(lr, 0)), 4), "p_value": round(p_value, 4),
        "reject_at_5pct": bool(p_value < 0.05),
        "transitions": {"n00": n00, "n01": n01, "n10": n10, "n11": n11},
        "verdict": ("PASS - breaches are independent"
                    if p_value >= 0.05 else
                    "FAIL - breaches cluster; the model misses sustained stress periods"),
    }


def basel_traffic_light(n_breaches: int, n_observations: int, confidence: float = 0.99) -> dict:
    """Basel Committee zone classification (250 trading days).

    The published 4/9 thresholds are defined **only for 99% VaR**, where 2.5
    breaches per 250 days are expected. Applying them verbatim to a 95% model
    (12.5 expected breaches) would flag every correctly-calibrated model as
    "red". For other levels we therefore derive equivalent zone boundaries from
    the same binomial cumulative probabilities Basel used (95% and 99.99%).
    """
    n_obs = max(n_observations, 1)
    scaled = n_breaches * (250 / n_obs)
    p = 1 - confidence

    if abs(confidence - 0.99) < 1e-9:
        green_max, yellow_max = 4, 9
        basis = "official Basel thresholds (99% VaR)"
    else:
        # Same construction as Basel: green = up to the 95th percentile of the
        # binomial breach count, yellow = up to the 99.99th percentile.
        green_max = int(stats.binom.ppf(0.95, 250, p))
        yellow_max = int(stats.binom.ppf(0.9999, 250, p))
        basis = f"binomial-equivalent thresholds derived for {confidence:.0%} VaR"

    if scaled <= green_max:
        zone, mult, action = "green", 3.00, "Model accepted - no capital penalty"
    elif scaled <= yellow_max:
        zone, mult, action = "yellow", 3.40, "Increased scrutiny; capital multiplier raised"
    else:
        zone, mult, action = "red", 4.00, "Model rejected - must be revised"

    return {
        "zone": zone, "scaled_breaches_per_250d": round(scaled, 1),
        "expected_per_250d": round(250 * p, 1),
        "green_threshold": green_max, "yellow_threshold": yellow_max,
        "capital_multiplier": mult, "action": action, "threshold_basis": basis,
    }


def backtest_var(returns: pd.Series, confidence: float = 0.95, window: int = 250,
                 method: str = "historical") -> dict:
    """Rolling out-of-sample VaR backtest with the full test battery.

    At each step VaR is estimated using ONLY prior data, then compared with the
    return that actually occurred. This is the only way to know whether a risk
    number means anything.
    """
    r = pd.Series(returns).dropna()
    if len(r) < window + 60:
        return {"error": f"need >= {window + 60} observations, got {len(r)}"}

    estimators = {
        "historical": lambda x: var_historical(x, confidence),
        "parametric": lambda x: var_parametric(x, confidence),
        "cornish_fisher": lambda x: var_cornish_fisher(x, confidence),
        "student_t": lambda x: var_student_t(x, confidence),
        "ewma": lambda x: var_ewma(x, confidence),
    }
    estimate = estimators.get(method, estimators["historical"])

    var_series, actuals, dates = [], [], []
    values = r.values
    for i in range(window, len(values)):
        var_series.append(estimate(pd.Series(values[i - window:i])))
        actuals.append(values[i])
        dates.append(r.index[i])

    var_arr = np.array(var_series)
    act_arr = np.array(actuals)
    breaches = act_arr < var_arr

    kupiec = kupiec_pof_test(breaches, confidence)
    indep = christoffersen_independence_test(breaches)
    cc_p = None
    if "lr_statistic" in kupiec and "lr_statistic" in indep:
        lr_cc = kupiec["lr_statistic"] + indep["lr_statistic"]
        cc_p = float(1 - stats.chi2.cdf(lr_cc, df=2))

    breach_losses = act_arr[breaches] - var_arr[breaches]
    return {
        "method": method, "confidence_level": confidence, "window": window,
        "n_observations": int(len(act_arr)),
        "n_breaches": int(breaches.sum()),
        "breach_rate": round(float(breaches.mean()), 4),
        "expected_rate": round(1 - confidence, 4),
        "kupiec_test": kupiec,
        "independence_test": indep,
        "conditional_coverage_p_value": round(cc_p, 4) if cc_p is not None else None,
        "basel": basel_traffic_light(int(breaches.sum()), len(act_arr), confidence),
        "average_var": round(float(var_arr.mean()), 5),
        "worst_breach": round(float(breach_losses.min()), 5) if len(breach_losses) else None,
        "average_breach_severity": round(float(breach_losses.mean()), 5) if len(breach_losses) else None,
        "model_valid": bool(not kupiec.get("reject_at_5pct", True)
                            and not indep.get("reject_at_5pct", True)),
        "first_date": str(dates[0].date()), "last_date": str(dates[-1].date()),
    }


def compare_var_methods(returns: pd.Series, confidence: float = 0.95,
                        window: int = 250) -> dict:
    """Rank estimators by backtest quality, not by which gives the nicest number."""
    results = {}
    for method in ("historical", "parametric", "cornish_fisher", "student_t", "ewma"):
        try:
            bt = backtest_var(returns, confidence, window, method)
            if "error" not in bt:
                results[method] = {
                    "breach_rate": bt["breach_rate"],
                    "expected_rate": bt["expected_rate"],
                    "kupiec_p": bt["kupiec_test"].get("p_value"),
                    "independence_p": bt["independence_test"].get("p_value"),
                    "basel_zone": bt["basel"]["zone"],
                    "average_var": bt["average_var"],
                    "valid": bt["model_valid"],
                }
        except Exception as exc:
            results[method] = {"error": str(exc)[:120]}

    valid = {k: v for k, v in results.items() if v.get("valid")}
    # Among valid models prefer the least conservative (least wasted capital)
    best = (max(valid.items(), key=lambda kv: kv[1]["average_var"])[0] if valid else None)
    return {
        "confidence_level": confidence, "window": window, "methods": results,
        "recommended": best,
        "rationale": (f"'{best}' passes both coverage tests while tying up the least capital."
                      if best else
                      "No estimator passed both tests - the return distribution is unstable; "
                      "prefer filtered historical simulation or EVT."),
    }


def comprehensive_var(returns: pd.Series, confidence: float = 0.95) -> dict:
    """All estimators side by side, plus the validated recommendation."""
    r = pd.Series(returns).dropna()
    if len(r) < 100:
        return {"error": "need >= 100 observations"}

    estimates = {
        "historical": {"var": var_historical(r, confidence)},
        "parametric_normal": {"var": var_parametric(r, confidence)},
        "cornish_fisher": {"var": var_cornish_fisher(r, confidence)},
        "student_t": {"var": var_student_t(r, confidence)},
        "ewma_riskmetrics": {"var": var_ewma(r, confidence), "method": "ewma_student_t"},
        "monte_carlo": var_monte_carlo(r, confidence),
        "filtered_historical": var_filtered_historical(r, confidence),
        "extreme_value": var_extreme_value(r, max(confidence, 0.975)),
    }
    for v in estimates.values():
        if "var" in v and v["var"] is not None:
            v["var"] = round(float(v["var"]), 5)
        if v.get("cvar") is not None:
            v["cvar"] = round(float(v["cvar"]), 5)

    spread = [v["var"] for v in estimates.values() if v.get("var")]
    return {
        "confidence_level": confidence,
        "n_observations": int(len(r)),
        "estimates": estimates,
        "range": {"most_conservative": round(min(spread), 5),
                  "least_conservative": round(max(spread), 5),
                  "disagreement": round(max(spread) - min(spread), 5)} if spread else {},
        "validation": compare_var_methods(r, confidence) if len(r) >= 310 else
                      {"note": "need >= 310 observations to backtest"},
        "interpretation": (
            "Where estimators disagree widely the tail is poorly determined - "
            "size positions off the most conservative figure."),
    }
