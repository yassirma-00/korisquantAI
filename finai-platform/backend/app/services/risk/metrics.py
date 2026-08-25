"""Portfolio risk & performance analytics (Sharpe, VaR, drawdown, optimisation)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.core.logging import get_logger
from app.utils.timeseries import TRADING_DAYS, annualise_return, annualise_vol

logger = get_logger(__name__)


# --------------------------------------------------------------- ratios
def sharpe_ratio(returns: pd.Series, risk_free: float = 0.02, periods: int = TRADING_DAYS) -> float:
    """Annualised Sharpe ratio.

    Zero-volatility input is mathematically undefined (division by zero). Rather
    than emit ``inf`` — which would poison downstream scoring and JSON responses —
    we return a large finite sentinel that preserves the sign of the excess
    return, and ``0.0`` when there is no excess return at all.
    """
    returns = pd.Series(returns).dropna()
    if len(returns) < 3:
        return 0.0
    excess = returns - risk_free / periods
    std = excess.std(ddof=1)
    if std > 1e-12:
        return float(excess.mean() / std * np.sqrt(periods))
    mean = float(excess.mean())
    if abs(mean) < 1e-12:
        return 0.0
    return float(np.sign(mean) * 99.0)


def downside_deviation(returns: pd.Series, target: float = 0.0,
                       periods: int = TRADING_DAYS) -> float:
    """Target semi-deviation: ``sqrt(mean(min(r - target, 0)^2))``, annualised.

    The textbook definition averages the squared shortfalls over **all**
    observations, not only the losing ones. Averaging over the losers alone —
    or worse, taking their standard deviation — measures the *dispersion of*
    losses rather than their *magnitude*, and those are different quantities:
    a series that loses exactly 2% every losing day has zero dispersion but a
    very real 2% downside deviation.
    """
    r = pd.Series(returns).dropna()
    if len(r) < 3:
        return 0.0
    shortfall = np.minimum(r.to_numpy() - target, 0.0)
    return float(np.sqrt(np.mean(shortfall ** 2)) * np.sqrt(periods))


def sortino_ratio(returns: pd.Series, risk_free: float = 0.02, periods: int = TRADING_DAYS) -> float:
    """Annualised Sortino ratio against the risk-free rate as the target.

    This used to divide by ``downside.std(ddof=1)`` — the standard deviation of
    the negative excess returns. That is not the downside deviation: it is how
    much the losses vary *around their own mean*, so a series whose every loss
    is identical scored 0.0 (perfectly safe) when its true Sortino was 12.4.
    It also silently returned 0.0 whenever fewer than three losing days
    existed, which reads as "no risk-adjusted return" for the best possible
    case.
    """
    returns = pd.Series(returns).dropna()
    if len(returns) < 3:
        return 0.0
    daily_target = risk_free / periods
    excess_mean = float((returns - daily_target).mean())
    dd = downside_deviation(returns, target=daily_target, periods=periods)
    if dd > 1e-12:
        return float(excess_mean * periods / dd)
    # No observation fell below the target: the ratio is unbounded, not zero.
    # Mirror sharpe_ratio's sentinel so the sign survives into the JSON.
    if abs(excess_mean) < 1e-12:
        return 0.0
    return float(np.sign(excess_mean) * 99.0)


def calmar_ratio(returns: pd.Series, periods: int = TRADING_DAYS) -> float:
    returns = pd.Series(returns).dropna()
    if returns.empty:
        return 0.0
    equity = (1 + returns).cumprod()
    mdd = float((equity / equity.cummax() - 1).min())
    ann = annualise_return(returns, periods)
    return float(ann / abs(mdd)) if mdd < -1e-9 else 0.0


def omega_ratio(returns: pd.Series, threshold: float = 0.0) -> float:
    returns = pd.Series(returns).dropna()
    gains = (returns[returns > threshold] - threshold).sum()
    losses = (threshold - returns[returns <= threshold]).sum()
    return float(gains / losses) if losses > 1e-12 else 0.0


def information_ratio(returns: pd.Series, benchmark: pd.Series, periods: int = TRADING_DAYS) -> float:
    aligned = pd.concat([pd.Series(returns), pd.Series(benchmark)], axis=1).dropna()
    if len(aligned) < 5:
        return 0.0
    active = aligned.iloc[:, 0] - aligned.iloc[:, 1]
    te = active.std(ddof=1)
    return float(active.mean() / te * np.sqrt(periods)) if te > 1e-12 else 0.0


# ----------------------------------------------------------------- risk
def value_at_risk(returns: pd.Series, confidence: float = 0.95,
                  method: str = "historical") -> float | None:
    """Return the loss quantile as a negative number, or ``None`` if unknown.

    Returning ``0.0`` for a series too short to estimate meant "this asset
    cannot lose money", which is the single most dangerous thing a risk system
    can say. ``None`` propagates to a dash in the UI instead.
    """
    returns = pd.Series(returns).dropna()
    if len(returns) < 10:
        return None
    if method == "parametric":
        from scipy.stats import norm
        return float(returns.mean() + norm.ppf(1 - confidence) * returns.std(ddof=1))
    if method == "cornish_fisher":
        from scipy.stats import norm
        z = norm.ppf(1 - confidence)
        s, k = float(returns.skew()), float(returns.kurtosis())
        z_cf = (z + (z**2 - 1) * s / 6 + (z**3 - 3*z) * k / 24 - (2*z**3 - 5*z) * s**2 / 36)
        # The Cornish-Fisher expansion is only valid while the mapping stays
        # monotone; far outside that domain it bends back and reports a
        # *smaller* loss for a more skewed, fatter-tailed series. Fall back to
        # the plain Gaussian quantile rather than publish an inverted number.
        if z_cf > z:
            z_cf = z
        return float(returns.mean() + z_cf * returns.std(ddof=1))
    return float(np.percentile(returns, (1 - confidence) * 100))


def conditional_var(returns: pd.Series, confidence: float = 0.95) -> float | None:
    """Expected shortfall: the mean loss *given* the VaR threshold is breached."""
    returns = pd.Series(returns).dropna()
    var = value_at_risk(returns, confidence)
    if var is None:
        return None
    tail = returns[returns <= var]
    return float(tail.mean()) if len(tail) else var


def drawdown_series(returns: pd.Series) -> pd.Series:
    equity = (1 + pd.Series(returns).dropna()).cumprod()
    return equity / equity.cummax() - 1


def ulcer_index(returns: pd.Series) -> float:
    dd = drawdown_series(returns)
    return float(np.sqrt((dd ** 2).mean()) * 100) if len(dd) else 0.0


def beta_alpha(returns: pd.Series, benchmark: pd.Series,
               risk_free: float = 0.02) -> tuple[float | None, float | None]:
    """CAPM beta and Jensen's alpha against a benchmark.

    Beta is ``cov(r, b) / var(b)`` measured on the **overlapping** dates only.
    Both series are normalised to tz-naive midnight first: an equity indexed at
    market close and a crypto series indexed at UTC midnight otherwise
    intersect on nothing, and an empty intersection used to return a confident
    beta of 0.0 — "moves independently of the market" — when the truth was
    "never compared".
    """
    r_raw, b_raw = pd.Series(returns).dropna(), pd.Series(benchmark).dropna()

    def _normalise(s: pd.Series) -> pd.Series:
        if isinstance(s.index, pd.DatetimeIndex):
            idx = s.index.tz_localize(None) if s.index.tz is not None else s.index
            return pd.Series(s.to_numpy(), index=idx.normalize())
        return s

    aligned = pd.concat([_normalise(r_raw), _normalise(b_raw)], axis=1,
                        join="inner").dropna()
    if len(aligned) < 20:
        return None, None
    r, b = aligned.iloc[:, 0], aligned.iloc[:, 1]
    var_b = float(b.var(ddof=1))
    if var_b <= 1e-12:
        return None, None
    beta = float(r.cov(b) / var_b)
    alpha = float(annualise_return(r) - (risk_free + beta * (annualise_return(b) - risk_free)))
    return beta, alpha


def full_metrics(returns: pd.Series, benchmark: pd.Series | None = None,
                 risk_free: float = 0.02) -> dict:
    returns = pd.Series(returns).dropna()
    if len(returns) < 3:
        return {"error": "insufficient data"}
    equity = (1 + returns).cumprod()
    dd = drawdown_series(returns)

    def _r(value: float | None, places: int) -> float | None:
        """Round, but keep an unknown quantity unknown."""
        return None if value is None else round(value, places)

    metrics = {
        "total_return": round(float(equity.iloc[-1] - 1), 4),
        "annualised_return": round(annualise_return(returns), 4),
        "annualised_volatility": round(annualise_vol(returns), 4),
        "downside_deviation": round(downside_deviation(returns, risk_free / TRADING_DAYS), 4),
        "sharpe_ratio": round(sharpe_ratio(returns, risk_free), 3),
        "sortino_ratio": round(sortino_ratio(returns, risk_free), 3),
        "calmar_ratio": round(calmar_ratio(returns), 3),
        "omega_ratio": round(omega_ratio(returns), 3),
        "max_drawdown": round(float(dd.min()), 4),
        "current_drawdown": round(float(dd.iloc[-1]), 4),
        "ulcer_index": round(ulcer_index(returns), 3),
        "var_95": _r(value_at_risk(returns, 0.95), 4),
        "var_99": _r(value_at_risk(returns, 0.99), 4),
        "cvar_95": _r(conditional_var(returns, 0.95), 4),
        "cvar_99": _r(conditional_var(returns, 0.99), 4),
        "var_95_cornish_fisher": _r(value_at_risk(returns, 0.95, "cornish_fisher"), 4),
        "skewness": round(float(returns.skew()), 3),
        "excess_kurtosis": round(float(returns.kurtosis()), 3),
        "best_day": round(float(returns.max()), 4),
        "worst_day": round(float(returns.min()), 4),
        "win_rate": round(float((returns > 0).mean()), 4),
        # No losing day makes this ratio infinite, not zero. Zero is the *worst*
        # possible profit factor, so reporting it for a flawless series inverted
        # the reading; mirror the Sharpe sentinel instead.
        "profit_factor": round(float(returns[returns > 0].sum() / abs(returns[returns < 0].sum())), 3)
                         if (returns < 0).any() else (99.0 if (returns > 0).any() else 0.0),
        "n_observations": int(len(returns)),
    }
    if benchmark is not None:
        beta, alpha = beta_alpha(returns, benchmark, risk_free)
        corr = float(pd.Series(returns).corr(pd.Series(benchmark)))
        metrics.update({
            "beta": _r(beta, 3), "alpha": _r(alpha, 4),
            "information_ratio": round(information_ratio(returns, benchmark), 3),
            "correlation_to_benchmark": None if pd.isna(corr) else round(corr, 3),
        })
    return metrics


# ------------------------------------------------------------ portfolio
def correlation_matrix(returns_matrix: pd.DataFrame) -> dict:
    corr = returns_matrix.corr()
    return {
        "symbols": list(corr.columns),
        "matrix": [[round(float(v), 4) for v in row] for row in corr.values],
        "average_correlation": round(float(corr.values[np.triu_indices_from(corr.values, k=1)].mean()), 4),
        "highest_pair": _extreme_pair(corr, highest=True),
        "lowest_pair": _extreme_pair(corr, highest=False),
    }


def _extreme_pair(corr: pd.DataFrame, highest: bool) -> dict:
    values = corr.values.copy()
    np.fill_diagonal(values, np.nan if highest else np.nan)
    if np.all(np.isnan(values)):
        return {}
    idx = np.nanargmax(values) if highest else np.nanargmin(values)
    i, j = divmod(int(idx), values.shape[1])
    return {"pair": [corr.columns[i], corr.columns[j]], "correlation": round(float(values[i, j]), 4)}


def risk_contribution(weights: np.ndarray, cov: np.ndarray) -> np.ndarray:
    port_vol = float(np.sqrt(weights @ cov @ weights))
    if port_vol < 1e-12:
        return np.zeros_like(weights)
    marginal = cov @ weights / port_vol
    return weights * marginal / port_vol


def optimise_portfolio(returns_matrix: pd.DataFrame, objective: str = "max_sharpe",
                       risk_free: float = 0.02, target_return: float | None = None,
                       allow_short: bool = False, n_simulations: int = 20_000,
                       max_weight: float | None = None) -> dict:
    """Mean-variance optimisation via random search + SLSQP refinement.

    ``max_weight`` caps any single position (e.g. 0.4) to avoid the degenerate
    corner solutions that unconstrained mean-variance optimisation produces on
    small, highly-correlated universes.
    """
    from scipy.optimize import minimize

    mu = returns_matrix.mean().values * TRADING_DAYS
    cov = returns_matrix.cov().values * TRADING_DAYS
    n = len(mu)
    rng = np.random.default_rng(42)

    def portfolio_stats(w: np.ndarray) -> tuple[float, float, float]:
        ret = float(w @ mu)
        vol = float(np.sqrt(max(w @ cov @ w, 1e-12)))
        return ret, vol, (ret - risk_free) / vol if vol > 1e-9 else 0.0

    # ------------- Monte-Carlo efficient frontier (also used for the chart)
    sims = rng.dirichlet(np.ones(n), size=n_simulations) if not allow_short else \
        rng.normal(0, 0.4, size=(n_simulations, n))
    if allow_short:
        sims = sims / np.abs(sims).sum(axis=1, keepdims=True)
    sim_ret = sims @ mu
    sim_vol = np.sqrt(np.einsum("ij,jk,ik->i", sims, cov, sims))
    sim_sharpe = np.where(sim_vol > 1e-9, (sim_ret - risk_free) / sim_vol, 0)

    objectives = {
        "max_sharpe": lambda w: -portfolio_stats(w)[2],
        "min_volatility": lambda w: portfolio_stats(w)[1],
        "max_return": lambda w: -portfolio_stats(w)[0],
        "risk_parity": lambda w: float(np.sum((risk_contribution(w, cov) - 1 / n) ** 2)),
    }
    if objective not in objectives:
        objective = "max_sharpe"

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    if target_return is not None:
        constraints.append({"type": "eq", "fun": lambda w: float(w @ mu) - target_return})

    # Cap concentration: default allows at most 40% in one name (never below 1/n)
    cap = max_weight if max_weight is not None else max(0.40, 1.0 / n)
    cap = float(np.clip(cap, 1.0 / n, 1.0))
    bounds = [(-1.0, cap) if allow_short else (0.0, cap)] * n

    best_sim = sims[int(np.argmax(sim_sharpe))] if objective == "max_sharpe" else \
        sims[int(np.argmin(sim_vol))] if objective == "min_volatility" else np.ones(n) / n
    result = minimize(objectives[objective], best_sim, method="SLSQP",
                      bounds=bounds, constraints=constraints,
                      options={"maxiter": 500, "ftol": 1e-9})
    weights = result.x if result.success else best_sim
    weights = np.clip(weights, -1 if allow_short else 0, cap)
    weights = weights / weights.sum()
    # Re-project if normalisation pushed a weight back above the cap
    for _ in range(8):
        excess = weights - cap
        if (excess <= 1e-6).all():
            break
        overflow = float(excess[excess > 0].sum())
        weights = np.minimum(weights, cap)
        room = cap - weights
        room_total = float(room.sum())
        if room_total <= 1e-9:
            break
        weights = weights + room / room_total * overflow

    ret, vol, sharpe = portfolio_stats(weights)
    rc = risk_contribution(weights, cov)

    # Efficient frontier sample for plotting
    order = np.argsort(sim_vol)
    frontier = []
    step = max(len(order) // 60, 1)
    best_r = -np.inf
    for k in order[::step]:
        if sim_ret[k] > best_r:
            best_r = sim_ret[k]
            frontier.append({"volatility": round(float(sim_vol[k]), 4),
                             "return": round(float(sim_ret[k]), 4),
                             "sharpe": round(float(sim_sharpe[k]), 3)})

    return {
        "objective": objective,
        "symbols": list(returns_matrix.columns),
        "weights": {sym: round(float(w), 4) for sym, w in zip(returns_matrix.columns, weights, strict=False)},
        "expected_annual_return": round(ret, 4),
        "expected_annual_volatility": round(vol, 4),
        "sharpe_ratio": round(sharpe, 3),
        "diversification_ratio": round(float(
            (weights @ np.sqrt(np.diag(cov))) / vol) if vol > 1e-9 else 0.0, 3),
        "risk_contribution": {sym: round(float(r), 4) for sym, r in zip(returns_matrix.columns, rc, strict=False)},
        "efficient_frontier": frontier,
        "converged": bool(result.success),
    }
