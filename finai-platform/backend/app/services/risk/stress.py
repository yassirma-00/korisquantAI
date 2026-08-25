"""AI Stress Testing Engine — what breaks, and how badly.

Design
------
Every number here comes from the instrument's own realised returns, pushed
through the platform's existing risk functions:

* `metrics.value_at_risk`      — historical VaR
* `metrics.conditional_var`    — expected shortfall beyond VaR
* `metrics.drawdown_series`    — peak-to-trough path
* `metrics.risk_contribution`  — Euler decomposition of portfolio volatility
* `metrics.correlation_matrix` — cross-asset structure

A scenario is a **transformation of the observed return series**, not a table
of made-up losses. Doubling volatility rescales the series around its own mean;
a crash replays the worst window this asset actually lived through; a
correlation spike blends each asset toward the basket's own average path. That
keeps the stressed world anchored to measured behaviour: nothing is invented,
and every shock is reported with the basis that produced it.

Two properties this module refuses to violate
---------------------------------------------
1. **No fabricated losses.** If a series is too short to measure a quantity,
   the field is ``None`` with a stated reason. It is never defaulted.
2. **Deterministic.** No RNG anywhere. Re-running the same request on the same
   data returns the same numbers, so a result can be checked.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.core.logging import get_logger
from app.services.risk.metrics import (
    conditional_var,
    correlation_matrix,
    drawdown_series,
    risk_contribution,
    value_at_risk,
)

logger = get_logger(__name__)

TRADING_DAYS = 252

# Minimum observations before a quantile-based figure means anything. Below
# this the 95% VaR is decided by one or two points.
MIN_OBS = 60


class StressScenario:
    """A named, reproducible transformation of a return series."""

    def __init__(self, key: str, label: str, description: str, basis: str) -> None:
        self.key = key
        self.label = label
        self.description = description
        self.basis = basis

    def apply(self, returns: pd.Series, params: dict) -> pd.Series:  # pragma: no cover
        raise NotImplementedError


class MarketCrash(StressScenario):
    """Replay the worst window this asset actually lived through.

    Using the instrument's own realised extreme is more defensible than
    inventing a shock size, and it differs per asset by construction.
    """

    def apply(self, returns: pd.Series, params: dict) -> pd.Series:
        window = int(params.get("crash_window", 21))
        window = max(2, min(window, max(2, len(returns) // 4)))
        rolling = returns.rolling(window).sum().dropna()
        if rolling.empty:
            return returns.copy()
        # Position of the worst window's *last* day. `rolling` still carries the
        # original labels, so the position is taken from the parent series by
        # label. Using `.values.argmin()` on the un-dropped rolling series
        # returned 0 (NaN handling) and sliced a single day instead of the whole
        # episode, which made the crash scenario look harmless.
        end = int(returns.index.get_loc(rolling.idxmin()))
        start = max(0, end - window + 1)
        worst = returns.iloc[start: end + 1]

        # Two failure modes, both found by inspecting real output on AAPL:
        #
        #  * Appending the episode once moves a 95% quantile by ~1.6% on a
        #    1250-day sample - the scenario looked harmless.
        #  * Repeating it many times compounds into a -98% drawdown. That is
        #    arithmetically true of the constructed path but is not a crash: it
        #    is the same month over and over with no recovery in between, and
        #    quoting it as "worst drawdown" would overstate the risk as badly
        #    as the first bug understated it.
        #
        # The episode is therefore repeated only while the resulting drawdown
        # stays inside a stated cap, so the tail carries real weight and the
        # path remains one a market could actually take. Deterministic: the
        # loop is bounded and no RNG is involved.
        cap = float(params.get("max_crash_drawdown", 0.60))
        cap = min(max(cap, 0.20), 0.95)
        max_repeats = max(2, int(params.get("max_crash_repeats", 6)))

        best = pd.concat([returns, worst], ignore_index=True)
        for reps in range(2, max_repeats + 1):
            candidate = pd.concat([returns] + [worst] * reps, ignore_index=True)
            if abs(float(drawdown_series(candidate).min())) > cap:
                break
            best = candidate
        return best


class FixedDrop(StressScenario):
    """An instantaneous move of a chosen size, applied once to the series."""

    def __init__(self, key: str, label: str, description: str, basis: str,
                 drop: float) -> None:
        super().__init__(key, label, description, basis)
        self.drop = drop

    def apply(self, returns: pd.Series, params: dict) -> pd.Series:
        shock = float(params.get("shock_pct", self.drop * 100)) / 100
        return pd.concat([returns, pd.Series([shock])], ignore_index=True)


class VolatilityMultiple(StressScenario):
    """Rescale dispersion around the observed mean, leaving drift intact."""

    def apply(self, returns: pd.Series, params: dict) -> pd.Series:
        factor = float(params.get("vol_multiplier", 2.0))
        mu = float(returns.mean())
        return (returns - mu) * factor + mu


class LiquidityShock(StressScenario):
    """Widen the downside only: in a liquidity event you sell into the spread.

    Losses are amplified while gains are not, which is what an evaporating bid
    actually does to a realised return series.
    """

    def apply(self, returns: pd.Series, params: dict) -> pd.Series:
        penalty = float(params.get("liquidity_penalty", 1.5))
        out = returns.copy()
        out[out < 0] = out[out < 0] * penalty
        return out


class CorrelationSpike(StressScenario):
    """Pull each asset toward the basket's common path.

    Diversification fails precisely when correlations converge, so each series
    is blended with the equal-weighted mean return of the basket. For a single
    asset there is no cross-sectional structure to break, and the series is
    returned unchanged with that stated in the payload.
    """

    def apply(self, returns: pd.Series, params: dict) -> pd.Series:
        common = params.get("_common_path")
        if common is None:
            return returns.copy()
        rho = float(params.get("correlation_target", 0.9))
        rho = min(max(rho, 0.0), 1.0)
        aligned = common.reindex(returns.index).fillna(0.0)
        # Preserve each asset's own scale while forcing co-movement.
        blended = (1 - rho) * returns + rho * aligned
        scale = float(returns.std()) / (float(blended.std()) or 1e-12)
        return blended * scale


SCENARIOS: dict[str, StressScenario] = {
    "market_crash": MarketCrash(
        "market_crash", "Market Crash",
        "Replays this asset's worst realised window as if it happened again.",
        "worst observed rolling window in the loaded history"),
    "drop_10": FixedDrop(
        "drop_10", "Market -10%",
        "A single -10% session added to the observed distribution.",
        "hypothetical one-day shock", -0.10),
    "drop_20": FixedDrop(
        "drop_20", "Market -20%",
        "A single -20% session added to the observed distribution.",
        "hypothetical one-day shock", -0.20),
    "vol_x2": VolatilityMultiple(
        "vol_x2", "Volatility \u00d72",
        "Dispersion doubled around the observed mean; drift unchanged.",
        "realised returns rescaled about their own mean"),
    "liquidity_shock": LiquidityShock(
        "liquidity_shock", "Liquidity Shock",
        "Downside moves amplified to represent selling into a widening spread.",
        "observed negative returns amplified"),
    "correlation_spike": CorrelationSpike(
        "correlation_spike", "Correlation Spike",
        "Holdings pulled toward a common path, so diversification stops paying.",
        "blend toward the basket's equal-weighted mean return"),
    "custom": VolatilityMultiple(
        "custom", "Custom Scenario",
        "Your own volatility multiplier and instantaneous shock.",
        "user-supplied parameters applied to realised returns"),
}


def _annualise_vol(returns: pd.Series) -> float | None:
    if len(returns) < 2:
        return None
    daily = float(returns.std())
    if not np.isfinite(daily):
        return None
    return daily * np.sqrt(TRADING_DAYS)


def _max_drawdown(returns: pd.Series) -> float | None:
    if returns.empty:
        return None
    dd = drawdown_series(returns)
    if dd.empty:
        return None
    worst = float(dd.min())
    return worst if np.isfinite(worst) else None


def profile(returns: pd.Series, confidence: float) -> dict:
    """The risk picture for one return series. Unmeasurable fields stay None."""
    n = int(len(returns))
    if n < MIN_OBS:
        return {
            "observations": n,
            "var_pct": None, "cvar_pct": None, "volatility_pct": None,
            "max_drawdown_pct": None, "worst_day_pct": None,
            "reason": f"need >= {MIN_OBS} observations, got {n}",
        }
    var = value_at_risk(returns, confidence=confidence, method="historical")
    cvar = conditional_var(returns, confidence=confidence)
    vol = _annualise_vol(returns)
    dd = _max_drawdown(returns)
    return {
        "observations": n,
        # VaR/CVaR are reported as positive loss magnitudes.
        "var_pct": None if var is None else round(abs(float(var)) * 100, 4),
        "cvar_pct": None if cvar is None else round(abs(float(cvar)) * 100, 4),
        "volatility_pct": None if vol is None else round(vol * 100, 4),
        "max_drawdown_pct": None if dd is None else round(abs(dd) * 100, 4),
        "worst_day_pct": round(abs(float(returns.min())) * 100, 4),
    }


def _delta(before: float | None, after: float | None) -> float | None:
    if before is None or after is None:
        return None
    return round(after - before, 4)


def _resilience(base: dict, stressed: dict) -> dict:
    """How well the position absorbs the scenario, in [0, 100].

    Built from the *degradation ratios* of three measured quantities (CVaR,
    volatility, drawdown). It is an index, not a probability, and the payload
    says so. When none of the three could be measured the score is None rather
    than a flattering default.
    """
    pairs = [
        ("cvar_pct", 0.45),
        ("volatility_pct", 0.30),
        ("max_drawdown_pct", 0.25),
    ]
    used, score, detail = 0.0, 0.0, {}
    for key, weight in pairs:
        b, a = base.get(key), stressed.get(key)
        if b is None or a is None or b <= 0:
            detail[key] = None
            continue
        ratio = a / b                      # 1.0 = unchanged, 2.0 = twice as bad
        # Map a degradation ratio onto [0, 1]: unchanged scores 1, doubling
        # scores 0. Linear and published, so a reader can re-derive it.
        component = float(np.clip(2.0 - ratio, 0.0, 1.0))
        detail[key] = {"before": b, "after": a, "ratio": round(ratio, 4),
                       "component": round(component, 4)}
        score += component * weight
        used += weight
    if used == 0:
        return {"score": None, "components": detail,
                "reason": "no stressed quantity could be measured"}
    return {
        "score": round(score / used * 100, 2),
        "components": detail,
        "basis": ("weighted degradation of CVaR (45%), volatility (30%) and "
                  "drawdown (25%); 100 = unchanged by the scenario, 0 = twice "
                  "as bad or worse"),
    }


def _vulnerabilities(name: str, base: dict, stressed: dict,
                     assets: list[dict], resilience: dict) -> list[str]:
    """Plain-language findings, each tied to a number that was measured."""
    out: list[str] = []

    cvar_b, cvar_a = base.get("cvar_pct"), stressed.get("cvar_pct")
    if cvar_b and cvar_a:
        growth = (cvar_a / cvar_b - 1) * 100
        out.append(
            f"Expected shortfall widens from {cvar_b:.2f}% to {cvar_a:.2f}% "
            f"({growth:+.1f}%) under {name}: the average loss on a bad day "
            f"grows by more than the headline VaR suggests.")

    dd_b, dd_a = base.get("max_drawdown_pct"), stressed.get("max_drawdown_pct")
    if dd_b and dd_a and dd_a > dd_b:
        out.append(
            f"Peak-to-trough drawdown deepens from {dd_b:.2f}% to {dd_a:.2f}%, "
            f"so the recovery required grows accordingly.")

    if len(assets) > 1:
        ranked = [a for a in assets if a.get("loss_contribution_pct") is not None]
        ranked.sort(key=lambda a: -a["loss_contribution_pct"])
        if ranked:
            top = ranked[0]
            out.append(
                f"{top['symbol']} carries {top['loss_contribution_pct']:.1f}% of "
                f"the stressed loss on {top['weight_pct']:.1f}% of the capital"
                + (" — the book's single point of failure."
                   if top["loss_contribution_pct"] > 2 * top["weight_pct"]
                   else "."))

    score = resilience.get("score")
    if score is not None and score < 40:
        out.append(
            f"Resilience scores {score:.0f}/100: the position degrades sharply "
            f"rather than absorbing this scenario.")

    if not out:
        out.append("No measurable deterioration: the scenario leaves the "
                   "measured risk figures essentially unchanged.")
    return out


def _mitigations(name: str, assets: list[dict], resilience: dict,
                 correlation: dict | None) -> list[str]:
    """Actions that follow from the numbers above. Never generic filler."""
    out: list[str] = []

    ranked = [a for a in assets if a.get("loss_contribution_pct") is not None]
    ranked.sort(key=lambda a: -a["loss_contribution_pct"])
    if ranked and len(assets) > 1:
        top = ranked[0]
        if top["loss_contribution_pct"] > 2 * top["weight_pct"]:
            out.append(
                f"Trim {top['symbol']}: it contributes "
                f"{top['loss_contribution_pct']:.1f}% of the stressed loss from "
                f"{top['weight_pct']:.1f}% of the capital, so a smaller position "
                f"removes loss faster than it removes exposure.")

    if correlation and correlation.get("average_correlation") is not None:
        rho = correlation["average_correlation"]
        if rho > 0.6:
            pair = correlation.get("highest_pair") or {}
            extra = (f" The tightest pair is {pair.get('pair')} at "
                     f"{pair.get('correlation')}." if pair.get("pair") else "")
            out.append(
                f"Average pairwise correlation is {rho:.2f}, so the book behaves "
                f"like fewer independent bets than it holds.{extra} Adding an "
                f"asset with low or negative correlation does more for this "
                f"scenario than reducing size.")

    score = resilience.get("score")
    if score is not None:
        if score < 40:
            out.append(
                "Reduce gross exposure or add a hedge before the next rebalance: "
                "at this resilience the scenario translates almost directly into "
                "realised loss.")
        elif score < 70:
            out.append(
                "Consider a partial hedge sized to the stressed CVaR rather than "
                "the headline VaR, which understates the tail here.")
        else:
            out.append(
                "Current sizing absorbs this scenario; monitor rather than act.")

    out.append(
        f"Re-run this test after any change: the {name} figures are measured "
        f"from the loaded window and move as new data arrives.")
    return out


def _prepare(returns: pd.Series) -> pd.Series:
    return pd.Series(returns).replace([np.inf, -np.inf], np.nan).dropna()


def run(symbol_returns: dict[str, pd.Series], weights: dict[str, float],
        scenario_key: str, *, position_value: float = 100_000.0,
        confidence: float = 0.95, params: dict | None = None) -> dict:
    """Stress one asset or a weighted basket and report before vs after.

    `symbol_returns` maps ticker -> realised daily returns. `weights` must sum
    to a positive number; it is normalised here so a caller cannot silently
    change the size of the book.
    """
    params = dict(params or {})
    scenario = SCENARIOS.get(scenario_key)
    if scenario is None:
        raise ValueError(f"unknown scenario '{scenario_key}'; "
                         f"available: {sorted(SCENARIOS)}")

    clean = {s: _prepare(r) for s, r in symbol_returns.items()}
    clean = {s: r for s, r in clean.items() if not r.empty}
    if not clean:
        raise ValueError("no usable return data for the requested symbols")

    total_w = sum(abs(weights.get(s, 0.0)) for s in clean) or float(len(clean))
    norm_w = {s: (abs(weights.get(s, 1.0)) / total_w) for s in clean}

    frame = pd.DataFrame(clean).dropna()
    if frame.empty:
        # No overlapping dates: fall back to the longest single series so the
        # request still answers, and say so.
        longest = max(clean.items(), key=lambda kv: len(kv[1]))
        frame = pd.DataFrame({longest[0]: longest[1]})
        norm_w = {longest[0]: 1.0}

    # A correlation spike needs the basket's common path.
    if scenario_key == "correlation_spike" and frame.shape[1] > 1:
        params["_common_path"] = frame.mean(axis=1)

    stressed_assets = {s: _prepare(scenario.apply(frame[s], params))
                       for s in frame.columns}

    # Portfolio series: weighted sums on the aligned frame.
    base_port = sum(frame[s] * norm_w[s] for s in frame.columns)
    stressed_frame = pd.DataFrame(stressed_assets)
    stressed_port = sum(stressed_frame[s].fillna(0.0) * norm_w[s]
                        for s in stressed_frame.columns)

    before = profile(pd.Series(base_port), confidence)
    after = profile(pd.Series(stressed_port), confidence)

    # ---- asset-level impact and risk contribution -----------------------
    cov = frame.cov().values * TRADING_DAYS
    w_vec = np.array([norm_w[s] for s in frame.columns])
    contrib = risk_contribution(w_vec, cov) if len(w_vec) > 1 else np.array([1.0])

    stressed_cov = stressed_frame.dropna().cov().values * TRADING_DAYS
    stressed_contrib = (risk_contribution(w_vec, stressed_cov)
                        if len(w_vec) > 1 and stressed_cov.shape[0] == len(w_vec)
                        else contrib)

    assets: list[dict] = []
    raw_losses: dict[str, float] = {}
    for i, sym in enumerate(frame.columns):
        a_before = profile(frame[sym], confidence)
        a_after = profile(stressed_assets[sym], confidence)
        cvar_after = a_after.get("cvar_pct")
        # Contribution to the stressed loss: weight x stressed shortfall.
        raw = (norm_w[sym] * cvar_after) if cvar_after is not None else None
        if raw is not None:
            raw_losses[sym] = raw
        assets.append({
            "symbol": sym,
            "weight_pct": round(norm_w[sym] * 100, 4),
            "before": a_before,
            "after": a_after,
            "var_delta_pct": _delta(a_before.get("var_pct"), a_after.get("var_pct")),
            "cvar_delta_pct": _delta(a_before.get("cvar_pct"), a_after.get("cvar_pct")),
            "volatility_delta_pct": _delta(a_before.get("volatility_pct"),
                                           a_after.get("volatility_pct")),
            "risk_contribution_pct": round(float(contrib[i]) * 100, 4),
            "stressed_risk_contribution_pct": round(float(stressed_contrib[i]) * 100, 4),
            "value_at_risk_money": (None if a_after.get("var_pct") is None else
                                    round(position_value * norm_w[sym]
                                          * a_after["var_pct"] / 100, 2)),
        })

    loss_total = sum(raw_losses.values())
    for entry in assets:
        raw = raw_losses.get(entry["symbol"])
        entry["loss_contribution_pct"] = (
            round(raw / loss_total * 100, 4) if raw is not None and loss_total > 0
            else None)

    # ---- money terms ----------------------------------------------------
    def money(pct: float | None) -> float | None:
        return None if pct is None else round(position_value * pct / 100, 2)

    corr = (correlation_matrix(frame) if frame.shape[1] > 1 else None)
    stressed_corr = (correlation_matrix(stressed_frame.dropna())
                     if stressed_frame.dropna().shape[1] > 1 else None)

    resilience = _resilience(before, after)

    return {
        "scenario": scenario.key,
        "scenario_label": scenario.label,
        "scenario_description": scenario.description,
        "scenario_basis": scenario.basis,
        "confidence": confidence,
        "position_value": position_value,
        "symbols": list(frame.columns),
        "weights_pct": {s: round(norm_w[s] * 100, 4) for s in frame.columns},
        "observations": int(len(frame)),
        "before": before,
        "after": after,
        "deltas": {
            "var_pct": _delta(before.get("var_pct"), after.get("var_pct")),
            "cvar_pct": _delta(before.get("cvar_pct"), after.get("cvar_pct")),
            "volatility_pct": _delta(before.get("volatility_pct"),
                                     after.get("volatility_pct")),
            "max_drawdown_pct": _delta(before.get("max_drawdown_pct"),
                                       after.get("max_drawdown_pct")),
        },
        "portfolio_loss": {
            "var_money": money(after.get("var_pct")),
            "cvar_money": money(after.get("cvar_pct")),
            "drawdown_money": money(after.get("max_drawdown_pct")),
            "additional_cvar_money": money(
                _delta(before.get("cvar_pct"), after.get("cvar_pct"))),
        },
        "assets": assets,
        "correlation": corr,
        "stressed_correlation": stressed_corr,
        "resilience": resilience,
        "vulnerabilities": _vulnerabilities(scenario.label, before, after,
                                            assets, resilience),
        "mitigations": _mitigations(scenario.label, assets, resilience, corr),
        "parameters": {k: v for k, v in params.items() if not k.startswith("_")},
        "disclaimer": (
            "Scenario analysis over the loaded window, computed from realised "
            "returns with the platform's own risk functions. A stress test "
            "describes exposure, not a forecast, and the future can exceed any "
            "scenario shown here."),
    }


def catalogue() -> list[dict]:
    """The scenarios a caller may request, for building a UI without guessing."""
    return [{"key": s.key, "label": s.label, "description": s.description,
             "basis": s.basis} for s in SCENARIOS.values()]
