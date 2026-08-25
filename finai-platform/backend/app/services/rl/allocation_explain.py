"""Per-asset attribution of the regime's influence on an allocation.

The single-asset explainer answers "did the regime change BUY into HOLD?". That
question does not survive the move to a portfolio: a continuous agent emits a
*weight vector*, so the regime does not flip a label — it shifts capital between
sleeves. Reporting only "the allocation changed" would hide the part a risk
manager actually needs, which is **which asset lost weight, and to whom**.

So this measures three things by counterfactual, re-querying the agent with
regime inputs neutralised:

* **Portfolio level** — how much total capital moved, and whether the shift was
  toward cash (de-risking) or between assets (rotation). Those are different
  behaviours and conflating them would make a defensive agent and an
  opportunistic one look identical.
* **Per asset** — the weight delta for each sleeve, tied to the regime that
  sleeve is actually in.
* **Per feature** — knocking out one regime input at a time to see how far the
  allocation moves in total.

Turnover, not direction, is the honest headline metric here. A weight vector has
no natural "action changed" boundary, so influence is graded by how much of the
book the regime moved: below half a percent of capital is noise on a softmax and
is reported as negligible rather than dressed up as insight.
"""

from __future__ import annotations

import numpy as np

from app.core.logging import get_logger
from app.services.rl.portfolio_regime import (
    PER_ASSET_FEATURES,
    PORTFOLIO_FEATURE_NAMES,
    feature_dim,
)
from app.services.rl.regime_features import REGIME_RISK, _neutral_row

logger = get_logger(__name__)

PER_ASSET_FEATURE_NAMES = ("risk", "directional_bias", "crash_probability")

# Fraction of the book that has to move before the regime is credited with
# influence. A softmax over near-identical logits jitters by a few basis points
# regardless of any input, so a threshold below this would report noise.
NEGLIGIBLE_TURNOVER = 0.005
DECISIVE_TURNOVER = 0.05


def _weights_from(agent, obs: np.ndarray) -> np.ndarray | None:
    """The agent's target weight vector for one observation."""
    try:
        raw = np.asarray(agent.act(obs, deterministic=True), dtype=float).ravel()
    except Exception as exc:      # pragma: no cover - defensive
        logger.debug("allocation query failed: %s", exc)
        return None
    exp = np.exp(raw - raw.max())
    return exp / exp.sum()


def _neutral_regime_block(provider, n_assets: int) -> np.ndarray:
    """The regime block as it would read with every asset unclassified."""
    neutral = _neutral_row()
    per_asset: list[float] = []
    for _ in range(n_assets):
        per_asset.extend((neutral.risk, neutral.bull, neutral.crash_prob))
    aggregate = [
        neutral.risk,      # mean risk
        0.0,               # worst crash probability
        0.0,               # dispersion: nothing to disagree about
        0.0,               # mean confidence
    ]
    return np.asarray(per_asset + aggregate, dtype=np.float32)


def explain_allocation_influence(agent, env, obs: np.ndarray,
                                 weights: np.ndarray) -> dict:
    """How the detected regimes shaped this weight vector.

    ``weights`` is the realised allocation including cash in the last slot.
    """
    provider = getattr(env, "regime", None)
    if provider is None or not getattr(env.cfg, "regime_aware", False):
        return {
            "available": False,
            "reason": ("This agent was trained without regime awareness, so the "
                       "market regime did not enter its allocation. Retrain with "
                       "regime_aware enabled to attribute it."),
        }

    symbols = list(env.symbols)
    n_assets = len(symbols)
    block = feature_dim(n_assets)
    rows = provider.rows(env.t)

    neutral_obs = np.array(obs, dtype=np.float32, copy=True)
    neutral_obs[-block:] = _neutral_regime_block(provider, n_assets)
    counterfactual = _weights_from(agent, neutral_obs)

    if counterfactual is None:
        return {"available": False,
                "reason": "The agent could not be re-queried for a counterfactual."}

    actual = np.asarray(weights, dtype=float).ravel()
    delta = actual - counterfactual
    # Half the L1 norm is the fraction of capital that actually changed hands;
    # the raw sum double-counts every move (one sleeve down, another up).
    turnover = float(np.abs(delta).sum() / 2.0)

    cash_delta = float(delta[-1]) if delta.size == n_assets + 1 else 0.0
    if turnover < NEGLIGIBLE_TURNOVER:
        influence = "negligible"
    elif turnover >= DECISIVE_TURNOVER:
        influence = "decisive"
    else:
        influence = "contributory"

    per_asset = []
    for i, symbol in enumerate(symbols):
        row = rows[i] if i < len(rows) else _neutral_row()
        per_asset.append({
            "symbol": symbol,
            "weight": round(float(actual[i]), 4),
            "weight_without_regime": round(float(counterfactual[i]), 4),
            "delta": round(float(delta[i]), 4),
            "regime": row.regime,
            "regime_label": row.label,
            "regime_confidence": round(row.confidence, 4),
            "risk": round(row.risk, 4),
            "crash_probability": round(row.crash_prob, 4),
            "direction": ("increased" if delta[i] > 1e-4
                          else "reduced" if delta[i] < -1e-4 else "unchanged"),
        })
    per_asset.sort(key=lambda a: abs(a["delta"]), reverse=True)

    # Which regime input moved the book, one knockout at a time.
    feature_contributions = []
    neutral_block = _neutral_regime_block(provider, n_assets)
    for i in range(block):
        probe = np.array(obs, dtype=np.float32, copy=True)
        probe[-block + i] = neutral_block[i]
        probe_weights = _weights_from(agent, probe)
        if probe_weights is None:
            continue
        moved = float(np.abs(actual - probe_weights).sum() / 2.0)
        if i < PER_ASSET_FEATURES * n_assets:
            asset_idx, feat_idx = divmod(i, PER_ASSET_FEATURES)
            name = f"{symbols[asset_idx]} {PER_ASSET_FEATURE_NAMES[feat_idx]}"
        else:
            name = PORTFOLIO_FEATURE_NAMES[i - PER_ASSET_FEATURES * n_assets]
        feature_contributions.append({
            "feature": name.replace("_", " "),
            "capital_moved": round(moved, 5),
        })
    feature_contributions.sort(key=lambda c: c["capital_moved"], reverse=True)

    riskiest = max(rows, key=lambda r: r.risk) if rows else _neutral_row()
    regimes = sorted({r.regime for r in rows})

    if influence == "negligible":
        summary = ("The detected regimes left this allocation essentially "
                   "unchanged: less than 0.5% of the book differs from what the "
                   "agent would hold with no regime information.")
    else:
        moved = [a for a in per_asset if a["direction"] != "unchanged"][:2]
        moves = ", ".join(
            f"{a['symbol']} {a['direction']} by {abs(a['delta']) * 100:.1f}pp "
            f"({a['regime_label']})" for a in moved)
        toward = ("into cash" if cash_delta > 1e-4
                  else "out of cash" if cash_delta < -1e-4
                  else "between assets")
        summary = (f"Regime awareness moved {turnover * 100:.1f}% of the book "
                   f"{toward}: {moves}.")

    return {
        "available": True,
        "influence": influence,
        "capital_moved": round(turnover, 5),
        "cash_delta": round(cash_delta, 5),
        # Rotation and de-risking are different behaviours; a single "changed"
        # flag would make a defensive agent look like an opportunistic one.
        "shift_type": ("de-risking" if cash_delta > 1e-4
                       else "re-risking" if cash_delta < -1e-4 else "rotation"),
        "per_asset": per_asset,
        "feature_contributions": feature_contributions[:8],
        "regimes_in_force": regimes,
        "all_assets_same_regime": len(regimes) == 1,
        "riskiest_asset": {
            "symbol": symbols[rows.index(riskiest)] if riskiest in rows else None,
            "regime": riskiest.regime,
            "risk": round(riskiest.risk, 4),
        } if rows else None,
        "mean_risk": round(float(np.mean([r.risk for r in rows])), 4) if rows else None,
        "summary": summary,
        "narrative": (
            f"{len(regimes)} regime{'s' if len(regimes) != 1 else ''} in force "
            f"across {n_assets} asset{'s' if n_assets != 1 else ''} "
            f"({', '.join(regimes)}). "
            + ("Every sleeve shares one regime, so diversification offers little "
               "protection here. " if len(regimes) == 1 else "")
            + summary),
        "scale_note": (
            f"Influence is graded by how much capital moved: below "
            f"{NEGLIGIBLE_TURNOVER:.1%} is treated as softmax noise, "
            f"{DECISIVE_TURNOVER:.0%} or more as decisive."),
    }


def regime_risk_reference() -> dict:
    """The ordinal risk scale, so a caller can read the numbers above."""
    return {k: round(v, 3) for k, v in sorted(REGIME_RISK.items(), key=lambda kv: kv[1])}
