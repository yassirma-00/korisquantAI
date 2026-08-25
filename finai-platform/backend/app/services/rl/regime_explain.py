"""Why the regime moved (or did not move) an RL agent's decision.

The RL page could already show *what* the agent chose and how confident it was.
What it could not show was *why the market environment mattered* — and once the
regime enters both the observation and the reward, that becomes the first
question a risk manager asks.

This module answers it with measurement rather than narration. The agent is
re-queried with the regime block replaced by a neutral reading, and the two
answers are compared. If the action changes, the regime is decisive; if only
the confidence moves, it is contributory; if nothing moves, the explanation
says so plainly instead of inventing an influence that is not there.

That counterfactual is the honest version of "the regime influenced this
decision". A template sentence asserting influence would read identically
whether or not any influence existed.
"""

from __future__ import annotations

import numpy as np

from app.core.logging import get_logger
from app.services.rl.regime_features import (
    REGIME_FEATURE_DIM,
    REGIME_FEATURE_NAMES,
    RegimeRow,
    _neutral_row,
)

logger = get_logger(__name__)

# What each regime implies for positioning, reused from the detector's own
# action table so the two panels cannot drift apart.
REGIME_STANCE = {
    "bull_market": "supports taking or holding exposure",
    "recovery": "supports rebuilding exposure gradually",
    "low_volatility": "supports carry and patience",
    "sideways": "argues for range discipline over trend-following",
    "high_volatility": "argues for smaller positions at the same conviction",
    "bear_market": "argues for reduced exposure and quality",
    "crash_risk": "argues for hedging and cutting gross exposure",
    "unknown": "gives no directional guidance",
}


def _q_vector(agent, obs: np.ndarray) -> np.ndarray | None:
    """Action values, when the agent exposes them."""
    if hasattr(agent, "q_values"):
        try:
            return np.asarray(agent.q_values(obs), dtype=float)
        except Exception as exc:      # pragma: no cover - defensive
            logger.debug("q_values failed: %s", exc)
    return None


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - values.max()
    exp = np.exp(shifted)
    return exp / exp.sum()


def explain_regime_influence(agent, env, obs: np.ndarray, action: int,
                             action_names: dict[int, str]) -> dict:
    """Counterfactual attribution of the regime block on one decision.

    Returns a payload that is safe to store in an audit log and to render.
    ``available`` is False when the agent is not regime-aware, rather than a
    fabricated explanation.
    """
    row: RegimeRow = (env.regime.at(env.t)
                      if getattr(env, "regime", None) is not None
                      else _neutral_row())

    if getattr(env, "regime", None) is None or not getattr(env.cfg, "regime_aware", False):
        return {
            "available": False,
            "regime": None,
            "reason": ("This agent was trained without regime awareness, so the "
                       "market regime did not enter its decision. Retrain with "
                       "regime_aware enabled to attribute it."),
        }

    chosen = action_names.get(action, str(action))
    base_q = _q_vector(agent, obs)

    # Counterfactual: same bar, neutral regime block.
    neutral_obs = np.array(obs, dtype=np.float32, copy=True)
    neutral_obs[-REGIME_FEATURE_DIM:] = _neutral_row().to_vector()

    try:
        if base_q is not None:
            alt_q = _q_vector(agent, neutral_obs)
            alt_action = int(np.argmax(alt_q)) if alt_q is not None else action
        else:
            alt_q = None
            alt_action = int(agent.act(neutral_obs, deterministic=True))
    except Exception as exc:      # pragma: no cover - defensive
        logger.debug("counterfactual failed: %s", exc)
        alt_q, alt_action = None, action

    changed = alt_action != action
    delta_conf = None
    if base_q is not None and alt_q is not None:
        delta_conf = float(_softmax(base_q)[action] - _softmax(alt_q)[action])

    if changed:
        influence = "decisive"
        summary = (
            f"The {row.label} regime changed this call: with the regime signal "
            f"removed the agent would have chosen "
            f"{action_names.get(alt_action, alt_action)} instead of {chosen}.")
    elif delta_conf is not None and abs(delta_conf) >= 0.02:
        influence = "contributory"
        direction = "raised" if delta_conf > 0 else "lowered"
        summary = (
            f"The {row.label} regime did not change the action, but it "
            f"{direction} confidence in {chosen} by "
            f"{abs(delta_conf) * 100:.1f} percentage points.")
    else:
        influence = "negligible"
        summary = (
            f"The {row.label} regime left this decision unchanged: the agent "
            f"would have chosen {chosen} regardless.")

    # Per-feature contribution: knock out one regime input at a time and see
    # how far the chosen action's value moves. Only meaningful for agents that
    # expose action values.
    contributions = []
    if base_q is not None:
        neutral_vec = _neutral_row().to_vector()
        actual_vec = row.to_vector()
        for i, name in enumerate(REGIME_FEATURE_NAMES):
            probe = np.array(obs, dtype=np.float32, copy=True)
            probe[-REGIME_FEATURE_DIM + i] = neutral_vec[i]
            probe_q = _q_vector(agent, probe)
            if probe_q is None:
                continue
            contributions.append({
                "feature": name,
                "value": round(float(actual_vec[i]), 4),
                "neutral": round(float(neutral_vec[i]), 4),
                "q_delta": round(float(base_q[action] - probe_q[action]), 5),
            })
        contributions.sort(key=lambda c: abs(c["q_delta"]), reverse=True)

    return {
        "available": True,
        "regime": row.regime,
        "regime_label": row.label,
        "regime_confidence": round(row.confidence, 4),
        "regime_probability": round(row.probability, 4),
        "risk_level": round(row.risk, 4),
        "directional_bias": round(row.bull, 4),
        "volatility_ratio": round(row.vol_ratio, 4),
        "crash_probability": round(row.crash_prob, 4),
        "drawdown": round(row.drawdown, 4),
        "risk_aversion_applied": round(row.risk_aversion, 4),
        "influence": influence,
        "counterfactual_action": action_names.get(alt_action, str(alt_action)),
        "action_changed": bool(changed),
        "confidence_delta": None if delta_conf is None else round(delta_conf, 4),
        "feature_contributions": contributions,
        "stance": REGIME_STANCE.get(row.regime, REGIME_STANCE["unknown"]),
        "summary": summary,
        "narrative": (
            f"{row.label} detected at {row.confidence * 100:.0f}% confidence "
            f"({row.probability * 100:.0f}% probability). This regime "
            f"{REGIME_STANCE.get(row.regime, REGIME_STANCE['unknown'])}. "
            f"Risk penalties in the agent's objective were scaled by "
            f"{row.risk_aversion:.2f}x. {summary}"),
    }
