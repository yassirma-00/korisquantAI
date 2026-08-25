"""AI Confidence Score: how much to trust the Buy / Hold / Sell call.

The recommendation engine already produced a confidence number, but it was
derived from four signal families only — the Risk Engine never entered it. A
score presented as "the AI's confidence" while ignoring the module that detects
crashes and bubbles overstates certainty in exactly the conditions where
certainty is least warranted.

This computes the score from five contributors, each bounded to [0, 1] and
weighted:

* **Model agreement** — do the signals point the same way? Disagreement is the
  single strongest reason to distrust a call.
* **Signal strength** — a composite near zero is a genuine "no opinion"; a
  strong reading is more actionable than a marginal one.
* **Model reliability** — the per-model quality the engine already measures
  (directional accuracy, backtest reward, indicator confirmation).
* **Coverage** — how many of the five families actually produced an opinion.
  An untrained RL agent is missing evidence, not neutral evidence.
* **Risk clarity** — the Risk Engine's own certainty. Elevated crash risk or a
  bubble reading *reduces* confidence in a bullish call rather than being
  ignored.

Every contributor is returned with its value, weight and a plain sentence, so
the score can be audited instead of trusted.
"""

from __future__ import annotations

import numpy as np

from app.core.logging import get_logger

logger = get_logger(__name__)

# Bands the UI renders. Chosen so that "Very High" is genuinely rare: with five
# noisy contributors, a score above 80% should mean the evidence really is
# unanimous, strong and complete.
BANDS = (
    (0.80, "Very High", "very-high"),
    (0.65, "High", "high"),
    (0.45, "Moderate", "moderate"),
    (0.25, "Low", "low"),
    (0.00, "Very Low", "very-low"),
)

WEIGHTS = {
    "agreement": 0.30,
    "strength": 0.20,
    "reliability": 0.20,
    "coverage": 0.15,
    "risk_clarity": 0.15,
}

SOURCE_LABELS = {
    "forecast": "Deep learning",
    "rl": "Reinforcement learning",
    "technical": "Technical analysis",
    "sentiment": "NLP sentiment",
    "risk": "Risk engine",
}


def _band(score: float) -> tuple[str, str]:
    for threshold, label, key in BANDS:
        if score >= threshold:
            return label, key
    return "Very Low", "very-low"


def _safe(value, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if np.isfinite(out) else default


def _agreement(signals: list[dict]) -> tuple[float, str]:
    """Do the available models point the same way?

    Measured over signals with an actual opinion: a model sitting at 0.0 is
    abstaining, and counting abstentions as agreement would inflate the score
    precisely when nothing is being said.
    """
    opinions = [_safe(s.get("score")) for s in signals
                if s.get("available") and abs(_safe(s.get("score"))) > 0.05]
    if len(opinions) < 2:
        # One opinion cannot agree or disagree with anything. 0.5 is the honest
        # midpoint, not a pass.
        return 0.5, ("Only one model expressed a view, so there is no "
                     "cross-model agreement to measure.")
    positive = sum(1 for o in opinions if o > 0)
    share = max(positive, len(opinions) - positive) / len(opinions)
    # share runs 0.5 (evenly split) to 1.0 (unanimous); rescale to 0..1.
    value = (share - 0.5) * 2

    # Two models agreeing is not the same evidence as four agreeing, but the
    # ratio above cannot tell them apart — both give 1.0. Scale unanimity by
    # how many voices produced it, or a lone surviving pair reads as consensus
    # and the score climbs precisely when most models failed to run.
    breadth = min(len(opinions) / 4.0, 1.0)
    value *= 0.55 + 0.45 * breadth

    if share >= 0.99:
        detail = (f"All {len(opinions)} models with a view agree"
                  + ("." if len(opinions) >= 4
                     else f", but only {len(opinions)} of 4 expressed one."))
    elif value >= 0.3:
        detail = f"{positive} of {len(opinions)} models are bullish — a majority, not a consensus."
    else:
        detail = f"Models are split {positive}/{len(opinions) - positive}; they contradict each other."
    return value, detail


def _strength(composite: float) -> tuple[float, str]:
    """A composite near zero is a real 'no opinion', not a confident hold."""
    magnitude = min(abs(composite) / 0.6, 1.0)
    if magnitude < 0.15:
        detail = (f"The combined score is {composite:+.2f} — close to neutral, "
                  "which is a weak basis for any action.")
    else:
        detail = f"The combined score is {composite:+.2f}, a clear directional reading."
    return magnitude, detail


def _reliability(signals: list[dict]) -> tuple[float, str]:
    available = [s for s in signals if s.get("available")]
    if not available:
        return 0.0, "No model produced a usable signal."
    values = [_safe(s.get("reliability")) for s in available]
    mean = sum(values) / len(values)
    weakest = min(available, key=lambda s: _safe(s.get("reliability")))
    return mean, (f"Average model quality {mean:.0%}; weakest is "
                  f"{SOURCE_LABELS.get(weakest['source'], weakest['source']).lower()} "
                  f"at {_safe(weakest.get('reliability')):.0%}.")


def _coverage(signals: list[dict], risk_available: bool) -> tuple[float, str]:
    """Missing evidence is not neutral evidence."""
    total = len(signals) + 1                      # the five families
    present = sum(1 for s in signals if s.get("available")) + int(risk_available)
    missing = [SOURCE_LABELS.get(s["source"], s["source"])
               for s in signals if not s.get("available")]
    if not risk_available:
        missing.append(SOURCE_LABELS["risk"])
    value = present / total if total else 0.0
    detail = (f"All {total} model families contributed."
              if not missing else
              f"{present} of {total} families contributed; missing: "
              f"{', '.join(m.lower() for m in missing)}.")
    return value, detail


def _risk_clarity(risk: dict, composite: float) -> tuple[float, str, bool]:
    """The Risk Engine's contribution.

    Two different things reduce confidence here, and they are not the same:

    * the risk scores could not be computed at all (unknown conditions), and
    * the scores are computable and say conditions are dangerous.

    Elevated crash risk under a bullish call is the case this exists to catch:
    the models may agree, but agreeing into a fragile tape deserves less
    confidence, not the same amount.
    """
    crash = (risk or {}).get("crash_risk") or {}
    bubble = (risk or {}).get("bubble") or {}
    crash_score = crash.get("crash_risk_score")
    bubble_score = bubble.get("bubble_score")

    if crash_score is None and bubble_score is None:
        return 0.35, ("Risk scores could not be computed for this window, so "
                      "conditions are unverified."), False

    crash_value = _safe(crash_score)
    bubble_value = _safe(bubble_score)
    # Calm, measurable conditions score high; danger scores low.
    danger = max(crash_value, bubble_value * 0.8)
    value = float(np.clip(1.0 - danger / 0.75, 0.0, 1.0))

    if danger < 0.3:
        detail = f"Risk conditions are calm (crash {crash_value:.0%}, bubble {bubble_value:.0%})."
    elif composite > 0.1:
        detail = (f"Crash risk {crash_value:.0%} and bubble {bubble_value:.0%} argue "
                  "against confidence in a bullish call.")
    else:
        detail = f"Elevated risk (crash {crash_value:.0%}, bubble {bubble_value:.0%})."
    return value, detail, True


def confidence_report(recommendation: dict) -> dict:
    """Build the AI Confidence Score from a recommendation payload."""
    signals = list(recommendation.get("signals") or [])
    risk = recommendation.get("risk") or {}
    composite = _safe(recommendation.get("composite_score"))

    agreement_v, agreement_d = _agreement(signals)
    strength_v, strength_d = _strength(composite)
    reliability_v, reliability_d = _reliability(signals)
    risk_v, risk_d, risk_ok = _risk_clarity(risk, composite)
    coverage_v, coverage_d = _coverage(signals, risk_ok)

    parts = [
        ("agreement", "Model agreement", agreement_v, agreement_d),
        ("strength", "Signal strength", strength_v, strength_d),
        ("reliability", "Model reliability", reliability_v, reliability_d),
        ("coverage", "Model coverage", coverage_v, coverage_d),
        ("risk_clarity", "Risk clarity", risk_v, risk_d),
    ]
    score = sum(WEIGHTS[key] * value for key, _label, value, _d in parts)
    score = float(np.clip(score, 0.0, 1.0))
    label, band_key = _band(score)

    contributors = [
        {
            "key": key,
            "label": label_text,
            "value": round(value, 4),
            "weight": WEIGHTS[key],
            # Points contributed out of 100, which is what the UI shows.
            "points": round(WEIGHTS[key] * value * 100, 1),
            "max_points": round(WEIGHTS[key] * 100, 1),
            "detail": detail,
        }
        for key, label_text, value, detail in parts
    ]

    # The one or two things that most held the score back — more useful than
    # restating what went right.
    shortfalls = sorted(contributors, key=lambda c: c["max_points"] - c["points"],
                        reverse=True)
    limiting = [c for c in shortfalls if c["max_points"] - c["points"] > 2][:2]

    action = recommendation.get("action", "HOLD")
    if score >= 0.65:
        headline = (f"The models broadly support this {action.replace('_', ' ')} call.")
    elif score >= 0.45:
        headline = (f"Moderate support for {action.replace('_', ' ')} — usable, "
                    "but not a high-conviction setup.")
    else:
        headline = (f"Weak support for {action.replace('_', ' ')}. Treat this as "
                    "a low-conviction reading.")

    return {
        "score": round(score, 4),
        "percent": round(score * 100, 1),
        "label": label,
        "band": band_key,
        "action": action,
        "headline": headline,
        "contributors": contributors,
        "limiting_factors": [c["label"] for c in limiting],
        "summary": (headline + " "
                    + (f"Held back mainly by {' and '.join(c['label'].lower() for c in limiting)}."
                       if limiting else "No single factor is holding it back.")),
        "bands": [{"from": round(t * 100), "label": lbl, "key": k} for t, lbl, k in reversed(BANDS)],
        "basis": ("A weighted read of model agreement, signal strength, measured "
                  "model quality, how many of the five model families contributed, "
                  "and how clear the risk picture is. It is not a probability that "
                  "the call will be profitable — nothing here is backtested against "
                  "realised outcomes."),
    }
