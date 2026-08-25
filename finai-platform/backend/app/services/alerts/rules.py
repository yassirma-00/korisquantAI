"""Evaluation of multi-condition alert rules, plus the one-click templates.

A rule is a list of conditions joined by AND or OR. Each condition names a
metric, an operator and a value; categorical conditions (market regime, AI
action) compare a label instead.

Two properties this module is careful about:

* **An unresolvable metric never counts as a pass.** If crash risk cannot be
  computed on the chosen window, the condition is false and the reason is
  recorded. Silently treating "unknown" as "true" would fire alerts on missing
  data, which is the worst possible failure for a risk tool.
* **Every trigger explains itself.** The stored alert carries the value each
  condition saw and the comparison it made, so a user reading the history can
  tell why it fired without re-deriving anything.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from app.core.logging import get_logger
from app.services.alerts.metrics import (
    ACTION_CHOICES,
    CATEGORICAL_METRICS,
    METRICS_BY_KEY,
    OPERATORS,
    REGIME_CHOICES,
    MetricResolver,
)

logger = get_logger(__name__)

PRIORITIES = ("low", "medium", "high", "critical")

# Priority decides how loudly a trigger is surfaced, so it maps onto the
# severity the alert list already understands.
PRIORITY_SEVERITY = {
    "low": "info", "medium": "info", "high": "warning", "critical": "critical",
}

# Legacy single-condition rules map onto the same metric registry, so old and
# new rules run through one evaluator rather than two divergent code paths.
LEGACY_RULE_MAP = {
    "price_above": ("price", "above"),
    "price_below": ("price", "below"),
    "pct_move": ("pct_change", "above"),
    "rsi": ("rsi", "above"),
}


# --------------------------------------------------------------- templates
# Each template is a complete, sensible rule. The thresholds are the ones the
# rest of the platform already uses for its own bands, so a "High Risk" alert
# fires at the same point the Risk panel turns red.
TEMPLATES: tuple[dict[str, Any], ...] = (
    {
        "key": "high_risk",
        "name": "High Risk Alert",
        "description": "Crash Risk Score enters the 'high' band while price is falling.",
        "priority": "critical",
        "logic": "AND",
        "period": "2y",
        "conditions": [
            {"metric": "crash_risk", "operator": "above", "value": 55},
            {"metric": "drawdown", "operator": "below", "value": -10},
        ],
    },
    {
        "key": "bull_entry",
        "name": "Bull Market Entry",
        "description": "Regime turns bullish with the 50-day above the 200-day.",
        "priority": "medium",
        "logic": "AND",
        "period": "2y",
        "conditions": [
            {"metric": "regime", "operator": "is", "value": "bull_market"},
            {"metric": "ma_cross", "operator": "above", "value": 0},
        ],
    },
    {
        "key": "bear_warning",
        "name": "Bear Market Warning",
        "description": "Regime turns bearish or the 50-day falls below the 200-day.",
        "priority": "high",
        "logic": "OR",
        "period": "2y",
        "conditions": [
            {"metric": "regime", "operator": "is", "value": "bear_market"},
            {"metric": "ma_cross", "operator": "below", "value": -2},
        ],
    },
    {
        "key": "high_volatility",
        "name": "High Volatility Alert",
        "description": "Realised volatility above 40% annualised.",
        "priority": "high",
        "logic": "OR",
        "period": "1y",
        "conditions": [
            {"metric": "volatility", "operator": "above", "value": 40},
            {"metric": "regime", "operator": "is", "value": "high_volatility"},
        ],
    },
    {
        "key": "drawdown",
        "name": "Portfolio Drawdown Alert",
        "description": "Price more than 15% below its peak for the period.",
        "priority": "high",
        "logic": "AND",
        "period": "2y",
        "conditions": [{"metric": "drawdown", "operator": "below", "value": -15}],
    },
    {
        "key": "ai_buy",
        "name": "AI Buy Signal",
        "description": "Ensemble recommendation turns constructive.",
        "priority": "medium",
        "logic": "AND",
        "period": "2y",
        "conditions": [
            {"metric": "ai_action", "operator": "is", "value": "BUY"},
            {"metric": "ai_score", "operator": "above", "value": 0.25},
        ],
    },
    {
        "key": "ai_sell",
        "name": "AI Sell Signal",
        "description": "Ensemble recommendation turns negative.",
        "priority": "high",
        "logic": "OR",
        "period": "2y",
        "conditions": [
            {"metric": "ai_action", "operator": "is", "value": "SELL"},
            {"metric": "ai_score", "operator": "below", "value": -0.25},
        ],
    },
    {
        "key": "bubble_risk",
        "name": "Bubble Risk Alert",
        "description": "Bubble Indicator enters the elevated band.",
        "priority": "high",
        "logic": "AND",
        "period": "5y",
        "conditions": [{"metric": "bubble_score", "operator": "above", "value": 60}],
    },
    {
        "key": "regime_change",
        "name": "Market Regime Change Alert",
        "description": "A regime is detected with high confidence — check the timeline.",
        "priority": "medium",
        "logic": "AND",
        "period": "2y",
        "conditions": [{"metric": "regime_probability", "operator": "above", "value": 70}],
    },
)

TEMPLATES_BY_KEY = {t["key"]: t for t in TEMPLATES}


def describe_condition(condition: dict) -> str:
    """Human phrasing for one condition, used in the UI and the alert message."""
    metric = condition.get("metric", "?")
    operator = condition.get("operator", "above")
    value = condition.get("value")
    if metric in CATEGORICAL_METRICS:
        return f"{metric.replace('_', ' ')} is {value}"
    spec = METRICS_BY_KEY.get(metric)
    label = spec.label if spec else metric
    unit = spec.unit if spec else ""
    word = {"above": "above", "below": "below", "equals": "equals",
            "crosses_above": "crosses above", "crosses_below": "crosses below"}.get(
        operator, operator)
    return f"{label} {word} {value}{unit}"


def normalise_conditions(rule) -> list[dict]:
    """The conditions to evaluate, whether the rule is new-style or legacy."""
    conditions = list(getattr(rule, "conditions", None) or [])
    if conditions:
        return conditions
    metric, operator = LEGACY_RULE_MAP.get(rule.rule_type, (rule.rule_type, "above"))
    # The legacy RSI rule flipped direction around 50; preserve that exactly,
    # or every pre-existing oversold alert would silently invert.
    if rule.rule_type == "rsi" and rule.threshold < 50:
        operator = "below"
    return [{"metric": metric, "operator": operator, "value": rule.threshold}]


def evaluate_rule(rule) -> dict:
    """Evaluate one rule against live data.

    Returns the verdict plus the evidence: every condition, the value observed,
    and whether it passed.
    """
    conditions = normalise_conditions(rule)
    period = getattr(rule, "period", None) or "6mo"
    resolver = MetricResolver(rule.symbol, period)

    results: list[dict] = []
    for condition in conditions:
        metric = condition.get("metric")
        operator = condition.get("operator", "above")
        target = condition.get("value")

        if metric in CATEGORICAL_METRICS:
            observed = resolver.label(metric)
            passed = (observed is not None
                      and str(observed).lower() == str(target).lower())
            results.append({
                "metric": metric, "operator": "is", "target": target,
                "observed": observed, "passed": bool(passed),
                "description": describe_condition(condition),
                "reason": None if observed is not None else "value unavailable",
            })
            continue

        observed = resolver.value(metric)
        if observed is None:
            # Unknown is not a pass. Firing on missing data would be worse than
            # staying quiet, and the reason is recorded so it can be diagnosed.
            results.append({
                "metric": metric, "operator": operator, "target": target,
                "observed": None, "passed": False,
                "description": describe_condition(condition),
                "reason": f"{metric} could not be computed over {period}",
            })
            continue

        compare = OPERATORS.get(operator, OPERATORS["above"])[1]
        try:
            passed = bool(compare(observed, float(target)))
        except (TypeError, ValueError):
            passed = False
        results.append({
            "metric": metric, "operator": operator, "target": target,
            "observed": round(observed, 6), "passed": passed,
            "description": describe_condition(condition), "reason": None,
        })

    logic = (getattr(rule, "logic", None) or "AND").upper()
    passed_flags = [r["passed"] for r in results]
    fired = all(passed_flags) if logic == "AND" else any(passed_flags)
    # An empty rule must never fire: `all([])` is True, which would make a
    # malformed rule alert on every single pass.
    if not results:
        fired = False

    return {
        "fired": fired,
        "logic": logic,
        "period": period,
        "conditions": results,
        "met": sum(passed_flags),
        "total": len(results),
    }


def build_message(rule, verdict: dict) -> str:
    """Why this alert exists, in one sentence a user can act on."""
    met = [c for c in verdict["conditions"] if c["passed"]]
    joiner = " and " if verdict["logic"] == "AND" else " or "
    parts = []
    for condition in met:
        observed = condition["observed"]
        shown = observed if isinstance(observed, str) else f"{observed:,.2f}"
        parts.append(f"{condition['description']} (now {shown})")
    body = joiner.join(parts) if parts else "conditions met"
    return f"{rule.symbol}: {body}."


def is_expired(rule, now: datetime | None = None) -> bool:
    expires = getattr(rule, "expires_at", None)
    if not expires:
        return False
    if expires.tzinfo is None:
        expires = expires.replace(tzinfo=UTC)
    return (now or datetime.now(UTC)) >= expires


def validate_conditions(conditions: list[dict]) -> list[str]:
    """Reject a rule that cannot ever be evaluated, at creation time.

    Saving a rule that silently never fires is the failure mode this prevents:
    the user believes they are covered when nothing is watching.
    """
    problems: list[str] = []
    if not conditions:
        return ["A rule needs at least one condition."]
    for index, condition in enumerate(conditions, start=1):
        metric = condition.get("metric")
        if metric in CATEGORICAL_METRICS:
            choices = REGIME_CHOICES if metric == "regime" else ACTION_CHOICES
            if str(condition.get("value")) not in choices:
                problems.append(
                    f"Condition {index}: '{condition.get('value')}' is not one of "
                    f"{', '.join(choices)}.")
            continue
        if metric not in METRICS_BY_KEY:
            problems.append(f"Condition {index}: unknown metric '{metric}'.")
            continue
        if condition.get("operator") not in OPERATORS:
            problems.append(
                f"Condition {index}: unknown operator '{condition.get('operator')}'.")
        try:
            float(condition.get("value"))
        except (TypeError, ValueError):
            problems.append(f"Condition {index}: '{condition.get('value')}' is not a number.")
    return problems
