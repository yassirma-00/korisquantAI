"""Training intelligence: convergence diagnosis, health scoring, recommendations.

Read-only analytics over what training already recorded. Nothing here retrains,
re-evaluates or recomputes a metric the backend produced — every figure is
either read from an agent's metadata sidecar or derived from it arithmetically,
and the derivation is named wherever it happens.

What the recorded data does and does not support
------------------------------------------------
Audited across the 14 trained agents on disk before this module was written:

* **Available per evaluation:** total_return, sharpe_ratio, sortino_ratio,
  max_drawdown, annualised_volatility, var_95, cvar_95, final_value.
* **Available per run:** episode rewards, losses, per-episode Sharpe, portfolio
  values, checkpoints, hyperparameters, seed, profile, fingerprint,
  `regime_aware` (which is what makes a legacy-vs-adaptive comparison real).
* **Derivable:** turnover from `total_transaction_cost / (fee + slippage) /
  capital`; per-episode win rate from the reward series; cumulative reward.
* **Not available, and therefore not invented:** wall-clock training duration
  (never timed — only evaluation seconds are), per-*trade* win rate (trades are
  not persisted in the sidecar), and multi-seed dispersion (every run on disk
  used seed 42, so a "mean ± std" would be one sample wearing an error bar).

Those three are reported as unavailable with the reason attached, rather than
filled with a plausible-looking number.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass

from app.core.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------- thresholds
#
# Calibration constants for the diagnosis, stated here rather than buried in
# the logic so a reader can disagree with them explicitly.
MIN_EPISODES_FOR_TREND = 6      # below this, any trend is noise
PLATEAU_BAND = 0.05             # |slope| under 5% of scale = flat
UNSTABLE_CV = 0.75              # reward std / |mean| above this = erratic
OVERFIT_GAP = 0.10              # train up, eval down by this much = overfitting

# Health-score weights. Sum to 1.0 (asserted below) so the composite is a
# genuine weighted mean rather than an arbitrary total.
HEALTH_WEIGHTS = {
    "performance": 0.30,    # did it beat its benchmark, out of sample
    "stability": 0.20,      # is the reward series erratic
    "convergence": 0.20,    # did learning actually settle
    "robustness": 0.15,     # does held-out agree with training
    "risk": 0.15,           # drawdown and tail behaviour
}
assert abs(sum(HEALTH_WEIGHTS.values()) - 1.0) < 1e-9

STATUS_LABELS = {
    "converged": "Converged",
    "improving": "Still improving",
    "plateaued": "Plateaued",
    "overfitting": "Overfitting",
    "unstable": "Unstable",
    "insufficient_data": "Not enough episodes",
}

# Action per diagnosis. Deliberately conservative: the cost of training a
# little longer is compute, the cost of shipping an overfitted agent is money.
RECOMMENDATIONS = {
    "improving": ("Continue Training",
                  "Reward is still trending up and the held-out score has not "
                  "turned down. Stopping now leaves performance on the table."),
    "converged": ("Stop Training",
                  "Reward has settled and the held-out score agrees. Further "
                  "episodes cost compute without changing the policy."),
    "plateaued": ("Reduce Learning Rate",
                  "Reward stopped moving while the held-out score is not "
                  "falling. A smaller step size often resolves the last few "
                  "percent; if it does not, tune the hyperparameters."),
    "overfitting": ("Stop Training",
                    "Training reward is rising while the held-out score falls. "
                    "The policy is memorising the training window; more "
                    "episodes make this worse, not better."),
    "unstable": ("Tune Hyperparameters",
                 "Episode reward swings far more than it trends. Lower the "
                 "learning rate or raise the batch size before reading "
                 "anything into the result."),
    "insufficient_data": ("Continue Training",
                          "Too few episodes to diagnose anything. Any verdict "
                          "here would be noise."),
}


@dataclass(frozen=True)
class Diagnosis:
    status: str
    confidence: float
    evidence: list[str]
    action: str
    rationale: str


def _safe_mean(values: list[float]) -> float:
    return float(statistics.fmean(values)) if values else 0.0


def _trend(series: list[float]) -> float:
    """Normalised slope of a series, in units of its own scale.

    Comparing the last third against the middle third is deliberately crude:
    a least-squares fit on 6 noisy episodes reports a confident slope that a
    single outlier can flip. Thirds are harder to fool.
    """
    if len(series) < MIN_EPISODES_FOR_TREND:
        return 0.0
    third = max(len(series) // 3, 1)
    early = _safe_mean(series[third:2 * third]) or _safe_mean(series[:third])
    late = _safe_mean(series[-third:])
    scale = max(abs(early), abs(late), 1e-9)
    return (late - early) / scale


def diagnose(training_history: dict, evaluations: list[dict]) -> Diagnosis:
    """Classify how a training run behaved, from its own recorded series."""
    rewards = list((training_history or {}).get("episode_rewards") or [])
    evidence: list[str] = []

    if len(rewards) < MIN_EPISODES_FOR_TREND:
        return Diagnosis(
            "insufficient_data", 0.0,
            [f"only {len(rewards)} episode(s) recorded; "
             f"{MIN_EPISODES_FOR_TREND} needed to read a trend"],
            *RECOMMENDATIONS["insufficient_data"])

    slope = _trend(rewards)
    mean_reward = _safe_mean(rewards)
    spread = float(statistics.pstdev(rewards)) if len(rewards) > 1 else 0.0
    cv = spread / max(abs(mean_reward), 1e-9)
    evidence.append(f"reward trend {slope:+.1%} over the last third of training")
    evidence.append(f"reward dispersion {cv:.2f}x its own mean")

    # Held-out trend, when periodic evaluation ran. This is the only way to
    # separate genuine learning from memorising the training window.
    scored = [e for e in (evaluations or []) if e.get("total_return") is not None]
    eval_slope = None
    if len(scored) >= 2:
        eval_series = [e["total_return"] for e in scored]
        eval_slope = eval_series[-1] - eval_series[0]
        evidence.append(
            f"held-out return moved {eval_slope:+.2%} between the first and "
            f"last of {len(scored)} evaluations")
    else:
        evidence.append(
            "no held-out trend: fewer than two evaluations were recorded, so "
            "overfitting cannot be ruled out")

    # Order matters. Overfitting is checked first because it can look like
    # healthy improvement if only the training curve is read.
    if eval_slope is not None and slope > PLATEAU_BAND and eval_slope < -OVERFIT_GAP:
        status, confidence = "overfitting", 0.85
    elif cv > UNSTABLE_CV:
        status, confidence = "unstable", 0.7
    elif abs(slope) <= PLATEAU_BAND:
        # Flat is only "converged" if the held-out score corroborates it;
        # otherwise it is a plateau that may still be recoverable.
        settled = eval_slope is None or eval_slope >= -OVERFIT_GAP / 2
        status = "converged" if settled and cv < UNSTABLE_CV / 2 else "plateaued"
        confidence = 0.75 if settled else 0.6
    elif slope > PLATEAU_BAND:
        status, confidence = "improving", 0.8
    else:
        status, confidence = "plateaued", 0.65

    action, rationale = RECOMMENDATIONS[status]
    return Diagnosis(status, round(confidence, 2), evidence, action, rationale)


# ------------------------------------------------------------- health score
def _contribution(name: str, raw: float | None, low: float, high: float,
                  weight: float, detail: str, invert: bool = False) -> dict:
    """One bounded term of the health score."""
    available = raw is not None
    value = None
    if available:
        span = (high - low) or 1.0
        value = (float(raw) - low) / span
        value = max(0.0, min(1.0, value))
        if invert:
            value = 1.0 - value
    return {
        "name": name, "weight": round(weight, 4),
        "raw": None if raw is None else round(float(raw), 6),
        "value": None if value is None else round(value, 4),
        "points": None if value is None else round(value * weight * 100, 2),
        "max_points": round(weight * 100, 2),
        "detail": detail, "available": bool(available),
    }


def health_score(meta: dict, diagnosis: Diagnosis) -> dict:
    """A 0-100 quality score for one trained model, with its arithmetic shown.

    Five dimensions, each normalised against an absolute reference range so two
    models are comparable. An unmeasurable dimension is dropped and its weight
    redistributed — never scored as zero, because zero means "measured, and
    bad".
    """
    performance = meta.get("test_performance") or {}
    baselines = meta.get("baselines") or {}
    history = meta.get("training_history") or {}
    rewards = list(history.get("episode_rewards") or [])

    # Single-asset runs benchmark against buy-and-hold; portfolio runs against
    # an equal-weight basket. Looking only for buy-and-hold silently dropped
    # the Performance term on every portfolio agent, redistributing its 30%
    # across the remaining dimensions — a SAC basket that LOST 8.6% ranked 4th
    # overall at 80.6% health, because the one dimension it failed was the one
    # being ignored. Prefer the alpha the run already computed against whatever
    # benchmark applies to it.
    alpha = performance.get("alpha_vs_buy_hold")
    if alpha is None:
        alpha = performance.get("alpha_vs_equal_weight")
    if alpha is None:
        benchmark = (baselines.get("buy_and_hold") or {}).get("total_return")
        if benchmark is None:
            benchmark = (performance.get("buy_and_hold_return")
                         or performance.get("equal_weight_return"))
        total_return = performance.get("total_return")
        alpha = (None if total_return is None or benchmark is None
                 else total_return - benchmark)

    cv = None
    if len(rewards) > 1:
        mean = _safe_mean(rewards)
        cv = float(statistics.pstdev(rewards)) / max(abs(mean), 1e-9)

    convergence_value = {
        "converged": 1.0, "improving": 0.7, "plateaued": 0.45,
        "unstable": 0.2, "overfitting": 0.05, "insufficient_data": None,
    }[diagnosis.status]

    scored = [e for e in ((meta.get("monitoring") or {}).get("evaluations") or [])
              if e.get("total_return") is not None]
    # Robustness = does the held-out result agree with the final report? A run
    # whose evaluations swung wildly is not robust even if it ended well.
    robustness = None
    if len(scored) >= 2:
        returns = [e["total_return"] for e in scored]
        robustness = float(statistics.pstdev(returns))

    drawdown = performance.get("max_drawdown")

    rows = [
        _contribution("Performance vs benchmark", alpha, -0.20, 0.20,
                      HEALTH_WEIGHTS["performance"],
                      "no benchmark recorded" if alpha is None
                      else f"{alpha:+.2%} against buy-and-hold out of sample"),
        _contribution("Stability", cv, 0.0, 1.5, HEALTH_WEIGHTS["stability"],
                      "needs at least two episodes" if cv is None
                      else f"reward dispersion {cv:.2f}x its mean (lower is better)",
                      invert=True),
        _contribution("Convergence", convergence_value, 0.0, 1.0,
                      HEALTH_WEIGHTS["convergence"],
                      "not enough episodes to diagnose"
                      if convergence_value is None
                      else f"diagnosed as {STATUS_LABELS[diagnosis.status].lower()}"),
        _contribution("Robustness", robustness, 0.0, 0.30,
                      HEALTH_WEIGHTS["robustness"],
                      "needs two or more evaluations" if robustness is None
                      else f"held-out return varied by {robustness:.2%} across "
                           f"{len(scored)} evaluations (lower is better)",
                      invert=True),
        _contribution("Risk", None if drawdown is None else abs(drawdown),
                      0.05, 0.50, HEALTH_WEIGHTS["risk"],
                      "no drawdown recorded" if drawdown is None
                      else f"max drawdown {abs(drawdown):.1%} (lower is better)",
                      invert=True),
    ]

    live = [r for r in rows if r["available"]]
    total_weight = sum(r["weight"] for r in live)
    if total_weight <= 0:
        return {"score": None, "grade": "unknown", "contributions": rows,
                "explanation": "Nothing measurable was recorded for this run."}

    score = sum(r["value"] * r["weight"] for r in live) / total_weight
    scale = 1.0 / total_weight
    for row in rows:
        if row["available"]:
            row["points"] = round(row["value"] * row["weight"] * scale * 100, 2)
            row["max_points"] = round(row["weight"] * scale * 100, 2)
        else:
            row["points"], row["max_points"] = None, 0.0

    grade = ("excellent" if score >= 0.80 else "good" if score >= 0.62
             else "fair" if score >= 0.45 else "poor")
    missing = [r["name"] for r in rows if not r["available"]]
    top = sorted(live, key=lambda r: r["points"], reverse=True)[:2]

    return {
        "score": round(score, 4),
        "percent": round(score * 100, 1),
        "grade": grade,
        "contributions": rows,
        "weight_redistributed": bool(missing),
        "unmeasured": missing,
        "explanation": (
            f"{grade.title()} ({score * 100:.0f}/100), carried mainly by "
            + ", ".join(f"{r['name'].lower()} ({r['points']:.0f} pts)" for r in top)
            + (f". {len(missing)} dimension(s) could not be measured "
               f"({', '.join(m.lower() for m in missing)}); the remaining "
               f"weights were rescaled." if missing else ".")),
    }


# ------------------------------------------------------------- derived stats
def derived_metrics(meta: dict) -> dict:
    """Figures computed from recorded data, with each derivation named."""
    performance = meta.get("test_performance") or {}
    env = meta.get("env_config") or {}
    history = meta.get("training_history") or {}
    rewards = list(history.get("episode_rewards") or [])

    # Turnover: the environment charged `transaction_cost + slippage` on every
    # unit of notional traded, so the total cost divided by that rate is the
    # notional, and dividing again by starting capital expresses it as a
    # multiple of the book.
    turnover = None
    cost = performance.get("total_transaction_cost")
    rate = float(env.get("transaction_cost", 0) or 0) + float(env.get("slippage", 0) or 0)
    capital = float(env.get("initial_balance", 0) or 0)
    if cost is not None and rate > 0 and capital > 0:
        turnover = float(cost) / rate / capital

    return {
        "cumulative_reward": round(sum(rewards), 2) if rewards else None,
        "mean_episode_reward": round(_safe_mean(rewards), 2) if rewards else None,
        # Per *episode*, not per trade: individual trades are not persisted in
        # the sidecar, so a trade-level win rate cannot be computed from it.
        "episode_win_rate": (round(sum(1 for r in rewards if r > 0) / len(rewards), 4)
                             if rewards else None),
        "win_rate_basis": "episodes with positive reward (trade-level P&L is not stored)",
        "turnover": None if turnover is None else round(turnover, 3),
        "turnover_basis": ("derived from total_transaction_cost / (fee + slippage) "
                           "/ initial capital"),
        "n_trades": performance.get("n_trades"),
        # Never timed by the training loop. Reporting evaluation seconds as if
        # it were training duration would be a fabricated headline figure.
        "training_duration_seconds": None,
        "training_duration_basis": (
            "not recorded: the training loop is never timed. "
            "eval_seconds below covers evaluation only."),
        "eval_seconds": (meta.get("monitoring") or {}).get("eval_seconds"),
    }


def seed_statistics(runs: list[dict]) -> dict:
    """Mean ± standard deviation across seeds, when there are enough seeds.

    Every agent on disk was trained with seed 42, so this almost always reports
    that dispersion is unavailable. That is the honest answer: a standard
    deviation over one sample is zero by construction, and drawing it as an
    error band would suggest a reproducibility check that never happened.
    """
    seeds = {r.get("seed") for r in runs if r.get("seed") is not None}
    returns = [(r.get("test_performance") or {}).get("total_return")
               for r in runs]
    returns = [r for r in returns if r is not None]

    if len(seeds) < 3 or len(returns) < 3:
        return {
            "available": False,
            "seeds": sorted(s for s in seeds),
            "runs": len(runs),
            "reason": (
                f"{len(seeds)} distinct seed(s) across {len(runs)} run(s). "
                "Mean ± standard deviation needs at least 3 independent seeds; "
                "below that the spread measures nothing."),
        }
    return {
        "available": True,
        "seeds": sorted(seeds),
        "runs": len(runs),
        "mean_return": round(_safe_mean(returns), 4),
        "std_return": round(float(statistics.pstdev(returns)), 4),
    }


def analyse_run(meta: dict) -> dict:
    """Full intelligence payload for one trained agent."""
    monitoring = meta.get("monitoring") or {}
    evaluations = monitoring.get("evaluations") or []
    diagnosis = diagnose(meta.get("training_history") or {}, evaluations)
    health = health_score(meta, diagnosis)
    scored = [e for e in evaluations if e.get("total_return") is not None]
    latest = scored[-1] if scored else None
    best = max(scored, key=lambda e: e["total_return"]) if scored else None

    performance = meta.get("test_performance") or {}
    return {
        "symbol": meta.get("symbol") or meta.get("portfolio_key"),
        "algo": meta.get("algo"),
        "experiment_id": meta.get("experiment_id"),
        "profile": meta.get("profile"),
        "seed": meta.get("seed"),
        "model_version": meta.get("hyperparameter_fingerprint"),
        "trained_at": meta.get("trained_at"),
        "regime_aware": bool(meta.get("regime_aware")),
        "episodes": len(((meta.get("training_history") or {})
                         .get("episode_rewards")) or []),
        "status": diagnosis.status,
        "status_label": STATUS_LABELS[diagnosis.status],
        "status_confidence": diagnosis.confidence,
        "evidence": diagnosis.evidence,
        "recommendation": {"action": diagnosis.action,
                           "rationale": diagnosis.rationale},
        "health": health,
        "metrics": {
            "total_return": performance.get("total_return"),
            "annualised_return": performance.get("annualised_return"),
            "sharpe_ratio": performance.get("sharpe_ratio"),
            "sortino_ratio": performance.get("sortino_ratio"),
            "max_drawdown": performance.get("max_drawdown"),
            "annualised_volatility": performance.get("annualised_volatility"),
            "alpha_vs_benchmark": performance.get("alpha_vs_buy_hold")
            or performance.get("alpha_vs_equal_weight"),
            # VaR/CVaR are recorded per evaluation, not in the final report.
            "var_95": (latest or {}).get("var_95"),
            "cvar_95": (latest or {}).get("cvar_95"),
            **derived_metrics(meta),
        },
        "evaluation": {
            "count": len(evaluations),
            "latest": latest,
            "best": best,
            "eval_freq": monitoring.get("eval_freq", 0),
        },
        "checkpoints": len(monitoring.get("checkpoints") or []),
        "stale": bool(meta.get("stale")),
    }
