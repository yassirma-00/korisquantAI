"""Automatic hyperparameter selection from the training environment.

What this is
------------
A rule-based tuner that reads six signals — algorithm, market regime, dataset
size, asset count, hardware and objective — and emits a complete configuration
plus a plain-language summary. It sits *on top of* `configs/`: the profiles
remain the source of truth, this chooses which one to start from and adjusts
specific keys with a stated reason for each.

What it is not
--------------
Not a search. It does not train candidates and compare them, so it cannot claim
to have found an optimum — it applies documented heuristics. Calling it
"optimal" would overstate what a rule table can know; the API says "recommended"
and every adjustment carries the reason it was made.

Timing estimates are measured, not guessed
------------------------------------------
Throughput was benchmarked on this platform's own environments (2-core CPU,
502-bar AAPL window), in **steady state**:

* native agents (DQN family, C51/IQN/Rainbow): **2.476 ms** per environment step
* SB3 agents (PPO, A2C, SAC, TD3, DDPG, TRPO, QR-DQN): **1.108 ms** per step

Steady state matters, and getting it wrong is easy. A first benchmark over two
episodes reported 0.181 ms/step for the native agents — 13x too fast — because
the replay buffer had not yet reached `min_buffer`, so `learn_step()` was
returning immediately and the measurement was mostly no-op. Validated against a
real run afterwards: the naive estimate was 10.8x under the true wall clock.
These figures are the marginal cost between a 10-episode and a 20-episode run,
which excludes that warm-up.

Even so, this covers the training loop only — not the data fetch or the final
evaluation — and it was measured on one machine. The API labels it an order of
magnitude, not a promise.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from app.core.logging import get_logger

logger = get_logger(__name__)

# Measured in steady state on this platform; see the module docstring.
# Seconds per environment step.
STEP_SECONDS = {"native": 0.002476, "sb3": 0.001108, "sb3_contrib": 0.001108}

# Fixed cost per run that is not part of the stepping loop: fetching history,
# building features and running the final evaluation. Measured at roughly 3-4 s
# on the benchmark machine. Omitting it made short runs look instant.
FIXED_OVERHEAD_SECONDS = 3.5

# High-level profiles offered to a standard user. Each maps onto a YAML profile
# that already exists, so nothing here bypasses the configuration system.
USER_PROFILES = {
    "conservative": {
        "label": "Conservative",
        "objective": "Protect capital. Heavier risk penalties, smaller trades.",
        "base": "conservative",
    },
    "balanced": {
        "label": "Balanced",
        "objective": "Default trade-off between return and drawdown.",
        "base": "default",
    },
    "high_performance": {
        "label": "High Performance",
        "objective": "Longer training and wider networks. Same objective, more compute.",
        "base": "high_performance",
    },
    "risk_aware": {
        "label": "Risk-Aware",
        "objective": "Adapt to the market regime; cautious only when conditions warrant.",
        "base": "risk_aware",
    },
    "ai_recommended": {
        "label": "AI Recommended",
        "objective": "Chosen automatically from the current environment.",
        "base": None,          # resolved by `recommend_profile`
    },
}

# Which base profile suits which detected regime. Deliberately asymmetric: the
# cost of being cautious in a bull market is opportunity, the cost of being
# aggressive into a crash is capital.
REGIME_PROFILE = {
    "crash_risk": "conservative",
    "bear_market": "conservative",
    "high_volatility": "risk_aware",
    "sideways": "default",
    "low_volatility": "default",
    "recovery": "risk_aware",
    "bull_market": "default",
    "unknown": "default",
}


@dataclass
class Environment:
    """The observable facts a recommendation is derived from."""

    algo: str
    backend: str = "native"
    action_space: str = "discrete"
    bars: int = 0
    n_assets: int = 1
    regime: str = "unknown"
    regime_confidence: float = 0.0
    volatility: float | None = None
    cpu_count: int = 1
    cuda: bool = False
    objective: str = "balanced"
    notes: list[str] = field(default_factory=list)


def detect_hardware() -> dict:
    """What the machine can actually offer. No GPU is not an error."""
    cuda = False
    try:
        import torch

        cuda = bool(torch.cuda.is_available())
    except Exception:      # pragma: no cover - torch optional at import time
        pass
    return {"cpu_count": os.cpu_count() or 1, "cuda": cuda}


def inspect_environment(symbol: str, algo: str, period: str = "3y",
                        objective: str = "balanced",
                        symbols: list[str] | None = None) -> Environment:
    """Gather the six signals. Every one is measured, none is assumed.

    Data and regime lookups fail soft: an unreachable provider must degrade the
    recommendation to a documented default, not abort the training request the
    user actually made.
    """
    from app.services.rl.catalogue import get_algorithm

    hardware = detect_hardware()
    spec = get_algorithm(algo)
    env = Environment(
        algo=algo,
        backend=(spec.backend if spec else "native"),
        action_space=(spec.action_space if spec else "discrete"),
        n_assets=len(symbols) if symbols else 1,
        objective=objective,
        cpu_count=hardware["cpu_count"],
        cuda=hardware["cuda"],
    )

    try:
        from app.services.data.market_data import market_data_service

        df = market_data_service.get_history(symbol, period=period).df
        env.bars = int(len(df))
        returns = df["close"].pct_change().dropna()
        if len(returns) > 20:
            env.volatility = float(returns.std() * (252 ** 0.5))
    except Exception as exc:      # pragma: no cover - provider dependent
        env.notes.append(f"market data unavailable ({str(exc)[:60]}); "
                         "sizing falls back to the profile defaults")
        logger.info("autotune could not read history for %s: %s", symbol, exc)
        return env

    try:
        from app.services.risk.regime import market_regime_detector

        verdict = market_regime_detector.detect(symbol, df, timeline_step=21)
        env.regime = verdict.get("regime") or "unknown"
        env.regime_confidence = float(verdict.get("confidence") or 0.0)
    except Exception as exc:      # pragma: no cover - provider dependent
        env.notes.append("market regime could not be classified; "
                         "regime-specific tuning was skipped")
        logger.info("autotune could not classify regime for %s: %s", symbol, exc)

    return env


def recommend_profile(env: Environment) -> tuple[str, str]:
    """Pick the base YAML profile, and say why."""
    if env.objective in ("conservative", "high_performance", "risk_aware"):
        return env.objective, f"you asked for the {env.objective} objective"

    base = REGIME_PROFILE.get(env.regime, "default")
    if env.regime_confidence < 0.4:
        # A low-confidence regime call is not a reason to change stance. Acting
        # on a coin-flip classification would make the recommendation swing
        # between runs for no measurable reason.
        return "default", (
            f"the market regime was classified as {env.regime.replace('_', ' ')} "
            f"but only at {env.regime_confidence:.0%} confidence, which is too "
            f"weak to tune on")
    return base, (
        f"the market is in a {env.regime.replace('_', ' ')} regime "
        f"({env.regime_confidence:.0%} confidence)")


def derive_overrides(env: Environment, base: str) -> list[dict]:
    """Adjustments on top of the base profile, each with its justification.

    Returned as a list rather than a dict so the UI can show *why* each value
    was chosen. A silent override is indistinguishable from a hardcoded value.
    """
    out: list[dict] = []

    # --- episode budget scales with how much data there is to learn from
    if env.bars:
        if env.bars < 300:
            episodes = 12
            reason = f"only {env.bars} bars: more episodes would re-read the same short window"
        elif env.bars < 800:
            episodes = 25
            reason = f"{env.bars} bars supports a moderate episode budget"
        else:
            episodes = 40
            reason = f"{env.bars} bars justifies a longer run"
        out.append({"path": "training.episodes", "value": episodes, "reason": reason})

    # --- batch size follows available cores, not preference
    if env.cpu_count >= 8:
        out.append({"path": "optimizer.batch_size", "value": 128,
                    "reason": f"{env.cpu_count} CPU cores available"})
    elif env.cpu_count <= 2:
        out.append({"path": "optimizer.batch_size", "value": 32,
                    "reason": f"only {env.cpu_count} CPU core(s); a smaller batch "
                              f"keeps each update cheap"})

    # --- device
    if env.cuda:
        out.append({"path": "training.device", "value": "cuda",
                    "reason": "a CUDA device was detected"})

    # --- network width scales with the observation, which grows with assets
    if env.n_assets >= 4:
        out.append({"path": "network.hidden", "value": [256, 256],
                    "reason": f"{env.n_assets} assets widen the observation vector"})

    # --- volatility drives risk penalties
    if env.volatility is not None and env.volatility > 0.45:
        out.append({"path": "risk.risk_penalty", "value": 0.30,
                    "reason": f"realised volatility is {env.volatility:.0%}, "
                              f"well above a typical equity"})
        out.append({"path": "risk.cvar_penalty", "value": 0.20,
                    "reason": "high volatility raises the value of penalising the tail"})

    # --- regime awareness, only when the classifier is confident enough to use
    if env.regime_confidence >= 0.5 and env.regime in (
            "crash_risk", "bear_market", "high_volatility", "recovery"):
        out.append({"path": "risk.regime_aware", "value": True,
                    "reason": f"a {env.regime.replace('_', ' ')} regime is where "
                              f"regime adaptation pays for its extra observation width"})

    # --- exploration: a plateau-prone continuous agent benefits from more
    if env.action_space == "continuous" and env.bars and env.bars > 500:
        out.append({"path": "policy_gradient.ent_coef", "value": 0.02,
                    "reason": "continuous control on a long window: a higher "
                              "entropy bonus delays premature convergence"})

    return out


def estimate_training(env: Environment, params: dict) -> dict:
    """Approximate wall-clock time, from measured throughput.

    This is the one number a user will plan around, so its basis is stated
    rather than implied. It is a linear extrapolation from a benchmark on one
    machine; it will be wrong on different hardware, and says so.
    """
    training = params.get("training") or {}
    episodes = int(training.get("episodes") or 20)
    total_timesteps = training.get("total_timesteps")

    bars = env.bars or 500
    steps = int(total_timesteps) if total_timesteps else episodes * bars
    rate = STEP_SECONDS.get(env.backend, STEP_SECONDS["native"])
    if env.cuda:
        # A GPU helps the update, not the environment step, which stays on CPU.
        # Claiming a large speed-up here would over-promise.
        rate *= 0.75

    seconds = steps * rate + FIXED_OVERHEAD_SECONDS
    return {
        "steps": steps,
        "seconds": round(seconds, 1),
        "human": _human_duration(seconds),
        "basis": (
            f"{steps:,} environment steps at {rate * 1000:.3f} ms/step "
            f"(steady state, {env.backend} backend) plus "
            f"{FIXED_OVERHEAD_SECONDS:.0f}s for the data fetch and final evaluation"),
        "caveat": ("Measured on a 2-core CPU with no GPU, and validated against "
                   "a real run. Different hardware will differ; treat this as an "
                   "order of magnitude, not a promise."),
    }


def _human_duration(seconds: float) -> str:
    if seconds < 60:
        return f"~{int(seconds)}s"
    if seconds < 3600:
        return f"~{seconds / 60:.0f} min"
    return f"~{seconds / 3600:.1f} h"


def expected_quality(env: Environment, params: dict) -> dict:
    """A bounded expectation of model quality, with the reasoning shown.

    Deliberately *not* a predicted return. Nothing here can forecast what a
    policy will earn; what the inputs support is whether the run is adequately
    resourced — enough data, enough episodes, an algorithm suited to the action
    space. Presenting that as an expected P&L would be the single most
    misleading thing this page could do.
    """
    factors: list[dict] = []

    bars = env.bars or 0
    data_score = min(bars / 1000.0, 1.0)
    factors.append({
        "name": "Data sufficiency", "value": round(data_score, 3),
        "detail": f"{bars} training bars (saturates at 1000)"})

    episodes = int((params.get("training") or {}).get("episodes") or 20)
    budget_score = min(episodes / 40.0, 1.0)
    factors.append({
        "name": "Training budget", "value": round(budget_score, 3),
        "detail": f"{episodes} episodes (saturates at 40)"})

    fit = 1.0 if _algo_fits(env) else 0.55
    factors.append({
        "name": "Algorithm fit", "value": fit,
        "detail": (f"{env.algo} suits a {env.action_space} action space"
                   if fit == 1.0 else
                   f"{env.algo} is being run outside its natural action space")})

    regime_fit = 1.0 if env.regime_confidence >= 0.5 else 0.7
    factors.append({
        "name": "Regime clarity", "value": regime_fit,
        "detail": (f"{env.regime.replace('_', ' ')} at "
                   f"{env.regime_confidence:.0%} confidence")})

    score = sum(f["value"] for f in factors) / len(factors)
    band = ("strong" if score >= 0.85 else "reasonable" if score >= 0.65
            else "limited" if score >= 0.45 else "weak")
    return {
        "score": round(score, 3),
        "percent": round(score * 100, 1),
        "band": band,
        "factors": factors,
        "meaning": (
            "How well-resourced this run is — data, episodes, algorithm fit and "
            "regime clarity. It is NOT a predicted return: nothing here can "
            "forecast what a policy will earn."),
    }


def _algo_fits(env: Environment) -> bool:
    if env.action_space == "continuous":
        return env.n_assets >= 1
    return env.n_assets == 1


def confidence(env: Environment) -> dict:
    """How much to trust the recommendation itself."""
    reasons: list[str] = []
    score = 1.0

    if not env.bars:
        score -= 0.35
        reasons.append("market data could not be read, so sizing is untuned")
    elif env.bars < 300:
        score -= 0.15
        reasons.append(f"only {env.bars} bars of history")

    if env.regime == "unknown":
        score -= 0.20
        reasons.append("the market regime is unknown")
    elif env.regime_confidence < 0.4:
        score -= 0.10
        reasons.append(f"regime confidence is only {env.regime_confidence:.0%}")

    if env.volatility is None:
        score -= 0.05
        reasons.append("realised volatility could not be measured")

    score = max(0.05, min(1.0, score))
    return {
        "score": round(score, 3),
        "percent": round(score * 100, 1),
        "reasons": reasons or ["every input signal was available and clear"],
        "basis": ("Confidence in the *recommendation*, not in the resulting "
                  "model. It falls when an input signal is missing or weak."),
    }


def recommend(symbol: str, algo: str, period: str = "3y",
              objective: str = "balanced",
              symbols: list[str] | None = None) -> dict:
    """The full recommendation: profile, overrides, estimates and summary."""
    from app.services.rl.hyperparams import hyperparameters

    env = inspect_environment(symbol, algo, period, objective, symbols)
    base, base_reason = recommend_profile(env)
    overrides = derive_overrides(env, base)

    override_map = {o["path"]: o["value"] for o in overrides}
    try:
        resolved = hyperparameters.resolve(algo, base, override_map)
        params = resolved.params
        fingerprint = resolved.fingerprint
    except Exception as exc:      # pragma: no cover - defensive
        # A rejected override must not block training: fall back to the base
        # profile untouched and say that is what happened.
        logger.warning("autotune overrides rejected, using %s as-is: %s", base, exc)
        resolved = hyperparameters.resolve(algo, base)
        params, fingerprint = resolved.params, resolved.fingerprint
        overrides = []
        env.notes.append(f"automatic adjustments were rejected ({str(exc)[:80]}); "
                         f"the {base} profile is used unmodified")

    timing = estimate_training(env, params)
    quality = expected_quality(env, params)
    trust = confidence(env)

    return {
        "symbol": symbol.upper(),
        "algo": algo,
        "objective": objective,
        "profile": base,
        "profile_label": USER_PROFILES.get(
            objective, {}).get("label", base.replace("_", " ").title()),
        "profile_reason": base_reason,
        "environment": {
            "bars": env.bars,
            "n_assets": env.n_assets,
            "regime": env.regime,
            "regime_confidence": round(env.regime_confidence, 3),
            "volatility": None if env.volatility is None else round(env.volatility, 4),
            "cpu_count": env.cpu_count,
            "cuda": env.cuda,
            "backend": env.backend,
            "action_space": env.action_space,
        },
        "adjustments": overrides,
        "estimated_training": timing,
        "expected_quality": quality,
        "confidence": trust,
        # Stored for reproducibility and experiment tracking, but the standard
        # UI shows only the summary above.
        "resolved_hyperparameters": params,
        "fingerprint": fingerprint,
        "notes": env.notes,
        "summary": _summary(env, base, base_reason, overrides, timing, quality, trust),
        "method": (
            "Rule-based selection from the observed environment, not a search: "
            "no candidate configurations were trained and compared, so this is "
            "a recommendation rather than a proven optimum."),
    }


def _summary(env, base, base_reason, overrides, timing, quality, trust) -> str:
    parts = [
        f"Using the {base.replace('_', ' ')} profile because {base_reason}.",
        f"{len(overrides)} automatic adjustment(s) applied for "
        f"{env.bars or 'unknown'} bars of data, {env.n_assets} asset(s) and "
        f"{env.cpu_count} CPU core(s).",
        f"Estimated {timing['human']} to train. "
        f"Setup quality {quality['band']} ({quality['percent']:.0f}/100), "
        f"recommendation confidence {trust['percent']:.0f}%.",
    ]
    return " ".join(parts)
