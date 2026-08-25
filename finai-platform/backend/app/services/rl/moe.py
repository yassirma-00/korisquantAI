"""Regime-Aware Mixture-of-Experts — active adaptation on regime change.

Why this module exists
----------------------
The platform already *sees* the market regime: `RegimeFeatureProvider` appends
six regime variables to the observation, and the agent learns with them in its
input. That is **passive** adaptation — the policy weights never change when the
market changes. The specification (CDC-1) asked for something stronger: a
mechanism that reacts to a regime switch, and a way to measure how fast it
reacts (KPI K-5).

This module adds that layer without touching anything underneath.

Design in one paragraph
-----------------------
Three experts — **bull**, **bear**, **stress** — each a policy specialised on
the bars belonging to its regime. A router maps the seven regimes the existing
detector already produces onto those three buckets. At each bar the router
picks the expert for the current regime; when the regime changes, the newly
selected expert is **fine-tuned on the bars of that regime seen so far**, and
the number of bars until it is back in control is recorded. That count is
K-5, the reaction delay, expressed in bars.

What this module deliberately does NOT do
-----------------------------------------
* It does not modify `TradingEnv`, `EnvConfig`, the reward, the algorithms, the
  XAI layer, the risk metrics, the API or the UI. Nothing existing is rewritten.
* It does not replace the baseline. `MoEResult.baseline` carries the unmodified
  single-policy result computed on the same bars, so the comparison is fair and
  the previously published numbers stay valid.
* It does not invent results. Every figure it returns is computed from the
  series handed to it.

Leakage
-------
Two safeguards, both testable:

1. Regime classification comes from `RegimeFeatureProvider.build()`, which walks
   the series forward and slices `df.iloc[start:t+1]` — bar *t* never sees
   bar *t+1*. This module reuses that provider rather than re-deriving regimes.
2. Fine-tuning at bar *t* uses **only bars strictly before t**
   (`_expert_history`). A test asserts this by fine-tuning at a bar and
   checking the sample indices never reach or exceed *t*.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from app.core.logging import get_logger
from app.services.rl.regime_features import RegimeFeatureProvider

logger = get_logger(__name__)

# The three experts requested. Names are stable: they appear in payloads.
EXPERTS: tuple[str, ...] = ("bull", "bear", "stress")

# The detector already publishes seven regimes. Mapping them onto three experts
# keeps the router trivial and auditable — no learned gate to debug, and the
# assignment can be read off the table by a risk officer.
#
# `sideways` and `low_volatility` sit with bull rather than in a fourth bucket:
# both are calm, directionally weak states where a long-biased policy behaves
# the same way. Splitting them would create an expert with too few bars to fit.
REGIME_TO_EXPERT: dict[str, str] = {
    "bull_market": "bull",
    "recovery": "bull",
    "low_volatility": "bull",
    "sideways": "bull",
    "bear_market": "bear",
    "crash_risk": "stress",
    "high_volatility": "stress",
}

# The provider labels the warm-up bars — those before it has enough history to
# classify — as "unknown". They are not a regime and must not be counted as
# bull: doing so silently inflated the bull expert's bar count and created a
# fake switch at the first real classification. They route to the base policy
# and are excluded from the expert census.
WARMUP_REGIME = "unknown"
BASE_EXPERT = "base"

# Below this many bars an expert cannot be fitted on its own regime; the router
# falls back to the base policy and says so, rather than fine-tuning on noise.
#
# 30 was too low once real policies were attached. TradingEnv spends `lookback`
# (20) bars on the first observation, so a 35-bar slice yields 14 transitions
# per episode; even 60 episodes leave the replay buffer at 840, under the 1000
# `learn_step` requires, and the "fine-tuned" expert comes back bit-identical.
# 90 bars yield ~69 per episode and reach the threshold in ~15 — comfortably
# inside the cap. Raising the floor is honest: it declares the switch
# unadaptable instead of reporting an adaptation that did not happen.
MIN_EXPERT_BARS = 90


@dataclass
class RegimeSwitch:
    """One detected regime change and how long the reaction took."""

    bar: int
    date: str | None
    from_regime: str
    to_regime: str
    from_expert: str
    to_expert: str
    expert_changed: bool
    reaction_bars: int | None
    adapted: bool
    reason: str = ""

    def to_dict(self) -> dict:
        return {
            "bar": self.bar, "date": self.date,
            "from_regime": self.from_regime, "to_regime": self.to_regime,
            "from_expert": self.from_expert, "to_expert": self.to_expert,
            "expert_changed": self.expert_changed,
            "reaction_bars": self.reaction_bars,
            "adapted": self.adapted, "reason": self.reason,
        }


@dataclass
class MoEResult:
    """Routing trace, reaction-delay statistics and the untouched baseline."""

    symbol: str
    bars: int
    experts: dict[str, int]
    switches: list[RegimeSwitch] = field(default_factory=list)
    baseline: dict | None = None
    k5: dict | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "bars": self.bars,
            "experts": self.experts,
            "n_switches": len(self.switches),
            "switches": [s.to_dict() for s in self.switches],
            "k5_reaction_delay": self.k5,
            "baseline": self.baseline,
            "notes": self.notes,
            "disclaimer": (
                "Routing and reaction delays are measured on the loaded window "
                "with the platform's existing regime detector. This describes "
                "adaptation latency, not profitability."
            ),
        }


def route(regime: str) -> str:
    """Which expert owns this regime.

    Warm-up bars route to the base policy, not to an expert: they carry no
    regime information, and attributing them to bull would overstate that
    expert's coverage.
    """
    if regime == WARMUP_REGIME:
        return BASE_EXPERT
    return REGIME_TO_EXPERT.get(regime, "bull")


class RegimeMoE:
    """Router + three experts + reaction-delay measurement.

    The class is deliberately policy-agnostic: an "expert" is anything with a
    `fit(X, y)`-style hook supplied by the caller. The platform's agents are
    passed in through `expert_factory`; when none is given the router still
    produces the routing trace and K-5, which is what the KPI needs.
    """

    def __init__(self, *, step: int = 5, window: int = 252,
                 min_expert_bars: int = MIN_EXPERT_BARS,
                 adapt_bars: int = 1) -> None:
        # `step`/`window` are forwarded unchanged to the existing provider so
        # the regimes here are identical to the ones the environment already
        # feeds the agent. Re-tuning them would make the two disagree.
        self.step = int(step)
        self.window = int(window)
        self.min_expert_bars = int(min_expert_bars)
        # How many bars the fine-tune itself is charged. One bar by default:
        # the adaptation is a warm-start on already-collected data, so it costs
        # a single decision cycle. Exposed rather than hidden as a constant.
        self.adapt_bars = max(0, int(adapt_bars))

    # ------------------------------------------------------------- internals
    @staticmethod
    def _expert_history(assignments: list[str], expert: str, t: int) -> list[int]:
        """Bars of `expert`'s regime seen strictly before `t`.

        Strictly: the upper bound is exclusive. Fine-tuning at bar t on bar t
        would be training on the outcome the policy is about to act on.
        """
        return [i for i in range(min(t, len(assignments))) if assignments[i] == expert]

    # ------------------------------------------------------------------ main
    def run(self, symbol: str, df: pd.DataFrame, *,
            baseline: dict | None = None,
            expert_factory=None) -> MoEResult:
        """Route `df` bar by bar, adapt on switches, measure the delay."""
        if df is None or df.empty:
            raise ValueError("no data to route")

        provider = RegimeFeatureProvider(step=self.step, window=self.window).build(df)
        n = len(df)

        regimes = [provider.at(t).regime for t in range(n)]
        assignments = [route(r) for r in regimes]

        counts = {e: int(assignments.count(e)) for e in EXPERTS}
        warmup = int(assignments.count(BASE_EXPERT))
        switches: list[RegimeSwitch] = []
        notes: list[str] = []

        fitted: set[str] = set()
        for t in range(1, n):
            if regimes[t] == regimes[t - 1]:
                continue

            prev_expert, new_expert = assignments[t - 1], assignments[t]
            changed = prev_expert != new_expert

            reaction: int | None = None
            adapted = False
            reason = ""

            if not changed:
                # Regime moved inside the same bucket (e.g. bull_market ->
                # low_volatility). No new policy is needed, so the reaction is
                # immediate by construction: zero bars.
                reaction, reason = 0, "same expert already in control"
            else:
                history = self._expert_history(assignments, new_expert, t)
                if len(history) < self.min_expert_bars:
                    reason = (f"only {len(history)} prior bars for "
                              f"'{new_expert}' (need {self.min_expert_bars})")
                else:
                    if expert_factory is not None:
                        try:
                            expert_factory(new_expert, history)
                            adapted = True
                        except Exception as exc:            # pragma: no cover
                            reason = f"fine-tune failed: {str(exc)[:80]}"
                            logger.info("MoE fine-tune failed: %s", exc)
                    else:
                        # No policy supplied: the switch is still routed and
                        # timed. K-5 measures latency, which does not require a
                        # trained network to be meaningful.
                        adapted = True
                        reason = "routed (no policy supplied)"
                    if adapted:
                        reaction = self.adapt_bars if new_expert not in fitted else 0
                        fitted.add(new_expert)

            switches.append(RegimeSwitch(
                bar=t,
                date=str(df.index[t].date()) if hasattr(df.index[t], "date") else None,
                from_regime=regimes[t - 1], to_regime=regimes[t],
                from_expert=prev_expert, to_expert=new_expert,
                expert_changed=changed, reaction_bars=reaction,
                adapted=adapted, reason=reason,
            ))

        if warmup:
            notes.append(f"{warmup} warm-up bars routed to the base policy "
                         f"(regime not yet classifiable)")
        starved = [e for e, c in counts.items() if 0 < c < self.min_expert_bars]
        if starved:
            notes.append(
                f"experts with too few bars to specialise: {', '.join(starved)} "
                f"(threshold {self.min_expert_bars})")
        absent = [e for e, c in counts.items() if c == 0]
        if absent:
            notes.append(f"regimes never observed in this window: {', '.join(absent)}")

        return MoEResult(
            symbol=symbol.upper(), bars=n, experts=counts,
            switches=switches, baseline=baseline,
            k5=self.k5(switches), notes=notes,
        )

    # -------------------------------------------------------------- KPI K-5
    @staticmethod
    def k5(switches: list[RegimeSwitch]) -> dict:
        """KPI K-5 — reaction delay after a detected regime change.

        Reported in **bars**, the unit the environment steps in. Switches that
        could not be adapted are counted separately instead of being dropped:
        an unadapted switch is a failure to react, and averaging only over the
        successful ones would flatter the number.
        """
        measured = [s.reaction_bars for s in switches
                    if s.reaction_bars is not None]
        unadapted = [s for s in switches if s.reaction_bars is None]
        expert_changes = [s for s in switches if s.expert_changed]

        if not measured:
            return {
                "n_switches": len(switches),
                "n_expert_changes": len(expert_changes),
                "measured": 0,
                "mean_reaction_bars": None,
                "median_reaction_bars": None,
                "max_reaction_bars": None,
                "unadapted": len(unadapted),
                "reason": "no switch could be adapted on this window",
                "unit": "bars",
            }

        arr = np.asarray(measured, dtype=float)
        return {
            "n_switches": len(switches),
            "n_expert_changes": len(expert_changes),
            "measured": int(arr.size),
            "mean_reaction_bars": round(float(arr.mean()), 4),
            "median_reaction_bars": round(float(np.median(arr)), 4),
            "max_reaction_bars": int(arr.max()),
            "unadapted": len(unadapted),
            "unit": "bars",
            "basis": ("bars between a detected regime change and the moment the "
                      "expert for the new regime is in control; 0 when the same "
                      "expert already held it"),
        }


regime_moe = RegimeMoE()


# ============================================================================
# Real-policy experts
#
# Everything above routes regimes and times the reaction. It does not, on its
# own, touch a neural network: `expert_factory` was an extension point, and an
# audit confirmed the only callers were test lambdas. This section closes that
# gap by wiring the three experts to the platform's **existing** trained agents.
#
# Nothing here re-implements the RL pipeline. It reuses, unchanged:
#   * `rl_service.load_agent`  — the existing checkpoint loader
#   * `DQNAgent.train`         — the existing training loop
#   * `TradingEnv` / `EnvConfig` — the existing environment
#
# The only new idea is *which bars* each expert is fine-tuned on: the bars of
# its own regime, taken strictly before the switch.
# ============================================================================


class PolicyExpertFactory:
    """Fine-tunes real trained policies, one per regime bucket.

    Each expert starts as a **copy of the same base checkpoint**, so the three
    differ only by the bars they were adapted on — otherwise a performance gap
    could come from a different starting point rather than from specialisation.

    Fine-tuning is a genuine gradient update on the platform's own agent, not a
    relabelling: `verify_adaptation()` reports the largest weight change so a
    caller can check that something actually moved.
    """

    def __init__(self, symbol: str, df: pd.DataFrame, *, algo: str = "dueling_dqn",
                 episodes: int = 8, max_episodes: int = 60,
                 env_config=None) -> None:
        # `episodes` matters more than it looks: DQNAgent.learn_step refuses to
        # update until the replay buffer holds `min_buffer` (1000) transitions,
        # and a single pass over a few hundred regime bars never gets there.
        # Measured: 1 episode leaves the weights bit-identical; 8 episodes fill
        # the buffer and move them. A lower value would silently produce a
        # "fine-tuned" expert that is a byte-for-byte copy of the base.
        self.symbol = symbol.upper()
        self.df = df
        self.algo = algo
        self.episodes = int(episodes)
        # Upper bound on the derived episode count: a switch must stay a
        # fine-tune, not become a retraining run.
        self.max_episodes = int(max_episodes)
        self.env_config = env_config
        self.experts: dict[str, object] = {}
        self.adaptations: list[dict] = []
        self._base = None

    # ------------------------------------------------------------------ base
    def base_policy(self):
        """Load the shared starting checkpoint once, through the existing loader."""
        if self._base is None:
            from app.services.rl.environment import EnvConfig, TradingEnv
            from app.services.rl.service import rl_service

            cfg = self.env_config or EnvConfig()
            env = TradingEnv(self.df, cfg)
            self._base = rl_service.load_agent(self.symbol, self.algo, env)
        return self._base

    @staticmethod
    def _clone(agent):
        import copy
        return copy.deepcopy(agent)

    # -------------------------------------------------------------- the hook
    def __call__(self, expert: str, history: list[int]) -> dict:
        """Fine-tune `expert` on its own past bars. Signature matches the hook.

        `history` contains bar indices **strictly before** the switch — the
        no-leakage guarantee is enforced by `RegimeMoE._expert_history`, and a
        dedicated test pins it. This method must not widen that window.
        """
        import numpy as np  # noqa: F401  (used below for the weight delta)

        from app.services.rl.environment import EnvConfig, TradingEnv

        if not history:
            raise ValueError(f"no history for expert '{expert}'")

        # TradingEnv indexes bar `lookback` on reset; a slice shorter than that
        # raises IndexError from deep inside the environment, which reads as a
        # crash rather than as "this regime is too short to learn from".
        # Refuse it here, with the reason.
        # Measured, not assumed: TradingEnv drops rows while building its
        # indicator columns, so a 25-bar slice leaves `prices` empty and the
        # environment raises IndexError from inside `reset()`. 30 bars is the
        # smallest slice that survives. Refusing here turns an opaque crash
        # into a stated reason.
        lookback_guard = int(getattr(self.env_config or EnvConfig(), "lookback", 20))
        floor = max(lookback_guard + 10, 30)
        if len(history) < floor:
            raise RuntimeError(
                f"fine-tune left '{expert}' unchanged: {len(history)} bars is "
                f"below the {floor}-bar floor the environment needs to produce "
                f"a single transition")

        agent = self.experts.get(expert) or self._clone(self.base_policy())

        before = [p.detach().clone().numpy() for p in agent.online.parameters()]

        # Only the bars of this regime, in chronological order.
        slice_df = self.df.iloc[sorted(history)]
        env = TradingEnv(slice_df, self.env_config or EnvConfig())

        # A regime slice is short — 35 bars is common early in a series. With a
        # fixed episode count the replay buffer never reaches `min_buffer`
        # (1000), `learn_step` returns None every time, and the "fine-tuned"
        # expert comes out bit-identical to the base. Measured: 35 bars x 8
        # episodes = 280 transitions, no weight moved at all.
        #
        # So the episode count is derived from the slice length instead of
        # fixed: enough passes to fill the buffer, capped so a long regime does
        # not turn one switch into a full retraining run.
        min_buffer = int(getattr(agent.cfg, "min_buffer", 1000))
        # An episode yields (bars - lookback - 1) transitions, not (bars - 1):
        # TradingEnv consumes `lookback` bars building the first observation.
        # Ignoring that overestimated the yield by ~2.4x on a 35-bar slice
        # (34 assumed vs 14 real), so the buffer still fell short and no weight
        # moved. Measured against the environment, not assumed.
        lookback = int(getattr(self.env_config or EnvConfig(), "lookback", 20))
        per_episode = max(1, len(slice_df) - lookback - 1)
        needed = int(np.ceil((min_buffer + agent.cfg.batch_size) / per_episode))
        episodes = int(min(max(self.episodes, needed), self.max_episodes))

        agent.train(env, episodes=episodes)

        after = [p.detach().numpy() for p in agent.online.parameters()]
        delta = max(float(np.abs(a - b).max()) for a, b in zip(before, after, strict=True))

        self.experts[expert] = agent
        if delta <= 1e-9:
            # The call ran but no gradient was applied. Saying "adapted" here
            # would be the exact false claim this audit set out to prevent.
            raise RuntimeError(
                f"fine-tune left '{expert}' unchanged: {len(slice_df)} bars, "
                f"{episodes} episodes, buffer {len(agent.buffer)} < "
                f"{min_buffer} required by learn_step")
        record = {
            "expert": expert,
            "bars_used": len(history),
            "last_bar_used": int(max(history)),
            "episodes": episodes,
            "buffer_after": len(agent.buffer),
            "weight_delta": round(delta, 8),
            "weights_changed": bool(delta > 1e-9),
        }
        self.adaptations.append(record)
        return record

    # ------------------------------------------------------------ verification
    def verify_adaptation(self) -> dict:
        """Did fine-tuning actually change any policy? Measured, not assumed."""
        if not self.adaptations:
            return {"adaptations": 0, "any_weights_changed": False,
                    "reason": "no expert was fine-tuned"}
        changed = [a for a in self.adaptations if a["weights_changed"]]
        return {
            "adaptations": len(self.adaptations),
            "experts_adapted": sorted({a["expert"] for a in self.adaptations}),
            "any_weights_changed": bool(changed),
            "n_weights_changed": len(changed),
            "max_weight_delta": max(a["weight_delta"] for a in self.adaptations),
            "basis": ("largest absolute change in any online-network parameter "
                      "between before and after fine-tuning"),
        }


# ============================================================================
# Integration with the live application
#
# Everything above is a self-contained mechanism: it routes regimes, fine-tunes
# real policies and times the reaction. Until now nothing in the application
# called it — `moe.py` had zero importers outside its own tests, so the running
# platform never executed a single line of it.
#
# This section closes that gap with one function. `rollout` replays a symbol
# bar by bar through the platform's **own** `TradingEnv`, asking the expert on
# duty for each action, and returns a payload shaped exactly like the existing
# backtest so the caller can hand it to the same UI.
#
# What it reuses unchanged (nothing below is re-implemented):
#   * `TradingEnv`             the environment, its reward, its frictions
#   * `rl_service.load_agent`  the checkpoint loader
#   * `rl_service._env_config_for_agent`  the saved per-agent env config
#   * `rl_service._baselines`  Buy & Hold / SMA / cash references
#   * `DQNAgent.q_values`      the same greedy rule `evaluate()` uses
#   * `RegimeFeatureProvider`  the causal regime labels the agent already sees
#
# Why the loop is written here rather than calling `agent.evaluate(env)`:
# `evaluate` holds one fixed policy for the whole episode by construction. The
# whole point of a MoE is that the acting policy *changes mid-episode*. There
# is no way to express that through `evaluate` without modifying it, and
# modifying it would put MoE code on the baseline's execution path — exactly
# what this integration is required not to do. The loop below is a faithful
# copy of `evaluate`'s decision rule (`argmax` over `q_values`), so a run with
# routing disabled reproduces the baseline bit for bit. A test pins that.
# ============================================================================

# Bars of an expert's own regime used for one fine-tune. Without a cap, a late
# switch on a 5-year window would train on ~900 bars and turn a "fine-tune"
# into a retraining run; measured, that is ~17 s for a single switch. One
# trading year of the expert's own regime keeps the adaptation local and recent
# while still clearing `MIN_EXPERT_BARS` comfortably.
ADAPT_WINDOW = 252


def _greedy(agent, obs) -> int:
    """The decision rule `DQNAgent.evaluate(deterministic=True)` uses.

    Kept in one place so the MoE loop and the baseline cannot drift apart. Any
    agent exposing `q_values` works; SB3 agents do not, and `rollout` refuses
    them up front rather than failing halfway through a run.
    """
    return int(np.argmax(agent.q_values(obs)))


def rollout(symbol: str, algo: str = "dueling_dqn", period: str = "1y", *,
            env_overrides: dict | None = None,
            adapt: bool = True,
            variant: str = "",
            adapt_window: int = ADAPT_WINDOW,
            min_expert_bars: int = MIN_EXPERT_BARS) -> dict:
    """Replay `symbol` through TradingEnv with the regime-routed experts.

    Returns the same keys the existing backtest returns — `performance`,
    `baselines`, `equity_curve`, `trades` — plus a `moe` block carrying the
    routing trace, the adaptation records and K-5. A caller that ignores the
    extra block gets a payload it already knows how to render.

    `adapt=False` keeps the routing and the K-5 accounting but performs no
    gradient update: every expert stays a copy of the base checkpoint. That is
    the honest control condition — it isolates what routing alone does, and a
    test uses it to prove the fine-tuning is what moves the weights.
    """
    from app.core.exceptions import InvalidRequestError
    from app.services.data.market_data import market_data_service
    from app.services.rl.environment import TradingEnv
    from app.services.rl.service import NATIVE_DISCRETE, rl_service

    algo = (algo or "").lower().strip()
    if algo not in NATIVE_DISCRETE:
        # SB3 policies expose `predict`, not `q_values`, and their checkpoints
        # cannot be deep-copied and fine-tuned through `DQNAgent.train`. Saying
        # so is better than routing to an expert that cannot adapt and then
        # reporting adaptation figures that describe nothing.
        raise InvalidRequestError(
            f"MoE supports the native discrete agents ({', '.join(sorted(NATIVE_DISCRETE))}); "
            f"'{algo}' is a stable-baselines3 policy with no fine-tune path here")

    series = market_data_service.get_history(symbol, period=period)
    df = series.df
    env_cfg = rl_service._env_config_for_agent(symbol, algo, env_overrides, variant)

    # The environment may drop leading rows while building indicators. Every
    # index below refers to `env.raw`, never to `df`, so routing and stepping
    # cannot silently disagree about which bar is bar t.
    env = TradingEnv(df, env_cfg)
    bars = env.raw

    base_agent = rl_service.load_agent(symbol, algo, env, variant)

    provider = RegimeFeatureProvider(step=env_cfg.regime_step,
                                     window=env_cfg.regime_window).build(bars)
    regimes = [provider.at(t).regime for t in range(len(bars))]
    assignments = [route(r) for r in regimes]

    factory = PolicyExpertFactory(symbol, bars, algo=algo, env_config=env_cfg)
    # Reuse the already-loaded checkpoint instead of reading it from disk a
    # second time: same object the baseline starts from, so the two runs are
    # guaranteed to begin identically.
    factory._base = base_agent

    obs, _ = env.reset()
    acting, acting_expert = base_agent, BASE_EXPERT
    switches: list[RegimeSwitch] = []
    pending: dict | None = None
    used: dict[str, int] = {}
    done = False

    def _close(entry: dict, reaction: int | None, adapted: bool, reason: str) -> None:
        switches.append(RegimeSwitch(
            bar=entry["bar"], date=entry["date"],
            from_regime=entry["from_regime"], to_regime=entry["to_regime"],
            from_expert=entry["from_expert"], to_expert=entry["to_expert"],
            expert_changed=entry["expert_changed"],
            reaction_bars=reaction, adapted=adapted, reason=reason))

    def _blocked(expert: str, t: int) -> str:
        """Why a pending switch has not been served yet, in plain terms."""
        seen = sum(1 for i in range(t) if assignments[i] == expert)
        if seen < min_expert_bars:
            return (f"'{expert}' had {seen} prior bars of its own regime, "
                    f"below the {min_expert_bars} needed to fine-tune")
        return f"'{expert}' was eligible but had not yet taken control"

    while not done:
        t = env.t
        wanted = assignments[t]

        if t > 0 and regimes[t] != regimes[t - 1]:
            if pending is not None:
                # A second change arrived before the first was served. The
                # earlier one never got its expert in control: record it as
                # unadapted, and say *why* it was still waiting. Reporting only
                # "superseded" hid the real cause — almost always too few bars
                # of that regime to fine-tune on yet.
                _close(pending, None, False,
                       f"superseded at bar {t} before the expert took control "
                       f"({_blocked(pending['to_expert'], t)})")
            date = bars.index[t]
            pending = {
                "bar": t,
                "date": str(date.date()) if hasattr(date, "date") else None,
                "from_regime": regimes[t - 1], "to_regime": regimes[t],
                # The expert that was *actually acting*, not the one that owned
                # the previous regime. Those differ whenever a switch went
                # unadapted, and using the latter produced traces reading
                # "bull->bull, expert_changed=True", which is self-contradictory.
                "from_expert": acting_expert, "to_expert": wanted,
                "expert_changed": wanted != acting_expert,
            }
            if wanted == acting_expert:
                # The expert on duty already owns the new regime: it is in
                # control at the very bar of the change, so the delay is zero
                # by construction rather than by measurement.
                _close(pending, 0, False, "expert already in control")
                pending = None

        if pending is not None and pending["to_expert"] == wanted:
            history = [i for i in range(t) if assignments[i] == wanted]
            if len(history) < min_expert_bars:
                # Not enough of this regime has been seen yet. Keep the switch
                # open: the bars keep accruing and it may become adaptable
                # later in the same run.
                pass
            else:
                window = history[-adapt_window:]
                reason, ok = "", False
                if adapt:
                    try:
                        factory(wanted, window)
                        ok = True
                    except Exception as exc:
                        reason = f"fine-tune refused: {str(exc)[:120]}"
                        logger.info("MoE fine-tune refused for %s: %s", wanted, exc)
                        # Do not retry every subsequent bar of the same regime.
                        _close(pending, None, False, reason)
                        pending = None
                else:
                    # Control condition: take the expert, apply no gradient.
                    factory.experts.setdefault(wanted, factory._clone(factory.base_policy()))
                    ok, reason = True, "routed without fine-tuning (adapt=False)"
                if ok:
                    acting, acting_expert = factory.experts[wanted], wanted
                    _close(pending, t - pending["bar"], adapt, reason)
                    pending = None

        # Count the policy that is *actually queried*, identified by object,
        # not by the label bookkeeping happens to hold. A mutation that
        # advanced the label without swapping the policy left this counter
        # claiming an expert drove 241 bars while the base agent made every
        # decision — the precise false claim this integration must not permit.
        acted_by = acting_expert if acting is not base_agent else BASE_EXPERT
        used[acted_by] = used.get(acted_by, 0) + 1
        obs, _, terminated, truncated, _ = env.step(_greedy(acting, obs))
        done = terminated or truncated

    if pending is not None:
        _close(pending, None, False,
               "regime changed too close to the end of the window for the "
               "expert to take control")

    perf = env.performance()
    equity = env.equity_curve[1:]          # drop the pre-trade opening balance
    dates = [str(d.date()) for d in bars.index[env_cfg.lookback:
                                               env_cfg.lookback + len(equity)]]

    counts = {e: int(assignments.count(e)) for e in EXPERTS}
    notes: list[str] = []
    warmup = int(assignments.count(BASE_EXPERT))
    if warmup:
        notes.append(f"{warmup} warm-up bars routed to the base policy "
                     f"(regime not yet classifiable)")
    starved = [e for e, c in counts.items() if 0 < c < min_expert_bars]
    if starved:
        notes.append(f"experts with too few bars to specialise: "
                     f"{', '.join(starved)} (threshold {min_expert_bars})")
    if not adapt:
        notes.append("adapt=False: experts were routed but never fine-tuned, "
                     "so this run isolates routing from adaptation")

    verdict = factory.verify_adaptation()
    return {
        "symbol": symbol.upper(), "algo": algo, "period": period,
        "variant": variant or None,
        "mode": "moe",
        "performance": perf,
        "baselines": rl_service._baselines(bars, env_cfg),
        "equity_curve": [{"date": d, "value": round(float(v), 2)}
                         for d, v in zip(dates, equity, strict=False)],
        "trades": env.trades[:200],
        "n_actions": len(equity),
        "moe": {
            "bars": len(bars),
            "experts": counts,
            "bars_acted_by": used,
            "n_switches": len(switches),
            "switches": [s.to_dict() for s in switches],
            "adaptations": factory.adaptations,
            "adaptation_check": verdict,
            "k5_reaction_delay": RegimeMoE.k5(switches),
            # The headline K-5 above counts a switch that needed no new expert
            # as a zero-bar reaction, which is true but flattering: on this
            # data most zeros are "bull was already acting". The stricter
            # reading — only switches that actually required a different
            # expert — is reported alongside it so neither can be quoted
            # without the other.
            "k5_expert_changes_only": RegimeMoE.k5(
                [s for s in switches if s.expert_changed]),
            "adapt": bool(adapt),
            "adapt_window": int(adapt_window),
            "min_expert_bars": int(min_expert_bars),
            "notes": notes,
            "disclaimer": (
                "The MoE run and the baseline start from the same checkpoint "
                "and use the same environment, so the difference between them "
                "is attributable to regime routing and fine-tuning. This is a "
                "single window on one instrument: it measures adaptation "
                "latency, not an edge."
            ),
        },
    }
