"""Catalogue of reinforcement-learning algorithms.

Every entry carries the information a user needs to *choose* an algorithm
rather than guess: what it does, where it shines, where it fails, and what it
costs to train. Availability is resolved at import time so the UI can grey out
anything the current install cannot actually run — we never advertise an
algorithm we cannot execute.

Honesty note on "performance": the figures below are qualitative ratings drawn
from the published literature (Atari/MuJoCo benchmarks), NOT measured trading
returns. Financial performance is instrument-specific and is only ever reported
from this platform's own out-of-sample backtests.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal

ActionSpace = Literal["discrete", "continuous", "both"]
Family = Literal["value_based", "policy_gradient", "actor_critic", "distributional"]

# IMPALA was removed: its only advantage is throughput from hundreds of parallel
# actors, which a single price series cannot provide, and it dragged in the very
# heavy Ray dependency for no practical gain on this platform.


# --------------------------------------------------------------- availability
def _probe() -> dict[str, bool]:
    have: dict[str, bool] = {}
    try:
        import stable_baselines3  # noqa: F401
        have["sb3"] = True
    except Exception:
        have["sb3"] = False
    try:
        import sb3_contrib  # noqa: F401
        have["sb3_contrib"] = True
    except Exception:
        have["sb3_contrib"] = False
    return have


BACKENDS = _probe()


@dataclass
class Algorithm:
    key: str
    name: str
    full_name: str
    family: Family
    action_space: ActionSpace
    year: int
    backend: str                     # native | sb3 | sb3_contrib
    description: str
    characteristics: list[str]
    advantages: list[str]
    limitations: list[str]
    performance: dict                # qualitative ratings, 1-5
    best_for: str
    hyperparameters: dict = field(default_factory=dict)
    paper: str = ""

    @property
    def available(self) -> bool:
        if self.backend == "native":
            return True
        return BACKENDS.get(self.backend, False)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["available"] = self.available
        d["requires"] = {
            "native": None, "sb3": "stable-baselines3", "sb3_contrib": "sb3-contrib",
        }[self.backend]
        return d


R = lambda s, sa, st, si: {  # noqa: E731 - compact rating helper
    "sample_efficiency": s, "stability": sa, "final_performance": st, "training_speed": si,
}


CATALOGUE: tuple[Algorithm, ...] = (
    # ============================================================ value based
    Algorithm(
        key="dqn", name="DQN", full_name="Deep Q-Network",
        family="value_based", action_space="discrete", year=2015, backend="native",
        description=(
            "The foundational deep RL algorithm. Learns an action-value function Q(s,a) with a "
            "neural network, stabilised by an experience replay buffer and a periodically-synced "
            "target network."),
        characteristics=[
            "Off-policy: reuses past experience many times",
            "Experience replay breaks temporal correlation between samples",
            "Target network prevents the bootstrap target from moving during updates",
            "ε-greedy exploration decayed over training",
        ],
        advantages=[
            "Simple, well understood and easy to debug",
            "Sample-efficient thanks to replay",
            "Q-values are directly interpretable as expected discounted reward",
        ],
        limitations=[
            "Systematically overestimates Q-values (max operator bias)",
            "Discrete actions only - cannot express portfolio weights",
            "Sensitive to reward scaling",
        ],
        performance=R(4, 3, 3, 4),
        best_for="A transparent baseline for single-asset BUY/HOLD/SELL decisions.",
        hyperparameters={"gamma": 0.99, "lr": 5e-4, "buffer_size": 50000, "batch_size": 64},
        paper="Mnih et al. 2015, Nature",
    ),
    Algorithm(
        key="double_dqn", name="Double DQN", full_name="Double Deep Q-Network",
        family="value_based", action_space="discrete", year=2016, backend="native",
        description=(
            "Fixes DQN's overestimation bias by decoupling action *selection* from action "
            "*evaluation*: the online network picks the next action, the target network scores it."),
        characteristics=[
            "Two networks with separated roles in the bootstrap target",
            "Identical cost to DQN - the fix is essentially free",
            "Markedly less optimistic value estimates",
        ],
        advantages=[
            "More accurate Q-values than vanilla DQN",
            "Better final policies with no extra compute",
            "Especially valuable in noisy domains such as finance",
        ],
        limitations=["Still discrete-only", "Does not address exploration"],
        performance=R(4, 4, 4, 4),
        best_for="A strict improvement over DQN; use it instead of plain DQN.",
        hyperparameters={"gamma": 0.99, "lr": 5e-4, "double": True},
        paper="van Hasselt et al. 2016, AAAI",
    ),
    Algorithm(
        key="dueling_dqn", name="Dueling DQN", full_name="Dueling Double Deep Q-Network",
        family="value_based", action_space="discrete", year=2016, backend="native",
        description=(
            "Splits the network into a state-value stream V(s) and an advantage stream A(s,a), "
            "recombined as Q = V + (A - mean A). The agent can learn which states are valuable "
            "without having to learn the effect of every action in them."),
        characteristics=[
            "Two-headed architecture sharing a common trunk",
            "Learns state value independently of action choice",
            "Combined here with Double DQN by default",
        ],
        advantages=[
            "Much faster learning when many actions are near-equivalent - exactly the HOLD-heavy "
            "situation of a trading agent",
            "More robust value estimates in noisy markets",
        ],
        limitations=["Discrete-only", "Slightly more parameters"],
        performance=R(4, 4, 4, 4),
        best_for="The recommended discrete default on this platform.",
        hyperparameters={"gamma": 0.99, "lr": 5e-4, "dueling": True, "double": True},
        paper="Wang et al. 2016, ICML",
    ),
    # ========================================================== distributional
    Algorithm(
        key="c51", name="C51", full_name="Categorical DQN (51 atoms)",
        family="distributional", action_space="discrete", year=2017, backend="native",
        description=(
            "Instead of the expected return, C51 learns the full probability *distribution* of "
            "returns over a fixed grid of 51 atoms. Risk is then visible in the model itself, not "
            "just in a scalar average."),
        characteristics=[
            "Models P(Z|s,a) over a fixed support [v_min, v_max]",
            "Trained by minimising cross-entropy after a projected Bellman update",
            "Yields a distribution the UI can plot per action",
        ],
        advantages=[
            "Distributional information is directly useful in finance: two actions with the same "
            "mean can have very different tail risk",
            "Usually more stable than expectation-based DQN",
        ],
        limitations=[
            "Requires choosing v_min/v_max in advance - a bad range silently truncates returns",
            "Slower per step than DQN",
        ],
        performance=R(4, 4, 4, 3),
        best_for="Risk-aware discrete trading where the tail matters as much as the mean.",
        hyperparameters={"n_atoms": 51, "v_min": -10.0, "v_max": 10.0},
        paper="Bellemare et al. 2017, ICML",
    ),
    Algorithm(
        key="qr_dqn", name="QR-DQN", full_name="Quantile Regression DQN",
        family="distributional", action_space="discrete", year=2018, backend="sb3_contrib",
        description=(
            "Learns the return distribution as a set of quantiles rather than a fixed grid, "
            "removing C51's need to pre-specify the value range. Trained with the quantile "
            "Huber loss."),
        characteristics=[
            "Adaptive support - no v_min/v_max to tune",
            "Directly estimates value-at-risk style quantiles",
            "Quantile Huber loss is robust to outliers",
        ],
        advantages=[
            "More flexible than C51",
            "Quantiles map naturally onto VaR/CVaR risk management",
        ],
        limitations=["Discrete-only", "Needs sb3-contrib installed"],
        performance=R(4, 4, 5, 3),
        best_for="Distributional RL when you want risk quantiles without tuning a value range.",
        hyperparameters={"n_quantiles": 170, "gamma": 0.99, "lr": 5e-5},
        paper="Dabney et al. 2018, AAAI",
    ),
    Algorithm(
        key="iqn", name="IQN", full_name="Implicit Quantile Network",
        family="distributional", action_space="discrete", year=2018, backend="native",
        description=(
            "Generalises QR-DQN by learning a continuous mapping from quantile level τ to value, "
            "so the distribution can be sampled at any resolution. Enables explicitly risk-averse "
            "policies by distorting the sampling of τ."),
        characteristics=[
            "Samples quantile levels stochastically during training",
            "Cosine embedding of τ conditions the network",
            "Supports risk-sensitive policies (CVaR objective) at no extra cost",
        ],
        advantages=[
            "The most expressive of the distributional family",
            "Can be made explicitly risk-averse - directly relevant to trading",
        ],
        limitations=["More complex to tune", "Discrete-only", "Slower convergence early on"],
        performance=R(4, 4, 5, 3),
        best_for="Risk-averse policies where downside quantiles drive the decision.",
        hyperparameters={"n_quantile_samples": 32, "embedding_dim": 64, "risk_distortion": "neutral"},
        paper="Dabney et al. 2018, ICML",
    ),
    Algorithm(
        key="rainbow", name="Rainbow", full_name="Rainbow DQN",
        family="distributional", action_space="discrete", year=2018, backend="native",
        description=(
            "Combines six independent DQN improvements - Double, Dueling, prioritised replay, "
            "multi-step returns, distributional (C51) and noisy exploration - into a single agent. "
            "State of the art among discrete value-based methods."),
        characteristics=[
            "Double + Dueling + Prioritised Replay + n-step + C51 + NoisyNets",
            "Prioritised replay focuses learning on surprising transitions",
            "NoisyNets replace ε-greedy with learned parameter noise",
        ],
        advantages=[
            "Best-in-class sample efficiency for discrete actions",
            "Each component addresses a distinct DQN weakness",
        ],
        limitations=[
            "Many interacting hyperparameters - hardest of the family to tune",
            "Slowest per-step of the DQN variants",
            "Gains over Dueling DQN shrink on small, noisy datasets like single-asset trading",
        ],
        performance=R(5, 4, 5, 2),
        best_for="Maximum discrete performance when you can afford the training budget.",
        hyperparameters={"n_atoms": 51, "n_step": 3, "per_alpha": 0.5, "per_beta": 0.4,
                         "noisy": True},
        paper="Hessel et al. 2018, AAAI",
    ),
    # ========================================================= policy gradient
    Algorithm(
        key="ppo", name="PPO", full_name="Proximal Policy Optimization",
        family="policy_gradient", action_space="both", year=2017, backend="sb3",
        description=(
            "Optimises the policy directly while clipping the update ratio so a single step can "
            "never move the policy too far. The most widely-used deep RL algorithm in practice, "
            "chiefly because it is hard to destabilise."),
        characteristics=[
            "On-policy with a clipped surrogate objective",
            "Generalised Advantage Estimation for low-variance advantages",
            "Handles discrete and continuous actions with the same code",
        ],
        advantages=[
            "Very stable and forgiving of hyperparameters",
            "Works for both single-asset actions and continuous portfolio weights",
            "Strong, reliable default",
        ],
        limitations=[
            "On-policy: discards experience after each update, so less sample-efficient than DQN",
            "Needs more environment steps to converge",
        ],
        performance=R(3, 5, 4, 4),
        best_for="The safest all-round choice, and the default for portfolio allocation.",
        hyperparameters={"n_steps": 512, "batch_size": 64, "n_epochs": 10, "clip_range": 0.2},
        paper="Schulman et al. 2017",
    ),
    Algorithm(
        key="a2c", name="A2C", full_name="Advantage Actor-Critic",
        family="actor_critic", action_space="both", year=2016, backend="sb3",
        description=(
            "Synchronous actor-critic: an actor proposes actions, a critic estimates state value, "
            "and the actor is updated along the advantage. Simpler and faster per step than PPO, "
            "but without the trust region."),
        characteristics=[
            "On-policy, synchronous updates",
            "Shared trunk between actor and critic",
            "Fewer moving parts than PPO",
        ],
        advantages=["Fast wall-clock training", "Low memory footprint", "Easy to reason about"],
        limitations=[
            "Noisier and less stable than PPO - no update clipping",
            "More sensitive to learning rate",
        ],
        performance=R(2, 3, 3, 5),
        best_for="Quick experiments and baselines when training time matters more than the last few %.",
        hyperparameters={"n_steps": 512, "gae_lambda": 0.95, "ent_coef": 0.01},
        paper="Mnih et al. 2016, ICML",
    ),
    Algorithm(
        key="trpo", name="TRPO", full_name="Trust Region Policy Optimization",
        family="policy_gradient", action_space="both", year=2015, backend="sb3_contrib",
        description=(
            "Enforces a hard KL-divergence constraint on each policy update, guaranteeing "
            "monotonic improvement in theory. PPO is its cheaper approximation."),
        characteristics=[
            "Hard KL trust region solved with conjugate gradient",
            "Guaranteed monotonic policy improvement under assumptions",
            "Second-order method",
        ],
        advantages=["Extremely stable updates", "Strong theoretical grounding"],
        limitations=[
            "Computationally expensive (conjugate gradient per update)",
            "In practice PPO matches it at a fraction of the cost",
        ],
        performance=R(3, 5, 4, 2),
        best_for="When update stability is paramount and compute is not the constraint.",
        hyperparameters={"target_kl": 0.01, "cg_max_steps": 15},
        paper="Schulman et al. 2015, ICML",
    ),
    # ============================================================== continuous
    Algorithm(
        key="ddpg", name="DDPG", full_name="Deep Deterministic Policy Gradient",
        family="actor_critic", action_space="continuous", year=2016, backend="sb3",
        description=(
            "Off-policy actor-critic for continuous control. A deterministic actor outputs the "
            "action directly and a critic evaluates it; exploration comes from added noise."),
        characteristics=[
            "Deterministic policy with additive exploration noise",
            "Replay buffer plus soft target updates",
            "Continuous actions - natural fit for portfolio weights",
        ],
        advantages=["Sample-efficient", "Directly outputs continuous allocations"],
        limitations=[
            "Notoriously brittle - prone to Q-value overestimation and divergence",
            "Largely superseded by TD3 and SAC",
        ],
        performance=R(4, 2, 3, 4),
        best_for="Historical reference; prefer TD3 or SAC in practice.",
        hyperparameters={"tau": 0.005, "buffer_size": 50000, "action_noise": "normal"},
        paper="Lillicrap et al. 2016, ICLR",
    ),
    Algorithm(
        key="td3", name="TD3", full_name="Twin Delayed DDPG",
        family="actor_critic", action_space="continuous", year=2018, backend="sb3",
        description=(
            "Repairs DDPG with three targeted fixes: twin critics taking the minimum (curbs "
            "overestimation), delayed policy updates, and target policy smoothing."),
        characteristics=[
            "Two critics, pessimistic minimum for the target",
            "Actor updated less frequently than the critics",
            "Noise added to target actions to smooth the value estimate",
        ],
        advantages=[
            "Far more stable than DDPG",
            "Strong performance on continuous allocation tasks",
        ],
        limitations=["Deterministic policy explores less well than SAC", "Continuous-only"],
        performance=R(4, 4, 4, 3),
        best_for="Continuous portfolio allocation when you want a deterministic policy.",
        hyperparameters={"policy_delay": 2, "target_noise": 0.2, "tau": 0.005},
        paper="Fujimoto et al. 2018, ICML",
    ),
    Algorithm(
        key="sac", name="SAC", full_name="Soft Actor-Critic",
        family="actor_critic", action_space="continuous", year=2018, backend="sb3",
        description=(
            "Maximum-entropy RL: the agent maximises reward *and* policy entropy, so it keeps "
            "exploring instead of collapsing onto one strategy. The entropy weight is tuned "
            "automatically."),
        characteristics=[
            "Stochastic policy with automatic temperature tuning",
            "Twin critics as in TD3",
            "Entropy bonus sustains exploration throughout training",
        ],
        advantages=[
            "Best sample efficiency of the continuous methods",
            "Very robust to hyperparameters",
            "Sustained exploration suits non-stationary markets, where a policy that stops "
            "exploring becomes stale as the regime shifts",
        ],
        limitations=["Continuous-only", "Entropy term needs care if rewards are badly scaled"],
        performance=R(5, 4, 5, 3),
        best_for="The recommended choice for multi-asset portfolio allocation.",
        hyperparameters={"tau": 0.005, "ent_coef": "auto", "buffer_size": 50000},
        paper="Haarnoja et al. 2018, ICML",
    ),
)

BY_KEY: dict[str, Algorithm] = {a.key: a for a in CATALOGUE}
DISCRETE_KEYS = {a.key for a in CATALOGUE if a.action_space in ("discrete", "both")}
CONTINUOUS_KEYS = {a.key for a in CATALOGUE if a.action_space in ("continuous", "both")}
AVAILABLE_KEYS = {a.key for a in CATALOGUE if a.available}


def get_algorithm(key: str) -> Algorithm | None:
    return BY_KEY.get(key.lower().strip())


def list_algorithms(action_space: str | None = None, family: str | None = None,
                    available_only: bool = False) -> list[Algorithm]:
    items = list(CATALOGUE)
    if action_space:
        items = [a for a in items if a.action_space in (action_space, "both")]
    if family:
        items = [a for a in items if a.family == family]
    if available_only:
        items = [a for a in items if a.available]
    return items


def comparison_table() -> list[dict]:
    """Compact side-by-side view for the UI comparison grid."""
    return [
        {
            "key": a.key, "name": a.name, "family": a.family,
            "action_space": a.action_space, "year": a.year,
            "available": a.available, "backend": a.backend,
            **a.performance,
            "overall": round(sum(a.performance.values()) / len(a.performance), 2),
            "best_for": a.best_for,
        }
        for a in CATALOGUE
    ]


def recommend_algorithm(action_space: str = "discrete", priority: str = "balanced") -> dict:
    """Suggest an algorithm for a given need, restricted to what is installed."""
    pool = list_algorithms(action_space=action_space, available_only=True)
    if not pool:
        return {"error": "no algorithm available for this action space"}
    key = {
        "sample_efficiency": "sample_efficiency",
        "stability": "stability",
        "performance": "final_performance",
        "speed": "training_speed",
    }.get(priority)
    if key:
        best = max(pool, key=lambda a: a.performance[key])
        reason = f"highest {priority.replace('_', ' ')} among installed {action_space} algorithms"
    else:
        best = max(pool, key=lambda a: sum(a.performance.values()))
        reason = "best overall balance of efficiency, stability and final performance"
    return {"recommended": best.key, "name": best.name, "reason": reason,
            "best_for": best.best_for,
            "alternatives": [a.key for a in sorted(
                pool, key=lambda x: -sum(x.performance.values()))[1:4]]}
