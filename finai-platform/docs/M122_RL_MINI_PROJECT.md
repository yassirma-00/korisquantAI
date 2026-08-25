# M122 — Reinforcement Learning Mini-Project Report

**Regime-Aware Observation Augmentation for Value-Based Deep RL in Single-Asset Trading: A Multi-Seed Reproduction and Negative Result**

El Maroufy Mohamed Yassir — Master MD3SI, Module M122

Supervised by Hamza Allaga

---

## Abstract

Deep reinforcement learning is increasingly applied to single-asset trading,
where an agent observes technical indicators and emits a buy/hold/sell action.
We reproduced a Dueling Double DQN baseline on a trading MDP with transaction
costs and slippage over two years of AAPL daily bars using five independent
seeds — a protocol the host platform could not previously support, as all
fourteen of its shipped agents share the seed 42. We then tested a modification
augmenting the observation with six market-regime features, hypothesising that
explicit regime context would stabilise the policy. The baseline returned
**+8.66 % ± 3.82 %** (mean ± 95 % CI, n = 5, sd = 3.08 %) against
**+5.21 % ± 8.83 %** (sd = 7.11 %) for the regime-aware arm; the −3.45-point
difference is not significant (Welch t = −0.995, df ≈ 5.4, p = 0.362,
d = −0.63) and variance more than doubled (2.31×). Both arms lost decisively to
a Buy & Hold benchmark returning **+21.17 %**, giving alpha of −12.51 and
−15.96 points, a pattern confirmed by a PPO ablation (−15.53 points). Regime
augmentation adds variance without measurable benefit, and no configuration
justifies deployment over inaction.

**Keywords** — deep reinforcement learning, Dueling DQN, market-regime
detection, multi-seed evaluation, negative result

---

## Table of Contents

| Section | Page |
|---|---|
| Abstract | 1 |
| Table of Contents | 1 |
| I. Introduction | 2 |
| II. Background and Related Work | 2 |
| III. Experimental Protocol | 3 |
| IV. Baseline Reproduction | 4 |
| V. Proposed Enhancement | 5 |
| VI. Results | 5 |
| VII. Ablation Study | 6 |
| VII-bis. Active Adaptation: Regime-Aware Mixture-of-Experts | 6 |
| VIII. Discussion | 7 |
| IX. Conclusion and Future Work | 8 |
| References | 8 |
| Appendix | 9 |

---

# I. Introduction

Single-asset trading is a natural candidate for reinforcement learning: the
problem is sequential, the reward is unambiguous (change in wealth), and the
action space is small. It is also unusually hostile. The signal-to-noise ratio
of daily equity returns is close to zero, the data-generating process is
non-stationary, and a policy that looks profitable on one random seed can be
unprofitable on the next.

That last property is the motivation for this mini-project. The host platform,
KorisQuant AI, ships fourteen trained agents across six algorithms — and every
one of them was trained with the random seed fixed at 42. Its own diagnostics
refuse to report dispersion for exactly this reason:

> "1 distinct seed across 14 runs. Mean ± standard deviation needs at least 3
> independent seeds; below that the spread measures nothing."

Any claim about whether a modification helps is therefore unfalsifiable on the
existing artefacts. This report (i) reproduces a baseline under a proper
multi-seed protocol, (ii) tests one specific modification — regime-aware
observation augmentation — and (iii) reports the outcome, which is negative.

**Contributions.**
1. A seed-varying protocol implemented without modifying any training logic.
2. A five-seed baseline reproduction with confidence intervals.
3. A controlled A/B evaluation of regime augmentation, with an effect size and
   a significance test.
4. An ablation across algorithm family, and an explicit comparison against a
   passive benchmark that both arms lose to.

---

# II. Background and Related Work

## A. MDP Formulation

The environment (`backend/app/services/rl/environment.py`) is a finite-horizon
MDP ⟨S, A, P, R, γ⟩ over daily bars:

- **State** `s_t ∈ S` — a window of normalised technical indicators concatenated
  with the agent's own position and cash fraction. Baseline observation
  dimension is **36**.
- **Action** `a_t ∈ A = {0: SELL, 1: HOLD, 2: BUY}` — discrete, long-only. No
  shorting and no leverage.
- **Transition** `P` — deterministic replay of the realised price path; the
  agent's orders affect only its own book, not the market.
- **Reward** `R` — change in portfolio value, net of a transaction fee and
  slippage charged on every executed order, minus a risk penalty term derived
  from drawdown and CVaR.
- **Discount** `γ = 0.99`.

Frictions matter here: a frictionless environment rewards high-turnover policies
that are unprofitable once realistic costs are applied.

## B. Algorithm Overview

The baseline is **Dueling Double DQN**, combining two corrections to vanilla DQN:

- **Double Q-learning** (van Hasselt et al., 2016) decouples action *selection*
  from action *evaluation*, correcting DQN's systematic overestimation bias.
- **Dueling architecture** (Wang et al., 2016) splits the network into a state
  value stream `V(s)` and an advantage stream `A(s,a)`, recombined as
  `Q = V + (A − mean A)`. This helps where many actions have similar value — the
  common case in trading, where HOLD and a marginal trade often differ little.

The ablation additionally uses **PPO** (Schulman et al., 2017), an on-policy
policy-gradient method that clips the update ratio so a single step cannot move
the policy too far.

## C. Prior Results on This Environment

The platform's own leaderboard, computed over its fourteen single-seed runs,
reports a mean fleet health of **53.8/100**, with 3 runs flagged *unstable*,
1 *plateaued*, 6 *still improving* and 4 with *insufficient episodes*. Notably,
the two worst performers by return are portfolio-level SAC agents at **−8.63 %**
and **−5.28 %**. Single-seed Dueling DQN on AAPL is recorded at +3.08 % with a
health score of 23.4 — the lowest of the fleet.

These figures cannot be compared against ours directly: they use different
episode budgets and no seed replication. They serve only to establish that the
environment is hard and that poor outcomes are common on it.

## D. Related Work on the Proposed Modification

Regime-switching models have a long history in finance (Hamilton, 1989), and
regime-conditioned RL has been proposed on the intuition that a single policy
cannot be simultaneously optimal in a bull market and a crash. The platform
implements an online classifier (`regime_features.py`) that emits six
descriptors per bar and, for baskets, `3n+4` for n assets.

Two implementation facts are relevant to reproducibility. First, the classifier
costs **9.8 ms per call**; naive per-step invocation would take 7.4 s for 752
bars, reduced to **1.19 s** by pre-computing per bar. Second, the feature block
is **opt-in** (`EnvConfig.regime_aware`, default `False`) because enabling it
changes the observation from 36 to 42 dimensions, which breaks every previously
trained agent with a shape error.

---

# III. Experimental Protocol

## A. Environment Specification

| Property | Value |
|---|---|
| Instrument | AAPL, daily bars |
| Data period | 2y (≈ 504 bars) |
| Train/test split | 80 % / 20 %, chronological (no shuffling) |
| Action space | Discrete(3) — SELL / HOLD / BUY, long-only |
| Observation dim | 36 (baseline) → 42 (regime-aware) |
| Initial balance | 100 000 |
| Transaction cost | applied per order (platform default) |
| Slippage | applied per order (platform default) |
| Reward | Δ wealth − risk penalty (drawdown, CVaR) |
| Benchmark | Buy & Hold on the same held-out window |

The split is chronological. A random split would leak future information into
training — a fault previously found and fixed in this codebase, where
`df.iloc[split-60:]` allowed the training window to overlap the test window.

## B. Implementation

All runs use the platform's own entry point, `rl_service.train_single_asset()`.
**No training logic was modified for this study.** Two documented extension
points carry the experimental manipulation:

- **Seed** — varied through a hyperparameter profile (`training.seed`), one
  profile per seed, created with `save_profile(..., merge=True)` so no other
  field in the profile is disturbed.
- **Regime flag** — passed through `env_overrides={"regime_aware": bool}`.

The driver is `scripts/multiseed_study.py`.

## C. Training Budget

8 episodes per run, on the 2y window. This is a deliberately small budget: the
study asks whether the modification helps *at equal cost*, not whether either
arm converges. Measured wall-clock time per run:

| Arm | Mean wall time | 95 % CI |
|---|---|---|
| Baseline (Dueling DQN) | 6.89 s | ± 0.76 s |
| Regime-aware (Dueling DQN) | 6.91 s | ± 0.08 s |
| Ablation (PPO) | 35.2 s | ± 0.7 s |

The modification is therefore **essentially free at run time** (+0.3 %), because
regime features are pre-computed per bar rather than per step.

## D. Seeds

**Five independent seeds: 1, 2, 3, 4, 5.** Each seed is used for both arms, so
the comparison is paired by seed on identical data.

Seed independence was verified rather than assumed: each run records the seed
actually resolved into its configuration and a SHA-256 configuration
fingerprint. All five fingerprints are distinct, and every `recorded_seed`
matches the requested one (Appendix B).

Five seeds is the minimum for a usable interval, not a comfortable sample. All
intervals below use the small-sample *t* critical value (t₀.₉₇₅,₄ = 2.776), not
the normal approximation.

## E. Metrics

| Metric | Definition |
|---|---|
| **Total return** | Held-out portfolio return, net of costs. Primary metric. |
| **Sharpe ratio** | Annualised mean excess return / annualised volatility |
| **Max drawdown** | Largest peak-to-trough decline on the held-out equity curve |
| **Alpha vs Buy & Hold** | Agent return − passive benchmark return over the same window |
| **Wall time** | Seconds per training run, measured with `perf_counter` |

Statistics reported as mean ± half-width of the 95 % *t*-interval. Comparison
between arms uses **Welch's t-test** (unequal variances) and **Cohen's d**
(pooled sd) for effect size.

## F. Hyperparameters

Resolved from `configs/`: `defaults.yaml` → `algorithms/dueling_dqn.yaml` →
`profiles/seed<N>.yaml`. Full YAML in Appendix A. Key values: learning rate
5 × 10⁻⁴, γ = 0.99, batch size 64, hidden (128, 128), replay buffer 50 000,
min buffer 1 000, target update 250, ε from 1.0 → 0.05.

Only `training.seed` and `training.episodes` differ between profiles; the
regime flag is an environment override, not a hyperparameter. Every other value
is identical across all ten runs.

---

# IV. Baseline Reproduction

Five seeds, Dueling Double DQN, stock 36-dimensional observation.

| Seed | Total return | Sharpe |
|---|---|---|
| 1 | +5.84 % | 0.796 |
| 2 | +7.34 % | 0.998 |
| 3 | +6.27 % | 3.472 |
| 4 | +11.17 % | 3.006 |
| 5 | +12.68 % | 1.750 |

**Baseline: +8.66 % ± 3.82 % (n = 5, sd = 3.08 %), Sharpe 2.004 ± 1.481.**

Two observations. First, the spread across seeds (5.84 % to 12.68 %) is more
than **twice** the width one would infer from any single run — which is exactly
why the platform's single-seed artefacts cannot support claims. Second, Sharpe
varies from 0.796 to 3.472 on identical data and identical hyperparameters; the
Sharpe interval (± 1.481) is so wide as to be nearly uninformative at n = 5.

---

# V. Proposed Enhancement

**Hypothesis.** A policy that cannot observe the prevailing market regime must
infer it implicitly from raw indicators. Supplying it explicitly should reduce
the burden on the network and stabilise behaviour across differing market
conditions — in particular, it should discourage holding through a crash.

**Implementation.** Setting `regime_aware=True` appends six features per bar
from the online classifier, expanding the observation from 36 to 42 dimensions.
The features describe trend direction and strength, volatility state, and
drawdown context. Everything else — architecture, optimiser, replay, reward,
data, split — is unchanged.

**Supporting prior measurement.** Under a regime-aware reward, holding a
position through a crash is penalised by **2 885 reward points** more than under
the standard reward, confirming the mechanism does change the incentive
landscape. Whether that translates into better held-out returns is the question
this study answers.

---

# VI. Results

## A. Primary comparison

| Arm | n | Mean return | 95 % CI | sd | Min | Max |
|---|---|---|---|---|---|---|
| Baseline | 5 | **+8.66 %** | [+4.84 %, +12.48 %] | 3.08 % | +5.84 % | +12.68 % |
| Regime-aware | 5 | **+5.21 %** | [−3.62 %, +14.04 %] | 7.11 % | −4.51 % | +15.33 % |

**Difference: −3.45 points** (regime − baseline).

| Test | Value | Interpretation |
|---|---|---|
| Welch t | −0.995 (df ≈ 5.4) | — |
| **p-value** | **0.362** | **not significant at α = 0.05** |
| Cohen d | −0.63 | medium effect, but CI spans zero |
| sd ratio | **2.31×** | variance more than doubled |

The regime-aware confidence interval **straddles zero** ([−3.62 %, +14.04 %]):
on this evidence the modification cannot even be said to be profitable, let
alone better than the baseline.

## B. Secondary metrics

| Metric | Baseline | Regime-aware |
|---|---|---|
| Sharpe ratio | 2.004 ± 1.481 | 1.128 ± 1.788 |
| Max drawdown | −6.93 % ± 5.86 % | −7.56 % ± 3.57 % |
| Alpha vs Buy & Hold | **−12.51 % ± 3.82 %** | **−15.96 % ± 8.83 %** |
| Wall time | 6.89 s ± 0.76 s | 6.91 s ± 0.08 s |

The one arguable benefit is drawdown *dispersion*: the regime arm's drawdown
interval is narrower (± 3.57 vs ± 5.86). Its mean drawdown is nonetheless
slightly worse, so this is a reduction in variability, not in risk.

## C. The result that matters most

**Buy & Hold on the same held-out window returned +21.17 %.**

Both arms lose to it decisively — by 12.51 points (baseline) and 15.96 points
(regime-aware). Neither confidence interval comes close to containing zero
alpha. Whatever the two arms are learning, at this budget it is worse than
doing nothing on a strongly trending asset.

Reporting only the RL-vs-RL comparison would have made the baseline look like a
success. It is not.

---

# VII. Ablation Study

**Question.** Is the negative result specific to the value-based family, or
does the environment itself resist improvement at this budget?

**Design.** Replace Dueling DQN with **PPO** — a different family (on-policy
policy gradient), same 5 seeds, same data, same split, same episode budget.

| Configuration | n | Mean return | 95 % CI | sd | Alpha vs B&H |
|---|---|---|---|---|---|
| Dueling DQN, baseline | 5 | +8.66 % | ± 3.82 % | 3.08 % | −12.51 % |
| Dueling DQN, regime-aware | 5 | +5.21 % | ± 8.83 % | 7.11 % | −15.96 % |
| **PPO, baseline** | 5 | **+5.64 %** | **± 3.72 %** | 2.99 % | **−15.53 %** |

**Findings.**

1. **The negative alpha is not algorithm-specific.** PPO also loses heavily to
   Buy & Hold (−15.53 points). Three of three configurations underperform the
   passive benchmark, with non-overlapping intervals around zero alpha.
2. **Seed variance is a property of the value-based arm with regime features,
   not of the environment.** Baseline Dueling DQN (sd 3.08 %) and PPO (sd
   2.99 %) have nearly identical dispersion; only the regime-aware arm inflates
   it (sd 7.11 %). This isolates the variance increase to the modification.
3. **Cost scales with family, not with the modification.** PPO takes 35.2 s per
   run versus 6.9 s for Dueling DQN — a 5.1× cost for no return advantage.

**Ablation conclusion.** The failure to beat Buy & Hold is a property of the
task and budget. The variance inflation is a property of the proposed
modification.

---

# VII-bis. Active Adaptation: A Regime-Aware Mixture-of-Experts

> Added after the five-seed study. Reported on **a single seed and a single
> instrument**, so it sits outside the module protocol above (divergence D-28).
> Every figure below was re-measured for this revision.

## Why a second enhancement

The modification evaluated in sections V–VII is **passive**: six regime features
enter the observation, but the policy weights never change when the market does.
That cannot react to a regime switch, only describe one, and it left KPI **K-5**
(reaction delay after a detected regime change) unmeasurable.

## Mechanism

Three experts — `bull`, `bear`, `stress` — each a deep copy of the same trained
checkpoint. A **lookup table**, not a learned gate, maps the seven regimes the
existing detector already emits onto those three buckets, so the routing is
auditable by inspection. On a switch, the incoming expert is fine-tuned on the
bars of **its own** regime seen **strictly before** the switch; the bars until it
takes control are recorded as K-5.

Two leakage guards, both testable: regime labels come from the existing causal
provider (`df.iloc[start:t+1]`, exclusive bound), and the fine-tune window uses
indices `< t`. A mutation widening the bound to `t+1` makes the test fail.

## Results (AAPL, `dueling_dqn`, single seed)

| | 1 year | 2 years |
|---|---:|---:|
| Bars | 251 | 501 |
| Regime switches detected | 8 | 19 |
| of which change expert | 8 | 11 |
| Real fine-tunes (weights moved) | 1 | 1 |
| largest parameter change | 0.0413 | 0.0468 |
| Bars driven by an expert | 11 / 230 | 241 / 480 |
| K-5, all switches (bars) | 60.0 | 0.0 |
| K-5, expert changes only (bars) | 60.0 | 0.0 |
| measured on | 1 of 8 | 1 of 11 |
| Switches with no reaction | 7 | 10 |
| **MoE total return** | **+4.19 %** | **+29.92 %** |
| Buy & Hold | +29.73 % | +38.65 % |

## Reading these numbers honestly

**The mechanism runs.** Fine-tuning is a genuine gradient step, verified on the
weights themselves (0.0413 and 0.0468). The factory *raises* rather than
reporting an adaptation when the replay buffer never reaches `min_buffer` and
the network comes back bit-identical.

**It fires rarely.** Only **1 of 8** (1y) and **1 of 11** (2y) expert changes
were served. The rest are counted as failures to react, not dropped: the
incoming expert had fewer than the 90 bars its regime needs. A mean over one
observation is not an average, and the interface says so.

**It does not beat the market.** The MoE trails Buy & Hold on both windows. One
instrument, one seed, two windows — no edge is claimed. The deliverable is a
measurable adaptation *latency*, not performance.

## Scope limits

Fine-tuning reuses the native DQN loop, so the MoE drives only the six native
discrete agents; SB3 policies expose `predict` rather than `q_values` and are
refused with **HTTP 422** instead of being routed to an expert that cannot adapt
(**D-29**). **MAML was not implemented**: the specification asked for drift
detection, meta-learning *or* MoE, and the third branch is the one built
(**D-30**).

# VIII. Discussion

## A. Why the modification did not help

Three explanations are consistent with the evidence, and this study cannot
separate them:

1. **Budget.** 8 episodes is small. Six extra input dimensions enlarge the
   parameter space the network must fit; with a fixed budget, the added capacity
   may simply be undertrained. This is the most plausible reading of a *higher
   variance, similar mean* outcome.
2. **Redundancy.** The regime descriptors are derived from the same price series
   as the technical indicators already in the observation. The network may
   already infer regime implicitly, so the extra features add noise, not
   information.
3. **Regime irrelevance on this sample.** The held-out window is a strong AAPL
   uptrend (Buy & Hold +21.17 %). A feature set whose value lies in recognising
   crashes has little to contribute where there is no crash — and the platform's
   own measurement that crash-holding is penalised by 2 885 points is precisely
   a *crash-scenario* result.

Explanation 3 also warns against over-generalising: a single instrument over a
single trending window is a narrow test bed.

## B. Threats to validity

- **n = 5.** Sufficient for an interval, weak for detecting a small effect. With
  sd ≈ 7 % in the regime arm, detecting a 3.45-point difference at 80 % power
  would need roughly 60 seeds per arm. This study is underpowered against small
  effects, and p = 0.362 should be read as *inconclusive*, not as proof of no
  effect.
- **One instrument, one window.** AAPL over 2y. No claim generalises beyond it.
- **One budget.** The comparison is at equal cost; a converged comparison could
  differ.
- **No hyperparameter tuning per arm.** The regime arm inherits hyperparameters
  tuned (loosely) for the 36-dimensional observation. A fair test of the
  modification's ceiling would re-tune for 42 dimensions.
- **Isolation Forest artefact (contextual).** The platform's anomaly detector
  labels 2 % of any window by construction (`contamination=0.02`); regime
  features are a separate module, but the same caution applies to any
  count-based descriptor.

## C. On reporting a negative result

The honest outcome here is that a plausible, cheap, well-motivated modification
did not work at this budget, and that the baseline it was compared against does
not beat buying and holding. Both facts are reported because a study that only
publishes favourable comparisons cannot be used to make a decision.

---

# IX. Conclusion and Future Work

We reproduced a Dueling Double DQN baseline on a realistic single-asset trading
MDP across five independent seeds — a protocol previously impossible on this
platform, whose fourteen shipped agents all share seed 42 — and tested whether
augmenting the observation with six market-regime features improves held-out
performance.

The baseline returned **+8.66 % ± 3.82 %**; the regime-aware arm returned
**+5.21 % ± 8.83 %**. The −3.45-point difference is **not significant**
(p = 0.362) and the modification **more than doubled seed variance** (2.31×) at
no run-time cost. An ablation to PPO (+5.64 % ± 3.72 %) showed the variance
inflation is specific to the modification, while the **negative alpha is not**:
all three configurations lose to a Buy & Hold benchmark returning **+21.17 %**.

**Future work.**
1. **Power.** Scale to ≥ 30 seeds per arm; at the observed variance, 5 seeds
   cannot resolve a 3-point effect.
2. **Budget sweep.** Repeat at 8 / 50 / 200 episodes to separate undertraining
   from redundancy.
3. **Regime-relevant windows.** Evaluate on a period containing a genuine
   drawdown (e.g. Feb–Mar 2020), where the mechanism should actually bind.
4. **Per-arm tuning.** Re-tune for the 42-dimensional observation before
   concluding on the modification's ceiling.
5. **Beat the passive benchmark first.** Until an arm produces positive alpha,
   comparisons between RL variants rank configurations that are all worse than
   inaction.

---

# References

[1] H. van Hasselt, A. Guez, D. Silver, "Deep Reinforcement Learning with Double
Q-learning," *AAAI*, 2016.

[2] Z. Wang et al., "Dueling Network Architectures for Deep Reinforcement
Learning," *ICML*, 2016.

[3] V. Mnih et al., "Human-level control through deep reinforcement learning,"
*Nature*, vol. 518, 2015.

[4] J. Schulman et al., "Proximal Policy Optimization Algorithms," arXiv
1707.06347, 2017.

[5] M. G. Bellemare, W. Dabney, R. Munos, "A Distributional Perspective on
Reinforcement Learning," *ICML*, 2017.

[6] J. D. Hamilton, "A New Approach to the Economic Analysis of Nonstationary
Time Series and the Business Cycle," *Econometrica*, vol. 57, 1989.

[7] P. Henderson et al., "Deep Reinforcement Learning that Matters," *AAAI*,
2018. — on seed variance and reporting standards.

[8] B. L. Welch, "The generalization of Student's problem when several different
population variances are involved," *Biometrika*, vol. 34, 1947.

[9] J. Cohen, *Statistical Power Analysis for the Behavioral Sciences*, 2nd ed.,
1988.

[10] P. Artzner et al., "Coherent Measures of Risk," *Mathematical Finance*,
vol. 9, 1999.

---

# Appendix

## A. Full Hyperparameter Configuration (YAML)

Resolution order: `defaults.yaml` → `algorithms/dueling_dqn.yaml` →
`profiles/seed<N>.yaml`.

```yaml
training:
  seed: <1|2|3|4|5>        # the only value varied across runs
  device: cpu
  episodes: 8
  test_fraction: 0.2
  timesteps_per_episode_floor: 10000
optimizer:
  learning_rate: 0.0005
  gamma: 0.99
  batch_size: 64
  grad_clip: 10.0
network:
  hidden: [128, 128]
  double: true
  dueling: true
replay:
  buffer_size: 50000
  min_buffer: 1000
  train_freq: 1
  target_update: 250
exploration:
  epsilon_start: 1.0
  epsilon_end: 0.05
  epsilon_decay_steps: 8000
environment:
  regime_aware: <false|true>   # the modification under test
  initial_balance: 100000
```

## B. Per-Seed Raw Final Returns

Held-out total return. `fingerprint` is the SHA-256 configuration hash recorded
by the platform; all five are distinct, confirming the seeds resolved
independently rather than silently collapsing to the default.

**Dueling DQN — baseline (36-dim observation)**

| Seed | Recorded seed | Fingerprint | Total return | Sharpe |
|---|---|---|---|---|
| 1 | 1 | `e72535604d9b` | +0.0584 | 0.796 |
| 2 | 2 | `aaaabaf4a22e` | +0.0734 | 0.998 |
| 3 | 3 | `31e2d045087e` | +0.0627 | 3.472 |
| 4 | 4 | `32216de2743c` | +0.1117 | 3.006 |
| 5 | 5 | `77235c2409b6` | +0.1268 | 1.750 |

Mean +0.0866, sd 0.0308, 95 % CI [+0.0484, +0.1248].

**Dueling DQN — regime-aware (42-dim observation)**

| Seed | Total return | Sharpe |
|---|---|---|
| 1 | +0.1533 | 3.102 |
| 2 | +0.0606 | 1.067 |
| 3 | +0.0600 | 0.843 |
| 4 | **−0.0451** | −0.909 |
| 5 | +0.0317 | 1.537 |

Mean +0.0521, sd 0.0711, 95 % CI [−0.0362, +0.1404].

**PPO — baseline (ablation)**

| Seed | Total return | Sharpe |
|---|---|---|
| 1 | +0.0150 | 0.370 |
| 2 | +0.0695 | 2.080 |
| 3 | +0.0965 | 4.572 |
| 4 | +0.0533 | 1.601 |
| 5 | +0.0475 | 2.820 |

Mean +0.0564, sd 0.0299, 95 % CI [+0.0192, +0.0936].

Raw JSON: `data/artifacts/multiseed_AAPL_dueling_dqn.json`,
`data/artifacts/multiseed_ablation_ppo.json`.

## C. Reproduction Command

```bash
# from the repository root
export PYTHONPATH=backend

# Main experiment: both arms, 5 seeds
python3 scripts/multiseed_study.py \
    --symbol AAPL --algo dueling_dqn \
    --episodes 8 --seeds 1,2,3,4,5 --arm both \
    --tag AAPL_dueling_dqn

# Ablation: PPO, baseline arm only
python3 scripts/multiseed_study.py \
    --symbol AAPL --algo ppo \
    --episodes 8 --seeds 1,2,3,4,5 --arm baseline \
    --tag ablation_ppo
```

The script creates one hyperparameter profile per seed and writes JSON to
`data/artifacts/`. It modifies no training code: the seed travels through
`training.seed` in a profile and the regime flag through `env_overrides`.

## D. Compute Budget

| Item | Value |
|---|---|
| Hardware | 2-core CPU container, no GPU |
| Runs | 15 (5 baseline + 5 regime-aware + 5 PPO ablation) |
| Dueling DQN wall time | 6.89 s ± 0.76 s per run |
| Regime-aware wall time | 6.91 s ± 0.08 s per run |
| PPO wall time | 35.2 s ± 0.7 s per run |
| **Total training time** | **≈ 245 s** (≈ 4 min) |

Wall time is measured with `perf_counter` around the training call. Note this is
the first study in this repository to time training at all — the platform's
duration estimates were previously derived from a calibrated per-step cost, not
measured end to end.

## E. Contribution Statement

Single-author project. The author designed the multi-seed protocol, implemented
`scripts/multiseed_study.py`, executed all 15 runs, performed the statistical
analysis (Welch's t-test, Cohen's d, small-sample t-intervals) and wrote the
report.

The trading environment, the 13-algorithm RL catalogue and the regime-feature
module are pre-existing components of the KorisQuant AI platform, also developed
by the author in a separate project. **No training logic was modified for this
study**; the seed and the regime flag were varied exclusively through the
platform's documented configuration and override interfaces.
