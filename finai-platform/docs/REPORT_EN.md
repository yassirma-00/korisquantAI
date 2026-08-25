# KorisQuant AI — Project Report

**Adaptive, Explainable Deep Reinforcement Learning
for Risk-Sensitive Portfolio Management**

*A web platform for financial analysis and portfolio management built on machine
learning, deep reinforcement learning and explainable AI*

Produced by **El Maroufy Mohamed Yassir** — supervised by **Hamza Allaga**

Final-year project

---

> **Method note.** Every value in this report was **measured by running the
> code** at the time of writing; the commands that produce them are in
> Appendix A. Where a number is a *design choice* rather than a measurement,
> that is stated. Where a quantity could not be measured it is reported as
> **unavailable, with the reason** — never estimated.
>
> Unfavourable results — negative R², failed VaR tests, RL agents beaten by a
> passive strategy — are reported as they stand. A report with one false number
> loses the reader's trust in all the others.

## Key figures (measured)

| Quantity | Value |
|---|---:|
| Automated tests (all green) | **776** |
| Lines of code (`.py`+`.js`+`.html`+`.css`) | **50 644** |
| Source files | **161** |
| API paths / operations | **127 / 135** |
| Web pages | **13** |
| Catalogue instruments | **32** |
| Deep-learning architectures | **5** |
| RL algorithms | **13** |
| Trained RL checkpoints on disk | **45** (21 baselines + 24 regime-aware twins) |
| Technical-indicator functions | **21** |
| Stress scenarios | **7** |
| Static analysis (`ruff`) | **0 violations** |
| Specification compliance | **18/30 (60 %)**, 9 documented divergences |

---

# 1 · Introduction

## 1.1 Context

Quantitative portfolio management combines three activities that usually live in
separate tools: **market analysis** (indicators, regimes, correlations),
**forecasting** (statistical or neural models) and **decision-making**
(allocation, sizing, risk control).

Retail platforms display indicators without explaining where they come from;
professional platforms are closed and expensive. Either way the model is a black
box: the user sees a "BUY" signal without knowing which data produced it, over
what window, or with what reliability. Regulation moves the other way — **SR
11-7** (US Federal Reserve) and the **EBA/ACPR** model-governance guidance
require traceability of model, version, data and out-of-sample performance.

## 1.2 Research question

> Can a financial decision-support platform be built in which every displayed
> number is traceable to its computation — including, and especially, when the
> result is unfavourable?

The difficulty is not training a model. It is building a system that **refuses
to lie**: one that shows "unavailable" rather than a misleading zero, that
reports when an agent loses to a passive strategy, and that documents that no
VaR estimator passes the independence test.

## 1.3 Objectives

| # | Objective | Section |
|---|---|---|
| O1 | Realistic MDP environment (costs, slippage, constraints) | §4.3 |
| O2 | Adaptive, regime-aware DRL agent | §4.3, §8.2.4 |
| O3 | Risk-sensitive reward (Sharpe, Sortino, CVaR, drawdown) | §4.3 |
| O4 | Native **and** post-hoc explainable AI | §4.5 |
| O5 | Coherent professional dashboard | §5 |
| O6 | Rigorous validation (walk-forward, stress tests) | §8 |
| O7 | Model governance (versioning, audit log, reproducibility) | §4.6 |

## 1.4 What the platform deliberately does not do

- **No real brokerage.** The portfolio is paper; no order reaches a market.
- **No investment advice.** Outputs are statistical estimates.
- **No multi-factor authentication** (§11.1) — blocking for public deployment.
- **No self-service password reset**: removed after an account-takeover flaw was
  found (§10).

---

# 2 · Background

## 2.1 Financial time-series forecasting

Price series are **non-stationary** and close to a random walk. Five
architectures were implemented: **LSTM** (Hochreiter & Schmidhuber, 1997),
**GRU** (Cho et al., 2014), **TCN** (Bai et al., 2018), **Transformer**
(Vaswani et al., 2017) and a **CNN-LSTM** hybrid.

**The metric is directional accuracy (DA), not RMSE.** A model can have low
squared error while being systematically wrong about direction — worthless for
deciding. See §8.1: R² is near zero and sometimes negative, and the report says so.

## 2.2 Reinforcement learning for trading

Trading is formulated as an MDP: state = market observation window + position,
action = buy/hold/sell (or a continuous weight), reward = wealth change
penalised by risk. Three families, 13 algorithms (§4.3):

- **Value-based** — DQN, Double DQN, Dueling DQN
- **Distributional** — C51, QR-DQN, IQN, Rainbow
- **Policy gradient / actor-critic** — PPO, A2C, TRPO, DDPG, TD3, SAC

The distributional approach (Bellemare et al., 2017) is particularly relevant in
finance: it learns the **distribution** of returns rather than only its
expectation, giving access to tail quantiles.

## 2.3 Risk measurement

**VaR** is a loss quantile and is *not coherent* in the sense of Artzner et al.
(1999) — it violates sub-additivity. **CVaR** (expected shortfall) is coherent
and is preferred here. Backtesting uses **Kupiec** (1995) for the *number* of
breaches and **Christoffersen** (1998) for their *independence*. The measured
result (§8.4.1) is instructive: most estimators pass Kupiec, **none passes
Christoffersen**.

## 2.4 Explainable AI

**SHAP** (Lundberg & Lee, 2017), **LIME** (Ribeiro et al., 2016) and
**counterfactuals** are implemented, plus a **native** RL explanation: regime
attribution by knockout (§4.5).

---

# 3 · Architecture

## 3.1 Overview

Layered architecture, monorepo, no client-side build step:

```text
+------------------------------------------------------------+
|  PRESENTATION - 13 HTML pages, 21 JS modules, 4 CSS         |
|  Plotly.js - light/dark theme - no framework                |
+-----------------------------+------------------------------+
                              | HTTP/JSON (HttpOnly cookie)
+-----------------------------v------------------------------+
|  API - FastAPI - 127 paths - 135 HTTP operations            |
|  AuthGuardMiddleware (default deny)                         |
+-----------------------------+------------------------------+
|  SERVICES - 11 business packages                            |
|  data - indicators - forecasting - rl - risk - nlp          |
|  recommendation - xai - alerts - chat - notifications       |
+-----------------------------+------------------------------+
|  PERSISTENCE - SQLite (9 tables) - Parquet cache            |
|  .pt/.zip models - YAML configs                             |
+------------------------------------------------------------+
```

## 3.2 Data model

Nine SQLite tables. Measured volumes: `users` 87 rows, `alerts` 2 920,
`alert_rules` 7, `portfolios` 1, `positions` 3, `transactions` 3,
`recommendation_log` 3, `model_registry` 0, `portfolio_snapshots` 0.

`recommendation_log` materialises the governance requirement: every
recommendation records its source, model version, algorithm, detected regime and
its influence, risk metrics and the explanation.

## 3.3 Data access strategy

Four-tier cascade: **memory cache** → **Parquet disk cache** → **online
provider** (Yahoo Finance) → **synthetic generator**, the last always labelled
`SIMULATED` in the interface. The label is a principle: the user must always
know whether the data is real.

## 3.4 Design decision: display period ≠ computation period

> "The selected period should only control the data displayed to the user, not
> the amount of historical data used by the analytical models."

`backend/app/utils/periods.py` separates `analysis_window()` (display) from
`model_bars()` (the bar floor each model needs). Measured effect: **8 distinct
responses for 8 periods**, against 4 out of 11 before.

---

# 4 · Implementation

## 4.1 Market data and indicators

**32 instruments**: 13 equities, 4 ETFs, 4 crypto, 4 FX, 4 indices,
3 commodities. **21 functions** in `technical.py`: SMA, EMA, WMA, RSI, MACD,
stochastic, ROC, Williams %R, CCI, Bollinger bands, True Range, ATR, Keltner
channels, historical volatility, ADX, Ichimoku, OBV, VWAP, Money Flow Index.

## 4.2 Deep forecasting

Five architectures, early stopping, normalisation fitted **on the training split
only** — necessary to avoid leakage. The *return* is predicted, never the raw
price. Confidence intervals are calibrated by **Adaptive Conformal Inference**,
which guarantees empirical coverage without distributional assumptions.

## 4.3 Reinforcement learning

All 13 catalogue algorithms are available: `dqn`, `double_dqn`, `dueling_dqn`
(value-based, discrete); `c51`, `qr_dqn`, `iqn`, `rainbow` (distributional,
discrete); `ppo`, `a2c`, `trpo` (both action spaces); `ddpg`, `td3`, `sac`
(continuous).

**MDP.** State = normalised indicator window + position + cash. Actions =
discrete {sell, hold, buy} or a continuous weight. Frictions = transaction cost
and slippage on every order. Reward = wealth change − risk penalty.

**Regime awareness.** `regime_features.py` adds **6 variables** (single asset)
or **3n+4** (basket of n assets) to the observation. Measured cost: `_classify`
= 9.8 ms per call; per-bar precomputation cuts 752 bars from 7.4 s to **1.19 s**.
The extension is **opt-in** (`EnvConfig.regime_aware`, default `False`):
enabling it by default would have broken the 11 already-trained agents, whose
input layer expects 36 dimensions rather than 42. Measured reward effect:
holding through a crash is penalised **2 885 points** more than in standard mode.

## 4.4 Risk engine

**Overall risk score, rebuilt.** The old score was `max(crash, bubble, anomaly)`,
each term relative to the asset's own history. Measured absurdity: **NVDA at
36.6 % annualised volatility was classed `low`** while **GLD at 28.5 % was
`high`**. The new score is a **weighted mean of 8 contributors on absolute
scales** and publishes each contribution — they sum back to the score.

| Spearman(volatility, score) | Before | After |
|---|---|---|
| | 0.76 | **0.976** |

**Available measures.** VaR (historical, parametric, Cornish-Fisher, Student-t,
EWMA, Monte-Carlo, filtered historical simulation, extreme value theory), CVaR,
drawdown, volatility, beta, Sharpe/Sortino/Calmar, anomaly detection (Isolation
Forest), regime detection, bubble indicator.

## 4.5 Explainable AI

| Method | Scope | Question answered |
|---|---|---|
| SHAP | local | Which variable drove *this* prediction? |
| LIME | local | How does the model behave *around* this state? |
| Permutation importance | global | Which variables matter in general? |
| Counterfactuals | local | What minimal change flips the decision? |
| Regime attribution | native (RL) | Was the regime decisive, contributory or negligible? |

Regime attribution works by **knockout**: neutralise a variable and measure the
decision shift. For baskets, turnover is **half** the L1 norm of the allocation
change — otherwise every transfer is counted twice.

## 4.6 Governance and reproducibility

**19 YAML files** (`defaults.yaml` + 13 algorithms + 5 profiles);
`ensure_configs()` recreates `configs/` additively if missing; every run records
profile, seed, resolved hyperparameters and a SHA-256 checkpoint fingerprint;
`log_rl_decision()` / `log_allocation_decision()` write the audit trail.

---

# 5 · User interface

**Scope.** 13 pages: 10 dashboard pages (Market Overview, Technical Analysis, AI
Forecasting, RL Agent, Recommendations, Explainability, Portfolio, Risk &
Alerts, Hyperparameters, Training Intelligence) plus the public landing page,
the authentication screen and AI Stress Testing.

**Design system**, built with **no backend change**: theme tokens (`theme.css`)
with a 5-level surface scale, a named type scale and a 4 px spacing scale, each
token declared **twice** (dark/light) with parity enforced by test; **no
hard-coded colour** outside the theme file (a test fails if a hex appears in a
page script); one button family with five intents; inline stroke-SVG icons (the
embedded preview has no network access, so a remote icon font would fail
silently); `prefers-reduced-motion` disables **all** animation.

**Accessibility.** Contrasts measured, not estimated: disabled button label
**8.93:1** (dark) and **8.55:1** (light); inactive navigation icon **4.63:1** /
**5.17:1** (AA threshold for graphical components: 3.0:1). Visible focus ring on
every interactive element; decorative icons marked `aria-hidden`.

> **Screenshots.** The gallery was removed from the repository to keep it under
> 30 MB. It is reproducible in full with `python3 scripts/capture_screens.py`
> (Appendix A), which writes 14 PNGs to `docs/screens/`. Captures come from the
> running application in a real browser (Chromium via Playwright, 1460 px
> viewport, ×2 density) — they are not mock-ups, and the numbers visible in them
> are what the platform computed at capture time.

---

# 6 · Measured results

## 6.1 Quantitative-finance grounding

Log returns $r_t = \ln(P_t/P_{t-1})$ are assumed non-stationary and fat-tailed —
verified here: measured excess kurtosis is positive and the GJR-GARCH win
(§6.4.2) confirms asymmetry. The platform **never** assumes normality: historical
VaR and CVaR are non-parametric.

VaR at level $\alpha$ is the quantile
$\mathrm{VaR}_\alpha = -\inf\{x : P(r \le x) \ge \alpha\}$; it is **not
sub-additive**, violating Artzner et al. (1999). CVaR corrects this:

$$\mathrm{CVaR}_\alpha = \mathbb{E}\big[\,r \;\big|\; r \le -\mathrm{VaR}_\alpha\,\big]$$

which is why the stress engine **ranks assets by CVaR, not VaR**. GJR-GARCH(1,1)
models leverage through a term active only on negative shocks:

$$\sigma_t^2 = \omega + (\alpha + \gamma \mathbb{1}_{\{\epsilon_{t-1} < 0\}})\,\epsilon_{t-1}^2 + \beta\sigma_{t-1}^2$$

Portfolio volatility is homogeneous of degree 1 in the weights, so it decomposes
exactly (Euler):

$$\sigma_p = \sum_i w_i \frac{\partial \sigma_p}{\partial w_i}, \qquad \mathrm{RC}_i = w_i \frac{(\Sigma w)_i}{\sigma_p}$$

This identity produces the counter-intuitive result of §6.5: GC=F carries 20 % of
capital for 4.5 % of risk.

> **On quantum computing.** This project does **quantitative** finance, not
> quantum computing: there is no Qiskit or Pennylane dependency and **0
> occurrences** of `quantum` in the source. The literature does explore QAOA for
> portfolio optimisation and quantum amplitude estimation for Monte-Carlo
> pricing; **none of it is implemented here**. Mentioning it without coding it
> would be an appeal to authority.

## 6.2 Deep forecasting

| Checkpoint | DA | vs chance | R² | RMSE |
|---|---:|---:|---:|---:|
| `KO_gru_h5` | **67.57 %** | **+17.6 pts** | 0.165 | 0.0312 |
| `AAPL_lstm_h5` | **62.57 %** | **+12.6 pts** | 0.0398 | 0.0346 |
| `AAPL_gru_h5` | 54.67 % | +4.7 pts | **−0.0534** | 0.0473 |
| `AAPL_cnn_lstm_h5` | 54.55 % | +4.6 pts | 0.0021 | 0.0353 |
| `AAPL_transformer_h5` | 54.01 % | +4.0 pts | **−0.0034** | 0.0353 |
| `EURUSD_X_gru_h5` | 53.61 % | +3.6 pts | **−0.0280** | 0.0081 |
| `AAPL_tcn_h5` | 52.94 % | +2.9 pts | 0.0360 | 0.0346 |

**Two models have a real edge.** `KO_gru_h5` and `AAPL_lstm_h5` are clearly above
chance and are the only two with a clearly positive R² — not a coincidence: a
model that captures variance tends also to capture direction.

**Five sit at the noise floor.** Between 52.94 % and 54.67 %, the gap to chance
is 2.9–4.7 points; over ~187 test observations that is not distinguishable from
luck.

**Three R² are negative** — `AAPL_gru` (−0.0534), `EURUSD_X_gru` (−0.0280),
`AAPL_transformer` (−0.0034). A negative R² means literally: *this model predicts
worse than the historical mean*. They are kept and shown as they are, because a
bad model removed from the report is still bad in the product.

**Why the Transformer disappoints.** It is the most data-hungry of the five:
self-attention must learn which sequence positions matter, with no temporal
prior. On ~1 250 bars it cannot, where an LSTM starts with a recurrent bias
suited to series. Dataset size, not architecture, explains the ranking.

## 6.3 Reinforcement learning

Each algorithm trained on **5 independent seeds**, same environment, same budget
(8 episodes), same AAPL 2-year window (400 training bars, 101 test). Returns are
net of transaction costs; the standard deviation is over the 5 seeds.

| Algorithm | Family | Mean return | σ | Min | Max |
|---|---|---:|---:|---:|---:|
| A2C | actor-critic | **+5.52 %** | **0.63** | +4.37 | +6.04 |
| SAC | actor-critic | +4.79 % | 2.59 | +1.28 | +7.08 |
| TRPO | policy gradient | +4.02 % | 1.23 | +2.07 | +5.40 |
| TD3 | actor-critic | +4.00 % | 1.85 | +1.68 | +7.01 |
| DDPG | actor-critic | +3.34 % | 2.71 | +0.27 | +6.54 |
| PPO | policy gradient | +2.94 % | 2.35 | +0.00 | +5.90 |
| Double DQN | value-based | +0.09 % | 0.57 | −0.38 | +1.18 |
| C51 / IQN / QR-DQN / Rainbow | distributional | **0.00 %** | **0.00** | 0.00 | 0.00 |
| DQN | value-based | −0.52 % | 1.85 | −4.04 | +1.44 |
| Dueling DQN | value-based | −1.86 % | 2.89 | −6.98 | +0.79 |
| **Buy & Hold** | *passive benchmark* | **+21.17 %** | — | — | — |

**Actor-critic methods dominate.** They learn a policy directly and handle
continuous actions natively; "invest 37 %" is a natural action for an allocation
problem. A2C is best **with the lowest dispersion** (σ = 0.63) — the only
algorithm whose 5 seeds fit inside a 1.7-point band.

**Value-based methods fail.** They estimate a value per discrete action then take
the maximum. At 8 episodes the Q estimate stays noisy and `max` amplifies that
noise (the overestimation bias Double DQN was invented for). Dueling DQN is worst
at **−1.86 %** with the **highest dispersion** (σ = 2.89, from −6.98 % to +0.79 %).

**The distributional family does not trade.** **0.00 % across all 5 seeds, σ
exactly 0.00.** Any position would produce seed-to-seed variation; zero
dispersion means zero decisions. Two independent measurements agree — **max
drawdown 0.0** and **Sharpe 0.0**, impossible for an exposed agent.

> **Methodological caveat.** The trade count is not recorded in these artefacts.
> The absence of trading is therefore **deduced** from three concordant
> quantities (zero return, zero dispersion, zero drawdown), not read from a
> counter. The deduction is solid but remains a deduction, and the report says so.

### The central result: nothing beats the passive strategy

- Best agent (A2C): **+5.52 %**
- Passive benchmark: **+21.17 %**
- **Gap: −15.65 points**

None of the 13 comes close. The likely cause is the conjunction of three measured
factors: an **8-episode budget** too short to converge, a **rising test window**
where doing nothing was optimal, and **transaction costs** that penalise any
activity.

> **Accepted consequence.** The specification's goal — "optimise investment
> decisions" — is **not met** (SPEC-2). The result is published at the head of
> the section rather than buried: a decision-support platform that hid its RL
> component losing to inaction would be misleading.

### The M122 study: regime augmentation does not help

Dedicated test on Dueling DQN, 5 seeds per arm:

| Arm | Return | IQM | 95 % bootstrap CI | σ |
|---|---:|---:|---|---:|
| Baseline | +8.66 % | +8.26 % | [+5.95, +12.30] | 3.08 % |
| Regime-aware | +5.21 % | +5.08 % | [−2.59, +13.00] | 7.11 % |

The intervals **overlap entirely** and the modified arm's **contains zero**.
Paired *t* = −0.804 (*p* = 0.467), Welch *p* = 0.362, *d* = −0.63. The
modification improves **one seed in five**. Adding 6 regime variables takes the
observation from 36 to 42 dimensions **without increasing the training budget**:
variance doubles (σ 3.08 → 7.11) with no gain in mean. That is an unfavourable
bias-variance trade, not an implementation defect.

## 6.4 Risk

### 6.4.1 Value at Risk: no estimator validated

Backtest on AAPL, 5 years, 1 253 observations, 95 % level.

| Estimator | Breaches | Kupiec *p* | Independence *p* | Verdict |
|---|---:|---:|---:|---|
| Historical | 5.58 % | 0.405 ✔ | **0.0032** ✘ | rejected |
| Parametric | 4.79 % | 0.754 ✔ | **0.0295** ✘ | rejected |
| Cornish-Fisher | 6.78 % | **0.0139** ✘ | **0.0015** ✘ | rejected |
| Student-t | 5.88 % | 0.212 ✔ | **0.0018** ✘ | rejected |
| EWMA | 6.98 % | **0.0065** ✘ | **0.0089** ✘ | rejected |
| Filtered historical | 5.58 % | 0.405 ✔ | **0.0032** ✘ | rejected |
| Monte-Carlo | 5.58 % | 0.405 ✔ | **0.0032** ✘ | rejected |

**The contrast between the two columns is the finding.** Five of seven pass
Kupiec: they promise 5 % exceptional losses and observe 4.8–5.9 %. Seen that way
the model "works". But **all seven fail independence**: breaches arrive in
**clusters** — volatility clustering, which an unconditional VaR assumes away.

> **Why it matters.** A model wrong 5 % of the time in scattered fashion is
> manageable. One wrong 5 % of the time *but concentrating its errors in the same
> week* exposes the holder to consecutive losses — exactly the scenario that
> ruins a portfolio.

No VaR is presented as "validated": `model_valid = False` propagates to the display.

### 6.4.2 GARCH: leverage is measurable

| Model | AIC | Reading |
|---|---:|---|
| **GJR-GARCH** | **4 709.8** | *best* — asymmetry accounted for |
| EGARCH | 4 721.8 | logarithmic asymmetry |
| GARCH | 4 723.3 | symmetric |

GJR wins by 13.5 AIC points over symmetric GARCH. Its only difference is a term
active **only when the return is negative**, so winning means **negative shocks
raise future volatility more than positive shocks of equal size** — the leverage
effect. The information criterion decided, not a preference.

### 6.4.3 Risk-engine coherence

Spearman(annualised volatility, overall score) = **0.967**; monotone in
volatility; **8 distinct responses for 8 selectable periods**, proving the
selector reaches the computation rather than decorating it.

### 6.4.4 Anomaly detection — an accepted limit

Isolation Forest runs at 2 % contamination, so it **labels 2 % of any window**,
however calm. That is a **relative ranking**, not detection in the strict sense.

## 6.5 Stress testing — one real case

Basket AAPL 50 % / MSFT 30 % / GC=F 20 %, *Market Crash* scenario, 1 253
observations, $100 000 position:

| Measure | Before | After | Change |
|---|---:|---:|---:|
| VaR 95 % | 2.05 % | 2.16 % | +0.11 pt |
| CVaR | 2.80 % | 3.03 % | +0.22 pt |
| Volatility | 20.32 % | 20.75 % | +0.43 pt |
| Max drawdown | 25.07 % | **48.06 %** | **+22.99 pts** |

**VaR barely moves while drawdown doubles.** That is the lesson of stress
testing: VaR is a *daily* quantile, insensitive to the sequencing of losses;
drawdown measures a **cumulative path**. A portfolio can look safe day by day and
lose half its value over a sequence.

**Euler decomposition contradicts weight intuition:**

| Asset | Weight | Risk contribution | Share of loss |
|---|---:|---:|---:|
| AAPL | 50.0 % | **63.6 %** | 52.8 % |
| MSFT | 30.0 % | 31.9 % | 31.3 % |
| GC=F | 20.0 % | **4.5 %** | 15.9 % |

GC=F carries **20 % of capital but 4.5 % of risk**: gold decorrelates from the
rest of the basket, damping instead of amplifying. **Weight does not measure
exposure** — precisely what a naive allocation ignores.

## 6.6 Active adaptation: regime-aware Mixture-of-Experts

The regime augmentation above is **passive**: the agent *sees* the regime, but
its weights never change when the market does. That cannot react to a switch,
only describe one, and it left KPI **K-5** (reaction delay) unmeasurable. A MoE
layer was added in `services/rl/moe.py` **without modifying the environment,
reward, algorithms, XAI, risk metrics, API or UI**.

**Three experts, an explicit router.** `bull`, `bear` and `stress` are each a
deep copy of the same trained checkpoint. A lookup table — not a learned gate —
maps the seven regimes the existing detector already emits onto those three
buckets, so a risk officer can read the assignment off the table:

| Expert | Regimes covered |
|---|---|
| **bull** | `bull_market`, `recovery`, `low_volatility`, `sideways` |
| **bear** | `bear_market` |
| **stress** | `crash_risk`, `high_volatility` |

On a switch the incoming expert is fine-tuned on the bars of **its own** regime
seen **strictly before** the switch, and the bars until it takes control are
recorded as K-5.

**No leakage — two testable guards.** Regime labels come from the existing causal
provider (`df.iloc[start:t+1]`, exclusive upper bound); the fine-tuning window
uses indices $< t$. A mutation widening the bound to $t+1$ makes the test fail.

**Measured, AAPL / `dueling_dqn`, single seed:**

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
| **MoE return** | **+4.19 %** | **+29.92 %** |
| Buy & Hold | +29.73 % | +38.65 % |

**Reading these honestly.** *The mechanism runs*: fine-tuning is a genuine
gradient step verified on the weights themselves, and the factory **raises**
rather than reporting an adaptation when the replay buffer never reaches
`min_buffer`. *It fires rarely*: only 1 of 8 and 1 of 11 expert changes were
served; the rest are counted as failures to react, because the incoming expert
had fewer than the 90 bars its regime needs. *It does not beat the market*: the
MoE trails Buy & Hold on both windows. One instrument, one seed — no edge is
claimed. The deliverable is a **measurable adaptation latency**, not performance.

**Integration.** The MoE is reachable from the existing backtest endpoint,
`GET /rl/backtest/{symbol}?moe=true`, imported lazily inside that branch so the
default path is untouched. With an unreachable expert threshold the rollout
reproduces the baseline **bit for bit** (identical `performance` dict and equity
curve), and the default response gains **neither** a `moe` key **nor** a `mode`
key. SB3 policies expose `predict` rather than `q_values` and are refused with
**HTTP 422** instead of being routed to an expert that cannot adapt.

**Regime-aware twins.** 24 checkpoints are stored under a `__regime` suffix
beside their baselines, so both can be compared rather than one replacing the
other. Only a regime-aware agent can attribute a decision to the market regime;
across the twins trained, regime awareness was **better on 9, worse on 7 and
identical on 5** — close to a coin flip, consistent with *p* = 0.467 above.

---

# 7 · Software quality and verification method

**a) Verify before asserting.** Every feature is tested in a **real browser**
(Playwright), not only by reading code.

**b) Prove a test catches the regression (mutation testing).** A passing test
proves nothing until it has been shown to **fail** when the defect is
reintroduced. Verified mutations include: volatility reused as expected movement;
constant 0.75 confidence; architecture resolver ignored; horizon substitution
allowed; verdict returned to the composite; crash neutralised; hard-coded
volatility; hard-coded resilience score; inert correlation spike; risk
contribution defaulting to 0; hard-coded theme button; colour token removed from
the light theme; topbar fallback back to 860 px; launcher reserve removed; MoE
leakage bound widened; `adapt=False` silently fine-tuning; experts never taking
control; failed fine-tune reported as success; dotted-ticker filename collision.
**Each made the corresponding test fail**, as required.

**c) Report false positives from one's own tools.** Several alerts raised by my
own probes proved **wrong** on verification, and honesty requires saying so
rather than "fixing" a non-problem:

- A bounding-box probe claimed the chat button still covered text; the
  authoritative test (`elementsFromPoint`) showed **no overlap**.
- A theme-parity test demanded the light theme redeclare `--font`, `--sp-*`,
  `--fs-*`; those tokens are **deliberately neutral**. Restricted to colours.
- An anti-contradiction test **reimplemented the merge logic locally**, so the
  mutation passed against its own copy. Rewritten to call the real `predict()`.
- A crash-materiality test used a **Gaussian** fixture with no fat tail, failing
  on correct code. Replaced with Student-t (df = 3).
- A probe reported three continuous-agent twins as broken; it had forced them
  into `TradingEnv` when the application correctly routes them to `PortfolioEnv`.
  **The probe was wrong, not the agents.**

**Discipline applied.** Never degrade a useful message to satisfy a coarse test —
tighten the test instead. Never rename or delete anything that would orphan real
user data. Flag a factually incorrect request before acting on it.

---

# 8 · Specification compliance

## 8.1 Specific objectives

| # | Requirement | State | Measured evidence |
|---|---|---|---|
| O-1 | Realistic simulation environment | ✅ | `PortfolioEnv` + `TradingEnv`; `transaction_cost = 0.001`, slippage modelled |
| O-2 | Adaptive DRL agent with regime-change detection | ⚠️ **SPEC-1** | `regime.py` + `regime_features.py`; **MoE implemented and wired in** (§6.6); **no MAML** |
| O-3 | Risk-sensitive reward | ✅ | `cvar_penalty = 0.10`, `drawdown_penalty = 0.35`, `cvar_alpha = 0.05` |
| O-4 | Native and post-hoc XAI | ✅ | SHAP, `allocation_explain.py`, `regime_explain.py` |
| O-5 | Dashboard for managers and control functions | ✅ | 13 pages incl. Risk & Alerts, Explainability, AI Stress Testing |
| O-6 | Backtesting, walk-forward, stress tests | ⚠️ **SPEC-2** | Backtesting ✅, stress tests ✅ (7 scenarios); **walk-forward present for forecasting, absent for RL** |
| O-7 | Governance documentation (SR 11-7, EBA/ACPR) | ⚠️ **SPEC-3** | Model card, audit log, version registry ✅; **no formal regulatory dossier** |

## 8.2 Functional description

F-1 multi-asset OHLCV ✅ (32 instruments, 6 classes) · F-2 macro/sentiment ✅
(`^VIX`, `nlp/`) · F-3 feature engineering ✅ (21 indicator functions) ·
F-4 data quality ⚠️ **SPEC-4** (*survivorship-bias adjustment absent*) ·
F-5 MDP formulation ✅ · F-6 weight vector summing to 1 ✅ · F-7 PPO/SAC/DDPG/TD3
✅ (all four present) · F-8 multi-algorithm comparison ✅ (13 algorithms, 65 runs)
· F-9 continual learning ⚠️ **SPEC-5** (*manual retraining only*) · F-10
timestamped audit log ✅ · F-11 natural-language decision summaries ✅ ·
F-12 alerts and guardrails ⚠️ **SPEC-6** (*no circuit breaker*).

## 8.3 Constraints and KPIs

C-1 auditability ✅ · C-2 GDPR ✅ (no personal data) · C-3 GPU + CI/CD ⚠️
**SPEC-7** (*CI with ruff + 776 tests ✅; CPU-only training*) · C-4 OMS/EMS
interoperability — out of scope by the specification · C-5 model-risk committee
❌ **SPEC-8** (*organisational, not code*).

| # | KPI | State | Measured value |
|---|---|---|---|
| K-1 | Net cumulative return vs benchmark | ✅ | Best RL **+5.52 %** vs Buy & Hold **+21.17 %** |
| K-2 | Sharpe and Sortino | ✅ | Sharpe 2.00 ± 1.19; Sortino 3.18 (baseline) |
| K-3 | Max drawdown and recovery time | ⚠️ **SPEC-9** | Drawdown ✅ (−6.93 % mean); **recovery time not computed** |
| K-4 | Realised VaR/CVaR vs targets | ✅ | 7 estimators backtested (§6.4.1) |
| K-5 | Reaction delay after regime change | ✅ | **60 bars** on 1 of 8 switches (1y); 0 bars on 1 of 11 (2y) — §6.6 |
| K-6 | Stability across market regimes | ⚠️ Partial | Regime study done; **single test regime** |
| K-7 | Explanation fidelity | ❌ **SPEC-3** | **Not measured** quantitatively |
| K-8 | Comprehensibility (user survey) | ❌ Out of scope | Requires a business-user panel |
| K-9 | Coverage of explained decisions | ✅ | 100 % — every displayed decision carries its per-signal contributions |

## 8.4 Compliance summary

| Specification category | Compliant | Divergent | Out of scope |
|---|---:|---:|---:|
| Specific objectives (7) | 4 | 3 | 0 |
| Functional description (12) | 8 | 4 | 0 |
| Constraints (5) | 2 | 1 | 2 |
| KPIs (9) | 4 | 4 | 1 |
| **Total (33)** | **18** | **12** | **3** |

**Compliance: 18/30 applicable requirements (60 %), 12 divergent lines reduced to
9 distinct causes (SPEC-1 … SPEC-9), 0 requirements silently dropped.**

Of the 9 causes, five (SPEC-1, 2, 4, 7, 9) are accepted technical or hardware
limits, three (SPEC-3, 5, 6) are unreached industrialisation, and one (SPEC-8) is
organisational.

> **The most important gap.** The specification targets an agent that "optimises
> investment decisions". Measured, **none of the 13 algorithms beats a passive
> strategy** (+5.52 % vs +21.17 %). The financial-performance objective is **not
> met**, and that result is published rather than concealed: it is the scientific
> conclusion of the work, not an implementation defect to be tuned away.

### Divergences in detail

**SPEC-1 — Regime adaptation: MoE, not MAML.** *Partially resolved.* A
regime-aware MoE was added (§6.6): 3 experts, explicit router over the 7 existing
regimes, fine-tuning triggered by the switch. **K-5, previously unmeasurable, now
is.** *Remaining gap.* The specification cited MAML **or** MoE; the second branch
is built, the first is not. Adaptation remains fine-tuning on past data, not
meta-learning.

**SPEC-2 — No walk-forward for RL.** Present for forecasting, absent for RL: the
8-episode budget (a CPU limit) made a rolling protocol unaffordable.

**SPEC-3 — No formal regulatory dossier, explanation fidelity unmeasured.** The
pieces exist (model card, audit trail, version registry); the dossier and a
quantitative fidelity metric do not.

**SPEC-4 — Survivorship bias not addressed.** The catalogue holds currently
listed instruments only; delisted ones are absent, which flatters historical
performance.

**SPEC-5 — No automatic periodic retraining.** Retraining is manual through the
interface.

**SPEC-6 — No circuit breaker.** Risk thresholds and alerts exist (2 920 in the
database) but nothing **automatically suspends** the agent on abnormal behaviour.
With no real order execution, operational risk remains nil.

**SPEC-7 — CPU training, not GPU.** ~7 s per forecasting model, ~8 s per RL run.
This is what capped the budget at 8 episodes and the study at 5 seeds.

**SPEC-8 — No model-risk committee review.** Organisational, out of reach for a
single-author academic deliverable.

**SPEC-9 — Recovery time not computed.** Max drawdown is measured; the recovery
time K-3 requires is not recorded, and is **reported as unavailable** rather than
estimated.

**On "quantum finance".** One request mentioned quantum finance. The project
implements **no quantum computing** — 0 occurrences of `quantum`, no Qiskit or
Pennylane. §6.1 covers the quantitative-finance foundations actually implemented,
each equation matching existing code.

---

# 9 · Defects found and fixed

| # | Defect | User-visible consequence | Status |
|---|---|---|---|
| 1 | Realised volatility presented as expected movement | Confident false prediction | fixed |
| 2 | Architecture hard-wired to `lstm` | 2 symbols declared "no model" when they had one | fixed |
| 3 | Verdict from the composite, opposite to the displayed figure | "INCREASE" above "−0.88 %" | fixed |
| 4 | Forecaster diluted to 30 % | Near-systematic NEUTRAL | fixed |
| 5 | `argmin` on a series with leading NaN | Crash lasted 1 day instead of 21 | fixed |
| 6 | Crash episode repeated excessively | Unrealistic −98.6 % drawdown | fixed |
| 7 | `?? 0` on risk contribution | "0 % risk" instead of "not measurable" | fixed |
| 8 | Hard-coded, empty theme button | Dead theme on `stress.html` | fixed |
| 9 | Topbar fallback at 860 px | Page dragged 49→195 px sideways | fixed |
| 10 | Insufficient bottom padding | Chat button covering data | fixed |
| 11 | Stale endpoint docstring | False contract published in `/openapi.json` | fixed |
| 12 | Pre-existing test encoding the bug (`abs()` if NEUTRAL) | False assumption turned green | tightened |
| 13 | Double rounding of the composite score | Prose said +0.334 while JSON published +0.335 | fixed |
| 14 | Dotted tickers (`MC.PA`) collapsing checkpoints | `with_suffix` overwrote the stem; 6 algorithms wrote to one file | fixed |
| 15 | `recommend_allocation` ignoring `variant` | Basket twins unreachable; the baseline served silently | fixed |

> **Defect 14 — found by training all 32 symbols.** Every save site calls
> `.with_suffix(".pt")`, and `Path` reads `.PA_dqn__regime` as an extension and
> *replaces* it, so all six algorithms for `MC.PA` wrote to `rl_MC.pt`. Four then
> failed to load with `unexpected keyword argument 'n_atoms'` — a C51 checkpoint
> read back as a DQN. Pre-existing; never exercised because no `.PA` agent had
> been trained before.

---

# 10 · Known limitations

## 10.1 Blocking before public deployment

SMTP not configured (password reset inoperative) · no multi-factor
authentication · no refresh-token rotation · `SECRET_KEY` must be changed ·
no lockout after repeated failures.

## 10.2 Scientific

- **Forecast coverage: 2 symbols of 32**, one horizon (5 days).
- **No VaR estimator passes Christoffersen.**
- **No RL agent beats Buy & Hold** (+21.17 %).
- **The M122 study is underpowered**: n = 5 seeds; about **61** would be needed
  for 80 % power.
- **Multi-seed impossible on the 14 shipped agents** (seed 42 frozen at training).
- **The MoE fires rarely** — 1 of 8 switches on a 1-year window — and shows no
  performance gain.

---

# 11 · Conclusion

The platform meets its **traceability** objective: every displayed number is
reproducible from a documented command, and unfavourable results are published
rather than hidden. Its **financial-performance** objective is **not met**: no
reinforcement-learning agent among the thirteen beats a passive strategy at this
training budget, and the report states so at the head of the results section
rather than in a footnote.

Three findings deserve to outlive the project. First, **honest measurement is
itself a deliverable**: the negative R², the seven VaR estimators failing
independence, and the distributional family learning not to trade are all more
informative than a tuned success story. Second, **regime awareness is not free**:
adding six variables at a fixed budget doubled variance without moving the mean.
Third, **an adaptation mechanism must be measured, not asserted**: the MoE only
earns the word "active" because the weight change is verified numerically, and it
fires far less often than its design suggests.

Future work follows directly from the documented divergences: raise the training
budget beyond 8 episodes on GPU (SPEC-7), add walk-forward validation for RL
(SPEC-2), and implement the meta-learning branch left open by SPEC-1.

---

# Appendix A · Verification commands

```bash
# Test count and per-file breakdown
export PYTHONPATH=backend
python3 -m pytest backend/tests -q -p no:cacheprovider --collect-only

# Full suite (run in two halves: ~5 min total)
python3 -m pytest backend/tests/test_api.py backend/tests/test_auth_and_brand.py \
  backend/tests/test_access_control.py -q -p no:cacheprovider
python3 -m pytest backend/tests/test_models_and_services.py backend/tests/test_chat.py \
  backend/tests/test_intelligence.py backend/tests/test_quant.py \
  backend/tests/test_data_and_indicators.py -q -p no:cacheprovider

# Static analysis (official CI scope)
python3 -m ruff check backend/app backend/tests

# Forecast coverage across the 32 instruments
python3 scripts/forecaster_coverage.py

# Code volume
find backend frontend scripts -type f \( -name "*.py" -o -name "*.js" \
  -o -name "*.html" -o -name "*.css" \) -not -path "*/__pycache__/*" | xargs wc -l | tail -1

# Reproduce the screenshot gallery into docs/screens/
bash scripts/run_server.sh start
python3 -m playwright install chromium
python3 scripts/capture_screens.py

# MoE rollout (regime routing + K-5)
curl -b cookies.txt "http://127.0.0.1:8000/api/v1/rl/backtest/AAPL?algo=dueling_dqn&period=1y&moe=true"
```

# Appendix B · Related documents

| Document | Content |
|---|---|
| `docs/RAPPORT_COMPLET.md` | Full French edition of this report |
| `docs/M122_RL_MINI_PROJECT.md` | M122 mini-project, Markdown edition |
| `docs/MOE_INTEGRATION.md` | MoE integration audit |
| `docs/REGIME_AWARE_TWINS.md` | Regime-aware twin checkpoints |
| `docs/SWEEP_ALL_SYMBOLS.md` | 32 × 6 training sweep |
| `docs/ARCHITECTURE.md` | Detailed technical architecture |
