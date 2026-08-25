# Architecture

Technical reference for the KorisQuant AI platform: layering, data flow, model design and the
engineering decisions that matter.

---

## 1. Layered design

```
┌───────────────────────────────────────────────────────────────────────┐
│ PRESENTATION   frontend/ — 8 static pages, vanilla JS + Plotly        │
│                served by FastAPI itself (no Node build step)          │
├───────────────────────────────────────────────────────────────────────┤
│ API            app/api/v1/ — 61 REST endpoints, Pydantic-validated    │
│                SafeJSONResponse guarantees finite, serialisable JSON  │
├───────────────────────────────────────────────────────────────────────┤
│ DOMAIN         app/services/ — the whole intelligence stack           │
│   data · indicators · forecasting · rl · nlp · risk · xai ·           │
│   recommendation · alerts                                             │
├───────────────────────────────────────────────────────────────────────┤
│ PERSISTENCE    app/db/ — SQLAlchemy 2.0 async ORM                     │
│                SQLite by default, PostgreSQL-ready                    │
├───────────────────────────────────────────────────────────────────────┤
│ INTEGRATION    providers (Yahoo/Finnhub/AlphaVantage/Polygon),        │
│                two-tier cache, deterministic synthetic engine         │
└───────────────────────────────────────────────────────────────────────┘
```

Dependencies point strictly downward. Services never import from `api/`, which makes every
service directly usable from a notebook, a script or a worker.

---

## 2. The hybrid data layer

The single most important reliability decision in the system: **a market-data request must
never simply fail.**

```
get_history(symbol, period, interval)
        │
        ├─ 1. fresh cache?            memory dict → parquet on disk      → source="cache"
        ├─ 2. live provider chain     yahoo → finnhub → alphavantage → polygon
        │                             each wrapped in try/except, returns None on failure
        ├─ 3. stale cache             any age — an old real price beats no price
        └─ 4. synthetic engine        deterministic, always succeeds     → source="synthetic"
```

Every response carries `source`, and the UI renders it as a colour-coded badge. Users always
know whether they are looking at live data, cached data, or a simulation. `DATA_MODE=live`
turns tier 4 into a `503`; `DATA_MODE=offline` skips tiers 2–3 entirely.

### The synthetic engine

Not random noise — it reproduces the stylised facts of financial time series:

| Property | Implementation |
|---|---|
| Volatility clustering | GARCH(1,1)-like recursion, ω=0.05, α=0.10, β=0.85 |
| Fat tails | Student-t innovations (ν=4.5), variance-normalised |
| Jumps / crashes | Poisson-timed shocks, amplified 2.5× for crypto |
| Mean reversion | Mild drag on cumulative drift keeps prices anchored |
| OHLC coherence | high ≥ max(open,close), low ≤ min(open,close), enforced by construction |
| Volume | Log-normal, correlated with \|return\|, 5-day seasonality |
| Determinism | SHA-256 of `(symbol, seed)` → identical output for identical inputs |

This is what makes the test suite fully offline and reproducible.

---

## 3. Feature engineering

`build_features()` produces 31 engineered features per bar, grouped as:

- **Returns** — 1/5/10/21-day and log returns
- **Volatility** — 10d/21d realised, plus the short/long ratio (regime detector)
- **Momentum** — RSI, MACD + histogram, Stochastic %K, CCI, ADX with ±DI
- **Mean reversion** — Bollinger %B and bandwidth
- **Trend position** — price relative to SMA20/SMA50/EMA12, SMA20-vs-SMA50 spread
- **Volume** — relative volume, OBV slope, Money Flow Index
- **Microstructure** — intraday range, close-vs-high, opening gap
- **Calendar** — day of week, month

`build_supervised()` joins features with forward-looking targets and drops incomplete rows.
Because the target is `close.shift(-horizon) / close - 1`, the last `horizon` rows are removed
automatically — **look-ahead bias is structurally impossible**, and a test asserts it.

---

## 4. Deep-learning forecasters

All five architectures share the contract `(batch, lookback, n_features) → (batch, 1)`.

| Model | Design | Best suited to |
|---|---|---|
| **LSTM** | 2 layers, 64 hidden, LayerNorm head | Long-range dependencies |
| **GRU** | 2 layers, 64 hidden | Faster, fewer parameters |
| **TCN** | Dilated causal conv, dilations 1/2/4, weight-norm, residual | Wide receptive field, parallel |
| **Transformer** | 2 encoder layers, 4 heads, pre-norm, sinusoidal PE, mean pooling | Global context |
| **CNN-LSTM** | Conv1d feature extraction → LSTM | Local patterns then sequence |

### Anti-leakage training protocol

1. Split **chronologically** into train / validation / test — never shuffled across time.
2. Fit `StandardScaler` on **train only**; transform the rest.
3. Build sliding windows *after* scaling, offsetting split indices by `lookback`.
4. Shuffle **windows** during training (not timesteps within a window).
5. Early-stop on validation loss; restore the best state dict.
6. Report metrics on the untouched test segment.

Targets are **returns**, not prices. Predicting price levels yields a deceptively high R²
because prices are near-random-walks — the model just learns "tomorrow ≈ today".

### Confidence intervals

Empirical, not Gaussian-assumed: residuals from the last ≤250 in-sample predictions give σ,
the band widens as √(step/horizon), and z=1.645 gives a 90% interval.

---

## 5. Reinforcement learning

### Environments

**`TradingEnv`** — single asset, discrete `{SELL, HOLD, BUY}`.
Observation = 31 normalised market features + 5 account features (exposure, P&L, drawdown,
recent volatility, time progress).

**`PortfolioEnv`** — multi-asset, continuous action over assets + cash, softmax-projected onto
the simplex (long-only by construction).

### The reward function

Naive PnL rewards produce agents that take catastrophic leverage. The reward is risk-adjusted:

```
reward = ( step_return
         − 0.15 × rolling_volatility
         + 0.35 × drawdown            (drawdown ≤ 0, so this is a penalty)
         − 0.02 × turnover )          × 100
```

with a −10 terminal penalty on ruin (equity < 20% of initial). Because the reward stream is
penalised, **Q-values are usually all negative** — only their *relative ordering* matters. The
UI states this explicitly and ranks the bars relative to the worst action.

### Frictions

Proportional transaction cost (default 10 bps) *and* slippage (5 bps) applied on execution
price. A test asserts that an environment with costs ends up poorer than an identical
cost-free one after the same trade sequence.

### Algorithms

| Algorithm | Implementation | Action space |
|---|---|---|
| DQN / Double DQN / Dueling DQN | **Native PyTorch** — replay buffer, target network, ε-decay | Discrete |
| PPO, A2C | Stable-Baselines3, native PPO fallback | Both |
| SAC, TD3 | Stable-Baselines3 | Continuous (portfolio) |

The DQN family is implemented natively so the discrete agent works even in a minimal install
without SB3.

### Honest evaluation

**Warm-up must not come from the training set.** Indicators need history before
they produce a value, and the obvious shortcut is to extend the test window
backwards into training data. That is data leakage: those bars are fitted on and
then scored. `TradingEnv` already solves this by starting at `t = lookback`, so
the first bars of the *test* set act as context and are never scored. The split
is strictly disjoint and a runtime assertion enforces it.

Agents train on the first 80% of history and are evaluated on the remaining 20%, always
against three baselines: **Buy & Hold**, **SMA(20/50) crossover**, and **cash**. `alpha_vs_buy_hold`
is surfaced prominently. Losing to Buy & Hold is reported, not hidden.

---

## 6. NLP pipeline

```
collect → classify → score sentiment → weight by impact → aggregate
```

**Sentiment** degrades gracefully across three tiers: FinBERT (if `transformers` is installed
and the network permits) → finance-tuned lexicon with negation and intensifier handling →
heuristic. The public API is identical regardless of tier, and the active backend is reported
in every response.

**Impact score** = `|sentiment| × category_weight × confidence × recency × relevance`, where
category weights encode that a Fed decision (0.95) matters more than a product launch (0.50),
and recency decays over a 14-day half-window.

---

## 7. Risk engine

| Detector | Method |
|---|---|
| Volatility spike | Rolling-21d realised vol z-scored against its own 63d regime |
| Return outlier | **Modified** z-score (median/MAD) — robust to the outliers it hunts |
| Volume anomaly | z-score on log volume |
| Structural break | CUSUM on standardised returns, resets after each detection |
| Multivariate | Isolation Forest over the full 31-feature matrix |
| Bubble | Log-price deviation from long-run trend + momentum + RSI + vol regime |
| Crash risk | Composite of vol regime, drawdown, negative skew, kurtosis, CVaR, losing streak |

Standard z-scores are deliberately avoided for return outliers: the mean and standard
deviation are themselves corrupted by the extreme values, so a genuine crash can hide inside
its own inflated σ. Median/MAD does not have this problem.

---

## 8. Recommendation fusion

```python
effective_weightᵢ = base_weightᵢ × (0.35 + 0.65 × reliabilityᵢ)
composite         = Σ(scoreᵢ × effective_weightᵢ) / Σ(effective_weightᵢ)
adjusted          = composite + risk_drag        # drag ≤ 0, magnitude-capped
```

Base weights: forecast 0.30, RL 0.25, technical 0.25, sentiment 0.20 — renormalised over
whatever is actually available, so an untrained model contributes nothing instead of injecting
a fabricated zero.

Reliability is measured, not assumed:
- forecast → out-of-sample directional accuracy, mapped from [45%, 70%] → [0, 1]
- RL → agent Sharpe and alpha vs Buy & Hold on unseen data
- technical → strength of indicator consensus
- sentiment → article count and analyser confidence

**Risk overlay invariant:** the drag is clamped to `min(raw_drag, composite)`, so it can pull a
bullish score to zero but never past it. Elevated risk means *stand aside*, not *go short*.
This is enforced by a test.

**Position sizing** targets 15% portfolio volatility, scaled by conviction and cut by a
risk-level haircut (low 1.0 → critical 0.2), hard-capped at 35%. Stops derive from ATR.

---

## 9. Explainability

| Technique | Implementation | Fallback |
|---|---|---|
| SHAP | TreeSHAP when `shap` is installed | Permutation-based sampled Shapley (batched: one `predict` call per permutation, not per feature) |
| LIME | Weighted ridge regression on Gaussian perturbations, exponential kernel | — (native) |
| Global importance | scikit-learn permutation importance + impurity importance | — |
| Counterfactual | Minimal single-feature σ-shift that flips the predicted direction | — |

A gradient-boosting **surrogate** is fitted on the same features and explained. This is fast,
model-agnostic and works uniformly across all five deep architectures.

LIME coefficients are *local sensitivities*, not additive shares of a baseline — the narrative
wording reflects that distinction rather than misreporting them as SHAP-style contributions.

---

## 10. Cross-cutting engineering

**NaN-safe JSON.** Indicators produce `NaN` during warm-up and `±Inf` on zero denominators;
`json.dumps` rejects both. `SafeJSONResponse` recursively normalises NaN/Inf → `null`, numpy
scalars → Python scalars, Timestamps → ISO strings. Applied once as the app-wide default
response class instead of sprinkling defensive code through 61 endpoints. A test asserts no
`NaN` token ever appears in a dashboard payload.

**Error taxonomy.** `KorisQuantError` subclasses carry their own HTTP status and machine-readable
code (`data_unavailable` 503, `model_not_trained` 409, `invalid_request` 422, `portfolio_error`
400), mapped centrally by one exception handler.

**Caching.** Memory TTL dict → parquet on disk → optional Redis, all behind one interface that
degrades silently when Redis is absent.

**Concurrency.** Batch quotes, dashboard aggregation and heatmaps fan out over a
`ThreadPoolExecutor` — provider calls are I/O-bound, so this is a large latency win.

---

## 11. Data model

```
users ──< portfolios ──< positions
                    ├──< transactions
                    └──< portfolio_snapshots
users ──< alerts
         alert_rules
         model_registry
         recommendation_log
```

`positions` carries a unique constraint on `(portfolio_id, symbol)`; buys update a weighted
average cost basis, and fully-closed positions are deleted rather than left at quantity zero.

---

## 12. Extension points

**A new indicator** — add the function to `services/indicators/technical.py`, register its key
in `INDICATOR_REGISTRY`, and wire it into `compute_indicators()`.

**A new architecture** — subclass `nn.Module` with the `(batch, lookback, features) → (batch, 1)`
contract, add it to `MODEL_REGISTRY`; training, prediction, comparison and the UI dropdown pick
it up automatically.

**A new data provider** — implement `available()`, `fetch_history()`, `fetch_quote()` and append
to `PROVIDER_CHAIN`. Returning `None` on failure is all the error handling required.

**A new RL algorithm** — expose `train()`, `act()`, `evaluate()`, `save()`; register it in
`SUPPORTED_ALGOS`.
