# KorisQuant AI — Final-Year Project Report
## Detailed outline & writing guide

> **How to use this document.** Part A is the *process* (the order in which to
> work). Part B is the *structure* (every chapter and section title, what goes
> inside, and which real figure from this repository to cite).
>
> Every number quoted below was measured in this repository. Where a figure is
> a *design choice* rather than a measurement, it says so. Do not invent
> numbers: a jury that checks one and finds it wrong will distrust all of them.

---

# PART A — The process (8 steps)

| # | Step | Output | Suggested time |
|---|------|--------|----------------|
| 1 | **Freeze the scope.** List what the platform does and — just as important — what it deliberately does *not* do (no live brokerage, no investment advice, no MFA yet). | 1 page of notes | half a day |
| 2 | **Collect the evidence.** Re-run the test suite, the lint, the install check and `scripts/diag_risk.py`. Save the console output — these become appendices and screenshots. | raw logs | half a day |
| 3 | **Capture the screenshots.** 8 dashboard pages × light theme, plus the sign-in screen and the landing page. Add close-ups of the Overall Risk breakdown and the regime panel. | ~15 images | 1 day |
| 4 | **Draw the diagrams.** Use case, class (DB), sequence (×2), deployment, layered architecture. Tools: PlantUML or draw.io. | 6 diagrams | 2 days |
| 5 | **Write the core chapters first** (3, 4, 5). They are the substance; the introduction and conclusion are easier once they exist. | ~40 pages | 1 week |
| 6 | **Write the framing chapters** (1, 2, 6) and the abstract. | ~20 pages | 3 days |
| 7 | **Assemble.** Table of contents, list of figures, list of tables, bibliography, appendices. | full document | 1 day |
| 8 | **Proofread against the code.** Open the report and the repository side by side; verify every figure, filename and endpoint still matches. | corrected document | half a day |

### Commands that produce citable evidence

```bash
# test suite (the count must match what the UI advertises)
export PYTHONPATH=backend && python3 -m pytest backend/tests -q

# static analysis
python3 -m ruff check backend/app backend/tests

# installation / integration self-check
python3 scripts/check_install.py

# risk-engine measurements: separation, monotonicity, period sensitivity
python3 scripts/diag_risk.py
```

### A note on honesty (this will earn marks)

The project's stated engineering principle is that unfavourable results are
reported, not hidden. Keep that in the report:

* out-of-sample R² on returns is near zero or negative — that is expected;
* **no VaR estimator passes the Christoffersen independence test** on this
  data — documented as a limitation, not silently dropped;
* RL agents that lose to Buy & Hold are shown losing;
* Isolation Forest labels 2% of *any* window by construction
  (`contamination=0.02`) — a design choice, stated as such.

A chapter that admits limits is stronger than one that claims perfection.

---

# PART B — Report structure

**Target length: 60–80 pages** excluding appendices.

---

## Front matter

| Element | Notes |
|---|---|
| Title page | *KorisQuant AI — An Explainable Machine-Learning Platform for Financial Analysis and Portfolio Management* |
| Dedication / Acknowledgements | 1 page |
| Abstract (English) | 250 words: problem, approach, results, contribution |
| Résumé (French) | same, translated |
| Table of contents | auto-generated |
| List of figures / List of tables | auto-generated |
| List of abbreviations | ML, DL, RL, XAI, VaR, CVaR, GARCH, EVT, SHAP, LIME, API, JWT, ACI, POT |

---

## Chapter 1 — General Introduction (5–6 pages)

### 1.1 Context and motivation
Retail investors face tools that are either black boxes or spreadsheets.
Financial ML has a credibility problem: backtests routinely lie.

### 1.2 Problem statement
Three concrete problems this project attacks:
1. **Opacity** — a "BUY" with no reason is unusable.
2. **Dishonest evaluation** — data leakage and shuffled splits inflate results.
3. **Silent failure** — a system that shows a number without saying where it
   came from (live? cached? simulated?) is dangerous.

### 1.3 Objectives
Functional and non-functional, phrased so they can be checked at the end.

### 1.4 Methodology
Iterative/incremental development; test-driven where behaviour is verifiable;
every claim validated by measurement (mention the browser-automation checks).

### 1.5 Report structure
One paragraph per chapter.

---

## Chapter 2 — State of the Art (10–12 pages)

### 2.1 Financial time series and their statistical properties
Non-stationarity, volatility clustering, fat tails, leverage effect.
→ *Justifies why GARCH-family models and EVT are used later.*

### 2.2 Technical analysis
Trend, momentum, volatility and volume families. Cite the 17 indicators
implemented (`backend/app/services/indicators/technical.py`).

### 2.3 Machine and deep learning for forecasting
LSTM, GRU, TCN, Transformer, CNN-LSTM — strengths and weaknesses.
Emphasise: **return-space targets**, not price levels.

### 2.4 Reinforcement learning for trading
MDP formulation, value-based vs policy-based, distributional RL.
Why RL differs from forecasting: it learns from the *consequences* of actions.

### 2.5 Explainable AI
SHAP (Shapley values), LIME, permutation importance, counterfactuals.
Why XAI is not optional in finance (accountability, regulation).

### 2.6 Quantitative risk management
VaR and its critique (not sub-additive), CVaR/Expected Shortfall as the
coherent measure, backtesting (Kupiec, Christoffersen, Basel traffic light),
Extreme Value Theory, conformal prediction.

### 2.7 Comparative study of existing solutions
Table comparing Bloomberg Terminal, TradingView, QuantConnect, Yahoo Finance
and this project across: explainability, honest backtesting, cost, RL support,
data-source transparency.

### 2.8 Conclusion — the gap this project fills

---

## Chapter 3 — Analysis and Design (12–15 pages)

### 3.1 Functional requirements
Grouped by module: market data, technical analysis, forecasting, RL,
recommendation, risk, XAI, portfolio, alerts, AI assistant, authentication.

### 3.2 Non-functional requirements
Performance, security, reliability (never silently fail), explainability,
accessibility (**WCAG AA verified on both themes**), maintainability.

### 3.3 Actors and use cases
Actors: *Visitor*, *Authenticated user*, *Background scheduler*, *AI assistant*.
→ **Figure: use-case diagram.**

### 3.4 Detailed use-case descriptions
Three, in table form (actor, preconditions, nominal flow, alternative flows):
"Analyse an instrument", "Train and evaluate an RL agent", "Assess risk".

### 3.5 Global architecture
Five layers, dependencies pointing strictly downward:

```
PRESENTATION   frontend/ — 10 pages, vanilla JS + Plotly, no build step
API            app/api/v1/ — 106 route handlers, Pydantic-validated
DOMAIN         app/services/ — 11 service packages
PERSISTENCE    app/db/ — SQLAlchemy 2.0 async, SQLite → PostgreSQL-ready
INTEGRATION    providers, two-tier cache, deterministic synthetic engine
```

→ **Figure: layered architecture diagram.**
Justify: services never import from `api/`, so each is usable from a notebook.

### 3.6 Data model
9 tables: `users`, `portfolios`, `positions`, `transactions`,
`portfolio_snapshots`, `alerts`, `alert_rules`, `model_registry`,
`recommendation_log`.
→ **Figure: class/ER diagram.**
Mention additive-only migrations (`db/migrations.py`) — never destructive.

### 3.7 Dynamic behaviour
→ **Figure: sequence diagram — "risk scan"** (browser → API → period
resolution → data layer → detectors → profiler → response).
→ **Figure: sequence diagram — "AI assistant tool call"** (message → model →
tool selection → platform service → grounded answer).

### 3.8 Technology choices and justification
FastAPI (async, native OpenAPI), PyTorch, Stable-Baselines3, SQLAlchemy 2.0,
Plotly, vanilla JS (no build step = no toolchain rot), Ollama.

---

## Chapter 4 — Implementation (20–25 pages) ⭐ *the core chapter*

### 4.1 Development environment
Python 3.13, virtualenv (explain the PEP 668 / `externally-managed-environment`
constraint on Kali), project layout, `scripts/run_server.sh`, `scripts/fix_venv.sh`.

### 4.2 The hybrid data layer
Four resolution tiers, each labelled in the response:

| Tier | Source | Badge |
|---|---|---|
| 1 | Fresh cache (memory → parquet) | `CACHED` |
| 2 | Yahoo → Finnhub → Alpha Vantage → Polygon | `YAHOO` … |
| 3 | Stale cache | `STALE CACHE` |
| 4 | Deterministic synthetic engine | `SIMULATED` |

**Key argument:** the UI always shows *which tier answered*. A number whose
provenance is unknown is not a measurement.
Universe: **32 instruments** — 13 equities, 4 ETFs, 4 crypto, 4 forex,
4 indices, 3 commodities.

### 4.3 Feature engineering and technical indicators
17 indicators; describe 3 in depth with their formulas (RSI, MACD, ATR).

### 4.4 Deep-learning forecasting
5 architectures (`lstm`, `gru`, `tcn`, `transformer`, `cnn_lstm`).
Chronological splits, scalers fitted on training data only, persisted with the
checkpoint. Report **directional accuracy** alongside RMSE/R².
> Measured: **LSTM 64.2% directional accuracy; other architectures ≈53%.**

### 4.5 Reinforcement learning
13 algorithms; environment with transaction costs (10 bps), slippage and
drawdown penalties. Discrete vs continuous action spaces and how a weight
vector is mapped to BUY/HOLD/SELL (±15% dead band).
Benchmarks: Buy & Hold, SMA crossover, momentum, RSI, cash.

### 4.6 The recommendation engine
```
composite = Σ (signalᵢ.score × weightᵢ × reliabilityᵢ) / Σ (weightᵢ × reliabilityᵢ)
```
Risk overlay **neutralises** a bullish signal but never inverts it into a
short — high risk is a reason to step aside, not to bet the other way.

### 4.7 The risk engine ⭐ *give this the most space*

#### 4.7.1 Absolute per-asset metrics
Volatility, VaR₉₅/₉₉, CVaR₉₅/₉₉, max & current drawdown, downside deviation,
beta/alpha vs an asset-class benchmark, Sharpe, Sortino, skewness, kurtosis.
All computed from the selected symbol's own returns over the selected window
(`backend/app/services/risk/profile.py`).

#### 4.7.2 The Overall Risk Score
Weighted mean of **8 bounded contributors**, each normalised against an
**absolute** reference range so two assets are comparable:

| Contributor | Weight | Scored against |
|---|---|---|
| Volatility | 20% | 10% → 80% annualised |
| Crash Risk Score | 18% | 0 → 1 |
| Tail risk (CVaR₉₅) | 16% | 1.5% → 10% daily |
| Maximum drawdown | 14% | 10% → 60% |
| Bubble Indicator | 10% | 0 → 1 |
| Return distribution | 10% | negative skew + excess kurtosis |
| Market beta | 7% | 0.5 → 2.0 |
| Recent anomalies | 5% | high-severity hits in 31 sessions |

Bands: **low < 30% ≤ moderate < 50% ≤ high < 72% ≤ critical**.
An unmeasurable contributor is **dropped and its weight redistributed** —
never counted as zero, because zero means "measured, and safe".

#### 4.7.3 Explainability of the score
Every contribution is published: measured value, the scale it was judged
against, and points contributed out of its maximum. **The rows sum to the
headline, and a test asserts it.**
→ **Figure: screenshot of the Overall Risk breakdown.**

#### 4.7.4 Display window vs computation window
The architectural rule: *the window a user looks at and the window a model
computes over are two different things.* `model_bars(period, model)` returns
`max(display window, model floor)`, so a 1-month selection still gives crash
risk 61 bars and the bubble 201 — no dashes, no synthetic fallback.

### 4.8 Explainable AI
SHAP (TreeSHAP or sampled Shapley), LIME, permutation importance,
counterfactuals. → **Figure: SHAP screenshot.**

### 4.9 NLP and sentiment
News collection, FinBERT-compatible sentiment with a lexicon fallback,
12-category classification, market-impact scoring.

### 4.10 Portfolio management
Paper trading, P&L, mean-variance and risk-parity optimisation with a 40%
position cap, efficient frontier, rebalancing.

### 4.11 The AI assistant
Ollama (`gpt-oss:20b`), **15 read-only tools**. It has no market data in its
prompt: every figure must come back from a tool call.
Security: the API key is **server-side only**; a key shipped in frontend JS is
a public key. Rate-limited to 20 requests/min per IP.

### 4.12 Security and access control
bcrypt with a SHA-256 pre-hash (so bcrypt's 72-byte cap cannot silently
truncate a long passphrase), JWT in an **HttpOnly** cookie, and a
**default-deny** `AuthGuardMiddleware`: a new endpoint is protected unless
explicitly allow-listed, so forgetting to guard something fails closed.

### 4.13 User interface
10 pages, dark/light themes, every colour a CSS custom property declared twice
(a test forbids hard-coded colours). Both palettes measured against WCAG AA.
Global time-range selector shared by all pages.

---

## Chapter 5 — Testing, Validation and Results (12–15 pages) ⭐ *second most important*

### 5.1 Testing strategy
**598 automated tests** across 9 files, fully offline and deterministic
(`DATA_MODE=offline`, fixed synthetic seed). Static analysis with ruff
(line-length 110; rules E, F, W, I, UP, B, C4, SIM).

### 5.2 Test typology
Unit (indicator maths checked by hand), integration (every API route),
security (access control, XSS), UI (browser automation with Playwright),
non-regression.

### 5.3 Mutation testing — proving the tests actually work
> **This section distinguishes a good report from an average one.**

A passing test proves nothing unless it fails when the code breaks. Method:
revert a fix, confirm the matching test fails, restore, confirm it passes.
Present a table of the mutations performed on the risk engine (6 mutations,
all caught).

### 5.4 Empirical results — the risk engine

| Property | Before | After |
|---|---|---|
| Spearman(volatility, Overall Risk) | 0.76 | **0.976** |
| Distinct scores across 10 instruments | — | **10 / 10** |
| Monotone in volatility (fixed path) | no | **yes** (0.040 → 0.669) |
| Distinct answers across periods | **4 / 11** | **8 / 8** |

→ **Table: the cross-asset matrix from `scripts/diag_risk.py`** (EURUSD=X to
^VIX, with vol, VaR, CVaR, maxDD, beta, Sharpe, Sortino, Overall).

### 5.5 Empirical results — the other models
* **Conformal prediction:** ACI calibrated on 6/6 assets
  (GC=F: split 0.759 → ACI 0.896).
* **Volatility:** GJR-GARCH wins on AIC (4702), leverage coefficient 0.109.
* **Regime detection:** 5/5 synthetic scenarios and 6/6 real SPY episodes;
  the COVID crash detected at 99–100%.
* **Forecasting:** LSTM 64.2% directional accuracy.
* **VaR:** *no* estimator passes Christoffersen — **reported as a limitation.**

### 5.6 Bugs found and fixed — a catalogue
Choose the 8–10 most instructive. Present each as *symptom → root cause →
fix → the test that now prevents it*. Strong candidates:

1. **Data leakage in RL** — `_split` did `df.iloc[split - 60:]`, giving the
   test set 60 bars of training data. ~28% of the test window contaminated;
   8 already-trained agents were purged.
2. **Sortino divided by the wrong quantity** — `std()` of losses measures how
   much losses *vary*, not how *large* they are. A series losing exactly 2%
   every losing day scored **0.0 when the true Sortino is 12.401**.
3. **The Overall Risk Score ignored absolute risk** — NVDA at 36.4%
   volatility scored `low` while GLD at 28.2% scored `high`.
4. **The period selector changed nothing** — 7 of 11 ranges returned the same
   answer (AAPL: crash 0.429, bubble 0.314 for all of them).
5. **`VaR` returned 0.0 on short data** — "this asset cannot lose money".
6. **Reflected XSS on `/auth.html?error=`** — exploited for real, then fixed
   (`innerHTML` → `textContent`).
7. **Account takeover** — `/auth/forgot-password` returned the reset token in
   its own HTTP response; the flow was removed entirely.
8. **Kurtosis counted twice** — `(kurt-3)/8` when pandas already returns
   excess kurtosis.
9. **Bubble false positive** — `resid_std or 1e-9` let a flat line score 0.35.

### 5.7 False positives in my own tests
> Also valuable to a jury: it shows critical distance.

* A test that "skipped" via a `PATH` without pytest — fixed with `sys.executable`.
* `test_regime_spells_never_print_an_absolute_date` **passed while the page
  printed "Apr 29, 2026"**: it asserted the presence of the buggy call itself.
* A synthetic "range" scenario that actually ended −24% — replaced with an
  Ornstein-Uhlenbeck process.

### 5.8 Performance
Suite runtime ≈ 4 min; response times per endpoint; CPU-only inference.

---

## Chapter 6 — Conclusion and Perspectives (4–5 pages)

### 6.1 Summary of the work
### 6.2 Objectives achieved
Map back to §1.3, one line each.

### 6.3 Limitations (be explicit)
* SMTP not configured — verification links are logged, not emailed.
* Missing before public deployment: MFA, refresh-token rotation, lockout after
  repeated failures, and `SECRET_KEY` must be changed or every session token
  is forgeable.
* No self-service password reset (deliberately removed).
* No VaR estimator passes Christoffersen.
* Responsive layout below 720 px not verified.
* Anchor ranges in the risk score are calibration choices, not measurements.

### 6.4 Skills acquired
### 6.5 Future work
Live broker integration, MFA, PostgreSQL + Redis in production, multi-asset RL,
transformer-based sentiment, mobile layout, SMTP.

---

## Bibliography
Aim for 25–35 references: Hochreiter & Schmidhuber (LSTM), Vaswani et al.
(Attention), Bai et al. (TCN), Mnih et al. (DQN), Schulman et al. (PPO/TRPO),
Bellemare et al. (C51), Lundberg & Lee (SHAP), Ribeiro et al. (LIME),
Artzner et al. (coherent risk measures), Christoffersen (1998), Kupiec (1995),
Bollerslev (GARCH), Glosten-Jagannathan-Runkle (GJR), Gibbs & Candès (ACI),
Basel Committee documents.

## Appendices
A. Installation and deployment guide (from `QUICKSTART.md`)
B. API endpoint reference (106 handlers)
C. Full test output
D. `scripts/diag_risk.py` output
E. Additional screenshots

---

# Verified project metrics

*Use these; do not round them upward.*

| Metric | Value |
|---|---|
| Backend Python | 17,189 lines |
| Test code | 6,443 lines |
| Frontend JS | 6,218 lines |
| Frontend CSS | 2,680 lines |
| Frontend HTML | 1,997 lines |
| **Total** | **≈ 34,500 lines** |
| Automated tests | **598**, all passing |
| Route handlers | 106 |
| Service packages | 11 |
| Database tables | 9 |
| HTML pages | 10 (8 dashboard + landing + auth) |
| Instruments | 32 |
| Technical indicators | 17 |
| DL architectures | 5 |
| RL algorithms | 13 |
| AI-assistant tools | 15 (read-only) |

---

# Figures to prepare

| # | Figure | Source |
|---|---|---|
| 1 | Use-case diagram | to draw |
| 2 | Layered architecture | to draw |
| 3 | Class / ER diagram (9 tables) | to draw |
| 4 | Sequence: risk scan | to draw |
| 5 | Sequence: AI assistant tool call | to draw |
| 6 | Deployment diagram | to draw |
| 7 | Data-layer decision tree (4 tiers) | to draw |
| 8–15 | Screenshots of the 8 dashboard pages | capture |
| 16 | Overall Risk breakdown (close-up) | `shots/risk_breakdown_light.png` |
| 17 | Risk page, full | `shots/risk_no_dates.png` |
| 18 | Volatility ladder / separation chart | plot from `diag_risk.py` |
