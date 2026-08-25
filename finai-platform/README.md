# KorisQuant AI  : Adaptive Explainable Deep Reinforcement Learning for Risk-Aware Portfolio Management

Intelligent Financial Analysis, Risk Management and Reinforcement-Learning Portfolio Platform

KorisQuant AI is an end-to-end quantitative finance platform designed for market analysis, financial forecasting, risk assessment, portfolio management, reinforcement-learning research, and explainable decision support.

The platform combines Machine Learning, Deep Learning, Reinforcement Learning, Natural Language Processing, quantitative risk modelling, and Explainable AI (XAI) in a unified architecture.

Its objective is not to provide deterministic investment predictions. Instead, the system evaluates market conditions, estimates risk, combines independent model signals, and produces transparent, reproducible, and auditable decision-support outputs.

Important: KorisQuant AI is a research and educational platform. It is not investment advice.

## Project Structure

The main project structure is:

<finai-platform>/
├── backend/
│   ├── app/
│   │   ├── core/              # configuration, logging, exceptions
│   │   ├── db/                # database models and sessions
│   │   ├── schemas/           # Pydantic request/response schemas
│   │   ├── api/
│   │   │   └── v1/            # REST API routes
│   │   ├── services/
│   │   │   ├── data/          # providers, cache, synthetic engine
│   │   │   ├── indicators/    # technical indicators and features
│   │   │   ├── forecasting/   # DL architectures and training
│   │   │   ├── rl/            # RL environments and agents
│   │   │   ├── nlp/           # sentiment and news
│   │   │   ├── risk/          # risk and anomaly analysis
│   │   │   ├── xai/           # SHAP, LIME, counterfactuals
│   │   │   ├── recommendation/# signal fusion
│   │   │   ├── chat/          # AI assistant
│   │   │   └── alerts/        # alert engine
│   │   ├── workers/            # background jobs
│   │   └── utils/              # time-series and serialization utilities
│   └── tests/                  # automated tests
│
├── frontend/
│   ├── index.html
│   ├── analysis.html
│   ├── forecast.html
│   ├── rl.html
│   ├── signals.html
│   ├── stress.html
│   ├── xai.html
│   ├── portfolio.html
│   ├── training.html
│   ├── hyperparams.html
│   └── risk.html
│
├── configs/
│   ├── defaults.yaml
│   ├── algorithms/
│   └── profiles/
│
├── scripts/
│   └── run_server.sh
│
├── infra/
│   └── docker/
│
├── docs/
│   └── ARCHITECTURE.md
│
├── requirements.txt
├── .env.example
├── docker-compose.yml
└── README.md

## Quick start

```bash
# Run these from the PROJECT ROOT (the folder containing README.md)
cd /path/to/the/project or open project with vscode 

# create a Virtual Environment

## Linux / macOS

python3 -m venv .venv
source .venv/bin/activate

## Windows

python -m venv .venv
.venv\Scripts\activate

# Verify the environment:

python --version
pip --version

### using a virtual environment is particularly important on Debian, Ubuntu, and Kali Linux systems because of PEP 668 restrictions on system-managed Python environments.

# 2 — install (CPU-only torch keeps this small and fast)
pip install torch --index-url https://download.pytorch.org/whl/cpu (optional)
# next
pip install -r requirements.txt 

or 
 
python -m pip install -r requirements.txt

# optional: unlocks QR-DQN and TRPO
pip install sb3-contrib

# 3 — run
bash scripts/run_server.sh start
#   ...or plain uvicorn:  cd backend && uvicorn app.main:app --reload

# 3 — open
 http://localhost:8000        

#   API reference is hidden by default; enable with EXPOSE_API_DOCS=true
```

Stop with `bash scripts/run_server.sh stop`. See **[QUICKSTART.md](QUICKSTART.md)** for
Docker, notebook usage, a guided first-5-minutes walkthrough and troubleshooting.

With Docker:

```bash
docker compose up --build                 # API + dashboard
docker compose --profile full up --build  # + PostgreSQL + Redis
```

No API keys are required — Yahoo Finance is used by default and the platform falls back to a
deterministic synthetic market engine when the network is unavailable

## The server script also supports:

bash scripts/run_server.sh stop
bash scripts/run_server.sh restart
bash scripts/run_server.sh logs





# Key Capabilities

KorisQuant AI integrates several complementary analytical engines.

### Deep Learning Forecasting

Supported architectures:

LSTM

GRU

TCN

Transformer

CNN-LSTM

# The forecasting module operates on financial return data and supports training, prediction, architecture comparison, and walk-forward validation.

Reinforcement Learning

The platform supports the following RL algorithms:

DQN

Double DQN

Dueling DQN

C51

IQN

Rainbow

QR-DQN

PPO

A2C

SAC

TD3

DDPG

TRPO

Discrete agents generate:

BUY / HOLD / SELL

Continuous-control algorithms such as SAC, TD3, and DDPG generate exposure or allocation weights.

Technical Analysis

The technical engine provides 17 indicators, including:

SMA

EMA

WMA

RSI

MACD

Bollinger Bands

ATR

ADX

Stochastic

CCI

Williams %R

OBV

VWAP

MFI

Keltner Channels

Ichimoku

Historical Volatility

NLP and Sentiment

The NLP layer provides:

financial-news collection;

FinBERT-compatible sentiment analysis;

lexicon-based fallback;

12-category news classification;

market-impact scoring.

Risk Management

The risk engine evaluates:

volatility;

VaR;

CVaR;

drawdown;

downside deviation;

beta;

Sharpe ratio;

Sortino ratio;

skewness;

kurtosis;

crash risk;

bubble risk;

market anomalies.

Explainable AI

The platform supports:

SHAP;

LIME;

permutation importance;

counterfactual analysis;

regime-influence analysis.

Portfolio Management

The portfolio module supports:

paper trading;

P&L tracking;

mean-variance optimization;

risk-parity allocation;

efficient-frontier analysis;

portfolio rebalancing;

position constraints.

# AI Assistant

The integrated AI Assistant uses Ollama and accesses platform information through controlled read-only tools.

The assistant is designed so that numerical market information comes from platform services rather than being invented by the language model.

Architecture

KorisQuant AI follows a modular architecture with five principal layers.

┌──────────────────────────────────────────────────────────────────────┐
│                         KORISQUANT AI                                │
├──────────────────────────────────────────────────────────────────────┤
│                         APPLICATION LAYER                            │
│ Dashboard · Portfolio · Risk · Forecast · RL · XAI · Alerts · Chat  │
├──────────────────────────────────────────────────────────────────────┤
│                          DECISION LAYER                              │
│ Recommendation Engine · RL Agents · Portfolio Allocation            │
├──────────────────────────────────────────────────────────────────────┤
│                    ANALYTICS / INTELLIGENCE LAYER                    │
│ Forecasting · Technical Analysis · NLP · Regime Detection · Risk    │
├──────────────────────────────────────────────────────────────────────┤
│                      RISK & EXPLAINABILITY                           │
│ SHAP · LIME · Counterfactuals · Risk Overlay · Position Sizing     │
├──────────────────────────────────────────────────────────────────────┤
│                           DATA LAYER                                 │
│ Live Providers · Cache · Synthetic Engine · Data Validation         │
└──────────────────────────────────────────────────────────────────────┘

The backend is implemented with FastAPI. The frontend uses HTML, JavaScript, and Plotly.

The project currently exposes 61 REST API routes and separates the main services into dedicated modules.


# Market Data Layer

The data layer resolves market requests through a controlled hierarchy:

1. Fresh cache
       ↓
2. Live provider chain
       ↓
3. Stale cache
       ↓
4. Deterministic synthetic engine

The platform explicitly reports the source of the returned data.

Possible source indicators include:

CACHED
YAHOO
STALE CACHE
SIMULATED

Synthetic data are never presented as real market observations.

The synthetic engine is designed for deterministic and reproducible experiments and uses statistical mechanisms such as volatility, fat tails, and jumps.

## Forecasting

The forecasting engine supports:

LSTM
GRU
TCN
Transformer
CNN-LSTM

Main API operations include:

/forecast/train
/forecast/predict
/forecast/compare

The forecasting framework uses financial returns rather than raw prices as the primary prediction target.

The evaluation process includes:

chronological splits;

training-only preprocessing;

directional accuracy;

RMSE;

R²;

walk-forward validation.

The platform does not assume that a high R² is necessary for useful financial forecasting. Out-of-sample return prediction can legitimately produce R² values close to zero or below zero.

## Reinforcement Learning

The RL engine supports both discrete and continuous action spaces.

Discrete Agents

DQN-family agents operate on a single instrument and generate:

BUY
HOLD
SELL

Supported native discrete algorithms include:

DQN
Double DQN
Dueling DQN
C51
IQN
Rainbow
QR-DQN

Continuous Agents

The following algorithms generate exposure or allocation weights:

PPO
A2C
SAC
TD3
DDPG
TRPO

Continuous agents can be used for:

single-asset exposure;

multi-asset portfolio allocation.

#  RL Training

Single-asset training is exposed through:

/rl/train

Multi-asset portfolio training is exposed through:

/rl/portfolio/train

The environment incorporates realistic trading constraints including transaction costs, slippage, drawdown penalties, volatility, and risk-related reward components.

## Market-Regime-Aware RL

The platform detects seven market regimes.

Regime information is incorporated into the RL observation through:

regime risk;

directional bias;

classifier confidence;

volatility ratio;

crash probability;

current drawdown.

For multi-asset portfolios, each asset receives its own regime track.

This is important because different assets can simultaneously operate under different market conditions.

The reward function additionally incorporates regime-dependent risk aversion, volatility, drawdown, turnover, and CVaR.

## Active Adaptation with Mixture-of-Experts

The platform includes an optional regime-aware Mixture-of-Experts (MoE) mechanism.

The architecture contains three specialized experts:

Bull
Bear
Stress

A regime router selects the appropriate expert according to the detected market state.

When sufficient historical observations exist, the incoming expert can be fine-tuned using data from its own regime, while respecting temporal causality.

The implementation does not interpret improved backtest performance as proof of a persistent trading edge. Its principal research objective is to evaluate adaptive behaviour and adaptation latency.

## Hyperparameter Management

Training parameters are stored in YAML configuration files.

configs/
├── defaults.yaml
├── algorithms/
└── profiles/

Configuration resolution follows:

defaults
   ↓
algorithm
   ↓
profile
   ↓
request overrides

Available high-level profiles include:

Conservative

Balanced

High Performance

Risk-Aware

AI Recommended

The configuration system performs:

bounds validation;

cross-field validation;

experiment identification;

configuration fingerprinting;

seed recording;

complete resolved-configuration storage.

This ensures that experiments remain reproducible.

## Training Intelligence

The Training Intelligence module provides:

convergence diagnosis;

overfitting detection;

instability detection;

training/evaluation comparison;

health scoring;

experiment leaderboards;

checkpoint history;

experiment reports.

Available metrics can include:

reward;

evaluation reward;

loss;

Sharpe;

Sortino;

maximum drawdown;

volatility;

VaR;

CVaR;

portfolio value.

Evaluation is performed on held-out data rather than using the training set as the primary performance reference.


## Checkpoints

The Checkpoint Manager records:

creation time;

training step;

episode;

algorithm;

model version;

experiment ID;

random seed;

configuration profile.

Checkpoint operations include:

comparison;

restoration;

deletion;

history inspection.

A restored checkpoint is explicitly identified because its original evaluation results may no longer describe the currently loaded model.


##  Recommendation Engine

The recommendation engine combines four signal families:

Deep Learning
      +
Reinforcement Learning
      +
Technical Analysis
      +
NLP / Sentiment

Each signal is weighted according to configured importance and measured reliability.

The composite score is:

Composite Score =
Σ(signal_score × weight × reliability)
/
Σ(weight × reliability)

A risk overlay is then applied.

Elevated crash or bubble risk can neutralize a bullish signal, but the risk layer does not automatically transform a bullish signal into a short position.

The final recommendation can include:

action;

confidence;

signal contribution;

model reliability;

risk score;

position size;

stop-loss;

take-profit;

horizon;

natural-language explanation;

SHAP attribution.

## Risk Management

Risk metrics are computed per asset and period.

The platform evaluates:

Volatility
VaR
CVaR
Maximum Drawdown
Downside Deviation
Beta
Sharpe
Sortino
Skewness
Kurtosis
Crash Risk
Bubble Risk
Anomalies

The Overall Risk Score is based on eight normalized contributors.

The score is designed as a quantitative measurement rather than a simple categorical label.

Missing measurements are not silently treated as zero. Instead, unavailable contributors are removed and their weights redistributed.

## Strategy Benchmarks

RL strategies are evaluated against conventional strategies under consistent transaction-cost assumptions.

Benchmarks include:

Buy & Hold
Moving-Average Crossover
Momentum
RSI
Cash

This comparison is essential because an RL strategy should not be considered successful merely because it produces positive returns. Its performance must be evaluated relative to appropriate baselines and on unseen data.

##  Portfolio Management

The portfolio module supports:

holdings;

equity curve;

P&L;

allocation;

rebalancing;

mean-variance optimization;

risk-parity optimization;

efficient frontier;

position constraints.

Continuous RL algorithms can directly generate portfolio weights for multi-asset allocation.

## AI Stress Testing

The dedicated stress-testing module supports scenarios such as:

Market Crash
-10% Shock
-20% Shock
Volatility ×2
Liquidity Shock
Correlation Spike
Custom Scenario

The system can compare before/after:

VaR;

CVaR;

volatility;

drawdown;

portfolio loss;

asset-level impact;

Euler risk contribution.

An AI-generated explanation summarizes the observed vulnerabilities and possible mitigation considerations.

## Explainable AI

The XAI layer supports:

SHAP
LIME
Permutation Importance
Counterfactual Analysis

The purpose is to explain why a model or recommendation produced a particular output.

For regime-aware RL, counterfactual analysis can remove or neutralize regime information and re-evaluate the decision.

This allows the platform to distinguish between:

Regime information materially influenced the decision

and:

Regime information had negligible influence

rather than automatically claiming that regime information affected every decision.

## AI Direction Prediction

KorisQuant AI includes a dedicated AI Direction Prediction module.

The purpose of this component is to estimate expected market direction over a selected horizon using available forecasting and signal-generation models.

The module follows an important principle:

 ## No unsupported numerical prediction is generated when the required trained model is unavailable.

When no trained forecaster exists for a symbol, the system reports that limitation and identifies the corresponding training workflow.

Predicted movement must therefore be interpreted as a model-derived estimate, not a guaranteed future return.

## NLP and Market Sentiment

The NLP module collects and processes financial news.

It supports:

sentiment classification;

news categorization;

market-impact estimation;

FinBERT-compatible processing;

deterministic lexicon fallback.

The resulting sentiment signal is integrated into the broader recommendation engine rather than being treated as an independent trading decision

## Alerts

The alert engine can monitor:

price changes;

volatility spikes;

signal changes;

risk escalation;

high-impact news;

custom conditions.

Alerts are exposed through:

/alerts/scan
/alerts/rules

### AI Assistant

The AI Assistant runs through Ollama.

Local Ollama

Start the service:

ollama serve

Pull a compatible model:

ollama pull llama3.1

Configure:

OLLAMA_BASE_URL=http://localhost:11434/v1
OLLAMA_MODEL=llama3.1

Ollama Cloud

Alternatively configure:

OLLAMA_API_KEY=|||||||||||||||

The key remains server-side.

The browser communicates only with the KorisQuant AI backend. It does not directly receive the Ollama credential.

The assistant uses controlled read-only tools to access:

quotes;

indicators;

forecasts;

RL agents;

risk analysis;

sentiment;

backtests;

market regimes;

performance reports;

SHAP explanations.

If live data are unavailable, the assistant explicitly identifies simulated data.

If a ticker is invalid, the system does not use the synthetic engine to manufacture a plausible-looking quote.

If a model is not trained, the assistant reports the missing model instead of fabricating a prediction


### Testing

Run the complete backend test suite:

cd backend
pytest tests/ -v

Run linting:

ruff check app tests

The project reports 776 automated tests designed to run offline and deterministically.

The tests cover:

indicator correctness;

OHLC invariants;

time-series leakage;

forecasting construction;

TCN causality;

transaction costs;

VaR/CVaR ordering;

optimizer constraints;

JSON safety;

API routes;

AI assistant tool integration;

model availability handling;

regime-aware behaviour;

MoE functionality.

The test suite has previously identified and helped correct issues such as RSI edge cases, infinite Sharpe-ratio values, and configuration startup failures.

## Reproducibility

Reproducibility is a core engineering requirement.

Each experiment records:

experiment ID;

algorithm;

profile;

fully resolved hyperparameters;

random seed;

configuration fingerprint;

contributing configuration files;

model version.

The complete resolved configuration is stored rather than only recording parameter differences.

This prevents later configuration changes from altering the interpretation of historical experiments.

## Research Methodology

KorisQuant AI applies several principles designed to reduce common errors in financial machine learning.

Chronological Evaluation

Financial observations are ordered in time and are not randomly shuffled across training and evaluation periods.

No Temporal Leakage

The test suite verifies that training and evaluation windows are disjoint.

Training-Only Preprocessing

Scalers are fitted on training data and then persisted with the model checkpoint.

Return-Based Targets

Forecasting models use return-space targets rather than raw prices to improve stationarity and comparability.

Out-of-Sample Evaluation

RL agents are evaluated on unseen data.

Benchmark Comparison

Results are compared against standard strategies.

Honest Reporting

Underperformance is reported rather than hidden.

#### Limitations

The platform has several important limitations.

# Market Non-Stationarity

Financial relationships change over time. A model that performs well under one market regime can fail under another.

# Forecasting Limitations

Out-of-sample R² for financial returns can be close to zero or negative.

# RL Limitations

RL strategies can underperform Buy & Hold, particularly after transaction costs.

# Backtesting Limitations

The current framework does not fully model:

market impact;

borrowing costs;

taxation;

survivorship bias.

# Synthetic Data

Synthetic observations are statistical simulations and are not substitutes for real market data.

# Research Scope

The available results should be interpreted as experimental evidence under specific datasets, assets, algorithms, and evaluation periods—not as proof of a universal trading advantage.

## API Overview

Major API groups include:

Market
/market/history
/market/quote
/market/correlation

Technical Analysis
/market/indicators/{symbol}

Forecasting
/forecast/train
/forecast/predict
/forecast/compare
/api/v1/forecast/backtest

Reinforcement Learning
/rl/train
/rl/portfolio/train
/rl/action/{symbol}
/rl/allocation
/rl/decisions

Risk
/risk/scan
/risk/crash
/risk/bubble
/risk/profile/{symbol}

Quantitative Finance
/quant/conformal
/quant/var
/quant/volatility
/quant/regime

Explainable AI
/xai/explain
/xai/counterfactual

Portfolio
/portfolio
/portfolio/optimise

Alerts
/alerts/scan
/alerts/rules

AI Assistant
/chat
/chat/health
/chat/tools


### Conclusion

KorisQuant AI is a modular quantitative-finance research platform that brings together market data engineering, Deep Learning, Reinforcement Learning, quantitative risk management, portfolio optimization, NLP, and Explainable AI.

Its architecture is designed not only to generate financial signals, but to provide a complete environment in which those signals can be:

generated;

evaluated;

benchmarked;

risk-adjusted;

explained;

monitored;

reproduced.

The central engineering principle is transparency over performance claims.

The platform therefore treats data provenance, temporal integrity, reproducibility, risk measurement, explainability, and honest reporting as first-class components of the system. 

## License

MIT License — see LICENSE

