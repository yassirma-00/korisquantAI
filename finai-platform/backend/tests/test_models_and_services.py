"""Tests for forecasting, RL, NLP, risk, XAI and the recommendation engine."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from app.services.forecasting.models import MODEL_REGISTRY, build_model, count_parameters
from app.services.forecasting.trainer import ForecastTrainer, TrainConfig
from app.services.nlp.news import classify, news_service
from app.services.nlp.sentiment import sentiment_analyzer
from app.services.risk.anomaly import anomaly_detector
from app.services.risk.metrics import (
    conditional_var,
    full_metrics,
    optimise_portfolio,
    sharpe_ratio,
    sortino_ratio,
    value_at_risk,
)
from app.services.rl.agents.dqn import DQNAgent, DQNConfig, ReplayBuffer
from app.services.rl.environment import EnvConfig, PortfolioEnv, TradingEnv
from app.services.xai.explainer import explainer


# ---------------------------------------------------------------- DL models
@pytest.mark.parametrize("name", sorted(MODEL_REGISTRY))
def test_every_architecture_forward_pass(name):
    model = build_model(name, n_features=12)
    out = model(torch.randn(4, 30, 12))
    assert out.shape == (4, 1)
    assert torch.isfinite(out).all()
    assert count_parameters(model) > 0


def test_build_model_rejects_unknown():
    with pytest.raises(ValueError, match="Unknown model"):
        build_model("not_a_model", n_features=5)


def test_tcn_is_causal():
    """No future timestep may influence an earlier position in the TCN feature map.

    We inspect the convolutional stack directly (before the final pooling head),
    because the model output is taken at the last timestep, where every input is
    legitimately in the past.
    """
    model = build_model("tcn", n_features=4).eval()
    x = torch.randn(1, 40, 4)
    t = 20
    with torch.no_grad():
        base = model.tcn(x.transpose(1, 2))[:, :, t]
        future_perturbed = x.clone()
        future_perturbed[0, t + 1:] += 100.0        # only the FUTURE is changed
        after = model.tcn(future_perturbed.transpose(1, 2))[:, :, t]
    assert torch.allclose(base, after, atol=1e-5), "output leaked information from the future"


def test_tcn_uses_recent_past():
    """Within its receptive field, recent history must actually matter."""
    model = build_model("tcn", n_features=4).eval()
    x = torch.randn(1, 40, 4)
    with torch.no_grad():
        base = model(x)
        recent = x.clone()
        recent[0, -1] += 100.0
        assert not torch.allclose(base, model(recent))


def test_train_and_predict_roundtrip(ohlcv, tmp_path):
    trainer = ForecastTrainer(model_dir=tmp_path)
    cfg = TrainConfig(model="gru", epochs=3, lookback=30, horizon=5)
    result = trainer.train("TEST", ohlcv, cfg)

    assert result.metrics["test"]["n_samples"] > 0
    assert 0 <= result.metrics["test"]["directional_accuracy"] <= 100
    assert result.n_parameters > 0
    assert trainer.is_trained("TEST", "gru", 5)

    prediction = trainer.predict("TEST", ohlcv, model_name="gru", horizon=5)
    assert prediction["direction"] in ("up", "down")
    assert 0 <= prediction["confidence"] <= 1
    assert len(prediction["forecast"]) == 5
    for point in prediction["forecast"]:
        assert point["lower"] <= point["price"] <= point["upper"]


def test_predict_without_checkpoint_raises(ohlcv, tmp_path):
    from app.core.exceptions import ModelNotTrainedError

    trainer = ForecastTrainer(model_dir=tmp_path)
    with pytest.raises(ModelNotTrainedError):
        trainer.predict("NEVER_TRAINED", ohlcv, model_name="lstm", horizon=5)


def test_training_rejects_short_series(short_ohlcv, tmp_path):
    from app.core.exceptions import InvalidRequestError

    trainer = ForecastTrainer(model_dir=tmp_path)
    with pytest.raises(InvalidRequestError):
        trainer.train("SHORT", short_ohlcv, TrainConfig(epochs=1, lookback=60))


# ------------------------------------------------------------------ RL env
def test_trading_env_reset_and_spaces(ohlcv):
    env = TradingEnv(ohlcv, EnvConfig(initial_balance=50_000))
    obs, _ = env.reset()
    assert obs.shape == env.observation_space.shape
    assert env.action_space.n == 3
    assert env.portfolio_value == pytest.approx(50_000)


def test_trading_env_buy_reduces_cash(ohlcv):
    env = TradingEnv(ohlcv, EnvConfig(initial_balance=100_000, trade_fraction=0.5))
    env.reset()
    cash_before = env.cash
    env.step(2)  # BUY
    assert env.cash < cash_before
    assert env.shares > 0


def test_trading_env_cannot_sell_without_position(ohlcv):
    env = TradingEnv(ohlcv, EnvConfig())
    env.reset()
    env.step(0)  # SELL with no holdings
    assert env.shares == pytest.approx(0.0)
    assert env.n_trades == 0


def test_transaction_costs_are_charged(ohlcv):
    free = TradingEnv(ohlcv, EnvConfig(transaction_cost=0.0, slippage=0.0))
    costly = TradingEnv(ohlcv, EnvConfig(transaction_cost=0.02, slippage=0.0))
    for env in (free, costly):
        env.reset()
        for _ in range(30):
            env.step(2)
            env.step(0)
    assert costly.portfolio_value < free.portfolio_value
    assert costly.total_cost > free.total_cost > -1e-9


def test_env_terminates_at_series_end(ohlcv):
    env = TradingEnv(ohlcv, EnvConfig())
    env.reset()
    done, steps = False, 0
    while not done and steps < 5000:
        _, _, terminated, truncated, _ = env.step(1)
        done = terminated or truncated
        steps += 1
    assert done and steps < 5000


def test_performance_report_keys(ohlcv):
    env = TradingEnv(ohlcv, EnvConfig())
    env.reset()
    for _ in range(60):
        env.step(np.random.randint(0, 3))
    perf = env.performance()
    for key in ("total_return", "sharpe_ratio", "max_drawdown", "buy_and_hold_return", "n_trades"):
        assert key in perf
    assert perf["max_drawdown"] <= 0


def test_portfolio_env_weights_sum_to_one(ohlcv, crypto_ohlcv):
    import pandas as pd

    matrix = pd.DataFrame({
        "A": ohlcv["close"], "B": crypto_ohlcv["close"].reindex(ohlcv.index).ffill().bfill(),
    }).dropna()
    env = PortfolioEnv(matrix, EnvConfig())
    env.reset()
    _, _, _, _, info = env.step(np.array([1.0, -0.5, 0.2]))
    assert sum(info["weights"].values()) == pytest.approx(1.0, abs=1e-3)
    assert all(w >= 0 for w in info["weights"].values()), "long-only projection"


# ---------------------------------------------------------------- RL agent
def test_replay_buffer_wraps():
    buf = ReplayBuffer(capacity=5, obs_dim=3)
    for i in range(12):
        buf.add(np.ones(3) * i, i % 3, float(i), np.ones(3), False)
    assert len(buf) == 5
    obs, actions, rewards, next_obs, dones = buf.sample(3)
    assert obs.shape == (3, 3) and actions.shape == (3,)


def test_dqn_learns_and_persists(ohlcv, tmp_path):
    env = TradingEnv(ohlcv.tail(320), EnvConfig())
    cfg = DQNConfig(min_buffer=64, batch_size=32, epsilon_decay_steps=200)
    agent = DQNAgent(env.observation_space.shape[0], env.action_space.n, cfg)
    history = agent.train(env, episodes=2, log_every=10)

    assert len(history["episode_rewards"]) == 2
    assert agent.steps > 0 and agent.grad_steps > 0
    assert agent.epsilon() < cfg.epsilon_start

    path = tmp_path / "agent.pt"
    agent.save(path)
    reloaded = DQNAgent.load(path)
    obs, _ = env.reset()
    assert np.allclose(agent.q_values(obs), reloaded.q_values(obs), atol=1e-6)


def test_dueling_and_double_flags_build():
    for double in (True, False):
        for dueling in (True, False):
            agent = DQNAgent(10, 3, DQNConfig(double=double, dueling=dueling))
            assert agent.q_values(np.zeros(10)).shape == (3,)


def test_agent_evaluation_is_deterministic(ohlcv):
    env = TradingEnv(ohlcv.tail(220), EnvConfig())
    agent = DQNAgent(env.observation_space.shape[0], 3, DQNConfig())
    first = agent.evaluate(env, deterministic=True)["performance"]
    second = agent.evaluate(env, deterministic=True)["performance"]
    assert first == second


# --------------------------------------------------------------------- NLP
@pytest.mark.parametrize("text,expected", [
    ("Company beats earnings estimates, profits surge to record highs", "positive"),
    ("Shares plunge after bankruptcy warning and massive losses", "negative"),
    ("The company will publish its quarterly report on Tuesday", "neutral"),
])
def test_sentiment_direction(text, expected):
    result = sentiment_analyzer.analyze(text)
    assert result.label == expected
    assert -1 <= result.score <= 1
    assert 0 <= result.confidence <= 1


def test_sentiment_handles_negation():
    positive = sentiment_analyzer.analyze("profits surged strongly").score
    negated = sentiment_analyzer.analyze("profits did not surge").score
    assert negated < positive


def test_empty_text_is_neutral():
    result = sentiment_analyzer.analyze("")
    assert result.label == "neutral" and result.score == 0.0


def test_sentiment_aggregate():
    results = sentiment_analyzer.analyze_batch([
        "record profits and strong growth", "massive losses and bankruptcy risk", "report published",
    ])
    agg = sentiment_analyzer.aggregate(results)
    assert agg["n"] == 3
    assert sum(agg["distribution"].values()) == 3


@pytest.mark.parametrize("text,category", [
    ("Quarterly earnings beat revenue estimates", "earnings"),
    ("The Federal Reserve raised interest rates", "monetary_policy"),
    ("Analysts upgrade the stock with a new price target", "analyst_rating"),
    ("Bitcoin and crypto markets rally on blockchain adoption", "crypto"),
])
def test_news_classification(text, category):
    assert classify(text) == category


def test_news_service_offline_fallback():
    items = news_service.get_news("AAPL", limit=6)
    assert len(items) == 6
    for item in items:
        assert item["title"] and item["sentiment"]["label"] in ("positive", "negative", "neutral")
        assert 0 <= item["impact_score"] <= 1


def test_sentiment_summary_shape():
    summary = news_service.sentiment_summary("AAPL", limit=8)
    assert summary["n"] > 0
    assert summary["label"] in ("positive", "negative", "neutral")
    assert "top_impact_news" in summary


# -------------------------------------------------------------------- risk
def test_sharpe_of_constant_returns_is_high():
    import pandas as pd

    steady = pd.Series([0.001] * 252)
    assert sharpe_ratio(steady) > 5


def test_sharpe_of_zero_volatility_is_finite():
    """Zero variance must never yield inf/NaN — it would break scoring and JSON."""
    import pandas as pd

    # Flat 0% returns sit *below* the risk-free rate -> negative excess, finite sentinel
    flat = sharpe_ratio(pd.Series([0.0] * 100))
    assert np.isfinite(flat) and flat < 0

    # Returns exactly at the risk-free rate -> no excess at all -> zero
    rf_daily = 0.02 / 252
    assert sharpe_ratio(pd.Series([rf_daily] * 100)) == 0.0

    # Constant positive excess -> large but finite
    steady = sharpe_ratio(pd.Series([0.001] * 100))
    assert np.isfinite(steady) and steady > 5


def test_var_ordering():
    import pandas as pd

    returns = pd.Series(np.random.default_rng(0).normal(0, 0.02, 2000))
    var95 = value_at_risk(returns, 0.95)
    var99 = value_at_risk(returns, 0.99)
    cvar95 = conditional_var(returns, 0.95)
    assert var99 < var95 < 0
    assert cvar95 <= var95, "CVaR must be at least as severe as VaR"


def test_sortino_ignores_upside_volatility():
    import pandas as pd

    upside = pd.Series([0.05, 0.06, 0.04, 0.05] * 50)
    mixed = pd.Series([0.05, -0.06, 0.04, -0.05] * 50)
    assert sortino_ratio(upside) > sortino_ratio(mixed)


def test_full_metrics_completeness(ohlcv):
    returns = ohlcv["close"].pct_change().dropna()
    metrics = full_metrics(returns)
    for key in ("sharpe_ratio", "max_drawdown", "var_95", "win_rate", "skewness"):
        assert key in metrics
    assert -1 <= metrics["max_drawdown"] <= 0
    assert 0 <= metrics["win_rate"] <= 1


def test_optimiser_respects_constraints(ohlcv, crypto_ohlcv):
    import pandas as pd

    returns = pd.DataFrame({
        "A": ohlcv["close"].pct_change(),
        "B": crypto_ohlcv["close"].reindex(ohlcv.index).ffill().pct_change(),
        "C": (ohlcv["close"] * 0.5 + 10).pct_change(),
    }).dropna()
    for objective in ("max_sharpe", "min_volatility", "risk_parity"):
        result = optimise_portfolio(returns, objective=objective, n_simulations=2000)
        weights = np.array(list(result["weights"].values()))
        assert weights.sum() == pytest.approx(1.0, abs=1e-3)
        assert (weights >= -1e-6).all(), "long-only by default"
        assert weights.max() <= 0.4 + 1e-6, "position cap enforced"


def test_min_volatility_beats_max_return_on_risk(ohlcv, crypto_ohlcv):
    import pandas as pd

    returns = pd.DataFrame({
        "A": ohlcv["close"].pct_change(),
        "B": crypto_ohlcv["close"].reindex(ohlcv.index).ffill().pct_change(),
    }).dropna()
    low = optimise_portfolio(returns, "min_volatility", n_simulations=2000)
    high = optimise_portfolio(returns, "max_return", n_simulations=2000)
    assert low["expected_annual_volatility"] <= high["expected_annual_volatility"] + 1e-6


def test_anomaly_scan_structure(ohlcv):
    scan = anomaly_detector.scan("TEST", ohlcv)
    assert scan["overall_risk_level"] in ("unknown", "low", "moderate", "high", "critical")
    assert isinstance(scan["anomalies"], list)
    assert 0 <= scan["crash_risk"]["crash_risk_score"] <= 1
    assert 0 <= scan["bubble"]["bubble_score"] <= 1


# ============================================ risk maths, checked by hand
def test_crash_score_does_not_subtract_three_from_excess_kurtosis(ohlcv):
    """pandas .kurtosis() is Fisher's definition — already excess, 0 for a
    normal. The score subtracted 3 from it anyway, so the whole tail-risk term
    stayed at zero for any series below excess kurtosis 3. A t(5) sample, which
    is emphatically fat-tailed, scored 0.12 on that term instead of 0.50."""
    import inspect

    from app.services.risk import anomaly

    source = inspect.getsource(anomaly.AnomalyDetector.crash_risk)
    assert "(kurt - 3)" not in source, "excess kurtosis is being reduced by 3 twice"

    # Behavioural check: fatter tails must raise the score, all else equal.
    calm = ohlcv.copy()
    fat = ohlcv.copy()
    close = fat["close"].to_numpy().copy()
    # Inject symmetric jumps: raises kurtosis without changing drift or skew.
    for i in range(60, len(close) - 1, 40):
        close[i] *= 1.06
        close[i + 1] *= 0.94
    fat["close"] = close
    calm_score = anomaly.anomaly_detector.crash_risk(calm)["crash_risk_score"]
    fat_score = anomaly.anomaly_detector.crash_risk(fat)["crash_risk_score"]
    assert fat_score > calm_score, (
        f"fat tails did not raise crash risk ({fat_score} vs {calm_score})")


def test_unavailable_scores_are_none_not_zero(short_ohlcv):
    """A score of 0.0 renders as a green 'no risk' bar. When the window is too
    short the honest answer is 'unknown', and the two must not look alike."""
    crash = anomaly_detector.crash_risk(short_ohlcv.head(40))
    bubble = anomaly_detector.bubble_indicator(short_ohlcv)

    assert crash["level"] == "insufficient_data"
    assert crash["crash_risk_score"] is None, "0.0 reads as 'no crash risk'"
    assert bubble["level"] == "insufficient_data"
    assert bubble["bubble_score"] is None, "0.0 reads as 'no bubble'"
    # And they must say what is missing.
    assert crash["bars_required"] and bubble["bars_required"]


def test_overall_risk_ignores_stale_anomalies(ohlcv):
    """A months-old structural break used to sit in the 'five most recent'
    slice and pin the headline to HIGH while every current measure said
    otherwise. GLD showed exactly this: escalated by a 136-day-old event."""
    import pandas as pd

    quiet = ohlcv.copy()
    # A violent shock, then a long calm stretch after it.
    idx = len(quiet) - 400
    quiet.iloc[idx, quiet.columns.get_loc("close")] *= 0.80

    scan = anomaly_detector.scan("TEST", quiet, lookback_days=720)
    stale = [a for a in scan["anomalies"]
             if a["date"] < str((quiet.index[-1] - pd.Timedelta(days=60)).date())
             and a.get("severity") in ("high", "critical")]
    escalated_by_anomalies = any(
        d["source"] == "recent_anomalies" for d in scan["risk_drivers"])
    if stale and not any(
            a["date"] >= str((quiet.index[-1] - pd.Timedelta(days=31)).date())
            and a.get("severity") in ("high", "critical")
            for a in scan["anomalies"]):
        assert not escalated_by_anomalies, (
            "an anomaly older than the recency window still escalates the headline")


def test_the_headline_explains_itself(ohlcv):
    """Overall Risk aggregates three inputs and can exceed both scores. Without
    the reasons on screen that disagreement looks like a bug."""
    scan = anomaly_detector.scan("TEST", ohlcv)
    assert scan["risk_drivers"], "no explanation for the headline level"
    for driver in scan["risk_drivers"]:
        assert {"source", "level", "detail"} <= set(driver)


def test_lookback_days_are_days_not_bars(ohlcv):
    """`.tail(n)` counts trading bars; lookback_days counts calendar days.
    Conflating them meant a 220-bar floor governed every request, so 1y and 10y
    fed the detectors an identical slice and the page never changed."""
    narrow = anomaly_detector.scan("TEST", ohlcv, lookback_days=30)
    wide = anomaly_detector.scan("TEST", ohlcv, lookback_days=720)

    assert wide["bars_analysed"] > narrow["bars_analysed"], \
        "a longer lookback analysed no more data"
    assert wide["window_start"] < narrow["window_start"]

    # The decisive check. `max(lookback_days, 220)` clamps every request below
    # 220 to exactly the floor, so the bar count is flat across small lookbacks
    # — that flatness is the bug. Converting days to bars makes it strictly
    # increasing. Asserting only an upper bound passed under both formulas.
    counts = [anomaly_detector.scan("TEST", ohlcv, lookback_days=d)["bars_analysed"]
              for d in (30, 90, 180)]
    assert counts[0] < counts[1] < counts[2], (
        f"bar count is flat across lookbacks {counts} — days are being read as bars")


def test_injected_crash_is_detected(ohlcv):
    shocked = ohlcv.copy()
    shocked.iloc[-5:, shocked.columns.get_loc("close")] *= 0.72   # -28% collapse
    shocked.iloc[-5:, shocked.columns.get_loc("low")] *= 0.70
    outliers = anomaly_detector.return_outliers(shocked)
    assert outliers, "a 28% multi-day collapse must register as an outlier"


def test_volume_spike_detected(ohlcv):
    shocked = ohlcv.copy()
    shocked.iloc[-1, shocked.columns.get_loc("volume")] *= 25
    assert anomaly_detector.volume_anomalies(shocked)


# --------------------------------------------------------------------- XAI
def test_shap_contributions(ohlcv):
    explanation = explainer.shap_values("TEST", ohlcv, horizon=5, n_samples=25, top_k=6)
    assert len(explanation.feature_importance) == 6
    assert explanation.narrative
    for item in explanation.feature_importance:
        assert item["direction"] in ("bullish", "bearish")


def test_lime_contributions(ohlcv):
    explanation = explainer.lime_explain("TEST", ohlcv, horizon=5, n_samples=200, top_k=5)
    assert len(explanation.feature_importance) == 5
    assert "local_r2" in explanation.details


def test_global_importance_sorted(ohlcv):
    result = explainer.global_importance("TEST", ohlcv, horizon=5, top_k=8)
    scores = [d["importance"] for d in result["permutation_importance"]]
    assert scores == sorted(scores, reverse=True)
    assert result["narrative"]


def test_counterfactual_shape(ohlcv):
    result = explainer.counterfactual("TEST", ohlcv, horizon=5)
    assert result["original_direction"] in ("up", "down")
    assert isinstance(result["counterfactuals"], list)
    assert result["narrative"]


def test_market_pulse_filters_noise_from_directional_lists():
    """Regression: a score of -0.0031 rendered as 'SPY -0.00' in the bearish list.

    A value that rounds to zero on screen is noise, not a directional call.
    """
    pulse = news_service.market_pulse(["AAPL", "MSFT", "SPY"])
    for entry in pulse["most_bullish"]:
        assert entry["score"] >= 0.01, f"near-zero score shown as bullish: {entry}"
    for entry in pulse["most_bearish"]:
        assert entry["score"] <= -0.01, f"near-zero score shown as bearish: {entry}"
    # the two lists must stay disjoint
    bulls = {e["symbol"] for e in pulse["most_bullish"]}
    bears = {e["symbol"] for e in pulse["most_bearish"]}
    assert not (bulls & bears)


def test_a_flat_series_is_not_a_bubble():
    """`resid_std or 1e-9` only caught an exact zero. A series that hugs its
    trend leaves ~1e-16 of floating-point residue, and dividing by that turned
    rounding noise into a 4-sigma stretch: a flat line scored as a moderate
    bubble, and a noiseless exponential ramp scored 0.42."""
    import numpy as np
    import pandas as pd

    idx = pd.bdate_range("2020-01-01", periods=400)
    for label, series in (
        ("flat line", np.full(400, 100.0)),
        ("noiseless exponential", 100 * np.exp(0.0004 * np.arange(400))),
    ):
        frame = pd.DataFrame({"close": series}, index=idx)
        result = anomaly_detector.bubble_indicator(frame)
        assert result["trend_deviation_sigma"] == 0.0, (
            f"{label} produced a {result['trend_deviation_sigma']}σ stretch from noise")
        assert result["bubble_score"] < 0.4, (
            f"{label} scored {result['bubble_score']} — a trendless series is not a bubble")


def test_score_components_sum_to_the_published_score(ohlcv):
    """The breakdown shown to the user must reconstruct the headline number,
    otherwise 'how this is calculated' is decoration."""
    crash = anomaly_detector.crash_risk(ohlcv)
    total = sum(c["weight"] * c["value"] for c in crash["components"])
    assert abs(total - crash["crash_risk_score"]) < 0.002, (
        f"crash components sum to {total:.4f}, score says {crash['crash_risk_score']}")

    bubble = anomaly_detector.bubble_indicator(ohlcv)
    total_b = sum(c["weight"] * c["value"] for c in bubble["components"])
    assert abs(total_b - bubble["bubble_score"]) < 0.002, (
        f"bubble components sum to {total_b:.4f}, score says {bubble['bubble_score']}")

    # Weights must be a partition of 1.0, or "pts / 22" means nothing.
    assert abs(sum(c["weight"] for c in crash["components"]) - 1.0) < 1e-9
    assert abs(sum(c["weight"] for c in bubble["components"]) - 1.0) < 1e-9


def test_every_score_publishes_its_scale(ohlcv):
    """A bare 0.41 invites 'out of what?'. The bands are what make the number
    readable, and they must match the thresholds the code actually uses."""
    crash = anomaly_detector.crash_risk(ohlcv)
    assert crash["scale"]["min"] == 0.0 and crash["scale"]["max"] == 1.0
    assert set(crash["scale"]["bands"]) == {"low", "moderate", "high", "critical"}
    bubble = anomaly_detector.bubble_indicator(ohlcv)
    assert set(bubble["scale"]["bands"]) == {"low", "moderate", "elevated", "extreme"}


def test_the_reported_window_never_exceeds_the_data(ohlcv):
    """Asking for 180 days of anomalies on three months of history claimed
    'anomalies since' a date three months before anything examined."""
    short = ohlcv.tail(63)
    scan = anomaly_detector.scan("TEST", short, lookback_days=180)
    basis = scan["basis"]["anomalies_from"]

    assert basis["reported_since"] >= str(short.index[0].date()), (
        "the window starts before the first available bar")
    assert basis["window_truncated"] is True
    assert str(short.index[0].date()) in basis["note"], \
        "the user is not told the window was shortened"


def test_the_page_says_which_sample_each_number_came_from(ohlcv):
    """Scores read the whole period, anomalies read the lookback window. On
    screen those look like one figure disagreeing with another."""
    scan = anomaly_detector.scan("TEST", ohlcv, lookback_days=180)
    basis = scan["basis"]
    assert basis["scores_from"]["bars"] == len(ohlcv)
    assert basis["anomalies_from"]["bars"] <= len(ohlcv)
    # Drawdown depends on the period's own peak; that must be stated.
    assert "period" in basis["scores_from"]["note"].lower()


# ============================================== market regime detection
def _regime_frame(prices, seed=1):
    import numpy as np
    import pandas as pd

    n = len(prices)
    idx = pd.bdate_range("2022-01-03", periods=n)
    close = pd.Series(prices, index=idx)
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "open": close * 0.999, "high": close * 1.012, "low": close * 0.988,
        "close": close, "volume": rng.integers(8e5, 1.2e6, n).astype(float),
    }, index=idx)


def _ou(n, sigma, seed, mu=100.0, theta=0.06):
    """Mean-reverting path — a genuine range.

    A zero-drift random walk is NOT a range: its variance compounds, so it
    wanders far from where it started. An early version of these tests used a
    walk and called it 'sideways', then blamed the classifier for reporting the
    -24% downtrend the walk had actually produced.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    x, out = mu, []
    for _ in range(n):
        x += theta * (mu - x) + rng.normal(0.0, sigma)
        out.append(x)
    return np.array(out)


def test_regime_detector_recognises_textbook_conditions():
    import numpy as np

    from app.services.risk.regime import market_regime_detector as detector

    rng = np.random.default_rng(11)
    n = 400
    cases = {
        "bull_market": 100 * np.cumprod(1 + rng.normal(0.0009, 0.008, n)),
        "bear_market": 100 * np.cumprod(1 + rng.normal(-0.0011, 0.011, n)),
        "low_volatility": _ou(n, 0.28, 4),
        "high_volatility": _ou(n, 4.2, 5),
        # A trend that is real but *slow*: enough drift to be an uptrend, with
        # a clean linear fit. This is the case the original formula got wrong —
        # it rewarded a small |trend| for "sideways", and a steadily compounding
        # advance never moves far inside any single 63-day window. Using only
        # the strong-drift series above passed under the broken formula too, so
        # it guarded nothing.
        #
        # (An earlier attempt used seed 7 at 0.0008 drift and asserted
        # "bull_market". That series actually ends 3% DOWN with an R² of 0.14
        # over the window the classifier sees — "sideways" was the correct
        # answer and the test label was wrong, not the code.)
        "bull_market_slow": 100 * np.exp(
            np.linspace(0, 0.32, n)
            + np.random.default_rng(7).normal(0, 0.010, n).cumsum() * 0.18),
    }
    expectations = dict.fromkeys(cases, None)
    expectations["bull_market_slow"] = "bull_market"
    crash = list(100 * np.cumprod(1 + rng.normal(0.0006, 0.008, n - 35)))
    crash += list(crash[-1] * np.cumprod(1 + rng.normal(-0.016, 0.038, 35)))
    cases["crash_risk"] = np.array(crash)

    for name, prices in cases.items():
        # Most keys name their own expected regime; the slow-trend case is the
        # exception, so it is looked up rather than assumed.
        expected = expectations.get(name) or name
        report = detector.detect("SYNTH", _regime_frame(prices), timeline_step=200)
        assert report["regime"] == expected, (
            f"{name} scenario was classified {report['regime']} "
            f"(p={report['probability']})")


def test_a_permanently_volatile_asset_is_not_crashing_every_day():
    """Crash risk keyed off the absolute volatility level, so an instrument
    that is always wild — a small cap, a crypto pair — read as crashing on
    every single bar. A crash is a change in volatility, not a level."""
    from app.services.risk.regime import market_regime_detector as detector

    calm_but_wild = _ou(400, sigma=4.2, seed=5)   # sustained ~58% annualised
    report = detector.detect("WILD", _regime_frame(calm_but_wild), timeline_step=200)
    assert report["regime"] != "crash_risk", (
        "sustained high volatility with no expansion was labelled a crash")
    assert report["context"]["volatility_ratio"] < 1.35, "this scenario does expand"


def test_regime_probabilities_are_a_distribution():
    import numpy as np

    from app.services.risk.regime import REGIMES
    from app.services.risk.regime import market_regime_detector as detector

    rng = np.random.default_rng(3)
    frame = _regime_frame(100 * np.cumprod(1 + rng.normal(0.0004, 0.01, 400)))
    report = detector.detect("SYNTH", frame, timeline_step=200)

    assert set(report["probabilities"]) == set(REGIMES), "a regime label is missing"
    total = sum(report["probabilities"].values())
    assert abs(total - 1.0) < 0.01, f"probabilities sum to {total}"
    assert report["probability"] == max(report["probabilities"].values())
    assert 0.0 <= report["confidence"] <= 0.95


def test_the_regime_timeline_never_looks_ahead():
    """Each point must be classified from the bars up to that date only. A
    timeline built on the full series would show a suspiciously prescient
    classifier."""
    import inspect

    import numpy as np

    from app.services.risk.regime import market_regime_detector as detector

    source = inspect.getsource(detector.history)
    assert "df.iloc[max(0, end - window):end]" in source, \
        "history() does not slice the frame up to each point"

    rng = np.random.default_rng(5)
    frame = _regime_frame(100 * np.cumprod(1 + rng.normal(0.0005, 0.01, 500)))
    timeline = detector.history(frame, step=25)
    assert timeline, "no timeline produced"
    assert all(p["date"] <= str(frame.index[-1].date()) for p in timeline)
    # Dates must advance monotonically for the chart to be readable.
    dates = [p["date"] for p in timeline]
    assert dates == sorted(dates)


def test_every_regime_has_an_action_and_a_reliability_note():
    """A classification with no instruction attached is trivia."""
    from app.services.risk.regime import (
        MODEL_RELIABILITY,
        REGIME_ACTIONS,
        REGIME_LABELS,
        REGIMES,
    )

    allowed = {"BUY", "HOLD", "REDUCE", "HEDGE", "SELL"}
    for regime in REGIMES:
        assert regime in REGIME_LABELS
        assert regime in MODEL_RELIABILITY
        action, rationale = REGIME_ACTIONS[regime]
        assert action in allowed, f"{regime} maps to unknown action {action}"
        assert len(rationale) > 40, f"{regime} has no usable rationale"


def test_short_history_is_refused_rather_than_guessed():
    import numpy as np

    from app.services.risk.regime import market_regime_detector as detector

    rng = np.random.default_rng(9)
    frame = _regime_frame(100 * np.cumprod(1 + rng.normal(0.0, 0.01, 40)))
    report = detector.detect("SHORT", frame)
    assert report["regime"] == "unknown"
    assert report["probability"] is None, "a guess was returned as a probability"
    assert "120" in report["reason"]


def test_the_insight_never_asserts_more_than_the_numbers_show():
    """The narrative is composed from computed values, so it cannot claim a
    confident call when the margin is thin."""
    from app.services.risk.regime import market_regime_detector as detector

    # A deliberately ambiguous series: no strong trend, ordinary volatility.
    frame = _regime_frame(_ou(400, sigma=1.4, seed=8))
    report = detector.detect("AMB", frame, timeline_step=200)
    insight = report["insight"].lower()
    if report["confidence"] < 0.5:
        assert "ambiguous" in insight, "a weak call is not described as weak"
    assert str(round(report["probability"] * 100)) in report["insight"]


# ==================================================== risk engine: correctness
#
# Each test below pins a defect that was measured on real data before the fix,
# not a hypothetical. The docstring records the observation.
def test_sortino_uses_downside_deviation_not_dispersion_of_losses():
    """Dividing by ``downside.std()`` measures how much losses vary around
    their own mean, not how large they are. A series whose every loss is
    exactly -2% has zero dispersion, so it scored 0.0 — "no risk-adjusted
    return" — when its true Sortino is 12.4."""
    import pandas as pd

    from app.services.risk.metrics import downside_deviation

    returns = pd.Series([0.03] * 60 + [-0.02] * 40)
    target = 0.02 / 252
    expected_dd = np.sqrt((np.minimum(returns - target, 0) ** 2).mean()) * np.sqrt(252)
    expected = float((returns - target).mean() * 252 / expected_dd)

    assert downside_deviation(returns, target) == pytest.approx(expected_dd, rel=1e-9)
    assert sortino_ratio(returns) == pytest.approx(expected, rel=1e-6)
    assert sortino_ratio(returns) > 5, "constant-size losses still score as riskless"


def test_var_is_unknown_rather_than_zero_on_too_little_data():
    """0.0 reads as "this asset cannot lose money" — the most dangerous thing
    a risk system can assert. The honest answer is None."""
    import pandas as pd

    assert value_at_risk(pd.Series([-0.05] * 9)) is None
    assert conditional_var(pd.Series([-0.05] * 9)) is None
    assert value_at_risk(pd.Series(np.random.default_rng(0).normal(0, 0.01, 300))) < 0


def test_profit_factor_is_not_zero_when_nothing_lost():
    """No losing day makes the ratio infinite; zero is the *worst* possible
    reading, so a flawless series was reported as the worst one."""
    import pandas as pd

    metrics = full_metrics(pd.Series([0.01] * 50))
    assert metrics["profit_factor"] > 1, "a series with no losses scored as the worst case"


def test_beta_refuses_to_answer_without_overlapping_dates():
    """An empty intersection used to return beta = 0.0 — "moves independently
    of the market" — when the truth was "never compared"."""
    import pandas as pd

    from app.services.risk.metrics import beta_alpha

    a = pd.Series(np.random.default_rng(1).normal(0, 0.01, 300),
                  index=pd.date_range("2020-01-01", periods=300, freq="B"))
    b = pd.Series(np.random.default_rng(2).normal(0, 0.01, 300),
                  index=pd.date_range("2015-01-01", periods=300, freq="B"))
    beta, alpha = beta_alpha(a, b)
    assert beta is None and alpha is None, "beta was invented from no shared history"


def test_overall_risk_score_tracks_absolute_volatility():
    """The old headline was max(crash_band, bubble_band, anomaly_band). Every
    crash term is relative to the asset's own history, so the score ignored
    absolute risk: NVDA at 36.4% annualised vol scored "low" while GLD at
    28.2% scored "high". Holding the return path fixed and scaling only sigma,
    the composite must rise monotonically."""
    import pandas as pd

    from app.services.risk.profile import risk_profiler

    z = np.random.default_rng(7).standard_normal(400)
    index = pd.date_range("2023-01-02", periods=400, freq="B")

    def frame(vol: float) -> pd.DataFrame:
        r = 0.0003 + z * (vol / np.sqrt(252))
        close = 100 * np.exp(np.cumsum(r))
        return pd.DataFrame({"open": close, "high": close * 1.005,
                             "low": close * 0.995, "close": close,
                             "volume": 1e6}, index=index)

    scores = []
    for vol in (0.05, 0.10, 0.20, 0.40, 0.80):
        df = frame(vol)
        profile = risk_profiler.profile(
            "T", df, crash=anomaly_detector.crash_risk(df),
            bubble=anomaly_detector.bubble_indicator(df))
        scores.append(profile["overall"]["score"])

    assert scores == sorted(scores), f"risk score is not monotone in volatility: {scores}"
    assert scores[-1] - scores[0] > 0.35, (
        f"a 16x change in volatility moved the score by only "
        f"{scores[-1] - scores[0]:.3f}")


def test_published_contributions_add_up_to_the_published_score():
    """Explainability is only real if the arithmetic checks out: a breakdown
    that does not sum to the headline is decoration."""
    from app.services.risk.profile import risk_profiler

    frame = generate_ohlcv_for_risk()
    profile = risk_profiler.profile(
        "T", frame, crash=anomaly_detector.crash_risk(frame),
        bubble=anomaly_detector.bubble_indicator(frame))
    overall = profile["overall"]
    total = sum(c["points"] for c in overall["contributions"] if c["points"] is not None)
    assert total == pytest.approx(overall["score"] * 100, abs=0.15), (
        f"contributions sum to {total:.2f} but the score is {overall['score'] * 100:.2f}")


def test_unmeasurable_contributors_are_dropped_not_counted_as_zero():
    """A missing input scored as 0.0 drags the composite toward "safe". Its
    weight must be redistributed, and the omission stated."""
    from app.services.risk.profile import risk_profiler

    frame = generate_ohlcv_for_risk()
    profile = risk_profiler.profile(
        "T", frame, crash=anomaly_detector.crash_risk(frame),
        bubble={"bubble_score": None, "level": "insufficient_data"})
    overall = profile["overall"]
    bubble_row = next(c for c in overall["contributions"] if c["key"] == "bubble")
    assert bubble_row["available"] is False
    assert bubble_row["points"] is None, "an unmeasured contributor scored points"
    assert overall["weight_redistributed"] is True
    live = [c for c in overall["contributions"] if c["available"]]
    assert sum(c["effective_weight"] for c in live) == pytest.approx(1.0, abs=1e-6)


def test_risk_scores_differ_between_assets_with_different_behaviour():
    """Two genuinely different instruments must not produce one shared score."""
    import pandas as pd

    from app.services.risk.profile import risk_profiler

    rng = np.random.default_rng(3)
    index = pd.date_range("2023-01-02", periods=400, freq="B")

    def frame(vol: float, drift: float, seed: int) -> pd.DataFrame:
        r = np.random.default_rng(seed).normal(drift, vol / np.sqrt(252), 400)
        close = 100 * np.exp(np.cumsum(r))
        return pd.DataFrame({"open": close, "high": close * 1.005,
                             "low": close * 0.995, "close": close,
                             "volume": rng.uniform(1e6, 2e6, 400)}, index=index)

    scores = []
    for vol, drift, seed in ((0.08, 0.0002, 11), (0.35, 0.0004, 12), (0.90, -0.001, 13)):
        df = frame(vol, drift, seed)
        profile = risk_profiler.profile(
            "X", df, crash=anomaly_detector.crash_risk(df),
            bubble=anomaly_detector.bubble_indicator(df))
        scores.append(profile["overall"]["score"])
    assert len(set(scores)) == len(scores), f"different assets shared a score: {scores}"


def generate_ohlcv_for_risk():
    """A 400-bar frame with enough history for every contributor."""
    import pandas as pd

    rng = np.random.default_rng(21)
    r = rng.standard_t(5, 400) / 100
    close = 100 * np.exp(np.cumsum(r))
    return pd.DataFrame(
        {"open": close, "high": close * 1.01, "low": close * 0.99,
         "close": close, "volume": rng.uniform(1e6, 3e6, 400)},
        index=pd.date_range("2023-01-02", periods=400, freq="B"))


# ==================================== regime-aware reinforcement learning
def _crash_frame(n: int = 700, seed: int = 4):
    """A series with a violent crash in the middle."""
    import pandas as pd

    rng = np.random.default_rng(seed)
    r = rng.normal(0.0006, 0.010, n)
    # Place the crash proportionally, so shorter frames still contain one
    # instead of silently broadcasting into an empty slice.
    start = int(n * 0.54)
    span = max(10, int(n * 0.07))
    r[start:start + span] = rng.normal(-0.030, 0.045, span)
    close = 100 * np.exp(np.cumsum(r))
    return pd.DataFrame(
        {"open": close, "high": close * 1.01, "low": close * 0.99,
         "close": close, "volume": 1e6},
        index=pd.date_range("2022-01-03", periods=n, freq="B"))


def test_regime_features_never_look_ahead():
    """Bar t must be classified from bars <= t only. An agent that can see
    tomorrow's regime backtests beautifully and fails live — the same class of
    bug that once cost this project eight trained agents.

    Comparing a full build against a truncated one is NOT enough, and that
    weaker version of this test passed against a deliberately leaking
    implementation: if the leak is present in both builds they agree with each
    other while both being wrong. So the slice handed to the classifier is
    captured directly and its right edge asserted.
    """
    from app.services.rl import regime_features as rf

    df = _crash_frame(400)
    seen: list[tuple] = []
    real = rf.market_regime_detector._classify

    def spy(slice_df, sentiment=None, with_history=False):
        seen.append((slice_df.index[0], slice_df.index[-1], len(slice_df)))
        return real(slice_df, sentiment=sentiment, with_history=with_history)

    rf.market_regime_detector._classify = spy
    try:
        provider = rf.build_provider(df)
    finally:
        rf.market_regime_detector._classify = real

    assert seen, "the detector was never called"
    # Every classification must end on the bar it is classifying, never later.
    for _first, last, _ in seen:
        assert last <= df.index[-1]
    # The decisive check: the k-th classification may not read beyond its bar.
    positions = {ts: i for i, ts in enumerate(df.index)}
    for i, (_, last, _) in enumerate(seen):
        bar = rf.MIN_BARS - 1 + i * provider.step
        bar = min(bar, len(df) - 1)
        assert positions[last] <= bar, (
            f"classification {i} read up to bar {positions[last]} "
            f"while classifying bar {bar} — future data leaked in")


def test_regime_awareness_is_opt_in_and_preserves_observation_width():
    """11 trained agents exist on disk. A network refuses an observation of the
    wrong width ('mat1 and mat2 shapes cannot be multiplied'), so enabling
    regime features by default would break every one of them."""
    from app.services.rl.environment import EnvConfig, TradingEnv
    from app.services.rl.regime_features import REGIME_FEATURE_DIM

    df = _crash_frame(300)
    assert EnvConfig().regime_aware is False, "regime awareness must default off"
    legacy = TradingEnv(df, EnvConfig())
    aware = TradingEnv(df, EnvConfig(regime_aware=True))
    assert (aware.observation_space.shape[0] - legacy.observation_space.shape[0]
            == REGIME_FEATURE_DIM)
    assert len(legacy.reset()[0]) == legacy.observation_space.shape[0]
    assert len(aware.reset()[0]) == aware.observation_space.shape[0]


def test_the_reward_punishes_crash_exposure_harder_when_regime_aware():
    """The whole point of an adaptive agent: the same action in the same bar
    should cost more when the market is in a crash regime."""
    from app.services.rl.environment import EnvConfig, TradingEnv

    df = _crash_frame()

    def run(aware: bool) -> tuple[float, float, int]:
        env = TradingEnv(df, EnvConfig(regime_aware=aware))
        env.reset()
        total = crash = 0.0
        bars = 0
        done = truncated = False
        while not (done or truncated):
            _, reward, done, truncated, info = env.step(2)   # 2 == BUY
            total += reward
            if info.get("regime") == "crash_risk":
                crash += reward
                bars += 1
        return total, crash, bars

    base_total, _, _ = run(False)
    aware_total, aware_crash, crash_bars = run(True)

    assert crash_bars > 0, "the scenario never entered a crash regime"
    assert aware_total < base_total, \
        "regime awareness did not change the reward at all"
    assert aware_crash < 0, "crash-regime bars were not penalised"


def test_risk_aversion_is_ordered_by_regime_severity():
    """A crash must not be treated more gently than a quiet bull market."""
    from app.services.rl.regime_features import REGIME_RISK, REGIME_RISK_AVERSION

    assert REGIME_RISK_AVERSION["crash_risk"] > REGIME_RISK_AVERSION["bear_market"]
    assert REGIME_RISK_AVERSION["bear_market"] > REGIME_RISK_AVERSION["sideways"]
    assert REGIME_RISK_AVERSION["sideways"] >= REGIME_RISK_AVERSION["bull_market"]
    assert REGIME_RISK["crash_risk"] == max(REGIME_RISK.values())
    assert REGIME_RISK["low_volatility"] == min(REGIME_RISK.values())


def test_unclassified_bars_are_not_reported_as_calm():
    """Before enough history exists the honest answer is 'unknown, assume
    ordinary' — not zero risk, which is a claim the data cannot support."""
    from app.services.rl.regime_features import REGIME_RISK, _neutral_row

    row = _neutral_row()
    assert row.classified is False
    assert row.confidence == 0.0
    assert row.risk == REGIME_RISK["sideways"], \
        "an unclassified bar is presented as calmer than a sideways market"


def test_cvar_penalty_only_charges_for_real_losses():
    """A 5th percentile above zero means the worst recent day was still a gain;
    penalising that would tax a winning streak."""
    from app.services.rl.environment import EnvConfig, TradingEnv

    df = _crash_frame(300)
    env = TradingEnv(df, EnvConfig())
    env.reset()
    env.equity_curve = list(np.linspace(100_000, 130_000, 40))   # monotone gains
    var, cvar = env._tail_risk()
    assert var == 0.0 and cvar == 0.0, "a rising equity curve was charged tail risk"


def test_regime_explanation_reports_no_influence_honestly():
    """A template asserting 'the regime influenced this decision' reads the
    same whether or not it did. The counterfactual must be able to say no."""
    from app.services.rl.environment import EnvConfig, TradingEnv
    from app.services.rl.regime_explain import explain_regime_influence

    df = _crash_frame(300)
    env = TradingEnv(df, EnvConfig())          # NOT regime aware
    env.reset()
    payload = explain_regime_influence(None, env, env._observation(), 1,
                                       {0: "SELL", 1: "HOLD", 2: "BUY"})
    assert payload["available"] is False
    assert "without regime awareness" in payload["reason"]


# ============================== regime-aware multi-asset allocation (PortfolioEnv)
def _two_asset_matrix(n: int = 650, seed: int = 11):
    """One asset that crashes mid-series, one that stays calm."""
    import pandas as pd

    rng = np.random.default_rng(seed)
    a = rng.normal(0.0005, 0.010, n)
    start, span = int(n * 0.58), max(20, int(n * 0.09))
    a[start:start + span] = rng.normal(-0.032, 0.05, span)
    b = rng.normal(0.0004, 0.007, n)
    return pd.DataFrame(
        {"CRASH": 100 * np.exp(np.cumsum(a)), "CALM": 100 * np.exp(np.cumsum(b))},
        index=pd.date_range("2022-01-03", periods=n, freq="B"))


def test_portfolio_regime_awareness_is_opt_in_and_sized_correctly():
    """Same contract as the single-asset env: adding features widens the
    observation, and a trained policy refuses the wrong width."""
    from app.services.rl.environment import EnvConfig, PortfolioEnv
    from app.services.rl.portfolio_regime import feature_dim

    matrix = _two_asset_matrix(400)
    legacy = PortfolioEnv(matrix, EnvConfig())
    aware = PortfolioEnv(matrix, EnvConfig(regime_aware=True))
    assert (aware.observation_space.shape[0] - legacy.observation_space.shape[0]
            == feature_dim(2))
    assert len(aware.reset()[0]) == aware.observation_space.shape[0]
    assert len(legacy.reset()[0]) == legacy.observation_space.shape[0]


def test_each_asset_gets_its_own_regime_track():
    """A single market-wide label would erase the difference an allocator
    exists to exploit: a crashing sleeve next to a calm one."""
    from app.services.rl.portfolio_regime import build_portfolio_provider

    matrix = _two_asset_matrix()
    provider = build_portfolio_provider(matrix)
    # Look just after the crash window.
    t = int(len(matrix) * 0.66)
    snapshot = provider.snapshot(t)
    per_asset = snapshot["per_asset"]
    assert set(per_asset) == {"CRASH", "CALM"}
    assert per_asset["CRASH"]["risk"] > per_asset["CALM"]["risk"], (
        f"the crashing asset was not scored riskier: {per_asset}")
    assert snapshot["regime_dispersion"] > 0, "regimes collapsed to one reading"


def test_risk_aversion_follows_the_allocation_not_just_the_market():
    """Holding cash through a crash must reduce the penalty, otherwise moving
    to safety only forgoes return instead of being rewarded."""
    from app.services.rl.portfolio_regime import build_portfolio_provider

    matrix = _two_asset_matrix()
    provider = build_portfolio_provider(matrix)
    t = int(len(matrix) * 0.66)

    all_crash = provider.aversion_for(t, np.array([1.0, 0.0, 0.0]))
    all_calm = provider.aversion_for(t, np.array([0.0, 1.0, 0.0]))
    all_cash = provider.aversion_for(t, np.array([0.0, 0.0, 1.0]))

    assert all_cash == pytest.approx(1.0, abs=1e-6), \
        "cash was charged a regime risk penalty"
    assert all_crash > all_calm, "concentrating in the crashing asset was not penalised"
    assert all_crash > all_cash, "holding the crasher cost no more than holding cash"


def test_portfolio_reward_punishes_crash_concentration_more_when_aware():
    from app.services.rl.environment import EnvConfig, PortfolioEnv

    matrix = _two_asset_matrix()

    def run(logits, aware):
        env = PortfolioEnv(matrix, EnvConfig(regime_aware=aware))
        env.reset()
        total = 0.0
        done = truncated = False
        while not (done or truncated):
            _, reward, done, truncated, _ = env.step(np.array(logits, dtype=float))
            total += reward
        return total

    concentrated = [4.0, -2.0, -2.0]     # ~all in the crashing asset
    crash_gap = run(concentrated, False) - run(concentrated, True)
    assert crash_gap > 0, "regime awareness did not penalise crash concentration"

    # A "cash" action cannot be *pure* cash: the action vector goes through a
    # softmax, so logits [-2,-2,4] still leave ~0.5% in each asset. That
    # residual genuinely carries regime risk, so demanding an exactly equal
    # reward here would be asserting something false — an earlier version of
    # this test did, and failed for the right reason. What must hold is that
    # the penalty scales with exposure: near-cash is charged orders of
    # magnitude less than concentration in the same crashing asset.
    cash = [-2.0, -2.0, 4.0]
    cash_gap = run(cash, False) - run(cash, True)
    assert cash_gap >= 0, "holding cash was rewarded for a crash"
    assert cash_gap < crash_gap / 100, (
        f"cash was penalised comparably to full exposure "
        f"({cash_gap:.3f} vs {crash_gap:.3f})")


def test_synthesised_ohlc_is_flagged_rather_than_hidden():
    """The price matrix is close-only; ADX needs high/low. Synthesising them is
    an acceptable fallback but must be visible, not silent."""
    from app.services.rl.portfolio_regime import build_portfolio_provider

    matrix = _two_asset_matrix(300)
    provider = build_portfolio_provider(matrix)          # no OHLCV supplied
    assert provider.ohlc_synthesised == {"CRASH": True, "CALM": True}
    assert provider.summary()["ohlc_synthesised"]["CRASH"] is True


def test_both_environments_share_one_tail_risk_implementation():
    """PortfolioEnv called _tail_risk while it lived only on TradingEnv, which
    would raise AttributeError on the first multi-asset step. Two copies of a
    risk formula also drift apart."""
    from app.services.rl.environment import PortfolioEnv, TradingEnv, _TailRiskMixin

    assert issubclass(TradingEnv, _TailRiskMixin)
    assert issubclass(PortfolioEnv, _TailRiskMixin)
    assert TradingEnv._tail_risk is PortfolioEnv._tail_risk, \
        "the two environments compute tail risk differently"


# ==================== per-asset attribution of regime influence on allocation
def _alloc_env(regime_aware: bool = True, n: int = 420):
    """A two-asset env whose sleeves are in genuinely different states."""
    from app.services.rl.environment import EnvConfig, PortfolioEnv

    matrix = _two_asset_matrix(n)
    return PortfolioEnv(matrix, EnvConfig(regime_aware=regime_aware))


class _FixedWeightAgent:
    """An agent whose output depends on the regime block in a known way.

    A real trained policy would make this test measure the policy rather than
    the attribution code. This one tilts toward the second asset in proportion
    to the first asset's risk feature, so the expected direction of every
    weight delta is known in advance.
    """

    def __init__(self, n_assets: int, block: int, sensitivity: float = 6.0):
        self.n_assets, self.block, self.sensitivity = n_assets, block, sensitivity

    def act(self, obs, deterministic=True):
        regime = np.asarray(obs)[-self.block:]
        first_risk = float(regime[0])          # risk of asset 0
        logits = np.zeros(self.n_assets + 1, dtype=float)
        logits[1] = self.sensitivity * first_risk
        return logits


def test_allocation_attribution_reports_per_asset_weight_deltas():
    """A weight vector has no 'the action flipped' moment. The useful question
    is which sleeve lost capital and to whom, so the attribution is per asset."""
    from app.services.rl.allocation_explain import explain_allocation_influence
    from app.services.rl.portfolio_regime import feature_dim

    env = _alloc_env()
    env.reset()
    env.t = int(len(env.prices) * 0.66)
    obs = env._observation()
    block = feature_dim(env.n_assets)
    agent = _FixedWeightAgent(env.n_assets, block)

    raw = agent.act(obs)
    exp = np.exp(raw - raw.max())
    weights = exp / exp.sum()

    payload = explain_allocation_influence(agent, env, obs, weights)
    assert payload["available"] is True
    symbols = {a["symbol"] for a in payload["per_asset"]}
    assert symbols == {"CRASH", "CALM"}, f"not every asset was attributed: {symbols}"
    for row in payload["per_asset"]:
        assert row["weight_without_regime"] is not None
        assert row["direction"] in ("increased", "reduced", "unchanged")
        # Each asset's delta must be tied to that asset's own regime.
        assert row["regime"], "an asset carries no regime label"
    assert payload["influence"] in ("decisive", "contributory", "negligible")
    assert payload["capital_moved"] >= 0


def test_allocation_attribution_separates_rotation_from_de_risking():
    """Moving to cash and rotating between assets are different behaviours;
    one 'changed' flag would make a defensive agent look opportunistic."""
    from app.services.rl.allocation_explain import explain_allocation_influence
    from app.services.rl.portfolio_regime import feature_dim

    env = _alloc_env()
    env.reset()
    env.t = int(len(env.prices) * 0.66)
    obs = env._observation()
    block = feature_dim(env.n_assets)
    agent = _FixedWeightAgent(env.n_assets, block)

    raw = agent.act(obs)
    exp = np.exp(raw - raw.max())
    payload = explain_allocation_influence(agent, env, obs, exp / exp.sum())
    assert payload["shift_type"] in ("de-risking", "re-risking", "rotation")
    # cash_delta and shift_type must agree, or the label is decorative.
    if payload["cash_delta"] > 1e-4:
        assert payload["shift_type"] == "de-risking"
    elif payload["cash_delta"] < -1e-4:
        assert payload["shift_type"] == "re-risking"


def test_allocation_attribution_measures_capital_not_direction():
    """Turnover is half the L1 norm: the raw sum double-counts every move
    because one sleeve's gain is another's loss."""
    from app.services.rl.allocation_explain import explain_allocation_influence
    from app.services.rl.portfolio_regime import feature_dim

    env = _alloc_env()
    env.reset()
    env.t = int(len(env.prices) * 0.66)
    obs = env._observation()
    block = feature_dim(env.n_assets)
    agent = _FixedWeightAgent(env.n_assets, block, sensitivity=9.0)

    raw = agent.act(obs)
    exp = np.exp(raw - raw.max())
    weights = exp / exp.sum()
    payload = explain_allocation_influence(agent, env, obs, weights)

    deltas = [a["delta"] for a in payload["per_asset"]] + [payload["cash_delta"]]
    # Weights are two simplexes, so the deltas must net to zero.
    assert abs(sum(deltas)) < 1e-3, f"weight deltas do not net to zero: {deltas}"
    assert payload["capital_moved"] <= 1.0 + 1e-9, "more than 100% of the book moved"

    # Assert the *definition*, not merely a loose bound. An earlier version of
    # this test only checked `<= 1.0`, which a doubled turnover still satisfies
    # — the mutation that removed the /2 passed it. Capital that changed hands
    # equals the weight gained by the winners, which is half the L1 norm.
    gained = sum(d for d in deltas if d > 0)
    assert payload["capital_moved"] == pytest.approx(gained, abs=2e-3), (
        f"capital_moved {payload['capital_moved']:.5f} is not the weight gained "
        f"by the winning sleeves ({gained:.5f}) — the L1 norm is double-counted")


def test_allocation_attribution_declines_when_the_agent_is_not_regime_aware():
    """No regime block means no attribution — stated, not fabricated."""
    from app.services.rl.allocation_explain import explain_allocation_influence

    env = _alloc_env(regime_aware=False)
    env.reset()
    obs = env._observation()
    payload = explain_allocation_influence(None, env, obs, np.array([0.5, 0.3, 0.2]))
    assert payload["available"] is False
    assert "without regime awareness" in payload["reason"]


def test_a_basket_where_every_asset_shares_a_regime_is_flagged():
    """Diversification stops helping when every sleeve is in the same state —
    that is the correlations-go-to-one problem, and it must be visible."""
    import pandas as pd

    from app.services.rl.portfolio_regime import build_portfolio_provider

    rng = np.random.default_rng(5)
    n = 420
    shared = rng.normal(0.0008, 0.008, n)
    matrix = pd.DataFrame(
        {"A": 100 * np.exp(np.cumsum(shared)),
         "B": 100 * np.exp(np.cumsum(shared * 1.01))},
        index=pd.date_range("2022-01-03", periods=n, freq="B"))
    snapshot = build_portfolio_provider(matrix).snapshot(n - 1)
    assert snapshot["all_assets_same_regime"] is True
    assert snapshot["regime_dispersion"] == pytest.approx(0.0, abs=1e-9)


# ============================ centralised hyperparameter configuration
def test_every_algorithm_has_a_config_file():
    """A missing file is not a silent fallback to defaults — it is a training
    request that cannot be served, so it must fail loudly at resolve time."""
    from app.services.rl.hyperparams import hyperparameters
    from app.services.rl.service import SUPPORTED_ALGOS

    configured = set(hyperparameters.algorithms())
    missing = {a for a in SUPPORTED_ALGOS if a not in configured}
    assert not missing, f"algorithms with no configs/algorithms/*.yaml: {sorted(missing)}"


def test_resolution_layers_in_the_documented_order():
    """defaults -> algorithm -> profile. A later layer must win."""
    from app.services.rl.hyperparams import hyperparameters

    defaults = hyperparameters.defaults()
    # c51 overrides the shared learning rate; the resolved value must be its own.
    c51 = hyperparameters.resolve("c51", "default")
    assert c51.get("optimizer.learning_rate") != defaults["optimizer"]["learning_rate"]
    assert c51.get("optimizer.learning_rate") == 0.00005

    # A profile overrides the algorithm in turn.
    aggressive = hyperparameters.resolve("c51", "aggressive")
    assert aggressive.get("optimizer.learning_rate") == 0.001
    assert aggressive.get("risk.risk_penalty") < c51.get("risk.risk_penalty")


def test_a_profile_does_not_erase_sibling_keys():
    """Shallow merging would let a profile that sets one risk coefficient wipe
    out every other key in that section."""
    from app.services.rl.hyperparams import hyperparameters

    conservative = hyperparameters.resolve("ppo", "conservative")
    risk = conservative.section("risk")
    for key in ("risk_penalty", "drawdown_penalty", "turnover_penalty",
                "cvar_penalty", "cvar_alpha", "regime_aware"):
        assert key in risk, f"the profile merge dropped risk.{key}"


def test_out_of_range_hyperparameters_are_refused():
    """A learning rate of 50 does not fail loudly on its own — it quietly
    produces a useless agent hours later."""
    from app.core.exceptions import InvalidRequestError
    from app.services.rl.hyperparams import hyperparameters

    for override in ({"optimizer.learning_rate": 50},
                     {"optimizer.gamma": 2.0},
                     {"replay.min_buffer": 99999, "replay.buffer_size": 1000},
                     {"exploration.epsilon_end": 0.9, "exploration.epsilon_start": 0.1},
                     {"network.hidden": "wide"}):
        with pytest.raises(InvalidRequestError):
            hyperparameters.resolve("ppo", "default", override)


def test_profile_names_cannot_escape_the_profiles_directory():
    """Profile names become filenames."""
    from app.core.exceptions import InvalidRequestError
    from app.services.rl.hyperparams import hyperparameters

    for bad in ("../../etc/passwd", "..", "a/b", "", "  ", "x" * 80):
        with pytest.raises(InvalidRequestError):
            hyperparameters.profile_config(bad)


def test_builtin_profiles_cannot_be_overwritten_or_deleted(tmp_path):
    """Overwriting `default` would leave no baseline and no way back.

    This runs against a *copy* of configs/ rather than the real directory. An
    earlier version pointed at the live files, and while mutation-testing the
    protection itself it wrote `learning_rate: 0.9` into the real
    `default.yaml` — a test that damages the repository when the code under
    test regresses is a worse failure mode than the regression.
    """
    import shutil

    from app.core.exceptions import InvalidRequestError
    from app.services.rl.hyperparams import CONFIG_DIR, HyperparameterManager

    sandbox = tmp_path / "configs"
    shutil.copytree(CONFIG_DIR, sandbox)
    manager = HyperparameterManager(sandbox)

    for name in ("default", "conservative", "aggressive", "risk_aware",
                 "high_performance"):
        with pytest.raises(InvalidRequestError):
            manager.save_profile(name, {"optimizer": {"learning_rate": 0.9}})
        with pytest.raises(InvalidRequestError):
            manager.delete_profile(name)

    # The shipped defaults must be untouched by the attempts above.
    assert manager.resolve("ppo", "default").get("optimizer.learning_rate") == 0.0003


def test_a_user_profile_round_trips_through_save_export_and_import(tmp_path):
    """Duplicate -> edit -> export -> import is the documented workflow."""
    import shutil

    from app.services.rl.hyperparams import CONFIG_DIR, HyperparameterManager

    sandbox = tmp_path / "configs"
    shutil.copytree(CONFIG_DIR, sandbox)
    manager = HyperparameterManager(sandbox)

    manager.duplicate_profile("conservative", "my_copy")
    manager.save_profile("my_copy", {"optimizer": {"learning_rate": 0.00042}})
    assert manager.resolve("ppo", "my_copy").get("optimizer.learning_rate") == 0.00042

    exported = manager.export_profile("my_copy")
    manager.delete_profile("my_copy")
    assert "my_copy" not in [p["key"] for p in manager.profiles()]

    manager.import_profile("restored", exported)
    assert manager.resolve("ppo", "restored").get("optimizer.learning_rate") == 0.00042


def test_the_fingerprint_identifies_the_configuration_not_its_description():
    """Two runs sharing a fingerprint used identical hyperparameters. Editing a
    description must not read as a different experiment."""
    from app.services.rl.hyperparams import fingerprint, hyperparameters

    a = hyperparameters.resolve("ppo", "default")
    b = hyperparameters.resolve("ppo", "default")
    assert a.fingerprint == b.fingerprint

    c = hyperparameters.resolve("ppo", "aggressive")
    assert a.fingerprint != c.fingerprint, "different profiles share a fingerprint"

    params = dict(a.params)
    params["meta"] = {"description": "a totally different note"}
    assert fingerprint(params) == a.fingerprint, "meta leaked into the fingerprint"


def test_the_rl_module_has_no_hardcoded_training_parameters():
    """The point of this migration: a literal here is a value no profile can
    reach, and one that a completed run cannot record."""
    import re
    from pathlib import Path

    rl_dir = Path(__file__).resolve().parents[1] / "app" / "services" / "rl"
    banned = re.compile(
        r"(learning_rate|lr)\s*=\s*\d|buffer_size\s*=\s*\d{3,}"
        r"|learning_starts\s*=\s*\d|exploration_fraction\s*=\s*0\.\d"
        r"|total_timesteps\s*=\s*\d{4,}")
    offenders: list[str] = []
    for path in rl_dir.rglob("*.py"):
        # Dataclass defaults are the fallback when a key is absent from YAML;
        # they are the schema, not the configuration. Only *call sites* count.
        for i, line in enumerate(path.read_text().splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith("#") or ":" in stripped.split("=")[0]:
                continue          # comment, or an annotated dataclass field
            if banned.search(stripped):
                offenders.append(f"{path.name}:{i}: {stripped[:90]}")
    assert not offenders, "hardcoded training parameters remain:\n" + "\n".join(offenders)


def test_training_records_a_reproducible_configuration(tmp_path):
    """A result that cannot be traced to its parameters cannot be reproduced."""
    from app.services.rl.service import rl_service

    meta = rl_service.train_single_asset(
        "AAPL", period="2y", algo="dueling_dqn", episodes=1, profile="conservative")
    for field in ("experiment_id", "profile", "hyperparameters",
                  "hyperparameter_fingerprint", "config_sources", "seed"):
        assert field in meta, f"the run does not record {field}"
    assert meta["profile"] == "conservative"
    # The record must be the full materialised set, not a diff: the YAML can
    # change afterwards and this run still has to be replayable.
    assert meta["hyperparameters"]["optimizer"]["learning_rate"]
    assert meta["hyperparameters"]["risk"]["risk_penalty"] == 0.30
    assert meta["env_config"]["risk_penalty"] == 0.30, \
        "the profile did not reach the environment"


# ================== periodic evaluation and checkpointing during training
class _StubAgent:
    """Records what the monitor asked of it, without training anything."""

    def __init__(self, returns=None):
        self.evaluated: list[int] = []
        self.saved: list[str] = []
        self._returns = list(returns or [])
        self._n = 0

    def evaluate(self, env, deterministic=True):
        self._n += 1
        self.evaluated.append(self._n)
        value = (self._returns[self._n - 1]
                 if self._n <= len(self._returns) else 0.05)
        return {"performance": {"total_return": value, "sharpe_ratio": 1.0,
                                "max_drawdown": -0.1, "final_value": 100_000}}

    def save(self, path):
        from pathlib import Path

        Path(path).write_bytes(b"checkpoint")
        self.saved.append(str(path))


def test_eval_freq_and_checkpoint_interval_actually_fire(tmp_path):
    """Both keys existed in configs/ and were recorded in every run's
    reproducibility block while nothing read them — a parameter that is stored
    and ignored is worse than a missing one, because the record claims it was
    applied."""
    from app.services.rl.monitor import TrainingMonitor

    agent = _StubAgent()
    monitor = TrainingMonitor(eval_env=object(), eval_freq=2,
                              checkpoint_interval=3,
                              checkpoint_dir=tmp_path, run_id="t")
    for episode in range(1, 11):
        monitor.on_episode_end(episode, 10, agent)

    evaluated = [e["episode"] for e in monitor.evaluations]
    checkpointed = [c["episode"] for c in monitor.checkpoints]
    assert evaluated == [2, 4, 6, 8], f"eval_freq=2 fired at {evaluated}"
    assert checkpointed == [3, 6, 9], f"checkpoint_interval=3 fired at {checkpointed}"
    # The final episode is skipped: the caller evaluates and saves there anyway,
    # so firing would duplicate both the work and the record.
    assert 10 not in evaluated and 10 not in checkpointed


def test_monitoring_is_off_by_default_and_is_a_true_no_op():
    """Zero must mean "never", not "every episode" — the previous behaviour
    has to survive untouched for anyone already training."""
    from app.services.rl.monitor import TrainingMonitor

    agent = _StubAgent()
    monitor = TrainingMonitor(eval_env=object(), eval_freq=0,
                              checkpoint_interval=0)
    for episode in range(1, 21):
        monitor.on_episode_end(episode, 20, agent)
    assert monitor.active is False
    assert agent.evaluated == [] and agent.saved == []
    assert monitor.summary()["enabled"] is False


def test_old_checkpoints_are_pruned_from_disk(tmp_path):
    """A 60-episode run at interval 1 would otherwise leave 60 model files."""
    from app.services.rl.monitor import TrainingMonitor

    agent = _StubAgent()
    monitor = TrainingMonitor(checkpoint_interval=1, checkpoint_dir=tmp_path,
                              run_id="t", max_checkpoints=3)
    for episode in range(1, 11):
        monitor.on_episode_end(episode, 20, agent)

    assert len(monitor.checkpoints) == 3
    on_disk = sorted(p.name for p in tmp_path.iterdir())
    assert len(on_disk) == 3, f"pruned entries were left on disk: {on_disk}"
    assert [c["episode"] for c in monitor.checkpoints] == [8, 9, 10]


def test_a_failed_evaluation_does_not_destroy_the_run(tmp_path):
    """Hours of training must not be lost to one bad mid-flight evaluation."""
    from app.services.rl.monitor import TrainingMonitor

    class _Exploding(_StubAgent):
        def evaluate(self, env, deterministic=True):
            raise RuntimeError("evaluation blew up")

    monitor = TrainingMonitor(eval_env=object(), eval_freq=1)
    monitor.on_episode_end(1, 5, _Exploding())      # must not raise
    assert monitor.evaluations and "error" in monitor.evaluations[0]


def test_checkpoints_carry_a_loadable_extension(tmp_path):
    """Native agents write exactly the path they are handed, so a bare stem
    produced extensionless files no loader would recognise."""
    from app.services.rl.monitor import TrainingMonitor

    agent = _StubAgent()
    monitor = TrainingMonitor(checkpoint_interval=1, checkpoint_dir=tmp_path,
                              run_id="t")
    monitor.on_episode_end(1, 5, agent)
    written = list(tmp_path.iterdir())
    assert written and all(p.suffix in (".pt", ".zip") for p in written), \
        f"checkpoint has no loadable extension: {[p.name for p in written]}"


def test_the_monitor_refuses_to_pick_a_best_model_on_the_test_window():
    """Selecting the best-scoring checkpoint would be choosing a model on the
    held-out data, inflating every figure reported from it afterwards."""
    from app.services.rl.monitor import TrainingMonitor

    agent = _StubAgent(returns=[0.01, 0.42, 0.02])
    monitor = TrainingMonitor(eval_env=object(), eval_freq=1)
    for episode in range(1, 4):
        monitor.on_episode_end(episode, 10, agent)
    summary = monitor.summary()
    assert summary["best_checkpoint"]["total_return"] == 0.42
    note = summary["selection_note"].lower()
    assert "test set" in note or "test window" in note, \
        "the best-score figure is published with no warning attached"


def test_training_records_what_the_monitor_actually_did():
    """The metadata has to distinguish "ran and found nothing" from "was off"."""
    from app.services.rl.service import rl_service

    meta = rl_service.train_single_asset(
        "AAPL", period="2y", algo="dueling_dqn", episodes=4,
        hyperparams={"evaluation": {"eval_freq": 2, "checkpoint_interval": 2}})
    monitoring = meta["monitoring"]
    assert monitoring["enabled"] is True
    assert monitoring["unit"] == "episodes"
    assert [e["episode"] for e in monitoring["evaluations"]] == [2]
    assert [c["episode"] for c in monitoring["checkpoints"]] == [2]
    # The evaluation must run on held-out data, so it can disagree with the
    # training reward — that disagreement is the entire point of the curve.
    assert monitoring["evaluations"][0]["total_return"] is not None


def test_an_unknown_episode_total_does_not_disable_monitoring(tmp_path):
    """On the timestep-driven paths (portfolio, continuous) the episode count
    is not known up front and 0 is passed. Treating that as a total made
    `episode >= 0` true immediately, so every evaluation and checkpoint was
    silently skipped while the run still reported monitoring as enabled."""
    from app.services.rl.monitor import TrainingMonitor

    agent = _StubAgent()
    monitor = TrainingMonitor(eval_env=object(), eval_freq=2,
                              checkpoint_interval=3, checkpoint_dir=tmp_path,
                              run_id="t")
    for episode in range(1, 8):
        monitor.on_episode_end(episode, 0, agent)      # 0 == "unknown"

    assert [e["episode"] for e in monitor.evaluations] == [2, 4, 6], \
        "an unknown episode total silently disabled periodic evaluation"
    assert [c["episode"] for c in monitor.checkpoints] == [3, 6]


# ================== automatic provisioning of hyperparameter configuration
def test_missing_configs_are_provisioned_automatically(tmp_path):
    """A fresh clone, or a container built without configs/, previously left
    every training request failing with "No configuration file for algorithm
    'ppo'" and no way to recover from the dashboard. Measured on an empty
    directory before this existed: zero profiles, every resolve raised."""
    from app.services.rl.hyperparams import HyperparameterManager

    manager = HyperparameterManager(tmp_path / "configs")
    assert manager.profiles() == [], "the fixture is not actually empty"

    report = manager.ensure_configs()
    assert report["created"], "nothing was provisioned"
    assert not report["failed"]

    keys = {p["key"] for p in manager.profiles()}
    assert {"default", "conservative", "aggressive", "risk_aware",
            "high_performance"} <= keys, f"missing built-in profiles: {keys}"

    # Every supported algorithm must be seeded, not just the profiles. A
    # partial seed leaves `resolve()` failing for whichever algorithm was
    # skipped, and checking only one of them would not notice.
    from app.services.rl.service import SUPPORTED_ALGOS

    configured = set(manager.algorithms())
    assert set(SUPPORTED_ALGOS) <= configured, \
        f"algorithms not provisioned: {sorted(set(SUPPORTED_ALGOS) - configured)}"

    # And the result must actually resolve, not merely exist on disk.
    for algo in sorted(SUPPORTED_ALGOS):
        assert manager.resolve(algo, "conservative").get("optimizer.learning_rate")


def test_provisioning_never_overwrites_a_user_edit(tmp_path):
    """Re-seeding on every boot would silently revert the user's own tuning.

    The edit is made to a file that *has* a template, because that is the only
    one seeding could overwrite. An earlier version of this test edited a
    user-created profile instead — which no template can touch — so removing
    the `if path.exists(): return` guard left it passing. It asserted nothing.
    """
    import yaml

    from app.services.rl import config_templates as templates
    from app.services.rl.hyperparams import HyperparameterManager

    manager = HyperparameterManager(tmp_path / "configs")
    manager.ensure_configs()

    # A built-in file, edited directly on disk the way the platform writes it.
    target = tmp_path / "configs" / "algorithms" / "ppo.yaml"
    edited = yaml.safe_load(target.read_text())
    edited["optimizer"]["learning_rate"] = 0.000777
    target.write_text(yaml.safe_dump(edited, sort_keys=False))
    assert "ppo" in templates.ALGORITHMS, "the fixture must edit a seeded file"

    for _ in range(3):                       # three restarts
        manager.ensure_configs()

    assert manager.resolve("ppo", "default").get("optimizer.learning_rate") == 0.000777, \
        "provisioning overwrote a file the user had already edited"

    # A user-created profile must survive too.
    manager.duplicate_profile("conservative", "mine")
    manager.save_profile("mine", {"optimizer": {"learning_rate": 0.000123}})
    manager.ensure_configs()
    assert manager.resolve("ppo", "mine").get("optimizer.learning_rate") == 0.000123


def test_a_deleted_builtin_profile_is_restored(tmp_path):
    """Self-healing: the platform must not stay broken because a file vanished."""
    from app.services.rl.hyperparams import HyperparameterManager

    manager = HyperparameterManager(tmp_path / "configs")
    manager.ensure_configs()
    (tmp_path / "configs" / "profiles" / "aggressive.yaml").unlink()
    assert "aggressive" not in {p["key"] for p in manager.profiles()}

    manager.ensure_configs()
    assert "aggressive" in {p["key"] for p in manager.profiles()}


def test_the_embedded_templates_match_the_shipped_yaml():
    """The templates are a seed for `configs/`, not a second source of truth.
    If they drift, a fresh install and an existing one behave differently."""
    import yaml

    from app.services.rl import config_templates as templates
    from app.services.rl.hyperparams import CONFIG_DIR

    on_disk = yaml.safe_load((CONFIG_DIR / "defaults.yaml").read_text())
    assert on_disk == templates.DEFAULTS, "defaults.yaml has drifted from its template"

    for algo, payload in templates.ALGORITHMS.items():
        shipped = yaml.safe_load((CONFIG_DIR / "algorithms" / f"{algo}.yaml").read_text())
        assert payload == shipped, f"algorithms/{algo}.yaml has drifted"


def test_a_schema_default_never_overrides_the_selected_profile():
    """`risk_penalty: float = Field(0.15)` was sent on every request and could
    not be told apart from a value the user typed, so choosing "Conservative"
    (risk_penalty 0.30) still trained at 0.15. None means "not specified"."""
    from app.schemas.common import TrainRLRequest
    from app.services.rl.service import rl_service

    request = TrainRLRequest(symbol="AAPL", profile="conservative")
    assert request.risk_penalty is None, \
        "a numeric default is indistinguishable from a user's choice"
    assert request.episodes is None and request.test_fraction is None

    overrides = {"initial_balance": request.initial_balance,
                 "transaction_cost": request.transaction_cost,
                 "risk_penalty": request.risk_penalty}
    cfg = rl_service._resolve_hyperparams(
        "dueling_dqn", request.profile, None, overrides,
        request.episodes, request.total_timesteps, request.test_fraction)
    assert cfg.get("risk.risk_penalty") == 0.30, \
        "the profile was overridden by a schema default the user never set"
    assert cfg.get("training.episodes") == 30      # conservative's own value


def test_saving_one_field_does_not_erase_the_rest_of_the_profile(tmp_path):
    """The dashboard sends only the fields the user touched, so a wholesale
    replace silently deleted everything else.

    Reproduced end to end before the fix: duplicating Conservative and then
    editing a single learning rate left a profile containing *only*
    `optimizer` — its risk penalties, trade fraction and regime settings gone —
    and training then quietly ran on the defaults those values existed to
    override, while still reporting the profile by name.
    """
    import shutil

    from app.services.rl.hyperparams import CONFIG_DIR, HyperparameterManager

    sandbox = tmp_path / "configs"
    shutil.copytree(CONFIG_DIR, sandbox)
    manager = HyperparameterManager(sandbox)

    manager.duplicate_profile("conservative", "mine")
    before = {k for k in manager.profile_config("mine") if k != "meta"}
    assert {"risk", "environment", "training"} <= before, before

    # A single-field edit, exactly what the UI sends.
    manager.save_profile("mine", {"optimizer": {"learning_rate": 0.000456}})

    after = manager.profile_config("mine")
    assert {k for k in after if k != "meta"} >= before, \
        f"sections were dropped: {before - {k for k in after if k != 'meta'}}"
    assert after["risk"]["risk_penalty"] == 0.30, "the profile's own risk settings were lost"
    assert after["optimizer"]["learning_rate"] == 0.000456, "the edit was not applied"

    # And it must still reach training that way.
    resolved = manager.resolve("ppo", "mine")
    assert resolved.get("risk.risk_penalty") == 0.30
    assert resolved.get("optimizer.learning_rate") == 0.000456


def test_importing_a_profile_replaces_rather_than_merges(tmp_path):
    """An imported file is a complete document. Merging it onto whatever was
    there would produce a profile matching neither the file nor the previous
    state — defeating the reproducibility import exists for."""
    import shutil

    from app.services.rl.hyperparams import CONFIG_DIR, HyperparameterManager

    sandbox = tmp_path / "configs"
    shutil.copytree(CONFIG_DIR, sandbox)
    manager = HyperparameterManager(sandbox)

    manager.duplicate_profile("conservative", "target")
    assert "risk" in manager.profile_config("target")

    manager.import_profile("target", "optimizer:\n  learning_rate: 0.00099\n")
    imported = manager.profile_config("target")
    assert {k for k in imported if k != "meta"} == {"optimizer"}, \
        "import merged with the previous contents instead of replacing them"
    assert imported["optimizer"]["learning_rate"] == 0.00099


# ============================== AI-driven automatic hyperparameter selection
def test_the_regime_drives_the_recommended_profile():
    """A crash and a bull market must not produce the same configuration —
    otherwise "AI-driven" is just a relabelled default."""
    from app.services.rl.autotune import Environment, recommend_profile

    crash = Environment(algo="ppo", regime="crash_risk", regime_confidence=0.9)
    bull = Environment(algo="ppo", regime="bull_market", regime_confidence=0.9)
    assert recommend_profile(crash)[0] == "conservative"
    assert recommend_profile(bull)[0] != "conservative"
    # And it must say why, not just decide.
    assert "crash" in recommend_profile(crash)[1]


def test_a_weak_regime_call_is_not_acted_on():
    """Tuning on a coin-flip classification would make the recommendation swing
    between runs for no measurable reason."""
    from app.services.rl.autotune import Environment, recommend_profile

    unsure = Environment(algo="ppo", regime="crash_risk", regime_confidence=0.2)
    profile, reason = recommend_profile(unsure)
    assert profile == "default"
    assert "confidence" in reason


def test_every_automatic_adjustment_carries_a_reason():
    """A silent override is indistinguishable from a hardcoded value."""
    from app.services.rl.autotune import Environment, derive_overrides

    env = Environment(algo="sac", backend="sb3", action_space="continuous",
                      bars=1200, n_assets=5, cpu_count=16, volatility=0.60,
                      regime="high_volatility", regime_confidence=0.8)
    overrides = derive_overrides(env, "default")
    assert overrides, "nothing was adjusted for a distinctive environment"
    for item in overrides:
        assert item["path"] and item["reason"], f"unexplained override: {item}"
        assert len(item["reason"]) > 10

    paths = {o["path"] for o in overrides}
    assert "optimizer.batch_size" in paths      # 16 cores
    assert "network.hidden" in paths            # 5 assets
    assert "risk.risk_penalty" in paths         # 60% volatility


def test_the_time_estimate_uses_steady_state_throughput():
    """A first benchmark over two episodes reported 0.181 ms/step for the
    native agents — 13x too fast, because the replay buffer had not reached
    min_buffer so learn_step() was a no-op. Validated against a real run, that
    estimate was 10.8x under the true wall clock."""
    from app.services.rl.autotune import STEP_SECONDS, Environment, estimate_training

    assert STEP_SECONDS["native"] > 0.001, \
        "the native rate looks like a warm-up measurement, not steady state"

    env = Environment(algo="dueling_dqn", backend="native", bars=500)
    estimate = estimate_training(env, {"training": {"episodes": 25}})
    # 25 * 500 * 2.476ms + overhead ~= 34s. Anything near 2s is the old bug.
    assert 20 < estimate["seconds"] < 60, estimate
    assert "steady state" in estimate["basis"]
    assert "order of magnitude" in estimate["caveat"]


def test_expected_quality_is_not_presented_as_a_predicted_return():
    """Nothing here can forecast what a policy will earn. Labelling a setup
    score as expected performance would be the most misleading thing this
    feature could do."""
    from app.services.rl.autotune import Environment, expected_quality

    env = Environment(algo="ppo", bars=1000, regime_confidence=0.9)
    quality = expected_quality(env, {"training": {"episodes": 40}})
    assert 0 <= quality["score"] <= 1
    assert "NOT a predicted return" in quality["meaning"]
    assert all(f["detail"] for f in quality["factors"])


def test_confidence_falls_when_an_input_signal_is_missing():
    """The score has to react to missing evidence, or it is decoration."""
    from app.services.rl.autotune import Environment, confidence

    complete = Environment(algo="ppo", bars=1000, regime="bull_market",
                           regime_confidence=0.9, volatility=0.25)
    blind = Environment(algo="ppo", bars=0, regime="unknown",
                        regime_confidence=0.0, volatility=None)
    assert confidence(complete)["score"] > confidence(blind)["score"] + 0.3
    assert any("regime" in r for r in confidence(blind)["reasons"])


def test_the_full_configuration_is_kept_for_reproducibility():
    """Hidden from a standard user is not the same as discarded: the run still
    has to be replayable from its own record."""
    from app.services.rl.autotune import recommend

    result = recommend("AAPL", "dueling_dqn", period="2y")
    params = result["resolved_hyperparameters"]
    for section in ("training", "optimizer", "network", "replay", "risk"):
        assert params.get(section), f"{section} is missing from the stored config"
    assert result["fingerprint"], "the configuration has no identifier"
    # And it must be a real resolution, not a sketch.
    assert params["training"]["seed"] is not None


def test_the_recommendation_never_claims_to_be_an_optimum():
    """It applies rules; it does not train candidates and compare them.
    Calling that "optimal" would overstate what a rule table can know."""
    from app.services.rl.autotune import recommend

    result = recommend("AAPL", "ppo", period="2y")
    assert "not a search" in result["method"].lower()
    assert "recommendation rather than a proven optimum" in result["method"]


def test_both_training_paths_measure_buy_and_hold_from_the_same_bar():
    """The discrete trainer computed Buy & Hold from `prices[env_cfg.lookback]`
    while the continuous trainer computed it from the first bar of the test
    matrix. Same instrument, same window, two different benchmarks: measured on
    AAPL/2y the discrete path reported +16.45% and the continuous path +22.72%,
    a 6.3-point gap that landed directly in `alpha_vs_buy_hold` and made
    discrete agents look better than continuous ones on identical data.

    The agent cannot trade before it has a full observation window, so the
    benchmark must start at `lookback` on both paths. This asserts the shared
    convention rather than a literal expression, so a refactor that keeps the
    behaviour does not fail it.
    """
    import re
    from pathlib import Path

    src = (Path(__file__).resolve().parents[1]
           / "app" / "services" / "rl" / "service.py").read_text()

    discrete = re.search(r"buy_hold = float\(prices\[-1\] / prices\[(\w+)\] - 1\)", src)
    assert discrete, "the discrete Buy & Hold computation moved or was renamed"
    assert discrete.group(1) == "start", (
        "the discrete baseline no longer starts at the lookback offset")

    continuous = re.search(
        r"buy_hold = float\(test_m\.iloc\[-1, 0\] / test_m\.iloc\[([^,]+), 0\] - 1\)", src)
    assert continuous, "the continuous Buy & Hold computation moved or was renamed"
    start_expr = continuous.group(1).strip()

    assert start_expr != "0", (
        "the continuous path benchmarks from bar 0, crediting Buy & Hold with "
        "the lookback bars the agent was never allowed to trade (+22.72% vs "
        "+16.45% on AAPL/2y)")
    # The start may be an inline expression or a local bound just above; either
    # is fine so long as it resolves from env_cfg.lookback.
    if "lookback" not in start_expr:
        binding = re.search(
            rf"{re.escape(start_expr)}\s*=\s*[^\n]*lookback[^\n]*", src)
        assert binding, (
            f"the continuous baseline starts at {start_expr!r}, which is not "
            "tied to env_cfg.lookback; the two families would not be comparable")
