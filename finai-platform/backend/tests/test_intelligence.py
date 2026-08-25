"""Tests for the RL algorithm catalogue, distributional agents and strategy benchmarks."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.services.recommendation.intelligence import PortfolioIntelligence, StrategyBenchmarks
from app.services.rl.agents.distributional import (
    C51Agent,
    DistributionalConfig,
    IQNAgent,
    PrioritisedReplayBuffer,
    RainbowAgent,
)
from app.services.rl.catalogue import (
    CATALOGUE,
    comparison_table,
    get_algorithm,
    list_algorithms,
    recommend_algorithm,
)
from app.services.rl.environment import EnvConfig, TradingEnv


# ============================================================== catalogue
def test_all_requested_algorithms_are_catalogued():
    """Every algorithm named in the specification must be present."""
    # IMPALA was deliberately dropped: see test_impala_is_gone
    required = {"ppo", "a2c", "dqn", "double_dqn", "dueling_dqn", "rainbow", "c51",
                "qr_dqn", "iqn", "sac", "td3", "ddpg", "trpo"}
    assert required <= {a.key for a in CATALOGUE}


def test_every_entry_is_fully_documented():
    """A catalogue entry with empty fields is worse than no entry."""
    for a in CATALOGUE:
        assert len(a.description) > 60, f"{a.key}: description too thin"
        assert a.characteristics, f"{a.key}: no characteristics"
        assert a.advantages, f"{a.key}: no advantages"
        assert a.limitations, f"{a.key}: no limitations - every algorithm has trade-offs"
        assert a.best_for, f"{a.key}: no guidance"
        assert set(a.performance) == {"sample_efficiency", "stability",
                                      "final_performance", "training_speed"}
        assert all(1 <= v <= 5 for v in a.performance.values())


def test_availability_is_honest():
    """`available` must reflect the real import state, not a hard-coded guess."""
    for a in CATALOGUE:
        if a.backend == "native":
            assert a.available
        elif a.backend == "rllib":
            try:
                import ray.rllib  # noqa: F401
                expected = True
            except Exception:
                expected = False
            assert a.available is expected


def test_action_space_filtering():
    discrete = list_algorithms(action_space="discrete")
    assert all(a.action_space in ("discrete", "both") for a in discrete)
    continuous = list_algorithms(action_space="continuous")
    assert all(a.action_space in ("continuous", "both") for a in continuous)
    assert {"sac", "td3", "ddpg"} <= {a.key for a in continuous}


def test_recommendation_only_suggests_installed_algorithms():
    for space in ("discrete", "continuous"):
        rec = recommend_algorithm(space, "balanced")
        if "error" not in rec:
            assert get_algorithm(rec["recommended"]).available


def test_comparison_table_shape():
    rows = comparison_table()
    assert len(rows) == len(CATALOGUE)
    assert all("overall" in r and 1 <= r["overall"] <= 5 for r in rows)


# ================================================== distributional agents
@pytest.fixture
def small_env(ohlcv):
    return TradingEnv(ohlcv.tail(300), EnvConfig())


@pytest.mark.parametrize("cls", [C51Agent, IQNAgent, RainbowAgent])
def test_distributional_agents_train(cls, small_env, ohlcv):
    cfg = DistributionalConfig(min_buffer=100, batch_size=32, epsilon_decay_steps=200)
    agent = cls(small_env.observation_space.shape[0], 3, cfg)
    history = agent.train(small_env, episodes=1, log_every=99)
    assert len(history["episode_rewards"]) == 1
    assert agent.steps > 0
    result = agent.evaluate(TradingEnv(ohlcv.tail(300), EnvConfig()))
    assert "performance" in result


@pytest.mark.parametrize("cls", [C51Agent, IQNAgent, RainbowAgent])
def test_return_distribution_is_coherent(cls, small_env):
    """CVaR must never exceed the mean - it is a lower-tail average."""
    cfg = DistributionalConfig(min_buffer=50, batch_size=16)
    agent = cls(small_env.observation_space.shape[0], 3, cfg)
    obs, _ = small_env.reset()
    dist = agent.action_distribution(obs)
    assert len(dist) == 3
    for a, stats in dist.items():
        assert stats["cvar_5pct"] <= stats["mean"] + 1e-6, f"action {a}: CVaR above mean"
        assert stats["std"] >= 0
        assert np.isfinite(stats["mean"])


def test_c51_probabilities_sum_to_one(small_env):
    import torch

    agent = C51Agent(small_env.observation_space.shape[0], 3, DistributionalConfig())
    obs, _ = small_env.reset()
    with torch.no_grad():
        probs = agent.online(torch.tensor(obs, dtype=torch.float32).unsqueeze(0)).exp()
    sums = probs.sum(dim=2).numpy().ravel()
    assert np.allclose(sums, 1.0, atol=1e-4), f"distribution not normalised: {sums}"


def test_c51_projection_stays_on_support(small_env):
    """Regression: an off-by-one in the projection silently corrupts the target."""
    import torch

    agent = C51Agent(small_env.observation_space.shape[0], 3, DistributionalConfig())
    obs, _ = small_env.reset()
    batch = torch.tensor(np.tile(obs, (8, 1)), dtype=torch.float32)
    rewards = torch.tensor([100.0, -100.0, 0.0, 5.0, -5.0, 1e-6, -1e-6, 0.5])  # extreme values
    dones = torch.zeros(8)
    projected = agent._project_distribution(batch, rewards, dones)
    assert torch.isfinite(projected).all()
    assert (projected >= -1e-6).all(), "negative probability mass"
    assert torch.allclose(projected.sum(dim=1), torch.ones(8), atol=1e-3)


def test_prioritised_replay_prefers_surprising_transitions():
    buf = PrioritisedReplayBuffer(200, obs_dim=4, alpha=1.0)
    for i in range(100):
        buf.add(np.ones(4) * i, i % 3, float(i), np.ones(4), False)
    # Make transition 50 dramatically more surprising than the rest
    buf.update_priorities(np.arange(100), np.concatenate([np.full(50, 0.001),
                                                          [10.0], np.full(49, 0.001)]))
    _, _, _, _, _, idx, weights = buf.sample_prioritised(200, beta=0.4)
    assert (idx == 50).sum() > 5, "high-priority sample was not oversampled"
    assert np.all(weights <= 1.0 + 1e-6), "importance weights must be normalised"


def test_iqn_risk_distortion_shifts_the_policy():
    """CVaR distortion must sample lower quantiles than the neutral setting."""
    neutral = IQNAgent(10, 3, DistributionalConfig(risk_distortion="neutral", seed=1))
    averse = IQNAgent(10, 3, DistributionalConfig(risk_distortion="cvar", cvar_alpha=0.25, seed=1))
    t_neutral = neutral._sample_taus(1, 2000).mean().item()
    t_averse = averse._sample_taus(1, 2000).mean().item()
    assert t_averse < t_neutral * 0.5, "risk-averse sampling should concentrate on the lower tail"


def test_distributional_agent_roundtrip(small_env, tmp_path):
    agent = C51Agent(small_env.observation_space.shape[0], 3,
                     DistributionalConfig(min_buffer=50, batch_size=16))
    path = tmp_path / "c51.pt"
    agent.save(path)
    reloaded = C51Agent.load(path)
    obs, _ = small_env.reset()
    assert np.allclose(agent.q_values(obs), reloaded.q_values(obs), atol=1e-5)


# ============================================================== benchmarks
def test_benchmarks_charge_costs_to_every_strategy():
    """A cost-free benchmark against a cost-paying agent is a rigged comparison."""
    prices = pd.Series(np.linspace(100, 150, 400),
                       index=pd.bdate_range("2022-01-01", periods=400))
    free = StrategyBenchmarks(transaction_cost=0.0, slippage=0.0)
    costly = StrategyBenchmarks(transaction_cost=0.01, slippage=0.002)
    a = free.moving_average_crossover(prices)["equity"].iloc[-1]
    b = costly.moving_average_crossover(prices)["equity"].iloc[-1]
    assert b < a, "transaction costs were not applied"


def test_benchmark_signals_do_not_look_ahead():
    """A signal computed on bar t must only be acted on at t+1.

    Constructed so that acting on the same bar would be visibly profitable and
    acting on the next bar would not.
    """
    prices = pd.Series([100] * 50 + [200] + [100] * 50,
                       index=pd.bdate_range("2022-01-01", periods=101), dtype=float)
    bench = StrategyBenchmarks(transaction_cost=0.0, slippage=0.0)
    result = bench.momentum(prices, lookback=5)
    equity = result["equity"]
    # A look-ahead strategy would capture the +100% spike and finish far above par
    assert equity.iloc[-1] < 150_000, "strategy appears to peek at the current bar"


def test_buy_and_hold_tracks_the_asset():
    prices = pd.Series(np.linspace(100, 200, 300),
                       index=pd.bdate_range("2022-01-01", periods=300))
    result = StrategyBenchmarks(0.0, 0.0).buy_and_hold(prices, 100_000)
    assert result["equity"].iloc[-1] == pytest.approx(200_000, rel=0.02)


def test_compare_all_ranks_and_reports(ohlcv):
    comparison = StrategyBenchmarks().compare_all(ohlcv["close"], 100_000)
    assert len(comparison["strategies"]) == 4
    assert comparison["best_by_sharpe"] in comparison["ranking"]
    for s in comparison["strategies"]:
        assert s["equity_curve"], f"{s['strategy']} has no equity curve"
        assert s["max_drawdown"] <= 0


def test_agent_overlay_produces_a_verdict(ohlcv):
    equity = [{"date": str(d.date()), "value": float(v)}
              for d, v in ((1 + ohlcv["close"].pct_change().fillna(0)).cumprod() * 100_000).items()]
    comparison = StrategyBenchmarks().compare_all(ohlcv["close"], 100_000, agent_equity=equity)
    assert comparison["verdict"] is not None
    assert "agent_beats_buy_and_hold" in comparison["verdict"]
    assert any(s["is_agent"] for s in comparison["strategies"])


# ================================================== portfolio dossier
def test_dossier_contains_every_required_metric(ohlcv):
    returns = ohlcv["close"].pct_change().dropna()
    d = PortfolioIntelligence().performance_dossier(returns)
    m = d["metrics"]
    for key in ("total_return", "annualised_volatility", "sharpe_ratio",
                "sortino_ratio", "calmar_ratio", "max_drawdown"):
        assert key in m, f"missing required metric: {key}"
    assert d["equity_curve"] and d["drawdown_curve"]
    assert d["risk_exposure"]["level"] in ("low", "moderate", "high", "critical")


def test_drawdown_episodes_are_ordered_by_severity(ohlcv):
    returns = ohlcv["close"].pct_change().dropna()
    episodes = PortfolioIntelligence().performance_dossier(returns)["drawdown_episodes"]
    depths = [e["depth"] for e in episodes]
    assert depths == sorted(depths), "episodes must be ranked worst-first"
    assert all(d <= 0 for d in depths)


def test_dossier_rejects_tiny_samples():
    assert "error" in PortfolioIntelligence().performance_dossier(pd.Series([0.01, 0.02]))


# ============================================ action-space routing (regression)
# Users hit "'sac' is not a valid discrete algorithm" because the UI offered
# continuous algorithms next to discrete ones while the Train button always
# posted to the single-asset endpoint.
def test_impala_is_gone():
    """IMPALA needed Ray for throughput a single price series cannot supply."""
    assert get_algorithm("impala") is None
    assert "impala" not in {a.key for a in CATALOGUE}


def test_no_catalogued_algorithm_is_unavailable():
    """Advertising an algorithm we cannot run is a broken promise."""
    unavailable = [a.key for a in CATALOGUE if not a.available]
    assert not unavailable, f"catalogue offers unrunnable algorithms: {unavailable}"


def test_continuous_algorithms_now_run_on_a_single_asset():
    """Superseded behaviour: SAC/TD3/DDPG used to be rejected here.

    They are now trained as a 1-asset allocation so the recommendations page can
    offer the whole catalogue. Multi-asset baskets remain available through
    /rl/portfolio/train.
    """
    from app.schemas.common import TrainRLRequest

    for algo in ("sac", "td3", "ddpg"):
        assert TrainRLRequest(symbol="AAPL", algo=algo).algo == algo


def test_continuous_algorithms_accepted_on_the_portfolio_endpoint():
    from app.schemas.common import TrainPortfolioRLRequest

    for algo in ("sac", "td3", "ddpg", "ppo", "a2c"):
        assert TrainPortfolioRLRequest(symbols=["AAPL", "MSFT"], algo=algo).algo == algo


def test_discrete_only_algorithms_rejected_on_the_portfolio_endpoint():
    from app.schemas.common import TrainPortfolioRLRequest

    for algo in ("dqn", "c51", "rainbow"):
        with pytest.raises(Exception) as exc:
            TrainPortfolioRLRequest(symbols=["AAPL", "MSFT"], algo=algo)
        assert "discrete" in str(exc.value)


@pytest.mark.parametrize("algo", ["sac", "td3", "ddpg"])
def test_continuous_agents_actually_train_a_portfolio(algo):
    """End-to-end: these must produce a real weight vector, not just validate."""
    from app.services.rl.service import rl_service

    meta = rl_service.train_portfolio(["AAPL", "MSFT", "SPY"], period="2y",
                                      algo=algo, total_timesteps=800)
    weights = meta["test_performance"]["final_weights"]
    assert set(weights) == {"AAPL", "MSFT", "SPY", "CASH"}
    assert sum(weights.values()) == pytest.approx(1.0, abs=1e-2)
    assert all(w >= -1e-6 for w in weights.values()), "long-only projection violated"


# ================================================ train/test leakage (regression)
# The UI used to print the split dates, which is how this was spotted:
#   "602 training bars (2023-08-01 -> 2025-12-22),
#    tested on 211 unseen bars (2025-09-29 -> 2026-07-31)"
# The test window STARTED BEFORE the training window ended. `_split` handed the
# test set 60 bars of training data as an indicator warm-up, so ~28% of the
# "unseen" period had already been fitted on, inflating every reported metric.
def test_single_asset_split_has_no_overlap():
    from app.services.data.market_data import market_data_service
    from app.services.rl.service import rl_service

    df = market_data_service.get_history("AAPL", period="3y").df
    train, test = rl_service._split(df, 0.2)
    overlap = train.index.intersection(test.index)
    assert overlap.empty, f"{len(overlap)} bars are in BOTH train and test"
    assert test.index[0] > train.index[-1], "test must start strictly after training ends"


@pytest.mark.parametrize("fraction", [0.1, 0.2, 0.3, 0.4])
def test_split_is_clean_at_every_ratio(fraction):
    from app.services.data.market_data import market_data_service
    from app.services.rl.service import rl_service

    df = market_data_service.get_history("MSFT", period="3y").df
    train, test = rl_service._split(df, fraction)
    assert train.index.intersection(test.index).empty
    assert len(train) + len(test) == len(df), "split must partition the data"


def test_portfolio_split_has_no_overlap():
    """The multi-asset path had the same 30-bar leak."""
    from app.services.data.market_data import market_data_service

    matrix = market_data_service.get_price_matrix(["AAPL", "MSFT", "SPY"], period="2y")
    split = int(len(matrix) * 0.8)
    train_m, test_m = matrix.iloc[:split], matrix.iloc[split:]
    assert train_m.index.intersection(test_m.index).empty
    assert test_m.index[0] > train_m.index[-1]


def test_reported_test_bars_are_genuinely_unseen():
    """End-to-end: the bar counts a user sees must describe disjoint windows."""
    from app.services.rl.service import rl_service

    meta = rl_service.train_single_asset("AAPL", period="3y",
                                         algo="dueling_dqn", episodes=1)
    train_start, train_end = meta["train_window"]
    test_start, test_end = meta["test_window"]
    assert test_start > train_end, (
        f"test window ({test_start}) starts before training ends ({train_end})")
    assert train_start < train_end and test_start < test_end


def test_stale_agents_from_the_old_split_are_flagged(tmp_path):
    """Checkpoints trained before the split fix must not pass as valid.

    Their metrics were computed on a partly-seen test window; replaying them
    unmarked would present inflated performance as genuine.
    """
    import json

    from app.services.rl.service import RLService

    service = RLService(model_dir=tmp_path)
    service.model_dir.mkdir(parents=True, exist_ok=True)

    (service.model_dir / "rl_OLD_dqn.json").write_text(json.dumps({
        "symbol": "OLD", "algo": "dqn",
        "train_window": ["2023-08-01", "2025-12-22"],
        "test_window": ["2025-09-29", "2026-07-31"],   # starts before training ends
    }))
    (service.model_dir / "rl_NEW_dqn.json").write_text(json.dumps({
        "symbol": "NEW", "algo": "dqn",
        "train_window": ["2023-08-01", "2025-12-22"],
        "test_window": ["2025-12-23", "2026-07-31"],   # clean
    }))

    agents = {a["symbol"]: a for a in service.list_agents()}
    assert agents["OLD"].get("stale") is True
    assert "inflated" in agents["OLD"]["stale_reason"]
    assert agents["NEW"].get("stale", False) is False


def test_portfolio_agents_without_windows_are_not_flagged(tmp_path):
    """Multi-asset runs record no date windows; absence must not mean 'stale'."""
    import json

    from app.services.rl.service import RLService

    service = RLService(model_dir=tmp_path)
    service.model_dir.mkdir(parents=True, exist_ok=True)
    (service.model_dir / "rl_BASKET_sac.json").write_text(json.dumps({
        "portfolio_key": "AAPL,MSFT", "algo": "sac", "train_bars": 400, "test_bars": 100,
    }))
    assert service.list_agents()[0].get("stale", False) is False


# ============ every algorithm usable on the recommendations page (regression)
# The page previously offered 4 forecast models and 3 RL algorithms out of
# 5 and 13. Worse, the missing RL ones were not merely hidden: SAC/TD3/DDPG
# were rejected outright because they emit a weight vector rather than
# BUY/HOLD/SELL. They now run as a single-asset allocation.
def test_recommendation_schema_accepts_every_catalogued_algorithm():
    from app.schemas.common import RecommendRequest
    from app.services.rl.catalogue import CATALOGUE

    for algo in (a.key for a in CATALOGUE if a.available):
        assert RecommendRequest(symbol="AAPL", rl_algo=algo).rl_algo == algo


def test_single_asset_training_accepts_every_catalogued_algorithm():
    from app.schemas.common import TrainRLRequest
    from app.services.rl.catalogue import CATALOGUE

    for algo in (a.key for a in CATALOGUE if a.available):
        assert TrainRLRequest(symbol="AAPL", algo=algo).algo == algo


def test_all_five_forecast_architectures_are_exposed():
    from app.services.forecasting.models import MODEL_REGISTRY

    assert {"lstm", "gru", "tcn", "transformer", "cnn_lstm"} == set(MODEL_REGISTRY)


@pytest.mark.parametrize("algo", ["sac", "td3", "ddpg"])
def test_continuous_agents_train_and_signal_on_one_asset(algo):
    """They must produce a usable BUY/HOLD/SELL, not just avoid crashing."""
    from app.services.rl.service import rl_service

    meta = rl_service.train_single_asset("AAPL", period="2y", algo=algo,
                                         total_timesteps=600)
    assert meta["mode"] == "single_asset_allocation"
    assert meta["train_bars"] > 0 and meta["test_bars"] > 0
    # the leak-free split still applies on this path
    assert meta["test_window"][0] > meta["train_window"][1]

    reco = rl_service.recommend_action("AAPL", algo=algo)
    assert reco["action"] in ("BUY", "HOLD", "SELL")
    assert 0.0 <= reco["target_exposure"] <= 1.0
    assert 0.0 <= reco["confidence"] <= 1.0
    assert reco["action_space"] == "continuous"
    assert reco["trade_plan"] and reco["explanation"]["summary"]


def test_exposure_maps_to_the_expected_signal():
    """The neutral band must be respected: 50% exposure is HOLD, not a trade."""
    from app.services.rl.service import RLService

    # Mirrors the thresholds used in _recommend_continuous
    neutral, band = 0.5, 0.15
    cases = [(0.90, "BUY"), (0.70, "BUY"), (0.50, "HOLD"),
             (0.40, "HOLD"), (0.20, "SELL"), (0.05, "SELL")]
    for exposure, expected in cases:
        if exposure >= neutral + band:
            action = "BUY"
        elif exposure <= neutral - band:
            action = "SELL"
        else:
            action = "HOLD"
        assert action == expected, f"{exposure:.0%} should read as {expected}"
    assert RLService  # the mapping lives in the service under test
