"""Gymnasium-compatible trading environments.

Two environments are provided:

``TradingEnv``          single asset, discrete actions {SELL, HOLD, BUY}
``PortfolioEnv``        multi asset, continuous target weights (SAC / TD3 / PPO)

Both model realistic frictions: proportional transaction costs, slippage,
optional short-selling ban, and a risk-adjusted reward that penalises
drawdown and volatility so the agent does not simply maximise raw PnL.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from app.core.config import settings
from app.services.indicators.features import build_features

ACTION_NAMES = {0: "SELL", 1: "HOLD", 2: "BUY"}


@dataclass
class EnvConfig:
    initial_balance: float = 100_000.0
    transaction_cost: float = 0.001
    slippage: float = 0.0005
    lookback: int = 20
    trade_fraction: float = 0.25     # fraction of buying power per BUY action
    reward_scaling: float = 100.0
    risk_penalty: float = 0.15       # weight on rolling volatility
    drawdown_penalty: float = 0.35   # weight on running drawdown
    turnover_penalty: float = 0.02   # discourages hyperactive trading
    allow_short: bool = False
    max_steps: int | None = None
    feature_columns: list[str] = field(default_factory=list)

    # ------------------------------------------------------ regime awareness
    # Off by default, and that is deliberate rather than timid: switching it on
    # widens the observation, and a network trained on the narrower vector
    # raises `mat1 and mat2 shapes cannot be multiplied` when handed the wider
    # one. The flag is persisted per agent, so old checkpoints keep loading
    # into the environment they were trained on.
    regime_aware: bool = False
    regime_step: int = 5             # re-classify every N bars
    regime_window: int = 252         # history each classification may read
    # Weight on the regime-scaled risk penalty. 0 keeps the reward untouched
    # while still letting the agent *observe* the regime — useful for isolating
    # which of the two mechanisms is doing the work.
    regime_reward_weight: float = 1.0
    cvar_penalty: float = 0.10       # weight on rolling CVaR of step returns
    cvar_alpha: float = 0.05         # tail probability for VaR / CVaR


class _TailRiskMixin:
    """Rolling VaR/CVaR of the equity curve, shared by both environments.

    Defined once rather than copied: PortfolioEnv called `_tail_risk` while it
    existed only on TradingEnv, which would have raised AttributeError the
    first time a multi-asset agent stepped. Two copies of a risk formula also
    drift, and a tail measure that differs between environments is worse than
    one that is merely imperfect.
    """

    def _tail_risk(self) -> tuple[float, float]:
        window = max(self.cfg.lookback, 20)
        equity = np.asarray(self.equity_curve[-(window + 1):], dtype=np.float64)
        if len(equity) < 12:
            return 0.0, 0.0
        rets = np.diff(equity) / np.maximum(equity[:-1], 1e-9)
        if rets.size < 10:
            return 0.0, 0.0
        var = float(np.quantile(rets, self.cfg.cvar_alpha))
        tail = rets[rets <= var]
        cvar = float(tail.mean()) if tail.size else var
        # Only losses are risk: a 5th percentile above zero means the worst
        # recent day was still a gain, which is not something to penalise.
        return max(-var, 0.0), max(-cvar, 0.0)


class TradingEnv(_TailRiskMixin, gym.Env):
    """Single-asset discrete-action trading environment."""

    metadata = {"render_modes": ["human"]}

    def __init__(self, df: pd.DataFrame, config: EnvConfig | None = None) -> None:
        super().__init__()
        self.cfg = config or EnvConfig()
        self.raw = df.copy()

        feats = build_features(df, dropna=False).ffill().bfill()
        if self.cfg.feature_columns:
            feats = feats[[c for c in self.cfg.feature_columns if c in feats.columns]]
        common = self.raw.index.intersection(feats.index)
        self.raw = self.raw.loc[common]
        self.features = feats.loc[common]

        # Normalise features once (z-score on the whole episode window)
        mu, sigma = self.features.mean(), self.features.std().replace(0, 1.0)
        self.norm_features = ((self.features - mu) / sigma).clip(-5, 5).fillna(0.0).values.astype(np.float32)
        self.prices = self.raw["close"].values.astype(np.float64)
        self.n_steps_total = len(self.prices)

        self.n_features = self.norm_features.shape[1]

        # Regime block: classified once per bar over `self.raw`, so the agent
        # reads a lookup rather than paying ~9.8 ms of classification inside
        # every step() call.
        self.regime = None
        self.n_regime_features = 0
        if self.cfg.regime_aware:
            from app.services.rl.regime_features import (
                REGIME_FEATURE_DIM,
                build_provider,
            )
            self.regime = build_provider(
                self.raw, step=self.cfg.regime_step, window=self.cfg.regime_window)
            self.n_regime_features = REGIME_FEATURE_DIM

        # observation = market features + 5 account features [+ 6 regime]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.n_features + 5 + self.n_regime_features,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)
        self.reset()

    @property
    def price(self) -> float:
        return float(self.prices[self.t])

    @property
    def portfolio_value(self) -> float:
        return self.cash + self.shares * self.price

    def _observation(self) -> np.ndarray:
        value = self.portfolio_value
        exposure = (self.shares * self.price) / value if value > 0 else 0.0
        pnl = value / self.cfg.initial_balance - 1.0
        drawdown = value / self.peak_value - 1.0 if self.peak_value > 0 else 0.0
        recent = self.equity_curve[-self.cfg.lookback:]
        recent_vol = float(np.std(np.diff(recent) / np.maximum(recent[:-1], 1e-9))) if len(recent) > 2 else 0.0
        account = np.array([
            exposure,
            np.clip(pnl, -1, 5),
            np.clip(drawdown, -1, 0),
            np.clip(recent_vol * 10, 0, 5),
            self.t / max(self.n_steps_total - 1, 1),
        ], dtype=np.float32)
        parts = [self.norm_features[self.t], account]
        if self.regime is not None:
            parts.append(self.regime.vector_at(self.t))
        return np.concatenate(parts).astype(np.float32)

    # --------------------------------------------------------------- gym API
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.t = self.cfg.lookback
        self.cash = float(self.cfg.initial_balance)
        self.shares = 0.0
        self.peak_value = float(self.cfg.initial_balance)
        self.equity_curve: list[float] = [float(self.cfg.initial_balance)]
        self.trades: list[dict] = []
        self.n_trades = 0
        self.total_cost = 0.0
        self._last_value = float(self.cfg.initial_balance)
        return self._observation(), {}

    def step(self, action: int):
        action = int(action)
        price = self.price
        prev_value = self.portfolio_value
        traded_notional = 0.0

        if action == 2:  # BUY
            budget = self.cash * self.cfg.trade_fraction
            exec_price = price * (1 + self.cfg.slippage)
            qty = budget / exec_price if exec_price > 0 else 0.0
            cost = qty * exec_price * self.cfg.transaction_cost
            if qty > 0 and budget > cost:
                self.cash -= qty * exec_price + cost
                self.shares += qty
                traded_notional = qty * exec_price
                self.total_cost += cost
                self.n_trades += 1
                self.trades.append({"step": self.t, "date": str(self.raw.index[self.t].date()),
                                    "action": "BUY", "price": exec_price, "qty": qty})
        elif action == 0:  # SELL
            qty = self.shares if not self.cfg.allow_short else self.shares + \
                (self.cash * self.cfg.trade_fraction / max(price, 1e-9))
            if qty > 0:
                exec_price = price * (1 - self.cfg.slippage)
                proceeds = qty * exec_price
                cost = proceeds * self.cfg.transaction_cost
                self.cash += proceeds - cost
                self.shares -= qty
                traded_notional = proceeds
                self.total_cost += cost
                self.n_trades += 1
                self.trades.append({"step": self.t, "date": str(self.raw.index[self.t].date()),
                                    "action": "SELL", "price": exec_price, "qty": qty})

        self.t += 1
        terminated = self.t >= self.n_steps_total - 1
        truncated = bool(self.cfg.max_steps and (self.t - self.cfg.lookback) >= self.cfg.max_steps)

        value = self.portfolio_value
        self.equity_curve.append(value)
        self.peak_value = max(self.peak_value, value)

        # ------------------------------ risk-adjusted reward ---------------
        step_return = (value - prev_value) / max(prev_value, 1e-9)
        drawdown = value / self.peak_value - 1.0
        recent = np.array(self.equity_curve[-self.cfg.lookback:])
        vol = float(np.std(np.diff(recent) / np.maximum(recent[:-1], 1e-9))) if len(recent) > 3 else 0.0
        turnover = traded_notional / max(prev_value, 1e-9)

        # Expected shortfall of the recent step returns. VaR alone says how far
        # the threshold is; CVaR says how bad it is *beyond* the threshold, and
        # it is the coherent measure (VaR is not sub-additive). Penalising it
        # pushes the policy away from strategies whose losing days are rare but
        # catastrophic — precisely what a Sharpe-like penalty on volatility
        # fails to distinguish from ordinary two-sided noise.
        var, cvar = self._tail_risk()

        # Risk aversion scales with the detected regime: the same drawdown is
        # worth punishing harder in a crash than in a quiet bull market.
        aversion = 1.0
        regime_row = None
        if self.regime is not None:
            regime_row = self.regime.at(self.t)
            aversion = 1.0 + self.cfg.regime_reward_weight * (regime_row.risk_aversion - 1.0)

        reward = (
            step_return
            - aversion * self.cfg.risk_penalty * vol
            + aversion * self.cfg.drawdown_penalty * drawdown   # negative -> penalty
            - aversion * self.cfg.cvar_penalty * cvar           # cvar >= 0 magnitude
            - self.cfg.turnover_penalty * turnover
        ) * self.cfg.reward_scaling

        if value <= self.cfg.initial_balance * 0.2:   # ruin
            reward -= 10.0
            terminated = True

        info = {
            "portfolio_value": value, "cash": self.cash, "shares": self.shares,
            "price": price, "action": ACTION_NAMES[action], "step_return": step_return,
            "drawdown": drawdown, "n_trades": self.n_trades,
            "volatility": vol, "var": var, "cvar": cvar,
            "risk_aversion": aversion,
        }
        if regime_row is not None:
            info["regime"] = regime_row.regime
            info["regime_confidence"] = regime_row.confidence
        return self._observation(), float(reward), bool(terminated), bool(truncated), info

    # -------------------------------------------------------------- reports
    def performance(self) -> dict:
        equity = pd.Series(self.equity_curve)
        rets = equity.pct_change().dropna()
        total_return = equity.iloc[-1] / equity.iloc[0] - 1
        periods = 252
        ann_return = (1 + total_return) ** (periods / max(len(equity), 1)) - 1
        ann_vol = float(rets.std() * np.sqrt(periods)) if len(rets) > 2 else 0.0
        sharpe = float(ann_return / ann_vol) if ann_vol > 1e-9 else 0.0
        downside = rets[rets < 0]
        sortino = float(ann_return / (downside.std() * np.sqrt(periods))) if len(downside) > 2 and downside.std() > 0 else 0.0
        dd = float((equity / equity.cummax() - 1).min())
        bh = float(self.prices[self.t] / self.prices[self.cfg.lookback] - 1)
        return {
            "final_value": round(float(equity.iloc[-1]), 2),
            "total_return": round(float(total_return), 4),
            "annualised_return": round(float(ann_return), 4),
            "annualised_volatility": round(ann_vol, 4),
            "sharpe_ratio": round(sharpe, 3),
            "sortino_ratio": round(sortino, 3),
            "max_drawdown": round(dd, 4),
            "calmar_ratio": round(float(ann_return / abs(dd)), 3) if dd < -1e-9 else 0.0,
            "n_trades": self.n_trades,
            "total_transaction_cost": round(self.total_cost, 2),
            "buy_and_hold_return": round(bh, 4),
            "alpha_vs_buy_hold": round(float(total_return - bh), 4),
        }


class PortfolioEnv(_TailRiskMixin, gym.Env):
    """Multi-asset continuous-allocation environment (target weight vector)."""

    metadata = {"render_modes": ["human"]}

    def __init__(self, price_matrix: pd.DataFrame, config: EnvConfig | None = None,
                 ohlcv: dict[str, pd.DataFrame] | None = None) -> None:
        super().__init__()
        self.cfg = config or EnvConfig()
        self.prices_df = price_matrix.dropna().copy()
        self.symbols = list(self.prices_df.columns)
        self.prices = self.prices_df.values.astype(np.float64)
        self.returns = np.vstack([
            np.zeros((1, len(self.symbols))),
            np.diff(self.prices, axis=0) / np.maximum(self.prices[:-1], 1e-9),
        ])
        self.n_assets = len(self.symbols)
        self.n_steps_total = len(self.prices)

        # Per-asset regime tracks. Assets are routinely in different regimes at
        # once, and that difference is the signal an allocator exists to use;
        # collapsing it to one market-wide label would discard it.
        self.regime = None
        self.n_regime_features = 0
        if self.cfg.regime_aware:
            from app.services.rl.portfolio_regime import (
                build_portfolio_provider,
                feature_dim,
            )
            self.regime = build_portfolio_provider(
                self.prices_df, ohlcv=ohlcv,
                step=self.cfg.regime_step, window=self.cfg.regime_window)
            self.n_regime_features = feature_dim(self.n_assets)

        obs_dim = (self.n_assets * (self.cfg.lookback + 3) + 3
                   + self.n_regime_features)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        # weights over assets + cash, squashed through softmax in `step`
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.n_assets + 1,), dtype=np.float32)
        self.reset()

    def _observation(self) -> np.ndarray:
        window = self.returns[self.t - self.cfg.lookback: self.t]
        flat = (window.T.ravel() * 20).astype(np.float32)
        mom = np.array([
            self.prices[self.t, i] / self.prices[self.t - min(21, self.t), i] - 1
            for i in range(self.n_assets)
        ], dtype=np.float32)
        vol = window.std(axis=0).astype(np.float32) * np.sqrt(252)
        value = self.portfolio_value
        drawdown = value / self.peak_value - 1 if self.peak_value else 0.0
        account = np.array([
            np.clip(value / self.cfg.initial_balance - 1, -1, 5),
            np.clip(drawdown, -1, 0),
            self.t / max(self.n_steps_total - 1, 1),
        ], dtype=np.float32)
        parts = [flat, mom, vol, self.weights[:-1], account]
        if self.regime is not None:
            parts.append(self.regime.vector_at(self.t))
        return np.concatenate(parts).astype(np.float32)

    @property
    def portfolio_value(self) -> float:
        return float(self.cash + np.dot(self.holdings, self.prices[self.t]))

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.t = self.cfg.lookback
        self.cash = float(self.cfg.initial_balance)
        self.holdings = np.zeros(self.n_assets)
        self.weights = np.zeros(self.n_assets + 1, dtype=np.float32)
        self.weights[-1] = 1.0   # all cash
        self.peak_value = float(self.cfg.initial_balance)
        self.equity_curve = [float(self.cfg.initial_balance)]
        self.weight_history: list[np.ndarray] = []
        self.total_cost = 0.0
        return self._observation(), {}

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float64).ravel()
        exp = np.exp(action - action.max())
        target = exp / exp.sum()                       # simplex projection (long-only)

        value_before = self.portfolio_value
        current_asset_value = self.holdings * self.prices[self.t]
        target_asset_value = target[:-1] * value_before
        delta = target_asset_value - current_asset_value
        turnover = float(np.abs(delta).sum() / max(value_before, 1e-9))
        cost = float(np.abs(delta).sum() * (self.cfg.transaction_cost + self.cfg.slippage))

        self.holdings = target_asset_value / np.maximum(self.prices[self.t], 1e-9)
        self.cash = value_before - target_asset_value.sum() - cost
        self.total_cost += cost
        self.weights = target.astype(np.float32)
        self.weight_history.append(self.weights.copy())

        self.t += 1
        terminated = self.t >= self.n_steps_total - 1
        truncated = bool(self.cfg.max_steps and (self.t - self.cfg.lookback) >= self.cfg.max_steps)

        value = self.portfolio_value
        self.equity_curve.append(value)
        self.peak_value = max(self.peak_value, value)

        step_return = (value - value_before) / max(value_before, 1e-9)
        recent = np.array(self.equity_curve[-self.cfg.lookback:])
        vol = float(np.std(np.diff(recent) / np.maximum(recent[:-1], 1e-9))) if len(recent) > 3 else 0.0
        drawdown = value / self.peak_value - 1
        var, cvar = self._tail_risk()

        # Risk aversion follows the *allocation*, not the market as a whole:
        # holding a crashing asset is penalised, holding cash through the same
        # crash is not. Weighting by `target` is what makes moving to cash an
        # actual reduction in penalty rather than only a forgone return.
        aversion = 1.0
        if self.regime is not None:
            aversion = 1.0 + self.cfg.regime_reward_weight * (
                self.regime.aversion_for(self.t - 1, target) - 1.0)

        reward = (
            step_return
            - aversion * self.cfg.risk_penalty * vol
            + aversion * self.cfg.drawdown_penalty * drawdown
            - aversion * self.cfg.cvar_penalty * cvar
            - self.cfg.turnover_penalty * turnover
        ) * self.cfg.reward_scaling

        info = {"portfolio_value": value, "weights": dict(zip(self.symbols + ["CASH"], target.round(4), strict=False)),
                "turnover": turnover, "drawdown": drawdown,
                "volatility": vol, "var": var, "cvar": cvar,
                "risk_aversion": aversion}
        if self.regime is not None:
            snapshot = self.regime.snapshot(self.t - 1)
            info["regime_mean_risk"] = snapshot.get("mean_risk")
            info["regimes_in_force"] = snapshot.get("regimes_in_force")
            info["worst_crash_probability"] = snapshot.get("worst_crash_probability")
        return self._observation(), float(reward), bool(terminated), bool(truncated), info

    def performance(self) -> dict:
        equity = pd.Series(self.equity_curve)
        rets = equity.pct_change().dropna()
        total_return = float(equity.iloc[-1] / equity.iloc[0] - 1)
        ann_return = (1 + total_return) ** (252 / max(len(equity), 1)) - 1
        ann_vol = float(rets.std() * np.sqrt(252)) if len(rets) > 2 else 0.0
        dd = float((equity / equity.cummax() - 1).min())
        equal_weight = float(np.mean(self.prices[self.t] / self.prices[self.cfg.lookback] - 1))
        return {
            "final_value": round(float(equity.iloc[-1]), 2),
            "total_return": round(total_return, 4),
            "annualised_return": round(float(ann_return), 4),
            "annualised_volatility": round(ann_vol, 4),
            "sharpe_ratio": round(float(ann_return / ann_vol), 3) if ann_vol > 1e-9 else 0.0,
            "max_drawdown": round(dd, 4),
            "total_transaction_cost": round(self.total_cost, 2),
            "equal_weight_return": round(equal_weight, 4),
            "alpha_vs_equal_weight": round(total_return - equal_weight, 4),
            "final_weights": dict(zip(self.symbols + ["CASH"], self.weights.round(4).tolist(), strict=False)),
        }


def make_env(df: pd.DataFrame, config: EnvConfig | None = None) -> TradingEnv:
    cfg = config or EnvConfig(
        initial_balance=settings.RL_INITIAL_BALANCE,
        transaction_cost=settings.RL_TRANSACTION_COST,
    )
    return TradingEnv(df, cfg)
