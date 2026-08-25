"""Intelligent portfolio layer: full performance analytics + strategy benchmarks.

Two capabilities the basic portfolio service does not provide:

1. **A complete performance dossier** — Sharpe, Sortino, Calmar, max drawdown,
   exposure, allocation, rolling history — computed from an actual equity path
   rather than a single snapshot.

2. **Honest strategy comparison** — Buy & Hold, Moving-Average Crossover and
   Momentum, all run through the *same* environment with the *same* transaction
   costs as the RL agent. Comparing a cost-free benchmark against a cost-paying
   agent is the most common way backtests flatter themselves; this module
   refuses to do that.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.core.logging import get_logger
from app.services.data.market_data import market_data_service
from app.services.indicators.technical import rsi, sma
from app.services.risk.metrics import full_metrics

logger = get_logger(__name__)

TRADING_DAYS = 252


# ============================================================== benchmarks
class StrategyBenchmarks:
    """Reference strategies evaluated under identical, realistic frictions."""

    def __init__(self, transaction_cost: float = 0.001, slippage: float = 0.0005) -> None:
        self.cost = transaction_cost
        self.slippage = slippage

    def _simulate(self, prices: pd.Series, signals: pd.Series,
                  initial: float = 100_000.0) -> dict:
        """Run a 0/1 exposure signal through a cost-aware equity simulation.

        ``signals`` must already be shifted so that a signal computed on bar *t*
        is only acted upon at *t+1* — otherwise the backtest peeks at data it
        could not have known.
        """
        prices = prices.astype(float)
        signals = signals.reindex(prices.index).fillna(0.0).clip(0, 1)
        returns = prices.pct_change().fillna(0.0)

        turnover = signals.diff().abs().fillna(signals.iloc[0])
        friction = turnover * (self.cost + self.slippage)
        strategy_returns = signals * returns - friction

        equity = (1 + strategy_returns).cumprod() * initial
        n_trades = int((turnover > 1e-9).sum())
        return {
            "equity": equity,
            "returns": strategy_returns,
            "n_trades": n_trades,
            "total_cost": float((friction * equity.shift(1).fillna(initial)).sum()),
        }

    def buy_and_hold(self, prices: pd.Series, initial: float = 100_000.0) -> dict:
        signals = pd.Series(1.0, index=prices.index)
        return self._simulate(prices, signals, initial)

    def moving_average_crossover(self, prices: pd.Series, fast: int = 20, slow: int = 50,
                                 initial: float = 100_000.0) -> dict:
        f, s = sma(prices, fast), sma(prices, slow)
        raw = (f > s).astype(float)
        return self._simulate(prices, raw.shift(1), initial)   # act on the NEXT bar

    def momentum(self, prices: pd.Series, lookback: int = 63, threshold: float = 0.0,
                 initial: float = 100_000.0) -> dict:
        mom = prices.pct_change(lookback)
        raw = (mom > threshold).astype(float)
        return self._simulate(prices, raw.shift(1), initial)

    def rsi_mean_reversion(self, prices: pd.Series, low: int = 30, high: int = 70,
                           initial: float = 100_000.0) -> dict:
        values = rsi(prices, 14)
        position, out = 0.0, []
        for v in values:
            if v < low:
                position = 1.0
            elif v > high:
                position = 0.0
            out.append(position)
        return self._simulate(prices, pd.Series(out, index=prices.index).shift(1), initial)

    def compare_all(self, prices: pd.Series, initial: float = 100_000.0,
                    agent_equity: list[dict] | None = None) -> dict:
        """Rank every reference strategy, optionally including the RL agent."""
        strategies = {
            "buy_and_hold": self.buy_and_hold(prices, initial),
            "ma_crossover_20_50": self.moving_average_crossover(prices, 20, 50, initial),
            "momentum_63d": self.momentum(prices, 63, 0.0, initial),
            "rsi_mean_reversion": self.rsi_mean_reversion(prices, 30, 70, initial),
        }

        rows = []
        for name, result in strategies.items():
            equity = result["equity"].dropna()
            if equity.empty:
                continue
            metrics = full_metrics(result["returns"].dropna())
            rows.append({
                "strategy": name,
                "label": name.replace("_", " ").title(),
                "final_value": round(float(equity.iloc[-1]), 2),
                "total_return": metrics.get("total_return"),
                "annualised_return": metrics.get("annualised_return"),
                "volatility": metrics.get("annualised_volatility"),
                "sharpe_ratio": metrics.get("sharpe_ratio"),
                "sortino_ratio": metrics.get("sortino_ratio"),
                "calmar_ratio": metrics.get("calmar_ratio"),
                "max_drawdown": metrics.get("max_drawdown"),
                "win_rate": metrics.get("win_rate"),
                "n_trades": result["n_trades"],
                "is_agent": False,
                "equity_curve": [
                    {"date": str(d.date()), "value": round(float(v), 2)}
                    for d, v in equity.iloc[:: max(len(equity) // 180, 1)].items()
                ],
            })

        if agent_equity:
            values = pd.Series([p["value"] for p in agent_equity])
            agent_returns = values.pct_change().dropna()
            metrics = full_metrics(agent_returns)
            rows.append({
                "strategy": "rl_agent", "label": "RL Agent",
                "final_value": round(float(values.iloc[-1]), 2),
                "total_return": metrics.get("total_return"),
                "annualised_return": metrics.get("annualised_return"),
                "volatility": metrics.get("annualised_volatility"),
                "sharpe_ratio": metrics.get("sharpe_ratio"),
                "sortino_ratio": metrics.get("sortino_ratio"),
                "calmar_ratio": metrics.get("calmar_ratio"),
                "max_drawdown": metrics.get("max_drawdown"),
                "win_rate": metrics.get("win_rate"),
                "n_trades": None, "is_agent": True,
                "equity_curve": agent_equity[:: max(len(agent_equity) // 180, 1)],
            })

        ranked = sorted(rows, key=lambda r: (r["sharpe_ratio"] or -99), reverse=True)
        best = ranked[0] if ranked else None
        bh = next((r for r in rows if r["strategy"] == "buy_and_hold"), None)
        agent = next((r for r in rows if r["is_agent"]), None)

        verdict = None
        if agent and bh:
            alpha = (agent["total_return"] or 0) - (bh["total_return"] or 0)
            verdict = {
                "agent_beats_buy_and_hold": bool(alpha > 0),
                "alpha_vs_buy_and_hold": round(alpha, 4),
                "note": (
                    "The agent outperformed passive exposure after costs."
                    if alpha > 0 else
                    "The agent did NOT beat buy-and-hold after costs. This is a common and "
                    "informative outcome: transaction costs and noise defeat most active "
                    "strategies on a single asset."),
            }

        return {
            "strategies": rows,
            "ranking": [r["strategy"] for r in ranked],
            "best_by_sharpe": best["strategy"] if best else None,
            "verdict": verdict,
            "cost_model": {"transaction_cost": self.cost, "slippage": self.slippage,
                           "note": "Every strategy pays identical costs - benchmarks are not cost-free."},
        }


# ================================================== portfolio intelligence
class PortfolioIntelligence:
    """Full analytics dossier for a portfolio or a single instrument."""

    def __init__(self) -> None:
        self.benchmarks = StrategyBenchmarks()

    def performance_dossier(self, returns: pd.Series, equity: pd.Series | None = None,
                            benchmark_returns: pd.Series | None = None,
                            initial_capital: float = 100_000.0) -> dict:
        returns = pd.Series(returns).dropna()
        if len(returns) < 20:
            return {"error": "need at least 20 return observations"}
        if equity is None:
            equity = (1 + returns).cumprod() * initial_capital

        metrics = full_metrics(returns, benchmark_returns)
        drawdown = equity / equity.cummax() - 1

        # Rolling views expose *when* a strategy worked, not just the average
        window = min(63, max(len(returns) // 4, 21))
        rolling_sharpe = (returns.rolling(window).mean() /
                          returns.rolling(window).std().replace(0, np.nan)) * np.sqrt(TRADING_DAYS)
        rolling_vol = returns.rolling(window).std() * np.sqrt(TRADING_DAYS)

        monthly = returns.resample("ME").apply(lambda r: (1 + r).prod() - 1) \
            if isinstance(returns.index, pd.DatetimeIndex) else pd.Series(dtype=float)

        underwater = self._drawdown_episodes(drawdown)
        return {
            "metrics": metrics,
            "equity_curve": [{"date": str(d.date()), "value": round(float(v), 2)}
                             for d, v in equity.items()] if isinstance(equity.index, pd.DatetimeIndex)
            else [{"index": i, "value": round(float(v), 2)} for i, v in enumerate(equity)],
            "drawdown_curve": [{"date": str(d.date()), "drawdown": round(float(v), 5)}
                               for d, v in drawdown.items()] if isinstance(drawdown.index, pd.DatetimeIndex) else [],
            "rolling_sharpe": [{"date": str(d.date()), "value": round(float(v), 3)}
                               for d, v in rolling_sharpe.dropna().items()]
            if isinstance(rolling_sharpe.index, pd.DatetimeIndex) else [],
            "rolling_volatility": [{"date": str(d.date()), "value": round(float(v), 4)}
                                   for d, v in rolling_vol.dropna().items()]
            if isinstance(rolling_vol.index, pd.DatetimeIndex) else [],
            "monthly_returns": [{"month": str(d.date())[:7], "return": round(float(v), 4)}
                                for d, v in monthly.dropna().items()] if len(monthly) else [],
            "drawdown_episodes": underwater,
            "risk_exposure": self._risk_exposure(metrics, returns),
        }

    @staticmethod
    def _drawdown_episodes(drawdown: pd.Series, threshold: float = -0.05,
                           top_n: int = 5) -> list[dict]:
        """Identify and rank the worst underwater periods."""
        episodes, in_dd, start, trough, trough_date = [], False, None, 0.0, None
        for date, value in drawdown.items():
            if not in_dd and value < threshold:
                in_dd, start, trough, trough_date = True, date, value, date
            elif in_dd:
                if value < trough:
                    trough, trough_date = value, date
                if value >= -1e-9:
                    episodes.append({
                        "start": str(start.date()) if hasattr(start, "date") else str(start),
                        "trough": str(trough_date.date()) if hasattr(trough_date, "date") else str(trough_date),
                        "recovered": str(date.date()) if hasattr(date, "date") else str(date),
                        "depth": round(float(trough), 4),
                        "duration_days": int((date - start).days) if hasattr(date, "days") or hasattr(start, "date") else None,
                    })
                    in_dd = False
        if in_dd:
            episodes.append({
                "start": str(start.date()) if hasattr(start, "date") else str(start),
                "trough": str(trough_date.date()) if hasattr(trough_date, "date") else str(trough_date),
                "recovered": None, "depth": round(float(trough), 4),
                "duration_days": None, "ongoing": True,
            })
        return sorted(episodes, key=lambda e: e["depth"])[:top_n]

    @staticmethod
    def _risk_exposure(metrics: dict, returns: pd.Series) -> dict:
        vol = metrics.get("annualised_volatility", 0) or 0
        dd = abs(metrics.get("max_drawdown", 0) or 0)
        var95 = abs(metrics.get("var_95", 0) or 0)
        score = float(np.clip(0.4 * min(vol / 0.5, 1) + 0.35 * min(dd / 0.4, 1)
                              + 0.25 * min(var95 / 0.05, 1), 0, 1))
        level = ("low" if score < 0.3 else "moderate" if score < 0.55
                 else "high" if score < 0.78 else "critical")
        return {
            "score": round(score, 3), "level": level,
            "annualised_volatility": round(vol, 4),
            "max_drawdown": round(-dd, 4),
            "daily_var_95": round(-var95, 4),
            "downside_deviation": round(float(returns[returns < 0].std() * np.sqrt(TRADING_DAYS)), 4)
            if (returns < 0).any() else 0.0,
            "tail_ratio": round(float(abs(np.percentile(returns, 95) / np.percentile(returns, 5))), 3)
            if len(returns) > 20 and np.percentile(returns, 5) != 0 else None,
            "interpretation": {
                "low": "Conservative risk profile suitable for capital preservation.",
                "moderate": "Balanced risk; drawdowns are recoverable within a normal horizon.",
                "high": "Aggressive profile - position sizes should be reduced accordingly.",
                "critical": "Extreme risk. Losses of this magnitude are difficult to recover from.",
            }[level],
        }

    def instrument_dossier(self, symbol: str, period: str = "2y",
                           benchmark: str = "SPY", initial_capital: float = 100_000.0,
                           agent_equity: list[dict] | None = None) -> dict:
        """Everything the dashboard needs about one instrument, in one call."""
        series = market_data_service.get_history(symbol, period=period)
        prices = series.df["close"]
        returns = prices.pct_change().dropna()

        bench_returns = None
        if benchmark and benchmark.upper() != symbol.upper():
            try:
                bench = market_data_service.get_history(benchmark, period=period).df["close"]
                bench_returns = bench.pct_change().dropna().reindex(returns.index).dropna()
                returns_aligned = returns.reindex(bench_returns.index)
            except Exception as exc:
                logger.debug("benchmark unavailable: %s", exc)
                returns_aligned = returns
        else:
            returns_aligned = returns

        dossier = self.performance_dossier(returns_aligned, benchmark_returns=bench_returns,
                                           initial_capital=initial_capital)
        comparison = self.benchmarks.compare_all(prices, initial_capital, agent_equity)
        return {
            "symbol": series.symbol,
            "name": series.instrument.name,
            "asset_class": series.instrument.asset_class,
            "period": period,
            "benchmark": benchmark,
            "data_source": series.source,
            "last_price": round(float(prices.iloc[-1]), 4),
            **dossier,
            "strategy_comparison": comparison,
        }


portfolio_intelligence = PortfolioIntelligence()
strategy_benchmarks = StrategyBenchmarks()
