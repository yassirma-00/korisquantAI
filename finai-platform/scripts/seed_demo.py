#!/usr/bin/env python3
"""Populate the platform with a demo portfolio and a few trained models.

Usage:
    python scripts/seed_demo.py            # quick: 1 forecaster + 1 agent
    python scripts/seed_demo.py --full     # all architectures + several agents
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.services.data.market_data import market_data_service          # noqa: E402
from app.services.forecasting.trainer import TrainConfig, forecast_trainer  # noqa: E402
from app.services.rl.service import rl_service                          # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="train every architecture")
    parser.add_argument("--symbol", default="AAPL")
    args = parser.parse_args()

    symbol = args.symbol.upper()
    print(f"→ fetching history for {symbol}")
    series = market_data_service.get_history(symbol, period="5y")
    print(f"  {len(series.df)} bars from '{series.source}'")

    models = ["lstm", "gru", "tcn", "transformer", "cnn_lstm"] if args.full else ["lstm"]
    for name in models:
        print(f"→ training {name}")
        result = forecast_trainer.train(symbol, series.df, TrainConfig(model=name, epochs=15))
        test = result.metrics["test"]
        print(f"  directional accuracy {test['directional_accuracy']:.1f}%  rmse {test['rmse']:.5f}")

    algos = ["dueling_dqn", "double_dqn", "ppo"] if args.full else ["dueling_dqn"]
    for algo in algos:
        print(f"→ training RL agent {algo}")
        try:
            meta = rl_service.train_single_asset(symbol, period="3y", algo=algo, episodes=10)
            perf = meta["test_performance"]
            print(f"  return {perf['total_return']:+.2%} vs B&H {perf['buy_and_hold_return']:+.2%} "
                  f"(alpha {perf['alpha_vs_buy_hold']:+.2%})")
        except Exception as exc:
            print(f"  skipped: {exc}")

    print("\nDone. Start the server and open http://localhost:8000")


if __name__ == "__main__":
    main()
