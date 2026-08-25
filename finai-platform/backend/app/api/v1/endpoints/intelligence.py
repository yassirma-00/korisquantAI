"""Endpoints for the RL algorithm catalogue, agent decisions and portfolio intelligence."""

from __future__ import annotations

from fastapi import APIRouter, Query

from app.core.exceptions import InvalidRequestError
from app.services.data.market_data import market_data_service
from app.services.data.universe import DEFAULT_WATCHLIST, list_instruments
from app.services.recommendation.intelligence import portfolio_intelligence, strategy_benchmarks
from app.services.rl.catalogue import (
    BACKENDS,
    comparison_table,
    get_algorithm,
    list_algorithms,
    recommend_algorithm,
)
from app.services.rl.service import rl_service

router = APIRouter(prefix="/intel", tags=["Intelligence"])


# =========================================================== RL catalogue
@router.get("/algorithms", summary="Full RL algorithm catalogue with descriptions")
async def algorithms(
    action_space: str | None = Query(None, pattern="^(discrete|continuous|both)$"),
    family: str | None = Query(None, pattern="^(value_based|policy_gradient|actor_critic|distributional)$"),
    available_only: bool = Query(False),
):
    """Every supported algorithm with its characteristics, advantages and limits.

    ``available`` reflects what this installation can actually run — the UI
    should disable anything false rather than let a training job fail later.
    """
    items = list_algorithms(action_space, family, available_only)
    return {
        "count": len(items),
        "algorithms": [a.to_dict() for a in items],
        "backends": BACKENDS,
        "families": {
            "value_based": "Learn Q(s,a) and act greedily. Sample-efficient, discrete only.",
            "distributional": "Learn the full return distribution, not just its mean - exposes tail risk.",
            "policy_gradient": "Optimise the policy directly. Stable, handles continuous actions.",
            "actor_critic": "Combine a policy (actor) with a value estimate (critic).",
        },
        "note": ("Performance ratings are qualitative, drawn from published benchmarks. "
                 "They are NOT trading returns - those come only from this platform's backtests."),
    }


@router.get("/algorithms/compare", summary="Side-by-side algorithm comparison table")
async def compare_algorithms():
    return {"algorithms": comparison_table(),
            "rating_scale": "1 (weak) to 5 (excellent)",
            "dimensions": {
                "sample_efficiency": "How much data is needed to learn",
                "stability": "How reliably training converges",
                "final_performance": "Quality of the converged policy",
                "training_speed": "Wall-clock cost per environment step",
            }}


@router.get("/algorithms/recommend", summary="Suggest an algorithm for a given need")
async def suggest_algorithm(
    action_space: str = Query("discrete", pattern="^(discrete|continuous)$"),
    priority: str = Query("balanced",
                          pattern="^(balanced|sample_efficiency|stability|performance|speed)$"),
):
    return recommend_algorithm(action_space, priority)


@router.get("/algorithms/{key}", summary="Detail for one algorithm")
async def algorithm_detail(key: str):
    algo = get_algorithm(key)
    if algo is None:
        raise InvalidRequestError(f"Unknown algorithm '{key}'",
                                  details={"available": [a.key for a in list_algorithms()]})
    return algo.to_dict()


# ============================================================== symbols
@router.get("/symbols", summary="Selectable instruments grouped by asset class")
async def symbols(q: str | None = Query(None), asset_class: str | None = Query(None)):
    """Powers the symbol picker: grouped, searchable, with live-quote metadata."""
    items = list_instruments(asset_class=asset_class, query=q)
    grouped: dict[str, list[dict]] = {}
    for inst in items:
        grouped.setdefault(inst.asset_class, []).append({
            "symbol": inst.symbol, "name": inst.name,
            "exchange": inst.exchange, "currency": inst.currency,
            "sector": inst.sector,
        })
    labels = {
        "equity": "Stocks", "etf": "ETFs", "crypto": "Cryptocurrencies",
        "index": "Indices", "forex": "Forex", "commodity": "Commodities",
    }
    return {
        "count": len(items),
        "groups": [{"key": k, "label": labels.get(k, k.title()), "instruments": v}
                   for k, v in sorted(grouped.items())],
        "default_watchlist": DEFAULT_WATCHLIST,
        "custom_symbols_allowed": True,
        "hint": "Any Yahoo Finance ticker works, even if not listed here (e.g. 'SHOP', 'DOGE-USD').",
    }


# ======================================================= agent decision
@router.get("/agent-decision/{symbol}", summary="Full RL agent decision with trade plan")
async def agent_decision(
    symbol: str,
    algo: str = Query("dueling_dqn"),
    period: str = Query("1y"),
    variant: str = Query("", description=(
        "Empty for the original agent, 'regime' for its regime-aware twin. "
        "Only the twin can attribute the market regime to a decision.")),
):
    """The complete AI Agent Decision panel: action, confidence, risk, sizing,
    stop-loss, take-profit, horizon and a natural-language explanation."""
    return rl_service.recommend_action(symbol, algo=algo, period=period,
                                       variant=variant)


@router.get("/agent-decision/{symbol}/available", summary="Which agents exist for this symbol")
async def available_agents(symbol: str):
    trained = rl_service.list_agents()
    sym = symbol.upper()
    matching = [a for a in trained if (a.get("symbol") or "").upper() == sym]
    return {
        "symbol": sym,
        "trained_algorithms": [
            {"algo": a["algo"],
             "trained_at": a.get("trained_at"),
             "test_performance": a.get("test_performance", {})}
            for a in matching
        ],
        "count": len(matching),
    }


# ==================================================== portfolio dossier
@router.get("/portfolio-analytics/{symbol}", summary="Complete performance dossier")
async def portfolio_analytics(
    symbol: str,
    period: str = Query("2y"),
    benchmark: str = Query("SPY"),
    capital: float = Query(100_000.0, gt=0),
    include_agent: bool = Query(False, description="Overlay a trained RL agent's equity curve"),
    algo: str = Query("dueling_dqn"),
):
    agent_equity = None
    if include_agent:
        try:
            backtest = rl_service.backtest(symbol, algo=algo, period=period)
            agent_equity = backtest.get("equity_curve")
        except Exception:
            agent_equity = None      # not trained yet - the dossier still works
    return portfolio_intelligence.instrument_dossier(
        symbol, period=period, benchmark=benchmark,
        initial_capital=capital, agent_equity=agent_equity)


@router.get("/benchmarks/{symbol}", summary="Buy & Hold vs MA Crossover vs Momentum vs RSI")
async def benchmarks(
    symbol: str,
    period: str = Query("2y"),
    capital: float = Query(100_000.0, gt=0),
    transaction_cost: float = Query(0.001, ge=0, le=0.05),
    include_agent: bool = Query(False),
    algo: str = Query("dueling_dqn"),
):
    """All reference strategies under identical, realistic transaction costs."""
    series = market_data_service.get_history(symbol, period=period)
    bench = strategy_benchmarks
    bench.cost = transaction_cost

    agent_equity = None
    if include_agent:
        try:
            agent_equity = rl_service.backtest(symbol, algo=algo, period=period).get("equity_curve")
        except Exception:
            agent_equity = None

    result = bench.compare_all(series.df["close"], capital, agent_equity)
    return {"symbol": series.symbol, "period": period,
            "data_source": series.source, **result}
