"""Reinforcement-learning endpoints (training, backtesting, live actions)."""

from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, Depends, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.logging import get_logger
from app.db.models import RecommendationLog
from app.db.session import get_db
from app.schemas.common import TrainPortfolioRLRequest, TrainRLRequest
from app.services.rl.agents.policy_gradient import SB3_AVAILABLE
from app.services.rl.audit import log_allocation_decision, log_rl_decision
from app.services.rl.service import SUPPORTED_ALGOS, rl_service

logger = get_logger(__name__)
router = APIRouter(prefix="/rl", tags=["Reinforcement Learning"])

RL_JOBS: dict[str, dict] = {}


@router.get("/algorithms", summary="Supported RL algorithms")
async def list_algorithms():
    return {
        "algorithms": [
            {"key": "dqn", "name": "Deep Q-Network", "action_space": "discrete",
             "native": True, "description": "Value-based baseline with replay buffer and target network."},
            {"key": "double_dqn", "name": "Double DQN", "action_space": "discrete",
             "native": True, "description": "Reduces Q-value overestimation bias."},
            {"key": "dueling_dqn", "name": "Dueling Double DQN", "action_space": "discrete",
             "native": True, "description": "Separate value and advantage streams - best discrete default."},
            {"key": "ppo", "name": "Proximal Policy Optimization", "action_space": "both",
             "native": SB3_AVAILABLE, "description": "Stable on-policy method, works for single asset and portfolios."},
            {"key": "a2c", "name": "Advantage Actor-Critic", "action_space": "both",
             "native": SB3_AVAILABLE, "description": "Synchronous actor-critic, fast but noisier."},
            {"key": "sac", "name": "Soft Actor-Critic", "action_space": "continuous",
             "native": SB3_AVAILABLE, "description": "Entropy-regularised, sample-efficient portfolio allocation."},
            {"key": "td3", "name": "Twin Delayed DDPG", "action_space": "continuous",
             "native": SB3_AVAILABLE, "description": "Deterministic policy with twin critics for allocation."},
        ],
        "supported": SUPPORTED_ALGOS,
        "stable_baselines3_available": SB3_AVAILABLE,
        "environments": {
            "TradingEnv": "Single asset, discrete actions {SELL, HOLD, BUY}",
            "PortfolioEnv": "Multi asset, continuous target weights over assets + cash",
        },
    }


@router.post("/train", summary="Train a single-asset trading agent")
async def train_agent(request: TrainRLRequest):
    return rl_service.train_single_asset(
        symbol=request.symbol, period=request.period, algo=request.algo,
        episodes=request.episodes, total_timesteps=request.total_timesteps,
        test_fraction=request.test_fraction, profile=request.profile,
        variant=request.variant,
        env_overrides={
            "initial_balance": request.initial_balance,
            "transaction_cost": request.transaction_cost,
            "risk_penalty": request.risk_penalty,
            "regime_aware": request.regime_aware,
            "cvar_penalty": request.cvar_penalty,
            "regime_reward_weight": request.regime_reward_weight,
        },
    )


@router.post("/train/async", summary="Train an agent in the background")
async def train_async(request: TrainRLRequest, background_tasks: BackgroundTasks):
    job_id = f"{request.symbol.upper()}_{request.algo}"
    RL_JOBS[job_id] = {"status": "queued", "request": request.model_dump()}

    def _run() -> None:
        RL_JOBS[job_id]["status"] = "running"
        try:
            meta = rl_service.train_single_asset(
                symbol=request.symbol, period=request.period, algo=request.algo,
                episodes=request.episodes, total_timesteps=request.total_timesteps,
                test_fraction=request.test_fraction, profile=request.profile,
                env_overrides={"initial_balance": request.initial_balance,
                               "transaction_cost": request.transaction_cost,
                               "risk_penalty": request.risk_penalty,
                               "regime_aware": request.regime_aware,
                               "cvar_penalty": request.cvar_penalty,
                               "regime_reward_weight": request.regime_reward_weight})
            RL_JOBS[job_id] = {"status": "completed",
                               "test_performance": meta["test_performance"],
                               "baselines": meta["baselines"],
                               "trained_at": meta["trained_at"]}
        except Exception as exc:
            logger.exception("RL async training failed")
            RL_JOBS[job_id] = {"status": "failed", "error": str(exc)[:400]}

    background_tasks.add_task(_run)
    return {"job_id": job_id, "status": "queued", "poll": f"{settings.API_V1_PREFIX}/rl/jobs/{job_id}"}


@router.get("/jobs/{job_id}", summary="Poll an RL training job")
async def job_status(job_id: str):
    return RL_JOBS.get(job_id, {"status": "unknown", "job_id": job_id})


@router.post("/portfolio/train", summary="Train a multi-asset allocation agent")
async def train_portfolio_agent(request: TrainPortfolioRLRequest):
    return rl_service.train_portfolio(
        symbols=request.symbols, period=request.period, algo=request.algo,
        total_timesteps=request.total_timesteps, profile=request.profile,
        variant=request.variant,
        env_overrides={"initial_balance": request.initial_balance,
                       "transaction_cost": request.transaction_cost,
                       "regime_aware": request.regime_aware,
                       "cvar_penalty": request.cvar_penalty,
                       "regime_reward_weight": request.regime_reward_weight},
    )


@router.get("/action/{symbol}", summary="Live action recommended by the agent")
async def recommend_action(
    symbol: str,
    algo: str = Query("dueling_dqn"),
    period: str = Query("1y"),
    audit: bool = Query(True, description="Record this decision in the audit log"),
    variant: str = Query("", description=(
        "Named variant of the same symbol+algo pair. Empty is the original "
        "agent; 'regime' is its regime-aware twin, trained with the same "
        "period, episodes and profile so the two can be compared.")),
    db: AsyncSession = Depends(get_db),
):
    """The agent's current call, with the regime evidence behind it.

    Every served decision is written to the audit trail by default: a decision
    log that only records what someone remembered to record is not an audit
    trail. Pass ``audit=false`` for exploratory queries that should not enter
    the governance record.
    """
    result = rl_service.recommend_action(symbol, algo=algo, period=period,
                                         variant=variant)
    if audit:
        row_id = await log_rl_decision(db, result)
        if row_id is not None:
            result["audit_id"] = row_id
    return result


@router.get("/allocation", summary="Live target weights from a multi-asset agent")
async def recommend_allocation(
    symbols: str = Query(..., description="Comma-separated basket, e.g. AAPL,MSFT,SPY"),
    algo: str = Query("sac"),
    period: str = Query("1y"),
    audit: bool = Query(True, description="Record this decision in the audit log"),
    variant: str = Query("", description=(
        "Empty for the original basket agent, 'regime' for its regime-aware "
        "twin, trained with the same symbols, period and timesteps.")),
    db: AsyncSession = Depends(get_db),
):
    """What a trained basket agent would hold right now, and why.

    `/rl/portfolio/train` could train these agents but nothing could ask them
    what to hold, so trained baskets sat on disk unusable. The response
    attributes the regime's influence **per asset**: a weight vector has no
    "the action flipped" moment, so the useful question is which sleeve lost
    capital and to whom.
    """
    parsed = [s.strip() for s in symbols.split(",") if s.strip()]
    result = rl_service.recommend_allocation(parsed, algo=algo, period=period,
                                             variant=variant)
    if audit:
        row_id = await log_allocation_decision(db, result)
        if row_id is not None:
            result["audit_id"] = row_id
    return result


@router.get("/decisions", summary="RL decision audit trail")
async def decision_log(
    symbol: str | None = Query(None),
    regime: str | None = Query(None),
    limit: int = Query(50, ge=1, le=500),
    db: AsyncSession = Depends(get_db),
):
    """Recorded RL decisions, most recent first.

    Filtering happens in SQL *before* the limit. Applying it afterwards would
    silently hide matching rows behind a page of non-matching ones — a bug this
    project already shipped once on the alert history.
    """
    # Both single-asset calls and basket rebalances are RL decisions; a filter
    # naming only one would silently hide half the audit trail.
    stmt = select(RecommendationLog).where(
        RecommendationLog.source.in_(("rl_agent", "rl_allocation")))
    if symbol:
        stmt = stmt.where(RecommendationLog.symbol == symbol.upper())
    if regime:
        stmt = stmt.where(RecommendationLog.regime == regime)
    stmt = stmt.order_by(RecommendationLog.created_at.desc()).limit(limit)

    rows = (await db.execute(stmt)).scalars().all()
    return {
        "count": len(rows),
        "filters": {"symbol": symbol, "regime": regime, "limit": limit},
        "decisions": [
            {
                "id": r.id,
                "symbol": r.symbol,
                "action": r.action,
                "confidence": r.confidence,
                "price_at_reco": r.price_at_reco,
                "algo": r.algo,
                "model_version": r.model_version,
                "regime": r.regime,
                "regime_confidence": r.regime_confidence,
                "regime_influence": r.regime_influence,
                "risk_metrics": r.risk_metrics,
                "explanation": r.explanation,
                "created_at": r.created_at.isoformat() if r.created_at else None,
            }
            for r in rows
        ],
    }


@router.get("/backtest/{symbol}", summary="Out-of-sample backtest of a trained agent")
async def backtest(
    symbol: str,
    algo: str = Query("dueling_dqn"),
    period: str = Query("1y"),
    initial_balance: float = Query(100_000.0, gt=0),
    transaction_cost: float = Query(0.001, ge=0, le=0.05),
    moe: bool = Query(False, description=(
        "Route each bar to a regime-specialised expert (bull / bear / stress), "
        "fine-tuning the incoming expert on its own past bars and measuring the "
        "reaction delay (KPI K-5). Native discrete agents only. Default off: "
        "the baseline path is untouched.")),
    variant: str = Query("", description=(
        "Named variant of the same symbol+algo pair. Empty is the original "
        "agent; 'regime' is its regime-aware twin, trained with the same "
        "period, episodes and profile so the two can be compared.")),
    moe_adapt: bool = Query(True, description=(
        "Only with moe=true. False routes the experts but applies no gradient "
        "update — the control condition that isolates routing from adaptation.")),
):
    """Out-of-sample backtest, optionally through the Mixture-of-Experts.

    `moe=false` (the default) is the original single-policy path, byte for
    byte: `rl_service.backtest` is called exactly as before and no MoE code is
    imported. `moe=true` swaps in the regime-routed rollout, which returns the
    same keys plus a `moe` block holding the routing trace and K-5.
    """
    env_overrides = {"initial_balance": initial_balance,
                     "transaction_cost": transaction_cost}
    if moe:
        # Imported lazily and only on this branch, so the default path cannot
        # be affected by anything in the MoE module.
        from app.services.rl.moe import rollout
        return rollout(symbol, algo=algo, period=period,
                       env_overrides=env_overrides, adapt=moe_adapt,
                       variant=variant)
    return rl_service.backtest(symbol, algo=algo, period=period,
                               env_overrides=env_overrides, variant=variant)


@router.get("/agents", summary="List trained agents")
async def list_agents():
    agents = rl_service.list_agents()
    return {
        "count": len(agents),
        "agents": [{
            "symbol": a.get("symbol") or a.get("portfolio_key"),
            "algo": a.get("algo"),
            "trained_at": a.get("trained_at"),
            "test_performance": a.get("test_performance", {}),
            "baselines": a.get("baselines", {}),
            "stale": a.get("stale", False),
            "stale_reason": a.get("stale_reason"),
        } for a in agents],
    }
