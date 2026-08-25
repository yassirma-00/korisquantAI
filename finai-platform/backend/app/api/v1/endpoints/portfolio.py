"""Portfolio management endpoints (paper trading)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.schemas.common import (
    CreatePortfolioRequest,
    OptimiseRequest,
    RebalanceRequest,
    TradeRequest,
)
from app.services.data.market_data import market_data_service
from app.services.recommendation.portfolio import portfolio_service
from app.services.risk.metrics import optimise_portfolio

router = APIRouter(prefix="/portfolio", tags=["Portfolio"])


@router.post("", summary="Create a portfolio")
async def create_portfolio(request: CreatePortfolioRequest, db: AsyncSession = Depends(get_db)):
    portfolio = await portfolio_service.create(
        db, name=request.name, initial_capital=request.initial_capital,
        description=request.description, base_currency=request.base_currency,
        strategy=request.strategy)
    return {"id": portfolio.id, "name": portfolio.name,
            "initial_capital": portfolio.initial_capital, "cash": portfolio.cash,
            "base_currency": portfolio.base_currency, "strategy": portfolio.strategy,
            "created_at": portfolio.created_at.isoformat()}


@router.get("", summary="List portfolios")
async def list_portfolios(db: AsyncSession = Depends(get_db)):
    portfolios = await portfolio_service.list_all(db)
    return {"count": len(portfolios), "portfolios": [
        {"id": p.id, "name": p.name, "initial_capital": p.initial_capital,
         "cash": round(p.cash, 2), "strategy": p.strategy,
         "created_at": p.created_at.isoformat()} for p in portfolios]}


@router.get("/{portfolio_id}", summary="Portfolio valuation")
async def get_portfolio(portfolio_id: int, db: AsyncSession = Depends(get_db)):
    return await portfolio_service.valuation(db, portfolio_id)


@router.delete("/{portfolio_id}", summary="Delete a portfolio")
async def delete_portfolio(portfolio_id: int, db: AsyncSession = Depends(get_db)):
    await portfolio_service.delete(db, portfolio_id)
    return {"deleted": portfolio_id}


@router.post("/{portfolio_id}/trade", summary="Execute a paper trade")
async def trade(portfolio_id: int, request: TradeRequest, db: AsyncSession = Depends(get_db)):
    return await portfolio_service.execute_trade(
        db, portfolio_id, symbol=request.symbol, side=request.side,
        quantity=request.quantity, notional=request.notional,
        price=request.price, notes=request.notes)


@router.get("/{portfolio_id}/transactions", summary="Transaction history")
async def transactions(portfolio_id: int, limit: int = Query(100, ge=1, le=1000),
                       db: AsyncSession = Depends(get_db)):
    rows = await portfolio_service.transactions(db, portfolio_id, limit)
    return {"count": len(rows), "transactions": [
        {"id": t.id, "symbol": t.symbol, "side": t.side, "quantity": t.quantity,
         "price": t.price, "fees": t.fees, "source": t.source, "notes": t.notes,
         "executed_at": t.executed_at.isoformat()} for t in rows]}


@router.get("/{portfolio_id}/analytics", summary="Full performance & risk analytics")
async def analytics(portfolio_id: int, period: str = Query("1y"),
                    benchmark: str = Query("SPY"), db: AsyncSession = Depends(get_db)):
    return await portfolio_service.analytics(db, portfolio_id, period=period, benchmark=benchmark)


@router.post("/{portfolio_id}/rebalance", summary="Plan (and optionally execute) a rebalance")
async def rebalance(portfolio_id: int, request: RebalanceRequest, db: AsyncSession = Depends(get_db)):
    if request.execute:
        return await portfolio_service.execute_rebalance(
            db, portfolio_id, target_weights=request.target_weights,
            objective=request.objective, period=request.period)
    return await portfolio_service.rebalance_plan(
        db, portfolio_id, target_weights=request.target_weights,
        objective=request.objective, period=request.period, tolerance=request.tolerance)


@router.post("/{portfolio_id}/snapshot", summary="Record a valuation snapshot")
async def snapshot(portfolio_id: int, db: AsyncSession = Depends(get_db)):
    row = await portfolio_service.take_snapshot(db, portfolio_id)
    return {"id": row.id, "total_value": row.total_value, "pnl": row.pnl,
            "pnl_pct": row.pnl_pct, "taken_at": row.taken_at.isoformat()}


@router.get("/{portfolio_id}/snapshots", summary="Historical snapshots")
async def snapshots(portfolio_id: int, db: AsyncSession = Depends(get_db)):
    rows = await portfolio_service.snapshots(db, portfolio_id)
    return {"count": len(rows), "snapshots": rows}


@router.post("/optimise", summary="Mean-variance / risk-parity optimisation")
async def optimise(request: OptimiseRequest):
    returns = market_data_service.get_returns_matrix(request.symbols, period=request.period)
    return optimise_portfolio(
        returns, objective=request.objective, risk_free=request.risk_free_rate,
        allow_short=request.allow_short)
