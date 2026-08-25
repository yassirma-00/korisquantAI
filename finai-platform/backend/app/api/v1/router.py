"""Aggregate router for API v1."""

from __future__ import annotations

from fastapi import APIRouter

from app.api.v1.endpoints import (
    auth,
    chat,
    dashboard,
    forecast,
    hyperparams,
    insights,
    intelligence,
    market,
    portfolio,
    quant,
    rl,
    training,
)

api_router = APIRouter()
api_router.include_router(market.router)
api_router.include_router(forecast.router)
api_router.include_router(rl.router)
api_router.include_router(hyperparams.router)
api_router.include_router(training.router)
api_router.include_router(portfolio.router)
api_router.include_router(dashboard.router)
api_router.include_router(quant.router)
api_router.include_router(intelligence.router)
api_router.include_router(insights.router)
api_router.include_router(chat.router)
api_router.include_router(auth.router)
