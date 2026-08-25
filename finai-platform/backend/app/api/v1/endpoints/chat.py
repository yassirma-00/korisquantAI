"""Conversational assistant endpoints.

The browser never talks to Ollama directly. It posts here, this module adds
the credential server-side, and only the finished answer travels back. That
keeps the API key out of devtools, out of the page source, and out of anyone
else's bill.
"""

from __future__ import annotations

import time
from collections import defaultdict, deque

from fastapi import APIRouter, Request

from app.core.config import settings
from app.core.logging import get_logger
from app.schemas.common import ChatRequest
from app.services.chat import tools
from app.services.chat.agent import run
from app.services.chat.provider import ChatUnavailableError, model_chain, probe

logger = get_logger(__name__)

router = APIRouter(prefix="/chat", tags=["AI Assistant"])

# In-process sliding window per client IP. Deliberately not Redis: this guards a
# free-tier quota on a single-node deployment, and an extra service for that
# would be over-engineering. Multi-node deployments should move it to Redis.
_HITS: dict[str, deque[float]] = defaultdict(deque)


def _rate_limited(client: str) -> tuple[bool, int]:
    window, limit = 60.0, settings.CHAT_RATE_LIMIT_PER_MIN
    now = time.monotonic()
    hits = _HITS[client]
    while hits and now - hits[0] > window:
        hits.popleft()
    if len(hits) >= limit:
        return True, int(window - (now - hits[0])) + 1
    hits.append(now)
    return False, 0


class RateLimitedError(ChatUnavailableError):
    status_code = 429
    code = "rate_limited"


@router.post("", summary="Ask the AI assistant")
async def chat(request: ChatRequest, http_request: Request):
    """Answer a question, calling platform tools for any real figure."""
    if not settings.CHAT_ENABLED:
        raise ChatUnavailableError(
            "The AI assistant is disabled on this deployment.",
            "Set CHAT_ENABLED=true to turn it back on.")

    client = http_request.client.host if http_request.client else "unknown"
    limited, retry_after = _rate_limited(client)
    if limited:
        raise RateLimitedError(
            f"Too many messages — please wait {retry_after}s before asking again.",
            f"Limit is {settings.CHAT_RATE_LIMIT_PER_MIN} messages per minute.")

    started = time.perf_counter()
    try:
        result = await run(
            user_message=request.message,
            history=[turn.model_dump() for turn in (request.history or [])],
            page=request.page,
            symbol=request.symbol,
        )
    except ChatUnavailableError as exc:
        # The user gets a generic message; the operator gets the real cause.
        # Logging it here is what makes that trade-off safe - the diagnosis is
        # narrowed, not thrown away.
        logger.warning("chat unavailable (%s): %s%s", exc.code, exc.message,
                       f" | {exc.detail}" if exc.detail else "")
        raise
    elapsed = (time.perf_counter() - started) * 1000
    logger.info("chat answered in %.0f ms via %s (%d tool calls)",
                elapsed, result.get("model"), len(result.get("tools_used", [])))
    return {**result, "elapsed_ms": round(elapsed, 1)}


# Fields the browser genuinely needs: enough to render the panel state, and
# nothing that describes how the backend is wired.
_PUBLIC_PROVIDER_FIELDS = ("reachable", "blocked", "user_message")


@router.get("/health", summary="Assistant availability and configuration")
async def chat_health():
    """Whether the assistant can actually answer, and with what.

    In development the full diagnosis is returned, because that is exactly what
    a developer needs. In production it is trimmed: endpoint URLs, config keys
    and remedies like "run ollama pull" describe the deployment, and anyone can
    open devtools. The panel only needs to know *that* it cannot answer and what
    to tell the user.
    """
    reachability = await probe() if settings.chat_available else {
        "reachable": False, "reason": "not configured"}

    payload = {
        "enabled": settings.CHAT_ENABLED,
        "configured": settings.chat_available,
        "available": settings.chat_available and reachability.get("reachable", False),
        "tool_count": len(tools.REGISTRY),
        "rate_limit_per_min": settings.CHAT_RATE_LIMIT_PER_MIN,
    }

    if settings.DEBUG:
        payload.update({
            "model": settings.OLLAMA_MODEL,
            "fallback_models": settings.OLLAMA_FALLBACK_MODELS,
            "model_chain": model_chain(),
            "provider": reachability,
        })
    else:
        payload["provider"] = {k: reachability[k] for k in _PUBLIC_PROVIDER_FIELDS
                               if k in reachability}
    return payload


@router.get("/tools", summary="Tools the assistant can call")
async def list_tools():
    """The assistant's capability surface, for transparency and debugging."""
    return {
        "count": len(tools.REGISTRY),
        "tools": [
            {"name": name, "description": schema["description"],
             "parameters": sorted((schema.get("parameters") or {}).get("properties", {}))}
            for name, (_, schema) in tools.REGISTRY.items()
        ],
    }


@router.get("/suggestions", summary="Starter prompts for the chat panel")
async def suggestions(page: str | None = None, symbol: str | None = None):
    """Context-aware prompts so the panel is never an empty box.

    Suggestions are templated per page: what is useful on the RL page is noise
    on the Portfolio page.
    """
    ticker = (symbol or "AAPL").upper()
    per_page = {
        "index": ["What is moving in the market today?",
                  f"Give me a quick read on {ticker}",
                  "Which of my watchlist symbols look risky?"],
        "analysis": [f"Explain {ticker}'s technical setup",
                     f"Is {ticker} overbought right now?",
                     f"What do the moving averages say about {ticker}?"],
        "forecast": [f"What does the LSTM forecast for {ticker}?",
                     "How accurate are these forecasts out of sample?",
                     "What does the prediction interval actually guarantee?"],
        "rl": [f"What does the RL agent recommend for {ticker}?",
               "Which RL algorithm should I use and why?",
               f"Did the agent beat buy & hold on {ticker}?"],
        "signals": [f"Why is the recommendation for {ticker} what it is?",
                    "What happens when the signals disagree?",
                    f"How much of {ticker} would the sizing model allocate?"],
        "xai": [f"Which features drive the prediction for {ticker}?",
                "How should I read a SHAP contribution?",
                "Is the surrogate model reliable here?"],
        "portfolio": [f"Compare trading strategies on {ticker}",
                      f"What are {ticker}'s risk-adjusted returns?",
                      "How is max drawdown calculated?"],
        "risk": [f"How risky is {ticker} right now?",
                 f"Any anomalies detected in {ticker}?",
                 "What is the difference between VaR and CVaR?"],
    }
    return {
        "page": page,
        "symbol": ticker,
        "suggestions": per_page.get(page or "", [
            f"Analyse {ticker} for me",
            "What can this platform do?",
            "Which models and agents are trained already?",
        ]),
    }
