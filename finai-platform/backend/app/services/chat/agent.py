"""The assistant's reasoning loop: prompt, tool orchestration, guardrails.

The loop is deliberately small: ask the model, run whatever tools it asked for,
feed the results back, repeat until it produces prose or we hit the round cap.
The interesting engineering is in the *constraints* around it.

Guardrails that matter in a finance product
-------------------------------------------
1. **No invented numbers.** The prompt forbids stating a figure that did not
   come from a tool result, and the tools are the only source of figures.
2. **Simulated data must be disclosed.** When live providers are unreachable the
   platform falls back to a synthetic engine. A price that looks real but isn't
   is dangerous, so ``data_source: synthetic`` has to be surfaced.
3. **No investment advice.** The platform is educational software. The
   assistant explains what the models output; it does not tell anyone to buy.
4. **Unfavourable results stay unfavourable.** If an RL agent lost to buy &
   hold, the assistant says so. Sugar-coating a backtest is how people lose
   money.
"""

from __future__ import annotations

import json
from typing import Any

from app.core.config import settings
from app.core.logging import get_logger
from app.services.chat import tools
from app.services.chat.provider import ChatCompletion, complete

logger = get_logger(__name__)


SYSTEM_PROMPT = """You are KorisQuant AI Assistant, the built-in analyst of the KorisQuant AI \
platform — a research tool for financial analysis and portfolio management \
powered by machine learning, deep learning and reinforcement learning.

## Your one hard rule
NEVER state a market figure — a price, an indicator value, a return, a Sharpe \
ratio, an accuracy — unless it came back from a tool call in this conversation. \
You have no memory of market data and no live prices in your context. If you \
need a number, call a tool. If a tool fails, say what failed. Guessing a \
plausible-looking number is the worst thing you can do here.

## What you can do
You have tools that read the platform's real engines: quotes, 17 technical \
indicators, deep-learning forecasts (LSTM/GRU/TCN/Transformer/CNN-LSTM), \
13 reinforcement-learning agents, the multi-signal recommendation engine, risk \
and anomaly scans, news sentiment, strategy backtests, regime detection, \
performance dossiers, and SHAP explainability.

Call tools whenever a question touches real data. Multiple tools in one turn is \
fine and often better — for "should I look at NVDA?" pull the recommendation, \
the risk scan and the technicals rather than one of them.

## How to report results
- Lead with the answer, then the evidence. Keep it tight; this is a chat panel, \
not a report. Markdown is supported: short paragraphs, bold for key numbers, \
compact bullet lists, tables only when comparing.
- Always name the source of a judgement: "the dueling-DQN agent", "the LSTM \
forecast", "the RSI reading" — never a vague "analysis shows".
- If a tool returns `is_simulated: true` or `data_source: synthetic`, state \
plainly that live providers were unreachable and these are simulated prices. \
If it returns `ticker_verified: false` or a `warning`, lead with that: tell the \
user the symbol could not be verified and do not present the figures as a real \
quote.
- Never mention tool names, function names or the word "tool" in your answer. \
The user sees a chat, not an API. Say "the platform's risk scan" or "the \
technical indicators", not `get_risk_assessment`.
- If a tool returns `model_not_trained` or `agent_unavailable`, explain what is \
missing and point at the page that trains it. Do not substitute a guess.
- Report bad results honestly. If an agent underperformed buy & hold, if R² is \
negative, if directional accuracy is near 50%, say so directly. Users of this \
platform are explicitly told the truth about model quality — that is a feature.
- Quantify uncertainty when the tool gives it to you: prediction intervals, \
confidence, coverage checks, signal agreement.

## Boundaries
You are educational and research software, not a financial adviser. Explain what \
the models say and what the numbers mean; never tell a user to buy or sell, and \
never promise a return. When a user asks "should I buy X", give them the \
platform's signal, its confidence, its risks, and let them decide.

Refuse politely and briefly if asked to do something outside financial analysis \
and this platform's features.

## Navigation help
The platform has 8 pages: Market Overview, Technical Analysis, AI Forecasting, \
RL Agent, Recommendations, Explainability, Portfolio, and Risk & Alerts. When a \
user would be better served by a page than by chat, point them there."""


def _page_context(page: str | None, symbol: str | None) -> str:
    """Tell the model where the user is, so it can be contextually useful."""
    if not page and not symbol:
        return ""
    hints = {
        "index": "the Market Overview dashboard (indices, watchlist, movers, alerts)",
        "analysis": "the Technical Analysis page (candlesticks, indicators, signals)",
        "forecast": "the AI Forecasting page (deep-learning price forecasts)",
        "rl": "the RL Agent page (training and inspecting trading agents)",
        "signals": "the Recommendations page (multi-signal fusion)",
        "xai": "the Explainability page (SHAP, LIME, counterfactuals)",
        "portfolio": "the Portfolio page (paper trading, analytics, strategy comparison)",
        "risk": "the Risk & Alerts page (anomalies, crash risk, VaR)",
    }
    parts = []
    if page:
        parts.append(f"The user is currently on {hints.get(page, page)}.")
    if symbol:
        parts.append(
            f"The instrument selected on screen is {symbol}. When they say "
            f"'this stock', 'it' or ask without naming a ticker, they mean {symbol}.")
    return " ".join(parts)


def build_messages(user_message: str, history: list[dict] | None = None,
                   page: str | None = None, symbol: str | None = None) -> list[dict]:
    """Assemble the request: system prompt, trimmed history, new message."""
    system = SYSTEM_PROMPT
    context = _page_context(page, symbol)
    if context:
        system = f"{system}\n\n## Current context\n{context}"

    messages: list[dict[str, Any]] = [{"role": "system", "content": system}]

    # Only user/assistant turns are replayed. Tool traffic from earlier turns is
    # dropped: it is bulky, and stale tool output invites the model to reuse an
    # old price instead of fetching the current one.
    for turn in (history or [])[-settings.CHAT_MAX_HISTORY:]:
        role, content = turn.get("role"), (turn.get("content") or "").strip()
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content[:4000]})

    messages.append({"role": "user", "content": user_message[:4000]})
    return messages


async def run(user_message: str, history: list[dict] | None = None,
              page: str | None = None, symbol: str | None = None) -> dict:
    """Answer one user message, calling platform tools as needed.

    Returns a payload carrying the prose answer plus the audit trail of which
    tools ran — the UI shows that trail so a user can see the answer was
    computed, not improvised.
    """
    messages = build_messages(user_message, history, page, symbol)
    schemas = tools.openai_tool_schemas()
    trail: list[dict] = []
    completion: ChatCompletion | None = None

    for round_index in range(settings.CHAT_MAX_TOOL_ROUNDS):
        completion = await complete(messages, tools=schemas)

        if not completion.wants_tools:
            break

        # Echo the assistant's tool request back verbatim: providers validate
        # that every tool result answers a call they actually saw.
        messages.append({
            "role": "assistant",
            "content": completion.content or None,
            "tool_calls": [call["raw"] for call in completion.tool_calls],
        })

        for call in completion.tool_calls:
            result = await tools.execute(call["name"], call["arguments"])
            trail.append({
                "tool": call["name"],
                "arguments": call["arguments"],
                "ok": "error" not in result,
                "error": result.get("error"),
            })
            messages.append({
                "role": "tool",
                "tool_call_id": call["id"],
                "name": call["name"],
                "content": json.dumps(result, default=str)[:12000],
            })

        logger.info("chat round %d ran %d tool(s)", round_index + 1,
                    len(completion.tool_calls))
    else:
        # Round cap hit while the model still wanted tools. Ask once more with
        # tools withheld so it has to answer from what it already gathered.
        logger.info("chat hit the %d-round tool cap; forcing a final answer",
                    settings.CHAT_MAX_TOOL_ROUNDS)
        messages.append({
            "role": "user",
            "content": ("Answer now using only the tool results already gathered. "
                        "Do not request further tools; note anything still missing."),
        })
        completion = await complete(messages, tools=None)

    content = (completion.content if completion else "").strip()
    if not content:
        # A model that returns nothing (all budget spent on reasoning tokens) is
        # better acknowledged than rendered as an empty bubble.
        content = ("I could not produce an answer for that. Try rephrasing, or "
                   "ask about a specific instrument — for example "
                   "\"analyse AAPL\" or \"what does the RL agent say about BTC-USD?\".")

    return {
        "reply": content,
        "model": completion.model if completion else "",
        "tools_used": trail,
        "usage": completion.usage if completion else {},
    }
