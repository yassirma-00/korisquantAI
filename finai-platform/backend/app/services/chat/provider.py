"""Ollama transport for the in-app assistant.

Ollama exposes an OpenAI-compatible ``/v1/chat/completions`` endpoint, so the
wire format is the familiar one. The same code drives two very different
deployments:

* **Ollama Cloud** — ``https://ollama.com/v1`` with an API key.
* **Local Ollama** — ``http://localhost:11434/v1``, no key, no network egress.

The local case is genuinely useful here: a finance tool that never sends market
questions to a third party is easier to justify, and it removes per-request
cost entirely.

Design notes
------------
* **The key never leaves the server.** The browser calls ``/api/v1/chat``; this
  module is the only place that holds the credential.
* **Model fallback is mandatory, not a nicety.** A cloud model can be saturated
  and a local one may simply not be pulled yet. A single-model client surfaces
  either as a broken chat; we walk a list instead.
* **Tool calling is non-negotiable.** The assistant's only access to real
  platform data is function calling. A model without it would answer from
  imagination — the exact failure this design exists to prevent.
* **Errors are surfaced, never invented.** Ollama's failure modes are specific
  (model not pulled, subscription required, daemon down) and each has a
  different remedy, so each gets its own typed exception and message.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import httpx

from app.core.config import settings
from app.core.exceptions import KorisQuantError
from app.core.logging import get_logger

logger = get_logger(__name__)


# What every end user sees when the assistant cannot answer, whatever the real
# cause. Deliberately generic: "Check OLLAMA_BASE_URL" is meaningless to someone
# using the dashboard, it leaks the internal architecture, and none of the real
# remedies (edit .env, run `ollama serve`, upgrade a plan) are things a user of a
# deployed instance can act on. The precise diagnosis is not lost — it goes to
# the server logs, where the person who *can* fix it will look.
USER_FACING_UNAVAILABLE = (
    "Unable to connect to the AI service. Please check your network connection, "
    "try again later, or contact the system developer if the problem persists.")


class ChatUnavailableError(KorisQuantError):
    """Raised when no model could answer. 503: the platform itself is fine.

    Carries two messages on purpose:

    * ``message``      - the operator diagnosis, written to the logs.
    * ``user_message`` - what the chat panel shows. Generic unless a subclass
      has something genuinely actionable to say to an end user.
    """

    status_code = 503
    code = "chat_unavailable"

    def __init__(self, message: str, detail: str | None = None,
                 user_message: str | None = None):
        super().__init__(message, details={"reason": detail} if detail else None)
        self.detail = detail
        self.user_message = user_message or USER_FACING_UNAVAILABLE

    def to_dict(self) -> dict:
        """Response body: the user-facing text, and nothing else.

        The operator diagnosis is deliberately absent even in development. It is
        already written to the server log by the endpoint, and shipping it here
        meant the frontend could render it by accident — which is exactly what
        happened: "Check OLLAMA_BASE_URL and your network connection." reached
        the chat panel. Not sending it removes the whole class of mistake.
        """
        return {"error": self.code, "message": self.user_message}


class DaemonUnreachableError(ChatUnavailableError):
    """Nothing is listening on the configured Ollama URL.

    Overwhelmingly the local case: the user set OLLAMA_BASE_URL to localhost but
    never started `ollama serve`. Saying "provider error" there would send them
    debugging the wrong thing.
    """

    code = "chat_daemon_unreachable"


class ModelNotPulledError(ChatUnavailableError):
    """A local Ollama has no copy of the requested model.

    One command fixes it, so the message names that command rather than leaving
    the user to guess.
    """

    code = "chat_model_not_pulled"


class SubscriptionRequiredError(ChatUnavailableError):
    """Every candidate model is gated behind an Ollama subscription.

    Distinct from a transient failure: retrying cannot help, and the remedy is
    either upgrading or choosing an un-gated model.
    """

    code = "chat_subscription_required"


class AuthenticationError(ChatUnavailableError):
    """The API key was rejected by Ollama Cloud."""

    code = "chat_auth_failed"


# Once an account-level condition is observed, remember it. Two reasons:
#   * the health endpoint would otherwise show a green "available" dot on an
#     assistant that fails on the very first message — a lie the UI repeats;
#   * every doomed attempt still costs a round-trip per model, so
#     short-circuiting turns a multi-second failure into an instant answer.
_account_block: dict[str, Any] = {}

# None of these conditions fix themselves, but the user may fix them without
# restarting the server (start the daemon, pull a model, upgrade). Re-probe
# periodically rather than latching forever on something already resolved.
_BLOCK_RECHECK_SECONDS = 120.0


def _blocked() -> dict[str, Any] | None:
    """Return the active account-level block, if it has not expired."""
    if not _account_block:
        return None
    until = float(_account_block.get("until") or 0)
    if until and until <= datetime.now(UTC).timestamp():
        _account_block.clear()
        return None
    return _account_block


def _note_block(kind: str, message: str, detail: str | None = None,
                user_message: str | None = None) -> None:
    _account_block.clear()
    _account_block.update({
        "kind": kind,
        "message": message,
        "detail": detail,
        "user_message": user_message,
        "until": datetime.now(UTC).timestamp() + _BLOCK_RECHECK_SECONDS,
    })


_BLOCK_TYPES: dict[str, type[ChatUnavailableError]] = {
    "daemon": DaemonUnreachableError,
    "auth": AuthenticationError,
    "subscription": SubscriptionRequiredError,
    "model": ModelNotPulledError,
}


def _block_error(block: dict[str, Any]) -> ChatUnavailableError:
    """Rebuild the typed exception for an already-known blocking condition."""
    cls = _BLOCK_TYPES.get(str(block.get("kind")), ChatUnavailableError)
    return cls(str(block.get("message")), block.get("detail"),
               user_message=block.get("user_message"))


@dataclass
class ChatCompletion:
    """One assistant turn: either free text, or a request to call tools."""

    content: str
    tool_calls: list[dict] = field(default_factory=list)
    model: str = ""
    usage: dict = field(default_factory=dict)
    reasoning: str | None = None

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_calls)


def _headers() -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    # A local daemon rejects nothing, but sending "Bearer None" would be worse
    # than sending nothing at all.
    if settings.OLLAMA_API_KEY:
        headers["Authorization"] = f"Bearer {settings.OLLAMA_API_KEY}"
    return headers


def model_chain() -> list[str]:
    """Primary model first, then the fallbacks, de-duplicated."""
    chain = [settings.OLLAMA_MODEL, *settings.OLLAMA_FALLBACK_MODELS]
    seen: set[str] = set()
    return [m for m in chain if m and not (m in seen or seen.add(m))]


def _strip_thinking(text: str) -> str:
    """Remove <think> blocks from user-visible content.

    Several models served by Ollama (gpt-oss, nemotron, qwen) are hybrid
    reasoning models and some return the chain of thought inline rather than in
    a separate field. Rendering it would leak the model's scratchpad into the
    chat panel.
    """
    if "<think>" not in text:
        return text.strip()
    buffer, depth, i = "", 0, 0
    while i < len(text):
        if text.startswith("<think>", i):
            depth += 1
            i += 7
        elif text.startswith("</think>", i):
            depth = max(depth - 1, 0)
            i += 8
        else:
            if depth == 0:
                buffer += text[i]
            i += 1
    return buffer.strip()


def _parse_choice(payload: dict, model: str) -> ChatCompletion:
    choices = payload.get("choices") or []
    if not choices:
        raise ValueError("response contained no choices")
    message = choices[0].get("message") or {}

    raw_calls = message.get("tool_calls") or []
    tool_calls: list[dict] = []
    for call in raw_calls:
        fn = call.get("function") or {}
        arguments = fn.get("arguments")
        # Ollama may send arguments as a JSON string or already as an object.
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments) if arguments.strip() else {}
            except json.JSONDecodeError:
                logger.warning("un-parseable tool arguments from %s: %r", model, arguments[:200])
                arguments = {}
        raw = dict(call)
        # Ollama frequently omits the call id, but the follow-up request has to
        # echo one back or the exchange is rejected.
        if not raw.get("id"):
            raw["id"] = f"call_{len(tool_calls)}"
        raw.setdefault("type", "function")
        tool_calls.append({
            "id": raw["id"],
            "name": fn.get("name", ""),
            "arguments": arguments if isinstance(arguments, dict) else {},
            "raw": raw,
        })

    return ChatCompletion(
        content=_strip_thinking(message.get("content") or ""),
        tool_calls=tool_calls,
        model=payload.get("model") or model,
        usage=payload.get("usage") or {},
        reasoning=message.get("reasoning") or message.get("reasoning_content"),
    )


def _classify_failure(status: int, payload: dict, text: str,
                      model: str) -> ChatUnavailableError | None:
    """Map an upstream error onto a typed, actionable exception.

    Returns None when the failure is model-specific and the chain should carry
    on to the next candidate.
    """
    error = payload.get("error")
    message = error.get("message") if isinstance(error, dict) else (error or "")
    haystack = f"{message} {text}".lower()

    if status in (401, 403):
        if "subscription" in haystack or "upgrade" in haystack:
            # Model-specific on Ollama Cloud: another model may well be free.
            return None
        err = AuthenticationError(
            "Ollama rejected the API key.",
            "Check OLLAMA_API_KEY in your .env file — the key may have been "
            "revoked or mistyped. Get one at https://ollama.com/settings/keys")
        _note_block("auth", err.message, err.detail, err.user_message)
        return err

    if status == 404 and ("not found" in haystack or "pull" in haystack):
        if settings.ollama_is_local:
            # Local: one command fixes it. Name the command.
            return ModelNotPulledError(
                f"The model '{model}' is not available on your local Ollama.",
                f"Run: ollama pull {model}",
                user_message=(f"The AI model '{model}' is not installed. "
                              f"Run 'ollama pull {model}', then try again."))
        return None      # cloud 404 on one model: try the next

    return None


async def complete(
    messages: list[dict[str, Any]],
    tools: list[dict] | None = None,
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> ChatCompletion:
    """Call Ollama, walking the model chain until one answers.

    Raises:
        DaemonUnreachableError: nothing is listening on OLLAMA_BASE_URL.
        AuthenticationError: Ollama Cloud rejected the key.
        ModelNotPulledError: a local daemon has no copy of the model.
        SubscriptionRequiredError: every candidate is behind a paid plan.
        ChatUnavailableError: every candidate failed for another reason.
    """
    if not settings.chat_available:
        raise ChatUnavailableError(
            "The AI assistant is not configured.",
            "Set OLLAMA_API_KEY in your .env file for Ollama Cloud, or point "
            "OLLAMA_BASE_URL at a local daemon (http://localhost:11434/v1).")

    known = _blocked()
    if known:
        raise _block_error(known)

    body: dict[str, Any] = {
        "messages": messages,
        "temperature": settings.CHAT_TEMPERATURE if temperature is None else temperature,
        "max_tokens": settings.CHAT_MAX_TOKENS if max_tokens is None else max_tokens,
    }
    if tools:
        body["tools"] = tools
        body["tool_choice"] = "auto"

    failures: list[str] = []
    gated: list[str] = []
    unreachable = 0
    timeouts = 0
    chain = model_chain()

    async with httpx.AsyncClient(timeout=settings.CHAT_TIMEOUT) as client:
        for model in chain:
            # Split the timeout budget: a connection either establishes quickly
            # or the service is unreachable, whereas generation legitimately
            # takes tens of seconds. Waiting the full CHAT_TIMEOUT just to learn
            # there is no route made an offline user stare at a spinner for
            # minutes before a verdict the first second already implied.
            timeout = httpx.Timeout(settings.CHAT_TIMEOUT,
                                    connect=settings.CHAT_CONNECT_TIMEOUT)
            try:
                response = await client.post(
                    f"{settings.OLLAMA_BASE_URL}/chat/completions",
                    headers=_headers(), json={**body, "model": model},
                    timeout=timeout,
                )
            except httpx.TimeoutException:
                # No route to the service times out on every model in turn. With
                # a 90 s timeout and a 4-model chain that is six minutes of
                # spinner before an error the first attempt already proved.
                timeouts += 1
                failures.append(f"{model}: timed out after {settings.CHAT_TIMEOUT:.0f}s")
                logger.warning("chat timeout on %s", model)
                if timeouts >= 2:
                    break
                continue
            except httpx.ConnectError as exc:
                # Connection refused is about the daemon, not the model, so
                # retrying other models is pure waste.
                unreachable += 1
                failures.append(f"{model}: cannot connect ({exc})")
                break
            except httpx.HTTPError as exc:
                failures.append(f"{model}: network error ({type(exc).__name__})")
                logger.warning("chat network error on %s: %s", model, exc)
                continue

            if response.status_code == 200:
                try:
                    payload = response.json()
                except ValueError:
                    failures.append(f"{model}: malformed JSON response")
                    continue
                if "error" in payload and not payload.get("choices"):
                    reason = payload["error"]
                    reason = reason.get("message", "unknown") if isinstance(reason, dict) else reason
                    failures.append(f"{model}: {reason}")
                    logger.warning("chat error body from %s: %s", model, reason)
                    continue
                try:
                    completion = _parse_choice(payload, model)
                except ValueError as exc:
                    failures.append(f"{model}: {exc}")
                    continue
                if failures:
                    logger.info("chat fell back to %s after %d failure(s)", model, len(failures))
                return completion

            try:
                payload = response.json()
            except ValueError:
                payload = {}
            text = response.text[:300].replace("\n", " ")

            fatal = _classify_failure(response.status_code, payload, text, model)
            if fatal is not None:
                logger.warning("chat blocked: %s", fatal.code)
                raise fatal

            if "subscription" in text.lower() or "upgrade" in text.lower():
                gated.append(model)
            failures.append(f"{model}: HTTP {response.status_code} {text}")
            logger.warning("chat HTTP %s on %s: %s", response.status_code, model, text)

    if unreachable or timeouts >= 2:
        where = ("your local Ollama daemon" if settings.ollama_is_local
                 else f"the Ollama server at {settings.OLLAMA_BASE_URL}")
        err = DaemonUnreachableError(
            f"Could not connect to {where}.",
            ("Start it with: ollama serve" if settings.ollama_is_local else
             "Check OLLAMA_BASE_URL and your network connection."),
            # A local daemon is the user's own process, so naming the command is
            # genuinely actionable. A remote endpoint is not something they can
            # fix, so they get the generic message instead of our internals.
            user_message=("The local AI service is not running. Start it with "
                          "'ollama serve', then try again."
                          if settings.ollama_is_local else None))
        _note_block("daemon", err.message, err.detail, err.user_message)
        raise err

    # Every model refused for the same paid-plan reason: retrying is pointless.
    if gated and len(gated) == len(chain):
        err = SubscriptionRequiredError(
            "Every configured model requires an Ollama subscription.",
            "Switch OLLAMA_MODEL to a model included in your plan (for example "
            "gpt-oss:20b), or upgrade at https://ollama.com/settings")
        _note_block("subscription", err.message, err.detail, err.user_message)
        raise err

    raise ChatUnavailableError(
        "Every available AI model refused the request. This is usually a "
        "temporary provider issue — try again in a few seconds.",
        " | ".join(failures[-3:]),
    )


async def probe() -> dict:
    """Cheap liveness check used by /api/v1/chat/health.

    Lists the models the endpoint actually serves. On a local daemon that also
    reveals the most common local problem — the daemon runs but the configured
    model was never pulled — which a plain connectivity check would miss.
    """
    if not settings.chat_available:
        return {"reachable": False,
                "reason": "OLLAMA_API_KEY is not set (required for Ollama Cloud)"}

    block = _blocked()
    if block:
        return {
            "reachable": False,
            "blocked": block.get("kind"),
            "reason": block.get("message"),          # operator diagnosis
            "remedy": block.get("detail"),           # operator remedy
            "user_message": block.get("user_message") or USER_FACING_UNAVAILABLE,
        }

    mode = "local" if settings.ollama_is_local else "cloud"
    try:
        timeout = httpx.Timeout(10.0, connect=settings.CHAT_CONNECT_TIMEOUT)
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(
                f"{settings.OLLAMA_BASE_URL}/models", headers=_headers())
    except httpx.ConnectError:
        return {
            "reachable": False, "mode": mode, "blocked": "daemon",
            "reason": f"nothing is listening at {settings.OLLAMA_BASE_URL}",
            "remedy": ("Start the daemon with: ollama serve" if settings.ollama_is_local
                       else "Check OLLAMA_BASE_URL and your network connection."),
            "user_message": ("The local AI service is not running. Start it with "
                             "'ollama serve', then reload this page."
                             if settings.ollama_is_local else USER_FACING_UNAVAILABLE),
        }
    except (httpx.TimeoutException, httpx.NetworkError) as exc:
        # A dropped wifi connection surfaces as a timeout, not ConnectError, so
        # without this branch it fell through to the generic handler and the
        # panel showed no `blocked` state at all - leaving the composer enabled
        # on an assistant that could not answer.
        return {
            "reachable": False, "mode": mode, "blocked": "daemon",
            "reason": f"{type(exc).__name__}: {exc}"[:160],
            "remedy": ("Start the daemon with: ollama serve" if settings.ollama_is_local
                       else "Check OLLAMA_BASE_URL and your network connection."),
            "user_message": ("The local AI service is not responding. Start it with "
                             "'ollama serve', then reload this page."
                             if settings.ollama_is_local else USER_FACING_UNAVAILABLE),
        }
    except Exception as exc:  # pragma: no cover - network dependent
        return {"reachable": False, "mode": mode, "blocked": "daemon",
                "reason": f"{type(exc).__name__}: {exc}"[:160],
                "user_message": USER_FACING_UNAVAILABLE}

    if response.status_code in (401, 403):
        return {"reachable": False, "mode": mode, "blocked": "auth",
                "reason": "the API key was rejected",
                "remedy": "Check OLLAMA_API_KEY at https://ollama.com/settings/keys",
                "user_message": USER_FACING_UNAVAILABLE}
    if response.status_code != 200:
        return {"reachable": False, "mode": mode,
                "reason": f"HTTP {response.status_code}"}

    try:
        available = [m["id"] for m in (response.json() or {}).get("data", [])]
    except ValueError:
        available = []

    result: dict[str, Any] = {
        "reachable": True,
        "mode": mode,
        "endpoint": settings.OLLAMA_BASE_URL,
        "models_available": len(available),
    }
    # Only meaningful locally: the cloud lists models the plan may still gate,
    # so absence there is not proof of anything.
    if settings.ollama_is_local and available and settings.OLLAMA_MODEL not in available:
        result.update({
            "reachable": False, "blocked": "model",
            "reason": f"'{settings.OLLAMA_MODEL}' has not been pulled",
            "remedy": f"Run: ollama pull {settings.OLLAMA_MODEL}",
            # Local-only condition, so the command is worth showing.
            "user_message": (f"The AI model '{settings.OLLAMA_MODEL}' is not "
                             f"installed. Run 'ollama pull {settings.OLLAMA_MODEL}', "
                             "then reload this page."),
        })
    return result
