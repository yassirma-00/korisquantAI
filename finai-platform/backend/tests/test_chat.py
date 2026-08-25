"""Tests for the in-app AI assistant.

Scope note
----------
These tests never call Ollama. A test that needs a live third-party API is a
test that fails when someone else's rate limit is hit, and CI runs with
``DATA_MODE=offline`` precisely to stay deterministic. So the provider is stubbed
and what is verified here is everything we actually own:

* the tool layer returns real platform data with the shape the model is promised
* failures degrade into explainable payloads instead of exceptions
* the agent loop feeds tool results back and stops correctly
* the API key never reaches the browser
* the frontend is wired on every page
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.services.chat import tools
from app.services.chat.agent import build_messages
from app.services.chat.provider import ChatCompletion

FRONTEND = Path(__file__).resolve().parents[2] / "frontend"


def settings_key_absent(payload) -> bool:
    """True when no Ollama credential appears anywhere in a response payload.

    Ollama keys look like ``<32 hex>.<24 url-safe chars>``; the configured key is
    also checked verbatim so the assertion still bites if the format changes.
    """
    import re

    from app.core.config import settings

    blob = json.dumps(payload)
    if settings.OLLAMA_API_KEY and settings.OLLAMA_API_KEY in blob:
        return False
    return not re.search(r"\b[0-9a-f]{32}\.[A-Za-z0-9_-]{20,}", blob)
PAGES = ("index", "analysis", "forecast", "rl", "signals", "xai", "portfolio", "risk")


# ============================================================== tool layer
@pytest.mark.asyncio
async def test_every_tool_has_a_schema_and_a_callable():
    """A tool advertised to the model but not implemented is a runtime 500."""
    for name, (func, schema) in tools.REGISTRY.items():
        assert callable(func), f"{name} is not callable"
        assert schema.get("description"), f"{name} has no description"
        assert "parameters" in schema, f"{name} declares no parameters"
        assert schema["parameters"].get("type") == "object", name


def test_tool_schemas_are_valid_openai_function_definitions():
    schemas = tools.openai_tool_schemas()
    assert len(schemas) == len(tools.REGISTRY)
    for entry in schemas:
        assert entry["type"] == "function"
        fn = entry["function"]
        assert fn["name"] in tools.REGISTRY
        assert isinstance(fn["description"], str) and fn["description"]
        for required in fn["parameters"].get("required", []):
            assert required in fn["parameters"]["properties"], \
                f"{fn['name']}: required arg '{required}' is not declared"


@pytest.mark.asyncio
async def test_unknown_tool_is_reported_not_raised():
    result = await tools.execute("get_the_winning_lottery_numbers", {})
    assert result["error"] == "unknown_tool"


@pytest.mark.asyncio
async def test_missing_required_argument_is_reported_not_raised():
    result = await tools.execute("get_quote", {})
    assert result["error"] == "missing_arguments"
    assert "symbol" in result["message"]


@pytest.mark.asyncio
async def test_unexpected_arguments_are_dropped():
    """Models hallucinate parameters; an unknown kwarg must not TypeError."""
    result = await tools.execute("get_quote", {"symbol": "AAPL", "exchange": "NASDAQ"})
    assert "error" not in result or result.get("error") != "bad_arguments"


@pytest.mark.asyncio
async def test_quote_tool_returns_real_platform_data():
    result = await tools.execute("get_quote", {"symbol": "AAPL"})
    assert result["symbol"] == "AAPL"
    assert isinstance(result["price"], (int, float))
    # The synthetic-data flag must always be present: the assistant is required
    # to disclose simulated prices, and it can only do that if we tell it.
    assert "is_simulated" in result
    assert "data_source" in result


@pytest.mark.asyncio
async def test_unknown_ticker_is_flagged_as_unverified():
    """The synthetic engine never fails, so a typo silently produces a
    plausible price. Without this flag the assistant would report a fabricated
    quote as real — the worst failure mode this product has."""
    result = await tools.execute("get_quote", {"symbol": "ZZZQQ999"})
    assert result["ticker_verified"] is False
    assert result["in_known_universe"] is False
    assert "warning" in result
    assert "SIMULATED" in result["warning"]


@pytest.mark.asyncio
async def test_known_ticker_is_marked_verified():
    result = await tools.execute("get_quote", {"symbol": "AAPL"})
    assert result["ticker_verified"] is True
    assert result["in_known_universe"] is True


@pytest.mark.asyncio
async def test_technical_analysis_tool_exposes_indicators_and_consensus():
    result = await tools.execute("get_technical_analysis", {"symbol": "AAPL"})
    assert result["symbol"] == "AAPL"
    assert result["consensus"] in ("bullish", "bearish", "neutral")
    assert result["indicators"]["rsi_14"] is not None
    assert 0 <= result["indicators"]["rsi_14"] <= 100


@pytest.mark.asyncio
async def test_untrained_forecast_explains_itself_instead_of_failing():
    """The assistant must be able to say *why* it cannot forecast, and where to
    fix it, rather than emitting a stack trace or inventing a number."""
    result = await tools.execute(
        "get_forecast", {"symbol": "NOSUCHMODEL", "model": "lstm", "horizon": 5})
    assert result["error"] == "model_not_trained"
    assert "AI Forecasting" in result["message"]


@pytest.mark.asyncio
async def test_missing_rl_agent_explains_itself():
    result = await tools.execute("get_agent_decision", {"symbol": "NOAGENT", "algo": "ppo"})
    assert result["error"] == "agent_unavailable"
    assert "RL Agent" in result["message"]


@pytest.mark.asyncio
async def test_catalogue_tools_need_no_arguments():
    for name in ("list_rl_algorithms", "list_trained_models", "get_platform_status"):
        result = await tools.execute(name, {})
        assert "error" not in result, f"{name}: {result}"


@pytest.mark.asyncio
async def test_algorithm_catalogue_tool_matches_the_real_catalogue():
    from app.services.rl.catalogue import CATALOGUE

    result = await tools.execute("list_rl_algorithms", {})
    assert result["count"] == len(CATALOGUE)
    assert {a["key"] for a in result["algorithms"]} == {a.key for a in CATALOGUE}


@pytest.mark.asyncio
async def test_tool_results_are_json_serialisable():
    """Tool output is serialised into the model transcript; a numpy float or a
    Timestamp in there raises at json.dumps time, mid-conversation."""
    for name in ("get_quote", "get_technical_analysis", "get_risk_assessment",
                 "get_market_regime", "compare_strategies"):
        result = await tools.execute(name, {"symbol": "AAPL"})
        json.dumps(result)          # must not raise


@pytest.mark.asyncio
async def test_tool_payloads_stay_small_enough_for_a_free_tier_context():
    """Strategy comparison carries equity curves internally; if those leak into
    the payload a single tool call can blow the model's context window."""
    for name in ("compare_strategies", "get_performance_metrics"):
        result = await tools.execute(name, {"symbol": "AAPL", "period": "1y"})
        size = len(json.dumps(result, default=str))
        assert size < 20_000, f"{name} returned {size}B — too large for the model context"


# ================================================================== prompt
def test_system_prompt_forbids_inventing_figures():
    from app.services.chat.agent import SYSTEM_PROMPT

    lowered = SYSTEM_PROMPT.lower()
    assert "never state a market figure" in lowered
    assert "not investment advice" in lowered or "financial adviser" in lowered


def test_page_context_resolves_pronouns_to_the_selected_symbol():
    messages = build_messages("analyse this stock", page="rl", symbol="NVDA")
    system = messages[0]["content"]
    assert "NVDA" in system
    assert "RL Agent" in system


def test_history_is_trimmed_and_tool_turns_are_not_replayed():
    history = [{"role": "user", "content": f"q{i}"} for i in range(40)]
    history.append({"role": "tool", "content": "should be dropped"})
    messages = build_messages("latest", history=history)
    assert all(m["role"] in ("system", "user", "assistant") for m in messages)
    assert not any("should be dropped" in (m.get("content") or "") for m in messages)
    assert messages[-1]["content"] == "latest"


# ============================================================== agent loop
@pytest.mark.asyncio
async def test_agent_feeds_tool_results_back_to_the_model(monkeypatch):
    """The whole point of the loop: the model asks, we execute, it answers with
    the real figure. Verifies the tool result actually reaches round two."""
    from app.services.chat import agent

    seen: list[list[dict]] = []
    calls = {"n": 0}

    async def fake_complete(messages, tools=None, **kwargs):
        seen.append(messages)
        calls["n"] += 1
        if calls["n"] == 1:
            return ChatCompletion(
                content="", model="stub",
                tool_calls=[{"id": "c1", "name": "get_quote",
                             "arguments": {"symbol": "AAPL"},
                             "raw": {"id": "c1", "type": "function",
                                     "function": {"name": "get_quote",
                                                  "arguments": '{"symbol":"AAPL"}'}}}])
        return ChatCompletion(content="AAPL is trading at the quoted price.", model="stub")

    monkeypatch.setattr(agent, "complete", fake_complete)
    result = await agent.run("what is AAPL at?")

    assert calls["n"] == 2, "the model was not re-invoked with the tool result"
    assert result["tools_used"][0]["tool"] == "get_quote"
    assert result["tools_used"][0]["ok"] is True
    tool_messages = [m for m in seen[1] if m.get("role") == "tool"]
    assert tool_messages, "tool output was never handed back to the model"
    assert "price" in tool_messages[0]["content"]


@pytest.mark.asyncio
async def test_agent_stops_at_the_round_cap(monkeypatch):
    """A model that loops on tool calls must not spin forever."""
    from app.core.config import settings
    from app.services.chat import agent

    calls = {"n": 0}

    async def always_tools(messages, tools=None, **kwargs):
        calls["n"] += 1
        if tools is None:                     # the forced final answer
            return ChatCompletion(content="Partial answer.", model="stub")
        return ChatCompletion(
            content="", model="stub",
            tool_calls=[{"id": f"c{calls['n']}", "name": "get_platform_status",
                         "arguments": {},
                         "raw": {"id": f"c{calls['n']}", "type": "function",
                                 "function": {"name": "get_platform_status",
                                              "arguments": "{}"}}}])

    monkeypatch.setattr(agent, "complete", always_tools)
    result = await agent.run("loop forever")

    assert calls["n"] == settings.CHAT_MAX_TOOL_ROUNDS + 1
    assert result["reply"] == "Partial answer."


@pytest.mark.asyncio
async def test_empty_model_response_becomes_a_useful_message(monkeypatch):
    """Free models sometimes spend their whole budget on reasoning tokens and
    return an empty string. An empty bubble looks like a broken app."""
    from app.services.chat import agent

    async def empty(messages, tools=None, **kwargs):
        return ChatCompletion(content="", model="stub")

    monkeypatch.setattr(agent, "complete", empty)
    result = await agent.run("hello")
    assert result["reply"].strip()
    assert "analyse" in result["reply"].lower()


# ==================================================================== API
def test_chat_health_never_leaks_the_api_key(client):
    payload = client.get("/api/v1/chat/health").json()
    assert payload["configured"] in (True, False)
    assert settings_key_absent(payload), "the Ollama API key leaked into a response"
    assert "api_key" not in payload


def test_chat_tools_endpoint_lists_the_registry(client):
    payload = client.get("/api/v1/chat/tools").json()
    assert payload["count"] == len(tools.REGISTRY)
    assert {t["name"] for t in payload["tools"]} == set(tools.REGISTRY)


def test_chat_suggestions_are_page_aware(client):
    rl = client.get("/api/v1/chat/suggestions", params={"page": "rl", "symbol": "nvda"}).json()
    assert rl["symbol"] == "NVDA"
    assert any("NVDA" in s for s in rl["suggestions"])
    risk = client.get("/api/v1/chat/suggestions", params={"page": "risk"}).json()
    assert rl["suggestions"] != risk["suggestions"]


def test_chat_rejects_an_empty_message(client):
    assert client.post("/api/v1/chat", json={"message": ""}).status_code == 422


def test_chat_rejects_an_oversized_message(client):
    assert client.post("/api/v1/chat", json={"message": "x" * 5000}).status_code == 422


def test_chat_reports_unavailability_instead_of_crashing(client, monkeypatch):
    """With no key configured the endpoint must answer 503 with an actionable
    message, not a 500."""
    from app.core.config import settings

    monkeypatch.setattr(settings, "OLLAMA_API_KEY", None)
    monkeypatch.setattr(settings, "OLLAMA_BASE_URL", "https://ollama.com/v1")
    response = client.post("/api/v1/chat", json={"message": "hello"})
    assert response.status_code == 503
    # The body is a user-facing message now: naming OLLAMA_API_KEY would be
    # meaningless to a user and would advertise the deployment. The operator
    # instruction lives in the log and in .env.example.
    body = json.dumps(response.json())
    assert "OLLAMA_API_KEY" not in body
    assert "contact the system developer" in body


# ================================================================ frontend
@pytest.mark.parametrize("page", PAGES)
def test_assistant_is_mounted_on_every_page(client, page):
    """A chatbot on 7 of 8 pages is a bug report waiting to happen."""
    html = client.get(f"/{page}.html").text
    assert "assets/js/chat.js" in html, f"{page}: the assistant script is missing"


@pytest.mark.parametrize("page", PAGES)
def test_chat_script_loads_after_its_dependencies(client, page):
    """chat.js calls api.chat() and getActiveSymbol() at DOMContentLoaded."""
    html = client.get(f"/{page}.html").text
    assert html.index("assets/js/chat.js") > html.index("assets/js/api.js"), \
        f"{page}: chat.js must load after api.js"


def test_api_client_exposes_the_chat_methods():
    js = (FRONTEND / "assets" / "js" / "api.js").read_text()
    for method in ("chat:", "chatHealth:", "chatTools:", "chatSuggestions:"):
        assert method in js, f"api.js is missing {method}"


def test_frontend_never_embeds_a_provider_key():
    """The single most important security property of this feature: the key is
    server-side only. A key in frontend JS is a public key.

    The check targets credentials and *calls* to the provider, not the string
    "ollama.com" itself — that also appears in help text telling the user where
    to get a key, which is not a leak. A blanket ban on the substring would have
    pushed us to degrade a useful message.
    """
    import re

    # Any absolute URL pointing at the provider, i.e. an actual direct call.
    # Only an actual network call counts. The URL also appears in help text
    # telling the user what to put in .env, which is not a leak — matching the
    # bare URL flagged that copy and would have pushed us to degrade it.
    direct_call = re.compile(
        r"""(?:fetch|XMLHttpRequest|axios[.\w]*)\s*\(\s*['"`][^'"`]*"""
        r"""(?:ollama\.com|:11434)""", re.I)

    for path in FRONTEND.rglob("*"):
        if path.suffix in (".js", ".html", ".css") and path.is_file():
            text = path.read_text()
            # Ollama keys look like <32 hex>.<24 url-safe chars>
            assert not re.search(r"\b[0-9a-f]{32}\.[A-Za-z0-9_-]{20,}", text), \
                f"an Ollama API key is exposed in {path.name}"
            assert "Bearer " not in text, \
                f"{path.name} sets a provider Authorization header"
            hit = direct_call.search(text)
            assert not hit, (f"{path.name} calls Ollama directly ({hit.group()}); "
                             "it must go through /api/v1/chat")


def test_chat_styles_use_theme_tokens_only():
    """A hard-coded colour survives a theme switch and becomes an orphan."""
    import re

    css = (FRONTEND / "assets" / "css" / "styles.css").read_text()
    start = css.index("AI ASSISTANT")
    block = css[start:]
    hits = [h for h in re.findall(r"#[0-9a-fA-F]{3,8}\b", block)
            if h.lower() not in ("#fff", "#ffffff")]
    assert not hits, f"hard-coded colours in the assistant styles: {set(hits)}"


def test_both_themes_define_the_assistant_tokens():
    """Chat tokens exist because --accent/--text-2 fail WCAG AA on the panel's
    lighter surface. Missing one in either theme reintroduces that."""
    import re

    css = (FRONTEND / "assets" / "css" / "theme.css").read_text()

    def tokens(selector: str) -> set[str]:
        start = css.index(selector)
        return set(re.findall(r"(--chat-[a-z0-9-]+):", css[start: css.index("}", start)]))

    dark, light = tokens('[data-theme="dark"]'), tokens('[data-theme="light"]')
    assert dark, "the dark theme declares no assistant tokens"
    assert dark == light, f"assistant tokens differ between themes: {dark ^ light}"


# ====================================================== configuration bug
def test_comma_separated_list_settings_parse_from_the_environment(monkeypatch):
    """Regression: pydantic-settings json.loads() list fields before validators
    run, so the `CORS_ORIGINS=*` line documented in .env.example crashed the app
    at import. Annotated[..., NoDecode] is what makes plain strings work."""
    from app.core.config import Settings

    monkeypatch.setenv("CORS_ORIGINS", "https://a.example,https://b.example")
    monkeypatch.setenv("OLLAMA_FALLBACK_MODELS", "model-a:7b,model-b:70b")
    settings = Settings()
    assert settings.CORS_ORIGINS == ["https://a.example", "https://b.example"]
    assert settings.OLLAMA_FALLBACK_MODELS == ["model-a:7b", "model-b:70b"]


def test_documented_env_example_values_do_not_crash_settings(monkeypatch):
    """`cp .env.example .env` is the documented first step; it must work."""
    from app.core.config import Settings

    monkeypatch.setenv("CORS_ORIGINS", "*")
    assert Settings().CORS_ORIGINS == ["*"]


def test_model_chain_is_deduplicated_and_primary_first(monkeypatch):
    from app.core.config import settings
    from app.services.chat.provider import model_chain

    monkeypatch.setattr(settings, "OLLAMA_MODEL", "primary:20b")
    monkeypatch.setattr(settings, "OLLAMA_FALLBACK_MODELS",
                        ["primary:20b", "backup:70b", "backup:70b"])
    assert model_chain() == ["primary:20b", "backup:70b"]


# ================================================================ security
def test_markdown_renderer_escapes_before_parsing():
    """Model output is untrusted input. The renderer escapes first and only
    then emits tags it created itself, so there is no path from model text to
    live HTML. Verified in a real browser too (no script executed, no <img>
    node created); this test guards the property at the source level."""
    js = (FRONTEND / "assets" / "js" / "chat.js").read_text()
    start = js.index("function renderMarkdown")
    body = js[start: js.index("\n}", start)]
    escape_at = body.index("const escape")
    # every rule that produces markup must come after the escaping step
    for rule in ("<strong>", "<code>", "<li>", "<table>"):
        assert body.index(rule) > escape_at, \
            f"markup rule {rule} runs before HTML escaping"
    # links are restricted to http(s): a javascript: URL must not survive
    assert "https?:\\/\\/" in body, "link rule does not restrict the URL scheme"


def test_user_messages_are_never_parsed_as_markup():
    js = (FRONTEND / "assets" / "js" / "chat.js").read_text()
    assert "bubble.textContent = content;" in js, \
        "user text must be assigned via textContent, never innerHTML"


def test_rate_limiter_is_per_client_and_bounded(monkeypatch):
    """Guards a shared free-tier quota: one noisy tab must not exhaust it."""
    from app.api.v1.endpoints import chat as chat_endpoint
    from app.core.config import settings

    monkeypatch.setattr(settings, "CHAT_RATE_LIMIT_PER_MIN", 3)
    chat_endpoint._HITS.clear()
    results = [chat_endpoint._rate_limited("10.0.0.1")[0] for _ in range(5)]
    assert results == [False, False, False, True, True]
    # a different client keeps its own budget
    assert chat_endpoint._rate_limited("10.0.0.2")[0] is False
    chat_endpoint._HITS.clear()


# ==================================================== account-level failures
def _mock_ollama(monkeypatch, handler, *, local=False):
    """Point the provider at an in-process fake Ollama endpoint."""
    import httpx

    from app.core.config import settings
    from app.services.chat import provider

    monkeypatch.setattr(settings, "OLLAMA_API_KEY", None if local else "testkey")
    monkeypatch.setattr(settings, "OLLAMA_BASE_URL",
                        "http://localhost:11434/v1" if local else "https://ollama.com/v1")
    real = httpx.AsyncClient
    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(provider.httpx, "AsyncClient",
                        lambda **kw: real(transport=transport, **kw))


@pytest.mark.asyncio
async def test_unreachable_daemon_stops_immediately(monkeypatch):
    """Connection refused is about the daemon, not the model, so retrying the
    other three candidates is pure waste — and 'provider error' would send a
    local user debugging the wrong thing entirely."""
    import httpx

    from app.services.chat import provider

    attempts = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        raise httpx.ConnectError("connection refused")

    _mock_ollama(monkeypatch, handler, local=True)
    with pytest.raises(provider.DaemonUnreachableError) as caught:
        await provider.complete([{"role": "user", "content": "hi"}])

    assert attempts["n"] == 1, "a dead daemon must stop the chain, not retry every model"
    assert "ollama serve" in (caught.value.detail or ""), \
        "the remedy must name the command that starts the daemon"


@pytest.mark.asyncio
async def test_model_not_pulled_names_the_pull_command(monkeypatch):
    """The classic local failure: the daemon runs, the model was never pulled.
    One command fixes it, so the message says which one."""
    import httpx

    from app.core.config import settings
    from app.services.chat import provider

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": 'model "x" not found, try pulling it first'})

    _mock_ollama(monkeypatch, handler, local=True)
    monkeypatch.setattr(settings, "OLLAMA_MODEL", "llama3.1")
    with pytest.raises(provider.ModelNotPulledError) as caught:
        await provider.complete([{"role": "user", "content": "hi"}])
    assert "ollama pull llama3.1" in (caught.value.detail or "")


@pytest.mark.asyncio
async def test_rejected_api_key_is_reported_distinctly(monkeypatch):
    """A bad key is not a transient error; retrying models cannot help."""
    import httpx

    from app.services.chat import provider

    attempts = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        return httpx.Response(401, json={"error": "invalid api key"})

    _mock_ollama(monkeypatch, handler)
    with pytest.raises(provider.AuthenticationError):
        await provider.complete([{"role": "user", "content": "hi"}])
    assert attempts["n"] == 1


@pytest.mark.asyncio
async def test_a_gated_model_falls_back_to_an_included_one(monkeypatch):
    """Observed live: Ollama Cloud gates 11 of its 18 models behind a paid plan
    and answers 403 'requires a subscription'. That is per-model, so the chain
    must continue rather than declaring the assistant dead."""
    import httpx

    from app.core.config import settings
    from app.services.chat import provider

    tried: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        import json as _json
        model = _json.loads(request.content)["model"]
        tried.append(model)
        if model == "gated:400b":
            return httpx.Response(
                403, json={"error": "this model requires a subscription, upgrade for access"})
        return httpx.Response(200, json={
            "model": model, "choices": [{"message": {"content": "recovered"}}]})

    _mock_ollama(monkeypatch, handler)
    monkeypatch.setattr(settings, "OLLAMA_MODEL", "gated:400b")
    monkeypatch.setattr(settings, "OLLAMA_FALLBACK_MODELS", ["gpt-oss:20b"])
    completion = await provider.complete([{"role": "user", "content": "hi"}])
    assert completion.content == "recovered"
    assert tried == ["gated:400b", "gpt-oss:20b"]


@pytest.mark.asyncio
async def test_all_models_gated_reports_a_subscription_problem(monkeypatch):
    """When every candidate is gated the chain has genuinely nothing left, and
    the remedy is a plan change, not a retry."""
    import httpx

    from app.core.config import settings
    from app.services.chat import provider

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            403, json={"error": "this model requires a subscription, upgrade for access"})

    _mock_ollama(monkeypatch, handler)
    monkeypatch.setattr(settings, "OLLAMA_MODEL", "gated-a")
    monkeypatch.setattr(settings, "OLLAMA_FALLBACK_MODELS", ["gated-b"])
    with pytest.raises(provider.SubscriptionRequiredError) as caught:
        await provider.complete([{"role": "user", "content": "hi"}])
    assert "gpt-oss:20b" in (caught.value.detail or ""), \
        "the remedy should name a model that works"


@pytest.mark.asyncio
async def test_thinking_blocks_never_reach_the_user(monkeypatch):
    """Several models served by Ollama are hybrid reasoning models and return
    the chain of thought inline. Rendering it would leak the scratchpad."""
    import httpx

    from app.services.chat import provider

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={
            "model": "gpt-oss:20b",
            "choices": [{"message": {
                "content": "<think>call a tool first</think>AAPL trades at **308.91**."}}]})

    _mock_ollama(monkeypatch, handler)
    completion = await provider.complete([{"role": "user", "content": "hi"}])
    assert "<think>" not in completion.content
    assert completion.content == "AAPL trades at **308.91**."


@pytest.mark.asyncio
async def test_tool_calls_without_an_id_are_still_usable(monkeypatch):
    """Ollama frequently omits `id` on tool calls, but the follow-up request has
    to echo one back or the exchange is rejected."""
    import httpx

    from app.services.chat import provider

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={
            "model": "gpt-oss:20b",
            "choices": [{"message": {"content": "", "tool_calls": [
                {"function": {"name": "get_quote", "arguments": '{"symbol":"AAPL"}'}}]}}]})

    _mock_ollama(monkeypatch, handler)
    completion = await provider.complete([{"role": "user", "content": "hi"}])
    assert completion.wants_tools
    call = completion.tool_calls[0]
    assert call["id"], "a synthetic id must be generated"
    assert call["raw"]["id"] == call["id"], "the echoed payload must carry the same id"


def test_error_bubbles_are_rendered_as_text_not_html():
    """An error often quotes an upstream payload; injecting it as HTML would
    both break the layout and hand markup control to a third party."""
    js = (FRONTEND / "assets" / "js" / "chat.js").read_text()
    start = js.index("} else if (role === 'error') {")
    block = js[start: start + 700]
    assert "line.textContent = part;" in block
    assert "innerHTML" not in block


# ==================================================== unavailable-state UX
def test_composer_is_disabled_when_the_assistant_cannot_answer():
    """Typing into a dead assistant only buys a delayed error. The panel must
    say what is wrong (and how to fix it) before a message is ever sent."""
    js = (FRONTEND / "assets" / "js" / "chat.js").read_text()
    assert "function disableComposer" in js
    # The message points at whoever can actually fix it, without naming env vars
    # the user has no access to.
    assert "contact the system developer" in js, \
        "the user is not told who can resolve this"
    start = js.index("function disableComposer")
    block = js[start: js.index("\n}\n", start)]
    assert "input.disabled = true" in block
    assert "send.disabled = true" in block


def test_welcome_screen_cannot_overwrite_the_unavailable_notice():
    """Regression: renderWelcome() awaits the suggestions endpoint, so it used
    to repaint over the 'assistant unavailable' notice that the health probe
    had already written — hiding the only explanation the user gets."""
    js = (FRONTEND / "assets" / "js" / "chat.js").read_text()
    start = js.index("async function renderWelcome")
    block = js[start: js.index("\nfunction renderThread", start)]
    guards = block.count("chatState.available === false")
    assert guards >= 2, (
        "renderWelcome must re-check availability after its await, not only "
        f"before it (found {guards} guard(s))")


# ================================================= account block state
@pytest.fixture(autouse=True)
def _clear_account_block():
    """The block latch is module-level; leaking it between tests would make
    unrelated cases fail with a spurious 'daemon unreachable'."""
    from app.services.chat import provider

    provider._account_block.clear()
    yield
    provider._account_block.clear()


@pytest.mark.asyncio
async def test_health_reports_a_known_block_instead_of_a_green_light():
    """A reachable endpoint is not the same as a usable one. Without this the
    panel shows a green 'available' dot on an assistant that fails on the very
    first message."""
    from app.services.chat import provider

    provider._note_block("daemon", "Could not connect to your local Ollama daemon.",
                         "Start it with: ollama serve")
    result = await provider.probe()
    assert result["reachable"] is False
    assert result["blocked"] == "daemon"
    assert "ollama serve" in result["remedy"]


@pytest.mark.asyncio
async def test_a_known_block_short_circuits_before_any_network_call(monkeypatch):
    """Once the condition is known the answer is already determined; burning
    four more round-trips only delays an honest error."""
    import httpx

    from app.core.config import settings
    from app.services.chat import provider

    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        raise AssertionError("a network call was made despite a known block")

    monkeypatch.setattr(settings, "OLLAMA_API_KEY", "testkey")
    real = httpx.AsyncClient
    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(provider.httpx, "AsyncClient",
                        lambda **kw: real(transport=transport, **kw))
    provider._note_block("auth", "Ollama rejected the API key.")

    with pytest.raises(provider.AuthenticationError):
        await provider.complete([{"role": "user", "content": "hi"}])
    assert calls["n"] == 0


def test_a_stale_block_expires_on_its_own():
    """These conditions are fixable without a server restart (start the daemon,
    pull the model, upgrade), so the latch must never be permanent."""
    from app.services.chat import provider

    provider._note_block("daemon", "x")
    assert provider._blocked() is not None
    provider._account_block["until"] = 1.0        # far in the past
    assert provider._blocked() is None
    assert not provider._account_block


@pytest.mark.asyncio
async def test_local_probe_detects_a_model_that_was_never_pulled(monkeypatch):
    """The daemon answers happily while the configured model is absent. Only a
    catalogue check catches that before the first message."""
    import httpx

    from app.core.config import settings
    from app.services.chat import provider

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [{"id": "mistral"}, {"id": "phi3"}]})

    _mock_ollama(monkeypatch, handler, local=True)
    monkeypatch.setattr(settings, "OLLAMA_MODEL", "llama3.1")
    result = await provider.probe()
    assert result["reachable"] is False
    assert result["blocked"] == "model"
    assert "ollama pull llama3.1" in result["remedy"]


def test_ui_distinguishes_the_failure_modes():
    """The status line still names the failure, so an operator glancing at the
    panel can tell the cases apart. The body text, however, comes from the
    backend's user_message — the remedy itself ("run ollama pull", "check
    OLLAMA_BASE_URL") is operator-only and stays out of the UI."""
    js = (FRONTEND / "assets" / "js" / "chat.js").read_text()
    for kind in ("daemon", "model", "auth", "subscription"):
        assert f"{kind}:" in js, f"the UI does not handle the '{kind}' block"
    assert "provider.user_message" in js, \
        "the UI ignores the user-facing message the backend computed"
    assert "provider.remedy" not in js, \
        "operator remedies must not be rendered in the chat panel"


def test_local_ollama_needs_no_api_key():
    """A local daemon is usable without a credential; requiring one would wrongly
    report a working setup as unconfigured."""
    from app.core.config import Settings

    local = Settings(_env_file=None, OLLAMA_API_KEY=None,
                     OLLAMA_BASE_URL="http://localhost:11434/v1")
    assert local.ollama_is_local is True
    assert local.chat_available is True

    cloud = Settings(_env_file=None, OLLAMA_API_KEY=None,
                     OLLAMA_BASE_URL="https://ollama.com/v1")
    assert cloud.ollama_is_local is False
    assert cloud.chat_available is False


# ================================================================ greeting
def test_panel_greets_the_user_before_they_type():
    """An empty panel makes the user guess what the assistant can do. The
    greeting states it up front, and it must survive a refactor of the welcome
    screen."""
    js = (FRONTEND / "assets" / "js" / "chat.js").read_text()
    assert "const CHAT_GREETING" in js, "the greeting constant is gone"
    assert "Hello! I'm KorisQuant AI Assistant." in js
    assert "financial analysis or investment decisions" in js, \
        "the greeting no longer says what the assistant is for"
    # The product name must be consistent: the panel, the greeting and the
    # system prompt all speaking a different name is how a rebrand half-lands.
    assert "FinAI" not in js, "the old product name survives in the chat widget"
    # rendered through the normal bubble renderer, so markdown/theming/a11y all
    # behave exactly as they do for a real answer
    assert "messageNode({ role: 'assistant', content: CHAT_GREETING })" in js


def test_greeting_is_never_replayed_to_the_model():
    """The greeting is UI chrome, not a turn anyone took. Pushing it into
    chatState.messages would send it as history on every single request —
    wasted tokens, and an invitation for the model to answer the greeting
    instead of the question."""
    js = (FRONTEND / "assets" / "js" / "chat.js").read_text()
    start = js.index("async function renderWelcome")
    body = js[start: js.index("\nfunction renderThread", start)]
    assert "chatState.messages.push" not in body, \
        "renderWelcome must not add the greeting to the transcript"
    assert "saveTranscript" not in body, \
        "the greeting must not be persisted as a conversation turn"


def test_greeting_only_shows_on_an_empty_thread():
    """It must not reappear above a restored conversation, nor be duplicated
    when the panel is closed and reopened."""
    js = (FRONTEND / "assets" / "js" / "chat.js").read_text()
    render = js[js.index("function renderThread"):]
    render = render[: render.index("\n}")]
    assert "if (!chatState.messages.length) { renderWelcome(); return; }" in render, \
        "renderThread must only greet when there is no history"

    welcome = js[js.index("async function renderWelcome"):]
    welcome = welcome[: welcome.index("\nfunction renderThread")]
    assert "chatState.messages.length) return;" in welcome, \
        "renderWelcome must bail out if a message arrived while it awaited"


# ============================================ user-facing vs operator errors
def test_users_never_see_internal_configuration_details():
    """"Check OLLAMA_BASE_URL" is meaningless to someone using the dashboard,
    leaks the deployment's architecture, and names a fix only the operator can
    apply. The panel gets a generic service message; the diagnosis is logged."""
    from app.services.chat.provider import (
        USER_FACING_UNAVAILABLE,
        ChatUnavailableError,
    )

    err = ChatUnavailableError(
        "Could not connect to the Ollama server at https://ollama.com/v1.",
        "Check OLLAMA_BASE_URL and your network connection.")

    assert err.user_message == USER_FACING_UNAVAILABLE
    assert "network connection" in err.user_message
    assert "contact the system developer" in err.user_message
    # the operator diagnosis is preserved for the logs, just not for the user
    assert "OLLAMA_BASE_URL" in (err.detail or "")

    body = err.to_dict()
    assert body["message"] == USER_FACING_UNAVAILABLE
    for leak in ("OLLAMA_BASE_URL", "ollama.com", "OLLAMA_API_KEY"):
        assert leak not in body["message"], f"{leak} leaked into the user message"


def test_production_responses_carry_no_diagnostic(monkeypatch):
    """In development the diagnosis is handy. In production anyone can open
    devtools, so endpoint URLs and config keys must not ship."""
    from app.core.config import settings
    from app.services.chat.provider import ChatUnavailableError

    err = ChatUnavailableError("Could not connect to https://ollama.com/v1.",
                               "Check OLLAMA_BASE_URL.")

    monkeypatch.setattr(settings, "DEBUG", False)
    production = err.to_dict()
    assert "diagnostic" not in production
    assert "details" not in production
    assert "ollama" not in json.dumps(production).lower()

    # The diagnosis is no longer shipped even in development: it kept finding
    # its way onto the screen. Developers read it in the server log instead,
    # which is where it was already being written.
    monkeypatch.setattr(settings, "DEBUG", True)
    assert "diagnostic" not in err.to_dict()


def test_locally_actionable_problems_keep_their_instructions():
    """Genericising everything would be over-correction: when the user runs the
    daemon themselves, naming the command is the whole point."""
    from app.services.chat.provider import (
        USER_FACING_UNAVAILABLE,
        ModelNotPulledError,
    )

    err = ModelNotPulledError(
        "The model 'llama3.1' is not available on your local Ollama.",
        "Run: ollama pull llama3.1",
        user_message="The AI model 'llama3.1' is not installed. "
                     "Run 'ollama pull llama3.1', then try again.")
    assert err.user_message != USER_FACING_UNAVAILABLE
    assert "ollama pull llama3.1" in err.user_message


def test_health_endpoint_hides_the_wiring_in_production(client, monkeypatch):
    from app.core.config import settings

    monkeypatch.setattr(settings, "DEBUG", False)
    payload = client.get("/api/v1/chat/health").json()
    blob = json.dumps(payload).lower()
    for leak in ("ollama.com", "localhost:11434", "ollama_api_key", "ollama_base_url"):
        assert leak not in blob, f"{leak} leaked from /chat/health in production"
    # the panel still gets what it needs to render a state
    assert "available" in payload and "tool_count" in payload


@pytest.mark.asyncio
async def test_an_unreachable_service_fails_fast(monkeypatch):
    """Regression: with no route to the service every model timed out in turn —
    4 models x 90 s meant six minutes of spinner before an error the first
    attempt already proved. The chain now gives up after the second timeout."""
    import httpx

    from app.core.config import settings
    from app.services.chat import provider

    attempts = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        raise httpx.ConnectTimeout("timed out")

    monkeypatch.setattr(settings, "OLLAMA_API_KEY", "testkey")
    monkeypatch.setattr(settings, "OLLAMA_BASE_URL", "https://ollama.com/v1")
    monkeypatch.setattr(settings, "OLLAMA_MODEL", "a")
    monkeypatch.setattr(settings, "OLLAMA_FALLBACK_MODELS", ["b", "c", "d"])
    real = httpx.AsyncClient
    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(provider.httpx, "AsyncClient",
                        lambda **kw: real(transport=transport, **kw))

    with pytest.raises(provider.DaemonUnreachableError) as caught:
        await provider.complete([{"role": "user", "content": "hi"}])

    assert attempts["n"] == 2, \
        f"gave up after {attempts['n']} attempts; the whole 4-model chain was walked"
    assert caught.value.user_message == provider.USER_FACING_UNAVAILABLE


def test_connect_timeout_is_shorter_than_the_generation_budget():
    """Connecting is fast or impossible; only generation is legitimately slow."""
    from app.core.config import settings

    assert settings.CHAT_CONNECT_TIMEOUT < settings.CHAT_TIMEOUT
    assert settings.CHAT_CONNECT_TIMEOUT <= 10, \
        "an unreachable service should be detected in seconds"


def test_frontend_shows_the_generic_message_and_no_config_keys():
    """Config keys must not reach the screen. Comments are stripped first: a
    comment explaining *why* OLLAMA_BASE_URL is hidden is not a leak, and
    banning the substring outright would punish documenting the decision."""
    import re

    js = (FRONTEND / "assets" / "js" / "chat.js").read_text()
    assert "Unable to connect to the AI service" in js
    assert "contact the system developer" in js

    code = re.sub(r"/\*.*?\*/", "", js, flags=re.S)       # block comments
    code = re.sub(r"^\s*//.*$", "", code, flags=re.M)      # line comments
    for leak in ("OLLAMA_BASE_URL", "OLLAMA_API_KEY", "localhost:11434", "ollama serve"):
        assert leak not in code, f"{leak} is shown to the user by chat.js"


# ====================================== no implementation details on screen
def test_error_responses_carry_only_the_user_message():
    """"Check OLLAMA_BASE_URL and your network connection." reached the chat
    panel because the detail was shipped in the response and the frontend
    rendered it. Not sending it at all removes the whole class of mistake —
    the diagnosis is already in the server log."""
    from app.services.chat.provider import ChatUnavailableError

    err = ChatUnavailableError(
        "Could not connect to the Ollama server at https://ollama.com/v1.",
        "Check OLLAMA_BASE_URL and your network connection.")
    body = err.to_dict()

    assert set(body) == {"error", "message"}, \
        f"the response carries more than the user message: {sorted(body)}"
    blob = json.dumps(body)
    for leak in ("OLLAMA_BASE_URL", "OLLAMA_API_KEY", "ollama.com", "diagnostic"):
        assert leak not in blob, f"{leak} still travels to the browser"


def test_frontend_never_renders_the_technical_detail():
    js = (FRONTEND / "assets" / "js" / "chat.js").read_text()
    assert "payload?.details?.reason" not in js, \
        "the error bubble still renders the operator detail"
    assert "provider.remedy" not in js, \
        "the panel still renders operator remedies"


def test_model_name_is_never_shown_to_the_user():
    """The model changes with the fallback chain, means nothing to the user and
    advertises the stack. The tool count and response time do carry meaning."""
    js = (FRONTEND / "assets" / "js" / "chat.js").read_text()
    assert "health.model" not in js, "the status line still names the model"
    assert "response.model" not in js, "the bubble footer still names the model"
    assert "data tools" in js, "the capability indicator was lost"
    assert "Answered in" in js, "the response time was lost"


# =========================================== absolute dates are not rendered
def test_pages_do_not_print_absolute_dates():
    """Dates were shown in five places (recommendation header, agent decision,
    transaction log, anomaly log, drawdown table). Chart axes are excluded on
    purpose: a price series without a time axis is unreadable."""
    js_dir = FRONTEND / "assets" / "js"
    banned = (
        "as of ${", "${d.as_of}", "${a.date}", "fmt.date(",
        "${e.start}", "${e.trough}", "${e.recovered}",
    )
    offenders = {}
    for path in list(js_dir.glob("*.js")) + list((js_dir / "pages").glob("*.js")):
        text = path.read_text()
        hits = [b for b in banned if b in text]
        if hits:
            offenders[path.name] = hits
    assert not offenders, f"absolute dates are still rendered: {offenders}"


def test_relative_timestamps_are_kept():
    """Over-correcting would hurt: "updated 2m ago" is how a user knows the
    quote is fresh, and "trained 3d ago" is how they judge a stale model."""
    components = (FRONTEND / "assets" / "js" / "components.js").read_text()
    overview = (FRONTEND / "assets" / "js" / "pages" / "overview.js").read_text()
    assert "fmt.timeAgo" in components, "news/alert freshness was removed"
    assert "fmt.timeAgo" in overview, "watchlist freshness was removed"


def test_chart_time_axes_are_untouched():
    """The axes are the one place a date belongs."""
    components = (FRONTEND / "assets" / "js" / "components.js").read_text()
    assert "type: 'date'" in components, "the candlestick time axis was removed"


def test_drawdown_table_keeps_what_matters_without_dates():
    """Depth, duration and whether it is over are the decision-relevant parts;
    the calendar dates were not."""
    js = (FRONTEND / "assets" / "js" / "pages" / "intelligence.js").read_text()
    table = js[js.index("function renderDrawdowns"):]
    table = table[: table.index("\n}")]
    assert "duration_days" in table and "e.depth" in table
    assert "recovered" in table, "the ongoing/recovered status was lost"
    assert "${e.start}" not in table and "${e.trough}" not in table
