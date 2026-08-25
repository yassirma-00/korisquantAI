"""The authentication wall, verification and password reset.

These tests exist because access control fails silently: a route that forgot
its guard looks perfectly normal until someone notices it never asked who they
were. The wall is asserted from the outside, with no session at all.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
FRONTEND = ROOT / "frontend"

PROTECTED_PAGES = ("/dashboard", "/index.html", "/analysis.html", "/forecast.html",
                   "/stress.html",
                   "/rl.html", "/signals.html", "/xai.html", "/portfolio.html",
                   "/risk.html")

PROTECTED_APIS = ("/api/v1/market/quote/AAPL", "/api/v1/dashboard/overview",
                  "/api/v1/portfolio", "/api/v1/rl/agents", "/api/v1/chat/health",
                  "/api/v1/forecast/models", "/api/v1/intel/algorithms")


# ============================================================ the wall
@pytest.mark.parametrize("page", PROTECTED_PAGES)
def test_pages_redirect_anonymous_visitors(anon_client, page):
    response = anon_client.get(page, follow_redirects=False)
    assert response.status_code == 303, f"{page} is reachable without signing in"
    assert response.headers["location"].startswith("/auth.html")


@pytest.mark.parametrize("endpoint", PROTECTED_APIS)
def test_apis_reject_anonymous_callers(anon_client, endpoint):
    """A page redirect would be parsed as JSON by a fetch(); APIs get a 401."""
    response = anon_client.get(endpoint, follow_redirects=False)
    assert response.status_code == 401, f"{endpoint} answered without a session"
    assert response.json()["error"] == "unauthorised"


def test_the_chatbot_is_behind_the_wall(anon_client):
    response = anon_client.post("/api/v1/chat", json={"message": "hello"},
                                follow_redirects=False)
    assert response.status_code == 401


def test_public_surface_stays_reachable(anon_client):
    """Locking the door on the sign-in screen would lock everyone out."""
    for path in ("/", "/auth.html", "/health"):
        assert anon_client.get(path).status_code == 200, f"{path} must stay public"
    # the auth screen's own assets have to load before anyone can sign in
    assert anon_client.get("/assets/css/auth.css").status_code == 200
    assert anon_client.get("/assets/js/auth.js").status_code == 200


def test_the_guard_defaults_to_deny():
    """A new route must be protected unless it is explicitly listed, so that
    forgetting to guard something fails closed rather than silently open."""
    guard = (ROOT / "backend" / "app" / "core" / "auth_guard.py").read_text()
    assert "PUBLIC_PATHS" in guard and "PUBLIC_PREFIXES" in guard
    source = guard.split("class AuthGuardMiddleware")[1]
    assert "_is_public(path)" in source, "the guard does not consult its allow-list"


def test_redirect_preserves_the_destination(anon_client):
    """Sending everyone to the dashboard would lose where they were going."""
    response = anon_client.get("/portfolio.html", follow_redirects=False)
    assert "next=/portfolio.html" in response.headers["location"]


def test_signing_in_opens_the_wall(client):
    """`client` is authenticated, so the same paths must now answer."""
    assert client.get("/dashboard").status_code == 200
    assert client.get("/api/v1/portfolio").status_code == 200


# ================================================== open-redirect guard
def test_next_parameter_cannot_leave_the_site():
    """`?next=https://evil.example` would turn a real login into a phishing
    hand-off. Only same-origin paths may be honoured."""
    js = (FRONTEND / "assets" / "js" / "auth.js").read_text()
    guard = js[js.index("function safeNext"): js.index("function el(")]
    assert "startsWith('/')" in guard, "an absolute URL would be accepted"
    assert "startsWith('//')" in guard, "a protocol-relative URL would be accepted"
    assert "auth.html" in guard, "a redirect loop back to the login page is possible"


# =========================================================== verification
def test_registration_requires_verification_and_issues_a_link(anon_client):
    response = anon_client.post("/api/v1/auth/register", json={
        "username": "verifyme", "email": "verifyme@example.com",
        "password": "Password123"})
    assert response.status_code == 200, response.text[:200]
    payload = response.json()
    assert payload["user"]["email_verified"] is False

    # With no SMTP configured the link is handed back rather than silently
    # dropped: pretending an email was sent leaves the user waiting forever.
    verification = payload["verification"]
    assert verification["delivered"] is False
    assert verification["link"] and "verify=" in verification["link"]

    token = verification["link"].split("verify=")[1]
    confirmed = anon_client.post(f"/api/v1/auth/verify?token={token}")
    assert confirmed.status_code == 200
    assert confirmed.json()["user"]["email_verified"] is True


def test_a_session_token_cannot_pass_as_a_verification_link(anon_client):
    """Purpose-scoping: one leaked token should compromise one capability."""
    registered = anon_client.post("/api/v1/auth/register", json={
        "username": "scopeuser", "email": "scope@example.com",
        "password": "Password123"}).json()
    session = registered["access_token"]

    misused = anon_client.post(f"/api/v1/auth/verify?token={session}")
    assert misused.status_code >= 400
    assert "not valid for that action" in misused.text


# ============================================ self-service reset is gone
# Removed at the owner's request. With no SMTP configured this flow returned
# the reset token in the HTTP response, so anyone who knew an email address
# could take over that account without ever reading the mailbox. The tests
# below assert the endpoints are absent, not merely unlinked.

def test_the_reset_endpoints_are_gone(anon_client):
    forgot = anon_client.post("/api/v1/auth/forgot-password",
                              json={"email": "alice@example.com"})
    assert forgot.status_code == 404, "forgot-password still answers"
    reset = anon_client.post("/api/v1/auth/reset-password",
                             json={"token": "x" * 20, "password": "Whatever123"})
    assert reset.status_code == 404, "reset-password still answers"


def test_an_email_address_alone_cannot_take_over_an_account(anon_client):
    """The exact attack that worked before removal, kept as a regression."""
    anon_client.post("/api/v1/auth/register", json={
        "username": "takeover", "email": "takeover@example.com",
        "password": "Password123"})

    stolen = anon_client.post("/api/v1/auth/forgot-password",
                              json={"email": "takeover@example.com"})
    assert stolen.status_code == 404
    assert "reset=" not in stolen.text, "a reset token is still handed out"

    # The owner's password must be untouched.
    assert anon_client.post("/api/v1/auth/login", json={
        "identifier": "takeover", "password": "Password123"}).status_code == 200


def test_no_code_path_can_still_mint_a_reset_token():
    """Leaving PURPOSE_RESET behind invites the flow back without its UI."""
    from app.core import security

    assert not hasattr(security, "PURPOSE_RESET")
    from app.services.notifications import mailer
    assert not hasattr(mailer, "send_password_reset")


def test_the_sign_in_screen_has_no_forgot_password_link():
    html = (FRONTEND / "auth.html").read_text()
    assert "Forgot password" not in html
    assert 'data-goto="forgot"' not in html
    assert "forgotForm" not in html and "viewForgot" not in html
    # The reset view was only reachable from an emailed link.
    assert "resetForm" not in html and "viewReset" not in html


def test_an_old_reset_link_explains_itself():
    """Links already sitting in inboxes must not land on a page that silently
    ignores the token."""
    js = (FRONTEND / "assets" / "js" / "auth.js").read_text()
    assert "params.get('reset')" in js, "an old ?reset= link is not handled"
    assert "no longer used" in js, "the page does not say what happened"


# ============================================================ remember me
def test_remember_me_extends_the_session(anon_client):
    from app.core.config import settings

    anon_client.post("/api/v1/auth/register", json={
        "username": "rememberer", "email": "remember@example.com",
        "password": "Password123"})
    anon_client.post("/api/v1/auth/logout")

    short = anon_client.post("/api/v1/auth/login", json={
        "identifier": "rememberer", "password": "Password123",
        "remember_me": False})
    brief = int(short.headers["set-cookie"].split("Max-Age=")[1].split(";")[0])

    long = anon_client.post("/api/v1/auth/login", json={
        "identifier": "rememberer", "password": "Password123",
        "remember_me": True})
    extended = int(long.headers["set-cookie"].split("Max-Age=")[1].split(";")[0])

    assert brief == settings.SESSION_MINUTES * 60
    assert extended == settings.SESSION_REMEMBER_MINUTES * 60
    assert extended > brief


# ====================================================== google is gone
# Google Sign-In was removed at the owner's request. These tests assert the
# absence, because a half-removed OAuth route is worse than none: dead
# endpoints keep accepting traffic and dead buttons keep making promises.

def test_no_google_endpoints_are_served(anon_client):
    """The routes must be gone, not merely unadvertised."""
    for path in ("/api/v1/auth/google/start", "/api/v1/auth/google/callback"):
        assert anon_client.get(path, follow_redirects=False).status_code == 404, \
            f"{path} still answers"


def test_the_sign_in_screen_offers_no_google_button():
    html = (FRONTEND / "auth.html").read_text()
    assert "googleLogin" not in html and "googleRegister" not in html
    assert "Continue with Google" not in html
    assert "Sign up with Google" not in html
    # The "or" dividers only existed to separate Google from the form.
    assert "auth-divider" not in html


def test_no_google_code_or_styling_survives():
    """Leftover CSS and JS is how a removed feature quietly comes back."""
    js = (FRONTEND / "assets" / "js" / "auth.js").read_text()
    css = (FRONTEND / "assets" / "css" / "auth.css").read_text()
    api = (FRONTEND / "assets" / "js" / "api.js").read_text()
    for name, text in (("auth.js", js), ("auth.css", css), ("api.js", api)):
        assert "google" not in text.lower(), f"{name} still mentions Google"


def test_the_backend_keeps_no_oauth_machinery():
    import importlib

    from app.core.config import settings

    assert not hasattr(settings, "GOOGLE_CLIENT_ID")
    assert not hasattr(settings, "google_enabled")
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("app.services.auth.google")


def test_config_no_longer_advertises_google(anon_client):
    body = anon_client.get("/api/v1/auth/config").json()
    assert "google_enabled" not in body
    assert "google_redirect_uris" not in body
    # What remains must still be answered, or the screen cannot configure itself.
    assert "require_auth" in body and "email_delivery" in body


def test_removing_google_left_no_dead_innerhtml_helper():
    """`bannerHTML` existed only to render the Google setup notice. An unused
    innerHTML sink on the credential screen is exactly the thing a later edit
    reaches for without remembering why it was dangerous."""
    js = (FRONTEND / "assets" / "js" / "auth.js").read_text()
    body = "\n".join(line for line in js.splitlines() if not line.strip().startswith("*"))
    assert "innerHTML" not in body, "an innerHTML sink survives on the auth screen"


# ================================================================== schema
def test_the_user_table_gained_its_new_columns():
    """create_all() never alters an existing table, so an installation that
    predates these columns would keep the old shape and fail at query time —
    with real accounts already inside it."""
    from app.db.migrations import ADDITIONS

    for column in ("auth_provider", "email_verified", "verified_at", "last_login"):
        assert column in ADDITIONS["users"], f"{column} has no migration"


def test_pre_existing_accounts_are_not_locked_out():
    """Accounts created before verification existed were made by someone who
    already had access; enforcing a new rule retroactively would strand them."""
    source = (ROOT / "backend" / "app" / "db" / "migrations.py").read_text()
    assert "UPDATE users SET email_verified = 1" in source
    assert "grandfathered" in source


def test_the_migration_never_destroys_anything():
    source = (ROOT / "backend" / "app" / "db" / "migrations.py").read_text()
    for destructive in ("DROP TABLE", "DROP COLUMN", "DELETE FROM", "TRUNCATE"):
        assert destructive not in source.upper(), \
            f"the migration contains {destructive}"


# ================================================================ auth page
def test_auth_page_offers_every_required_method():
    html = (FRONTEND / "auth.html").read_text()
    for element in ('id="loginForm"', 'id="registerForm"', 'id="rememberMe"'):
        assert element in html, f"the auth screen is missing {element}"
    assert "auth-strength" in html, "there is no live password strength feedback"


def test_auth_page_uses_theme_tokens_only():
    import re

    css = (FRONTEND / "assets" / "css" / "auth.css").read_text()
    hits = [h for h in re.findall(r"#[0-9a-fA-F]{3,8}\b", css)
            if h.lower() not in ("#fff", "#ffffff")]
    assert not hits, f"hard-coded colours on the auth screen: {set(hits)}"


def test_no_password_ever_appears_in_a_response(client):
    """Belt and braces: the hash must not leak through any account endpoint."""
    for endpoint in ("/api/v1/auth/me", "/api/v1/auth/status"):
        body = json.dumps(client.get(endpoint).json())
        assert "hashed_password" not in body
        assert "$2b$" not in body, "a bcrypt hash leaked into a response"


# ==================================================== validation messages
def test_validation_errors_are_readable_not_raw_pydantic(anon_client):
    """Reported from the UI: submitting username "ma" and email
    "yassirma@2021" printed

        [{"type":"string_too_short","loc":["body","username"],...}]

    into the form. FastAPI's default handler returns a list of dicts under
    "detail"; the frontend had nothing to do with that and stringified it.
    Every form on the platform shared the bug, not just registration.
    """
    response = anon_client.post("/api/v1/auth/register", json={
        "username": "ma", "email": "yassirma@2021", "password": "Password123"})
    assert response.status_code == 422
    payload = response.json()

    assert isinstance(payload.get("message"), str), "the reply has no readable message"
    for leak in ("string_too_short", '"loc"', '"ctx"', "value_error"):
        assert leak not in json.dumps(payload), f"{leak} leaked to the client"

    problems = payload["problems"]
    assert len(problems) == 2, "both problems should be reported at once"
    assert any("at least 3 characters" in p for p in problems)
    assert any("2021" in p for p in problems), \
        "the email problem does not name the offending domain"


def test_every_validation_error_type_is_humanised(anon_client):
    """A field label, not the raw attribute name, and a full sentence."""
    cases = [
        ({"email": "a@b.com", "password": "Password123"}, "Username is required."),

        ({"username": "ok", "email": "a@b.com", "password": "sh"},
         "Password must be at least 8 characters long."),
        ({"username": "bad name!", "email": "a@b.com", "password": "Password123"},
         "Username contains characters that are not allowed."),
    ]
    for body, expected in cases:
        payload = anon_client.post("/api/v1/auth/register", json=body).json()
        assert expected in payload["problems"], \
            f"{body} produced {payload['problems']} instead of {expected!r}"


def test_a_min_length_of_one_reads_as_required(anon_client):
    """"must be at least 1 characters long" is ungrammatical and a clumsy way
    to say a field is empty. Asserted on the login form, which is where
    min_length=1 actually applies."""
    payload = anon_client.post("/api/v1/auth/login",
                               json={"identifier": "", "password": ""}).json()
    assert payload["problems"] == ["Username or email is required.",
                                   "Password is required."]


def test_email_validation_explains_the_actual_mistake(anon_client):
    """"Enter a valid email address" does not tell someone who typed
    "@gmail" what is wrong with it."""
    checks = {
        "yassirma@gmail": "not a complete domain",
        "yassirma": "including the @ sign",
        "@gmail.com": "before the @ sign",
        "yassirma@.com": "not valid",
    }
    for address, expected in checks.items():
        payload = anon_client.post("/api/v1/auth/register", json={
            "username": "typotest", "email": address,
            "password": "Password123"}).json()
        assert any(expected in p for p in payload["problems"]), \
            f"{address!r} produced {payload['problems']}"


def test_a_complete_address_is_accepted(anon_client):
    """The rejection must be precise, not merely strict: the corrected form of
    the reported address has to work."""
    response = anon_client.post("/api/v1/auth/register", json={
        "username": "yassirma", "email": "yassirma@2021.com",
        "password": "Password123"})
    assert response.status_code == 200, response.text[:200]
    assert response.json()["user"]["email"] == "yassirma@2021.com"


def test_the_frontend_never_stringifies_a_validation_payload():
    js = (FRONTEND / "assets" / "js" / "api.js").read_text()
    block = js[js.index("if (!response.ok)"): js.index("return payload;")]
    assert "JSON.stringify(msg)" not in block, \
        "the client can still print raw Pydantic errors into a form"
    assert "typeof payload?.detail === 'string'" in block, \
        "a list-shaped `detail` would still be rendered"
    assert "error.problems" in block, "per-field problems are dropped"


def test_the_auth_form_lists_multiple_problems():
    js = (FRONTEND / "assets" / "js" / "auth.js").read_text()
    assert "auth-error-list" in js, "several problems are run into one sentence"
    # server text must never be parsed as HTML
    block = js[js.index("function showError"): js.index("function errorParts")]
    assert "innerHTML" not in block
    assert "item.textContent = problem" in block


# ==================================================== injection on the auth screen
def test_the_auth_banner_never_parses_untrusted_text_as_html():
    """Regression, exploited for real in a browser: `?error=` was written with
    innerHTML, so a crafted link ran script on the page where passwords are
    typed. The default path must set textContent."""
    source = (FRONTEND / "assets" / "js" / "auth.js").read_text()
    body = source.split("function banner(", 1)[1].split("\n}", 1)[0]
    assert "textContent" in body, "banner() builds HTML from its argument"
    assert "innerHTML = message" not in body

    # The query parameter must not reach the markup path unescaped.
    init = source.split("params.get('error')", 1)[1][:600]
    assert "bannerHTML(\n        `<strong>" not in init or "escapeHTML(message)" in init, (
        "the error parameter is interpolated into markup without escaping")


# ================================================== risk page period controls
def test_the_risk_page_uses_the_global_range_not_a_local_one():
    """The per-page selector was replaced by the shared control. Keeping both
    would give two widgets that disagree about the window."""
    html = (FRONTEND / "risk.html").read_text()
    assert 'id="kPeriod"' not in html, "the local period selector survives"
    assert "data-timerange" in html, "the risk page lost the global control"


def test_changing_the_range_rescans_without_a_button_press():
    """Leaving the charts on the old window while the control claims otherwise
    is worse than not offering the control."""
    js = (FRONTEND / "assets" / "js" / "pages" / "risk.js").read_text()
    assert "onTimeRangeChange(" in js, "the risk page ignores the global range"
    assert "el('kLookback').addEventListener('change'" in js, \
        "the anomaly window no longer triggers a rescan"


def test_every_risk_element_id_is_unique():
    """Two nodes shared id="kLookback" after the selector was added, so
    getElementById silently read whichever came first."""
    import collections
    import re

    html = (FRONTEND / "risk.html").read_text()
    counts = collections.Counter(re.findall(r'\sid="([^"]+)"', html))
    duplicates = {k: v for k, v in counts.items() if v > 1}
    assert not duplicates, f"duplicate element ids on risk.html: {duplicates}"


def test_an_uncomputable_score_is_not_drawn_as_zero():
    """null must render as a grey dash, never a green zero bar: one says
    'unknown', the other says 'no risk'."""
    js = (FRONTEND / "assets" / "js" / "pages" / "risk.js").read_text()
    assert "function scoreCard" in js, "there is no shared card renderer"
    card = js[js.index("function scoreCard"): js.index("async function runScan")]
    assert "null" in card and "undefined" in card, "the card cannot express 'unknown'"
    # The old code multiplied `score || 0` into a bar width.
    assert "crash.crash_risk_score || 0" not in js
    assert "bubble.bubble_score || 0" not in js
    # The VaR marker must not be pinned to 0% when there is no VaR.
    assert "(crash.var_95_daily || 0) * 100" not in js, \
        "a missing VaR still draws a line at 0%"


def test_scores_are_shown_as_percentages_with_their_scale():
    """"0.41" invites "out of what?". The card shows 41% against a stated
    0-100% scale, with the band thresholds beside it."""
    js = (FRONTEND / "assets" / "js" / "pages" / "risk.js").read_text()
    card = js[js.index("function scoreCard"): js.index("async function runScan")]
    assert "(score * 100).toFixed(0)}%" in card, "the score is not rendered as a percentage"
    assert "fmt.num(score, 2)" not in card, "the raw 0-1 decimal is still displayed"
    assert "scale 0–100%" in card, "the scale is not stated"
    assert "score-bands" in card, "the band thresholds are not shown"


def test_each_score_can_be_expanded_into_its_components():
    """A weighted composite that cannot be inspected is a number users have to
    take on faith."""
    js = (FRONTEND / "assets" / "js" / "pages" / "risk.js").read_text()
    assert "How this is calculated" in js
    assert "block.components" in js, "the per-term breakdown is never rendered"
    css = (FRONTEND / "assets" / "css" / "styles.css").read_text()
    for cls in ("score-detail", "score-parts", "score-part-bar", "risk-basis"):
        assert f".{cls}" in css, f"{cls} has no styling"


def test_the_page_states_which_sample_feeds_which_number():
    """Anomalies come from the lookback window; crash risk and the bubble read
    their own windows, floored at each model's minimum. Side by side and
    unexplained, they read as a contradiction — so the strip states the size of
    each sample, in bars rather than dates."""
    js = (FRONTEND / "assets" / "js" / "pages" / "risk.js").read_text()
    assert "riskBasis" in js, "there is no basis strip"
    assert "basis.crash_window" in js and "basis.anomalies_from" in js
    assert "window_truncated" in js, "a shortened window is not surfaced"
    html = (FRONTEND / "risk.html").read_text()
    assert 'id="riskBasis"' in html


def test_no_absolute_dates_are_printed_outside_the_charts():
    """Dates belong on a chart's x-axis. Repeating them in the surrounding
    text duplicated the axis when the windows matched it, and misled when they
    did not: at a 1M selection the chart spans 23 bars while crash risk reads
    61 and the bubble 201 — three date ranges above a chart showing a fourth.

    Plotly renders axis ticks itself, so any date *interpolated into HTML* on
    this page is by definition outside a chart.
    """
    import re

    js = (FRONTEND / "assets" / "js" / "pages" / "risk.js").read_text()
    # Server-supplied date fields, interpolated into markup.
    banned = ("period_start", "period_end", "reported_since",
              "cWin.start", "cWin.end", "bWin.start", "bWin.end")
    for field in banned:
        assert f"${{{field}" not in js and f"${{m.{field}" not in js \
            and f"${{r.{field}" not in js, (
            f"{field} is rendered as an absolute date outside a chart")
    # And the window descriptions must be expressed as a length instead.
    assert "function span(bars)" in js, "no bar-count-to-duration helper"
    assert re.search(r"span\(cWin\.bars\)", js), \
        "the crash window is not described by its length"


# ============================================ market regime detection panel
def test_the_watchlist_scan_card_is_gone_from_risk():
    """Replaced by Market Regime Detection at the owner's request. The shared
    scanWatchlist API stays — the Overview page and the alert rules still use
    it — but this card and its handler must not linger."""
    html = (FRONTEND / "risk.html").read_text()
    assert "Watchlist Alert Scan" not in html
    assert 'id="alertScanBox"' not in html and 'id="scanAllBtn"' not in html
    js = (FRONTEND / "assets" / "js" / "pages" / "risk.js").read_text()
    assert "alertScanBox" not in js and "scanAllBtn" not in js


def test_the_regime_panel_is_wired_into_the_page():
    html = (FRONTEND / "risk.html").read_text()
    for node in ('id="regimeBox"', 'id="regimeTimeline"', 'id="regimeSpells"',
                 'id="regimeRefreshBtn"', "Market Regime Detection"):
        assert node in html, f"the regime panel is missing {node}"

    js = (FRONTEND / "assets" / "js" / "pages" / "risk.js").read_text()
    assert "api.marketRegime(" in js, "the panel never calls the endpoint"
    # It must follow the symbol and period, or it describes a different
    # instrument from the charts above it.
    assert "onTimeRangeChange(() => { refreshAll() })" in js
    assert "loadRegime()" in js


def test_the_regime_panel_shows_evidence_not_just_a_verdict():
    js = (FRONTEND / "assets" / "js" / "pages" / "risk.js").read_text()
    for piece in ("r.probabilities", "r.factors", "r.insight",
                  "r.action_rationale", "r.confidence_basis", "r.timeline"):
        assert piece in js, f"the panel never renders {piece}"
    css = (FRONTEND / "assets" / "css" / "styles.css").read_text()
    for cls in ("rg-hero", "rg-prob", "rg-factor", "rg-spell", "rg-action"):
        assert f".{cls}" in css, f"{cls} has no styling"


def test_the_regime_panel_links_to_the_modules_it_informs():
    """A regime reading is only useful where positions are taken."""
    js = (FRONTEND / "assets" / "js" / "pages" / "risk.js").read_text()
    for target in ("/portfolio.html", "/signals.html", "/rl.html"):
        assert target in js, f"no link through to {target}"


def test_the_regime_endpoint_returns_a_usable_payload(client):
    body = client.get("/api/v1/risk/regime/AAPL?period=1y&include_sentiment=false")
    assert body.status_code == 200, body.text
    data = body.json()
    for field in ("regime", "label", "probability", "confidence", "probabilities",
                  "factors", "action", "insight", "timeline", "related",
                  "confidence_basis"):
        assert field in data, f"the payload is missing {field}"
    assert data["related"]["portfolio"].startswith("/api/v1/portfolio")


def test_regime_spells_never_print_an_absolute_date():
    """`fmt.timeAgo` falls back to a formatted date past 30 days, and regime
    spells routinely span months.

    The previous version of this test asserted the presence of
    ``ended ${fmt.timeAgo(s.to)}`` — which was the bug, not the fix. It passed
    while the page rendered "brief, Apr 29, 2026", because timeAgo's own
    fallback prints a date beyond 30 days. Assert that spell ages do not reach
    for timeAgo at all.
    """
    js = (FRONTEND / "assets" / "js" / "pages" / "risk.js").read_text()
    block = js[js.index("const spellAge"):]
    block = block[: block.index("\n  };")]
    assert "months" in block, "no duration wording"
    assert "fmt.timeAgo" not in block, (
        "spell ages call fmt.timeAgo, which prints an absolute date past 30 days")
    assert "brief" in block, "a single-point spell still prints a raw date"


# ================================================ shared symbol autocomplete
def _pages_with_symbol_inputs():
    """Every HTML page and the symbol-ish inputs it declares."""
    import re

    found = {}
    for path in FRONTEND.glob("*.html"):
        html = path.read_text()
        ids = []
        for match in re.finditer(r'<input[^>]*id="([^"]+)"[^>]*>', html):
            iid = match.group(1)
            if any(k in iid.lower() for k in ("symbol", "ticker", "benchmark")):
                ids.append(iid)
        if ids:
            found[path.name] = (html, ids)
    return found


def test_every_symbol_input_has_an_autocomplete_panel():
    """A bare text box accepts a typo silently: the request then fails, or
    worse, quietly analyses the wrong instrument."""
    missing = {}
    for name, (html, ids) in _pages_with_symbol_inputs().items():
        for iid in ids:
            if f'id="{iid}Panel"' not in html:
                missing.setdefault(name, []).append(iid)
    assert not missing, f"symbol inputs with no picker panel: {missing}"


def test_pages_with_a_picker_actually_load_it():
    """The panel markup is inert without symbolpicker.js, and the page script
    then throws ReferenceError: SymbolPicker is not defined."""
    offenders = []
    for name, (html, _ids) in _pages_with_symbol_inputs().items():
        if "sp-panel" not in html:
            continue
        assert "symbolpicker.js" in html, f"{name} declares a panel but never loads the picker"
        # Order matters: the class must exist before the page script runs.
        picker_at = html.index("symbolpicker.js")
        page_script = html.rfind("/assets/js/pages/")
        if page_script != -1 and picker_at > page_script:
            offenders.append(name)
    assert not offenders, f"symbolpicker.js loads after the page script on: {offenders}"


def test_the_picker_supports_keyboard_navigation():
    """Arrow keys, Enter and Escape are how this control is used without a
    mouse; the ARIA roles are how a screen reader knows it is a list at all."""
    js = (FRONTEND / "assets" / "js" / "symbolpicker.js").read_text()
    for key in ("ArrowDown", "ArrowUp", "Enter", "Escape", "Home", "End"):
        assert f"'{key}'" in js, f"the picker does not handle {key}"
    for aria in ("role', 'combobox", "aria-expanded", "aria-activedescendant",
                 "role', 'listbox"):
        assert aria in js, f"missing ARIA wiring: {aria}"
    css = (FRONTEND / "assets" / "css" / "styles.css").read_text()
    assert ".sp-item.sp-active" in css, "the keyboard highlight is invisible"


def test_the_picker_searches_names_as_well_as_tickers():
    """Typing "apple" must find AAPL — most people know the company, not the
    ticker."""
    js = (FRONTEND / "assets" / "js" / "symbolpicker.js").read_text()
    start = js.index("static score(")
    # `render()` also appears earlier as a call, so anchor on the declaration
    # that follows the scorer rather than the first occurrence of the word.
    block = js[start: js.index("\n  render()", start)]
    assert "name.startsWith(q)" in block
    assert "name.split(" in block, "a word inside the company name is not matched"


def test_multi_select_baskets_replaced_the_comma_separated_boxes():
    """"AAPL,, MSFT " parsed without complaint and offered no way to see what
    the universe contained."""
    for page, field in (("portfolio.js", "optSymbols"), ("analysis.js", "corrSymbols")):
        js = (FRONTEND / "assets" / "js" / "pages" / page).read_text()
        assert f"'{field}'" in js and "multi: true" in js, \
            f"{page} does not use a multi-select picker for {field}"
        assert f"el('{field}').value.split(',')" not in js, \
            f"{page} still parses {field} as a comma-separated string"


def test_multi_select_does_not_hijack_the_active_symbol():
    """Adding an asset to a basket must not retarget every other panel on the
    page to it — the basket is a set, not a navigation choice."""
    js = (FRONTEND / "assets" / "js" / "symbolpicker.js").read_text()
    assert "syncActive" in js
    block = js[js.index("select(symbol) {"): js.index("deselect(symbol)")]
    assert "if (this.syncActive) setActiveSymbol" in block, \
        "the picker sets the global symbol unconditionally"
def test_history_rows_show_the_trigger_values():
    js = (FRONTEND / "assets" / "js" / "pages" / "risk.js").read_text()
    assert "a.triggers" in js, "the history never renders the values that fired"
    assert "hist-triggers" in js
    css = (FRONTEND / "assets" / "css" / "styles.css").read_text()
    for cls in ("tmpl-card", "cond-row", "rule-card", "hist-row", "trig"):
        assert f".{cls}" in css, f"{cls} has no styling"


def test_a_hidden_button_stays_hidden():
    """`.btn { display: inline-flex }` outranks the HTML `hidden` attribute, so
    'Cancel edit' rendered even when the builder was not editing anything —
    the same fault that once exposed an unconfigured Google button."""
    css = (FRONTEND / "assets" / "css" / "styles.css").read_text()
    assert ".btn[hidden]" in css, "hidden is overridden by the display rule"


def test_builder_controls_are_not_stretched_to_full_width():
    """The global `input, select { width: 100% }` rule stacked the condition
    builder into one control per row."""
    css = (FRONTEND / "assets" / "css" / "styles.css").read_text()
    block = css[css.index(".cond-row select"):]
    block = block[: block.index(".rules-toolbar")]
    assert "width: auto" in block, "condition controls inherit the full-width rule"
    assert ".cond-metric" in block and "width: 200px" in block


# =========================================== AI Confidence Score (dashboard)
def test_the_alert_rule_builder_is_gone():
    """Replaced by the AI Confidence Score card at the owner's request. A
    half-removed module leaves handlers bound to elements that no longer
    exist, which throws on page load."""
    html = (FRONTEND / "risk.html").read_text()
    for node in ('id="alertModule"', 'id="conditionRows"', 'id="alertTemplates"',
                 'id="rulesBox"', 'id="createRuleBtn"', 'data-bulk='):
        assert node not in html, f"the rule builder still declares {node}"

    js = (FRONTEND / "assets" / "js" / "pages" / "risk.js").read_text()
    for symbol in ("loadRules", "submitRule", "bulkAction", "renderTemplates",
                   "loadAlertCatalogue", "conditionRow"):
        assert symbol not in js, f"dead handler {symbol} survives in risk.js"


def test_no_handler_targets_a_removed_element():
    """Every ui.el('x') on the risk page must correspond to real markup."""
    import re

    html = (FRONTEND / "risk.html").read_text()
    js = (FRONTEND / "assets" / "js" / "pages" / "risk.js").read_text()
    referenced = set(re.findall(r"ui\.el\('([A-Za-z0-9_]+)'\)", js))
    declared = set(re.findall(r'id="([A-Za-z0-9_]+)"', html))
    # Ignore ids created dynamically by the page itself.
    dynamic = set(re.findall(r"id=\"?\$\{[^}]+\}", js))
    orphans = {r for r in referenced - declared if r not in dynamic}
    assert not orphans, f"risk.js addresses elements that do not exist: {sorted(orphans)}"


def test_the_alert_history_survived_the_removal():
    """99% of stored alerts come from the automatic scanner, not custom rules.
    Removing the history with the builder would have orphaned them."""
    html = (FRONTEND / "risk.html").read_text()
    assert "Alert History" in html and 'id="historyBox"' in html
    js = (FRONTEND / "assets" / "js" / "pages" / "risk.js").read_text()
    assert "loadHistory" in js


def test_the_confidence_card_sits_beside_the_recommendation():
    """The point of the card is instant comparison: a verdict read without its
    reliability is how a coin-flip becomes a conviction call."""
    html = (FRONTEND / "index.html").read_text()
    assert 'id="verdictRow"' in html, "there is no verdict row"
    assert 'id="verdictCard"' in html and 'id="confidenceCard"' in html
    row = html[html.index('id="verdictRow"'):]
    row = row[: row.index("</div>\n\n      <div class=\"grid grid-4")]
    assert row.index('id="verdictCard"') < row.index('id="confidenceCard"'), \
        "the confidence card is not next to the recommendation"
    assert "grid-2" in html[html.index('id="verdictRow"') - 60:html.index('id="verdictRow"')], \
        "the pair is not laid out side by side"


def test_the_card_renders_gauge_bar_badge_and_factors():
    js = (FRONTEND / "assets" / "js" / "pages" / "overview.js").read_text()
    for piece in ("confidenceGauge", "conf-badge", "conf-bar", "conf-factor",
                  "report.contributors", "report.summary"):
        assert piece in js, f"the confidence card never renders {piece}"

    css = (FRONTEND / "assets" / "css" / "styles.css").read_text()
    for cls in ("conf-gauge", "conf-gauge-fg", "conf-badge", "conf-bar",
                "conf-factor", "verdict-action"):
        assert f".{cls}" in css, f"{cls} has no styling"
    # All five bands need a colour, or one renders unstyled.
    for band in ("very-high", "high", "moderate", "low", "very-low"):
        assert f".conf-{band}" in css, f"band {band} has no colour"


def test_the_gauge_animates_from_empty():
    """Setting the final dash offset inline renders the end state instantly —
    there is nothing to transition from."""
    js = (FRONTEND / "assets" / "js" / "pages" / "overview.js").read_text()
    assert "requestAnimationFrame" in js, "the animation is applied in the same frame"
    assert "data-target" in js and "strokeDashoffset" in js
    css = (FRONTEND / "assets" / "css" / "styles.css").read_text()
    block = css[css.index(".conf-gauge-fg"):]
    block = block[: block.index("}")]
    assert "transition" in block, "the ring has no transition"
    # Motion must be defeatable.
    assert "prefers-reduced-motion" in css


def test_the_dashboard_refreshes_the_card_with_the_symbol():
    """A confidence score describing a different instrument from the verdict
    beside it is worse than no score."""
    js = (FRONTEND / "assets" / "js" / "pages" / "overview.js").read_text()
    init = js[js.index("DOMContentLoaded"):]
    assert "loadConfidence()" in init, "the card never loads"
    assert init.count("loadConfidence()") >= 3, \
        "the card does not follow symbol changes and refresh"


# ==================================================== global time range (UI)
TIMED_PAGES = ("index", "analysis", "forecast", "rl", "signals", "xai",
               "portfolio", "risk")


def test_every_page_hosts_the_shared_control():
    """Six pages each carried a different period widget offering a different
    subset. Switching pages silently changed the window you were looking at."""
    missing = []
    for page in TIMED_PAGES:
        html = (FRONTEND / f"{page}.html").read_text()
        if "data-timerange" not in html or "timerange.js" not in html:
            missing.append(page)
    assert not missing, f"pages without the global time range: {missing}"


def test_the_component_loads_before_the_page_script():
    """timerange.js defines getTimeRange(); a page script that runs first
    throws ReferenceError on load."""
    late = []
    for page in TIMED_PAGES:
        html = (FRONTEND / f"{page}.html").read_text()
        component = html.index("timerange.js")
        page_script = html.rfind("/assets/js/pages/")
        if page_script != -1 and component > page_script:
            late.append(page)
    assert not late, f"timerange.js loads after the page script on: {late}"


def test_the_old_per_page_selectors_are_gone():
    """Leaving them would give two controls that disagree about the window."""
    leftovers = {}
    for page, widget in (("analysis", 'id="periodChips"'),
                         ("portfolio", 'id="pPeriodChips"'),
                         ("xai", 'id="xPeriod"'),
                         ("risk", 'id="kPeriod"')):
        html = (FRONTEND / f"{page}.html").read_text()
        if widget in html:
            leftovers[page] = widget
    assert not leftovers, f"duplicate period controls survive: {leftovers}"


def test_pages_read_and_subscribe_to_the_global_range():
    """A control nothing listens to is decoration."""
    for js in ("overview", "analysis", "forecast", "rl", "signals", "xai",
               "portfolio", "risk"):
        src = (FRONTEND / "assets" / "js" / "pages" / f"{js}.js").read_text()
        assert "initTimeRange()" in src, f"{js}.js never renders the control"
        assert "onTimeRangeChange(" in src, f"{js}.js ignores range changes"


def test_the_selection_persists_across_pages():
    """Session storage, not local: the window follows you while navigating but
    a new session starts clean rather than resurrecting last week's setting."""
    src = (FRONTEND / "assets" / "js" / "timerange.js").read_text()
    assert "sessionStorage" in src
    assert "localStorage" not in src, "the range would outlive the session"


def test_the_control_is_rendered_from_the_server_catalogue():
    """Hard-coding the list is how a page ends up offering a range the backend
    rejects."""
    src = (FRONTEND / "assets" / "js" / "timerange.js").read_text()
    assert "api.timeRanges()" in src, "the catalogue is never fetched"
    # The built-in fallback must still be valid keys, not invented ones.
    for key in ("'1d'", "'ytd'", "'max'"):
        assert key in src


def test_history_charts_are_interactive():
    """Zoom, pan, a range slider and unified hover are what make a history
    chart usable rather than a picture."""
    src = (FRONTEND / "assets" / "js" / "timerange.js").read_text()
    for feature in ("rangeslider", "rangeselector", "hovermode", "scrollZoom",
                    "dragmode", "transition"):
        assert feature in src, f"history charts lack {feature}"

    shared = (FRONTEND / "assets" / "js" / "components.js").read_text()
    assert "historyChartLayout" in shared, \
        "the shared line renderer does not use the interactive layout"


def test_the_control_is_keyboard_operable_and_animated():
    src = (FRONTEND / "assets" / "js" / "timerange.js").read_text()
    assert "ArrowRight" in src and "ArrowLeft" in src, "no keyboard navigation"
    assert "aria-pressed" in src and "role=\"group\"" in src.replace("'", '"')

    css = (FRONTEND / "assets" / "css" / "styles.css").read_text()
    block = css[css.index(".tr-marker"):]
    block = block[: block.index("}")]
    assert "transition" in block, "the highlight jumps instead of sliding"
    assert "prefers-reduced-motion" in css


def test_local_overrides_are_labelled_as_such():
    """Training length is a model hyperparameter, not a view. Keeping it local
    is right, but it must say so or it looks like the global control failed."""
    html = (FRONTEND / "forecast.html").read_text()
    assert 'id="fTrainPeriod"' in html, "the training period was folded into the global range"
    # The bare "(local)" tag was replaced by an explicit tooltip: it says what
    # the control does instead of leaving the user to infer it from a word.
    assert "Independent of the time range at the top" in html, \
        "the local override is not marked as independent"


# ================================= forecast page: display vs training window
def test_the_two_forecast_windows_are_kept_apart():
    """The chart fetch read the *training* selector, so picking 1M at the top
    redrew nothing while quietly requesting 5y."""
    js = (FRONTEND / "assets" / "js" / "pages" / "forecast.js").read_text()
    block = js[js.index("function formValues()"): js.index("async function trainModel")]
    assert "trainPeriod: ui.el('fTrainPeriod').value" in block
    assert "displayPeriod: getTimeRange()" in block

    # Training consumes the local selector...
    train = js[js.index("async function trainModel"): js.index("async function predict")]
    assert "period: v.trainPeriod" in train, "training ignores the local history selector"
    # ...and the charts consume the global one. Assert the *history* call
    # specifically: checking that `displayPeriod` appears somewhere in the
    # function passed even when the chart fetch was reverted to the training
    # window, because api.predict still used it. That is the exact bug
    # reported, so the test has to name the exact call.
    predict = js[js.index("async function predict"):]
    predict = predict[: predict.index("\n}")]
    assert "api.history(v.symbol, v.displayPeriod" in predict, \
        "the chart fetch does not use the global display window"
    assert "api.predict(v.symbol, v.model, v.horizon, v.displayPeriod" in predict, \
        "the forecast call does not use the global display window"
    assert "v.trainPeriod" not in predict, \
        "the training window leaks into the display path"


def test_the_forecast_chart_is_not_truncated_to_a_fixed_length():
    """`.slice(-180)` capped every window at ~9 months, so selecting 5Y drew
    the same chart as 1Y."""
    js = (FRONTEND / "assets" / "js" / "pages" / "forecast.js").read_text()
    assert "candles.slice(-180)" not in js, "the chart is still capped at 180 bars"


def test_training_curves_have_a_renderer_and_an_empty_state():
    js = (FRONTEND / "assets" / "js" / "pages" / "forecast.js").read_text()
    assert "function renderTrainingCurves" in js
    assert "async function loadTrainingCurves" in js
    # The exact message the owner asked for.
    assert "No training history available." in js
    assert "Train the model to generate training curves." in js
    html = (FRONTEND / "forecast.html").read_text()
    assert 'id="lossMeta"' in html, "there is nowhere to report the run's metrics"


def test_the_curves_follow_the_model_they_describe():
    """Switching architecture, horizon or symbol points at a different
    checkpoint; leaving the old curves on screen attributes one model's
    training to another."""
    js = (FRONTEND / "assets" / "js" / "pages" / "forecast.js").read_text()
    init = js[js.index("DOMContentLoaded"):]
    assert "loadTrainingCurves()" in init, "curves never load on page open"
    for control in ("fModel", "fHorizon", "fSymbol"):
        assert f"ui.el('{control}').addEventListener('change', loadTrainingCurves)" in init, \
            f"changing {control} leaves stale curves on screen"


def test_training_does_not_react_to_the_global_range():
    """Retraining every time someone zooms a chart would be both slow and
    wrong: the training window is a hyperparameter."""
    js = (FRONTEND / "assets" / "js" / "pages" / "forecast.js").read_text()
    handler = js[js.index("onTimeRangeChange(() =>"):]
    handler = handler[: handler.index(";")]
    assert "trainModel" not in handler, "changing the view retrains the model"
    assert "predict" in handler


def test_the_period_selector_explains_its_reach():
    """The control looks like a chart toolbar, so users reasonably wonder how
    far it reaches."""
    js = (FRONTEND / "assets" / "js" / "timerange.js").read_text()
    tip = ("Controls the time range displayed by all charts and historical "
           "market data on this page.")
    assert tip in js, "the info tooltip text is missing"
    assert 'class="tr-info"' in js
    # Keyboard users need it too: hover-only help is invisible to them.
    assert 'tabindex="0"' in js

    css = (FRONTEND / "assets" / "css" / "styles.css").read_text()
    assert ".tr-info::after" in css, "the tooltip has no styling"
    assert ".tr-info:focus-visible::after" in css, "the tooltip is hover-only"


def test_the_training_selector_is_labelled_as_independent():
    html = (FRONTEND / "forecast.html").read_text()
    block = html[html.index('for="fTrainPeriod"'): html.index("</select>", html.index('id="fTrainPeriod"'))]
    assert "Training history" in block
    assert "Independent of the time range at the top" in block, \
        "nothing tells the user the two controls differ"
    # A 1Y training option was missing entirely.
    assert 'value="1y"' in block


# ===================================== period selectors after the 1W/2W removal
def test_no_page_offers_a_weekly_range():
    js = (FRONTEND / "assets" / "js" / "timerange.js").read_text()
    assert "'1w'" not in js and "'2w'" not in js, \
        "the fallback list still offers a weekly range"
    assert "label: '1W'" not in js and "label: '2W'" not in js


def test_the_rl_history_selector_offers_only_long_windows():
    """An RL agent needs enough episodes of varied conditions to learn a policy
    rather than memorise one regime."""
    html = (FRONTEND / "rl.html").read_text()
    block = html[html.index('<select id="rPeriod">'):]
    block = block[: block.index("</select>")]
    import re
    values = re.findall(r'value="([^"]+)"', block)
    assert values == ["1y", "3y", "5y", "10y"], values
    # It is a training hyperparameter, so it must say it is not the view.
    label = html[html.index('for="rPeriod"'): html.index('<select id="rPeriod">')]
    assert "Training history" in label
    assert "Independent of the time range at the top" in label


def test_the_portfolio_page_has_exactly_one_period_control():
    """The Intelligence panel kept its own #iPeriod dropdown that the global
    control never touched, so one page analysed two different windows at once
    and the panel never refreshed."""
    html = (FRONTEND / "portfolio.html").read_text()
    assert 'id="iPeriod"' not in html, "a second, orphaned period selector survives"
    assert "data-timerange" in html

    js = (FRONTEND / "assets" / "js" / "pages" / "intelligence.js").read_text()
    assert "ui.el('iPeriod')" not in js, "the panel still reads the removed dropdown"
    assert "period: getTimeRange()" in js, "the panel ignores the global range"
    assert "onTimeRangeChange(" in js, "the panel never refreshes on a range change"


def test_a_rolling_statistic_refuses_to_draw_a_degenerate_axis():
    """With one or two points Plotly falls back to a millisecond time axis
    ("23:59:59.999"), which reads as a rendering fault rather than "the window
    is too short"."""
    js = (FRONTEND / "assets" / "js" / "pages" / "intelligence.js").read_text()
    block = js[js.index("function renderRolling"):]
    block = block[: block.index("\n}")]
    assert "rolling.length < 5" in block, "a near-empty rolling series is still plotted"
    assert "select 3M or more" in block, "the message does not say how to fix it"


# ============================ button busy state vs state-disabled (regression)
def test_a_state_disabled_button_does_not_claim_to_be_working():
    """`observeBusyState` toggled `.is-busy` on the bare `disabled` attribute,
    so every button disabled to express a *state* — Save on a built-in
    hyperparameter profile, Compare before two checkpoints are ticked — drew a
    progress bar that animated forever on a control that was doing nothing.
    The UI claimed to be working while it was idle.

    Busy must require the disable to follow a press of that same button.
    """
    js = (FRONTEND / "assets" / "js" / "animate.js").read_text()
    block = js[js.index("function observeBusyState"):]
    block = block[: block.index("\n}")]

    # The plain `btn.disabled` toggle is exactly the bug.
    assert "toggle('is-busy', btn.disabled)" not in block, \
        "any disabled button is still treated as in-flight"
    # A press has to be recorded, and consulted when deciding busy.
    assert "addEventListener('click'" in block, \
        "nothing records that the user pressed this button"
    assert "pressedAt" in block and "performance.now()" in block, \
        "the busy decision does not consider when the button was pressed"
    assert "btn.disabled" in block, "a button that is not disabled must never be busy"


def test_the_busy_bar_is_visible_on_both_themes():
    """The travelling bar was a hard-coded white, which vanished against the
    light theme's near-white disabled surface."""
    css = (FRONTEND / "assets" / "css" / "styles.css").read_text()
    block = css[css.index(".btn.is-busy::before"):]
    block = block[: block.index("}")]
    assert "rgba(255, 255, 255" not in block and "#fff" not in block, \
        "the progress bar is a hard-coded white"
    assert "var(--" in block, "the progress bar colour does not come from a token"


def test_a_disabled_button_stays_legible():
    """Fading a gradient button with opacity alone dropped 'Save' to 2.76:1 on
    the light topbar and left the button shape at 1.03:1 against its
    background: the control disappeared instead of reading as unavailable."""
    css = (FRONTEND / "assets" / "css" / "styles.css").read_text()
    block = css[css.index(".btn:disabled,"):]
    block = block[: block.index("}")]
    assert "opacity: 1" in block, "the disabled state still relies on opacity"
    assert "background-image: none" in block, \
        "a disabled variant keeps its gradient and still reads as primary"
    for prop in ("color:", "border-color:", "background:"):
        assert prop in block, f"the disabled state does not pin {prop}"
