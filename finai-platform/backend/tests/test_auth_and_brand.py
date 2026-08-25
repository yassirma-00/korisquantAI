"""Authentication, and the completeness of the KorisQuant AI rename.

The rebrand tests exist because a half-finished rename is the normal outcome:
the visible headings get changed, and the old name survives in a page title, a
placeholder or a log line where nobody looks until a user screenshots it.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
FRONTEND = ROOT / "frontend"
PAGES = ("index", "analysis", "forecast", "rl", "signals", "xai", "portfolio", "risk")


# ================================================================= rebrand
def test_no_source_file_still_says_finai():
    """The database filename is the one deliberate exception: renaming it would
    orphan every portfolio and transaction already stored."""
    allowed = {"finai.db"}
    offenders: dict[str, list[str]] = {}

    for path in ROOT.rglob("*"):
        if path.suffix not in {".py", ".js", ".html", ".css", ".md", ".sh"}:
            continue
        # `tests` is excluded on purpose: the assertions that *guard* the
        # rename necessarily quote the old name, and flagging them would make
        # the guard impossible to write. Everything users can see is covered.
        if any(part in {"node_modules", "__pycache__", ".git", "data", "tests"}
               for part in path.parts):
            continue
        hits = [
            line.strip()[:90]
            for line in path.read_text(encoding="utf-8", errors="ignore").splitlines()
            if "finai" in line.lower()
            and not any(a in line for a in allowed)
            # the storage-key migration must keep naming the old prefix
            and "finai:" not in line
        ]
        if hits:
            offenders[str(path.relative_to(ROOT))] = hits[:3]

    assert not offenders, f"the old product name survives in: {offenders}"


@pytest.mark.parametrize("page", PAGES)
def test_every_dashboard_page_is_branded(client, page):
    html = client.get(f"/{page}.html").text
    assert "KorisQuant AI" in html, f"{page}.html is not branded"
    assert "FinAI" not in html, f"{page}.html still shows the old name"


def test_chat_assistant_is_renamed_everywhere():
    js = (FRONTEND / "assets" / "js" / "chat.js").read_text()
    prompt = (ROOT / "backend" / "app" / "services" / "chat" / "agent.py").read_text()

    assert "KorisQuant AI Assistant" in js, "the panel header keeps the old name"
    assert "KorisQuant AI Assistant" in prompt, "the system prompt keeps the old name"
    assert "FinAI" not in js and "FinAI" not in prompt


def test_greeting_is_the_specified_sentence():
    js = (FRONTEND / "assets" / "js" / "chat.js").read_text()
    assert "Hello! I'm KorisQuant AI Assistant." in js
    assert "How can I help you " in js
    assert "with your financial analysis or investment decisions today?" in js


def test_storage_keys_are_migrated_not_dropped():
    """Changing the localStorage prefix without carrying values over would
    silently reset every user's theme, watchlist and chat transcript — a
    regression caused purely by a cosmetic rename."""
    theme = (FRONTEND / "assets" / "js" / "theme.js").read_text()
    assert "korisquant:theme" in theme
    assert "migrateStorageKeys" in theme, "there is no migration for the old prefix"
    assert "finai:" in theme, "the migration does not reference the old prefix"

    # The inline anti-flash script runs before theme.js, so on the first load
    # after the rename it is the only thing preventing a wrong-theme flash.
    for page in PAGES:
        head = (FRONTEND / f"{page}.html").read_text().split("</head>")[0]
        assert "korisquant:theme" in head and "finai:theme" in head, \
            f"{page}.html cannot read the pre-rebrand theme key"


# ================================================================ landing
def test_landing_page_is_served_at_the_root(client):
    response = client.get("/")
    assert response.status_code == 200
    body = response.text
    assert "KorisQuant AI" in body
    for section in ('id="features"', 'id="platform"', 'id="faq"'):
        assert section in body, f"the landing page is missing {section}"


def test_dashboard_moved_but_still_reachable(client):
    assert client.get("/dashboard").status_code == 200
    assert 'class="sidebar"' in client.get("/dashboard").text


def test_landing_covers_every_advertised_capability():
    html = (FRONTEND / "landing.html").read_text()
    for capability in ("AI financial analysis", "Portfolio optimisation",
                       "Market prediction", "Reinforcement learning",
                       "Explainable AI", "Risk management",
                       "Technical analysis", "KorisQuant AI Assistant"):
        assert capability in html, f"the landing page does not mention {capability!r}"


def test_landing_has_the_required_calls_to_action():
    html = (FRONTEND / "landing.html").read_text()
    assert "Get Started" in html
    assert 'id="navLogin"' in html and 'id="navRegister"' in html
    # real screenshots, not placeholder art
    assert "screen-dashboard.png" in html
    for image in ("screen-dashboard.png", "screen-analysis.png", "screen-rl.png"):
        assert (FRONTEND / "assets" / "img" / image).exists(), f"{image} is missing"


def test_landing_footer_is_complete():
    html = (FRONTEND / "landing.html").read_text()
    assert "@korisquant.ai" in html, "no contact address"
    assert "All rights reserved" in html, "no copyright line"
    assert 'data-legal="privacy"' in html, "no privacy policy link"
    assert "Disclaimer" in html, "the not-investment-advice notice is missing"


def test_landing_uses_theme_tokens_only():
    """A hard-coded colour would survive a theme switch and leave the marketing
    page drifting away from the product it advertises."""
    import re

    css = (FRONTEND / "assets" / "css" / "landing.css").read_text()
    hits = [h for h in re.findall(r"#[0-9a-fA-F]{3,8}\b", css)
            if h.lower() not in ("#fff", "#ffffff")]
    assert not hits, f"hard-coded colours on the landing page: {set(hits)}"


# =========================================================== authentication
def test_password_hashing_is_not_reversible():
    from app.core.security import hash_password, verify_password

    hashed = hash_password("Password123")
    assert "Password123" not in hashed
    assert hashed.startswith("$2b$"), "passwords are not bcrypt-hashed"
    assert verify_password("Password123", hashed)
    assert not verify_password("Password124", hashed)


def test_long_passwords_are_not_silently_truncated():
    """bcrypt caps at 72 bytes. Without a pre-hash a 100-character passphrase
    would validate against its own first 72 characters, quietly discarding the
    entropy the user believed they had."""
    from app.core.security import hash_password, verify_password

    base = "A" * 100
    hashed = hash_password(base + "ending-one")
    assert verify_password(base + "ending-one", hashed)
    assert not verify_password(base + "ending-two", hashed), \
        "two long passwords sharing a 72-byte prefix are treated as identical"


def test_a_malformed_hash_reads_as_a_wrong_password():
    from app.core.security import verify_password

    assert verify_password("anything", "not-a-bcrypt-hash") is False


def test_tokens_are_signed_and_verified():
    from app.core.security import AuthError, create_access_token, decode_access_token

    token = create_access_token(7, "alice")
    assert decode_access_token(token)["sub"] == "7"

    tampered = token[:-4] + ("aaaa" if not token.endswith("aaaa") else "bbbb")
    with pytest.raises(AuthError):
        decode_access_token(tampered)


def test_expired_tokens_are_rejected():
    from app.core.security import AuthError, create_access_token, decode_access_token

    with pytest.raises(AuthError):
        decode_access_token(create_access_token(1, "bob", expires_minutes=-1))


def test_register_login_and_me(client):
    registered = client.post("/api/v1/auth/register", json={
        "username": "brandnew", "email": "brandnew@example.com",
        "password": "Password123", "full_name": "Brand New"})
    assert registered.status_code == 200, registered.text[:200]
    payload = registered.json()
    assert payload["user"]["username"] == "brandnew"
    # the hash must never travel to a client
    assert "hashed_password" not in json.dumps(payload)

    me = client.get("/api/v1/auth/me")
    assert me.status_code == 200
    assert me.json()["user"]["username"] == "brandnew"

    client.post("/api/v1/auth/logout")
    assert client.get("/api/v1/auth/me").status_code == 401

    signed_in = client.post("/api/v1/auth/login", json={
        "identifier": "brandnew@example.com", "password": "Password123"})
    assert signed_in.status_code == 200, "signing in by email should work"


def test_weak_passwords_are_refused(client):
    response = client.post("/api/v1/auth/register", json={
        "username": "weakuser", "email": "weak@example.com", "password": "abcdefgh"})
    assert response.status_code >= 400
    assert "number" in response.text


def test_duplicate_accounts_are_refused(client):
    client.post("/api/v1/auth/register", json={
        "username": "takenname", "email": "taken@example.com", "password": "Password123"})
    # username differing only by case must still collide
    clash = client.post("/api/v1/auth/register", json={
        "username": "TAKENNAME", "email": "other@example.com", "password": "Password123"})
    assert clash.status_code >= 400
    assert "already registered" in clash.text


def test_login_does_not_reveal_whether_an_account_exists(client):
    """Different messages for "no such user" and "wrong password" hand an
    attacker a way to enumerate valid accounts."""
    client.post("/api/v1/auth/register", json={
        "username": "realuser", "email": "real@example.com", "password": "Password123"})

    missing = client.post("/api/v1/auth/login",
                          json={"identifier": "ghost", "password": "Password123"})
    wrong = client.post("/api/v1/auth/login",
                        json={"identifier": "realuser", "password": "WrongPass123"})
    assert missing.json()["message"] == wrong.json()["message"]


def test_status_endpoint_never_errors_when_signed_out(client):
    client.post("/api/v1/auth/logout")
    response = client.get("/api/v1/auth/status")
    assert response.status_code == 200, "the landing page must be able to ask safely"
    assert response.json()["authenticated"] is False


def test_session_cookie_is_httponly(client):
    """A token readable by JavaScript turns any XSS into account takeover."""
    response = client.post("/api/v1/auth/register", json={
        "username": "cookieuser", "email": "cookie@example.com",
        "password": "Password123"})
    header = response.headers.get("set-cookie", "")
    assert "korisquant_session" in header
    assert "HttpOnly" in header, "the session cookie is readable by scripts"
    assert "SameSite=lax" in header.replace("samesite", "SameSite")


def test_the_advertised_test_count_is_true():
    """The landing and sign-in pages both boast a number of passing tests.

    Two different stale numbers were on screen at once (422 and 393) while the
    suite actually held 474: a claim nobody re-checked because nothing failed
    when it drifted. A marketing number about test coverage that is itself
    untested is exactly the kind of thing this project should not ship, so the
    figure is pinned to the real collection count here.
    """
    import os
    import re
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    # sys.executable, not "python3": a hand-built PATH found a system
    # interpreter with no pytest installed, and this test skipped itself into a
    # green tick that verified nothing.
    env = {**os.environ, "PYTHONPATH": str(root / "backend")}
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "backend/tests", "--collect-only", "-q"],
        cwd=root, capture_output=True, text=True, env=env,
    )
    # Two output shapes: a "N tests collected" summary when run directly, and
    # bare per-file counts when nested inside another pytest process. Handle
    # both, or this silently stops checking anything.
    total = re.search(r"(\d+) tests? collected", result.stdout)
    if total:
        actual = int(total.group(1))
    else:
        per_file = re.findall(r"^\S+\.py: (\d+)$", result.stdout, re.MULTILINE)
        assert per_file, (
            "could not count the suite; collection failed:\n"
            f"{result.stdout[-500:]}\n{result.stderr[-500:]}")
        actual = sum(int(n) for n in per_file)

    frontend = root / "frontend"
    claims = {
        "auth.html": int(re.search(
            r"<strong>(\d+)</strong><span>Tests passing</span>",
            (frontend / "auth.html").read_text()).group(1)),
        "landing.html": int(re.search(
            r'data-count="(\d+)">\d+</span>\s*<span class="lp-stat-label">Automated tests',
            (frontend / "landing.html").read_text()).group(1)),
    }
    for page, claimed in claims.items():
        assert claimed == actual, (
            f"{page} advertises {claimed} passing tests but the suite has {actual}")


# ==================================================== install instructions
def test_the_docs_never_invent_a_project_folder_name():
    """QUICKSTART said `cd korisquant-platform` while the directory shipped as
    `finai-platform`, so the first command failed for anyone following along.

    Naming any fixed folder is wrong anyway — users rename and relocate the
    checkout. The instructions must point at the project root generically.
    """
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    for name in ("QUICKSTART.md", "README.md"):
        for line in (root / name).read_text().splitlines():
            stripped = line.strip()
            if stripped.startswith("cd ") and "-platform" in stripped:
                raise AssertionError(
                    f"{name} hard-codes a folder name: `{stripped}`. Use a "
                    "generic path, since the checkout can be renamed.")


def test_install_instructions_cover_pep_668():
    """Debian, Ubuntu and Kali refuse system-wide pip installs. Without a
    virtualenv step the documented first command dies with
    `externally-managed-environment` before anything else can be tried."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    for name in ("QUICKSTART.md", "README.md"):
        text = (root / name).read_text()
        install = text[: text.index("pip install -r requirements.txt")]
        assert "python3 -m venv" in install, \
            f"{name} tells the reader to pip install before creating a virtualenv"
        assert "activate" in install, f"{name} never activates the virtualenv"


def test_the_launcher_prefers_the_projects_virtualenv():
    """`python3` was hard-coded, so a fully installed .venv sitting in the
    project root was ignored unless it happened to be activated — the server
    then died on ModuleNotFoundError with the packages plainly installed."""
    from pathlib import Path

    script = (Path(__file__).resolve().parents[2] / "scripts" / "run_server.sh").read_text()
    assert '"$ROOT/.venv/bin/python"' in script, "the project virtualenv is not looked for"
    assert 'setsid "$PYTHON"' in script, "the server still launches with a fixed interpreter"
    # An explicitly activated environment must still win.
    assert "VIRTUAL_ENV" in script, "an activated virtualenv is ignored"


def test_a_missing_dependency_is_explained_not_traced():
    """A stack trace tells the reader what broke; it does not tell them that
    the fix is a virtualenv."""
    from pathlib import Path

    script = (Path(__file__).resolve().parents[2] / "scripts" / "run_server.sh").read_text()
    assert "externally-managed-environment" in script, \
        "the error message does not connect to what the user actually saw"
    assert "python3 -m venv .venv" in script, "no copy-pasteable fix is offered"


def test_a_moved_project_is_diagnosed_not_misdiagnosed():
    """Moving the folder breaks the virtualenv's recorded paths, and the
    symptom is Kali's `externally-managed-environment` error — which looks
    like "no virtualenv" but is actually "the virtualenv moved".

    Telling that user to create one sends them to rebuild what they already
    have, and to re-download gigabytes of wheels for nothing.
    """
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    script = (root / "scripts" / "run_server.sh").read_text()
    assert "still points at its old location" in script, \
        "a moved project is reported as a missing virtualenv"
    assert "fix_venv.sh" in script, "the launcher does not point at the repair"


def test_the_venv_repair_script_exists_and_preserves_packages():
    """The repair must be in place. Deleting and recreating works too, but
    throws away a working ~2 GB install to fix a path string."""
    from pathlib import Path

    repair = Path(__file__).resolve().parents[2] / "scripts" / "fix_venv.sh"
    assert repair.exists(), "there is no venv repair script"
    body = repair.read_text()

    # Both halves are needed: recreating over the directory rewrites activate,
    # but leaves console-script shebangs (bin/pip) pointing at the old path.
    #
    # Check executable lines only. Searching the whole file matched the word
    # in the explanatory comment, so deleting the actual command still passed —
    # the test verified that the fix was *described*, not that it ran.
    code = "\n".join(line for line in body.splitlines()
                     if line.strip() and not line.strip().startswith("#"))
    assert "python3 -m venv" in code, "activate is never rewritten"
    assert "--force-reinstall" in code, \
        "pip's launcher shebang is never regenerated, so bin/pip stays broken"
    # It must not wipe the environment it is repairing.
    for destructive in ("rm -rf \"$VENV\"", "rm -rf $VENV", "--clear"):
        assert destructive not in body, f"the repair destroys the venv ({destructive})"


def test_the_docs_explain_the_move_failure():
    from pathlib import Path

    quickstart = (Path(__file__).resolve().parents[2] / "QUICKSTART.md").read_text()
    assert "fix_venv.sh" in quickstart, "the repair is undocumented"
    assert "externally-managed-environment" in quickstart, \
        "the docs never connect the repair to the error the user actually sees"
    assert "which pip" in quickstart, "no way to confirm which pip is active"


# ================================ brand mark + navigation icons (regression)
DASHBOARD_PAGES = PAGES + ("hyperparams", "training", "stress")


def test_the_brand_mark_is_the_same_letter_everywhere():
    """The rename shipped a half-finished mark: landing and sign-in showed "K"
    for KorisQuant while all ten dashboard pages still showed "Fi" — the FinAI
    initials — in the sidebar tile. A user signing in watched the logo change
    letter, which reads as two different products.

    `test_no_source_file_still_says_finai` did not catch it because "Fi" alone
    is not the string "finai".
    """
    marks = {}
    for page in DASHBOARD_PAGES:
        html = (FRONTEND / f"{page}.html").read_text()
        found = re.findall(r'class="brand-logo">([^<]*)<', html)
        assert found, f"{page}.html has no brand mark"
        marks[page] = found[0].strip()

    assert set(marks.values()) == {"K"}, \
        f"the sidebar mark is not consistently 'K': {marks}"

    # And it must agree with the public screens, which were already correct.
    landing = (FRONTEND / "landing.html").read_text()
    auth = (FRONTEND / "auth.html").read_text()
    assert re.search(r'class="lp-logo">K<', landing), "the landing mark drifted"
    assert re.search(r'class="auth-logo">K<', auth), "the sign-in mark drifted"


def test_navigation_uses_real_icons_not_typographic_glyphs():
    """The rail used characters (◈ ◉ ◭ ⬡ ✦ ◇ ▤ ⚠ ⚙ ◎) as icons. Those render at
    whatever weight and baseline the fallback font happens to give them, cannot
    inherit a stroke width, and several carry the wrong meaning — ⚠ is a warning
    sign, used merely to label the risk page.
    """
    glyphs = "◈◉◭⬡✦◇▤⚠⚙◎"
    for page in DASHBOARD_PAGES:
        html = (FRONTEND / f"{page}.html").read_text()
        icons = re.findall(r'<span class="nav-icon">(.*?)</span>', html, re.S)
        # Nine, not ten: the training page was hidden from the rail on request.
        # The count stays pinned so a stray or missing entry is still caught.
        assert len(icons) == 10, f"{page}.html has {len(icons)} nav icons, expected 10"
        for inner in icons:
            assert "<svg" in inner, f"{page}.html still uses a glyph icon: {inner[:20]!r}"
            assert not any(g in inner for g in glyphs), \
                f"{page}.html still contains a typographic glyph"


def test_navigation_icons_are_inlined_and_inherit_colour():
    """An icon font or CDN sprite would fail silently in the sandboxed preview,
    which has no network. Inline SVG also has to inherit `currentColor`, or the
    active/hover states cannot tint it."""
    html = (FRONTEND / "index.html").read_text()
    block = html[html.index('<aside class="sidebar"'): html.index("</aside>")]
    assert 'stroke="currentColor"' in block, "icons do not inherit the nav colour"
    assert "<link" not in block and "icon-font" not in block, \
        "the sidebar pulls an external icon resource"
    assert 'aria-hidden="true"' in block, \
        "decorative icons are not hidden from screen readers"

    css = (FRONTEND / "assets" / "css" / "styles.css").read_text()
    assert ".nav-icon svg" in css, "the inline icons have no sizing rule"


def test_upgrading_the_icons_left_navigation_intact():
    """The icons were swapped by a script. It must not have touched the anchor,
    its href or its label — `highlightNav()` matches on href."""
    # training.html is deliberately absent: it was hidden from the rail on
    # request. Every other destination must survive the icon rewrite.
    targets = ("index.html", "analysis.html", "forecast.html", "rl.html",
               "signals.html", "xai.html", "portfolio.html", "risk.html",
               "hyperparams.html", "stress.html")
    for page in DASHBOARD_PAGES:
        html = (FRONTEND / f"{page}.html").read_text()
        for target in targets:
            assert f'<a class="nav-item" href="{target}">' in html, \
                f"{page}.html lost its link to {target}"
        for label in ("Market Overview", "Technical Analysis", "Risk & Alerts",
                      "Hyperparameters", "AI Stress Testing"):
            assert label in html, f"{page}.html lost the {label!r} label"


def test_no_page_hardcodes_a_theme_toggle():
    """The switch is injected by `mountThemeToggle`, which is idempotent: it
    skips a topbar that already contains a `.theme-toggle`.

    Regression: stress.html shipped its own empty `<button class="theme-toggle">`.
    The mount then did nothing, so the page had a button with no knob, no icons
    and no click handler — dark/light was dead on that page only, while every
    other page worked. Markup cannot own this element.
    """
    offenders = {}
    for page in DASHBOARD_PAGES:
        html = (FRONTEND / f"{page}.html").read_text()
        if 'class="theme-toggle"' in html:
            offenders[page] = "declares .theme-toggle in markup"
    assert not offenders, (
        f"pages hardcode the theme switch instead of letting theme.js mount it: "
        f"{offenders}")

    # And the mounting path must still exist, or removing the markup would
    # simply leave every page without a switch.
    theme_js = (FRONTEND / "assets" / "js" / "theme.js").read_text()
    assert "mountThemeToggle('.topbar')" in theme_js, \
        "nothing injects the theme switch any more"
    assert "toggleTheme" in theme_js, "the switch has no click behaviour"

    # Every dashboard page must load the controller and have a topbar to host it.
    for page in DASHBOARD_PAGES:
        html = (FRONTEND / f"{page}.html").read_text()
        assert "/assets/js/theme.js" in html, f"{page}.html does not load theme.js"
        assert 'class="topbar"' in html, f"{page}.html has no topbar to mount into"


def test_both_themes_define_the_same_tokens():
    """A token defined for one theme only renders as an invalid value on the
    other, which shows up as an invisible or unstyled element rather than an
    error. Parity is the only way to catch it before a user does."""
    import re

    css = (FRONTEND / "assets" / "css" / "theme.css").read_text()

    def tokens(block: str) -> set[str]:
        return set(re.findall(r"(--[a-z0-9-]+)\s*:", block))

    # The dark palette is declared as `:root, [data-theme="dark"] { ... }`, so
    # anchor on the data-theme selector rather than on ":root" alone — the
    # latter also matches a later, unrelated :root block of layout tokens.
    dark = re.search(r'\[data-theme=["\']dark["\']\]\s*\{(.*?)\n\}', css, re.S)
    light = re.search(r'\[data-theme=["\']light["\']\]\s*\{(.*?)\n\}', css, re.S)
    assert dark and light, "theme.css no longer defines both palettes"

    # Only colour-bearing tokens need parity. Geometry, type scale, easings and
    # durations are deliberately theme-agnostic: they are declared once in
    # :root and inherited, so requiring the light block to redeclare them would
    # be demanding duplication, not catching a bug.
    structural = ("--font", "--mono", "--radius", "--sp-", "--fs-", "--t-",
                  "--ease", "--sidebar-w", "--topbar-h")

    def colours(names: set[str]) -> set[str]:
        return {n for n in names if not n.startswith(structural)}

    dark_tokens = colours(tokens(dark.group(1)))
    light_tokens = colours(tokens(light.group(1)))
    missing = dark_tokens - light_tokens
    assert not missing, (
        f"light theme does not define these colour tokens: {sorted(missing)}")
    assert len(dark_tokens) > 20, \
        "the colour-token filter matched almost nothing; the check is vacuous"


def test_the_topbar_scrolls_before_it_pushes_the_page():
    """The topbar is a nowrap flex row: title, range picker, search, actions.

    Its scroll fallback started at 860px, but the row stops fitting around
    1050px — while the sidebar is still on screen. Between those widths it
    pushed the document instead of scrolling, so the whole page could be
    dragged sideways (measured: 49px at 1024, 195px at 870) and the sticky
    topbar tore away from the content under it. iPad landscape and 11" laptops
    sit inside that band.
    """
    import re

    css = (FRONTEND / "assets" / "css" / "styles.css").read_text()

    # Find every media query that gives the topbar a horizontal scroll.
    breakpoints = []
    for match in re.finditer(
            r"@media\s*\(max-width:\s*(\d+)px\)\s*\{(.*?)\n\}", css, re.S):
        width, block = int(match.group(1)), match.group(2)
        if re.search(r"\.topbar[^{]*\{[^}]*overflow-x:\s*auto", block, re.S):
            breakpoints.append(width)

    assert breakpoints, "no breakpoint gives the topbar a scroll fallback"
    assert max(breakpoints) >= 1100, (
        f"the topbar only starts scrolling at {max(breakpoints)}px, but the row "
        f"stops fitting around 1050px; widths between those push the page")

    # The scroll only works if the children are allowed to keep their size and
    # the row is permitted to shrink below its content.
    guard = css[css.index("@media (max-width: 1100px)"):]
    guard = guard[: guard.index("\n}")]
    assert "flex-shrink: 0" in guard, \
        "topbar children may shrink, so the row will not scroll"


def test_the_page_reserves_room_for_the_chat_launcher():
    """The launcher is fixed at bottom-right. Without a reserved strip it sat on
    top of real data once a page was scrolled to the end — the confidence scale
    label on Market Overview, a risk percentage on Risk & Alerts, a timestamp on
    Hyperparameters."""
    import re

    css = (FRONTEND / "assets" / "css" / "styles.css").read_text()

    fab = re.search(r"\.chat-fab\s*\{(.*?)\n\}", css, re.S)
    assert fab, "the chat launcher is gone"
    height = re.search(r"height:\s*(\d+)px", fab.group(1))
    inset = re.search(r"bottom:\s*(\d+)px", fab.group(1))
    assert height and inset, "the launcher has no fixed footprint to reserve for"
    needed = int(height.group(1)) + int(inset.group(1))

    content = re.search(r"\.content\s*\{(.*?)\n\}", css, re.S)
    assert content, ".content rule not found"
    # Stop at the semicolon, not the first ')': the expression contains
    # `var(--sp-6)`, so a lazy `[^)]*` captured nothing useful.
    declaration = re.search(r"padding-bottom:\s*calc\((.*?)\);",
                            content.group(1), re.S)
    assert declaration, ".content reserves no room for the fixed launcher"
    # The pad is a calc() of the normal spacing token plus a pixel reserve; it
    # is the pixel term that has to cover the launcher.
    pixels = [int(n) for n in re.findall(r"(\d+)px", declaration.group(1))]
    assert pixels, "the reserve is not expressed in pixels"
    assert max(pixels) >= needed, (
        f"reserved {max(pixels)}px but the launcher occupies {needed}px")


def test_the_chat_launcher_is_named_not_just_a_symbol():
    """The launcher rendered a single glyph, "◈" — a decorative diamond that
    tells a first-time user nothing about what the button does. Discovering the
    assistant required hovering for a tooltip or clicking to find out.

    It now carries the word "Assistant" beside a speech-bubble icon. This pins
    the label so a future restyle cannot quietly return to a bare symbol.
    """
    js = (FRONTEND / "assets" / "js" / "chat.js").read_text()
    css = (FRONTEND / "assets" / "css" / "styles.css").read_text()

    block = js[js.index("fab.className = 'chat-fab'"):]
    block = block[: block.index("const panel")]

    # Strip comment lines before checking for the old glyph: the code comment
    # explaining the fix quotes "◈" itself, and matching that is a false
    # positive — the test must look at what is rendered, not at prose.
    rendered = "\n".join(line for line in block.splitlines()
                         if not line.lstrip().startswith("//"))

    assert "Assistant" in rendered, "the launcher shows no visible name"
    assert "\u25c8" not in rendered, "the ambiguous diamond glyph is back"
    assert "<svg" in rendered, "the launcher lost its icon"
    # The accessible name must survive too: the visible label alone is not
    # enough for a screen reader if the icon is not hidden from it.
    assert 'aria-label' in block and "assistant" in block.lower()
    assert 'aria-hidden="true"' in block, "the icon is exposed to screen readers"

    # A label needs a shape that can hold it: a fixed 54px circle would clip it.
    rule = re.search(r"\.chat-fab\s*\{(.*?)\n\}", css, re.S)
    assert rule, "the launcher rule is gone"
    body = rule.group(1)
    assert "border-radius: 50%" not in body, \
        "the launcher is still a circle; the label cannot fit"
    assert "white-space: nowrap" in body, "the label may wrap inside the button"
    # Height is what the page reserves at the foot of .content; keep it pinned.
    assert re.search(r"height:\s*54px", body), \
        "the launcher height changed; the reserved strip no longer matches"
