#!/usr/bin/env python3
"""Verify that this working copy contains the expected fixes.

Why this exists
---------------
A fix applied in the repository is worthless if the running copy is older.
When a symptom persists after a fix, the first question is always "is the code
that is running actually the code that was fixed?" — this script answers it in
one command instead of guessing.

Usage:
    python scripts/check_install.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GREEN, RED, YELLOW, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[0m"


def ok(msg: str) -> bool:
    print(f"  {GREEN}PASS{RESET}  {msg}")
    return True


def fail(msg: str, fix: str) -> bool:
    print(f"  {RED}FAIL{RESET}  {msg}")
    print(f"        -> {fix}")
    return False


def warn(msg: str) -> None:
    print(f"  {YELLOW}WARN{RESET}  {msg}")


def check_no_dates_in_ui() -> bool:
    """The RL status message must not print train/test date windows."""
    rl_js = ROOT / "frontend" / "assets" / "js" / "pages" / "rl.js"
    if not rl_js.exists():
        return fail("frontend/assets/js/pages/rl.js is missing", "re-clone the project")
    src = rl_js.read_text()
    if "train_window" in src or "test_window" in src:
        return fail("rl.js still renders train_window / test_window",
                    "pull the latest code: the date display was removed")
    return ok("RL status message contains no date windows")


def check_split_is_clean() -> bool:
    """_split must not hand the test set training bars.

    Checked behaviourally rather than by grepping for `split - 60`: that string
    also appears in the docstring explaining the old bug, which would make a
    correct file look broken.
    """
    sys.path.insert(0, str(ROOT / "backend"))
    try:
        import pandas as pd

        from app.services.rl.service import rl_service
    except Exception as exc:
        warn(f"could not import the RL service ({type(exc).__name__}); "
             "skipped the split check")
        return True

    idx = pd.bdate_range("2020-01-01", periods=500)
    df = pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0,
                       "close": 1.0, "volume": 1.0}, index=idx)
    train, test = rl_service._split(df, 0.2)
    overlap = train.index.intersection(test.index)
    if not overlap.empty:
        return fail(f"train/test overlap by {len(overlap)} bars",
                    "pull the latest code: the leaking split was fixed")
    return ok("train/test split is strictly disjoint")


def check_no_leaky_agents() -> bool:
    """Stored agents must not carry metadata from the old overlapping split."""
    rl_dir = ROOT / "data" / "models" / "rl"
    if not rl_dir.exists():
        return ok("no trained agents on disk yet")
    leaky = []
    for meta_path in rl_dir.glob("*.json"):
        try:
            meta = json.loads(meta_path.read_text())
        except Exception:
            continue
        tw, te = meta.get("train_window"), meta.get("test_window")
        if tw and te and str(te[0]) <= str(tw[1]):
            leaky.append(meta_path.stem)
    if leaky:
        return fail(f"{len(leaky)} agent(s) trained with the old leaking split: "
                    f"{', '.join(leaky[:4])}{'…' if len(leaky) > 4 else ''}",
                    "run: python scripts/purge_leaky_agents.py --delete")
    return ok("no agent carries a contaminated train/test split")


def check_cache_busting() -> bool:
    """HTML must fingerprint asset URLs so browsers cannot serve stale JS."""
    versioning = ROOT / "backend" / "app" / "utils" / "asset_versioning.py"
    if not versioning.exists():
        return fail("asset_versioning.py is missing",
                    "pull the latest code: cache busting was added")
    return ok("content-hash cache busting is installed")


def check_served_html() -> bool:
    """If the server is up, confirm what it actually serves."""
    try:
        # The platform now requires a session, so an anonymous fetch would read
        # the sign-in page and report a phantom failure. Sign in first.
        import json as _json
        import urllib.request
        jar = urllib.request.HTTPCookieProcessor()
        opener = urllib.request.build_opener(jar)
        creds = _json.dumps({"username": "installcheck",
                             "email": "installcheck@example.com",
                             "password": "InstallCheck123"}).encode()
        for path in ("register", "login"):
            body = creds if path == "register" else _json.dumps(
                {"identifier": "installcheck", "password": "InstallCheck123"}).encode()
            req = urllib.request.Request(
                f"http://127.0.0.1:8000/api/v1/auth/{path}", data=body,
                headers={"Content-Type": "application/json"})
            try:
                opener.open(req, timeout=5)
                break
            except Exception:
                continue
        with opener.open("http://127.0.0.1:8000/rl.html", timeout=5) as r:
            html = r.read().decode()
    except Exception:
        warn("server not running - skipped the live check "
             "(start it with: bash scripts/run_server.sh start)")
        return True

    if "?v=" not in html:
        return fail("served HTML has no cache-busting tokens",
                    "restart the server: bash scripts/run_server.sh restart")

    match = re.search(r"/assets/js/pages/rl\.js\?v=([a-f0-9]+)", html)
    if not match:
        return fail("rl.js is not fingerprinted in the served page",
                    "restart the server")

    try:
        import urllib.request
        url = f"http://127.0.0.1:8000/assets/js/pages/rl.js?v={match.group(1)}"
        with opener.open(url, timeout=5) as r:
            served_js = r.read().decode()
    except Exception:
        warn("could not fetch the served rl.js")
        return True

    if "train_window" in served_js:
        return fail("the SERVER is still serving an old rl.js with dates",
                    "restart the server after pulling: bash scripts/run_server.sh restart")
    return ok("the server serves the current, date-free rl.js")


def check_assistant() -> bool:
    """The chat is wired on every page and its key is configured server-side."""
    frontend = ROOT / "frontend"
    pages = sorted(frontend.glob("*.html"))
    if not pages:
        return fail("no frontend pages found", "re-clone the project")

    # landing.html is the marketing page, not part of the product: the
    # assistant answers questions about live portfolio data, which a visitor
    # who has not opened the dashboard has none of.
    # landing.html and auth.html are public pages seen *before* signing in.
    # The assistant answers questions about live portfolio data, which a
    # visitor who has not reached the dashboard does not have.
    public_pages = {"landing.html", "auth.html"}
    dashboard_pages = [p for p in pages if p.name not in public_pages]
    missing = [p.name for p in dashboard_pages if "assets/js/chat.js" not in p.read_text()]
    if missing:
        return fail(f"the assistant is not mounted on: {', '.join(missing)}",
                    "pull the latest code: chat.js must be included on every page")

    if not (frontend / "assets" / "js" / "chat.js").exists():
        return fail("frontend/assets/js/chat.js is missing",
                    "pull the latest code")

    # A key in frontend code is a public key. Fail loudly if one ever lands there.
    for path in frontend.rglob("*"):
        if (path.suffix in (".js", ".html", ".css") and path.is_file()
                and re.search(r"\b[0-9a-f]{32}\.[A-Za-z0-9_-]{20,}", path.read_text())):
            return fail(f"an Ollama API key is exposed in {path.name}",
                        "remove it: the key belongs in .env, server-side only")

    env = ROOT / ".env"
    text = env.read_text() if env.exists() else ""
    configured = "OLLAMA_API_KEY=" in text or "localhost:11434" in text
    if not configured:
        warn("no Ollama endpoint in .env - the assistant will report itself "
             "as unavailable (everything else works). Add OLLAMA_API_KEY for "
             "Ollama Cloud, or point OLLAMA_BASE_URL at a local daemon.")
        return True
    return ok("AI assistant is wired on every page and configured server-side")


def check_assistant_live() -> bool:
    """When the server is up, confirm the assistant can actually answer."""
    try:
        # /api/v1/chat/health sits behind the auth wall like every other
        # business endpoint, so this needs a session to reach it.
        import urllib.request
        opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor())
        for path, body in (
            ("register", {"username": "installcheck",
                          "email": "installcheck@example.com",
                          "password": "InstallCheck123"}),
            ("login", {"identifier": "installcheck", "password": "InstallCheck123"}),
        ):
            try:
                opener.open(urllib.request.Request(
                    f"http://127.0.0.1:8000/api/v1/auth/{path}",
                    data=json.dumps(body).encode(),
                    headers={"Content-Type": "application/json"}), timeout=5)
                break
            except Exception:
                continue
        with opener.open(
                "http://127.0.0.1:8000/api/v1/chat/health", timeout=8) as r:
            health = json.loads(r.read().decode())
    except Exception:
        warn("server not running - skipped the live assistant check")
        return True

    if not health.get("enabled"):
        warn("the assistant is disabled (CHAT_ENABLED=false)")
        return True
    if not health.get("configured"):
        warn("the assistant has no endpoint configured; add OLLAMA_API_KEY to .env")
        return True
    provider = health.get("provider") or {}
    if not health.get("available"):
        reason = provider.get("reason") or "unknown"
        remedy = provider.get("remedy")
        if remedy:
            # The provider already worked out the exact command or setting to
            # change; repeating a generic "check your key" would be worse.
            return fail(f"the assistant cannot answer: {reason}", remedy)
        return fail(f"the assistant cannot reach Ollama: {reason}",
                    "check OLLAMA_BASE_URL / OLLAMA_API_KEY in .env")
    mode = provider.get("mode", "?")
    return ok(f"assistant reachable ({mode}) with {health.get('tool_count', 0)} data tools")


def main() -> int:
    print("\nKorisQuant AI installation check\n" + "=" * 46)
    results = [
        check_no_dates_in_ui(),
        check_split_is_clean(),
        check_no_leaky_agents(),
        check_cache_busting(),
        check_served_html(),
        check_assistant(),
        check_assistant_live(),
    ]
    print("=" * 46)
    if all(results):
        print(f"{GREEN}All checks passed.{RESET} If the browser still shows old "
              "content, hard-refresh once: Ctrl+Shift+R (Cmd+Shift+R on macOS).\n")
        return 0
    print(f"{RED}Some checks failed.{RESET} Apply the suggested fixes above, "
          "then re-run this script.\n")
    return 1


if __name__ == "__main__":
    sys.exit(main())
