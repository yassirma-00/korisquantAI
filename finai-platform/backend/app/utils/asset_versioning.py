"""Content-hash cache busting for frontend assets.

Why headers alone were not enough
---------------------------------
``Cache-Control`` only helps on the *next* fetch. A browser that already stored
``/assets/js/api.js`` under heuristic freshness will reuse that entry **without
contacting the server**, so a newly-added header never gets a chance to apply.
The user keeps seeing the stale file until they manually hard-refresh — which is
not an acceptable thing to ask of every user after every deploy.

The fix is to change the *URL*, not the headers. A cache is keyed by URL, so
``/assets/js/api.js?v=6f2a1c`` simply cannot resolve to the old entry. The token
is a hash of the file's own contents, so:

* edit a file  -> its hash changes -> browsers fetch it exactly once
* leave it be  -> hash is stable   -> browsers keep caching it normally

HTML is served ``no-store`` because it is the map that points at those hashed
URLs; if the map itself were cached the whole scheme would stall.
"""

from __future__ import annotations

import hashlib
import re
from functools import lru_cache
from pathlib import Path

from app.core.logging import get_logger

logger = get_logger(__name__)

# src="/assets/..." or href="/assets/..." with no existing query string
ASSET_REF = re.compile(r'''((?:src|href)=["'])(/assets/[^"'?]+)(["'])''')


def _hash_file(path: Path) -> str:
    try:
        return hashlib.blake2b(path.read_bytes(), digest_size=6).hexdigest()
    except OSError:
        return "0"


def _fingerprint(frontend_dir: Path, url_path: str) -> str:
    """Content hash for an /assets/... URL, or '0' when the file is absent."""
    relative = url_path.lstrip("/").removeprefix("assets/")
    return _hash_file(frontend_dir / "assets" / relative)


def _version_key(frontend_dir: Path) -> tuple:
    """Cheap cache key: mtimes of every asset. Changes whenever any file does."""
    assets = frontend_dir / "assets"
    if not assets.exists():
        return ()
    return tuple(sorted(
        (str(p.relative_to(assets)), p.stat().st_mtime_ns)
        for p in assets.rglob("*") if p.is_file()
    ))


@lru_cache(maxsize=32)
def _render(html_path_str: str, frontend_dir_str: str, _key: tuple) -> str:
    """Rewrite asset references with content hashes. Cached until a file changes."""
    html_path, frontend_dir = Path(html_path_str), Path(frontend_dir_str)
    html = html_path.read_text(encoding="utf-8")

    def replace(match: re.Match) -> str:
        prefix, url, suffix = match.groups()
        return f"{prefix}{url}?v={_fingerprint(frontend_dir, url)}{suffix}"

    rewritten, count = ASSET_REF.subn(replace, html)
    logger.debug("versioned %d asset refs in %s", count, html_path.name)
    return rewritten


def render_versioned_html(html_path: Path, frontend_dir: Path) -> str:
    """Return the page with every local asset URL fingerprinted.

    Safe to call per request: the result is memoised and only recomputed when an
    asset's mtime changes, which also makes edits appear instantly in dev.
    """
    return _render(str(html_path), str(frontend_dir), _version_key(frontend_dir))


def clear_cache() -> None:
    _render.cache_clear()
