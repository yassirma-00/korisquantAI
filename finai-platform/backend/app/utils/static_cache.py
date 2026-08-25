"""Static-file serving with correct revalidation semantics.

The problem this solves
-----------------------
Starlette's ``StaticFiles`` emits ``ETag`` and ``Last-Modified`` but no
``Cache-Control``. With no explicit directive a browser falls back to *heuristic
caching*: it invents its own freshness lifetime (commonly 10% of the file's age)
and serves the cached copy without asking the server. After an update the user
then loads a **mix** of old and new assets — e.g. a fresh ``intelligence.js``
calling into a stale ``api.js``, which fails with
``api.portfolioAnalytics is not a function``.

``Cache-Control: no-cache`` does *not* disable caching. It means "cache it, but
revalidate before every reuse". Combined with the existing ETag the browser
sends ``If-None-Match`` and receives a 304 with an empty body when nothing
changed — so the file is still served from disk, at the cost of one tiny
conditional request. Correctness with essentially no bandwidth penalty.

Hashed/immutable assets (``app.abc123.js``) would instead deserve
``immutable, max-age=31536000``; this project serves unhashed filenames, so
revalidation is the right default.
"""

from __future__ import annotations

from starlette.staticfiles import StaticFiles
from starlette.types import Scope

# Code assets must never go stale relative to one another.
REVALIDATE_SUFFIXES = (".js", ".css", ".html", ".json", ".map")
# Content-addressed media can be cached hard.
LONG_CACHE_SUFFIXES = (".woff", ".woff2", ".ttf", ".otf", ".png", ".jpg",
                       ".jpeg", ".gif", ".svg", ".webp", ".ico")


class RevalidatingStaticFiles(StaticFiles):
    """StaticFiles that tells browsers exactly how to cache each asset type."""

    def file_response(self, full_path, stat_result, scope: Scope, status_code: int = 200):
        response = super().file_response(full_path, stat_result, scope, status_code)
        path = str(full_path).lower()

        # A request carrying ?v=<content-hash> can never refer to different bytes:
        # if the file changes, the HTML emits a new URL. Such a response is safe
        # to cache hard, which removes the revalidation round-trip entirely.
        query = scope.get("query_string", b"").decode("latin-1")
        if "v=" in query:
            response.headers["Cache-Control"] = "public, max-age=31536000, immutable"
            return response

        if path.endswith(REVALIDATE_SUFFIXES):
            # Unversioned code asset (direct hit): always check with the server.
            response.headers["Cache-Control"] = "no-cache, must-revalidate"
        elif path.endswith(LONG_CACHE_SUFFIXES):
            response.headers["Cache-Control"] = "public, max-age=604800"
        else:
            response.headers["Cache-Control"] = "no-cache"

        return response
