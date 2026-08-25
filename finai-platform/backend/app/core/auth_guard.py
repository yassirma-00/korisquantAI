"""Middleware that closes the platform to anonymous visitors.

Enforced at the edge, on purpose
--------------------------------
Protecting each router by hand means the next endpoint someone adds is public
until they remember to guard it — and nobody notices, because the failure is
silent. A default-deny middleware inverts that: a new route is protected unless
it is explicitly listed here, so forgetting fails closed.

Pages redirect, APIs return 401. A browser asking for a page should land on the
sign-in screen; a fetch() should get a status its caller can act on, not a
login page parsed as JSON.
"""

from __future__ import annotations

from fastapi import Request
from fastapi.responses import RedirectResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app.core.config import settings
from app.core.logging import get_logger
from app.utils.json_response import SafeJSONResponse

logger = get_logger(__name__)

# Reachable without a session.
PUBLIC_PATHS: set[str] = {
    "/",                    # landing page
    "/landing.html",
    "/auth.html",           # sign-in / sign-up screen
    "/health",
    "/api",
    "/favicon.ico",
}

PUBLIC_PREFIXES: tuple[str, ...] = (
    "/assets/",             # CSS, JS and images the auth screen itself needs
    "/api/v1/auth/",        # the endpoints used to *obtain* a session
)

# The API reference is developer-facing and stays behind the wall with
# everything else; EXPOSE_API_DOCS already governs whether it exists at all.
API_PREFIX = "/api/"


def _is_public(path: str) -> bool:
    if path in PUBLIC_PATHS:
        return True
    return any(path.startswith(prefix) for prefix in PUBLIC_PREFIXES)


class AuthGuardMiddleware(BaseHTTPMiddleware):
    """Require a valid session for everything not explicitly public."""

    async def dispatch(self, request: Request, call_next):
        if not settings.REQUIRE_AUTH:
            return await call_next(request)

        path = request.url.path
        # Preflight carries no cookies by design; blocking it would break CORS
        # before the real request is ever sent.
        if request.method == "OPTIONS" or _is_public(path):
            return await call_next(request)

        if self._has_valid_session(request):
            return await call_next(request)

        if path.startswith(API_PREFIX):
            return SafeJSONResponse(
                status_code=401,
                content={"error": "unauthorised",
                         "message": "You need to sign in to access this."})

        # Carry the destination so the user lands where they were going rather
        # than always on the dashboard.
        target = path
        if request.url.query:
            target = f"{path}?{request.url.query}"
        return RedirectResponse(f"/auth.html?next={target}", status_code=303)

    @staticmethod
    def _has_valid_session(request: Request) -> bool:
        """Signature and expiry only — no database round trip per request.

        A deactivated account still fails at the endpoint, where `current_user`
        loads the row. Doing that lookup here would put a query in front of
        every static asset and API call for no security gain.
        """
        from app.api.v1.endpoints.auth import COOKIE_NAME
        from app.core.security import AuthError, decode_access_token

        token = request.cookies.get(COOKIE_NAME)
        if not token:
            header = request.headers.get("Authorization", "")
            if header.startswith("Bearer "):
                token = header[7:]
        if not token:
            return False
        try:
            decode_access_token(token)
            return True
        except AuthError:
            return False
