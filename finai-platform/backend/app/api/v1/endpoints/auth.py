"""Registration, sign-in and email verification.

The session token is returned in the response body *and* set as an HttpOnly
cookie. The cookie is what the browser uses: a token kept in localStorage is
readable by any script on the page, so a single XSS becomes account takeover.
The body copy exists for non-browser clients (scripts, tests) with no cookie jar.
"""

from __future__ import annotations

import re
import secrets
from datetime import UTC, datetime

from fastapi import APIRouter, Depends, Query, Request, Response
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.logging import get_logger
from app.core.security import (
    PURPOSE_VERIFY,
    AuthError,
    create_access_token,
    create_link_token,
    decode_access_token,
    decode_link_token,
    hash_password,
    password_problems,
    verify_password,
)
from app.db.models import User
from app.db.session import get_db
from app.schemas.common import (
    LoginRequest,
    RegisterRequest,
)
from app.services.notifications import mailer

logger = get_logger(__name__)

router = APIRouter(prefix="/auth", tags=["Authentication"])

COOKIE_NAME = "korisquant_session"


def _set_session_cookie(response: Response, token: str, *, remember: bool) -> None:
    minutes = (settings.SESSION_REMEMBER_MINUTES if remember
               else settings.SESSION_MINUTES)
    response.set_cookie(
        COOKIE_NAME, token,
        max_age=minutes * 60,
        httponly=True,          # unreachable from JavaScript, so XSS cannot steal it
        samesite="lax",         # survives normal navigation, blocks cross-site POSTs
        # Secure would break plain-HTTP local development, so it follows the
        # environment rather than being hard-coded either way.
        secure=settings.ENVIRONMENT == "production",
        path="/",
    )


def _public(user: User) -> dict:
    """The subset of a user that is safe to send back. Never the hash."""
    return {
        "id": user.id,
        "username": user.username,
        "email": user.email,
        "full_name": user.full_name,
        "auth_provider": user.auth_provider,
        "email_verified": bool(user.email_verified),
        "risk_profile": user.risk_profile,
        "created_at": user.created_at.isoformat() if user.created_at else None,
    }


def _issue_session(response: Response, user: User, *, remember: bool) -> str:
    minutes = (settings.SESSION_REMEMBER_MINUTES if remember
               else settings.SESSION_MINUTES)
    token = create_access_token(user.id, user.username, expires_minutes=minutes)
    _set_session_cookie(response, token, remember=remember)
    return token


async def _unique_username(db: AsyncSession, preferred: str) -> str:
    """Derive a free username from an email or display name.

    Colliding with an existing account would either fail the insert or, worse,
    silently merge two people into one login.
    """
    base = re.sub(r"[^A-Za-z0-9_.-]", "", (preferred or "user").split("@")[0])[:24]
    base = base or "user"
    candidate = base
    for _ in range(50):
        exists = await db.execute(
            select(User.id).where(func.lower(User.username) == candidate.lower()))
        if exists.scalar_one_or_none() is None:
            return candidate
        candidate = f"{base}{secrets.randbelow(9000) + 1000}"
    return f"{base}{secrets.token_hex(4)}"


async def current_user(request: Request,
                       db: AsyncSession = Depends(get_db)) -> User:
    """Resolve the signed-in user, or raise 401.

    Accepts the cookie first (browsers) and falls back to a bearer header
    (scripts and tests).
    """
    token = request.cookies.get(COOKIE_NAME)
    if not token:
        header = request.headers.get("Authorization", "")
        if header.startswith("Bearer "):
            token = header[7:]
    if not token:
        raise AuthError("You need to sign in to access this.")

    claims = decode_access_token(token)
    user = await db.get(User, int(claims["sub"]))
    if user is None or not user.is_active:
        raise AuthError("This account is no longer available.")
    return user


# ============================================================ registration
@router.post("/register", summary="Create an account")
async def register(request: RegisterRequest, response: Response,
                   db: AsyncSession = Depends(get_db)):
    username = request.username.strip()
    email = request.email.strip().lower()

    problems = password_problems(request.password)
    if problems:
        raise AuthError("Your password must " + ", ".join(problems) + ".")

    # Case-insensitive uniqueness: "Alice" and "alice" being different accounts
    # is a support ticket waiting to happen.
    existing = await db.execute(
        select(User).where(
            (func.lower(User.username) == username.lower()) | (User.email == email)))
    if existing.scalar_one_or_none() is not None:
        raise AuthError("That username or email is already registered.")

    user = User(
        username=username, email=email,
        hashed_password=hash_password(request.password),
        full_name=request.full_name or None,
        auth_provider="local",
        email_verified=False,
    )
    db.add(user)
    await db.flush()

    token = create_link_token(user.id, PURPOSE_VERIFY, hours=24,
                              extra={"email": email})
    delivery = mailer.send_verification(email, user.username, token)

    session = _issue_session(response, user, remember=bool(request.remember_me))
    logger.info("account created: %s (#%s)", user.username, user.id)
    return {
        "user": _public(user),
        "access_token": session,
        "token_type": "bearer",
        "verification": delivery,
    }


# ================================================================== login
@router.post("/login", summary="Sign in")
async def login(request: LoginRequest, response: Response,
                db: AsyncSession = Depends(get_db)):
    identifier = request.identifier.strip()
    result = await db.execute(
        select(User).where(
            (func.lower(User.username) == identifier.lower())
            | (User.email == identifier.lower())))
    user = result.scalar_one_or_none()

    # One message for both "no such user" and "wrong password": saying which
    # one failed hands an attacker a way to enumerate valid accounts.
    if user is None or not user.hashed_password \
            or not verify_password(request.password, user.hashed_password):
        # An account with no password reaches the same branch —
        # so the reply also cannot reveal *how* an address is registered.
        logger.info("failed sign-in attempt for %r", identifier[:40])
        raise AuthError("Incorrect username or password.")
    if not user.is_active:
        raise AuthError("This account has been deactivated.")

    user.last_login = datetime.now(UTC)
    token = _issue_session(response, user, remember=bool(request.remember_me))
    return {"user": _public(user), "access_token": token, "token_type": "bearer"}


@router.post("/logout", summary="Sign out")
async def logout(response: Response):
    response.delete_cookie(COOKIE_NAME, path="/")
    return {"signed_out": True}


@router.get("/me", summary="The signed-in user")
async def me(user: User = Depends(current_user)):
    return {"user": _public(user)}


@router.get("/status", summary="Whether the caller is signed in")
async def status(request: Request, db: AsyncSession = Depends(get_db)):
    """Never raises: the auth screen and the dashboard header both need to ask
    "is anyone signed in?" without treating "no" as an error."""
    try:
        user = await current_user(request, db)
        return {"authenticated": True, "user": _public(user)}
    except AuthError:
        return {"authenticated": False, "user": None}


@router.get("/config", summary="Which sign-in methods are available")
async def config():
    """What the sign-in screen may offer. Email and password is the only route."""
    return {
        "email_delivery": settings.email_enabled,
        "require_auth": settings.REQUIRE_AUTH,
    }


# ==================================================== email verification
@router.post("/verify/resend", summary="Send another verification link")
async def resend_verification(user: User = Depends(current_user)):
    if user.email_verified:
        return {"already_verified": True}
    token = create_link_token(user.id, PURPOSE_VERIFY, hours=24,
                              extra={"email": user.email})
    return {"already_verified": False,
            **mailer.send_verification(user.email, user.username, token)}


@router.post("/verify", summary="Confirm an email address")
async def verify_email(token: str = Query(...), db: AsyncSession = Depends(get_db)):
    claims = decode_link_token(token, PURPOSE_VERIFY)
    user = await db.get(User, int(claims["sub"]))
    if user is None:
        raise AuthError("This verification link is no longer valid.")

    # The address may have changed since the link was issued; verifying the old
    # one would mark the *current* address confirmed without proof.
    if claims.get("email") and claims["email"] != user.email:
        raise AuthError("This link was sent to a different email address.")

    if not user.email_verified:
        user.email_verified = True
        user.verified_at = datetime.now(UTC)
        logger.info("email verified for %s (#%s)", user.username, user.id)
    return {"verified": True, "user": _public(user)}
