"""Password hashing and JWT session tokens.

Scope and honesty
-----------------
This gives KorisQuant AI real authentication: passwords are hashed with bcrypt
and sessions are signed JWTs. That is a genuine improvement over an open
dashboard, but it is not a hardened production identity system. Deliberately
absent: refresh-token rotation, password reset, email verification, MFA, and
account lockout after repeated failures. Anyone deploying this publicly should
add them, and `SECRET_KEY` must be changed from its default or every token
becomes forgeable.

Choices worth stating
---------------------
* **bcrypt, not a fast hash.** SHA-256 over a password is trivially brute
  forced; bcrypt's work factor is the whole point.
* **72-byte pre-hash.** bcrypt silently truncates at 72 bytes, so a longer
  password would validate against its own prefix. We SHA-256 first, which
  removes the ceiling entirely instead of rejecting long passwords.
* **Signed tokens, not server sessions.** The platform is stateless and often
  runs as a single process; a JWT keeps it that way with no session store.
"""

from __future__ import annotations

import base64
import hashlib
from datetime import UTC, datetime, timedelta
from typing import Any

import bcrypt
import jwt

from app.core.config import settings
from app.core.exceptions import KorisQuantError

ALGORITHM = "HS256"

# Purpose-scoped tokens. A verification link must not double as a session, and
# a reset link must not verify an email: one leaked link should compromise one
# capability, not the account.
#
# PURPOSE_RESET was removed with the self-service password reset. Leaving the
# constant behind would let a future caller mint reset tokens again for a flow
# that no longer has a UI or an owner.
PURPOSE_SESSION = "session"
PURPOSE_VERIFY = "verify_email"


class AuthError(KorisQuantError):
    """Bad credentials, or a missing/expired token."""

    status_code = 401
    code = "unauthorised"


def _prepare(password: str) -> bytes:
    """SHA-256 then base64, so bcrypt never sees more than 72 bytes.

    Without this bcrypt truncates at 72 bytes and a 100-character passphrase
    would authenticate against its first 72 characters — the extra entropy the
    user believed they had would be silently discarded.
    """
    digest = hashlib.sha256(password.encode("utf-8")).digest()
    return base64.b64encode(digest)


def hash_password(password: str) -> str:
    return bcrypt.hashpw(_prepare(password), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(_prepare(password), hashed.encode("utf-8"))
    except (ValueError, TypeError):
        # A malformed hash in the database must read as "wrong password",
        # never as a crash that leaks how storage is shaped.
        return False


def create_access_token(user_id: int, username: str,
                        expires_minutes: int | None = None) -> str:
    minutes = expires_minutes or settings.ACCESS_TOKEN_EXPIRE_MINUTES
    now = datetime.now(UTC)
    payload = {
        "sub": str(user_id),
        "username": username,
        "purpose": PURPOSE_SESSION,
        "iat": now,
        "exp": now + timedelta(minutes=minutes),
    }
    return jwt.encode(payload, settings.SECRET_KEY, algorithm=ALGORITHM)


def create_link_token(user_id: int, purpose: str, *, hours: int = 24,
                      extra: dict[str, Any] | None = None) -> str:
    """A short-lived, single-purpose token for an emailed link."""
    now = datetime.now(UTC)
    payload = {
        "sub": str(user_id),
        "purpose": purpose,
        "iat": now,
        "exp": now + timedelta(hours=hours),
        **(extra or {}),
    }
    return jwt.encode(payload, settings.SECRET_KEY, algorithm=ALGORITHM)


def decode_link_token(token: str, expected_purpose: str) -> dict[str, Any]:
    """Validate a link token *and* that it was issued for this purpose.

    Without the purpose check a verification link would be accepted as a
    password-reset link, so one leaked email would hand over the account.
    """
    claims = decode_access_token(token)
    if claims.get("purpose") != expected_purpose:
        raise AuthError("This link is not valid for that action.")
    return claims


def decode_access_token(token: str) -> dict[str, Any]:
    """Validate a token and return its claims.

    Raises:
        AuthError: expired, tampered with, or simply not a token.
    """
    try:
        return jwt.decode(token, settings.SECRET_KEY, algorithms=[ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise AuthError("Your session has expired. Please sign in again.") from None
    except jwt.InvalidTokenError:
        raise AuthError("Invalid session. Please sign in again.") from None


def password_problems(password: str) -> list[str]:
    """Human-readable reasons a password is too weak, empty when it is fine.

    Returning every failing rule at once matters: revealing them one at a time
    turns choosing a password into a guessing game.
    """
    problems: list[str] = []
    if len(password) < 8:
        problems.append("be at least 8 characters long")
    if not any(c.isalpha() for c in password):
        problems.append("contain at least one letter")
    if not any(c.isdigit() for c in password):
        problems.append("contain at least one number")
    return problems
