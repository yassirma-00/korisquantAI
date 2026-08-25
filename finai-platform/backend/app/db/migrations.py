"""Lightweight additive schema migrations.

Why this exists
---------------
``Base.metadata.create_all`` creates missing *tables* but never alters an
existing one. When the User model gained its email-verification columns,
an installation that already had a ``users`` table would keep the old shape and
every query naming a new column would fail at runtime — with real accounts and
portfolios already inside it.

This runs on startup, adds only what is missing, and is safe to run repeatedly.
It deliberately never drops or rewrites anything: the worst outcome of a bug
here should be a column that is not added, not data that is gone. A project
that outgrows this should move to Alembic.
"""

from __future__ import annotations

from sqlalchemy import inspect, text
from sqlalchemy.ext.asyncio import AsyncConnection

from app.core.logging import get_logger

logger = get_logger(__name__)

# table -> column -> DDL type (plus default) used when the column is absent.
ADDITIONS: dict[str, dict[str, str]] = {
    "users": {
        "auth_provider": "VARCHAR(32) DEFAULT 'local'",
        "email_verified": "BOOLEAN DEFAULT 0",
        "verified_at": "DATETIME",
        "last_login": "DATETIME",
    },
    # Alert rules gained multi-condition logic, priority, expiry and delivery
    # options. Every column is nullable or defaulted so rules saved under the
    # single-threshold schema keep evaluating exactly as before.
    "alert_rules": {
        "conditions": "JSON",
        "logic": "VARCHAR(8) DEFAULT 'AND'",
        "name": "VARCHAR(120)",
        "priority": "VARCHAR(16) DEFAULT 'medium'",
        "period": "VARCHAR(8) DEFAULT '6mo'",
        "notify_in_app": "BOOLEAN DEFAULT 1",
        "notify_email": "BOOLEAN DEFAULT 0",
        "notify_push": "BOOLEAN DEFAULT 0",
        "expires_at": "DATETIME",
        "recurring": "BOOLEAN DEFAULT 1",
        "trigger_count": "INTEGER DEFAULT 0",
        "template": "VARCHAR(48)",
    },
    # The decision log gained model-governance fields: which model version and
    # algorithm produced the call, the market regime in force, how much that
    # regime actually moved the decision, and the risk figures at the time.
    # All nullable, so rows written under the older schema stay readable.
    "recommendation_log": {
        "source": "VARCHAR(32) DEFAULT 'engine'",
        "model_version": "VARCHAR(64)",
        "algo": "VARCHAR(32)",
        "regime": "VARCHAR(32)",
        "regime_confidence": "FLOAT",
        "regime_influence": "VARCHAR(16)",
        "risk_metrics": "JSON",
        "regime_explanation": "JSON",
    },
}


def _existing_columns(sync_conn, table: str) -> set[str]:
    inspector = inspect(sync_conn)
    if table not in inspector.get_table_names():
        return set()          # create_all will build it from the model
    return {c["name"] for c in inspector.get_columns(table)}


async def run_migrations(conn: AsyncConnection) -> list[str]:
    """Add any missing columns. Returns what was applied, for the log."""
    applied: list[str] = []

    for table, columns in ADDITIONS.items():
        present = await conn.run_sync(_existing_columns, table)
        if not present:
            continue
        for column, ddl in columns.items():
            if column in present:
                continue
            await conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}"))
            applied.append(f"{table}.{column}")

    # Accounts that predate email verification were created by a human who
    # already had access. Locking them out to enforce a rule introduced after
    # the fact would be a regression caused purely by an upgrade, so they are
    # grandfathered in rather than stranded.
    if any(a.startswith("users.email_verified") for a in applied):
        result = await conn.execute(text(
            "UPDATE users SET email_verified = 1, verified_at = created_at "
            "WHERE email_verified = 0 OR email_verified IS NULL"))
        if result.rowcount:
            applied.append(f"users: {result.rowcount} pre-existing account(s) grandfathered")

    if applied:
        logger.info("schema migration applied: %s", ", ".join(applied))
    return applied
