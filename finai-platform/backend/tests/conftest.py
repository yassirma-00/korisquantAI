"""Shared pytest fixtures.

Tests run fully offline (``DATA_MODE=offline``) so they are deterministic and
never depend on a live market-data provider.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

os.environ.setdefault("DATA_MODE", "offline")
os.environ.setdefault("SYNTHETIC_SEED", "1234")

_TMP = Path(tempfile.mkdtemp(prefix="korisquant_test_"))
os.environ.setdefault("DATABASE_URL", f"sqlite+aiosqlite:///{_TMP/'test.db'}")
os.environ.setdefault("SYNC_DATABASE_URL", f"sqlite:///{_TMP/'test.db'}")

import pytest  # noqa: E402

from app.core.config import settings  # noqa: E402
from app.services.data.synthetic import generate_ohlcv  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def _isolate_artifacts():
    """Keep model checkpoints and caches out of the real data directory."""
    settings.MODEL_DIR = _TMP / "models"
    settings.ARTIFACT_DIR = _TMP / "artifacts"
    settings.CACHE_DIR = _TMP / "cache"
    settings.ensure_dirs()
    yield


@pytest.fixture
def ohlcv():
    """A deterministic 3-year daily OHLCV frame."""
    return generate_ohlcv("TEST", periods=760)


@pytest.fixture
def short_ohlcv():
    return generate_ohlcv("SHORT", periods=90)


@pytest.fixture
def crypto_ohlcv():
    return generate_ohlcv("BTC-USD", periods=500)


@pytest.fixture
def client():
    """FastAPI test client, already signed in.

    The platform now requires authentication for every page and business
    endpoint, so an anonymous client would only ever exercise the redirect.
    Registering here means each test starts where a real user starts: signed
    in. Tests that care about the *unauthenticated* case use `anon_client`.
    """
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as test_client:
        registered = test_client.post("/api/v1/auth/register", json={
            "username": "fixtureuser", "email": "fixture@example.com",
            "password": "FixturePass123"})
        # A re-used database may already hold the account; sign in instead.
        if registered.status_code != 200:
            test_client.post("/api/v1/auth/login", json={
                "identifier": "fixtureuser", "password": "FixturePass123"})
        yield test_client


@pytest.fixture
def anon_client():
    """A client with no session, for asserting that the wall is really there."""
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="session")
def tiny_regime_agent():
    """Train one small regime-aware agent so RL tests exercise a real policy.

    Skipping when no checkpoint happens to exist means the test proves nothing
    on a clean machine — which is exactly where a regression would land.
    """
    from app.services.rl.service import rl_service

    symbol, algo = "AAPL", "dueling_dqn"
    rl_service.train_single_asset(
        symbol, period="2y", algo=algo, episodes=2,
        env_overrides={"regime_aware": True})
    return symbol, algo
