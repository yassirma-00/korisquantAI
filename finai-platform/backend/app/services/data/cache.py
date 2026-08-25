"""Two-tier cache: in-process TTL dict + parquet/CSV disk cache.

Redis is used transparently when ``REDIS_URL`` is configured and the client
library is installed; otherwise everything degrades gracefully to local memory.
"""

from __future__ import annotations

import contextlib
import json
import threading
import time
from pathlib import Path
from typing import Any

import pandas as pd

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

_LOCK = threading.RLock()
_MEMORY: dict[str, tuple[float, Any]] = {}

_redis_client = None
if settings.REDIS_URL:
    try:  # pragma: no cover - optional dependency
        import redis  # type: ignore

        _redis_client = redis.Redis.from_url(settings.REDIS_URL, socket_timeout=2)
        _redis_client.ping()
        logger.info("Redis cache enabled")
    except Exception as exc:  # pragma: no cover
        logger.warning("Redis unavailable (%s); falling back to in-memory cache", exc)
        _redis_client = None


# --------------------------------------------------------------------- KV API
def cache_get(key: str) -> Any | None:
    now = time.time()
    with _LOCK:
        hit = _MEMORY.get(key)
        if hit and hit[0] > now:
            return hit[1]
        _MEMORY.pop(key, None)
    if _redis_client is not None:  # pragma: no cover
        try:
            raw = _redis_client.get(key)
            if raw:
                return json.loads(raw)
        except Exception:
            pass
    return None


def cache_set(key: str, value: Any, ttl: int | None = None) -> None:
    ttl = ttl or settings.CACHE_TTL_SECONDS
    with _LOCK:
        _MEMORY[key] = (time.time() + ttl, value)
        if len(_MEMORY) > 2000:
            for stale in [k for k, (exp, _) in _MEMORY.items() if exp < time.time()][:500]:
                _MEMORY.pop(stale, None)
    if _redis_client is not None:  # pragma: no cover
        with contextlib.suppress(Exception):
            _redis_client.setex(key, ttl, json.dumps(value, default=str))


def cache_clear() -> None:
    with _LOCK:
        _MEMORY.clear()


# ------------------------------------------------------------------ DataFrame
def _frame_path(key: str) -> Path:
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in key)
    return Path(settings.CACHE_DIR) / f"{safe}.parquet"


def frame_get(key: str, max_age: int | None = None) -> pd.DataFrame | None:
    """Read a cached frame from memory then disk."""
    cached = cache_get(f"frame::{key}")
    if isinstance(cached, pd.DataFrame):
        return cached.copy()
    path = _frame_path(key)
    if not path.exists():
        return None
    age = time.time() - path.stat().st_mtime
    if max_age is not None and age > max_age:
        return None
    try:
        df = pd.read_parquet(path)
        cache_set(f"frame::{key}", df, ttl=settings.CACHE_TTL_SECONDS)
        return df.copy()
    except Exception as exc:
        logger.debug("Frame cache read failed for %s: %s", key, exc)
        return None


def frame_set(key: str, df: pd.DataFrame) -> None:
    if df is None or df.empty:
        return
    with _LOCK:
        _MEMORY[f"frame::{key}"] = (time.time() + settings.CACHE_TTL_SECONDS, df.copy())
    try:
        df.to_parquet(_frame_path(key))
    except Exception as exc:  # pragma: no cover - pyarrow may be missing
        logger.debug("Frame cache write skipped for %s: %s", key, exc)


def stale_frame(key: str) -> pd.DataFrame | None:
    """Any cached copy regardless of age - the last-resort fallback."""
    return frame_get(key, max_age=None)
