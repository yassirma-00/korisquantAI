"""FastAPI application entry point.

Run locally:
    uvicorn app.main:app --reload --port 8000

Docs:
    http://localhost:8000/docs      (Swagger UI)
    http://localhost:8000/redoc     (ReDoc)
    http://localhost:8000/          (bundled dashboard)
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

from app.api.v1.router import api_router
from app.core.auth_guard import AuthGuardMiddleware
from app.core.config import PROJECT_ROOT, settings
from app.core.exceptions import KorisQuantError
from app.core.logging import configure_logging, get_logger
from app.db.session import init_db
from app.schemas.common import HealthResponse
from app.services.data.market_data import market_data_service
from app.utils.asset_versioning import render_versioned_html
from app.utils.json_response import SafeJSONResponse
from app.utils.static_cache import RevalidatingStaticFiles

configure_logging()
logger = get_logger(__name__)

FRONTEND_DIR = PROJECT_ROOT / "frontend"


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting %s v%s [%s] data_mode=%s",
                settings.APP_NAME, settings.APP_VERSION, settings.ENVIRONMENT, settings.DATA_MODE)
    settings.ensure_dirs()
    # Hyperparameter YAML files are provisioned automatically: a fresh clone or
    # a container built without configs/ would otherwise fail every training
    # request with no way to recover from the dashboard. Additive only — files
    # the user has edited are never overwritten.
    try:
        from app.services.rl.hyperparams import hyperparameters

        hyperparameters.ensure_configs()
    except Exception as exc:      # pragma: no cover - must never block boot
        logger.warning("hyperparameter configs could not be provisioned: %s", exc)
    await init_db()

    scheduler = None
    if settings.SCHEDULER_ENABLED:
        try:
            from app.workers.scheduler import start_scheduler
            scheduler = start_scheduler()
            logger.info("Background scheduler started")
        except Exception as exc:
            logger.warning("Scheduler could not start: %s", exc)

    yield

    if scheduler:
        scheduler.shutdown(wait=False)
    logger.info("Shutdown complete")


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=(
        "Intelligent web platform for financial analysis and portfolio management.\n\n"
        "**Capabilities**\n"
        "- Multi-asset market data (equities, crypto, ETFs, commodities, forex, indices)\n"
        "- 17 technical indicators and rule-based signal consensus\n"
        "- Deep-learning forecasting: LSTM, GRU, TCN, Transformer, CNN-LSTM\n"
        "- Reinforcement learning: DQN / Double DQN / Dueling DQN / PPO / A2C / SAC / TD3\n"
        "- Financial NLP: news collection, FinBERT-compatible sentiment, impact scoring\n"
        "- Risk engine: anomalies, volatility spikes, bubble and crash-risk scoring\n"
        "- Explainable AI: SHAP, LIME, permutation importance, counterfactuals\n"
        "- Paper-trading portfolios with mean-variance / risk-parity optimisation\n\n"
        "_Educational and research software. Not investment advice._"
    ),
    lifespan=lifespan,
    # The API reference is intentionally NOT linked from the dashboard (cleaner,
    # user-facing UI) but remains reachable by URL at /docs for developers.
    # Set EXPOSE_API_DOCS=false to disable the routes completely.
    docs_url="/docs" if settings.EXPOSE_API_DOCS else None,
    redoc_url="/redoc" if settings.EXPOSE_API_DOCS else None,
    openapi_url="/openapi.json" if settings.EXPOSE_API_DOCS else None,
    default_response_class=SafeJSONResponse,
)

# Order matters: middleware added later runs earlier. The guard is registered
# after CORS so it runs *before* it, refusing an unauthenticated request without
# doing any of the work behind it.
app.add_middleware(AuthGuardMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    started = time.perf_counter()
    response = await call_next(request)
    elapsed = (time.perf_counter() - started) * 1000
    response.headers["X-Process-Time-ms"] = f"{elapsed:.1f}"
    if elapsed > 3000:
        logger.warning("slow request %s %s -> %.0f ms", request.method, request.url.path, elapsed)
    return response


@app.exception_handler(KorisQuantError)
async def korisquant_exception_handler(request: Request, exc: KorisQuantError):
    logger.warning("%s on %s: %s", exc.code, request.url.path, exc.message)
    return SafeJSONResponse(status_code=exc.status_code, content=exc.to_dict())


# Field labels for validation messages. Pydantic reports the raw attribute
# name, which is an implementation detail: "full_name" is not what the form
# calls it, and neither is "identifier".
_FIELD_LABELS = {
    "username": "Username",
    "email": "Email",
    "password": "Password",
    "full_name": "Full name",
    "identifier": "Username or email",
    "token": "Link",
    "symbol": "Symbol",
    "notional": "Amount",
    "quantity": "Quantity",
    "message": "Message",
}


def _humanise_validation(errors: list[dict]) -> list[str]:
    """Turn Pydantic's error dicts into sentences a person can act on."""
    out: list[str] = []
    for error in errors:
        # loc is ("body", "username") - the last element is the field itself.
        location = [str(part) for part in error.get("loc", []) if part != "body"]
        field = location[-1] if location else ""
        label = _FIELD_LABELS.get(field, field.replace("_", " ").capitalize() or "Value")
        kind = error.get("type", "")
        context = error.get("ctx") or {}
        message = str(error.get("msg", "is invalid"))

        if kind == "string_too_short":
            minimum = context.get("min_length")
            if minimum == 1:
                # "must be at least 1 characters long" is both ungrammatical
                # and a clumsy way to say the field is empty.
                out.append(f"{label} is required.")
            elif minimum:
                plural = "character" if minimum == 1 else "characters"
                out.append(f"{label} must be at least {minimum} {plural} long.")
            else:
                out.append(f"{label} is too short.")
        elif kind == "string_too_long":
            maximum = context.get("max_length")
            plural = "character" if maximum == 1 else "characters"
            out.append(f"{label} must be at most {maximum} {plural} long."
                       if maximum else f"{label} is too long.")
        elif kind == "missing":
            out.append(f"{label} is required.")
        elif kind == "string_pattern_mismatch":
            out.append(f"{label} contains characters that are not allowed.")
        elif kind in ("greater_than", "greater_than_equal"):
            out.append(f"{label} must be greater than {context.get('gt', context.get('ge'))}.")
        elif kind in ("less_than", "less_than_equal"):
            out.append(f"{label} must be less than {context.get('lt', context.get('le'))}.")
        else:
            # Pydantic prefixes custom validator messages with "Value error, ";
            # that prefix means nothing to the person reading it.
            cleaned = message.removeprefix("Value error, ").strip()
            cleaned = cleaned[0].upper() + cleaned[1:] if cleaned else "is invalid"
            if not cleaned.endswith((".", "!", "?")):
                cleaned += "."
            out.append(cleaned if cleaned.lower().startswith(label.lower())
                       else f"{label}: {cleaned}")
    return out


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Return readable text instead of Pydantic's internal error objects.

    Without this, FastAPI's default handler serialises a list of dicts under
    "detail". The frontend has nothing sensible to do with that, so it
    JSON.stringify()s the lot and the user sees
    `[{"type":"string_too_short","loc":["body","username"]...}]` in a form.
    Every form on the platform shared this failure, not just registration.
    """
    problems = _humanise_validation(exc.errors())
    logger.info("validation failed on %s: %s", request.url.path, problems)
    return SafeJSONResponse(
        status_code=422,
        content={
            "error": "invalid_request",
            "message": " ".join(problems) or "Some of the details you entered are invalid.",
            "problems": problems,
        },
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled error on %s", request.url.path)
    return SafeJSONResponse(
        status_code=500,
        content={"error": "internal_error", "message": str(exc)[:500],
                 "path": str(request.url.path)},
    )


app.include_router(api_router, prefix=settings.API_V1_PREFIX)


@app.get("/health", response_model=HealthResponse, tags=["System"], summary="Service health")
async def health():
    try:
        import torch  # noqa: F401
        torch_available = True
    except Exception:
        torch_available = False
    from app.services.rl.agents.policy_gradient import SB3_AVAILABLE

    data_health = market_data_service.health()
    return HealthResponse(
        status="healthy",
        version=settings.APP_VERSION,
        environment=settings.ENVIRONMENT,
        data_mode=settings.DATA_MODE,
        live_providers=data_health["live_providers"],
        sb3_available=SB3_AVAILABLE,
        torch_available=torch_available,
        universe_size=data_health["universe_size"],
        timestamp=datetime.now(UTC).isoformat(),
    )


@app.get("/api", tags=["System"], summary="API index")
async def api_index():
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "docs": "/docs" if settings.EXPOSE_API_DOCS else "disabled",
        "health": "/health",
        "modules": {
            "market": f"{settings.API_V1_PREFIX}/market",
            "forecast": f"{settings.API_V1_PREFIX}/forecast",
            "reinforcement_learning": f"{settings.API_V1_PREFIX}/rl",
            "portfolio": f"{settings.API_V1_PREFIX}/portfolio",
            "recommendations": f"{settings.API_V1_PREFIX}/signals",
            "news_nlp": f"{settings.API_V1_PREFIX}/news",
            "risk": f"{settings.API_V1_PREFIX}/risk",
            "explainable_ai": f"{settings.API_V1_PREFIX}/xai",
            "alerts": f"{settings.API_V1_PREFIX}/alerts",
            "dashboard": f"{settings.API_V1_PREFIX}/dashboard",
        },
    }


# --------------------------------------------------------------- frontend
if FRONTEND_DIR.exists():
    # RevalidatingStaticFiles adds Cache-Control so browsers never mix a fresh
    # page script with a stale api.js (see app/utils/static_cache.py).
    app.mount("/assets", RevalidatingStaticFiles(directory=FRONTEND_DIR / "assets"), name="assets")

    # HTML is the map pointing at content-hashed asset URLs, so it must never be
    # cached: a stale map would keep requesting stale assets forever.
    _HTML_HEADERS = {
        "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
        "Pragma": "no-cache",
        "Expires": "0",
    }

    def _html_response(path: Path) -> HTMLResponse:
        return HTMLResponse(render_versioned_html(path, FRONTEND_DIR), headers=_HTML_HEADERS)

    @app.get("/", include_in_schema=False)
    async def serve_landing():
        """The marketing page is the front door; the dashboard lives at
        /dashboard. Anyone who bookmarked / for the dashboard is redirected
        there by landing.html when they are already signed in, so the change
        does not strand existing users."""
        landing = FRONTEND_DIR / "landing.html"
        if landing.exists():
            return _html_response(landing)
        index = FRONTEND_DIR / "index.html"
        if index.exists():
            return _html_response(index)
        return JSONResponse({"message": "Frontend not built", "api": "/api", "docs": "/docs"})

    @app.get("/auth.html", include_in_schema=False)
    async def serve_auth():
        page = FRONTEND_DIR / "auth.html"
        if page.exists():
            return _html_response(page)
        return JSONResponse({"error": "not_found"}, status_code=404)

    @app.get("/dashboard", include_in_schema=False)
    async def serve_dashboard():
        index = FRONTEND_DIR / "index.html"
        if index.exists():
            return _html_response(index)
        return JSONResponse({"error": "not_found"}, status_code=404)

    @app.get("/{page}.html", include_in_schema=False)
    async def serve_page(page: str):
        candidate = FRONTEND_DIR / f"{Path(page).name}.html"
        if candidate.exists():
            return _html_response(candidate)
        return JSONResponse({"error": "not_found", "page": page}, status_code=404)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
