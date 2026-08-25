"""Hyperparameter management: view, edit, duplicate, import, export, switch."""

from __future__ import annotations

from fastapi import APIRouter, Body, Query
from fastapi.responses import PlainTextResponse

from app.core.logging import get_logger
from app.services.rl.hyperparams import SECTIONS, hyperparameters

logger = get_logger(__name__)
router = APIRouter(prefix="/hyperparams", tags=["Hyperparameters"])


@router.get("/catalogue", summary="Algorithms, profiles and the schema")
async def catalogue():
    """Everything the management page needs to render itself."""
    algos = []
    for key in hyperparameters.algorithms():
        meta = (hyperparameters.algorithm_config(key).get("meta") or {})
        algos.append({
            "key": key,
            "family": meta.get("family"),
            "action_space": meta.get("action_space"),
            "backend": meta.get("backend", "native"),
            "requires": meta.get("requires"),
            "description": meta.get("description", ""),
        })
    return {
        "algorithms": algos,
        "profiles": hyperparameters.profiles(),
        "sections": list(SECTIONS),
        "defaults": hyperparameters.defaults(),
    }


@router.get("/resolve", summary="The exact parameters a training run would use")
async def resolve(
    algo: str = Query("dueling_dqn"),
    profile: str = Query("default"),
):
    """The fully materialised set, plus where each layer came from.

    This is what the page shows for editing: a diff would leave the user
    guessing what the effective value actually is.
    """
    return hyperparameters.resolve(algo, profile).to_dict()


@router.get("/profiles", summary="List hyperparameter profiles")
async def list_profiles():
    return {"profiles": hyperparameters.profiles()}


@router.get("/profiles/{name}", summary="One profile's overrides")
async def get_profile(name: str):
    return {"key": name, "config": hyperparameters.profile_config(name)}


@router.post("/profiles/{name}", summary="Create or update a profile")
async def save_profile(name: str, payload: dict = Body(...)):
    """Built-ins are rejected here on purpose — duplicate them instead."""
    return hyperparameters.save_profile(
        name,
        payload.get("config") or {},
        description=payload.get("description", ""))


@router.post("/profiles/{name}/duplicate", summary="Copy a profile")
async def duplicate_profile(name: str, payload: dict = Body(...)):
    return hyperparameters.duplicate_profile(name, payload.get("new_name", ""))


@router.delete("/profiles/{name}", summary="Delete a user profile")
async def delete_profile(name: str):
    return hyperparameters.delete_profile(name)


@router.get("/profiles/{name}/export", summary="Download a profile as YAML",
            response_class=PlainTextResponse)
async def export_profile(name: str):
    return PlainTextResponse(
        hyperparameters.export_profile(name),
        headers={"Content-Disposition": f'attachment; filename="{name}.yaml"'})


@router.post("/profiles/{name}/import", summary="Upload a profile from YAML")
async def import_profile(name: str, payload: dict = Body(...)):
    return hyperparameters.import_profile(name, payload.get("yaml", ""))


@router.get("/experiments", summary="Recorded training runs and their configs")
async def experiments(limit: int = Query(50, ge=1, le=500)):
    """Every trained agent's reproducibility record.

    Read from the checkpoint metadata rather than a separate table: the config
    that produced a model belongs with that model, and cannot drift away from
    it. Agents trained before this feature have no experiment_id and are
    reported as such instead of being hidden.
    """
    runs = []
    for meta in hyperparameters_agents():
        runs.append({
            "experiment_id": meta.get("experiment_id"),
            "symbol": meta.get("symbol") or meta.get("portfolio_key"),
            "algo": meta.get("algo"),
            "profile": meta.get("profile"),
            "seed": meta.get("seed"),
            "fingerprint": meta.get("hyperparameter_fingerprint"),
            "trained_at": meta.get("trained_at"),
            "test_performance": meta.get("test_performance", {}),
            "has_config": bool(meta.get("hyperparameters")),
            "legacy": meta.get("experiment_id") is None,
        })
    runs.sort(key=lambda r: str(r.get("trained_at") or ""), reverse=True)
    return {"count": len(runs), "experiments": runs[:limit]}


@router.get("/experiments/{experiment_id}", summary="One run's full configuration")
async def experiment(experiment_id: str):
    from app.core.exceptions import InvalidRequestError

    for meta in hyperparameters_agents():
        if meta.get("experiment_id") == experiment_id:
            return {
                "experiment_id": experiment_id,
                "symbol": meta.get("symbol") or meta.get("portfolio_key"),
                "algo": meta.get("algo"),
                "profile": meta.get("profile"),
                "seed": meta.get("seed"),
                "fingerprint": meta.get("hyperparameter_fingerprint"),
                "sources": meta.get("config_sources", []),
                "hyperparameters": meta.get("hyperparameters", {}),
                "env_config": meta.get("env_config", {}),
                "test_performance": meta.get("test_performance", {}),
                "trained_at": meta.get("trained_at"),
            }
    raise InvalidRequestError(f"No experiment '{experiment_id}'.")


def hyperparameters_agents() -> list[dict]:
    """Metadata for every trained agent on disk."""
    from app.services.rl.service import rl_service

    return rl_service.list_agents()


# ============================================== smart (AI-driven) selection
@router.get("/smart/profiles", summary="High-level training profiles")
async def smart_profiles():
    """The only choice a standard user makes."""
    from app.services.rl.autotune import USER_PROFILES

    return {
        "profiles": [{"key": key, **value} for key, value in USER_PROFILES.items()],
        "default": "ai_recommended",
        "note": ("Each maps onto a YAML profile in configs/, so the automatic "
                 "path and the advanced path share one configuration system."),
    }


@router.get("/smart/recommend", summary="Analyse the environment and configure")
async def smart_recommend(
    symbol: str = Query("AAPL"),
    algo: str = Query("dueling_dqn"),
    period: str = Query("3y"),
    objective: str = Query("balanced", description=
                           "conservative | balanced | high_performance | risk_aware"),
    symbols: str | None = Query(None, description="Comma-separated basket"),
):
    """Inspect the environment and return a ready-to-train configuration.

    The full parameter set is included for reproducibility and for Advanced
    Mode, but the standard UI renders only `summary`, `estimated_training`,
    `expected_quality` and `confidence`.
    """
    from app.services.rl.autotune import recommend

    basket = [s.strip() for s in symbols.split(",") if s.strip()] if symbols else None
    return recommend(symbol, algo, period=period, objective=objective,
                     symbols=basket)
