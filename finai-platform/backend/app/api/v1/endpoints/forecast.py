"""Deep-learning forecasting endpoints."""

from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, Query

from app.core.config import settings
from app.core.logging import get_logger
from app.schemas.common import CompareModelsRequest, PredictRequest, TrainForecastRequest
from app.services.data.market_data import market_data_service
from app.services.forecasting.models import MODEL_REGISTRY
from app.services.forecasting.trainer import TrainConfig, forecast_trainer

logger = get_logger(__name__)
router = APIRouter(prefix="/forecast", tags=["AI Forecasting"])

TRAINING_JOBS: dict[str, dict] = {}


@router.get("/models", summary="Available forecasting architectures")
async def list_models():
    return {
        "models": [
            {"key": "lstm", "name": "Long Short-Term Memory",
             "description": "Recurrent network capturing long-range temporal dependencies."},
            {"key": "gru", "name": "Gated Recurrent Unit",
             "description": "Lighter recurrent alternative to LSTM, faster to train."},
            {"key": "tcn", "name": "Temporal Convolutional Network",
             "description": "Dilated causal convolutions with a wide receptive field."},
            {"key": "transformer", "name": "Transformer Encoder",
             "description": "Self-attention over the sequence; captures global context."},
            {"key": "cnn_lstm", "name": "CNN-LSTM Hybrid",
             "description": "Convolutional feature extraction followed by recurrent modelling."},
        ],
        "registry": sorted(MODEL_REGISTRY.keys()),
        "default_lookback": settings.DEFAULT_LOOKBACK,
        "default_horizon": settings.DEFAULT_HORIZON,
    }


@router.post("/train", summary="Train a forecasting model")
async def train_model(request: TrainForecastRequest):
    series = market_data_service.get_history(request.symbol, period=request.period)
    config = TrainConfig(
        model=request.model, lookback=request.lookback, horizon=request.horizon,
        epochs=request.epochs, batch_size=request.batch_size,
        learning_rate=request.learning_rate, target=request.target,
    )
    result = forecast_trainer.train(request.symbol, series.df, config)
    return {
        "symbol": result.symbol, "model": request.model,
        "data_source": series.source, "bars_used": len(series.df),
        "n_parameters": result.n_parameters, "metrics": result.metrics,
        "history": result.history, "trained_at": result.trained_at,
        "checkpoint": result.checkpoint,
    }


@router.post("/train/async", summary="Train in the background (non-blocking)")
async def train_async(request: TrainForecastRequest, background_tasks: BackgroundTasks):
    job_id = f"{request.symbol.upper()}_{request.model}_h{request.horizon}"
    TRAINING_JOBS[job_id] = {"status": "queued", "request": request.model_dump()}

    def _run() -> None:
        TRAINING_JOBS[job_id]["status"] = "running"
        try:
            series = market_data_service.get_history(request.symbol, period=request.period)
            config = TrainConfig(
                model=request.model, lookback=request.lookback, horizon=request.horizon,
                epochs=request.epochs, batch_size=request.batch_size,
                learning_rate=request.learning_rate, target=request.target)
            result = forecast_trainer.train(request.symbol, series.df, config)
            TRAINING_JOBS[job_id] = {"status": "completed", "metrics": result.metrics,
                                     "trained_at": result.trained_at}
        except Exception as exc:
            logger.exception("async training failed")
            TRAINING_JOBS[job_id] = {"status": "failed", "error": str(exc)[:400]}

    background_tasks.add_task(_run)
    return {"job_id": job_id, "status": "queued",
            "poll": f"{settings.API_V1_PREFIX}/forecast/jobs/{job_id}"}


@router.get("/jobs/{job_id}", summary="Poll a background training job")
async def job_status(job_id: str):
    return TRAINING_JOBS.get(job_id, {"status": "unknown", "job_id": job_id})


@router.post("/predict", summary="Forecast future prices")
async def predict(request: PredictRequest):
    series = market_data_service.get_history(request.symbol, period=request.period)
    prediction = forecast_trainer.predict(
        request.symbol, series.df, model_name=request.model, horizon=request.horizon)
    prediction["data_source"] = series.source
    return prediction


@router.get("/predict/{symbol}", summary="Forecast (GET convenience)")
async def predict_get(
    symbol: str,
    model: str = Query("lstm"),
    horizon: int = Query(5, ge=1, le=60),
    period: str = Query("2y"),
    auto_train: bool = Query(True, description="Train automatically if no checkpoint exists"),
):
    # `period` is the window the caller wants to *display*. Feature engineering
    # needs far more than that: at 1mo (22 bars) the 21-day rolling features
    # leave zero usable rows, and the model then rejects the frame with
    # "Feature mismatch, missing: [...]" — which looked like a broken
    # checkpoint rather than too short a window.
    from app.utils.periods import compute_period

    fit_period = compute_period(period, "forecast")
    series = market_data_service.get_history(symbol, period=fit_period)
    if auto_train and not forecast_trainer.is_trained(symbol, model, horizon):
        logger.info("auto-training %s/%s", symbol, model)
        forecast_trainer.train(symbol, series.df,
                               TrainConfig(model=model, horizon=horizon, epochs=15))
    prediction = forecast_trainer.predict(symbol, series.df, model_name=model, horizon=horizon)
    prediction["data_source"] = series.source
    prediction["display_period"] = period
    prediction["computed_over"] = fit_period
    return prediction


@router.post("/compare", summary="Train and compare several architectures")
async def compare_models(request: CompareModelsRequest):
    series = market_data_service.get_history(request.symbol, period=request.period)
    results = []
    for model_name in request.models:
        try:
            config = TrainConfig(model=model_name, horizon=request.horizon, epochs=request.epochs)
            result = forecast_trainer.train(request.symbol, series.df, config)
            test = result.metrics.get("test", {})
            results.append({
                "model": model_name, "n_parameters": result.n_parameters,
                "rmse": test.get("rmse"), "mae": test.get("mae"), "r2": test.get("r2"),
                "directional_accuracy": test.get("directional_accuracy"),
                "train_seconds": result.metrics.get("train_seconds"),
                "epochs_run": result.metrics.get("epochs_run"),
            })
        except Exception as exc:
            results.append({"model": model_name, "error": str(exc)[:300]})

    ranked = sorted([r for r in results if r.get("directional_accuracy") is not None],
                    key=lambda r: (-r["directional_accuracy"], r["rmse"]))
    return {
        "symbol": request.symbol.upper(), "horizon": request.horizon,
        "data_source": series.source, "bars": len(series.df),
        "results": results,
        "best_model": ranked[0]["model"] if ranked else None,
        "ranking": [r["model"] for r in ranked],
    }


@router.post("/backtest", summary="Walk-forward validation")
async def walk_forward(
    symbol: str = Query(...),
    model: str = Query("lstm"),
    period: str = Query("5y"),
    horizon: int = Query(5, ge=1, le=30),
    folds: int = Query(3, ge=2, le=6),
):
    series = market_data_service.get_history(symbol, period=period)
    config = TrainConfig(model=model, horizon=horizon, epochs=12)
    return forecast_trainer.walk_forward(symbol, series.df, config, folds=folds)


@router.get("/trained", summary="List trained forecasting checkpoints")
async def list_trained():
    import json

    models = []
    for meta in sorted(settings.MODEL_DIR.glob("forecast_*.json")):
        try:
            payload = json.loads(meta.read_text())
            models.append({
                "symbol": payload.get("symbol"),
                "model": payload.get("config", {}).get("model"),
                "horizon": payload.get("config", {}).get("horizon"),
                "trained_at": payload.get("trained_at"),
                "test_metrics": payload.get("metrics", {}).get("test", {}),
            })
        except Exception:
            continue
    return {"count": len(models), "models": models}


@router.get("/training-history/{symbol}", summary="Loss curves from the last training run")
async def training_history(symbol: str, model: str = Query("lstm"),
                           horizon: int = Query(5, ge=1, le=60)):
    """The per-epoch curves recorded when this model was trained.

    They were already written to a sidecar JSON beside every checkpoint but
    never served, so the Training Curves panel could only ever show something
    in the same browser session that ran the training. Reload the page and it
    went blank with no explanation — which reads as broken, not as empty.
    """
    import json

    path = settings.MODEL_DIR / f"forecast_{symbol.upper()}_{model}_h{horizon}.json"
    if not path.exists():
        # Not an error: an untrained model is a normal state. The UI needs to
        # tell the two apart, so say which it is rather than returning 404.
        return {
            "symbol": symbol.upper(), "model": model, "horizon": horizon,
            "trained": False, "history": None,
            "message": ("No training history available. Train the model to "
                        "generate training curves."),
        }

    payload = json.loads(path.read_text())
    history = payload.get("history") or {}
    config = payload.get("config", {})
    return {
        "symbol": payload.get("symbol", symbol.upper()),
        "model": config.get("model", model),
        "horizon": config.get("horizon", horizon),
        "trained": True,
        "trained_at": payload.get("trained_at"),
        "history": {
            "train_loss": history.get("train_loss", []),
            "val_loss": history.get("val_loss", []),
            "lr": history.get("lr", []),
        },
        "epochs_run": len(history.get("train_loss", [])),
        "config": {
            "lookback": config.get("lookback"),
            "epochs_requested": config.get("epochs"),
            "bars_used": payload.get("bars_used"),
        },
        "metrics": payload.get("metrics", {}),
    }
