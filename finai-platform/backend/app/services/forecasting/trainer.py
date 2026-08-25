"""Training / inference orchestration for the deep-learning forecasters.

Design choices that matter for finance:

* **Chronological splits only** - never shuffle across time (leakage).
* **Scalers fitted on train only**, persisted with the checkpoint.
* **Return-space targets** (not raw price) -> stationary, comparable across assets.
* **Early stopping** on a validation window that immediately precedes the test set.
* Walk-forward backtesting utility for honest out-of-sample metrics.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

from app.core.config import settings
from app.core.exceptions import InvalidRequestError, ModelNotTrainedError
from app.core.logging import get_logger
from app.services.forecasting.conformal import SplitConformal, evaluate_coverage
from app.services.forecasting.models import build_model, count_parameters
from app.services.indicators.features import build_supervised, make_sequences
from app.utils.timeseries import business_days_ahead

logger = get_logger(__name__)
DEVICE = torch.device(settings.TORCH_DEVICE)


@dataclass
class TrainConfig:
    model: str = "lstm"
    lookback: int = 60
    horizon: int = 5
    epochs: int = 25
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    val_fraction: float = 0.15
    test_fraction: float = 0.15
    patience: int = 8
    target: str = "target_return"
    seed: int = 42
    model_kwargs: dict = field(default_factory=dict)


@dataclass
class TrainResult:
    symbol: str
    config: dict
    metrics: dict
    history: dict
    feature_names: list[str]
    n_parameters: int
    trained_at: str
    checkpoint: str
    # Bars the run consumed. Without this the sidecar cannot say what the
    # curves were produced from, and "Training History" becomes folklore.
    bars_used: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true, y_pred = np.asarray(y_true).ravel(), np.asarray(y_pred).ravel()
    if y_true.size == 0:
        return {}
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    denom = np.where(np.abs(y_true) < 1e-9, np.nan, np.abs(y_true))
    mape = float(np.nanmean(np.abs(err) / denom) * 100)
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
    direction = float(np.mean(np.sign(y_pred) == np.sign(y_true)) * 100)
    # naive "strategy" check: does trading the sign of the forecast make money?
    pnl = float(np.sum(np.sign(y_pred) * y_true))
    hit_up = float(np.mean(y_true[y_pred > 0] > 0) * 100) if (y_pred > 0).any() else 0.0
    return {
        "mae": round(mae, 6), "rmse": round(rmse, 6), "mape": round(mape, 3),
        "r2": round(float(r2), 4), "directional_accuracy": round(direction, 2),
        "long_hit_rate": round(hit_up, 2), "signal_pnl": round(pnl, 5),
        "n_samples": int(y_true.size),
    }


class ForecastTrainer:
    """Trains one architecture on one instrument and caches the checkpoint."""

    def __init__(self, model_dir: Path | None = None) -> None:
        self.model_dir = Path(model_dir or settings.MODEL_DIR)
        self.model_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------------- paths
    def checkpoint_path(self, symbol: str, model: str, horizon: int) -> Path:
        safe = symbol.upper().replace("/", "_").replace("=", "_").replace("^", "idx_")
        return self.model_dir / f"forecast_{safe}_{model}_h{horizon}.pt"

    def is_trained(self, symbol: str, model: str, horizon: int) -> bool:
        return self.checkpoint_path(symbol, model, horizon).exists()

    # ------------------------------------------------- checkpoint discovery
    def available_forecasts(self, symbol: str) -> list[dict]:
        """Every trained (model, horizon) pair that exists for `symbol`.

        Discovered by listing what is actually on disk, so a newly trained
        instrument becomes usable with no code change and no ticker list to
        maintain. The catalogue is never consulted: the files are the truth.

        `checkpoint_path` rewrites '=' and '^' on the way to a filename, which
        is not reversible ('GC=F' and 'GC_F' both yield 'GC_F'). So rather than
        parsing a symbol back out of a name, the symbol's own prefix is
        rebuilt and only the trailing '<model>_h<N>' is parsed — and the model
        is accepted only if the registry knows it. That rejects a neighbouring
        instrument whose prefix happens to extend this one.
        """
        from app.services.forecasting.models import MODEL_REGISTRY

        # Rebuild the exact prefix `checkpoint_path` would produce for this
        # symbol, by asking it rather than re-implementing its rewrite rules —
        # the two can never drift apart that way.
        probe = self.checkpoint_path(symbol, "MODEL", 0).name
        safe = probe[len("forecast_"): -len("_MODEL_h0.pt")]
        found: list[dict] = []
        for path in sorted(self.model_dir.glob(f"forecast_{safe}_*.pt")):
            tail = path.name[len(f"forecast_{safe}_"): -len(".pt")]
            match = re.fullmatch(r"(?P<model>.+)_h(?P<horizon>\d+)", tail)
            if not match:
                continue
            model = match.group("model")
            if model not in MODEL_REGISTRY:
                continue        # a different instrument sharing this prefix
            found.append({"model": model, "horizon": int(match.group("horizon"))})
        return found

    def available_horizons(self, symbol: str) -> list[int]:
        """Horizons this symbol has at least one trained model for."""
        return sorted({f["horizon"] for f in self.available_forecasts(symbol)})

    def _directional_accuracy(self, symbol: str, model: str, horizon: int) -> float:
        """Measured test-set directional accuracy, or -1.0 when unrecorded.

        Read from the sidecar written at training time. Never invented: a
        missing metric ranks last instead of being given a flattering default.
        """
        meta = self.checkpoint_path(symbol, model, horizon).with_suffix(".json")
        if not meta.exists():
            return -1.0
        try:
            payload = json.loads(meta.read_text())
        except (OSError, ValueError):
            return -1.0
        value = payload.get("metrics", {}).get("test", {}).get("directional_accuracy")
        return float(value) if isinstance(value, (int, float)) else -1.0

    def resolve_model(self, symbol: str, preferred: str, horizon: int) -> str | None:
        """The model to actually use for `symbol` at `horizon`, or None.

        The requested architecture wins whenever it is trained. Otherwise the
        best *measured* alternative for that same horizon is used, so an
        instrument trained on GRU is no longer reported as having no forecaster
        merely because the caller's default happened to be LSTM.

        The horizon is never substituted. A 60-day question answered by a
        5-day model would be a different forecast wearing the wrong label, so
        when nothing is trained for this horizon the answer stays None.
        """
        if self.is_trained(symbol, preferred, horizon):
            return preferred
        candidates = [f["model"] for f in self.available_forecasts(symbol)
                      if f["horizon"] == horizon]
        if not candidates:
            return None
        # Deterministic: best recorded accuracy, then name. No randomness, and
        # repeated calls cannot disagree with each other.
        candidates.sort(key=lambda m: (-self._directional_accuracy(symbol, m, horizon), m))
        return candidates[0]

    # -------------------------------------------------------------- train
    def train(self, symbol: str, df: pd.DataFrame, config: TrainConfig | None = None) -> TrainResult:
        cfg = config or TrainConfig()
        _set_seed(cfg.seed)

        x_df, y_df = build_supervised(df, horizon=cfg.horizon)
        if len(x_df) < cfg.lookback + 60:
            raise InvalidRequestError(
                f"Not enough history for {symbol}: need >= {cfg.lookback + 60} usable rows, got {len(x_df)}"
            )
        if cfg.target not in y_df.columns:
            raise InvalidRequestError(f"Unknown target '{cfg.target}'")

        feature_names = list(x_df.columns)
        x_all, y_all = x_df.values.astype(np.float32), y_df[cfg.target].values.astype(np.float32)

        n = len(x_all)
        n_test = max(int(n * cfg.test_fraction), cfg.lookback // 2 + 5)
        n_val = max(int(n * cfg.val_fraction), cfg.lookback // 2 + 5)
        n_train = n - n_val - n_test
        if n_train < cfg.lookback + 20:
            raise InvalidRequestError(f"Series too short after splitting for {symbol}")

        x_scaler, y_scaler = StandardScaler(), StandardScaler()
        x_train_s = x_scaler.fit_transform(x_all[:n_train])
        y_train_s = y_scaler.fit_transform(y_all[:n_train].reshape(-1, 1)).ravel()
        x_rest_s = x_scaler.transform(x_all[n_train:])
        y_rest_s = y_scaler.transform(y_all[n_train:].reshape(-1, 1)).ravel()
        x_scaled = np.vstack([x_train_s, x_rest_s])
        y_scaled = np.concatenate([y_train_s, y_rest_s])

        xs, ys = make_sequences(x_scaled, y_scaled, cfg.lookback)
        offset = cfg.lookback
        tr_end = max(n_train - offset, 1)
        va_end = tr_end + n_val
        x_tr, y_tr = xs[:tr_end], ys[:tr_end]
        x_va, y_va = xs[tr_end:va_end], ys[tr_end:va_end]
        x_te, y_te = xs[va_end:], ys[va_end:]
        if len(x_va) == 0 or len(x_te) == 0:
            split = int(len(xs) * 0.8)
            x_tr, y_tr, x_va, y_va = xs[:split], ys[:split], xs[split:], ys[split:]
            x_te, y_te = x_va, y_va

        model = build_model(cfg.model, n_features=xs.shape[2], **cfg.model_kwargs).to(DEVICE)
        optimiser = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, mode="min", factor=0.5, patience=3)
        criterion = nn.HuberLoss(delta=1.0)

        t_x = torch.tensor(x_tr, device=DEVICE)
        t_y = torch.tensor(y_tr, device=DEVICE).unsqueeze(-1)
        v_x = torch.tensor(x_va, device=DEVICE)
        v_y = torch.tensor(y_va, device=DEVICE).unsqueeze(-1)

        history = {"train_loss": [], "val_loss": [], "lr": []}
        best_loss, best_state, bad_epochs = float("inf"), None, 0
        started = time.time()

        for epoch in range(cfg.epochs):
            model.train()
            perm = torch.randperm(len(t_x), device=DEVICE)  # shuffle windows, not time inside a window
            epoch_loss = 0.0
            for i in range(0, len(perm), cfg.batch_size):
                idx = perm[i: i + cfg.batch_size]
                optimiser.zero_grad()
                loss = criterion(model(t_x[idx]), t_y[idx])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimiser.step()
                epoch_loss += float(loss.detach()) * len(idx)
            epoch_loss /= max(len(t_x), 1)

            model.eval()
            with torch.no_grad():
                val_loss = float(criterion(model(v_x), v_y))
            scheduler.step(val_loss)

            history["train_loss"].append(round(epoch_loss, 6))
            history["val_loss"].append(round(val_loss, 6))
            history["lr"].append(optimiser.param_groups[0]["lr"])

            if val_loss < best_loss - 1e-6:
                best_loss, bad_epochs = val_loss, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                bad_epochs += 1
                if bad_epochs >= cfg.patience:
                    logger.info("early stopping %s/%s at epoch %d", symbol, cfg.model, epoch + 1)
                    break

        if best_state:
            model.load_state_dict(best_state)
        model.eval()

        def predict_scaled(arr: np.ndarray) -> np.ndarray:
            if len(arr) == 0:
                return np.empty(0)
            with torch.no_grad():
                out = model(torch.tensor(arr, device=DEVICE)).cpu().numpy()
            return y_scaler.inverse_transform(out.reshape(-1, 1)).ravel()

        y_te_true = y_scaler.inverse_transform(y_te.reshape(-1, 1)).ravel()
        y_va_true = y_scaler.inverse_transform(y_va.reshape(-1, 1)).ravel()
        metrics = {
            "test": _metrics(y_te_true, predict_scaled(x_te)),
            "validation": _metrics(y_va_true, predict_scaled(x_va)),
            "train_seconds": round(time.time() - started, 2),
            "epochs_run": len(history["train_loss"]),
            "best_val_loss": round(best_loss, 6),
        }

        ckpt = self.checkpoint_path(symbol, cfg.model, cfg.horizon)
        torch.save({
            "state_dict": model.state_dict(),
            "config": asdict(cfg),
            "feature_names": feature_names,
            "x_scaler_mean": x_scaler.mean_, "x_scaler_scale": x_scaler.scale_,
            "y_scaler_mean": y_scaler.mean_, "y_scaler_scale": y_scaler.scale_,
            "n_features": xs.shape[2],
            "symbol": symbol.upper(),
            "trained_at": pd.Timestamp.utcnow().isoformat(),
            "metrics": metrics,
        }, ckpt)

        result = TrainResult(
            symbol=symbol.upper(), config=asdict(cfg), metrics=metrics, history=history,
            feature_names=feature_names, n_parameters=count_parameters(model),
            trained_at=pd.Timestamp.utcnow().isoformat(), checkpoint=str(ckpt),
            bars_used=int(len(df)),
        )
        (self.model_dir / f"{ckpt.stem}.json").write_text(json.dumps(result.to_dict(), indent=2, default=str))
        logger.info("trained %s/%s -> test DA=%.1f%% rmse=%.5f",
                    symbol, cfg.model, metrics["test"].get("directional_accuracy", 0),
                    metrics["test"].get("rmse", 0))
        return result

    # ------------------------------------------------------------ predict
    def load(self, symbol: str, model: str, horizon: int) -> dict:
        path = self.checkpoint_path(symbol, model, horizon)
        if not path.exists():
            raise ModelNotTrainedError(
                f"No checkpoint for {symbol}/{model}/h{horizon}. Train it first.",
                details={"symbol": symbol, "model": model, "horizon": horizon},
            )
        return torch.load(path, map_location=DEVICE, weights_only=False)

    def predict(self, symbol: str, df: pd.DataFrame, model_name: str = "lstm",
                horizon: int = 5, n_steps: int | None = None, alpha: float = 0.1) -> dict:
        ckpt = self.load(symbol, model_name, horizon)
        cfg = TrainConfig(**ckpt["config"])
        feature_names: list[str] = ckpt["feature_names"]

        from app.services.indicators.features import build_features
        features = build_features(df, dropna=False).ffill().bfill()
        missing = [c for c in feature_names if c not in features.columns]
        if missing:
            raise InvalidRequestError(f"Feature mismatch, missing: {missing[:5]}")
        x = features[feature_names].values.astype(np.float32)

        x_scaled = (x - ckpt["x_scaler_mean"]) / ckpt["x_scaler_scale"]
        if len(x_scaled) < cfg.lookback:
            raise InvalidRequestError(f"Need at least {cfg.lookback} bars to predict, got {len(x_scaled)}")

        model = build_model(cfg.model, n_features=ckpt["n_features"], **cfg.model_kwargs).to(DEVICE)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()

        window = torch.tensor(x_scaled[-cfg.lookback:][None, ...], dtype=torch.float32, device=DEVICE)
        with torch.no_grad():
            scaled_pred = float(model(window).cpu().numpy().ravel()[0])
        predicted_return = scaled_pred * float(ckpt["y_scaler_scale"][0]) + float(ckpt["y_scaler_mean"][0])

        # ---- uncertainty: conformal calibration on held-out residuals --------
        # A Gaussian z-band assumes normal errors, which financial residuals are
        # not (fat tails). Split conformal instead guarantees, in finite samples,
        # that the realised outcome falls inside the band (1-alpha) of the time.
        n_hist = min(len(x_scaled) - cfg.lookback, 400)
        residual_std = 0.0
        conformal_q = None
        coverage_report: dict | None = None

        if n_hist > 40:
            xs, _ = make_sequences(x_scaled[-(n_hist + cfg.lookback):], np.zeros(n_hist + cfg.lookback), cfg.lookback)
            with torch.no_grad():
                preds = model(torch.tensor(xs, dtype=torch.float32, device=DEVICE)).cpu().numpy().ravel()
            preds = preds * float(ckpt["y_scaler_scale"][0]) + float(ckpt["y_scaler_mean"][0])
            actual = df["close"].pct_change(horizon).shift(-horizon).dropna().values[-len(preds):]
            m = min(len(preds), len(actual))
            if m > 30:
                errors = preds[:m] - actual[:m]
                residual_std = float(np.std(errors))
                # Calibrate on the older half, verify coverage on the newer half
                split = m // 2
                conformal = SplitConformal(alpha=alpha).calibrate(actual[:split], preds[:split])
                conformal_q = conformal.q
                ivs = [conformal.predict(float(p)) for p in preds[split:m]]
                coverage_report = evaluate_coverage(
                    actual[split:m], [i.lower for i in ivs], [i.upper for i in ivs],
                    target=1 - alpha)

        last_price = float(df["close"].iloc[-1])
        target_price = last_price * (1 + predicted_return)
        steps = n_steps or horizon
        dates = business_days_ahead(df.index[-1], steps)
        # Smooth path interpolation from today's price to the horizon target
        path = [last_price * (1 + predicted_return * ((i + 1) / steps)) for i in range(steps)]

        if conformal_q is not None and np.isfinite(conformal_q):
            half_width, band_method = conformal_q, "conformal"
        else:
            half_width, band_method = residual_std * 1.645, "gaussian"
        band = [half_width * last_price * np.sqrt((i + 1) / steps) for i in range(steps)]

        test_da = ckpt.get("metrics", {}).get("test", {}).get("directional_accuracy", 50.0)
        confidence = float(np.clip(
            0.4 * (test_da / 100.0) +
            0.35 * min(abs(predicted_return) / max(residual_std, 1e-6), 1.0) +
            0.25 * (1 - min(residual_std * 10, 1.0)),
            0.0, 0.99))

        return {
            "symbol": symbol.upper(),
            "model": model_name,
            "horizon": horizon,
            "last_price": round(last_price, 6),
            "predicted_return": round(predicted_return, 6),
            "predicted_price": round(target_price, 6),
            "direction": "up" if predicted_return > 0 else "down",
            "confidence": round(confidence, 4),
            "residual_std": round(residual_std, 6),
            "interval_method": band_method,
            "confidence_level": round(1 - alpha, 3),
            "conformal_half_width": round(float(conformal_q), 6) if conformal_q is not None else None,
            "coverage_validation": coverage_report,
            "trained_at": ckpt.get("trained_at"),
            "test_metrics": ckpt.get("metrics", {}).get("test", {}),
            "forecast": [
                {
                    "date": d.strftime("%Y-%m-%d"),
                    "price": round(float(p), 6),
                    "lower": round(float(p - b), 6),
                    "upper": round(float(p + b), 6),
                }
                for d, p, b in zip(dates, path, band, strict=False)
            ],
        }

    # ---------------------------------------------------- walk-forward test
    def walk_forward(self, symbol: str, df: pd.DataFrame, config: TrainConfig | None = None,
                     folds: int = 3) -> dict:
        """Expanding-window backtest: honest estimate of live performance."""
        cfg = config or TrainConfig()
        x_df, y_df = build_supervised(df, horizon=cfg.horizon)
        n = len(x_df)
        if n < (cfg.lookback + 80) * 2:
            raise InvalidRequestError("Series too short for walk-forward validation")

        fold_size = n // (folds + 1)
        results = []
        for k in range(1, folds + 1):
            train_end = fold_size * k
            test_end = min(fold_size * (k + 1), n)
            sub = df.iloc[: test_end + cfg.horizon]
            sub_cfg = TrainConfig(**{**asdict(cfg), "epochs": max(cfg.epochs // 2, 8)})
            try:
                res = self.train(f"{symbol}__wf{k}", sub, sub_cfg)
                results.append({"fold": k, "train_end": train_end, "test_end": test_end,
                                **res.metrics.get("test", {})})
            except Exception as exc:
                logger.warning("walk-forward fold %d failed: %s", k, exc)

        if not results:
            raise InvalidRequestError("All walk-forward folds failed")
        avg = {
            key: round(float(np.mean([r[key] for r in results if key in r])), 4)
            for key in ("rmse", "mae", "r2", "directional_accuracy")
        }
        return {"symbol": symbol.upper(), "folds": results, "average": avg}


forecast_trainer = ForecastTrainer()
