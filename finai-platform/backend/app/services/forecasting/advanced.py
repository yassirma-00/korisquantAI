"""Advanced forecasting: ensembles, GARCH volatility and regime detection.

These are the components that measurably improve results on financial data,
as opposed to simply adding more layers to a neural network:

* **Ensembles** — averaging decorrelated models is the single most reliable
  accuracy gain in noisy regimes. Includes inverse-error and directional-skill
  weighting, not just a naive mean.
* **GARCH volatility** — returns are barely predictable; *volatility* genuinely
  is (it clusters). EGARCH/GJR also capture the leverage effect: bad news
  raises volatility more than equivalent good news.
* **Regime detection** — a model tuned on a calm bull market is dangerous in a
  crisis. Gaussian-mixture and jump-based classification of the current regime
  lets the platform state which environment it is operating in.
* **Directional calibration** — converts a raw score into a probability that is
  actually reliable, measured by Brier score and reliability curves.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from app.core.logging import get_logger

logger = get_logger(__name__)

try:  # pragma: no cover - optional
    from arch import arch_model
    ARCH_AVAILABLE = True
except Exception:  # pragma: no cover
    ARCH_AVAILABLE = False


# ============================================================== ensembling
@dataclass
class EnsembleResult:
    prediction: float
    weights: dict
    member_predictions: dict
    dispersion: float
    agreement: float
    method: str

    def to_dict(self) -> dict:
        return {
            "prediction": round(self.prediction, 6),
            "weights": {k: round(v, 4) for k, v in self.weights.items()},
            "member_predictions": {k: round(v, 6) for k, v in self.member_predictions.items()},
            "dispersion": round(self.dispersion, 6),
            "agreement": round(self.agreement, 4),
            "method": self.method,
        }


class ForecastEnsemble:
    """Combine several model forecasts into one better-calibrated estimate.

    ``dispersion`` (spread across members) is a genuine uncertainty signal:
    when models disagree sharply, confidence should fall regardless of how
    strong the mean signal looks.
    """

    METHODS = ("mean", "median", "inverse_error", "directional", "trimmed")

    def __init__(self, method: str = "inverse_error") -> None:
        if method not in self.METHODS:
            raise ValueError(f"Unknown ensemble method '{method}'. Use one of {self.METHODS}")
        self.method = method

    def combine(self, predictions: dict[str, float],
                metrics: dict[str, dict] | None = None) -> EnsembleResult:
        names = [k for k, v in predictions.items() if v is not None and np.isfinite(v)]
        if not names:
            raise ValueError("No finite member predictions to combine")
        values = np.array([predictions[k] for k in names], dtype=float)
        metrics = metrics or {}

        if self.method == "mean" or len(names) == 1:
            weights = np.ones(len(names)) / len(names)
        elif self.method == "median":
            weights = np.zeros(len(names))
            weights[int(np.argsort(values)[len(values) // 2])] = 1.0
        elif self.method == "trimmed" and len(names) >= 3:
            order = np.argsort(values)
            weights = np.zeros(len(names))
            keep = order[1:-1]                      # drop the extremes
            weights[keep] = 1.0 / len(keep)
        elif self.method == "inverse_error":
            # Weight ∝ 1/RMSE: models with lower out-of-sample error count more
            rmse = np.array([max(metrics.get(k, {}).get("rmse", 1.0) or 1.0, 1e-9) for k in names])
            weights = (1.0 / rmse) / np.sum(1.0 / rmse)
        elif self.method == "directional":
            # Weight by skill *above chance*; a 50%-accurate model gets ~nothing
            da = np.array([metrics.get(k, {}).get("directional_accuracy", 50.0) or 50.0 for k in names])
            edge = np.clip(da - 50.0, 0.0, None)
            weights = np.ones(len(names)) / len(names) if edge.sum() < 1e-9 else edge / edge.sum()
        else:
            weights = np.ones(len(names)) / len(names)

        prediction = float(np.dot(weights, values))
        dispersion = float(np.std(values))
        signs = np.sign(values)
        nonzero = signs[signs != 0]
        agreement = (float(max((nonzero == 1).sum(), (nonzero == -1).sum()) / len(nonzero))
                     if len(nonzero) else 0.5)

        return EnsembleResult(
            prediction=prediction,
            weights=dict(zip(names, weights.tolist(), strict=False)),
            member_predictions={k: float(predictions[k]) for k in names},
            dispersion=dispersion, agreement=agreement, method=self.method,
        )


# ========================================================= GARCH volatility
class VolatilityForecaster:
    """GARCH-family volatility forecasting.

    Volatility is the part of a return series that is genuinely forecastable:
    it clusters, mean-reverts, and reacts asymmetrically to bad news. A good
    volatility forecast improves position sizing and VaR far more than a
    marginally better return forecast.
    """

    MODELS = ("garch", "egarch", "gjr")

    def __init__(self, model: str = "gjr", dist: str = "t") -> None:
        self.model = model.lower()
        self.dist = dist          # Student-t captures fat tails; normal does not
        self.fitted = None
        self.scale = 100.0        # arch works best on percentage returns

    def fit(self, returns: pd.Series) -> VolatilityForecaster:
        if not ARCH_AVAILABLE:
            raise RuntimeError("the 'arch' package is required for GARCH models")
        r = pd.Series(returns).dropna() * self.scale
        if len(r) < 100:
            raise ValueError(f"GARCH needs >= 100 observations, got {len(r)}")

        spec = {"garch": {"vol": "GARCH", "p": 1, "q": 1},
                "egarch": {"vol": "EGARCH", "p": 1, "q": 1},
                "gjr": {"vol": "GARCH", "p": 1, "o": 1, "q": 1}}[self.model]
        am = arch_model(r, mean="Constant", dist=self.dist, **spec)
        self.fitted = am.fit(disp="off", show_warning=False)
        return self

    def forecast(self, horizon: int = 5) -> dict:
        if self.fitted is None:
            raise RuntimeError("fit() must be called first")
        # EGARCH and GJR have no closed-form multi-step variance: the recursion is
        # non-linear in the shock, so `arch` refuses analytic forecasts beyond h=1.
        # Simulation is the correct route (and is what the literature prescribes).
        needs_simulation = horizon > 1 and self.model in ("egarch", "gjr")
        if needs_simulation:
            f = self.fitted.forecast(horizon=horizon, reindex=False,
                                     method="simulation", simulations=1000)
        else:
            f = self.fitted.forecast(horizon=horizon, reindex=False)
        var_path = np.asarray(f.variance.values[-1], dtype=float)
        daily_vol = np.sqrt(var_path) / self.scale
        cumulative = float(np.sqrt(var_path.sum()) / self.scale)

        params = self.fitted.params.to_dict()
        alpha = float(params.get("alpha[1]", 0.0))
        beta = float(params.get("beta[1]", 0.0))
        gamma = float(params.get("gamma[1]", 0.0))
        persistence = alpha + beta + gamma / 2

        return {
            "model": self.model.upper(),
            "distribution": self.dist,
            "horizon": horizon,
            "daily_volatility": [round(float(v), 6) for v in daily_vol],
            "annualised_volatility": round(float(daily_vol[-1] * np.sqrt(252)), 4),
            "cumulative_volatility": round(cumulative, 6),
            "persistence": round(persistence, 4),
            "half_life_days": (round(float(np.log(0.5) / np.log(persistence)), 1)
                               if 0 < persistence < 1 else None),
            "leverage_effect": round(gamma, 4) if gamma else None,
            "interpretation": self._interpret(persistence, gamma),
            "log_likelihood": round(float(self.fitted.loglikelihood), 2),
            "aic": round(float(self.fitted.aic), 2),
            "forecast_method": "simulation" if needs_simulation else "analytic",
        }

    @staticmethod
    def _interpret(persistence: float, gamma: float) -> str:
        parts = []
        if persistence > 0.98:
            parts.append("Volatility shocks are extremely persistent - elevated risk will decay slowly.")
        elif persistence > 0.9:
            parts.append("Volatility is persistent; expect the current regime to continue for weeks.")
        else:
            parts.append("Volatility mean-reverts quickly.")
        if gamma and gamma > 0.02:
            parts.append("A clear leverage effect: negative returns raise volatility more than positive ones.")
        return " ".join(parts)

    def diagnostics(self) -> dict:
        """Did the model actually fit? Standardised residuals should be white noise."""
        if self.fitted is None:
            raise RuntimeError("fit() must be called first")
        from statsmodels.stats.diagnostic import acorr_ljungbox

        std_resid = pd.Series(self.fitted.std_resid).dropna()
        lb = acorr_ljungbox(std_resid, lags=[10], return_df=True)
        lb_sq = acorr_ljungbox(std_resid ** 2, lags=[10], return_df=True)
        p_resid = float(lb["lb_pvalue"].iloc[0])
        p_sq = float(lb_sq["lb_pvalue"].iloc[0])
        return {
            "ljung_box_p_residuals": round(p_resid, 4),
            "ljung_box_p_squared": round(p_sq, 4),
            "residual_autocorrelation": bool(p_resid < 0.05),
            "remaining_arch_effects": bool(p_sq < 0.05),
            "well_specified": bool(p_resid >= 0.05 and p_sq >= 0.05),
            "excess_kurtosis": round(float(std_resid.kurtosis()), 3),
            "note": ("Standardised residuals show no remaining structure - the model is adequate."
                     if p_resid >= 0.05 and p_sq >= 0.05 else
                     "Residual structure remains; treat the volatility forecast with caution."),
        }


# ========================================================= regime detection
class RegimeDetector:
    """Identify the market regime the model is currently operating in.

    Uses a Gaussian mixture over (return, volatility) plus trend and drawdown
    context. Knowing the regime is often more actionable than a point forecast:
    it tells you whether the model's training distribution still applies.
    """

    REGIMES = ("crisis", "bear", "sideways", "bull", "euphoria")

    def detect(self, df: pd.DataFrame, window: int = 63) -> dict:
        close = df["close"]
        returns = close.pct_change().dropna()
        if len(returns) < window * 2:
            return {"regime": "unknown", "reason": "insufficient history"}

        recent = returns.tail(window)
        vol_recent = float(recent.std() * np.sqrt(252))
        vol_long = float(returns.std() * np.sqrt(252)) or 1e-9
        vol_ratio = vol_recent / vol_long
        trend = float(close.iloc[-1] / close.iloc[-window] - 1)
        equity = (1 + returns).cumprod()
        drawdown = float(equity.iloc[-1] / equity.cummax().iloc[-1] - 1)
        skew = float(recent.skew())
        down_share = float((recent < 0).mean())

        # Rule layer: transparent and auditable, unlike a black-box classifier
        if vol_ratio > 1.8 and drawdown < -0.15:
            regime, confidence = "crisis", min(0.6 + vol_ratio / 10, 0.95)
        elif trend < -0.10 and down_share > 0.5:
            regime, confidence = "bear", 0.7
        elif trend > 0.20 and vol_ratio < 1.2:
            regime, confidence = "euphoria", 0.65
        elif trend > 0.05:
            regime, confidence = "bull", 0.7
        else:
            regime, confidence = "sideways", 0.6

        # Statistical layer: unsupervised confirmation
        gmm_label = None
        try:
            from sklearn.mixture import GaussianMixture

            feats = pd.DataFrame({
                "ret": returns.rolling(21).mean(),
                "vol": returns.rolling(21).std(),
            }).dropna()
            if len(feats) > 120:
                gmm = GaussianMixture(n_components=3, random_state=42, n_init=3)
                labels = gmm.fit_predict(feats.values)
                order = np.argsort(gmm.means_[:, 1])       # sort by volatility
                mapping = {int(order[0]): "low_volatility",
                           int(order[1]): "medium_volatility",
                           int(order[2]): "high_volatility"}
                gmm_label = mapping[int(labels[-1])]
        except Exception as exc:  # pragma: no cover
            logger.debug("GMM regime detection failed: %s", exc)

        return {
            "regime": regime,
            "confidence": round(float(confidence), 3),
            "statistical_regime": gmm_label,
            "trend_63d": round(trend, 4),
            "volatility_ratio": round(vol_ratio, 3),
            "annualised_volatility": round(vol_recent, 4),
            "current_drawdown": round(drawdown, 4),
            "return_skew": round(skew, 3),
            "down_day_share": round(down_share, 3),
            "model_reliability": self._reliability(regime),
            "guidance": self._guidance(regime),
        }

    @staticmethod
    def _reliability(regime: str) -> str:
        return {
            "crisis": "low - models trained on normal conditions extrapolate badly here",
            "bear": "moderate - trend signals work, mean-reversion signals often fail",
            "sideways": "moderate - mean-reversion favoured, trend signals whipsaw",
            "bull": "good - most historical patterns remain applicable",
            "euphoria": "low - late-stage momentum reverses violently",
        }.get(regime, "unknown")

    @staticmethod
    def _guidance(regime: str) -> str:
        return {
            "crisis": "Reduce gross exposure, widen stops, prefer cash. Correlations converge to 1.",
            "bear": "Favour defensive assets; rallies are frequently short-lived.",
            "sideways": "Range strategies over trend-following; keep position sizes modest.",
            "bull": "Trend-following is favoured; maintain trailing stops.",
            "euphoria": "Take profits progressively; tail risk is materially underpriced.",
        }.get(regime, "")


# =================================================== probability calibration
class DirectionalCalibrator:
    """Turn a raw model score into a *trustworthy* probability.

    A model that says "80% confident" should be right about 80% of the time.
    Isotonic regression enforces that, and the Brier score measures whether it
    worked.
    """

    def __init__(self) -> None:
        self.calibrator = None
        self.fitted = False

    def fit(self, scores: np.ndarray, outcomes: np.ndarray) -> DirectionalCalibrator:
        from sklearn.isotonic import IsotonicRegression

        scores = np.asarray(scores, float).ravel()
        outcomes = np.asarray(outcomes, float).ravel()
        mask = np.isfinite(scores) & np.isfinite(outcomes)
        scores, outcomes = scores[mask], outcomes[mask]
        if len(scores) < 40:
            return self
        self.calibrator = IsotonicRegression(y_min=0.02, y_max=0.98, out_of_bounds="clip")
        self.calibrator.fit(scores, outcomes)
        self.fitted = True
        return self

    def predict_proba(self, score: float) -> float:
        if not self.fitted:
            # Logistic fallback keeps the output in a sane range
            return float(1 / (1 + np.exp(-np.clip(score * 8, -10, 10))))
        return float(np.clip(self.calibrator.predict([score])[0], 0.02, 0.98))

    @staticmethod
    def evaluate(probabilities: np.ndarray, outcomes: np.ndarray, n_bins: int = 5) -> dict:
        p = np.asarray(probabilities, float).ravel()
        y = np.asarray(outcomes, float).ravel()
        mask = np.isfinite(p) & np.isfinite(y)
        p, y = p[mask], y[mask]
        if len(p) < 10:
            return {"error": "insufficient data"}

        brier = float(np.mean((p - y) ** 2))
        base = float(y.mean())
        brier_base = float(np.mean((base - y) ** 2))
        skill = 1 - brier / brier_base if brier_base > 1e-12 else 0.0

        bins = np.linspace(0, 1, n_bins + 1)
        reliability, ece = [], 0.0
        for i in range(n_bins):
            sel = (p >= bins[i]) & (p < bins[i + 1] if i < n_bins - 1 else p <= bins[i + 1])
            if sel.sum() >= 3:
                conf, acc = float(p[sel].mean()), float(y[sel].mean())
                reliability.append({"bin": f"{bins[i]:.1f}-{bins[i+1]:.1f}",
                                    "predicted": round(conf, 3),
                                    "observed": round(acc, 3), "n": int(sel.sum())})
                ece += (sel.sum() / len(p)) * abs(conf - acc)

        return {
            "brier_score": round(brier, 4),
            "brier_skill_score": round(float(skill), 4),
            "expected_calibration_error": round(float(ece), 4),
            "base_rate": round(base, 4),
            "well_calibrated": bool(ece < 0.1),
            "reliability_curve": reliability,
            "interpretation": ("Probabilities are trustworthy." if ece < 0.1 else
                               "Probabilities are poorly calibrated - treat confidence figures as ordinal only."),
        }


forecast_ensemble = ForecastEnsemble()
regime_detector = RegimeDetector()
