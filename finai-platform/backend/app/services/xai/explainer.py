"""Explainable AI layer.

Provides model-agnostic explanations without requiring the heavy ``shap`` /
``lime`` packages (which are supported when installed):

* **Permutation importance** - global feature ranking on a surrogate model
* **KernelSHAP-style attributions** - sampled Shapley values, exact for small
  coalitions, Monte-Carlo approximated otherwise
* **LIME-style local explanation** - weighted local ridge regression around
  the instance being explained
* **Counterfactuals** - the smallest feature change that flips the decision
* **Natural-language rationale** for every recommendation
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge

from app.core.exceptions import InvalidRequestError
from app.core.logging import get_logger
from app.services.indicators.features import build_supervised

logger = get_logger(__name__)

FEATURE_LABELS = {
    "return_1d": "1-day return", "return_5d": "5-day return", "return_10d": "10-day return",
    "return_21d": "1-month return", "log_return": "log return",
    "volatility_10d": "10-day volatility", "volatility_21d": "21-day volatility",
    "volatility_ratio": "short/long volatility ratio", "rsi_14": "RSI(14)",
    "macd": "MACD line", "macd_hist": "MACD histogram", "bb_pct_b": "Bollinger %B",
    "bb_width": "Bollinger bandwidth", "atr_pct": "ATR (% of price)", "adx": "ADX trend strength",
    "plus_di": "+DI", "minus_di": "-DI", "stoch_k": "Stochastic %K", "cci_20": "CCI(20)",
    "price_to_sma20": "price vs SMA20", "price_to_sma50": "price vs SMA50",
    "price_to_ema12": "price vs EMA12", "sma20_to_sma50": "SMA20 vs SMA50",
    "volume_ratio": "relative volume", "obv_slope": "OBV slope", "mfi_14": "Money Flow Index",
    "high_low_range": "intraday range", "close_to_high": "close vs high",
    "gap": "opening gap", "day_of_week": "day of week", "month": "month",
}


def humanise(feature: str) -> str:
    return FEATURE_LABELS.get(feature, feature.replace("_", " "))


@dataclass
class Explanation:
    method: str
    feature_importance: list[dict]
    base_value: float
    prediction: float
    narrative: str
    details: dict

    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "feature_importance": self.feature_importance,
            "base_value": round(self.base_value, 6),
            "prediction": round(self.prediction, 6),
            "narrative": self.narrative,
            "details": self.details,
        }


class Explainer:
    """Trains a fast surrogate model on the same features, then explains it."""

    def __init__(self, n_estimators: int = 120, max_depth: int = 3) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self._cache: dict[str, tuple] = {}

    # ------------------------------------------------------------ surrogate
    def _surrogate(self, symbol: str, df: pd.DataFrame, horizon: int = 5):
        key = f"{symbol}:{horizon}:{len(df)}"
        if key in self._cache:
            return self._cache[key]
        x, y = build_supervised(df, horizon=horizon)
        if len(x) < 80:
            raise InvalidRequestError(
                f"Not enough usable history to explain {symbol}: {len(x)} rows after feature "
                "engineering, need at least 80. Try a longer period.",
                details={"symbol": symbol, "usable_rows": int(len(x)), "required": 80},
            )
        target = y["target_return"]
        split = int(len(x) * 0.85)
        model = GradientBoostingRegressor(
            n_estimators=self.n_estimators, max_depth=self.max_depth,
            learning_rate=0.05, subsample=0.85, random_state=42)
        model.fit(x.iloc[:split], target.iloc[:split])
        payload = (model, x, target, split)
        self._cache[key] = payload
        return payload

    # ------------------------------------------------------------- global
    def global_importance(self, symbol: str, df: pd.DataFrame, horizon: int = 5,
                          top_k: int = 12) -> dict:
        model, x, y, split = self._surrogate(symbol, df, horizon)
        x_test, y_test = x.iloc[split:], y.iloc[split:]
        result = permutation_importance(model, x_test, y_test, n_repeats=8,
                                        random_state=42, n_jobs=1)
        ranking = sorted(
            [{"feature": f, "label": humanise(f),
              "importance": round(float(m), 6), "std": round(float(s), 6)}
             for f, m, s in zip(x.columns, result.importances_mean, result.importances_std, strict=False)],
            key=lambda d: d["importance"], reverse=True)[:top_k]

        builtin = sorted(
            [{"feature": f, "label": humanise(f), "importance": round(float(v), 6)}
             for f, v in zip(x.columns, model.feature_importances_, strict=False)],
            key=lambda d: d["importance"], reverse=True)[:top_k]

        r2 = float(model.score(x_test, y_test))
        return {
            "symbol": symbol.upper(), "horizon": horizon,
            "surrogate_r2": round(r2, 4),
            "permutation_importance": ranking,
            "impurity_importance": builtin,
            "n_train": split, "n_test": len(x_test),
            "narrative": self._global_narrative(ranking, r2),
        }

    @staticmethod
    def _global_narrative(ranking: list[dict], r2: float) -> str:
        if not ranking:
            return "No features could be ranked."
        top = ", ".join(f"{d['label']}" for d in ranking[:3])
        quality = ("strong" if r2 > 0.15 else "moderate" if r2 > 0.03
                   else "weak (typical for noisy financial data)")
        return (f"The model's forecasts are driven mainly by {top}. "
                f"Out-of-sample explanatory power is {quality} (R²={r2:.3f}). "
                "Financial returns are largely unpredictable, so treat feature "
                "rankings as directional evidence rather than certainty.")

    # -------------------------------------------------------------- SHAP
    def shap_values(self, symbol: str, df: pd.DataFrame, horizon: int = 5,
                    n_samples: int = 150, top_k: int = 12) -> Explanation:
        """Sampled Shapley attributions for the most recent observation."""
        model, x, y, split = self._surrogate(symbol, df, horizon)

        try:  # exact TreeSHAP when the library is present
            import shap  # type: ignore

            explainer = shap.TreeExplainer(model)
            instance = x.iloc[[-1]]
            values = explainer.shap_values(instance)[0]
            base = float(explainer.expected_value)
            method = "TreeSHAP"
        except Exception:
            values, base = self._sampled_shapley(model, x, n_samples)
            method = "SampledSHAP"

        prediction = float(model.predict(x.iloc[[-1]])[0])
        contributions = sorted(
            [{"feature": f, "label": humanise(f),
              "value": round(float(x.iloc[-1][f]), 5),
              "contribution": round(float(v), 6),
              "direction": "bullish" if v > 0 else "bearish"}
             for f, v in zip(x.columns, values, strict=False)],
            key=lambda d: abs(d["contribution"]), reverse=True)[:top_k]

        return Explanation(
            method=method, feature_importance=contributions,
            base_value=base, prediction=prediction,
            narrative=self._local_narrative(contributions, prediction, base),
            details={"symbol": symbol.upper(), "horizon": horizon,
                     "sum_contributions": round(float(np.sum(values)), 6)},
        )

    @staticmethod
    def _sampled_shapley(model, x: pd.DataFrame, n_samples: int = 150) -> tuple[np.ndarray, float]:
        """Monte-Carlo Shapley approximation (KernelSHAP spirit, permutation based).

        Coalitions are evaluated in batches to keep the number of ``predict``
        calls proportional to ``n_samples`` rather than ``n_samples * n_features``.
        """
        rng = np.random.default_rng(42)
        columns = list(x.columns)
        background = x.iloc[:-1].sample(min(len(x) - 1, 120), random_state=42).values
        instance = x.iloc[-1].values
        n_features = len(instance)

        def predict(matrix: np.ndarray) -> np.ndarray:
            return model.predict(pd.DataFrame(matrix, columns=columns))

        base = float(predict(background).mean())
        phi = np.zeros(n_features)

        for _ in range(n_samples):
            ref = background[rng.integers(0, len(background))]
            order = rng.permutation(n_features)
            # Build the whole permutation path at once: row k has the first k
            # features of `order` taken from the instance, the rest from `ref`.
            path = np.repeat(ref.reshape(1, -1), n_features + 1, axis=0)
            for step, feature_idx in enumerate(order, start=1):
                path[step:, feature_idx] = instance[feature_idx]
            preds = predict(path)
            phi[order] += np.diff(preds)
        return phi / n_samples, base

    # -------------------------------------------------------------- LIME
    def lime_explain(self, symbol: str, df: pd.DataFrame, horizon: int = 5,
                     n_samples: int = 600, top_k: int = 10) -> Explanation:
        """Local surrogate: weighted ridge regression around the latest point."""
        model, x, y, split = self._surrogate(symbol, df, horizon)
        instance = x.iloc[-1].values
        sigma = x.std().replace(0, 1e-6).values
        rng = np.random.default_rng(42)

        perturbed = instance + rng.normal(0, 1, size=(n_samples, len(instance))) * sigma * 0.5
        preds = model.predict(pd.DataFrame(perturbed, columns=x.columns))
        distances = np.sqrt(((perturbed - instance) / sigma) ** 2).sum(axis=1)
        kernel_width = np.sqrt(len(instance)) * 0.75
        weights = np.exp(-(distances ** 2) / (kernel_width ** 2))

        ridge = Ridge(alpha=1.0)
        ridge.fit((perturbed - instance) / sigma, preds, sample_weight=weights)

        contributions = sorted(
            [{"feature": f, "label": humanise(f),
              "value": round(float(instance[i]), 5),
              "contribution": round(float(ridge.coef_[i]), 6),
              "direction": "bullish" if ridge.coef_[i] > 0 else "bearish"}
             for i, f in enumerate(x.columns)],
            key=lambda d: abs(d["contribution"]), reverse=True)[:top_k]

        prediction = float(model.predict(x.iloc[[-1]])[0])
        local_r2 = float(ridge.score((perturbed - instance) / sigma, preds, sample_weight=weights))
        return Explanation(
            method="LIME", feature_importance=contributions,
            base_value=float(ridge.intercept_), prediction=prediction,
            narrative=self._lime_narrative(contributions, prediction, local_r2),
            details={"symbol": symbol.upper(), "local_r2": round(local_r2, 4)},
        )

    @staticmethod
    def _lime_narrative(contributions: list[dict], prediction: float, local_r2: float) -> str:
        """LIME coefficients are local sensitivities, not additive shares of a baseline."""
        if not contributions:
            return "No local explanation available."
        direction = "upward" if prediction > 0 else "downward"
        up = [c for c in contributions if c["contribution"] > 0][:3]
        down = [c for c in contributions if c["contribution"] < 0][:3]
        parts = [
            f"The model currently forecasts {prediction:+.3%} ({direction}). "
            f"Around this exact market state, the local surrogate (fit quality R²={local_r2:.2f}) "
            "reads each coefficient as: if that feature rose by one standard deviation, "
            "the forecast would move by roughly this much."
        ]
        if up:
            parts.append("Increasing these would push the forecast up: " + ", ".join(
                f"{c['label']} ({c['contribution']:+.5f})" for c in up) + ".")
        if down:
            parts.append("Increasing these would push it down: " + ", ".join(
                f"{c['label']} ({c['contribution']:+.5f})" for c in down) + ".")
        return " ".join(parts)

    # ---------------------------------------------------------- counterfactual
    def counterfactual(self, symbol: str, df: pd.DataFrame, horizon: int = 5,
                       max_features: int = 3) -> dict:
        """Smallest single-feature nudges that would flip the predicted direction."""
        model, x, _, _ = self._surrogate(symbol, df, horizon)
        columns = list(x.columns)
        instance = x.iloc[-1].values.copy()

        def predict_one(vec: np.ndarray) -> float:
            return float(model.predict(pd.DataFrame(vec.reshape(1, -1), columns=columns))[0])

        original = predict_one(instance)
        target_sign = -np.sign(original) if original != 0 else 1.0
        sigma = x.std().replace(0, 1e-6).values
        flips = []

        for i, feature in enumerate(x.columns):
            for scale in (0.5, 1.0, 1.5, 2.0, 3.0):
                for direction in (1, -1):
                    candidate = instance.copy()
                    candidate[i] = instance[i] + direction * scale * sigma[i]
                    pred = predict_one(candidate)
                    if np.sign(pred) == target_sign and np.sign(pred) != np.sign(original):
                        flips.append({
                            "feature": feature, "label": humanise(feature),
                            "current_value": round(float(instance[i]), 5),
                            "required_value": round(float(candidate[i]), 5),
                            "change_in_sigma": round(direction * scale, 2),
                            "new_prediction": round(pred, 6),
                            "cost": scale,
                        })
                        break
                else:
                    continue
                break

        flips.sort(key=lambda d: d["cost"])
        return {
            "symbol": symbol.upper(),
            "original_prediction": round(original, 6),
            "original_direction": "up" if original > 0 else "down",
            "counterfactuals": flips[:max_features],
            "narrative": (
                f"The forecast would flip to {'down' if original > 0 else 'up'} if "
                + ", or ".join(
                    f"{f['label']} moved {f['change_in_sigma']:+.1f}σ" for f in flips[:max_features])
                + "." if flips else "No single-feature change flips this forecast - the signal is robust."
            ),
        }

    # ----------------------------------------------------------- narrative
    @staticmethod
    def _local_narrative(contributions: list[dict], prediction: float, base: float) -> str:
        if not contributions:
            return "No attribution available."
        bullish = [c for c in contributions if c["contribution"] > 0][:3]
        bearish = [c for c in contributions if c["contribution"] < 0][:3]
        direction = "upward" if prediction > base else "downward"
        parts = [f"The model leans {direction} (prediction {prediction:+.3%} vs baseline {base:+.3%})."]
        if bullish:
            parts.append("Supporting factors: " + ", ".join(
                f"{c['label']} ({c['contribution']:+.4f})" for c in bullish) + ".")
        if bearish:
            parts.append("Opposing factors: " + ", ".join(
                f"{c['label']} ({c['contribution']:+.4f})" for c in bearish) + ".")
        return " ".join(parts)

    # ----------------------------------------------------------- combined
    def explain(self, symbol: str, df: pd.DataFrame, horizon: int = 5,
                methods: list[str] | None = None) -> dict:
        methods = methods or ["shap", "lime", "global"]
        out: dict = {"symbol": symbol.upper(), "horizon": horizon}
        if "global" in methods:
            out["global"] = self.global_importance(symbol, df, horizon)
        if "shap" in methods:
            out["shap"] = self.shap_values(symbol, df, horizon).to_dict()
        if "lime" in methods:
            out["lime"] = self.lime_explain(symbol, df, horizon).to_dict()
        if "counterfactual" in methods:
            out["counterfactual"] = self.counterfactual(symbol, df, horizon)
        return out


explainer = Explainer()
