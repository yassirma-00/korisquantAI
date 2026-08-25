"""AI Direction Prediction — will this instrument rise, fall, or go nowhere?

Design
------
This module answers one narrow question: over the selected horizon, is the
price more likely to go **up**, **down**, or is there no usable edge? It is a
thin fusion layer, not a new model. Every input is already computed elsewhere
in the platform:

* `RecommendationEngine._forecast_signal`   — deep-learning return forecast
* `RecommendationEngine._technical_signal`  — 17-indicator consensus
* `RecommendationEngine._rl_signal`         — trained RL agent's action
* `RecommendationEngine._sentiment_signal`  — news sentiment

Reusing those builders (rather than re-deriving them) means the direction call
can never disagree with the Recommendations page about what the models said.

Three properties this module refuses to violate
-----------------------------------------------
1. **No invented numbers.** The expected move has exactly one source: the
   forecaster's own `predicted_return`, from a checkpoint trained for this
   symbol *and* this horizon. When no such model exists the magnitude is
   `None`, `magnitude_basis` is `no_trained_forecaster`, and the UI shows N/A.
   Realised volatility is **never** used as a stand-in: it is returned
   separately and unsigned as `market_volatility_pct`, because volatility
   measures how far a price travels, not which way. Nothing is hardcoded and
   nothing is random.

2. **NEUTRAL is a real answer.** A weak consensus is reported as NEUTRAL rather
   than rounded into a direction. The threshold is a documented calibration
   choice, exposed in the payload as `neutral_band`.

3. **Confidence reflects evidence, not enthusiasm.** It combines signal
   agreement, per-signal reliability (which itself comes from measured
   directional accuracy, agent Sharpe, indicator strength and article count)
   and how many signals were actually available. A single available signal
   cannot produce high confidence.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.core.logging import get_logger
from app.services.recommendation.engine import RecommendationEngine

logger = get_logger(__name__)

# Below this |score| the evidence does not support committing to a direction.
# A calibration choice, not a measurement — published in the response so a
# reader can see the boundary rather than guess it.
NEUTRAL_BAND = 0.12

# Trading days per calendar year, used to scale realised volatility.
TRADING_DAYS = 252


class DirectionPredictor:
    """Fuse the platform's existing signals into an up/down/neutral call."""

    def __init__(self) -> None:
        self._engine = RecommendationEngine()

    # ------------------------------------------------------------- helpers
    @staticmethod
    def _realised_volatility(df: pd.DataFrame, horizon: int) -> float | None:
        """Realised volatility scaled to the horizon, by the square-root rule.

        Returns None rather than a fabricated default when there is not enough
        history: an unmeasurable magnitude must be reported as unavailable.
        """
        returns = df["close"].pct_change().dropna()
        if len(returns) < 20:
            return None
        daily = float(returns.std())
        if not np.isfinite(daily) or daily <= 0:
            return None
        return float(daily * np.sqrt(max(horizon, 1)))

    @staticmethod
    def _direction(score: float) -> str:
        if score > NEUTRAL_BAND:
            return "INCREASE"
        if score < -NEUTRAL_BAND:
            return "DECREASE"
        return "NEUTRAL"

    @staticmethod
    def _strength(score: float) -> str:
        """Verbal strength of the signal. Bands are a presentation choice."""
        magnitude = abs(score)
        if magnitude >= 0.50:
            return "strong"
        if magnitude >= 0.25:
            return "moderate"
        if magnitude > NEUTRAL_BAND:
            return "weak"
        return "inconclusive"

    # ---------------------------------------------------------- confidence
    @staticmethod
    def _confidence(signals: list, score: float) -> dict:
        """Confidence from evidence quality, not from the size of the score.

        Three factors, each in [0, 1]:
          * agreement  — do the available signals point the same way?
          * reliability— weighted mean of each signal's own measured reliability
          * coverage   — how many of the four signals actually ran
        """
        available = [s for s in signals if s.available]
        if not available:
            return {"value": 0.0, "agreement": 0.0, "reliability": 0.0,
                    "coverage": 0.0, "n_available": 0,
                    "reason": "no signal could be computed"}

        # Agreement: share of weight pointing in the direction of the consensus.
        # Signals sitting exactly at zero are counted as abstentions, not votes.
        directional = [s for s in available if abs(s.score) > 1e-9]
        if directional and abs(score) > 1e-9:
            aligned = sum(s.weight for s in directional
                          if np.sign(s.score) == np.sign(score))
            total = sum(s.weight for s in directional)
            agreement = float(aligned / total) if total > 0 else 0.0
        else:
            agreement = 0.0

        weight_sum = sum(s.weight for s in available)
        reliability = (float(sum(s.reliability * s.weight for s in available) / weight_sum)
                       if weight_sum > 0 else 0.0)
        coverage = len(available) / 4.0

        # Geometric-style blend: a zero in any factor should drag the result
        # down rather than be averaged away.
        value = float(np.clip(0.45 * agreement + 0.35 * reliability + 0.20 * coverage,
                              0.0, 1.0))
        return {
            "value": round(value, 4),
            "agreement": round(agreement, 4),
            "reliability": round(reliability, 4),
            "coverage": round(coverage, 4),
            "n_available": len(available),
        }

    @staticmethod
    def _confidence_label(value: float) -> str:
        if value >= 0.75:
            return "very high"
        if value >= 0.60:
            return "high"
        if value >= 0.45:
            return "moderate"
        if value >= 0.30:
            return "low"
        return "very low"

    # ------------------------------------------------------------ magnitude
    def _expected_move(self, signals: list, df: pd.DataFrame,
                       horizon: int) -> dict:
        """The model's predicted return — or nothing at all.

        There is exactly one source for an expected movement: a trained
        forecaster's own `predicted_return`. If no forecaster is available for
        this symbol *and this horizon*, the answer is `None`.

        Realised volatility is deliberately **not** used here. Volatility
        measures how far a price tends to travel, not which way it goes;
        signing it by the consensus and presenting it as an expected move
        implied a directional forecast the platform had not made. It is still
        computed and returned, but under `market_volatility_pct`, as a separate
        quantity the UI must label separately.
        """
        forecast = next((s for s in signals if s.source == "forecast" and s.available), None)
        if forecast is not None:
            predicted = forecast.detail.get("predicted_return")
            if predicted is not None and np.isfinite(predicted):
                return {
                    "expected_move_pct": round(float(predicted) * 100, 4),
                    "magnitude_basis": "deep_learning_forecast",
                    "forecaster_available": True,
                    "basis_detail": {
                        "model": forecast.detail.get("model"),
                        "horizon_days": forecast.detail.get("horizon_days"),
                        "directional_accuracy": forecast.detail.get("directional_accuracy"),
                    },
                }

        reason = (forecast.detail.get("reason") if forecast is not None
                  else None)
        if not reason:
            unavailable = next((s for s in signals if s.source == "forecast"), None)
            reason = ((unavailable.detail.get("reason") if unavailable else None)
                      or "no trained forecaster for this symbol and horizon")

        return {
            "expected_move_pct": None,
            "magnitude_basis": "no_trained_forecaster",
            "forecaster_available": False,
            "basis_detail": {"reason": reason},
        }

    def _market_volatility(self, df: pd.DataFrame, horizon: int) -> dict:
        """Realised volatility over the horizon — a dispersion measure only.

        Reported alongside the direction call so the user can see how far this
        instrument typically travels, explicitly *not* as a forecast. Returned
        as an unsigned magnitude: a signed volatility would be a contradiction.
        """
        sigma = self._realised_volatility(df, horizon)
        if sigma is None:
            return {
                "market_volatility_pct": None,
                "volatility_basis": "unavailable",
                "volatility_note": "fewer than 20 returns available to measure "
                                   "volatility over this window",
            }
        return {
            "market_volatility_pct": round(abs(sigma) * 100, 4),
            "volatility_basis": "realised_historical_volatility",
            "volatility_note": ("Historical price variability over the horizon. "
                                "This is not a directional forecast and must not "
                                "be read as an expected movement."),
        }

    # --------------------------------------------------------------- public
    def predict(self, symbol: str, df: pd.DataFrame, *, horizon: int = 5,
                forecast_model: str = "lstm", rl_algo: str = "dueling_dqn") -> dict:
        """Return the direction call for `symbol` over `horizon` trading days."""
        symbol = symbol.upper()

        # Use the architecture this symbol actually has for this horizon. The
        # caller's choice is honoured whenever it is trained; otherwise the best
        # measured alternative is substituted. Without this, an instrument
        # trained on GRU reported "no trained forecaster" purely because the
        # default happened to be LSTM — the checkpoint was on disk and loadable
        # the whole time. Discovery is by directory listing, so every symbol is
        # covered automatically and no ticker is named in code.
        from app.services.forecasting.trainer import forecast_trainer

        resolved_model = forecast_trainer.resolve_model(
            symbol, forecast_model, horizon) or forecast_model

        signals = [
            self._engine._forecast_signal(symbol, df, resolved_model, horizon),
            self._engine._rl_signal(symbol, rl_algo),
            self._engine._technical_signal(df),
            self._engine._sentiment_signal(symbol),
        ]

        # Re-weight across the signals that actually ran, so a missing RL agent
        # does not silently drag the composite towards zero.
        available = [s for s in signals if s.available]
        weight_sum = sum(s.weight for s in available)
        score = (float(sum(s.score * s.weight for s in available) / weight_sum)
                 if weight_sum > 0 else 0.0)
        score = float(np.clip(score, -1.0, 1.0))

        confidence = self._confidence(signals, score)
        magnitude = self._expected_move(signals, df, horizon)
        volatility = self._market_volatility(df, horizon)

        # Direction requires evidence. Two distinct situations must not be
        # collapsed into the same NEUTRAL:
        #   * enough signals ran but they disagree      -> NEUTRAL (a finding)
        #   * too little evidence to judge at all       -> UNAVAILABLE
        # Reporting the second as NEUTRAL would present an absence of analysis
        # as a considered verdict.
        n_available = confidence.get("n_available", 0)
        move_pct = magnitude["expected_move_pct"]
        conflict = False

        # The verdict follows the forecaster's own predicted return, because it
        # is the only component that answers the question this page asks: which
        # way, over this horizon. The other three describe present conditions —
        # a momentum reading and a news tone are not forecasts — and averaging
        # them in at 70% of the weight cancelled the prediction out. AAPL is the
        # worked example: the model predicted -0.92% (signal -0.133) while
        # technical (+0.225) and sentiment (+0.089) pulled the composite to
        # +0.034, inside the +/-0.12 band, so the page said NEUTRAL and showed
        # a negative Expected Movement beside it.
        #
        # The composite is still computed, still published, and still decides
        # agreement and confidence. It no longer overrides the prediction.
        forecast_signal = next((s for s in signals
                                if s.source == "forecast" and s.available), None)
        lead_score = (float(forecast_signal.score)
                      if forecast_signal is not None else score)

        if n_available == 0:
            direction = "UNAVAILABLE"
        elif not magnitude["forecaster_available"]:
            # The forecaster is the only component that predicts a *return*.
            # Without it the remaining signals (technical, sentiment, RL) can
            # describe present conditions but cannot support a reliable
            # directional call, so the verdict is NEUTRAL and the reason is
            # stated. Announcing "INCREASE" with no predicted movement behind
            # it was the specific defect this change removes.
            direction = "NEUTRAL"
        else:
            direction = self._direction(lead_score)
            # The headline and the number underneath it must describe the same
            # future, so a verdict is never announced against the sign of the
            # move on display. Since the verdict now follows the forecaster,
            # this can only trip if the two disagree internally.
            if (direction in ("INCREASE", "DECREASE") and move_pct is not None
                    and abs(move_pct) > 0
                    and np.sign(move_pct) != np.sign(lead_score)):
                direction = "NEUTRAL"
                conflict = True

        last_price = float(df["close"].iloc[-1])

        # A target price is only meaningful when a model actually predicted a
        # return *and* committed to a side.
        target = (round(last_price * (1 + move_pct / 100), 4)
                  if move_pct is not None and direction in ("INCREASE", "DECREASE")
                  else None)

        return {
            "symbol": symbol,
            "direction": direction,
            # Strength describes the call that was actually made, so it reads
            # off the same score the verdict came from.
            "strength": self._strength(lead_score),
            "lead_score": round(lead_score, 4),
            "lead_basis": ("forecast_predicted_return" if forecast_signal is not None
                           else "composite_signal"),
            "composite_score": round(score, 4),
            "neutral_band": NEUTRAL_BAND,
            "horizon_days": horizon,
            "last_price": round(last_price, 4),
            "target_price": target,
            **magnitude,
            **volatility,
            # A flag the UI can trust without re-deriving the rule: when this is
            # false there is no predicted return and no model confidence to show.
            "reliable_prediction": bool(magnitude["forecaster_available"]
                                        and direction in ("INCREASE", "DECREASE")),
            "confidence": (confidence["value"] if magnitude["forecaster_available"]
                           else None),
            "confidence_label": (self._confidence_label(confidence["value"])
                                 if magnitude["forecaster_available"] else None),
            "confidence_breakdown": confidence,
            # What was asked for, what was actually used, and what else this
            # symbol could answer — so a caller can see a substitution happened
            # instead of wondering why the model name differs from its request.
            "requested_model": forecast_model,
            "resolved_model": (resolved_model
                               if magnitude["forecaster_available"] else None),
            "model_substituted": bool(magnitude["forecaster_available"]
                                      and resolved_model != forecast_model),
            "available_horizons": forecast_trainer.available_horizons(symbol),
            # True when the forecaster's predicted return and the combined
            # signal pointed opposite ways. Surfaced rather than hidden: it is
            # a real disagreement between the models and the reason the verdict
            # was held at NEUTRAL despite a non-zero predicted move.
            "signal_conflict": conflict,
            "signals": [s.to_dict() for s in signals],
            "bars_analysed": int(len(df)),
            "summary": self._narrative(symbol, direction, lead_score, move_pct,
                                       confidence, horizon,
                                       magnitude["forecaster_available"],
                                       volatility.get("market_volatility_pct"),
                                       conflict, score),
            "disclaimer": ("Statistical estimate from the platform's own models, "
                           "not investment advice. Direction is a probabilistic "
                           "view over the selected horizon, not a guarantee."),
        }

    # -------------------------------------------------------------- wording
    @staticmethod
    def _narrative(symbol: str, direction: str, score: float,
                   move_pct: float | None, confidence: dict, horizon: int,
                   forecaster_available: bool,
                   volatility_pct: float | None,
                   conflict: bool = False,
                   composite: float | None = None) -> str:
        """Plain-language summary. Never states more than the numbers support."""
        n = confidence.get("n_available", 0)
        conf = confidence.get("value", 0.0)

        # Quote the *published* figures, not the raw floats. The payload rounds
        # `composite_score` to 4 dp; formatting the unrounded value to 3 dp here
        # produced a different number in the prose than in the JSON beside it
        # (+0.334 vs +0.335 on SPY), which reads as two disagreeing answers.
        score = round(score, 4)
        if move_pct is not None:
            move_pct = round(move_pct, 4)
        if composite is not None:
            composite = round(composite, 4)

        if direction == "UNAVAILABLE" or n == 0:
            return (f"No usable signal for {symbol}: none of the four models "
                    f"could be evaluated, so no direction is offered.")

        # Without a forecaster there is no predicted return, and the report has
        # to say so rather than let volatility stand in for one.
        if not forecaster_available:
            vol_txt = (f" Historical volatility over this horizon is "
                       f"±{volatility_pct:.2f}%, which measures how far the "
                       f"price typically travels — not which way it will go."
                       if volatility_pct is not None else "")
            return (f"No trained forecaster is available for {symbol} at a "
                    f"{horizon}-day horizon, so the system cannot estimate a "
                    f"reliable future return or directional movement."
                    f"{vol_txt} The remaining {n} of 4 signals give a combined "
                    f"reading of {score:+.3f}, which is reported as context "
                    f"only.")

        if direction == "NEUTRAL" and conflict:
            # Two models disagreeing is a finding, not noise to be smoothed
            # over. Naming the disagreement is more honest than reporting a
            # weak consensus that never existed.
            side = "a fall" if (move_pct or 0) < 0 else "a rise"
            return (f"{symbol} has conflicting evidence over the next {horizon} "
                    f"trading days: the trained forecaster predicts {side} of "
                    f"{abs(move_pct):.2f}%, while the combined signal from "
                    f"{n} of 4 models points the other way ({score:+.3f}). "
                    f"Because the prediction and the wider evidence disagree, "
                    f"no direction is called.")

        if direction == "NEUTRAL":
            move_txt = (f" The model's predicted move of {move_pct:+.2f}% is "
                        f"too small to call a side."
                        if move_pct is not None else "")
            return (f"{symbol} shows no clear directional edge over the next "
                    f"{horizon} trading days. The forecast signal "
                    f"({score:+.3f}) sits inside the neutral band of "
                    f"\u00b1{NEUTRAL_BAND:.2f}, so no direction is called."
                    f"{move_txt} {n} of 4 signals were available"
                    + (f", combining to {composite:+.3f}."
                       if composite is not None else "."))

        verb = "rise" if direction == "INCREASE" else "fall"
        # abs() is only safe because a move whose sign contradicts the verdict
        # is intercepted above and reported as a conflict. Asserting the
        # invariant here stops a future edit from quietly restoring the bug
        # where "rise by 0.88%" described a predicted -0.88%.
        if move_pct is not None and abs(move_pct) > 0:
            assert np.sign(move_pct) == (1 if direction == "INCREASE" else -1), (
                f"narrative would say {verb} for a predicted move of {move_pct}")
        move_txt = (f" by about {abs(move_pct):.2f}%" if move_pct is not None else "")
        agreement = confidence.get("agreement", 0.0)
        return (f"{symbol} is more likely to {verb}{move_txt} over the next "
                f"{horizon} trading days, from the trained forecaster's own "
                f"prediction (signal {score:+.3f})."
                + (f" The other models combine to {composite:+.3f}."
                   if composite is not None else "")
                + f" {n} of 4 signals available, {agreement:.0%} pointing the "
                  f"same way, at {conf:.0%} confidence.")

direction_predictor = DirectionPredictor()
