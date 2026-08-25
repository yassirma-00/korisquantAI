"""Intelligent recommendation engine.

Fuses four independent evidence streams into a single, explainable decision:

    1. Deep-learning price forecast        (weight ~0.30)
    2. Reinforcement-learning agent action (weight ~0.25)
    3. Technical indicator consensus       (weight ~0.25)
    4. News sentiment                      (weight ~0.20)

Weights are re-normalised over the signals that are actually available and are
adjusted by each signal's own reliability (e.g. a forecaster with 48% directional
accuracy is discounted towards zero). The output includes a confidence score, a
risk assessment, position sizing and a human-readable rationale.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from app.core.logging import get_logger
from app.services.data.market_data import market_data_service
from app.services.forecasting.trainer import forecast_trainer
from app.services.indicators.technical import compute_indicators, signal_summary
from app.services.nlp.news import news_service
from app.services.risk.anomaly import anomaly_detector
from app.services.risk.metrics import full_metrics
from app.services.rl.service import rl_service
from app.services.xai.explainer import explainer

logger = get_logger(__name__)

ACTIONS = ("STRONG_BUY", "BUY", "HOLD", "SELL", "STRONG_SELL")

BASE_WEIGHTS = {"forecast": 0.30, "rl": 0.25, "technical": 0.25, "sentiment": 0.20}


@dataclass
class SignalContribution:
    source: str
    score: float            # signed [-1, 1]
    weight: float
    reliability: float
    available: bool
    detail: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "source": self.source, "score": round(self.score, 4),
            "weight": round(self.weight, 4), "reliability": round(self.reliability, 4),
            "weighted_contribution": round(self.score * self.weight, 4),
            "available": self.available, "detail": self.detail,
        }


class RecommendationEngine:
    # ------------------------------------------------------ signal builders
    def _forecast_signal(self, symbol: str, df: pd.DataFrame, model: str, horizon: int) -> SignalContribution:
        try:
            pred = forecast_trainer.predict(symbol, df, model_name=model, horizon=horizon)
        except Exception as exc:
            logger.info("forecast signal unavailable for %s: %s", symbol, exc)
            return SignalContribution("forecast", 0.0, 0.0, 0.0, False,
                                      {"reason": str(exc)[:160]})
        expected = pred["predicted_return"]
        vol = pred.get("residual_std") or 0.02
        score = float(np.clip(expected / max(vol, 1e-6) / 2.0, -1, 1))
        da = pred.get("test_metrics", {}).get("directional_accuracy", 50.0)
        reliability = float(np.clip((da - 45) / 25, 0.0, 1.0))
        return SignalContribution(
            "forecast", score, BASE_WEIGHTS["forecast"], reliability, True,
            {"predicted_return": expected, "predicted_price": pred["predicted_price"],
             "horizon_days": horizon, "model": model,
             "directional_accuracy": da, "confidence": pred["confidence"]})

    def _rl_signal(self, symbol: str, algo: str) -> SignalContribution:
        try:
            reco = rl_service.recommend_action(symbol, algo=algo)
        except Exception as exc:
            logger.info("RL signal unavailable for %s: %s", symbol, exc)
            return SignalContribution("rl", 0.0, 0.0, 0.0, False, {"reason": str(exc)[:160]})
        mapping = {"BUY": 1.0, "HOLD": 0.0, "SELL": -1.0}
        score = mapping.get(reco["action"], 0.0) * reco.get("confidence", 0.5) * 1.4
        perf = reco.get("agent_test_performance", {})
        sharpe = perf.get("sharpe_ratio", 0.0)
        alpha = perf.get("alpha_vs_buy_hold", 0.0)
        reliability = float(np.clip(0.35 + sharpe / 3.0 + alpha, 0.0, 1.0))
        return SignalContribution(
            "rl", float(np.clip(score, -1, 1)), BASE_WEIGHTS["rl"], reliability, True,
            {"action": reco["action"], "q_values": reco.get("q_values", {}),
             "agent_sharpe": sharpe, "agent_alpha_vs_bh": alpha, "algo": algo})

    def _technical_signal(self, df: pd.DataFrame) -> SignalContribution:
        enriched = compute_indicators(df, ["sma", "ema", "rsi", "macd", "bbands",
                                           "atr", "adx", "stoch", "mfi"])
        summary = signal_summary(enriched)
        if not summary:
            return SignalContribution("technical", 0.0, 0.0, 0.0, False, {})
        buy, sell = summary["buy_votes"], summary["sell_votes"]
        total = max(buy + sell + summary["neutral_votes"], 1)
        score = (buy - sell) / total * 1.8
        reliability = float(np.clip(0.45 + summary["strength"] * 0.55, 0, 1))
        return SignalContribution(
            "technical", float(np.clip(score, -1, 1)), BASE_WEIGHTS["technical"], reliability, True,
            {"consensus": summary["consensus"], "buy_votes": buy, "sell_votes": sell,
             "indicators": summary["indicators"]})

    def _sentiment_signal(self, symbol: str) -> SignalContribution:
        try:
            summary = news_service.sentiment_summary(symbol, limit=12)
        except Exception as exc:
            return SignalContribution("sentiment", 0.0, 0.0, 0.0, False, {"reason": str(exc)[:160]})
        if not summary.get("n"):
            return SignalContribution("sentiment", 0.0, 0.0, 0.0, False, {"reason": "no news"})
        score = float(np.clip(summary["score"] * 2.2, -1, 1))
        reliability = float(np.clip(0.3 + min(summary["n"], 15) / 25 + summary["confidence"] * 0.3, 0, 1))
        return SignalContribution(
            "sentiment", score, BASE_WEIGHTS["sentiment"], reliability, True,
            {"label": summary["label"], "n_articles": summary["n"],
             "bullish_ratio": summary.get("bullish_ratio"),
             "bearish_ratio": summary.get("bearish_ratio"),
             "backend": summary.get("backend"),
             "top_news": summary.get("top_impact_news", [])[:3]})

    # ----------------------------------------------------------- decision
    @staticmethod
    def _to_action(score: float) -> str:
        if score >= 0.45:
            return "STRONG_BUY"
        if score >= 0.15:
            return "BUY"
        if score <= -0.45:
            return "STRONG_SELL"
        if score <= -0.15:
            return "SELL"
        return "HOLD"

    @staticmethod
    def _position_size(score: float, action: str, risk: dict, metrics: dict) -> dict:
        """Volatility-targeted sizing, capped and reduced when risk is elevated.

        The platform is long-only, so a bearish call is an instruction to *reduce*,
        not to allocate: the target weight for SELL / STRONG_SELL is zero and the
        conviction figure is reported as the strength of the exit signal instead.
        """
        vol = max(metrics.get("annualised_volatility", 0.25), 0.05)
        target_vol = 0.15
        base = min(target_vol / vol, 1.0)
        conviction = min(abs(score) / 0.6, 1.0)
        risk_level = risk.get("overall_risk_level", "moderate")
        risk_haircut = {"low": 1.0, "moderate": 0.75, "high": 0.45, "critical": 0.2}.get(risk_level, 0.7)
        atr_pct = risk.get("crash_risk", {}).get("atr_pct", 2.0)

        bearish = action in ("SELL", "STRONG_SELL")
        if bearish:
            weight = 0.0
            trim = float(np.clip(conviction, 0.0, 1.0))
            rationale = (
                f"Bearish signal (conviction {conviction:.0%}): target allocation is 0% — this is a "
                f"long-only platform, so the recommendation is to trim or exit rather than short. "
                f"Suggested reduction of any existing position: {trim:.0%}."
            )
        else:
            weight = float(np.clip(base * conviction * risk_haircut, 0.0, 0.35))
            trim = 0.0
            rationale = (
                f"Sized to a {target_vol:.0%} volatility target on an asset with {vol:.0%} realised "
                f"volatility, scaled by conviction ({conviction:.0%}) and a {risk_level}-risk haircut."
            )

        return {
            "suggested_portfolio_weight": round(weight, 4),
            "suggested_trim_fraction": round(trim, 4),
            "direction": "reduce" if bearish else "accumulate" if weight > 0 else "hold",
            "conviction": round(conviction, 4),
            "max_weight_cap": 0.35,
            "volatility_target": target_vol,
            "asset_volatility": round(vol, 4),
            "stop_loss_pct": round(min(atr_pct * 2.0, 15.0), 2),
            "take_profit_pct": round(min(atr_pct * 3.5, 30.0), 2),
            "rationale": rationale,
        }

    # -------------------------------------------------------------- public
    def recommend(
        self,
        symbol: str,
        period: str = "2y",
        forecast_model: str = "lstm",
        horizon: int = 5,
        rl_algo: str = "dueling_dqn",
        include_xai: bool = True,
    ) -> dict:
        symbol = symbol.upper().strip()
        series = market_data_service.get_history(symbol, period=period)
        df = series.df

        signals = [
            self._forecast_signal(symbol, df, forecast_model, horizon),
            self._rl_signal(symbol, rl_algo),
            self._technical_signal(df),
            self._sentiment_signal(symbol),
        ]

        # ---- weighted fusion over available signals, discounted by reliability
        effective = [(s, s.weight * (0.35 + 0.65 * s.reliability)) for s in signals if s.available]
        total_weight = sum(w for _, w in effective) or 1.0
        composite = sum(s.score * w for s, w in effective) / total_weight

        risk = anomaly_detector.scan(symbol, df)
        returns = df["close"].pct_change().dropna()
        metrics = full_metrics(returns)

        # Risk overlay: temper bullishness when crash/bubble risk is elevated.
        # The drag can neutralise a long signal but never invert it into a short —
        # elevated risk is a reason to step aside, not a reason to bet the other way.
        # `.get(key, 0.0)` does NOT protect against a key that exists holding
        # None — and both scores are deliberately None when the window is too
        # short to compute them. That crashed the whole recommendation on any
        # period under ~200 bars. Treating an unknown score as zero drag is the
        # right fallback: absent evidence must not manufacture a penalty.
        crash_score = risk.get("crash_risk", {}).get("crash_risk_score") or 0.0
        bubble_score = risk.get("bubble", {}).get("bubble_score") or 0.0
        if composite > 0:
            raw_drag = 0.35 * crash_score + 0.25 * bubble_score
            risk_drag = -min(raw_drag, composite)
        else:
            risk_drag = 0.0
        adjusted = float(np.clip(composite + risk_drag, -1, 1))
        action = self._to_action(adjusted)

        agreement = self._agreement(signals)
        confidence = float(np.clip(
            0.35 * abs(adjusted)
            + 0.30 * agreement
            + 0.20 * (sum(s.reliability for s, _ in effective) / max(len(effective), 1))
            + 0.15 * (len(effective) / len(signals)), 0.0, 0.98))

        sizing = self._position_size(adjusted, action, risk, metrics)
        xai_payload = None
        if include_xai:
            try:
                shap_exp = explainer.shap_values(symbol, df, horizon=horizon, top_k=8)
                xai_payload = shap_exp.to_dict()
            except Exception as exc:
                logger.info("XAI unavailable for %s: %s", symbol, exc)

        return {
            "symbol": symbol,
            "name": series.instrument.name,
            "asset_class": series.instrument.asset_class,
            "generated_at": datetime.now(UTC).isoformat(),
            "as_of": str(df.index[-1].date()),
            # Echo the window this recommendation was actually computed over.
            # Without it a caller cannot tell a fresh result from a stale one,
            # which is exactly how the frontend sending no period at all went
            # unnoticed: every answer looked plausible.
            "period": period,
            "bars_analysed": int(len(df)),
            "period_start": str(df.index[0].date()),
            "data_source": series.source,
            "last_price": round(float(df["close"].iloc[-1]), 4),
            "action": action,
            "composite_score": round(adjusted, 4),
            "raw_score": round(composite, 4),
            "risk_adjustment": round(risk_drag, 4),
            "confidence": round(confidence, 4),
            "signal_agreement": round(agreement, 4),
            "signals": [s.to_dict() for s in signals],
            "risk": {
                "overall_level": risk["overall_risk_level"],
                "crash_risk": risk["crash_risk"],
                "bubble": risk["bubble"],
                "recent_anomalies": risk["anomalies"][:5],
            },
            "performance_metrics": metrics,
            "position_sizing": sizing,
            "explanation": self._narrative(symbol, action, adjusted, signals, risk, confidence),
            "xai": xai_payload,
            "disclaimer": ("Educational and research output produced by statistical models. "
                           "Not investment advice. Past performance does not guarantee future results."),
        }

    @staticmethod
    def _agreement(signals: list[SignalContribution]) -> float:
        scores = [s.score for s in signals if s.available and abs(s.score) > 0.05]
        if len(scores) < 2:
            return 0.5
        signs = [np.sign(s) for s in scores]
        majority = max(signs.count(1), signs.count(-1))
        return float(majority / len(signs))

    @staticmethod
    def _narrative(symbol: str, action: str, score: float,
                   signals: list[SignalContribution], risk: dict, confidence: float) -> dict:
        available = [s for s in signals if s.available]
        bullish = [s for s in available if s.score > 0.1]
        bearish = [s for s in available if s.score < -0.1]
        neutral = [s for s in available if abs(s.score) <= 0.1]
        missing = [s.source for s in signals if not s.available]

        label_map = {
            "forecast": "the deep-learning price forecast",
            "rl": "the reinforcement-learning agent",
            "technical": "technical indicator consensus",
            "sentiment": "news sentiment",
        }

        lines = [f"**{action.replace('_', ' ')}** on {symbol} with a composite score of {score:+.2f} "
                 f"and {confidence:.0%} confidence."]
        if bullish:
            lines.append("Bullish evidence from " + ", ".join(
                f"{label_map[s.source]} ({s.score:+.2f})" for s in bullish) + ".")
        if bearish:
            lines.append("Bearish evidence from " + ", ".join(
                f"{label_map[s.source]} ({s.score:+.2f})" for s in bearish) + ".")
        if neutral:
            lines.append("Neutral: " + ", ".join(label_map[s.source] for s in neutral) + ".")

        crash = risk.get("crash_risk", {})
        # `.get(key, 0)` does not protect against a key that exists and holds
        # None — it returns the None. On a short window the crash model
        # declines to guess and reports None rather than a false 0.0, so
        # `period=1mo` raised "unsupported format string passed to
        # NoneType.__format__" and the whole recommendation returned a 500.
        # `or 0.0` is the pattern already used elsewhere for exactly this.
        crash_score = crash.get("crash_risk_score")
        crash_var = crash.get("var_95_daily")
        crash_dd = crash.get("current_drawdown")
        # Say "not measurable" rather than print a 0.00 that reads as "no risk".
        score_text = "not measurable on this window" if crash_score is None \
            else f"{crash_score:.2f}"
        var_text = "n/a" if crash_var is None else f"{crash_var:.2%}"
        dd_text = "n/a" if crash_dd is None else f"{crash_dd:.1%}"
        lines.append(f"Risk level is **{risk.get('overall_risk_level', 'unknown')}** "
                     f"(crash-risk score {score_text}, "
                     f"daily VaR₉₅ {var_text}, "
                     f"current drawdown {dd_text}). "
                     f"{crash.get('recommendation', '')}")
        if missing:
            lines.append("Signals unavailable (models not yet trained): " + ", ".join(missing) + ".")

        return {
            "summary": " ".join(lines),
            "key_drivers": [
                {"source": s.source, "label": label_map[s.source], "score": round(s.score, 3),
                 "reliability": round(s.reliability, 3)}
                for s in sorted(available, key=lambda x: abs(x.score * x.reliability), reverse=True)
            ],
            "missing_signals": missing,
        }

    # ------------------------------------------------------------ batch
    def screen(self, symbols: list[str], **kwargs) -> list[dict]:
        """Rank a watchlist by composite score (light mode: no XAI)."""
        kwargs.setdefault("include_xai", False)
        out = []
        for sym in symbols:
            try:
                reco = self.recommend(sym, **kwargs)
                out.append({
                    "symbol": reco["symbol"], "name": reco["name"],
                    "action": reco["action"], "score": reco["composite_score"],
                    "confidence": reco["confidence"], "last_price": reco["last_price"],
                    "risk_level": reco["risk"]["overall_level"],
                    "suggested_weight": reco["position_sizing"]["suggested_portfolio_weight"],
                })
            except Exception as exc:
                logger.warning("screen failed for %s: %s", sym, exc)
                out.append({"symbol": sym.upper(), "error": str(exc)[:200]})
        return sorted(out, key=lambda d: d.get("score", -99), reverse=True)


recommendation_engine = RecommendationEngine()
