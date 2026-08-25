"""Decision audit trail for RL recommendations.

``recommendation_log`` existed as a table but nothing ever wrote to it, so the
platform had a decision log with no decisions in it. Model-governance reviews
(SR 11-7, EBA/ACPR guidance on AI) ask a simple question — *for this decision,
which model version produced it, on what evidence, and under what market
conditions?* — and an empty table cannot answer it.

This writes one row per RL recommendation, capturing the model identity, the
regime that was in force, how much that regime actually moved the call, and the
risk metrics at the time.

Two deliberate properties:

* **Never fails the caller.** An audit write that raises would turn a working
  recommendation into a 500. Failures are logged and swallowed; a missing audit
  row is a reporting gap, a broken recommendation is an outage.
* **Model version is derived from the checkpoint**, not hand-maintained. A
  version string someone has to remember to bump is a version string that goes
  stale, and a stale one is worse than none because it is believed.
"""

from __future__ import annotations

import contextlib
import hashlib
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.db.models import RecommendationLog

logger = get_logger(__name__)


def model_version(agent_path: Path | str | None, trained_at: str | None) -> str:
    """A short, stable identifier for the exact artefact that decided.

    Content-addressed: the hash changes when the weights change, so two rows
    carrying the same version were genuinely produced by the same model. A
    manually-maintained semantic version cannot promise that.
    """
    stamp = (trained_at or "unknown")[:19]
    if agent_path:
        for suffix in (".pt", ".zip"):
            candidate = Path(str(agent_path)).with_suffix(suffix)
            if candidate.exists():
                digest = hashlib.sha256()
                with candidate.open("rb") as fh:
                    # Hash a bounded prefix: enough to distinguish artefacts
                    # without reading hundreds of megabytes on every call.
                    digest.update(fh.read(1_048_576))
                return f"{stamp}+{digest.hexdigest()[:12]}"
    return f"{stamp}+nohash"


async def log_rl_decision(db: AsyncSession, recommendation: dict) -> int | None:
    """Persist one RL decision. Returns the row id, or None if it could not."""
    try:
        explanation = recommendation.get("regime_explanation") or {}
        plan = recommendation.get("trade_plan") or recommendation
        risk_metrics = {
            k: recommendation.get(k) or plan.get(k)
            for k in ("position_size", "stop_loss", "take_profit",
                      "risk_reward_ratio", "expected_return")
            if (recommendation.get(k) is not None or plan.get(k) is not None)
        }
        for key in ("risk_level", "volatility", "var", "cvar"):
            if recommendation.get(key) is not None:
                risk_metrics[key] = recommendation[key]
        # Regime-derived risk context, so a reviewer sees the conditions the
        # decision was taken under without joining another table.
        for key in ("volatility_ratio", "crash_probability", "drawdown",
                    "risk_aversion_applied", "risk_level"):
            if explanation.get(key) is not None:
                risk_metrics.setdefault(key, explanation[key])

        row = RecommendationLog(
            symbol=str(recommendation.get("symbol", "?")).upper(),
            action=str(recommendation.get("action", "HOLD"))[:16],
            composite_score=float(recommendation.get("composite_score")
                                  or recommendation.get("confidence") or 0.0),
            confidence=float(recommendation.get("confidence") or 0.0),
            price_at_reco=float(recommendation.get("last_price") or 0.0),
            signals={"q_values": recommendation.get("q_values") or {},
                     "return_distribution": recommendation.get("return_distribution")},
            explanation={"summary": explanation.get("summary"),
                         "narrative": explanation.get("narrative")},
            source="rl_agent",
            model_version=recommendation.get("model_version"),
            algo=recommendation.get("algo"),
            regime=explanation.get("regime"),
            regime_confidence=explanation.get("regime_confidence"),
            regime_influence=explanation.get("influence"),
            risk_metrics=risk_metrics,
            regime_explanation=explanation,
        )
        db.add(row)
        await db.commit()
        await db.refresh(row)
        return int(row.id)
    except Exception as exc:      # pragma: no cover - never break the caller
        logger.warning("could not write decision audit row: %s", exc)
        with contextlib.suppress(Exception):
            await db.rollback()
        return None


async def log_allocation_decision(db: AsyncSession, allocation: dict) -> int | None:
    """Persist one multi-asset allocation decision.

    A basket has no single symbol and no BUY/SELL label, so the row records the
    portfolio key as the symbol and REBALANCE as the action. The weights and
    the per-asset regime attribution go into the JSON columns — flattening them
    into one score would lose exactly the detail a reviewer needs.
    """
    try:
        explanation = allocation.get("regime_explanation") or {}
        weights = {a["symbol"]: a["weight"] for a in allocation.get("allocation", [])}
        weights["CASH"] = allocation.get("cash_weight")

        largest = allocation.get("largest_position") or {}
        risk_metrics = {
            "cash_weight": allocation.get("cash_weight"),
            "largest_position": largest.get("symbol"),
            "largest_weight": largest.get("weight"),
            "capital_moved_by_regime": explanation.get("capital_moved"),
            "shift_type": explanation.get("shift_type"),
            "mean_regime_risk": explanation.get("mean_risk"),
            "all_assets_same_regime": explanation.get("all_assets_same_regime"),
        }

        regimes = explanation.get("regimes_in_force") or []
        row = RecommendationLog(
            symbol=str(allocation.get("portfolio_key", "?"))[:32],
            action="REBALANCE",
            composite_score=float(explanation.get("capital_moved") or 0.0),
            confidence=float(1.0 - (allocation.get("cash_weight") or 0.0)),
            price_at_reco=0.0,      # a basket has no single price
            signals={"weights": weights,
                     "per_asset": explanation.get("per_asset") or []},
            explanation={"summary": explanation.get("summary"),
                         "narrative": explanation.get("narrative")},
            source="rl_allocation",
            model_version=allocation.get("model_version"),
            algo=allocation.get("algo"),
            # Several regimes can be in force at once across a basket; join
            # them rather than pick one and imply the rest do not exist.
            regime=",".join(regimes)[:32] if regimes else None,
            regime_confidence=None,
            regime_influence=explanation.get("influence"),
            risk_metrics=risk_metrics,
            regime_explanation=explanation,
        )
        db.add(row)
        await db.commit()
        await db.refresh(row)
        return int(row.id)
    except Exception as exc:      # pragma: no cover - never break the caller
        logger.warning("could not write allocation audit row: %s", exc)
        with contextlib.suppress(Exception):
            await db.rollback()
        return None
