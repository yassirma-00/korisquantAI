"""Periodic evaluation and checkpointing during RL training.

``eval_freq`` and ``checkpoint_interval`` existed in ``configs/`` but nothing
read them: they were recorded in every run's reproducibility block while the
training loops ran straight through to the end. A parameter that is stored and
ignored is worse than a missing one, because the record claims it was applied.

This wires them into all five loops (DQN, C51/IQN, Rainbow, SB3, native PPO)
through one object rather than five copies of the same bookkeeping.

Unit: episodes, everywhere
--------------------------
Three of the loops count episodes and two count timesteps, so "every 5" would
otherwise mean two different things depending on the algorithm — and on the
timestep loops it would mean *five environment steps*, evaluating hundreds of
times per episode. The SB3 callback therefore counts completed episodes from
the ``dones`` vector, so one number means one thing across the platform.

Checkpoints are periodic, not best-by-score
-------------------------------------------
Deliberate. The evaluation environment is the held-out test window, so keeping
"the checkpoint with the best eval score" would be selecting a model *on the
test set* and every figure reported afterwards would be optimistically biased.
The intermediate scores are a learning curve to look at, not a selection
criterion. ``best_checkpoint`` is reported for information and carries that
warning with it.
"""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from pathlib import Path

from app.core.logging import get_logger

logger = get_logger(__name__)

# Cap on retained intermediate checkpoints. Each SB3 model is a few MB and a
# 60-episode run at interval 1 would otherwise leave 60 of them per agent.
MAX_CHECKPOINTS = 5


def _tail_metrics(equity_curve, alpha: float = 0.05) -> dict:
    """VaR and CVaR of an evaluation run, derived from its equity curve.

    `env.performance()` reports volatility and drawdown but not the tail
    measures, while `evaluate()` already returns the equity curve both are
    computed from. Deriving them here keeps the training loop untouched.

    Returned as **negative** numbers, matching the platform's convention
    everywhere else (a VaR of -2.1% is a 2.1% loss), so a chart can plot them
    on the same axis as returns without a sign flip.
    """
    import numpy as np

    if not equity_curve or len(equity_curve) < 12:
        # Below ~12 points a 5% quantile is one observation. Reporting it
        # would dress a single bad day up as a tail estimate.
        return {"var_95": None, "cvar_95": None}
    equity = np.asarray(equity_curve, dtype=float)
    returns = np.diff(equity) / np.maximum(equity[:-1], 1e-9)
    if returns.size < 10:
        return {"var_95": None, "cvar_95": None}
    var = float(np.quantile(returns, alpha))
    tail = returns[returns <= var]
    cvar = float(tail.mean()) if tail.size else var
    return {"var_95": round(var, 6), "cvar_95": round(cvar, 6)}


class TrainingMonitor:
    """Runs periodic evaluations and writes checkpoints during training.

    Disabled by default (``eval_freq=0``, ``checkpoint_interval=0``), which is
    exactly the previous behaviour — enabling it is opt-in per profile.
    """

    def __init__(self, eval_env=None, eval_freq: int = 0,
                 checkpoint_interval: int = 0,
                 checkpoint_dir: Path | str | None = None,
                 run_id: str = "run", deterministic: bool = True,
                 max_checkpoints: int = MAX_CHECKPOINTS) -> None:
        self.eval_env = eval_env
        self.eval_freq = max(0, int(eval_freq or 0))
        self.checkpoint_interval = max(0, int(checkpoint_interval or 0))
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.run_id = run_id
        self.deterministic = bool(deterministic)
        self.max_checkpoints = max(1, int(max_checkpoints))

        self.evaluations: list[dict] = []
        self.checkpoints: list[dict] = []
        self._eval_seconds = 0.0
        self._warned_no_env = False

    # ----------------------------------------------------------- properties
    @property
    def evaluating(self) -> bool:
        return self.eval_freq > 0 and self.eval_env is not None

    @property
    def checkpointing(self) -> bool:
        return self.checkpoint_interval > 0 and self.checkpoint_dir is not None

    @property
    def active(self) -> bool:
        return self.evaluating or self.checkpointing

    # -------------------------------------------------------------- hooks
    def on_episode_end(self, episode: int, total_episodes: int, agent) -> None:
        """Call once per finished episode. ``episode`` is 1-based.

        ``total_episodes`` may be 0 on the timestep-driven paths, where the
        episode count is not known in advance. That must mean "unknown", not
        "already finished": treating it as a total made ``episode >= 0`` true
        immediately and silently skipped every evaluation and checkpoint on
        those runs, while still reporting monitoring as enabled.
        """
        is_last = total_episodes > 0 and episode >= total_episodes
        if self.eval_freq and (episode % self.eval_freq == 0) and not is_last:
            self._evaluate(episode, agent)
        if self.checkpoint_interval and (episode % self.checkpoint_interval == 0) \
                and not is_last:
            self._checkpoint(episode, agent)

    # ---------------------------------------------------------- evaluation
    def _evaluate(self, episode: int, agent) -> dict | None:
        if self.eval_env is None:
            if not self._warned_no_env:
                logger.info("eval_freq is set but no evaluation environment was "
                            "provided; skipping periodic evaluation")
                self._warned_no_env = True
            return None
        started = time.perf_counter()
        try:
            result = agent.evaluate(self.eval_env, deterministic=self.deterministic)
            perf = result.get("performance", {}) or {}
        except Exception as exc:      # pragma: no cover - never kill a run
            # A failed mid-training evaluation must not destroy hours of
            # training. Record the failure instead of raising.
            logger.warning("periodic evaluation failed at episode %d: %s", episode, exc)
            self.evaluations.append({"episode": episode, "error": str(exc)[:200]})
            return None
        elapsed = time.perf_counter() - started
        self._eval_seconds += elapsed

        entry = {
            "episode": int(episode),
            "total_return": perf.get("total_return"),
            "sharpe_ratio": perf.get("sharpe_ratio"),
            "max_drawdown": perf.get("max_drawdown"),
            "final_value": perf.get("final_value"),
            "seconds": round(elapsed, 3),
            # Sortino and annualised volatility are already computed by
            # `env.performance()` and were simply being dropped here.
            # PortfolioEnv does not report Sortino, hence the `.get`.
            "sortino_ratio": perf.get("sortino_ratio"),
            "annualised_volatility": perf.get("annualised_volatility"),
            # VaR/CVaR are not in `performance()`, but `evaluate()` already
            # returns the equity curve they are computed from. Deriving them
            # here is post-processing of data the evaluation produced anyway —
            # it does not touch the training loop or the agent.
            **_tail_metrics(result.get("equity_curve")),
        }
        self.evaluations.append(entry)
        logger.info("eval @ episode %d | return=%.2f%% sharpe=%.2f dd=%.2f%%",
                    episode, (perf.get("total_return") or 0) * 100,
                    perf.get("sharpe_ratio") or 0.0,
                    (perf.get("max_drawdown") or 0) * 100)
        return entry

    # --------------------------------------------------------- checkpoints
    def _checkpoint(self, episode: int, agent) -> dict | None:
        if self.checkpoint_dir is None:
            return None
        try:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            stem = self.checkpoint_dir / f"{self.run_id}_ep{episode:04d}"
            # Native agents write exactly the path they are handed, so passing
            # a bare stem produced extensionless files that no loader would
            # recognise. SB3 appends .zip itself and rejects a path that
            # already ends in .pt, so the suffix has to be chosen per backend.
            sb3_backed = hasattr(agent, "model") and hasattr(agent.model, "save")
            target = stem if sb3_backed else stem.with_suffix(".pt")
            agent.save(target)
            written = next((p for p in (stem.with_suffix(".pt"),
                                        stem.with_suffix(".zip"), stem)
                            if p.exists()), None)
            entry = {
                "episode": int(episode),
                "path": str(written) if written else str(stem),
                "bytes": written.stat().st_size if written else None,
                # Provenance, so the Checkpoint Manager can identify a file
                # without re-deriving it from the filename.
                "created_at": datetime.now(UTC).isoformat(),
                # Gradient/environment steps taken so far. Native agents track
                # `steps`; SB3 exposes `num_timesteps`. Neither is guaranteed,
                # hence the fallback to None rather than a fabricated 0.
                "training_step": (getattr(agent, "steps", None)
                                  or getattr(getattr(agent, "model", None),
                                             "num_timesteps", None)),
                "run_id": self.run_id,
            }
            self.checkpoints.append(entry)
            self._prune()
            logger.info("checkpoint @ episode %d -> %s", episode,
                        Path(entry["path"]).name)
            return entry
        except Exception as exc:      # pragma: no cover - never kill a run
            logger.warning("checkpoint failed at episode %d: %s", episode, exc)
            return None

    def _prune(self) -> None:
        """Keep only the most recent ``max_checkpoints`` files on disk."""
        while len(self.checkpoints) > self.max_checkpoints:
            oldest = self.checkpoints.pop(0)
            path = Path(oldest["path"])
            try:
                if path.exists():
                    path.unlink()
            except OSError as exc:    # pragma: no cover - filesystem dependent
                logger.debug("could not remove old checkpoint %s: %s", path, exc)

    # -------------------------------------------------------------- report
    def summary(self) -> dict:
        """What actually happened, for the training metadata."""
        scored = [e for e in self.evaluations if e.get("total_return") is not None]
        best = max(scored, key=lambda e: e["total_return"]) if scored else None
        return {
            "enabled": self.active,
            "unit": "episodes",
            "eval_freq": self.eval_freq,
            "checkpoint_interval": self.checkpoint_interval,
            "evaluations": self.evaluations,
            "checkpoints": self.checkpoints,
            "eval_seconds": round(self._eval_seconds, 2),
            "best_checkpoint": best,
            # Stated wherever the number is: a reader who picks the best
            # intermediate score has selected a model on the test window.
            "selection_note": (
                "Intermediate evaluations run on the held-out test window and are "
                "a learning curve, not a selection criterion. Checkpoints are "
                "saved on a fixed interval; choosing the best-scoring one would "
                "be selecting a model on the test set and would inflate every "
                "figure reported from it."),
        }

    def write_history(self, path: Path | str) -> None:
        """Persist the curve next to the model, for the UI to plot."""
        try:
            Path(path).write_text(json.dumps(self.summary(), indent=2, default=str))
        except OSError as exc:        # pragma: no cover - filesystem dependent
            logger.debug("could not write monitor history: %s", exc)


def make_monitor(cfg, eval_env=None, checkpoint_dir: Path | str | None = None,
                 run_id: str = "run") -> TrainingMonitor:
    """Build a monitor from a resolved hyperparameter config."""
    evaluation = cfg.section("evaluation") if hasattr(cfg, "section") else (cfg or {})
    return TrainingMonitor(
        eval_env=eval_env,
        eval_freq=int(evaluation.get("eval_freq", 0) or 0),
        checkpoint_interval=int(evaluation.get("checkpoint_interval", 0) or 0),
        checkpoint_dir=checkpoint_dir,
        run_id=run_id,
        deterministic=bool(evaluation.get("deterministic", True)),
    )


# ------------------------------------------------------------------- SB3
def make_sb3_callback(monitor: TrainingMonitor, agent, total_episodes: int):
    """Adapt the monitor to Stable-Baselines3's callback protocol.

    SB3 drives its own loop, so the hook has to be inverted. Episodes are
    counted from the ``dones`` vector rather than from timesteps, so
    ``eval_freq`` keeps meaning "every N episodes" here too.
    """
    try:
        from stable_baselines3.common.callbacks import BaseCallback
    except Exception:      # pragma: no cover - SB3 optional
        return None
    if not monitor.active:
        return None

    class _MonitorCallback(BaseCallback):
        def __init__(self) -> None:
            super().__init__(verbose=0)
            self.episodes = 0

        def _on_step(self) -> bool:
            dones = self.locals.get("dones")
            if dones is None:
                done = self.locals.get("done")
                dones = [done] if done is not None else []
            for finished in dones:
                if not finished:
                    continue
                self.episodes += 1
                monitor.on_episode_end(self.episodes, total_episodes, agent)
            return True      # never abort training from the monitor

    return _MonitorCallback()
