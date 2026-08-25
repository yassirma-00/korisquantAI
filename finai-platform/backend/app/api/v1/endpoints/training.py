"""Training monitoring: progress curves, checkpoint manager, experiment summary.

Read-only over what training already writes. Every figure here comes from an
agent's own metadata sidecar (`data/models/rl/*.json`) — nothing is recomputed
and no training logic is touched, so a number shown on this page is by
construction the number that run actually produced.

One deliberate exception is called out where it happens: VaR and CVaR are
derived from the equity curve the evaluation already returned, because
`env.performance()` never reported them. That is post-processing of recorded
data, not a second implementation of the metric.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Body, Query

from app.core.exceptions import InvalidRequestError
from app.core.logging import get_logger
from app.services.rl.service import rl_service

logger = get_logger(__name__)
router = APIRouter(prefix="/training", tags=["Training Monitoring"])


def _agent_key(meta: dict) -> str:
    return str(meta.get("symbol") or meta.get("portfolio_key") or "?")


def _find_run(symbol: str, algo: str) -> dict:
    """The metadata sidecar for one trained agent."""
    wanted_symbol, wanted_algo = symbol.upper().strip(), algo.lower().strip()
    for meta in rl_service.list_agents():
        if (_agent_key(meta).upper() == wanted_symbol
                and str(meta.get("algo", "")).lower() == wanted_algo):
            return meta
    raise InvalidRequestError(
        f"No trained {wanted_algo} agent for {wanted_symbol}.",
        details={"available": [
            {"symbol": _agent_key(m), "algo": m.get("algo")}
            for m in rl_service.list_agents()]})


@router.get("/runs", summary="Trained runs available for monitoring")
async def list_runs():
    """Every trained agent, newest first, with just enough to populate a picker."""
    runs = []
    for meta in rl_service.list_agents():
        monitoring = meta.get("monitoring") or {}
        history = meta.get("training_history") or {}
        evaluations = monitoring.get("evaluations") or []
        # How many distinct metrics this run actually recorded. Runs trained
        # before Sortino/VaR/CVaR were captured have evaluations but only four
        # series, and picking a default by evaluation *count* alone opened the
        # page on one of those — four metric tabs out of nine, which reads as a
        # broken page rather than an older run.
        metric_keys = {k for e in evaluations for k, v in e.items()
                       if v is not None and k not in ("episode", "seconds")}
        runs.append({
            "symbol": _agent_key(meta),
            "algo": meta.get("algo"),
            "experiment_id": meta.get("experiment_id"),
            "profile": meta.get("profile"),
            "trained_at": meta.get("trained_at"),
            "episodes": len(history.get("episode_rewards") or []),
            "evaluations": len(evaluations),
            "checkpoints": len(monitoring.get("checkpoints") or []),
            "metrics_recorded": len(metric_keys),
            "monitoring_enabled": bool(monitoring.get("enabled")),
            "stale": bool(meta.get("stale")),
        })
    runs.sort(key=lambda r: str(r.get("trained_at") or ""), reverse=True)
    return {"count": len(runs), "runs": runs}


@router.get("/progress/{symbol}", summary="Training and evaluation curves")
async def progress(symbol: str, algo: str = Query("dueling_dqn")):
    """Per-episode training curves plus the periodic evaluation series.

    The two are returned as separate series on purpose. Training reward is
    measured on the *training* window and evaluation reward on the held-out
    one; plotting them as a single line would imply a continuity that does not
    exist, and would hide the gap between them — which is exactly the signal
    worth watching for overfitting.
    """
    meta = _find_run(symbol, algo)
    history = meta.get("training_history") or {}
    monitoring = meta.get("monitoring") or {}
    evaluations = monitoring.get("evaluations") or []

    rewards = history.get("episode_rewards") or []
    training_series = [
        {
            "episode": i + 1,
            "reward": rewards[i] if i < len(rewards) else None,
            "loss": (history.get("losses") or [None] * len(rewards))[i]
            if i < len(history.get("losses") or []) else None,
            "sharpe_ratio": (history.get("sharpe") or [None] * len(rewards))[i]
            if i < len(history.get("sharpe") or []) else None,
            "portfolio_value": (history.get("final_values") or [None] * len(rewards))[i]
            if i < len(history.get("final_values") or []) else None,
        }
        for i in range(len(rewards))
    ]

    # Only the metrics that were actually recorded are advertised. Claiming a
    # series the backend never produced would render an empty chart with no
    # explanation of why.
    available = {
        "reward": bool(rewards),
        "loss": any(p["loss"] is not None for p in training_series),
        "sharpe_ratio": any(p["sharpe_ratio"] is not None for p in training_series),
        "portfolio_value": any(p["portfolio_value"] is not None for p in training_series),
    }
    for key in ("total_return", "sharpe_ratio", "sortino_ratio", "max_drawdown",
                "annualised_volatility", "var_95", "cvar_95", "final_value"):
        available[f"eval_{key}"] = any(
            e.get(key) is not None for e in evaluations)

    return {
        "symbol": _agent_key(meta),
        "algo": meta.get("algo"),
        "experiment_id": meta.get("experiment_id"),
        "profile": meta.get("profile"),
        "seed": meta.get("seed"),
        "episodes": len(training_series),
        "training": training_series,
        "evaluations": evaluations,
        "checkpoints": monitoring.get("checkpoints") or [],
        "eval_freq": monitoring.get("eval_freq", 0),
        "checkpoint_interval": monitoring.get("checkpoint_interval", 0),
        "monitoring_enabled": bool(monitoring.get("enabled")),
        "available_series": available,
        "note": (
            "Training reward is measured on the training window; evaluation "
            "metrics on the held-out test window. A widening gap between them "
            "is the overfitting signal."
            if evaluations else
            "No periodic evaluations were recorded for this run: eval_freq was "
            "0. Set it in a hyperparameter profile and retrain to populate the "
            "evaluation series."),
        "selection_note": monitoring.get("selection_note"),
    }


@router.get("/checkpoints", summary="Checkpoint manager")
async def list_checkpoints(symbol: str | None = Query(None),
                           algo: str | None = Query(None)):
    """Every checkpoint on disk, with the run that produced it.

    `exists` is checked rather than assumed: the retention policy prunes older
    files, so a recorded checkpoint may legitimately be gone. Listing it as
    restorable when the file has been deleted would fail only at restore time.
    """
    out = []
    for meta in rl_service.list_agents():
        key, meta_algo = _agent_key(meta), str(meta.get("algo") or "")
        if symbol and key.upper() != symbol.upper():
            continue
        if algo and meta_algo.lower() != algo.lower():
            continue
        monitoring = meta.get("monitoring") or {}
        for entry in monitoring.get("checkpoints") or []:
            path = Path(entry.get("path", ""))
            out.append({
                # Checkpoints written before `created_at`/`training_step` were
                # recorded simply lack those keys. Defaulting them to None here
                # keeps one stable response shape for the UI; back-filling the
                # sidecars with invented timestamps would corrupt the record.
                "created_at": None,
                "training_step": None,
                "run_id": None,
                **entry,
                "filename": path.name,
                "exists": path.exists(),
                "symbol": key,
                "algo": meta_algo,
                "experiment_id": meta.get("experiment_id"),
                "model_version": meta.get("hyperparameter_fingerprint"),
                "seed": meta.get("seed"),
                "profile": meta.get("profile"),
                "trained_at": meta.get("trained_at"),
            })
    out.sort(key=lambda c: str(c.get("created_at") or ""), reverse=True)
    return {
        "count": len(out),
        "checkpoints": out,
        "retention": {
            "max_checkpoints": _max_checkpoints(),
            "note": ("Only the most recent checkpoints of each run are kept on "
                     "disk; older entries are pruned automatically and are "
                     "listed here with exists=false."),
        },
    }


def _max_checkpoints() -> int:
    from app.services.rl.monitor import MAX_CHECKPOINTS

    return MAX_CHECKPOINTS


@router.post("/checkpoints/compare", summary="Compare two checkpoints")
async def compare_checkpoints(payload: dict = Body(...)):
    """Side-by-side diff of two checkpoints and their nearest evaluations.

    A checkpoint is only scored if an evaluation landed on the *same* episode.
    Interpolating from a neighbouring one would invent a number, and the whole
    point of this panel is that every figure is one the run really recorded.
    """
    left, right = payload.get("left") or {}, payload.get("right") or {}
    if not left or not right:
        raise InvalidRequestError("Two checkpoints are required to compare.")

    def describe(ref: dict) -> dict:
        meta = _find_run(ref.get("symbol", ""), ref.get("algo", ""))
        monitoring = meta.get("monitoring") or {}
        episode = int(ref.get("episode", 0))
        checkpoint = next((c for c in monitoring.get("checkpoints") or []
                           if int(c.get("episode", -1)) == episode), None)
        evaluation = next((e for e in monitoring.get("evaluations") or []
                           if int(e.get("episode", -1)) == episode), None)
        return {
            "symbol": _agent_key(meta),
            "algo": meta.get("algo"),
            "episode": episode,
            "checkpoint": checkpoint,
            "evaluation": evaluation,
            "evaluation_available": evaluation is not None,
            "experiment_id": meta.get("experiment_id"),
            "model_version": meta.get("hyperparameter_fingerprint"),
            "seed": meta.get("seed"),
            "profile": meta.get("profile"),
            "hyperparameters": meta.get("hyperparameters", {}),
        }

    a, b = describe(left), describe(right)

    metrics: list[dict] = []
    if a["evaluation"] and b["evaluation"]:
        for key in ("total_return", "sharpe_ratio", "sortino_ratio",
                    "max_drawdown", "annualised_volatility", "var_95",
                    "cvar_95", "final_value"):
            left_value, right_value = a["evaluation"].get(key), b["evaluation"].get(key)
            if left_value is None and right_value is None:
                continue
            delta = (None if left_value is None or right_value is None
                     else round(right_value - left_value, 6))
            metrics.append({"metric": key, "left": left_value,
                            "right": right_value, "delta": delta})

    # Which hyperparameters actually differ, so a comparison across profiles
    # says *why* the two runs are not alike.
    differences: list[dict] = []
    for section, values in (a["hyperparameters"] or {}).items():
        if section == "meta" or not isinstance(values, dict):
            continue
        other = (b["hyperparameters"] or {}).get(section) or {}
        for key, value in values.items():
            if other.get(key) != value:
                differences.append({"parameter": f"{section}.{key}",
                                    "left": value, "right": other.get(key)})

    return {
        "left": {k: v for k, v in a.items() if k != "hyperparameters"},
        "right": {k: v for k, v in b.items() if k != "hyperparameters"},
        "metrics": metrics,
        "hyperparameter_differences": differences,
        "comparable": bool(metrics),
        "note": (
            "Both checkpoints have an evaluation on their own episode."
            if metrics else
            "At least one checkpoint has no evaluation on the same episode, so "
            "there is nothing to compare on. Align eval_freq with "
            "checkpoint_interval to score every checkpoint."),
    }


@router.post("/checkpoints/restore", summary="Promote a checkpoint to the live agent")
async def restore_checkpoint(payload: dict = Body(...)):
    """Copy a checkpoint over the active agent file.

    The current agent is backed up first. Restoring is destructive — it
    overwrites the model every prediction endpoint loads — and a restore that
    turns out to be wrong with no way back would be worse than no restore at
    all.
    """
    import shutil

    symbol = str(payload.get("symbol") or "")
    algo = str(payload.get("algo") or "")
    episode = int(payload.get("episode") or 0)

    meta = _find_run(symbol, algo)
    monitoring = meta.get("monitoring") or {}
    entry = next((c for c in monitoring.get("checkpoints") or []
                  if int(c.get("episode", -1)) == episode), None)
    if entry is None:
        raise InvalidRequestError(
            f"No checkpoint at episode {episode} for {symbol}/{algo}.")

    source = Path(entry["path"])
    if not source.exists():
        raise InvalidRequestError(
            f"The checkpoint file is gone: {source.name}. Retention keeps only "
            f"the {_max_checkpoints()} most recent per run.")

    target = rl_service.agent_path(_agent_key(meta), str(meta.get("algo")))
    target = target.with_suffix(source.suffix)
    backup = target.with_suffix(source.suffix + ".bak")
    if target.exists():
        shutil.copy2(target, backup)
    shutil.copy2(source, target)
    logger.info("checkpoint restored: %s -> %s", source.name, target.name)

    return {
        "restored": True,
        "symbol": _agent_key(meta),
        "algo": meta.get("algo"),
        "episode": episode,
        "from": source.name,
        "to": target.name,
        "backup": backup.name if backup.exists() else None,
        "warning": ("The live agent now holds a mid-training snapshot. Its "
                    "recorded test performance was measured on the final "
                    "model, so it no longer describes what is loaded."),
    }


@router.delete("/checkpoints", summary="Delete a checkpoint file")
async def delete_checkpoint(symbol: str = Query(...), algo: str = Query(...),
                            episode: int = Query(...)):
    """Remove one checkpoint from disk.

    The metadata entry is kept and marked `exists: false` rather than rewritten:
    the run's record is evidence of what happened during training, and editing
    it to hide a deleted file would make the reproducibility block a partial
    account of the run.
    """
    meta = _find_run(symbol, algo)
    monitoring = meta.get("monitoring") or {}
    entry = next((c for c in monitoring.get("checkpoints") or []
                  if int(c.get("episode", -1)) == episode), None)
    if entry is None:
        raise InvalidRequestError(
            f"No checkpoint at episode {episode} for {symbol}/{algo}.")

    path = Path(entry["path"])
    existed = path.exists()
    if existed:
        path.unlink()
        logger.info("checkpoint deleted: %s", path.name)
    return {
        "deleted": existed,
        "filename": path.name,
        "episode": episode,
        "note": ("The training record still lists this checkpoint, now with "
                 "exists=false. The run's history is not rewritten."),
    }


@router.get("/summary/{symbol}", summary="Experiment summary card")
async def summary(symbol: str, algo: str = Query("dueling_dqn")):
    """Headline numbers for one run."""
    meta = _find_run(symbol, algo)
    monitoring = meta.get("monitoring") or {}
    history = meta.get("training_history") or {}
    evaluations = monitoring.get("evaluations") or []
    scored = [e for e in evaluations if e.get("total_return") is not None]

    best = max(scored, key=lambda e: e["total_return"]) if scored else None
    latest = scored[-1] if scored else None
    checkpoints = monitoring.get("checkpoints") or []

    # Elapsed time is not recorded as a duration anywhere, so it is not
    # invented. What *is* known is how long the evaluations took.
    return {
        "symbol": _agent_key(meta),
        "algo": meta.get("algo"),
        "algorithm_family": (meta.get("hyperparameters", {})
                             .get("meta", {}).get("family")),
        "experiment_id": meta.get("experiment_id"),
        "profile": meta.get("profile"),
        "seed": meta.get("seed"),
        "model_version": meta.get("hyperparameter_fingerprint"),
        "trained_at": meta.get("trained_at"),
        "total_episodes": len(history.get("episode_rewards") or []),
        "train_bars": meta.get("train_bars"),
        "test_bars": meta.get("test_bars"),
        "eval_freq": monitoring.get("eval_freq", 0),
        "checkpoint_interval": monitoring.get("checkpoint_interval", 0),
        "n_evaluations": len(evaluations),
        "n_checkpoints": len(checkpoints),
        "checkpoints_on_disk": sum(1 for c in checkpoints
                                   if Path(c.get("path", "")).exists()),
        "best_evaluation": best,
        "latest_evaluation": latest,
        "eval_seconds": monitoring.get("eval_seconds"),
        "final_performance": meta.get("test_performance", {}),
        "baselines": meta.get("baselines", {}),
        "monitoring_enabled": bool(monitoring.get("enabled")),
        "selection_note": monitoring.get("selection_note"),
    }


# ==================================================== training intelligence
@router.get("/intelligence", summary="All runs with diagnosis, health and ranking")
async def intelligence(
    symbol: str | None = Query(None, description="Filter by instrument"),
    algo: str | None = Query(None, description="Filter by algorithm"),
    status: str | None = Query(None, description="converged | improving | plateaued | overfitting | unstable"),
    search: str | None = Query(None, description="Free text over symbol and algorithm"),
):
    """Every trained run, diagnosed and scored, plus global aggregates.

    One request feeds the whole dashboard: splitting it per panel would let the
    leaderboard and the table disagree while one was still loading.
    """
    from app.services.rl.intelligence import analyse_run, seed_statistics

    runs = [analyse_run(meta) for meta in rl_service.list_agents()]

    def keep(run: dict) -> bool:
        if symbol and str(run.get("symbol", "")).upper() != symbol.upper():
            return False
        if algo and str(run.get("algo", "")).lower() != algo.lower():
            return False
        if status and run.get("status") != status:
            return False
        if search:
            needle = search.lower()
            haystack = f"{run.get('symbol')} {run.get('algo')} {run.get('profile')}".lower()
            if needle not in haystack:
                return False
        return True

    filtered = [r for r in runs if keep(r)]
    filtered.sort(key=lambda r: str(r.get("trained_at") or ""), reverse=True)

    scored = [r for r in filtered if (r["health"] or {}).get("score") is not None]

    # Per-symbol leaderboard: best algorithm for each instrument.
    by_symbol: dict[str, list[dict]] = {}
    for run in scored:
        by_symbol.setdefault(str(run["symbol"]), []).append(run)
    leaderboard = []
    for sym, entries in sorted(by_symbol.items()):
        ranked = sorted(entries, key=lambda r: r["health"]["score"], reverse=True)
        leaderboard.append({
            "symbol": sym,
            "best": {"algo": ranked[0]["algo"],
                     "health": ranked[0]["health"]["percent"],
                     "total_return": ranked[0]["metrics"]["total_return"],
                     "status": ranked[0]["status_label"]},
            "ranking": [{"rank": i + 1, "algo": r["algo"],
                         "health": r["health"]["percent"],
                         "total_return": r["metrics"]["total_return"],
                         "sharpe_ratio": r["metrics"]["sharpe_ratio"]}
                        for i, r in enumerate(ranked)],
        })

    overall = sorted(scored, key=lambda r: r["health"]["score"], reverse=True)

    # Legacy vs adaptive. Reported as a raw comparison, not a verdict: the two
    # groups differ in more than the regime flag (different symbols, episode
    # counts and dates), so this is a starting point for a controlled
    # experiment rather than evidence on its own.
    adaptive = [r for r in scored if r["regime_aware"]]
    legacy = [r for r in scored if not r["regime_aware"]]

    def _summarise(group: list[dict]) -> dict:
        if not group:
            return {"runs": 0}
        health = [r["health"]["score"] for r in group]
        returns = [r["metrics"]["total_return"] for r in group
                   if r["metrics"]["total_return"] is not None]
        return {
            "runs": len(group),
            "mean_health": round(sum(health) / len(health) * 100, 1),
            "mean_return": (round(sum(returns) / len(returns), 4)
                            if returns else None),
        }

    status_counts: dict[str, int] = {}
    for run in filtered:
        status_counts[run["status"]] = status_counts.get(run["status"], 0) + 1

    return {
        "count": len(filtered),
        "total_runs": len(runs),
        "runs": filtered,
        "filters": {"symbol": symbol, "algo": algo, "status": status,
                    "search": search},
        "facets": {
            "symbols": sorted({str(r["symbol"]) for r in runs}),
            "algorithms": sorted({str(r["algo"]) for r in runs}),
            "statuses": sorted({r["status"] for r in runs}),
        },
        "leaderboard": leaderboard,
        "overall_ranking": [
            {"rank": i + 1, "symbol": r["symbol"], "algo": r["algo"],
             "health": r["health"]["percent"], "grade": r["health"]["grade"],
             "total_return": r["metrics"]["total_return"],
             "status": r["status_label"]}
            for i, r in enumerate(overall[:20])
        ],
        "global": {
            "trained_models": len(runs),
            "symbols_covered": len({str(r["symbol"]) for r in runs}),
            "algorithms_used": len({str(r["algo"]) for r in runs}),
            "mean_health": (round(sum(r["health"]["score"] for r in scored)
                                  / len(scored) * 100, 1) if scored else None),
            "status_counts": status_counts,
            "needs_attention": [
                {"symbol": r["symbol"], "algo": r["algo"],
                 "status": r["status_label"],
                 "action": r["recommendation"]["action"]}
                for r in filtered
                if r["status"] in ("overfitting", "unstable")
            ],
        },
        "adaptive_vs_legacy": {
            "adaptive": _summarise(adaptive),
            "legacy": _summarise(legacy),
            "note": ("Grouped by the regime_aware flag. The two groups also "
                     "differ in symbol, episode count and training date, so "
                     "this is a starting point for a controlled comparison, "
                     "not evidence on its own."),
        },
        "seed_statistics": seed_statistics(rl_service.list_agents()),
    }


@router.get("/report/{symbol}", summary="Downloadable training report")
async def report(symbol: str, algo: str = Query("dueling_dqn")):
    """A self-contained Markdown report for one training session."""
    from app.services.rl.intelligence import analyse_run

    meta = _find_run(symbol, algo)
    run = analyse_run(meta)
    metrics, health = run["metrics"], run["health"]

    def pct(value) -> str:
        return "—" if value is None else f"{value * 100:.2f}%"

    def num(value, digits: int = 3) -> str:
        return "—" if value is None else f"{value:.{digits}f}"

    lines = [
        f"# Training Report — {run['symbol']} / {str(run['algo']).upper()}",
        "",
        f"- **Experiment:** `{run['experiment_id'] or 'not recorded'}`",
        f"- **Model version:** `{run['model_version'] or 'n/a'}`",
        f"- **Profile:** {run['profile'] or 'n/a'}",
        f"- **Seed:** {run['seed'] if run['seed'] is not None else 'not recorded'}",
        f"- **Trained at:** {run['trained_at']}",
        f"- **Regime-aware:** {'yes' if run['regime_aware'] else 'no'}",
        "",
        "## Diagnosis",
        "",
        f"**{run['status_label']}** (confidence {run['status_confidence']:.0%})",
        "",
        *[f"- {item}" for item in run["evidence"]],
        "",
        f"**Recommended action — {run['recommendation']['action']}.** "
        f"{run['recommendation']['rationale']}",
        "",
        "## Health score",
        "",
        f"**{health.get('percent', '—')} / 100 ({health.get('grade', 'unknown')})**",
        "",
        "| Dimension | Points | Max | Detail |",
        "|---|---:|---:|---|",
        *[f"| {c['name']} | {'—' if c['points'] is None else c['points']} | "
          f"{c['max_points']} | {c['detail']} |" for c in health.get("contributions", [])],
        "",
        "## Performance (held-out test window)",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Total return | {pct(metrics['total_return'])} |",
        f"| Annualised return | {pct(metrics['annualised_return'])} |",
        f"| Alpha vs benchmark | {pct(metrics['alpha_vs_benchmark'])} |",
        f"| Sharpe ratio | {num(metrics['sharpe_ratio'], 2)} |",
        f"| Sortino ratio | {num(metrics['sortino_ratio'], 2)} |",
        f"| Max drawdown | {pct(metrics['max_drawdown'])} |",
        f"| Annualised volatility | {pct(metrics['annualised_volatility'])} |",
        f"| VaR 95% (last evaluation) | {pct(metrics['var_95'])} |",
        f"| CVaR 95% (last evaluation) | {pct(metrics['cvar_95'])} |",
        f"| Turnover | {num(metrics['turnover'], 2)}x capital |",
        f"| Trades | {metrics['n_trades'] if metrics['n_trades'] is not None else '—'} |",
        f"| Episode win rate | {pct(metrics['episode_win_rate'])} |",
        f"| Cumulative reward | {num(metrics['cumulative_reward'], 1)} |",
        "",
        "## Not recorded",
        "",
        f"- **Training duration:** {metrics['training_duration_basis']}",
        f"- **Win rate basis:** {metrics['win_rate_basis']}",
        f"- **Turnover basis:** {metrics['turnover_basis']}",
        "",
        f"_{run['evaluation']['count']} evaluation(s), "
        f"{run['checkpoints']} checkpoint(s)._",
        "",
        "> Educational and research software. Past performance measured on a "
        "held-out window is not evidence of future results.",
    ]
    from fastapi.responses import PlainTextResponse

    filename = f"training_report_{run['symbol']}_{run['algo']}.md".replace(",", "-")
    return PlainTextResponse(
        "\n".join(lines), media_type="text/markdown",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'})
