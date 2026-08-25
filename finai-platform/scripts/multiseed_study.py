#!/usr/bin/env python3
"""Multi-seed RL study: baseline vs regime-aware observation.

Why this script exists
----------------------
Every agent shipped in `data/models/rl/` was trained with the seed frozen at 42,
so the platform correctly refuses to report mean ± std:

    "1 distinct seed across 14 runs. Mean ± standard deviation needs at least
     3 independent seeds; below that the spread measures nothing."

A mini-project report that quotes confidence intervals therefore cannot reuse
those runs. This script produces the missing evidence by running the *same*
training entry point across several seeds, twice: once with the stock
observation, once with `regime_aware=True` (the modification under test).

It changes no training logic. The seed is varied the way the platform already
supports it — through a hyperparameter profile (`training.seed`) — and the
regime flag through the documented `env_overrides` channel.

Output: JSON to `data/artifacts/multiseed_<tag>.json`, plus a printed summary.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "backend"))

from app.services.rl.hyperparams import hyperparameters as H
from app.services.rl.service import rl_service

ARTIFACTS = pathlib.Path(__file__).resolve().parents[1] / "data" / "artifacts"


def ensure_seed_profiles(seeds: list[int], episodes: int) -> None:
    """One profile per seed. Additive merge: nothing else in the profile moves."""
    for s in seeds:
        H.save_profile(
            f"seed{s}",
            {"training": {"seed": s, "episodes": episodes}},
            description=f"Multi-seed protocol: seed {s}",
            merge=True,
        )


def run_one(symbol: str, algo: str, seed: int, period: str,
            episodes: int, regime_aware: bool) -> dict:
    """A single training run. Returns the held-out metrics plus provenance."""
    started = time.perf_counter()
    meta = rl_service.train_single_asset(
        symbol=symbol,
        algo=algo,
        period=period,
        episodes=episodes,
        profile=f"seed{seed}",
        env_overrides={"regime_aware": regime_aware},
    )
    wall = time.perf_counter() - started

    perf = meta.get("test_performance") or {}
    # Episode rewards are what makes a learning curve plottable. The first
    # version of this script dropped them, so the study had numbers but no
    # figures; keeping them costs a few KB per run.
    history = meta.get("training_history") or {}
    return {
        "episode_rewards": history.get("episode_rewards"),
        "train_window": meta.get("train_window"),
        "test_window": meta.get("test_window"),
        "train_bars": meta.get("train_bars"),
        "test_bars": meta.get("test_bars"),
        "baselines": meta.get("baselines"),
        "seed": seed,
        "regime_aware": regime_aware,
        "wall_seconds": round(wall, 2),
        "episodes": meta.get("episodes"),
        "profile": meta.get("profile"),
        "recorded_seed": (meta.get("hyperparameters") or {})
                         .get("training", {}).get("seed"),
        "fingerprint": meta.get("hyperparameter_fingerprint"),
        "total_return": perf.get("total_return"),
        "sharpe_ratio": perf.get("sharpe_ratio"),
        "sortino_ratio": perf.get("sortino_ratio"),
        "max_drawdown": perf.get("max_drawdown"),
        "alpha_vs_buy_hold": perf.get("alpha_vs_buy_hold"),
        "buy_hold_return": perf.get("buy_hold_return"),
    }


def summarise(runs: list[dict], key: str) -> dict:
    """Mean, sample std and a 95% t-interval. Reported as unavailable below 3
    seeds rather than filled with a plausible-looking number."""
    xs = [r[key] for r in runs if isinstance(r.get(key), (int, float))]
    n = len(xs)
    if n == 0:
        return {"n": 0, "available": False,
                "reason": f"{key} was not recorded by any run"}
    mean = statistics.fmean(xs)
    if n < 3:
        return {"n": n, "mean": mean, "available": False,
                "reason": "a 95% interval needs at least 3 independent seeds"}
    sd = statistics.stdev(xs)
    # two-sided 95% t critical values, small samples
    tcrit = {3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447,
             8: 2.365, 9: 2.306, 10: 2.262}.get(n, 1.96)
    half = tcrit * sd / (n ** 0.5)
    return {"n": n, "mean": mean, "std": sd, "available": True,
            "ci95_half_width": half,
            "ci95": [mean - half, mean + half],
            "min": min(xs), "max": max(xs), "values": xs}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="AAPL")
    ap.add_argument("--algo", default="dueling_dqn")
    ap.add_argument("--period", default="2y")
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--seeds", default="1,2,3,4,5")
    ap.add_argument("--arm", choices=("baseline", "regime", "both"), default="both")
    ap.add_argument("--algos", default=None,
                    help="comma-separated list; runs one artefact per algorithm")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    ensure_seed_profiles(seeds, args.episodes)

    # Batch mode: one artefact per algorithm, so a long grid can be run in
    # chunks and resumed. Each file has the same shape as a single-algo run.
    if args.algos:
        for algo in [a.strip() for a in args.algos.split(",") if a.strip()]:
            out = ARTIFACTS / f"multiseed_grid_{algo}.json"
            if out.exists():
                print(f"[skip] {algo}: {out.name} already exists")
                continue
            rows = []
            for seed in seeds:
                print(f"[{algo}] seed={seed} …", flush=True)
                try:
                    row = run_one(args.symbol, algo, seed, args.period,
                                  args.episodes, False)
                except Exception as exc:                  # noqa: BLE001
                    row = {"seed": seed, "algo": algo, "error": str(exc)}
                    print(f"    FAILED: {exc}", flush=True)
                else:
                    row["algo"] = algo
                    print(f"    return={row['total_return']} "
                          f"sharpe={row['sharpe_ratio']} ({row['wall_seconds']}s)",
                          flush=True)
                rows.append(row)
            ok = [r for r in rows if "error" not in r]
            payload = {
                "config": {"symbol": args.symbol, "algo": algo,
                           "period": args.period, "episodes": args.episodes,
                           "seeds": seeds},
                "runs": {"baseline": rows},
                "summary": {"baseline": {
                    m: summarise(ok, m) for m in
                    ("total_return", "sharpe_ratio", "sortino_ratio",
                     "max_drawdown", "alpha_vs_buy_hold", "wall_seconds")}},
            }
            ARTIFACTS.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(payload, indent=2))
            tr = payload["summary"]["baseline"]["total_return"]
            if tr.get("available"):
                print(f"  -> {algo}: {tr['mean']:+.4f} +/- {tr['ci95_half_width']:.4f}"
                      f" (n={tr['n']})\n")
            else:
                print(f"  -> {algo}: unavailable ({tr.get('reason')})\n")
        return 0

    arms = {"baseline": False, "regime": True}
    if args.arm != "both":
        arms = {args.arm: arms[args.arm]}

    results: dict[str, list[dict]] = {}
    for arm, flag in arms.items():
        results[arm] = []
        for seed in seeds:
            print(f"[{arm}] seed={seed} …", flush=True)
            try:
                row = run_one(args.symbol, args.algo, seed,
                              args.period, args.episodes, flag)
            except Exception as exc:                      # noqa: BLE001
                row = {"seed": seed, "regime_aware": flag, "error": str(exc)}
                print(f"    FAILED: {exc}", flush=True)
            else:
                print(f"    return={row['total_return']} sharpe={row['sharpe_ratio']}"
                      f" ({row['wall_seconds']}s)", flush=True)
            results[arm].append(row)

    payload = {
        "config": vars(args) | {"seeds": seeds},
        "runs": results,
        "summary": {
            arm: {m: summarise([r for r in rows if "error" not in r], m)
                  for m in ("total_return", "sharpe_ratio", "sortino_ratio",
                            "max_drawdown", "alpha_vs_buy_hold", "wall_seconds")}
            for arm, rows in results.items()
        },
    }

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    tag = args.tag or f"{args.symbol}_{args.algo}"
    out = ARTIFACTS / f"multiseed_{tag}.json"
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nwritten: {out}")

    for arm, mets in payload["summary"].items():
        tr = mets["total_return"]
        if tr.get("available"):
            print(f"  {arm:9} return {tr['mean']:+.4f} ± {tr['ci95_half_width']:.4f}"
                  f"  (n={tr['n']}, sd={tr['std']:.4f})")
        else:
            print(f"  {arm:9} return unavailable — {tr.get('reason')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
