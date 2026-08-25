#!/usr/bin/env python3
"""Train a regime-aware twin of every existing agent, side by side.

Why
---
`regime_explain.py` answers "this agent was trained without regime awareness"
for 17 of the 21 shipped checkpoints, because they were trained before the
feature existed and the UI had no control for it. This produces the missing
half of the comparison **without discarding the baseline**: each twin is saved
under the `__regime` suffix, so `rl_AAPL_dqn.pt` and `rl_AAPL_dqn__regime.pt`
coexist and can be compared.

Fair comparison, not a fresh start
----------------------------------
Every twin reuses the *original* run's own settings, read back from its
metadata sidecar: period, episodes or total_timesteps, profile, and the basket
for portfolio agents. Nothing is normalised to a single episode count — an
agent trained for 8 episodes on 2y is retrained for 8 episodes on 2y, so the
only variable that changes is `regime_aware`.

Usage
-----
    python3 scripts/retrain_regime_aware.py --dry-run     # show the plan
    python3 scripts/retrain_regime_aware.py               # run everything
    python3 scripts/retrain_regime_aware.py --only AAPL_dqn AAPL_c51
    python3 scripts/retrain_regime_aware.py --discrete-only
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

VARIANT = "regime"


def _plan() -> list[dict]:
    """One entry per existing baseline agent, carrying its own settings."""
    from app.services.rl.service import rl_service

    jobs: list[dict] = []
    for meta_file in sorted(rl_service.model_dir.glob("rl_*.json")):
        try:
            meta = json.loads(meta_file.read_text())
        except Exception as exc:  # noqa: BLE001 - a corrupt sidecar must not abort the plan
            print(f"  ! unreadable metadata {meta_file.name}: {exc}")
            continue

        # Skip the twins themselves, or a rerun would train twins of twins.
        if meta.get("variant"):
            continue
        if VARIANT in meta_file.stem:
            continue

        algo = str(meta.get("algo") or "").lower()
        if not algo:
            continue

        env_cfg = meta.get("env_config") or {}
        job = {
            "name": meta_file.stem[3:],
            "algo": algo,
            "period": meta.get("period") or "2y",
            "profile": meta.get("profile") or "default",
            "episodes": meta.get("episodes"),
            "total_timesteps": meta.get("total_timesteps"),
            "symbol": meta.get("symbol"),
            "symbols": meta.get("symbols"),
            "was_regime_aware": env_cfg.get("regime_aware"),
            "baseline_return": (meta.get("test_performance") or {}).get("total_return"),
        }
        jobs.append(job)
    return jobs


def _run_one(job: dict) -> dict:
    from app.services.rl.service import rl_service

    kwargs = {
        "period": job["period"],
        "algo": job["algo"],
        "profile": job["profile"],
        "variant": VARIANT,
        "env_overrides": {"regime_aware": True},
    }
    if job["episodes"] is not None:
        kwargs["episodes"] = int(job["episodes"])
    if job["total_timesteps"] is not None:
        kwargs["total_timesteps"] = int(job["total_timesteps"])

    if job["symbols"]:
        return rl_service.train_portfolio(symbols=list(job["symbols"]), **kwargs)
    return rl_service.train_single_asset(symbol=job["symbol"], **kwargs)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    ap.add_argument("--only", nargs="*", help="restrict to these agent names")
    ap.add_argument("--discrete-only", action="store_true",
                    help="skip SB3 agents (the MoE only drives native discrete ones)")
    args = ap.parse_args()

    from app.services.rl.service import NATIVE_DISCRETE

    jobs = _plan()
    if args.only:
        wanted = {w.lower() for w in args.only}
        jobs = [j for j in jobs if j["name"].lower() in wanted]
    if args.discrete_only:
        jobs = [j for j in jobs if j["algo"] in NATIVE_DISCRETE]

    if not jobs:
        print("nothing to do")
        return 1

    print(f"{len(jobs)} agent(s) to twin under the '__{VARIANT}' suffix\n")
    print(f"{'AGENT':<26} {'ALGO':<12} {'PERIOD':<7} {'EP/TS':<9} {'PROFILE':<9} BASELINE")
    print("-" * 82)
    for j in jobs:
        budget = (f"{j['episodes']}ep" if j["episodes"] is not None
                  else f"{j['total_timesteps']}ts" if j["total_timesteps"] is not None
                  else "cfg")
        print(f"{j['name']:<26} {j['algo']:<12} {j['period']:<7} {budget:<9} "
              f"{j['profile']:<9} {j['baseline_return']}")
    if args.dry_run:
        print("\n--dry-run: nothing was trained")
        return 0

    print()
    results = []
    for i, j in enumerate(jobs, 1):
        t0 = time.time()
        print(f"[{i}/{len(jobs)}] {j['name']} … ", end="", flush=True)
        try:
            meta = _run_one(j)
            perf = (meta.get("test_performance") or {}).get("total_return")
            aware = meta.get("regime_aware")
            ok = aware is True
            print(f"{time.time() - t0:6.1f}s  return {perf}  regime_aware={aware}"
                  f"{'' if ok else '   << NOT REGIME-AWARE'}")
            results.append({**j, "twin_return": perf, "twin_regime_aware": aware,
                            "seconds": round(time.time() - t0, 1), "error": None})
        except Exception as exc:  # noqa: BLE001 - one bad agent must not stop the batch
            print(f"FAILED after {time.time() - t0:.1f}s: {str(exc)[:120]}")
            results.append({**j, "twin_return": None, "twin_regime_aware": None,
                            "seconds": round(time.time() - t0, 1),
                            "error": str(exc)[:300]})

    out = ROOT / "data" / "artifacts" / "regime_aware_retrain.json"
    out.write_text(json.dumps({"variant": VARIANT, "results": results}, indent=2))

    done = [r for r in results if r["twin_regime_aware"] is True]
    failed = [r for r in results if r["error"]]
    print(f"\n{len(done)}/{len(results)} twins trained and confirmed regime-aware")
    if failed:
        print(f"{len(failed)} failed:")
        for r in failed:
            print(f"  {r['name']}: {r['error'][:110]}")
    print(f"summary written to {out.relative_to(ROOT)}")
    return 0 if done else 1


if __name__ == "__main__":
    raise SystemExit(main())
