#!/usr/bin/env python3
"""Train every catalogue algorithm on every catalogue symbol, regime-aware.

Scope
-----
32 symbols x 13 algorithms = 416 runs, each saved under the ``__regime``
suffix so the existing baseline checkpoints are never touched.

Per-combination settings
------------------------
Hyperparameters are **not** invented. For each algorithm the reference run is
the existing baseline agent for that algorithm (all 13 exist for AAPL), and its
own recorded settings are reused verbatim: period, episodes or total_timesteps,
and profile. A symbol that already has its own baseline for that algorithm
takes precedence over the reference, so `MSFT/dueling_dqn` keeps its 3 episodes
rather than inheriting AAPL's 8.

Resumable
---------
State lives in ``data/artifacts/train_all_regime.json``. A run that is already
recorded as done, or whose checkpoint exists on disk, is skipped. This matters:
the full sweep was measured at roughly ten hours, far longer than any single
shell invocation survives.

Usage
-----
    python3 scripts/train_all_regime_aware.py --plan          # list, train nothing
    python3 scripts/train_all_regime_aware.py --budget 1500   # train for 25 min
    python3 scripts/train_all_regime_aware.py --verify        # audit checkpoints
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
STATE = ROOT / "data" / "artifacts" / "train_all_regime.json"


def _reference_settings() -> dict[str, dict]:
    """Original settings per algorithm, read from the shipped baselines."""
    from app.services.rl.service import rl_service

    ref: dict[str, dict] = {}
    per_symbol: dict[tuple[str, str], dict] = {}
    for meta_file in sorted(rl_service.model_dir.glob("rl_*.json")):
        if VARIANT in meta_file.stem:
            continue
        try:
            meta = json.loads(meta_file.read_text())
        except Exception as exc:  # noqa: BLE001 - a corrupt sidecar must not abort
            print(f"  ! unreadable {meta_file.name}: {exc}")
            continue
        if meta.get("symbols"):          # basket agents are a different shape
            continue
        algo = str(meta.get("algo") or "").lower()
        symbol = str(meta.get("symbol") or "").upper()
        if not algo or not symbol:
            continue
        settings = {
            "period": meta.get("period") or "2y",
            "episodes": meta.get("episodes"),
            "total_timesteps": meta.get("total_timesteps"),
            "profile": meta.get("profile") or "default",
            "source": f"{symbol}/{algo}",
        }
        per_symbol[(symbol, algo)] = settings
        ref.setdefault(algo, settings)
    return {"by_algo": ref, "by_pair": per_symbol}


def build_plan() -> tuple[list[dict], list[dict]]:
    """Every symbol x algorithm pair, plus the ones that cannot be planned."""
    from app.services.data.universe import UNIVERSE
    from app.services.rl.catalogue import BY_KEY

    refs = _reference_settings()
    plan: list[dict] = []
    skipped: list[dict] = []

    for inst in UNIVERSE:
        for algo, spec in BY_KEY.items():
            if not spec.available:
                skipped.append({"symbol": inst.symbol, "algo": algo,
                                "reason": f"backend '{spec.backend}' not installed"})
                continue
            settings = (refs["by_pair"].get((inst.symbol.upper(), algo))
                        or refs["by_algo"].get(algo))
            if settings is None:
                skipped.append({"symbol": inst.symbol, "algo": algo,
                                "reason": "no baseline run exists for this algorithm, "
                                          "so its original settings are unknown"})
                continue
            plan.append({
                "symbol": inst.symbol, "algo": algo,
                "asset_class": inst.asset_class,
                **{k: v for k, v in settings.items()},
            })
    return plan, skipped


def _load_state() -> dict:
    if STATE.exists():
        try:
            return json.loads(STATE.read_text())
        except Exception as exc:  # noqa: BLE001 - a corrupt state file must not
            # block a ten-hour sweep; restarting the ledger only re-checks disk.
            print(f'  ! unreadable state file, starting a fresh ledger: {exc}')
    return {"done": {}, "failed": {}}


def _save_state(state: dict) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(state, indent=2, default=str))


def _key(job: dict) -> str:
    return f"{job['symbol']}|{job['algo']}"


def checkpoint_exists(symbol: str, algo: str) -> bool:
    from app.services.rl.service import NATIVE_DISCRETE, rl_service

    base = rl_service.agent_path(symbol, algo, VARIANT)
    suffix = ".pt" if algo in NATIVE_DISCRETE else ".zip"
    return base.with_suffix(suffix).exists() and \
        rl_service.meta_path(symbol, algo, VARIANT).exists()


def run_one(job: dict) -> dict:
    from app.services.rl.service import rl_service

    kwargs = {
        "symbol": job["symbol"], "algo": job["algo"], "period": job["period"],
        "profile": job["profile"], "variant": VARIANT,
        "env_overrides": {"regime_aware": True},
    }
    if job.get("episodes") is not None:
        kwargs["episodes"] = int(job["episodes"])
    if job.get("total_timesteps") is not None:
        kwargs["total_timesteps"] = int(job["total_timesteps"])
    return rl_service.train_single_asset(**kwargs)


def cmd_plan(plan: list[dict], skipped: list[dict]) -> int:
    from collections import Counter

    print(f"PLAN — {len(plan)} combinaisons symbole x algorithme\n")
    print(f"{'SYMBOL':<10} {'CLASS':<10} {'ALGO':<12} {'PERIOD':<7} "
          f"{'BUDGET':<10} {'PROFILE':<9} SETTINGS FROM")
    print("-" * 92)
    for j in plan:
        budget = (f"{j['episodes']} ep" if j.get("episodes") is not None
                  else f"{j['total_timesteps']} ts" if j.get("total_timesteps") is not None
                  else "config")
        print(f"{j['symbol']:<10} {j['asset_class']:<10} {j['algo']:<12} "
              f"{j['period']:<7} {budget:<10} {j['profile']:<9} {j['source']}")

    print(f"\nsymbols  : {len(Counter(j['symbol'] for j in plan))}")
    print(f"algos    : {len(Counter(j['algo'] for j in plan))}")
    print(f"total    : {len(plan)}")
    if skipped:
        print(f"\nSKIPPED ({len(skipped)}) — each with a stated reason:")
        for s in skipped:
            print(f"  {s['symbol']:<10} {s['algo']:<12} {s['reason']}")
    else:
        print("\nSKIPPED: none — every symbol x algorithm pair is planned.")
    return 0


def cmd_verify(plan: list[dict]) -> int:
    """Does every planned combination have a usable checkpoint?"""
    from app.services.rl.service import rl_service

    missing, bad, ok = [], [], 0
    for job in plan:
        symbol, algo = job["symbol"], job["algo"]
        if not checkpoint_exists(symbol, algo):
            missing.append(f"{symbol}/{algo}")
            continue
        try:
            meta = json.loads(rl_service.meta_path(symbol, algo, VARIANT).read_text())
        except Exception as exc:  # noqa: BLE001 - report, do not abort the audit
            bad.append(f"{symbol}/{algo}: unreadable metadata ({exc})")
            continue
        if (meta.get("env_config") or {}).get("regime_aware") is not True:
            bad.append(f"{symbol}/{algo}: checkpoint is not regime-aware")
            continue
        # Existence is not validity. A `.PA` ticker once collapsed six
        # checkpoints onto one filename and this audit still called them all
        # valid, because it never opened them. Load the policy for real.
        try:
            from app.services.data.market_data import market_data_service
            from app.services.rl.environment import TradingEnv

            df = market_data_service.get_history(symbol, period="1y").df
            cfg = rl_service._env_config_for_agent(symbol, algo, None, VARIANT)
            env = TradingEnv(df, cfg)
            rl_service.load_agent(symbol, algo, env, VARIANT)
            width = env.observation_space.shape[0]
        except Exception as exc:  # noqa: BLE001 - report every failure, audit all
            bad.append(f"{symbol}/{algo}: will not load ({str(exc)[:80]})")
            continue
        saved_width = meta.get("obs_dim") or meta.get("observation_dim")
        if saved_width and int(saved_width) != width:
            bad.append(f"{symbol}/{algo}: env is {width}-wide, checkpoint {saved_width}")
            continue
        ok += 1

    print(f"checkpoints valides : {ok}/{len(plan)}")
    if missing:
        print(f"manquants ({len(missing)}): {', '.join(missing[:20])}"
              + (" …" if len(missing) > 20 else ""))
    if bad:
        print(f"invalides ({len(bad)}):")
        for b in bad[:20]:
            print("  ", b)
    return 0 if (not missing and not bad) else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", action="store_true", help="print the plan and exit")
    ap.add_argument("--verify", action="store_true", help="audit the checkpoints and exit")
    ap.add_argument("--budget", type=float, default=1500.0,
                    help="stop starting new runs after this many seconds")
    ap.add_argument("--only-symbol", nargs="*", help="restrict to these symbols")
    ap.add_argument("--only-algo", nargs="*", help="restrict to these algorithms")
    args = ap.parse_args()

    plan, skipped = build_plan()
    if args.only_symbol:
        wanted = {s.upper() for s in args.only_symbol}
        plan = [j for j in plan if j["symbol"].upper() in wanted]
    if args.only_algo:
        wanted = {a.lower() for a in args.only_algo}
        plan = [j for j in plan if j["algo"] in wanted]

    if args.plan:
        return cmd_plan(plan, skipped)
    if args.verify:
        return cmd_verify(plan)

    state = _load_state()
    todo = [j for j in plan
            if _key(j) not in state["done"] and not checkpoint_exists(j["symbol"], j["algo"])]
    print(f"{len(plan)} planned · {len(plan) - len(todo)} already done · {len(todo)} to run")
    print(f"budget {args.budget:.0f}s\n", flush=True)

    started = time.time()
    for job in todo:
        if time.time() - started > args.budget:
            print(f"\nbudget reached — {len(todo)} remained when this batch started")
            break
        t0 = time.time()
        label = f"{job['symbol']}/{job['algo']}"
        print(f"  {label:<24} ", end="", flush=True)
        try:
            meta = run_one(job)
            perf = (meta.get("test_performance") or {}).get("total_return")
            aware = meta.get("regime_aware")
            elapsed = round(time.time() - t0, 1)
            if aware is not True:
                raise RuntimeError("saved checkpoint is not regime-aware")
            state["done"][_key(job)] = {"return": perf, "seconds": elapsed}
            state["failed"].pop(_key(job), None)
            print(f"{elapsed:7.1f}s  return {perf}")
        except Exception as exc:  # noqa: BLE001 - one failure must not stop the sweep
            elapsed = round(time.time() - t0, 1)
            state["failed"][_key(job)] = {"error": str(exc)[:300], "seconds": elapsed}
            print(f"{elapsed:7.1f}s  FAILED {str(exc)[:90]}")
        _save_state(state)

    done, failed = len(state["done"]), len(state["failed"])
    print(f"\ncumulative: {done} done · {failed} failed · {len(plan) - done} remaining")
    if failed:
        print("failures so far:")
        for k, v in list(state["failed"].items())[:10]:
            print(f"  {k}: {v['error'][:100]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
