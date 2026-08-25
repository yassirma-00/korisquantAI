#!/usr/bin/env python3
"""Remove RL agents trained with the old, leaking train/test split.

Background
----------
Before the fix, ``RLService._split`` handed the test set 60 bars of *training*
data as an indicator warm-up, so roughly a quarter of each "unseen" window had
already been fitted on. Agents trained then still sit on disk with inflated
metrics baked into their metadata, and the UI happily replays those numbers.

Retraining is the only honest remedy: the checkpoint itself saw contaminated
data, so its reported performance cannot be salvaged by recomputing metadata.

Usage
-----
    python scripts/purge_leaky_agents.py            # list what would go
    python scripts/purge_leaky_agents.py --delete   # actually remove them
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RL_DIR = ROOT / "data" / "models" / "rl"


def is_leaky(meta: dict) -> bool:
    """True when the recorded test window starts before training ended."""
    train_window = meta.get("train_window")
    test_window = meta.get("test_window")
    if not train_window or not test_window:
        return False                      # portfolio agents record no windows
    return str(test_window[0]) <= str(train_window[1])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--delete", action="store_true",
                        help="remove the affected checkpoints (default: dry run)")
    args = parser.parse_args()

    if not RL_DIR.exists():
        print("No RL agents on disk - nothing to do.")
        return 0

    leaky: list[tuple[Path, dict]] = []
    clean = 0
    for meta_path in sorted(RL_DIR.glob("*.json")):
        try:
            meta = json.loads(meta_path.read_text())
        except Exception:
            continue
        if is_leaky(meta):
            leaky.append((meta_path, meta))
        else:
            clean += 1

    if not leaky:
        print(f"All {clean} stored agents use a clean split. Nothing to purge.")
        return 0

    print(f"{len(leaky)} agent(s) were trained with the leaking split "
          f"({clean} already clean):\n")
    for meta_path, meta in leaky:
        tw, te = meta["train_window"], meta["test_window"]
        overlap_note = f"test starts {te[0]} but training ran to {tw[1]}"
        print(f"  {meta_path.stem:34s} {meta.get('algo', '?'):12s} {overlap_note}")

    if not args.delete:
        print("\nDry run. Re-run with --delete to remove them, then retrain "
              "from the RL page (or scripts/seed_demo.py).")
        return 0

    removed = 0
    for meta_path, _ in leaky:
        for suffix in (".json", ".pt", ".zip"):
            candidate = meta_path.with_suffix(suffix)
            if candidate.exists():
                candidate.unlink()
                removed += 1
    print(f"\nRemoved {removed} file(s). Retrain the agents you need; the new "
          f"split is strictly disjoint.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
