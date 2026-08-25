#!/usr/bin/env python3
"""Report which catalogue symbols have a usable forecaster, and which do not.

Reads the checkpoint directory and the instrument universe; trains nothing and
writes nothing. Every number below is measured at run time, so the report
cannot drift from reality the way a hand-maintained list would.

    python3 scripts/forecaster_coverage.py
    python3 scripts/forecaster_coverage.py --horizons 1,5,30,60
"""
from __future__ import annotations

import argparse
import collections
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "backend"))

from app.services.data.universe import UNIVERSE            # noqa: E402
from app.services.forecasting.trainer import forecast_trainer   # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--horizons", default="1,5,30,60",
                        help="comma-separated horizons to probe")
    args = parser.parse_args()
    horizons = [int(h) for h in args.horizons.split(",") if h.strip()]

    covered: dict[str, list[dict]] = {}
    missing: list[tuple[str, str]] = []
    for ins in sorted(UNIVERSE, key=lambda i: (i.asset_class, i.symbol)):
        found = forecast_trainer.available_forecasts(ins.symbol)
        if found:
            covered[ins.symbol] = found
        else:
            missing.append((ins.symbol, ins.asset_class))

    print(f"checkpoint directory : {forecast_trainer.model_dir}")
    print(f"catalogue symbols    : {len(UNIVERSE)}")
    print(f"with a forecaster    : {len(covered)}")
    print(f"without any model    : {len(missing)}")

    all_h = sorted({f["horizon"] for v in covered.values() for f in v})
    print(f"horizons trained     : {all_h or '(none)'}")

    if covered:
        print("\n-- symbols with a trained model " + "-" * 45)
        for symbol, found in covered.items():
            pairs = ", ".join(f"{f['model']}/h{f['horizon']}" for f in found)
            resolved = {h: forecast_trainer.resolve_model(symbol, "lstm", h)
                        for h in horizons}
            usable = ", ".join(f"h{h}->{m}" for h, m in resolved.items() if m)
            print(f"  {symbol:<10} {pairs}")
            print(f"  {'':<10} resolves: {usable or '(no probed horizon)'}")

    if missing:
        by_class: dict[str, list[str]] = collections.defaultdict(list)
        for symbol, asset_class in missing:
            by_class[asset_class].append(symbol)
        print("\n-- symbols with no model (Expected Movement is N/A) " + "-" * 25)
        for asset_class, symbols in sorted(by_class.items()):
            print(f"  {asset_class:<10} ({len(symbols):2d}) {', '.join(symbols)}")

    print("\nA symbol listed as missing has no checkpoint file at all; that is a\n"
          "training gap, not a loading fault. Train one with:\n"
          "  POST /api/v1/forecast/train  {\"symbol\": \"<TICKER>\", "
          "\"model\": \"lstm\", \"horizon\": 5}\n"
          "It becomes usable immediately — discovery is by directory listing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
