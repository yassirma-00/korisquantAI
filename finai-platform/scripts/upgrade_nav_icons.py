#!/usr/bin/env python3
"""Replace the sidebar's Unicode glyphs with inline stroke SVG icons.

Why
---
The navigation used typographic characters (◈ ◉ ◭ ⬡ ✦ ◇ ▤ ⚠ ⚙ ◎) as icons.
Those are glyphs from whatever font happens to resolve them, so they render at
inconsistent weights and baselines across platforms, cannot inherit a stroke
width, and several carry unintended semantics (⚠ is a warning sign, used here
merely to mean "risk page"). They also make the rail look sketched rather than
designed.

These are replaced with a matched 16px stroke icon set, inlined as SVG so no
network request, icon font or build step is introduced — the in-app preview has
no network access, so an external icon library would silently fail.

Purely presentational: the `<a class="nav-item" href=...>` element, its href,
its label text and the `.nav-icon` wrapper are all untouched, so `highlightNav()`
and every navigation test keep working exactly as before.

Idempotent: running it twice is a no-op.
"""
from __future__ import annotations

import pathlib
import re
import sys

FRONTEND = pathlib.Path(__file__).resolve().parents[1] / "frontend"

PAGES = ("index", "analysis", "forecast", "rl", "signals", "xai",
         "portfolio", "risk", "hyperparams", "training")

# A 24x24 stroke set, drawn on the same grid so weights match.
# Keyed by the destination page, which is stable — unlike the label text.
ICONS: dict[str, str] = {
    # Market Overview — candlestick chart
    "index.html": (
        '<path d="M4 20V10M4 7V4M9 20v-4M9 13V4M15 20v-3M15 14V4M20 20v-8M20 9V4"/>'
        '<rect x="2.5" y="7" width="3" height="3" rx="1"/>'
        '<rect x="7.5" y="13" width="3" height="3" rx="1"/>'
        '<rect x="13.5" y="14" width="3" height="3" rx="1"/>'
        '<rect x="18.5" y="9" width="3" height="3" rx="1"/>'
    ),
    # Technical Analysis — trend line with points
    "analysis.html": (
        '<path d="M3 20h18"/><path d="M4 16l5-5 4 3 6-7"/>'
        '<circle cx="9" cy="11" r="1.4"/><circle cx="13" cy="14" r="1.4"/>'
    ),
    # AI Forecasting — projection into a dashed future
    "forecast.html": (
        '<path d="M3 20h18"/><path d="M4 15l5-4 3 2"/>'
        '<path d="M12 13l4-3 4 2" stroke-dasharray="3 2.5"/>'
        '<circle cx="12" cy="13" r="1.4"/>'
    ),
    # RL Agent — node learning from feedback
    "rl.html": (
        '<circle cx="12" cy="12" r="3"/>'
        '<path d="M12 3v3M12 18v3M3 12h3M18 12h3"/>'
        '<path d="M6.5 6.5l1.8 1.8M15.7 15.7l1.8 1.8M17.5 6.5l-1.8 1.8M8.3 15.7l-1.8 1.8"/>'
    ),
    # Recommendations — signal spark
    "signals.html": (
        '<path d="M12 3l2.1 5.2L19.5 10l-5.4 1.8L12 17l-2.1-5.2L4.5 10l5.4-1.8z"/>'
        '<path d="M18 17l.9 2.1 2.1.9-2.1.9-.9 2.1-.9-2.1-2.1-.9 2.1-.9z"/>'
    ),
    # Explainability — a lens over the model
    "xai.html": (
        '<circle cx="11" cy="11" r="6"/><path d="M15.5 15.5L21 21"/>'
        '<path d="M8.5 12.5l2-2.5 2 2 2-3"/>'
    ),
    # Portfolio — allocation ring
    "portfolio.html": (
        '<circle cx="12" cy="12" r="8"/><path d="M12 4v8l6 3.6"/>'
    ),
    # Risk & Alerts — shield
    "risk.html": (
        '<path d="M12 3l7 3v5.5c0 4.3-2.9 7.9-7 9.5-4.1-1.6-7-5.2-7-9.5V6z"/>'
        '<path d="M12 9v3.5"/><circle cx="12" cy="15.6" r="0.9" fill="currentColor" stroke="none"/>'
    ),
    # Hyperparameters — sliders
    "hyperparams.html": (
        '<path d="M4 7h6M14 7h6M4 12h10M18 12h2M4 17h3M11 17h9"/>'
        '<circle cx="12" cy="7" r="2"/><circle cx="16" cy="12" r="2"/><circle cx="9" cy="17" r="2"/>'
    ),
    # Training Intelligence — rising bars under a pulse.
    # Retained deliberately: the page is hidden from the sidebar but still
    # served, and this entry only restyles an <a> that already exists. It
    # never inserts one, so keeping it cannot resurrect the nav link.
    "training.html": (
        '<path d="M3 20h18"/><path d="M6 20v-5M11 20v-9M16 20v-6M21 20v-11"/>'
        '<path d="M3 8l4-3 4 2.5L15 4"/>'
    ),
}

SVG = ('<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" '
       'stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" '
       'aria-hidden="true" focusable="false">{paths}</svg>')


def upgrade(html: str) -> tuple[str, int]:
    """Swap each nav item's glyph for the SVG matching its destination."""
    changed = 0

    def replace(match: re.Match[str]) -> str:
        nonlocal changed
        href, icon_inner = match.group("href"), match.group("inner")
        paths = ICONS.get(href)
        if paths is None:
            return match.group(0)
        if "<svg" in icon_inner:          # already upgraded
            return match.group(0)
        changed += 1
        return match.group(0).replace(
            f'<span class="nav-icon">{icon_inner}</span>',
            f'<span class="nav-icon">{SVG.format(paths=paths)}</span>',
        )

    pattern = re.compile(
        r'<a class="nav-item" href="(?P<href>[^"]+)">'
        r'\s*<span class="nav-icon">(?P<inner>.*?)</span>',
        re.S,
    )
    return pattern.sub(replace, html), changed


def main() -> int:
    total = 0
    for name in PAGES:
        page = FRONTEND / f"{name}.html"
        original = page.read_text()
        updated, n = upgrade(original)
        if n:
            page.write_text(updated)
        total += n
        print(f"  {name}.html: {n} icon(s) upgraded")
    print(f"total: {total}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
