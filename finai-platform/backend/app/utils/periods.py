"""One definition of a time range, shared by every page and every endpoint.

The important idea here is that **the window a user looks at and the window a
model computes over are two different things.**

Selecting "1M" should not tell the crash-risk model to fit on 22 bars — it
cannot, and the honest output would be a dash. Professional terminals do not
behave that way: they show you the month you asked for while the analytics
underneath quietly read whatever history they need. So every range carries two
numbers:

* ``display`` — the period fetched for charts and tables, exactly what was asked
  for.
* ``compute`` — the minimum period a model needs to produce a stable answer.

An endpoint fetches ``compute`` history, runs the model on all of it, and then
trims the *presentation* to ``display``. The user sees a one-month chart backed
by a model that saw five years.
"""

from __future__ import annotations

from dataclasses import dataclass

# Analytical floors, in trading bars, taken from what each model actually
# refuses to run below (see anomaly.py, regime.py, explainer.py).
MIN_BARS = {
    "crash_risk": 60,
    "regime": 120,
    "bubble": 200,
    "forecast": 250,
    "rl": 250,
    "xai": 200,
    "default": 250,
}


@dataclass(frozen=True)
class TimeRange:
    key: str            # what the UI sends: "1d", "5d", "3mo", "ytd", "max"
    label: str          # what the segmented control shows: "1D", "5D", "3M"
    display: str        # period string for the chart fetch
    interval: str       # bar size that makes this window meaningful
    compute: str        # period string for model fitting
    approx_bars: int    # rough display length, for trimming and for tests
    group: str          # intraday | days | months | years


# The thirteen ranges, in the order the control renders them.
#
# 1D and 5D use intraday bars on purpose: one daily bar is not a chart, and the
# old pipeline silently answered a 1D request with synthetic data because the
# real fetch returned too few rows to pass its length check.
RANGES: tuple[TimeRange, ...] = (
    TimeRange("1d",  "1D",  "1d",  "5m",  "1y",  78,   "intraday"),
    TimeRange("5d",  "5D",  "5d",  "30m", "1y",  65,   "intraday"),
    TimeRange("1mo", "1M",  "1mo", "1d",  "2y",  22,   "months"),
    TimeRange("3mo", "3M",  "3mo", "1d",  "2y",  63,   "months"),
    TimeRange("6mo", "6M",  "6mo", "1d",  "2y",  126,  "months"),
    TimeRange("ytd", "YTD", "ytd", "1d",  "2y",  150,  "months"),
    TimeRange("1y",  "1Y",  "1y",  "1d",  "2y",  252,  "years"),
    TimeRange("3y",  "3Y",  "3y",  "1d",  "5y",  756,  "years"),
    TimeRange("5y",  "5Y",  "5y",  "1d",  "5y",  1260, "years"),
    TimeRange("10y", "10Y", "10y", "1d",  "10y", 2520, "years"),
    TimeRange("max", "MAX", "max", "1d",  "max", 99999, "years"),
)

BY_KEY: dict[str, TimeRange] = {r.key: r for r in RANGES}

# Older links and saved settings used yfinance-style strings that are not keys
# in the new control. Map them rather than 422-ing a bookmark.
# 1W and 2W were removed: on daily bars they are 5 and 10 points, too few for
# the indicators and analytics this platform draws. Existing links fall back
# to the nearest surviving window instead of erroring.
ALIASES = {"2y": "1y", "1wk": "5d", "1w": "5d", "2wk": "1mo", "2w": "1mo",
           "1mon": "1mo", "12mo": "1y"}


def resolve(key: str | None, default: str = "1y") -> TimeRange:
    """Look up a range, tolerating legacy spellings and unknown input."""
    if not key:
        return BY_KEY[default]
    normalised = str(key).strip().lower()
    normalised = ALIASES.get(normalised, normalised)
    return BY_KEY.get(normalised, BY_KEY[default])


def compute_period(key: str | None, model: str = "default") -> str:
    """The history a model should fit on, regardless of what is displayed.

    Always at least as long as the display window: asking for 10Y must not make
    a model fit on 2Y just because that is its floor.
    """
    selected = resolve(key)
    needed_bars = MIN_BARS.get(model, MIN_BARS["default"])
    candidates = [selected.compute, selected.display]
    # Pick the longest of {model floor, requested display, declared compute}.
    floor_key = _period_for_bars(needed_bars)
    candidates.append(floor_key)
    return max(candidates, key=_rank)


def analysis_window(key: str | None, model: str = "default") -> str:
    """History for a *diagnostic* measurement, as opposed to model training.

    ``compute_period`` deliberately over-fetches: a neural net or an RL agent
    wants as much history as it can get, so each range declares a ``compute``
    floor well above its display window. That is right for training and wrong
    for measurement.

    A risk metric is a *description of the selected window*. Over-fetching
    makes it describe a different window than the one on screen, and because
    every range from 1D to 1Y declares ``compute="2y"``, seven of the eleven
    ranges collapsed onto one identical answer — the period selector moved and
    nothing changed. Measured on AAPL: 1D, 5D, 1M, 3M, 6M, YTD and 1Y all
    returned crash 0.429 / bubble 0.314.

    So this returns the *shortest* period that is both (a) at least the display
    window and (b) long enough for the model's floor. Short windows still get
    the history they need — nothing ever falls back to a dash or to synthetic
    data — while a window that is already long enough is used exactly as
    selected.
    """
    selected = resolve(key)
    floor_key = _period_for_bars(MIN_BARS.get(model, MIN_BARS["default"]))
    return max([selected.display, floor_key], key=_rank)


def model_bars(key: str | None, model: str = "default") -> int:
    """How many trailing bars a *measurement* should actually read.

    The rule the platform committed to: a model retrieves the minimum history
    it needs for a reliable computation, and otherwise describes the window the
    user selected. So this is ``max(display window, model floor)`` in bars.

    Worked through for the risk page:

    ==========  =========================  ========================
    Selection   Crash Risk (floor 60)      Bubble (floor 200)
    ==========  =========================  ========================
    1M (22)     60 bars  (floor governs)   200 bars (floor governs)
    3M (63)     63 bars  (window governs)  200 bars (floor governs)
    1Y (252)    252 bars (window governs)  252 bars (window governs)
    5Y (1260)   1260 bars                  1260 bars
    ==========  =========================  ========================

    Neither model ever runs under its floor, so no panel shows a dash and
    nothing falls back to synthetic data; and past the floor the selection
    genuinely drives the number.
    """
    selected = resolve(key)
    floor = MIN_BARS.get(model, MIN_BARS["default"])
    # +1 because these floors are stated in *returns*, and N price bars yield
    # only N-1 returns. Slicing exactly 60 bars for a model that requires 60
    # returns handed it 59 and tripped its own `insufficient_data` guard — the
    # dash the whole display/compute split exists to prevent.
    return max(int(selected.approx_bars), int(floor) + 1)


def _rank(period: str) -> int:
    order = ["1d", "5d", "1mo", "3mo", "6mo", "ytd", "1y", "2y", "3y", "5y", "10y", "max"]
    return order.index(period) if period in order else 0


def _period_for_bars(bars: int) -> str:
    """Shortest standard period that comfortably yields `bars` trading days."""
    if bars <= 60:
        return "6mo"
    if bars <= 130:
        return "1y"
    if bars <= 260:
        return "2y"
    if bars <= 760:
        return "5y"
    return "10y"


def catalogue() -> list[dict]:
    """What the frontend renders in the segmented control."""
    return [
        {"key": r.key, "label": r.label, "display": r.display,
         "interval": r.interval, "compute": r.compute, "group": r.group}
        for r in RANGES
    ]
