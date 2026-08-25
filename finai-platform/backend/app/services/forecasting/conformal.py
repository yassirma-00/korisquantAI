"""Conformal prediction: intervals with a *proven* coverage guarantee.

Why this matters more than a better point forecast
--------------------------------------------------
On financial returns, no model reliably predicts direction much better than a
coin flip. What *is* achievable — and far more useful for risk management — is
an interval you can trust: "the next 5-day return lands in [-3.1%, +4.2%] with
90% probability".

Split conformal prediction gives exactly that. Under the single assumption of
**exchangeability** of calibration and test residuals, coverage is guaranteed in
finite samples:

    P(y ∈ C(x)) ≥ 1 − α

No distributional assumption, no asymptotics, valid for *any* underlying model.

Three variants are implemented:

* ``SplitConformal``      — constant-width interval; the baseline guarantee
* ``MondrianConformal``   — separate calibration per volatility regime, so the
                            band widens in turbulent markets instead of relying
                            on an average that fits neither calm nor crisis
* ``AdaptiveConformal``   — online quantile update (ACI, Gibbs & Candès 2021)
                            that keeps empirical coverage on target even when
                            the return distribution drifts — essential for
                            non-stationary markets

References: Vovk et al. (2005); Lei et al. (2018); Gibbs & Candès (2021);
Romano et al. (2019) for the normalised (locally-weighted) variant.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def _finite(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float).ravel()
    return a[np.isfinite(a)]


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """The finite-sample-corrected (1-α) quantile used by split conformal.

    The ``⌈(n+1)(1-α)⌉ / n`` correction is what turns an empirical quantile into
    a *guarantee* rather than an approximation.
    """
    scores = _finite(scores)
    n = len(scores)
    if n == 0:
        return float("nan")
    level = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
    return float(np.quantile(scores, level, method="higher"))


@dataclass
class ConformalInterval:
    point: float
    lower: float
    upper: float
    alpha: float
    method: str
    width: float
    n_calibration: int
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "point": round(self.point, 6),
            "lower": round(self.lower, 6),
            "upper": round(self.upper, 6),
            "alpha": self.alpha,
            "confidence_level": round(1 - self.alpha, 4),
            "method": self.method,
            "width": round(self.width, 6),
            "n_calibration": self.n_calibration,
            **self.details,
        }


class SplitConformal:
    """Constant-width conformal intervals from held-out absolute residuals."""

    method = "split_conformal"

    def __init__(self, alpha: float = 0.1) -> None:
        self.alpha = alpha
        self.q: float | None = None
        self.scores: np.ndarray = np.array([])

    def calibrate(self, y_true: np.ndarray, y_pred: np.ndarray) -> SplitConformal:
        residuals = np.abs(_finite(np.asarray(y_true) - np.asarray(y_pred)))
        self.scores = residuals
        self.q = conformal_quantile(residuals, self.alpha)
        return self

    def predict(self, point: float) -> ConformalInterval:
        if self.q is None or not np.isfinite(self.q):
            raise RuntimeError("SplitConformal must be calibrated first")
        return ConformalInterval(
            point=point, lower=point - self.q, upper=point + self.q,
            alpha=self.alpha, method=self.method, width=2 * self.q,
            n_calibration=len(self.scores),
        )


class MondrianConformal:
    """Volatility-conditional conformal: a separate quantile per regime.

    A single average band is simultaneously too wide in calm markets and far too
    narrow in a crisis. Bucketing the calibration set by realised volatility
    keeps the guarantee while making the interval *informative*.
    """

    method = "mondrian_conformal"

    def __init__(self, alpha: float = 0.1, n_bins: int = 3) -> None:
        self.alpha = alpha
        self.n_bins = n_bins
        self.edges: np.ndarray | None = None
        self.q_by_bin: dict[int, float] = {}
        self.q_global: float = float("nan")
        self.counts: dict[int, int] = {}

    def _bin(self, vol: float) -> int:
        """Map a volatility level to its calibration bucket.

        Uses ``side="left"`` so the convention matches calibration, where a
        bucket is defined as ``(edge_{b-1}, edge_b]`` — a value exactly on an
        edge belongs to the *lower* bucket. With ``side="right"`` a value equal
        to an edge would be scored against the wrong (wider) regime.
        """
        if self.edges is None or len(self.edges) == 0:
            return 0
        return int(np.clip(np.searchsorted(self.edges, vol, side="left"), 0, self.n_bins - 1))

    def calibrate(self, y_true: np.ndarray, y_pred: np.ndarray,
                  vol: np.ndarray) -> MondrianConformal:
        residuals = np.abs(np.asarray(y_true, float) - np.asarray(y_pred, float))
        vol = np.asarray(vol, float)
        mask = np.isfinite(residuals) & np.isfinite(vol)
        residuals, vol = residuals[mask], vol[mask]

        self.q_global = conformal_quantile(residuals, self.alpha)
        if len(residuals) < self.n_bins * 12:      # too few points to bucket safely
            self.edges = None
            return self

        qs = np.linspace(0, 100, self.n_bins + 1)[1:-1]
        edges = np.unique(np.percentile(vol, qs))

        # Volatility is often near piecewise-constant (two distinct regimes), so
        # percentile edges can coincide with the values themselves. An edge equal
        # to the series maximum leaves the top bucket empty, which silently
        # destroys the per-regime guarantee. Keep only interior edges that
        # actually split the data, then size n_bins to what the data supports.
        interior = [e for e in edges if (vol > e).any() and (vol <= e).any()]
        self.edges = np.array(interior) if interior else None
        self.n_bins = len(interior) + 1 if interior else 1
        for b in range(self.n_bins):
            lo = -np.inf if b == 0 else self.edges[b - 1]
            hi = np.inf if b == self.n_bins - 1 else self.edges[b]
            sel = residuals[(vol > lo) & (vol <= hi)]
            self.counts[b] = int(len(sel))
            # Fall back to the global quantile when a bucket is sparse:
            # a guarantee from 8 points is not a guarantee.
            self.q_by_bin[b] = (conformal_quantile(sel, self.alpha)
                                if len(sel) >= 12 else self.q_global)
        return self

    def predict(self, point: float, vol: float) -> ConformalInterval:
        b = self._bin(vol)
        q = self.q_by_bin.get(b, self.q_global)
        if not np.isfinite(q):
            q = self.q_global
        return ConformalInterval(
            point=point, lower=point - q, upper=point + q, alpha=self.alpha,
            method=self.method, width=2 * q,
            n_calibration=sum(self.counts.values()) or 0,
            details={"volatility_bin": b, "n_bins": self.n_bins,
                     "bin_counts": self.counts,
                     "regime": ["calm", "normal", "turbulent"][min(b, 2)]},
        )


class AdaptiveConformal:
    """Adaptive Conformal Inference — online recalibration under drift.

    After each observation the effective miscoverage level is nudged:

        α_{t+1} = α_t + γ (α_target − 1{y_t ∉ C_t})

    If recent intervals miss too often the bands widen; if they are needlessly
    wide they tighten. This tracks regime changes automatically, which fixed
    quantiles cannot do.
    """

    method = "adaptive_conformal"

    def __init__(self, alpha: float = 0.1, gamma: float = 0.02, window: int = 250) -> None:
        self.alpha_target = alpha
        self.alpha_t = alpha
        self.gamma = gamma
        self.window = window
        self.scores: list[float] = []
        self.coverage_history: list[int] = []
        self._feedback_active = False   # flipped on by the first update() call
        self._full_scores: np.ndarray = np.array([])

    def calibrate(self, y_true: np.ndarray, y_pred: np.ndarray) -> AdaptiveConformal:
        residuals = np.abs(np.asarray(y_true, float) - np.asarray(y_pred, float))
        residuals = _finite(residuals)
        # Rolling window drives the ONLINE quantile (it must forget stale regimes),
        # but the full calibration set backs the open-loop fallback: truncating to
        # the window there silently under-covers whenever the tail of the
        # calibration period happened to be calmer than the test period.
        self.scores = list(residuals[-self.window:])
        self._full_scores = residuals.copy()
        # Replay the calibration set to let alpha settle at a sensible level
        for i in range(1, len(residuals)):
            q = conformal_quantile(np.array(residuals[max(0, i - self.window):i]),
                                   np.clip(self.alpha_t, 0.005, 0.5))
            if np.isfinite(q):
                covered = int(residuals[i] <= q)
                self.coverage_history.append(covered)
                self.alpha_t = float(np.clip(
                    self.alpha_t + self.gamma * (self.alpha_target - (1 - covered)),
                    0.005, 0.5))
        return self

    def predict(self, point: float) -> ConformalInterval:
        # ACI only stays on target when outcomes are fed back via `update()`.
        # Used open-loop (predict repeatedly with no feedback) alpha_t freezes at
        # whatever the calibration replay produced and coverage drifts. Detect
        # that case and fall back to the plain split-conformal quantile, which
        # keeps the finite-sample guarantee without requiring feedback.
        if self._feedback_active:
            alpha_eff = float(np.clip(self.alpha_t, 0.005, 0.5))
            pool = np.array(self.scores)
        else:
            alpha_eff = self.alpha_target
            pool = self._full_scores if len(self._full_scores) else np.array(self.scores)
        q = conformal_quantile(pool, alpha_eff)
        if not np.isfinite(q):
            q = 0.0
        recent = self.coverage_history[-100:]
        return ConformalInterval(
            point=point, lower=point - q, upper=point + q,
            alpha=self.alpha_target, method=self.method, width=2 * q,
            n_calibration=len(self.scores),
            details={
                "effective_alpha": round(alpha_eff, 4),
                "online_feedback": self._feedback_active,
                "recent_empirical_coverage": round(float(np.mean(recent)), 4) if recent else None,
            },
        )

    def update(self, y_true: float, interval: ConformalInterval) -> None:
        """Feed a realised outcome back in (online use).

        Calling this activates true ACI behaviour: alpha adapts to keep the
        realised coverage on target as the return distribution drifts.
        """
        self._feedback_active = True
        covered = int(interval.lower <= y_true <= interval.upper)
        self.coverage_history.append(covered)
        self.alpha_t = float(np.clip(
            self.alpha_t + self.gamma * (self.alpha_target - (1 - covered)), 0.005, 0.5))
        self.scores.append(abs(y_true - interval.point))
        self.scores = self.scores[-self.window:]


def evaluate_coverage(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray,
                      target: float = 0.9) -> dict:
    """Empirical validation — does the interval actually deliver its promise?"""
    y_true = np.asarray(y_true, float)
    lower = np.asarray(lower, float)
    upper = np.asarray(upper, float)
    mask = np.isfinite(y_true) & np.isfinite(lower) & np.isfinite(upper)
    y_true, lower, upper = y_true[mask], lower[mask], upper[mask]
    if len(y_true) == 0:
        return {"error": "no valid observations"}

    covered = (y_true >= lower) & (y_true <= upper)
    coverage = float(covered.mean())
    widths = upper - lower
    # Interval score (Gneiting & Raftery): rewards narrow bands, penalises misses
    alpha = 1 - target
    penalty_low = (2 / alpha) * np.clip(lower - y_true, 0, None)
    penalty_high = (2 / alpha) * np.clip(y_true - upper, 0, None)
    interval_score = float(np.mean(widths + penalty_low + penalty_high))

    return {
        "empirical_coverage": round(coverage, 4),
        "target_coverage": target,
        "coverage_gap": round(coverage - target, 4),
        "calibrated": bool(abs(coverage - target) <= 0.05),
        "mean_width": round(float(widths.mean()), 6),
        "median_width": round(float(np.median(widths)), 6),
        "interval_score": round(interval_score, 6),
        "n_observations": int(len(y_true)),
    }
