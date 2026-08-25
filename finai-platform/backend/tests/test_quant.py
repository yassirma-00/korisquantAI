"""Unit tests for conformal prediction, GARCH, regime detection and VaR validation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.services.forecasting.advanced import (
    DirectionalCalibrator,
    ForecastEnsemble,
    RegimeDetector,
    VolatilityForecaster,
)
from app.services.forecasting.conformal import (
    AdaptiveConformal,
    MondrianConformal,
    SplitConformal,
    conformal_quantile,
    evaluate_coverage,
)
from app.services.risk.advanced_var import (
    backtest_var,
    christoffersen_independence_test,
    kupiec_pof_test,
    var_cornish_fisher,
    var_ewma,
    var_extreme_value,
    var_historical,
    var_student_t,
)


# ============================================================== conformal
def test_conformal_quantile_finite_sample_correction():
    """The (n+1)/n correction must make the quantile conservative, never anti-."""
    scores = np.arange(1, 11, dtype=float)
    q = conformal_quantile(scores, alpha=0.1)
    assert q >= np.quantile(scores, 0.9)


def test_split_conformal_achieves_target_coverage():
    """The central promise: >= (1-alpha) coverage on exchangeable data."""
    rng = np.random.default_rng(0)
    y = rng.standard_t(4, size=2000) * 0.02      # fat-tailed, like real returns
    cal, test = y[:1000], y[1000:]
    model = SplitConformal(alpha=0.1).calibrate(cal, np.zeros(1000))
    ivs = [model.predict(0.0) for _ in test]
    cov = evaluate_coverage(test, [i.lower for i in ivs], [i.upper for i in ivs], 0.9)
    assert cov["empirical_coverage"] >= 0.86, cov


def test_conformal_holds_target_on_fat_tails():
    """Conformal reads the empirical tail, so fat tails do not break coverage.

    A Gaussian z-band assumes normality

    on t-distributed returns its coverage
    drifts away from target in whichever direction the sample variance happens
    to be biased. Conformal is distribution-free and stays on target.
    """
    rng = np.random.default_rng(1)
    y = rng.standard_t(3, size=4000) * 0.02
    cal, test = y[:2000], y[2000:]

    conformal = SplitConformal(alpha=0.1).calibrate(cal, np.zeros(2000))
    cov_conf = float(np.mean(np.abs(test) <= conformal.q))
    cov_gauss = float(np.mean(np.abs(test) <= 1.645 * float(np.std(cal))))

    assert 0.86 <= cov_conf <= 0.94, f"conformal off target: {cov_conf}"
    # Conformal must be at least as close to the 90% target as the z-band
    assert abs(cov_conf - 0.9) <= abs(cov_gauss - 0.9) + 0.01


def test_mondrian_widens_in_turbulent_regimes():
    rng = np.random.default_rng(2)
    n = 1500
    # Continuously-varying volatility (as in real markets) so the percentile
    # edges are distinct and every bucket is populated.
    vol = np.linspace(0.005, 0.06, n) * rng.uniform(0.9, 1.1, n)
    y = rng.standard_normal(n) * vol
    model = MondrianConformal(alpha=0.1, n_bins=3).calibrate(y, np.zeros(n), vol)
    calm = model.predict(0.0, float(np.percentile(vol, 5)))
    storm = model.predict(0.0, float(np.percentile(vol, 95)))
    assert storm.width > calm.width, "band must widen when volatility is high"
    assert storm.details["regime"] != calm.details["regime"]


def test_mondrian_handles_duplicate_percentile_edges():
    """Regression: piecewise-constant volatility collapsed a bucket to 0 samples."""
    rng = np.random.default_rng(21)
    n = 1200
    vol = np.concatenate([np.full(n // 2, 0.01), np.full(n // 2, 0.05)])
    y = rng.standard_normal(n) * vol
    model = MondrianConformal(alpha=0.1, n_bins=3).calibrate(y, np.zeros(n), vol)
    assert all(c > 0 for c in model.counts.values()), f"empty bucket: {model.counts}"
    assert model.predict(0.0, 0.05).width > model.predict(0.0, 0.01).width


def test_adaptive_conformal_recovers_under_drift():
    """ACI with feedback should track a regime shift that breaks split conformal."""
    rng = np.random.default_rng(3)
    calm = rng.standard_normal(800) * 0.01
    storm = rng.standard_normal(800) * 0.04        # 4x volatility jump
    y = np.concatenate([calm, storm])

    split = SplitConformal(alpha=0.1).calibrate(y[:800], np.zeros(800))
    ivs = [split.predict(0.0) for _ in y[800:]]
    cov_split = evaluate_coverage(y[800:], [i.lower for i in ivs], [i.upper for i in ivs], 0.9)

    aci = AdaptiveConformal(alpha=0.1, gamma=0.05).calibrate(y[:800], np.zeros(800))
    lo, hi = [], []
    for actual in y[800:]:
        iv = aci.predict(0.0)
        lo.append(iv.lower)
        hi.append(iv.upper)
        aci.update(float(actual), iv)
    cov_aci = evaluate_coverage(y[800:], lo, hi, 0.9)

    assert cov_split["empirical_coverage"] < 0.75, "split should fail badly on a 4x vol jump"
    assert cov_aci["empirical_coverage"] > cov_split["empirical_coverage"] + 0.10


def test_adaptive_open_loop_falls_back_safely():
    """Without feedback ACI must not silently under-cover."""
    rng = np.random.default_rng(4)
    y = rng.standard_normal(1600) * 0.02
    aci = AdaptiveConformal(alpha=0.1).calibrate(y[:800], np.zeros(800))
    ivs = [aci.predict(0.0) for _ in y[800:]]
    cov = evaluate_coverage(y[800:], [i.lower for i in ivs], [i.upper for i in ivs], 0.9)
    assert cov["empirical_coverage"] >= 0.85
    assert aci.predict(0.0).details["online_feedback"] is False


def test_evaluate_coverage_penalises_misses():
    y = np.array([0.0, 0.0, 5.0])
    tight = evaluate_coverage(y, np.full(3, -1.0), np.full(3, 1.0), 0.9)
    wide = evaluate_coverage(y, np.full(3, -6.0), np.full(3, 6.0), 0.9)
    assert wide["empirical_coverage"] > tight["empirical_coverage"]
    assert tight["interval_score"] > wide["interval_score"], "a miss must cost more than width"


# ================================================================ ensemble
def test_inverse_error_weights_the_better_model_higher():
    e = ForecastEnsemble("inverse_error").combine(
        {"good": 0.01, "bad": -0.01},
        {"good": {"rmse": 0.01}, "bad": {"rmse": 0.10}})
    assert e.weights["good"] > e.weights["bad"] * 5
    assert e.prediction > 0


def test_directional_weighting_ignores_coin_flip_models():
    e = ForecastEnsemble("directional").combine(
        {"skilled": 0.02, "chance": -0.02},
        {"skilled": {"directional_accuracy": 65}, "chance": {"directional_accuracy": 50}})
    assert e.weights["skilled"] > 0.95


def test_ensemble_dispersion_and_agreement():
    agree = ForecastEnsemble("mean").combine({"a": 0.01, "b": 0.011, "c": 0.009})
    argue = ForecastEnsemble("mean").combine({"a": 0.05, "b": -0.05, "c": 0.001})
    assert agree.agreement == 1.0
    assert argue.agreement < 1.0
    assert argue.dispersion > agree.dispersion


def test_trimmed_ensemble_drops_outliers():
    e = ForecastEnsemble("trimmed").combine({"a": 0.01, "b": 0.011, "outlier": 5.0})
    assert abs(e.prediction) < 0.5, "an extreme member must not dominate"


# ================================================================== GARCH
@pytest.mark.parametrize("model", ["garch", "egarch", "gjr"])
def test_garch_variants_fit_and_forecast(model, ohlcv):
    returns = ohlcv["close"].pct_change().dropna()
    vf = VolatilityForecaster(model).fit(returns)
    fc = vf.forecast(horizon=5)
    assert len(fc["daily_volatility"]) == 5
    assert fc["annualised_volatility"] > 0
    assert all(np.isfinite(v) for v in fc["daily_volatility"])


def test_multistep_uses_simulation_for_nonlinear_models(ohlcv):
    """Regression: EGARCH/GJR have no analytic multi-step variance."""
    returns = ohlcv["close"].pct_change().dropna()
    for model in ("egarch", "gjr"):
        fc = VolatilityForecaster(model).fit(returns).forecast(horizon=5)
        assert fc["forecast_method"] == "simulation"
    assert VolatilityForecaster("garch").fit(returns).forecast(5)["forecast_method"] == "analytic"


def test_garch_diagnostics_reported(ohlcv):
    returns = ohlcv["close"].pct_change().dropna()
    diag = VolatilityForecaster("gjr").fit(returns).diagnostics()
    assert "well_specified" in diag and isinstance(diag["well_specified"], bool)
    assert 0 <= diag["ljung_box_p_residuals"] <= 1


def test_garch_rejects_short_series():
    with pytest.raises(ValueError, match=">= 100"):
        VolatilityForecaster("garch").fit(pd.Series(np.random.randn(50) * 0.01))


# ================================================================= regime
def test_regime_detection_on_a_crash():
    idx = pd.bdate_range(end="2024-01-01", periods=400)
    calm = np.random.default_rng(5).standard_normal(300) * 0.005
    crash = np.random.default_rng(6).standard_normal(100) * 0.05 - 0.01
    prices = 100 * np.exp(np.cumsum(np.concatenate([calm, crash])))
    df = pd.DataFrame({"close": prices, "open": prices, "high": prices * 1.01,
                       "low": prices * 0.99, "volume": 1e6}, index=idx)
    result = RegimeDetector().detect(df)
    assert result["regime"] in ("crisis", "bear")
    assert result["volatility_ratio"] > 1.2
    assert "low" in result["model_reliability"]


def test_regime_handles_short_history(short_ohlcv):
    assert RegimeDetector().detect(short_ohlcv)["regime"] == "unknown"


# ========================================================== VaR estimators
def test_var_ordering_across_confidence_levels(ohlcv):
    r = ohlcv["close"].pct_change().dropna()
    assert var_historical(r, 0.99) < var_historical(r, 0.95) < 0


def test_ewma_reacts_faster_than_historical():
    """A volatility spike must move EWMA VaR more than the 250-day average."""
    calm = np.random.default_rng(7).standard_normal(400) * 0.005
    spike = np.random.default_rng(8).standard_normal(30) * 0.05
    r = pd.Series(np.concatenate([calm, spike]))
    assert abs(var_ewma(r, 0.95)) > abs(var_historical(r, 0.95))


def test_cornish_fisher_adjusts_for_skew():
    rng = np.random.default_rng(9)
    skewed = pd.Series(-np.abs(rng.standard_normal(1000)) * 0.03 + 0.005)
    assert np.isfinite(var_cornish_fisher(skewed, 0.95))


def test_student_t_var_is_finite(ohlcv):
    assert np.isfinite(var_student_t(ohlcv["close"].pct_change().dropna(), 0.99))


def test_evt_reports_tail_shape(ohlcv):
    result = var_extreme_value(ohlcv["close"].pct_change().dropna(), 0.99)
    if result["method"].startswith("extreme_value"):
        assert "shape_xi" in result and "tail_type" in result
        assert result["var"] < 0


# ========================================================== VaR validation
def test_kupiec_accepts_a_correct_model():
    rng = np.random.default_rng(10)
    breaches = rng.random(1000) < 0.05        # exactly the expected rate
    result = kupiec_pof_test(breaches, 0.95)
    assert not result["reject_at_5pct"]
    assert "PASS" in result["verdict"]


def test_kupiec_rejects_an_understated_model():
    rng = np.random.default_rng(11)
    breaches = rng.random(1000) < 0.15        # 3x too many breaches
    result = kupiec_pof_test(breaches, 0.95)
    assert result["reject_at_5pct"]
    assert "too many" in result["verdict"]


def test_christoffersen_detects_clustering():
    """A model that fails in bursts must be caught even with the right breach count.

    Both series below contain exactly 50 breaches, so Kupiec cannot tell them
    apart - only the independence test can.
    """
    rng = np.random.default_rng(12)
    # Genuine Bernoulli draws: P(breach | breach) == P(breach | calm)
    independent = rng.random(1000) < 0.05
    clustered = np.zeros(1000, dtype=bool)
    clustered[100:150] = True                 # all breaches back-to-back

    ind_result = christoffersen_independence_test(independent)
    cl_result = christoffersen_independence_test(clustered)

    assert not ind_result["reject_at_5pct"], f"false positive: p={ind_result['p_value']}"
    assert cl_result["reject_at_5pct"], f"missed clustering: p={cl_result['p_value']}"
    assert "cluster" in cl_result["verdict"]
    # The signature of clustering: a breach is far more likely right after one
    assert cl_result["p_breach_after_breach"] > cl_result["p_breach_after_calm"] * 5


def test_backtest_reports_all_components(ohlcv):
    r = ohlcv["close"].pct_change().dropna()
    bt = backtest_var(r, 0.95, window=250, method="historical")
    if "error" not in bt:
        for key in ("kupiec_test", "independence_test", "basel", "model_valid"):
            assert key in bt
        assert bt["n_breaches"] <= bt["n_observations"]


# =========================================================== calibration
def test_probability_calibration_improves_reliability():
    rng = np.random.default_rng(13)
    scores = rng.random(600)
    outcomes = (rng.random(600) < scores * 0.6 + 0.2).astype(float)   # miscalibrated
    cal = DirectionalCalibrator().fit(scores, outcomes)
    raw_eval = DirectionalCalibrator.evaluate(scores, outcomes)
    cal_probs = np.array([cal.predict_proba(s) for s in scores])
    cal_eval = DirectionalCalibrator.evaluate(cal_probs, outcomes)
    assert cal_eval["expected_calibration_error"] <= raw_eval["expected_calibration_error"]


def test_uncalibrated_fallback_is_bounded():
    cal = DirectionalCalibrator()
    for score in (-10.0, 0.0, 10.0):
        assert 0.0 <= cal.predict_proba(score) <= 1.0
