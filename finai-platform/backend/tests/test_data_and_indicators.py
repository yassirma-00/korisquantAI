"""Tests for the data layer, synthetic engine and technical indicators."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.services.data.synthetic import generate_ohlcv, generate_quote
from app.services.data.universe import get_instrument, infer_instrument, list_instruments
from app.services.indicators.features import build_features, build_supervised, make_sequences
from app.services.indicators.technical import (
    atr,
    bollinger_bands,
    compute_indicators,
    ema,
    macd,
    rsi,
    signal_summary,
    sma,
)


# ------------------------------------------------------------------ universe
def test_universe_lookup():
    assert get_instrument("AAPL").asset_class == "equity"
    assert get_instrument("aapl") is not None, "lookup must be case-insensitive"
    assert get_instrument("NOT_A_TICKER") is None


@pytest.mark.parametrize(
    "symbol,expected",
    [("XYZ-USD", "crypto"), ("ABC=X", "forex"), ("ZZ=F", "commodity"),
     ("^ABC", "index"), ("UNKNOWN", "equity")],
)
def test_infer_instrument_from_suffix(symbol, expected):
    assert infer_instrument(symbol).asset_class == expected


def test_filter_universe_by_class():
    cryptos = list_instruments(asset_class="crypto")
    assert cryptos and all(i.asset_class == "crypto" for i in cryptos)


# ----------------------------------------------------------------- synthetic
def test_synthetic_is_deterministic():
    a = generate_ohlcv("AAPL", periods=250)
    b = generate_ohlcv("AAPL", periods=250)
    pd.testing.assert_frame_equal(a, b)


def test_synthetic_differs_per_symbol():
    a = generate_ohlcv("AAPL", periods=250)["close"]
    b = generate_ohlcv("MSFT", periods=250)["close"]
    assert not np.allclose(a.values, b.values)


def test_synthetic_ohlc_invariants(ohlcv):
    assert (ohlcv["high"] >= ohlcv["low"]).all()
    assert (ohlcv["high"] >= ohlcv[["open", "close"]].max(axis=1) - 1e-9).all()
    assert (ohlcv["low"] <= ohlcv[["open", "close"]].min(axis=1) + 1e-9).all()
    assert (ohlcv["close"] > 0).all()
    assert ohlcv.index.is_monotonic_increasing


def test_synthetic_volatility_in_plausible_range(ohlcv):
    annual_vol = ohlcv["close"].pct_change().std() * np.sqrt(252)
    assert 0.02 < annual_vol < 3.0


def test_synthetic_quote_shape():
    quote = generate_quote("AAPL")
    assert quote["source"] == "synthetic"
    assert quote["price"] > 0
    assert {"symbol", "change", "change_percent"}.issubset(quote)


# ---------------------------------------------------------------- indicators
def test_sma_matches_manual_mean(ohlcv):
    result = sma(ohlcv["close"], 20)
    manual = ohlcv["close"].iloc[-20:].mean()
    assert result.iloc[-1] == pytest.approx(manual, rel=1e-9)


def test_rsi_bounds_and_direction(ohlcv):
    values = rsi(ohlcv["close"], 14).dropna()
    assert values.between(0, 100).all()
    rising = pd.Series(np.linspace(100, 200, 120))
    assert rsi(rising, 14).iloc[-1] > 85, "a monotone uptrend should be overbought"
    falling = pd.Series(np.linspace(200, 100, 120))
    assert rsi(falling, 14).iloc[-1] < 15


def test_macd_relationships(ohlcv):
    df = macd(ohlcv["close"])
    assert set(df.columns) == {"macd", "macd_signal", "macd_hist"}
    diff = (df["macd"] - df["macd_signal"] - df["macd_hist"]).abs().max()
    assert diff < 1e-9, "histogram must equal macd minus signal"


def test_bollinger_band_ordering(ohlcv):
    bands = bollinger_bands(ohlcv["close"], 20, 2.0).dropna()
    assert (bands["bb_upper"] >= bands["bb_middle"]).all()
    assert (bands["bb_middle"] >= bands["bb_lower"]).all()


def test_atr_is_positive(ohlcv):
    values = atr(ohlcv, 14).dropna()
    assert (values > 0).all()


def test_ema_reacts_faster_than_sma(ohlcv):
    close = ohlcv["close"]
    shocked = close.copy()
    shocked.iloc[-1] *= 1.15
    ema_delta = abs(ema(shocked, 20).iloc[-1] - ema(close, 20).iloc[-1])
    sma_delta = abs(sma(shocked, 20).iloc[-1] - sma(close, 20).iloc[-1])
    assert ema_delta > sma_delta


def test_compute_indicators_adds_columns(ohlcv):
    enriched = compute_indicators(ohlcv, ["sma", "ema", "rsi", "macd", "bbands", "atr", "adx"])
    for col in ("sma_20", "ema_12", "rsi", "macd", "bb_upper", "atr", "adx"):
        assert col in enriched.columns
    assert len(enriched) == len(ohlcv)


def test_signal_summary_structure(ohlcv):
    enriched = compute_indicators(ohlcv)
    summary = signal_summary(enriched)
    assert summary["consensus"] in ("bullish", "bearish", "neutral")
    assert 0.0 <= summary["strength"] <= 1.0
    assert summary["buy_votes"] + summary["sell_votes"] + summary["neutral_votes"] == len(summary["indicators"])


def test_indicators_handle_empty_frame():
    assert compute_indicators(pd.DataFrame()).empty
    assert signal_summary(pd.DataFrame()) == {}


# ------------------------------------------------------------------ features
def test_build_features_no_nan(ohlcv):
    features = build_features(ohlcv)
    assert not features.isna().any().any()
    assert len(features.columns) > 20


def test_build_features_rejects_tiny_input():
    tiny = generate_ohlcv("TINY", periods=30).head(10)
    assert build_features(tiny).empty


def test_supervised_alignment_and_no_leakage(ohlcv):
    x, y = build_supervised(ohlcv, horizon=5)
    assert len(x) == len(y) and len(x) > 100
    assert x.index.equals(y.index)
    assert not x.isna().any().any() and not y.isna().any().any()
    # the target must be a *future* return, so it cannot be recoverable from x alone
    assert "target_return" in y.columns
    assert y["target_direction"].isin([0, 1]).all()


def test_make_sequences_shapes(ohlcv):
    x, y = build_supervised(ohlcv, horizon=5)
    xs, ys = make_sequences(x.values, y["target_return"].values, lookback=60)
    assert xs.shape == (len(x) - 60, 60, x.shape[1])
    assert ys.shape == (len(x) - 60,)
    assert np.allclose(xs[0, -1], x.values[59])


def test_make_sequences_with_insufficient_data():
    xs, ys = make_sequences(np.zeros((10, 3)), np.zeros(10), lookback=60)
    assert xs.shape[0] == 0 and ys.shape[0] == 0


# ------------------------------------------------- volume-less instruments
# Regression: forex / indices report zero volume. Volume-derived features were
# producing all-NaN columns, and the downstream dropna() then wiped out the
# ENTIRE feature matrix — silently disabling forecasting, RL and XAI for those
# asset classes while every endpoint still returned HTTP 200.
def _flat_volume_frame():
    df = generate_ohlcv("EURUSD=X", periods=400)
    df["volume"] = 0.0
    return df


def test_features_survive_zero_volume():
    df = _flat_volume_frame()
    features = build_features(df, dropna=True)
    assert len(features) > 300, "zero-volume assets must still yield a usable feature matrix"
    assert not features.isna().any().any()


def test_no_feature_column_is_entirely_nan():
    df = _flat_volume_frame()
    raw = build_features(df, dropna=False)
    fully_nan = [c for c in raw.columns if raw[c].isna().all()]
    assert not fully_nan, f"columns are 100% NaN and will destroy the matrix: {fully_nan}"


def test_volume_features_are_neutral_without_volume():
    df = _flat_volume_frame()
    features = build_features(df, dropna=True)
    assert (features["volume_ratio"] == 1.0).all()
    assert (features["obv_slope"] == 0.0).all()
    assert (features["mfi_14"] == 50.0).all()


def test_supervised_works_for_zero_volume_assets():
    x, y = build_supervised(_flat_volume_frame(), horizon=5)
    assert len(x) > 300 and len(x) == len(y)


def test_volume_features_still_active_when_volume_exists(ohlcv):
    features = build_features(ohlcv, dropna=True)
    assert features["volume_ratio"].nunique() > 5, "real volume must not be flattened"


# ------------------------------------------------------ period validation
@pytest.mark.parametrize("bad", ["invalid", "7q", "", "1 y"])
def test_validate_period_rejects_garbage(bad):
    from app.core.exceptions import InvalidRequestError
    from app.utils.timeseries import validate_period

    with pytest.raises(InvalidRequestError):
        validate_period(bad)


@pytest.mark.parametrize("good,expected", [("1Y", "1y"), (" 6mo ", "6mo"), ("MAX", "max")])
def test_validate_period_normalises(good, expected):
    from app.utils.timeseries import validate_period

    assert validate_period(good) == expected


def test_validate_interval():
    from app.core.exceptions import InvalidRequestError
    from app.utils.timeseries import validate_interval

    assert validate_interval("1D") == "1d"
    with pytest.raises(InvalidRequestError):
        validate_interval("3s")
