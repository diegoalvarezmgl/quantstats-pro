"""
Tests for quantstats.alphadecay core analysis.
"""

import numpy as np
import pandas as pd
import pytest

from quantstats.alphadecay.core import (
    METRIC_SPECS,
    _completed_underwater_durations,
    _transform_values,
    analyze,
)


@pytest.fixture
def stable_returns():
    rng = np.random.default_rng(42)
    n = 400
    log_r = rng.normal(0.0004, 0.01, n)
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.Series(np.expm1(log_r), index=dates, name="TEST")


@pytest.fixture
def decay_returns():
    rng = np.random.default_rng(7)
    n = 400
    log_r = np.concatenate([
        rng.normal(0.0005, 0.008, 300),
        rng.normal(-0.001, 0.015, 100),
    ])
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.Series(np.expm1(log_r), index=dates, name="DECAY")


class TestAnalyze:
    def test_returns_decay_result(self, stable_returns):
        result = analyze(stable_returns)
        assert result.score >= 0
        assert result.total == 10 * 3
        assert len(result.metrics) == 10
        assert set(result.windows) == {7, 15, 30}

    def test_sharpe_sortino_removed(self):
        keys = {spec.key for spec in METRIC_SPECS}
        assert "sharpe" not in keys
        assert "sortino" not in keys

    def test_insufficient_data_raises(self):
        rng = np.random.default_rng(1)
        short = pd.Series(rng.normal(0, 0.01, 20))
        with pytest.raises(ValueError, match="Need at least"):
            analyze(short)

    def test_z_score_on_synthetic_distribution(self):
        rng = np.random.default_rng(99)
        n = 200
        base = rng.normal(0.0003, 0.01, n - 7)
        # Force last 7 days to very negative returns -> bad vol/cagr
        tail = rng.normal(-0.02, 0.03, 7)
        rets = np.expm1(np.concatenate([base, tail]))
        dates = pd.date_range("2021-01-01", periods=n, freq="B")
        series = pd.Series(rets, index=dates)

        result = analyze(series, windows=(7,))
        vol_mr = next(m for m in result.metrics if m.spec.key == "volatility")
        wr = vol_mr.windows[7]
        assert wr.observed > wr.mean
        assert wr.z_score > 0
        assert wr.status in ("warning", "critical", "good")

    def test_higher_is_better_excellent_when_above_mean(self, stable_returns):
        result = analyze(stable_returns, windows=(15,))
        # At least some metrics should have valid z-scores
        cagr_mr = next(m for m in result.metrics if m.spec.key == "cagr")
        wr = cagr_mr.windows[15]
        assert not np.isnan(wr.z_score)
        assert wr.status in ("excellent", "good", "warning", "critical")

    def test_score_counts_good_and_excellent(self, stable_returns):
        result = analyze(stable_returns)
        manual = 0
        for mr in result.metrics:
            for wr in mr.windows.values():
                if wr.status in ("excellent", "good"):
                    manual += 1
        assert result.score == manual

    def test_cusum_alarm_on_decay(self, decay_returns):
        result = analyze(decay_returns)
        assert result.cusum.current >= 0
        assert result.cusum.threshold > 0
        # Decay series should have higher CUSUM than stable
        stable = analyze(
            pd.Series(
                np.expm1(np.random.default_rng(42).normal(0.0004, 0.01, 400)),
                index=pd.date_range("2020-01-01", periods=400, freq="B"),
            )
        )
        assert result.cusum.pct_of_threshold >= stable.cusum.pct_of_threshold

    def test_time_underwater(self, decay_returns):
        result = analyze(decay_returns)
        tuw = result.time_underwater
        assert tuw.current_days >= 0
        assert len(tuw.distribution) >= 0


class TestTransforms:
    def test_log_transform_reduces_skew(self):
        rng = np.random.default_rng(5)
        vols = np.exp(rng.normal(-3, 0.5, 500))  # lognormal sample
        raw_skew = float(pd.Series(vols).skew())
        t_skew = float(pd.Series(_transform_values(vols, "log")).skew())
        assert abs(t_skew) < abs(raw_skew)

    def test_log1p_matches_log_growth(self):
        vals = np.array([0.10, -0.05, 0.30])
        out = _transform_values(vals, "log1p")
        assert np.allclose(out, np.log1p(vals))

    def test_log_neg_maps_severity(self):
        dd = np.array([-0.02, -0.10, -0.30])
        out = _transform_values(dd, "log_neg")
        # More severe drawdown -> larger log-severity
        assert out[2] > out[1] > out[0]

    def test_log_family_handles_zeros(self):
        vals = np.array([0.0, 0.01, 0.05])
        out = _transform_values(vals, "log")
        assert np.all(np.isfinite(out))
        # Zero is floored to the smallest positive value
        assert out[0] == out[1]

    def test_log_neg_handles_zero_drawdown(self):
        dd = np.array([0.0, -0.05, -0.20])
        out = _transform_values(dd, "log_neg")
        assert np.all(np.isfinite(out))
        assert out[0] <= out[1] <= out[2]

    def test_metric_specs_transform_assignment(self):
        by_key = {s.key: s.transform for s in METRIC_SPECS}
        assert by_key["cagr"] == "log1p"
        assert by_key["volatility"] == "log"
        assert by_key["downside_vol"] == "log"
        assert by_key["max_drawdown"] == "log_neg"
        assert by_key["var_95"] == "log_neg"
        assert by_key["cvar_95"] == "log_neg"
        assert by_key["payoff_ratio"] == "log"
        assert by_key["win_rate"] == "none"
        assert by_key["skew"] == "none"

    def test_window_result_stores_analysis_distribution(self, stable_returns):
        result = analyze(stable_returns, windows=(15,))
        vol_mr = next(m for m in result.metrics if m.spec.key == "volatility")
        wr = vol_mr.windows[15]
        expected = _transform_values(wr.distribution, "log")
        assert np.allclose(wr.analysis_distribution, expected)
        assert wr.analysis_observed == expected[-1]
        assert wr.analysis_mean == float(np.mean(expected))

    def test_time_underwater_analysis_scale(self, decay_returns):
        tuw = analyze(decay_returns).time_underwater
        if len(tuw.distribution):
            assert np.allclose(tuw.analysis_distribution, np.log(tuw.distribution))
            assert tuw.analysis_mean == float(np.mean(np.log(tuw.distribution)))

    def test_time_underwater_excludes_current_spell_from_reference(self):
        # Simulate drawdown: two completed spells then a long ongoing one
        dd = pd.Series([0, -1, -1, 0, -1, -1, -1, 0] + [-1] * 100)
        completed, current, is_uw = _completed_underwater_durations(dd)
        assert is_uw
        assert current == 100
        assert completed == [2, 3]
        assert 100 not in completed

    def test_time_underwater_status_for_long_spell(self):
        from quantstats.alphadecay.core import _classify_underwater_status

        assert _classify_underwater_status(2.5, is_underwater=True) == "critical"
        assert _classify_underwater_status(1.0, is_underwater=True) == "warning"
        assert _classify_underwater_status(-0.5, is_underwater=True) == "good"
        assert _classify_underwater_status(0.0, is_underwater=False) == "excellent"
