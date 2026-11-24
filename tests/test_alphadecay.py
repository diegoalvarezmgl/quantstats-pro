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
        # reference excludes the 15 rolling values that overlap the current window
        assert wr.analysis_mean == float(np.mean(expected[:-15]))

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

        # Thresholds match the lower-is-better metric traffic lights
        assert _classify_underwater_status(3.5, is_underwater=True) == "critical"
        assert _classify_underwater_status(2.5, is_underwater=True) == "warning"
        assert _classify_underwater_status(0.89, is_underwater=True) == "good"
        assert _classify_underwater_status(-0.5, is_underwater=True) == "excellent"
        assert _classify_underwater_status(0.0, is_underwater=False) == "excellent"


class TestUnderwaterChartBounds:
    def test_caps_axis_when_long_tail_outlier(self):
        from quantstats.alphadecay.plotting import _underwater_hist_bounds

        dist = np.array([5, 8, 10, 12, 15, 20, 25, 30, 500], dtype=float)
        upper, hist, n_beyond, longest = _underwater_hist_bounds(
            dist, mean=13.0, current_days=12, is_underwater=True
        )
        assert upper < 500
        assert upper >= 15  # covers mean/current
        assert n_beyond == 1
        assert longest == 500
        assert 500 not in hist

    def test_extends_for_long_current_spell(self):
        from quantstats.alphadecay.plotting import _underwater_hist_bounds

        dist = np.array([5, 10, 15, 20, 25], dtype=float)
        upper, _, n_beyond, _ = _underwater_hist_bounds(
            dist, mean=15.0, current_days=102, is_underwater=True
        )
        assert upper >= 102 * 1.25
        assert n_beyond == 0


class TestVectorisedParity:
    """The vectorised fast path must match the reference quantstats formulas."""

    @pytest.mark.parametrize("window", [7, 15, 30])
    def test_matches_reference_implementation(self, window):
        from quantstats.alphadecay.core import (
            _rolling_metric_series,
            _rolling_metrics_matrix,
        )

        rng = np.random.default_rng(11)
        n = 90
        rets = pd.Series(
            np.expm1(rng.normal(0.0004, 0.012, n)),
            index=pd.date_range("2022-01-01", periods=n, freq="B"),
        )
        fast = _rolling_metrics_matrix(rets, window, periods=252.0)
        for spec in METRIC_SPECS:
            slow = (
                _rolling_metric_series(rets, window, spec.key, rf=0.0, periods=252.0)
                .to_numpy(dtype=float)[window - 1 :]
            )
            np.testing.assert_allclose(
                fast[spec.key],
                slow,
                rtol=1e-9,
                atol=1e-12,
                equal_nan=True,
                err_msg=f"metric={spec.key} window={window}",
            )

    def test_all_positive_window_edge_case(self):
        # No losses: downside vol = 0, max DD = 0, payoff = NaN, win rate = 1
        from quantstats.alphadecay.core import (
            _rolling_metric_series,
            _rolling_metrics_matrix,
        )

        rets = pd.Series([0.01] * 6 + [0.02] * 6)
        out = _rolling_metrics_matrix(rets, 6, periods=252.0)
        assert np.all(out["downside_vol"] == 0.0)
        assert np.all(out["max_drawdown"] == 0.0)
        assert np.all(np.isnan(out["payoff_ratio"]))
        assert np.all(out["win_rate"] == 1.0)
        # CVaR must match the reference even in the degenerate all-positive case
        # (constant windows -> std=0 -> VaR is NaN in both implementations)
        for key in ("var_95", "cvar_95"):
            slow = (
                _rolling_metric_series(rets, 6, key, rf=0.0, periods=252.0)
                .to_numpy(dtype=float)[5:]
            )
            np.testing.assert_allclose(
                out[key], slow, rtol=1e-9, equal_nan=True, err_msg=key
            )


class TestReferencePeriod:
    """0.5.0: the reference must not contain the period being judged."""

    @staticmethod
    def _decaying(n=600, seed=0):
        rng = np.random.default_rng(seed)
        idx = pd.bdate_range("2020-01-01", periods=n)
        half = n // 2
        vals = np.r_[rng.normal(0.002, 0.01, half), rng.normal(-0.001, 0.01, n - half)]
        return pd.Series(vals, index=idx, name="strat")

    def test_cusum_baseline_is_more_sensitive_than_full_sample(self):
        r = self._decaying()
        ref = analyze(r, baseline=0.5).cusum
        legacy = analyze(r, baseline=None).cusum
        assert ref.baseline_end == 300
        assert legacy.baseline_end is None
        # the full-sample target is dragged down by the decayed half
        assert ref.target_mean > legacy.target_mean
        assert ref.series.max() > legacy.series.max()

    def test_zscore_reference_excludes_current_window(self):
        rng = np.random.default_rng(1)
        idx = pd.bdate_range("2020-01-01", periods=400)
        vals = rng.normal(0.0005, 0.01, 400)
        vals[-30:] = rng.normal(0.0005, 0.05, 30)  # volatility shock at the end
        res = analyze(pd.Series(vals, index=idx), windows=(30,))
        wr = next(m for m in res.metrics if m.spec.key == "volatility").windows[30]
        full_std = float(np.std(wr.analysis_distribution, ddof=1))
        full_z = (wr.analysis_observed - float(np.mean(wr.analysis_distribution))) / full_std
        assert wr.z_score > full_z > 0

    def test_benchmark_removes_market_exposure(self):
        rng = np.random.default_rng(2)
        idx = pd.bdate_range("2020-01-01", periods=500)
        bench = pd.Series(rng.normal(0.0004, 0.012, 500), index=idx)
        strat = 1.5 * bench + rng.normal(0.0, 0.002, 500)
        res = analyze(strat, benchmark=bench)
        assert res.beta == pytest.approx(1.5, abs=0.05)
        assert res.returns.std() < strat.std() / 3

    def test_invalid_baseline(self):
        with pytest.raises(ValueError, match="baseline"):
            analyze(self._decaying(), baseline=1.5)

    def test_to_dict_is_strict_json(self):
        import json

        d = analyze(self._decaying()).to_dict()
        parsed = json.loads(json.dumps(d, allow_nan=False))
        assert parsed["health"]["total"] == 30
        assert parsed["cusum"]["baseline_observations"] == 300
        assert set(parsed["metrics"]["volatility"]["windows"]) == {"7", "15", "30"}
