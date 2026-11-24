"""Tests for quantstats.validity (backtest validity analytics)."""

import json
import math

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

import quantstats as qs
from quantstats import validity as v


def _series(n=1260, mean=0.0006, vol=0.01, seed=1, start="2018-01-01"):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n)
    return pd.Series(rng.normal(mean, vol, n), index=idx, name="strat")


class TestSharpeSignificance:
    def test_dsr_with_one_trial_equals_stats_psr(self):
        r = _series()
        dsr = v.deflated_sharpe_ratio(r, n_trials=1)
        expected = qs.stats.probabilistic_sharpe_ratio(r)
        assert dsr["dsr"] == pytest.approx(expected, rel=1e-9)
        assert dsr["psr"] == pytest.approx(expected, rel=1e-9)
        assert dsr["sharpe_hurdle"] == 0.0

    def test_dsr_decreases_with_trials(self):
        r = _series()
        values = [v.deflated_sharpe_ratio(r, n_trials=n)["dsr"] for n in (1, 10, 100)]
        assert values[0] > values[1] > values[2]

    def test_published_worked_example(self):
        """Bailey & López de Prado (2014): N=100, T=1250, SR 2.5 annualised,
        V[SR]=0.5 annualised, skew -3, kurtosis 10 → SR0≈0.1132, DSR≈0.90."""
        sr = 2.5 / math.sqrt(250)
        hurdle = v.expected_max_sharpe(100, math.sqrt(0.5 / 250))
        assert hurdle == pytest.approx(0.1132, abs=2e-4)
        dsr = v.probabilistic_sharpe_ratio(sr, 1250, -3.0, 10.0, sr_benchmark=hurdle)
        assert dsr == pytest.approx(0.9004, abs=2e-3)

    def test_min_track_record_is_psr_inverse(self):
        sr, skew, kurt = 0.08, -0.7, 5.0
        n = v.min_track_record_length(sr, skew, kurt, confidence=0.95)
        # PSR's standard error uses n - 1, as does MinTRL's leading 1
        z = sr / math.sqrt((1 - skew * sr + (kurt - 1) / 4 * sr**2) / (n - 1))
        assert norm.cdf(z) == pytest.approx(0.95, abs=1e-9)

    def test_min_track_record_infinite_below_benchmark(self):
        assert v.min_track_record_length(-0.01) == float("inf")
        assert v.min_track_record_length(0.05, sr_benchmark=0.06) == float("inf")

    def test_selection_false_positive_rate_is_controlled(self):
        """Best of N zero-skill strategies should rarely pass the deflated test."""
        rng = np.random.default_rng(0)
        n_trials, t, reps = 20, 500, 300
        passes = naive = 0
        for _ in range(reps):
            mat = rng.normal(0.0, 0.01, (t, n_trials))
            sr = mat.mean(axis=0) / mat.std(axis=0, ddof=1)
            best = mat[:, sr.argmax()]
            res = v.deflated_sharpe_ratio(best, n_trials=n_trials)
            passes += res["dsr"] > 0.95
            naive += res["psr"] > 0.95
        assert passes / reps <= 0.07
        # ...whereas ignoring selection flags most of them
        assert naive / reps > 0.4


class TestBootstrap:
    def test_interval_contains_observed_and_is_reproducible(self):
        r = _series()
        a = v.bootstrap_confidence_intervals(r, n_boot=400, seed=3)
        b = v.bootstrap_confidence_intervals(r, n_boot=400, seed=3)
        for key in ("sharpe", "cagr", "volatility", "max_drawdown"):
            assert a[key]["low"] <= a[key]["observed"] <= a[key]["high"]
            assert a[key] == b[key]
        assert a["max_drawdown"]["high"] <= 0

    def test_observed_matches_stats(self):
        r = _series()
        res = v.bootstrap_confidence_intervals(r, n_boot=50)
        assert res["sharpe"]["observed"] == pytest.approx(qs.stats.sharpe(r), rel=1e-9)
        assert res["max_drawdown"]["observed"] == pytest.approx(
            qs.stats.max_drawdown(r), rel=1e-9
        )

    def test_indices_form_contiguous_blocks(self):
        rng = np.random.default_rng(0)
        idx = v.core._stationary_bootstrap_indices(200, 50, 10.0, rng)
        assert idx.shape == (50, 200)
        assert idx.min() >= 0 and idx.max() < 200
        step = (np.diff(idx, axis=1) % 200) == 1
        # mean block of 10 → about 90% of steps continue the current block
        assert 0.85 < step.mean() < 0.95


class TestSplit:
    def test_detects_degradation(self):
        good = _series(n=756, mean=0.0015, seed=2)
        bad = _series(n=504, mean=-0.0005, seed=3, start="2021-01-01")
        r = pd.concat([good, bad])
        r.index = pd.bdate_range("2018-01-01", periods=len(r))
        res = v.split_analysis(r, r.index[756])
        assert res["in_sample"]["n"] == 756
        assert res["out_of_sample"]["n"] == 504
        assert res["sharpe_change_z"] < 0
        assert res["degradation_p_value"] < 0.05

    def test_fraction_split(self):
        r = _series(n=1000)
        res = v.split_analysis(r, 0.3)
        assert res["out_of_sample"]["n"] == 300

    def test_rejects_tiny_segments_and_int(self):
        r = _series(n=200)
        with pytest.raises(ValueError, match="at least"):
            v.split_analysis(r, r.index[-5])
        with pytest.raises(ValueError, match="fraction"):
            v.split_analysis(r, 100)


class TestDecay:
    def test_recovers_alpha_and_beta(self):
        rng = np.random.default_rng(7)
        idx = pd.bdate_range("2015-01-01", periods=2520)
        b = pd.Series(rng.normal(0.0003, 0.01, 2520), index=idx)
        r = 0.0004 + 0.8 * b + rng.normal(0, 0.004, 2520)
        res = v.decay_analysis(r, b)
        assert res["kind"] == "alpha"
        assert res["beta"] == pytest.approx(0.8, abs=0.03)
        # standard error of the annualised alpha here is about 0.02
        assert res["edge_annual"] == pytest.approx(0.0004 * 252, abs=0.05)
        assert res["edge_t_stat"] > 2

    def test_detects_linear_decay(self):
        rng = np.random.default_rng(6)
        n = 2520
        idx = pd.bdate_range("2015-01-01", periods=n)
        drift = np.linspace(0.002, -0.0005, n)
        r = pd.Series(drift + rng.normal(0, 0.01, n), index=idx)
        res = v.decay_analysis(r)
        assert res["kind"] == "mean_return"
        assert res["trend_t_stat"] < -1.96
        # 0.0025/day lost over 10 years ≈ -6.3% annualised edge per year
        assert res["trend_annual_per_year"] == pytest.approx(-0.063, abs=0.03)

    def test_no_trend_on_stationary_series(self):
        res = v.decay_analysis(_series(n=2520, seed=11))
        assert abs(res["trend_t_stat"]) < 2.5

    def test_benchmark_without_overlap_raises(self):
        r = _series(n=100)
        b = _series(n=100, start="2030-01-01")
        with pytest.raises(ValueError, match="shared"):
            v.decay_analysis(r, b)


class TestPBO:
    def test_noise_gives_about_half(self):
        rng = np.random.default_rng(7)
        trials = pd.DataFrame(rng.normal(0, 0.01, (1600, 40)))
        res = v.probability_of_backtest_overfitting(trials, n_splits=10)
        assert 0.3 < res["pbo"] < 0.7
        assert res["n_combinations"] == math.comb(10, 5)

    def test_genuine_winner_gives_low_pbo(self):
        rng = np.random.default_rng(8)
        data = rng.normal(0, 0.01, (1600, 40))
        data[:, 0] += 0.002
        res = v.probability_of_backtest_overfitting(pd.DataFrame(data), n_splits=10)
        assert res["pbo"] < 0.05
        assert res["prob_oos_loss"] < 0.05

    def test_requires_dataframe_with_columns(self):
        with pytest.raises(ValueError, match="one column"):
            v.probability_of_backtest_overfitting(pd.DataFrame({"a": [0.1, 0.2]}))


class TestValidate:
    def test_full_run_and_strict_json(self):
        rng = np.random.default_rng(9)
        r = _series(n=1260, seed=9)
        bench = pd.Series(rng.normal(0.0003, 0.01, 1260), index=r.index)
        trials = pd.DataFrame(rng.normal(0, 0.01, (1260, 12)), index=r.index)
        trials[0] = r
        res = v.validate(r, bench, trials=trials, split=0.3, n_boot=200)
        d = res.to_dict()
        text = json.dumps(d, allow_nan=False)
        assert json.loads(text)["sharpe"]["n_trials"] == 12
        assert d["sample"]["start"].startswith("2018-01-01")
        assert d["overfitting"]["pbo"] is not None
        assert "logits" not in d["overfitting"]
        assert "samples" not in d["bootstrap"]
        assert d["split"]["out_of_sample"]["n"] == 378
        assert {f["severity"] for f in d["findings"]} <= {"info", "caution", "warning"}
        assert "Backtest validity" in res.summary()

    def test_missing_trials_is_flagged(self):
        res = v.validate(_series(), n_boot=100)
        codes = {f.code for f in res.findings}
        assert "N_TRIALS_NOT_PROVIDED" in codes
        assert "NO_OUT_OF_SAMPLE_SPLIT" in codes
        assert res.sharpe["dsr"] == pytest.approx(res.sharpe["psr"])

    def test_noise_strategy_is_not_endorsed(self):
        res = v.validate(_series(mean=0.0, seed=4), n_trials=50, n_boot=200)
        codes = {f.code for f in res.findings}
        assert "SHARPE_SIGNIFICANT_AFTER_SELECTION" not in codes
        assert codes & {
            "SHARPE_NOT_SIGNIFICANT_AFTER_SELECTION",
            "SHARPE_BELOW_SELECTION_HURDLE",
            "SHARPE_NOT_POSITIVE",
        }

    def test_strict_json_with_degenerate_values(self):
        """Constant returns give NaN Sharpe; the dict must still be valid JSON."""
        idx = pd.bdate_range("2020-01-01", periods=60)
        res = v.validate(pd.Series(0.0, index=idx), n_boot=20)
        d = json.loads(res.to_json())
        assert d["sharpe"]["sharpe"] is None

    def test_rejects_prices(self):
        idx = pd.bdate_range("2020-01-01", periods=100)
        prices = pd.Series(np.linspace(100, 120, 100), index=idx)
        with pytest.raises(ValueError, match="price series"):
            v.validate(prices)

    def test_requires_periods_without_dates(self):
        r = pd.Series(np.random.default_rng(0).normal(0, 0.01, 100))
        with pytest.raises(ValueError, match="periods"):
            v.validate(r)
        assert v.validate(r, periods=252, n_boot=20).periods == 252

    def test_too_short(self):
        with pytest.raises(ValueError, match="at least 30"):
            v.validate(_series(n=10))

    def test_infers_monthly_and_crypto_periods(self):
        rng = np.random.default_rng(1)
        monthly = pd.Series(
            rng.normal(0.01, 0.04, 120),
            index=pd.date_range("2010-01-01", periods=120, freq="MS"),
        )
        assert v.validate(monthly, n_boot=20).periods == 12
        daily = pd.Series(
            rng.normal(0.001, 0.03, 400),
            index=pd.date_range("2022-01-01", periods=400, freq="D"),
        )
        assert v.validate(daily, n_boot=20).periods == 365

    def test_rf_reduces_sharpe(self):
        r = _series()
        a = v.validate(r, n_boot=20).sharpe["sharpe"]
        b = v.validate(r, rf=0.05, n_boot=20).sharpe["sharpe"]
        assert b < a
