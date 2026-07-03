"""
Tests for the multi-model Monte Carlo engine (quantstats.montecarlo).

Stochastic models are tested deterministically via a seeded RNG: we assert on
output shape, reproducibility and that recovered moments are within tolerance of
the calibrated parameters.
"""

import numpy as np
import pandas as pd
import pytest

from quantstats.montecarlo import analytics, run_models
from quantstats.montecarlo.core import infer_periods_per_year
from quantstats.montecarlo.registry import available_models, get_model


@pytest.fixture
def gbm_like_returns():
    """Returns generated from a known GBM so calibration is checkable."""
    rng = np.random.default_rng(0)
    n = 2000
    mu, sigma = 0.0004, 0.01
    log_r = rng.normal(mu, sigma, n)
    simple = np.expm1(log_r)
    dates = pd.date_range("2015-01-01", periods=n, freq="B")
    return pd.Series(simple, index=dates, name="Asset")


ALL_MODELS = [
    "gbm",
    "shuffle",
    "bootstrap",
    "block_bootstrap",
    "trimmed_1pct",
    "jump_diffusion",
    "garch",
    "heston",
    "bayesian",
    "bayesian_bootstrap",
]

MC_MODELS = [m for m in ALL_MODELS if m != "trimmed_1pct"]
STRESS_MODELS = ["trimmed_1pct"]


class TestRegistry:
    def test_all_models_registered(self):
        names = available_models()
        for m in ALL_MODELS:
            assert m in names
        assert "trimmed_5pct" not in names

    def test_unknown_model_raises(self):
        with pytest.raises(KeyError):
            get_model("does_not_exist")

    def test_trimmed_model_is_stress_category(self):
        assert get_model("trimmed_1pct").category == "stress"

    def test_gbm_is_montecarlo_category(self):
        assert get_model("gbm").category == "montecarlo"


class TestModelInterface:
    @pytest.mark.parametrize("name", ALL_MODELS)
    def test_simulate_shape(self, name, gbm_like_returns):
        model = get_model(name).calibrate(gbm_like_returns)
        rng = np.random.default_rng(123)
        out = model.simulate(horizon=50, sims=20, rng=rng)
        assert out.shape == (50, 20)
        assert np.isfinite(out).all()

    @pytest.mark.parametrize("name", ALL_MODELS)
    def test_reproducible_with_seed(self, name, gbm_like_returns):
        model = get_model(name).calibrate(gbm_like_returns)
        a = model.simulate(50, 20, np.random.default_rng(7))
        b = model.simulate(50, 20, np.random.default_rng(7))
        np.testing.assert_allclose(a, b)

    def test_simulate_before_calibrate_raises(self):
        model = get_model("gbm")
        with pytest.raises(RuntimeError):
            model.simulate(10, 10, np.random.default_rng(0))


class TestGBMCalibration:
    def test_recovers_moments(self, gbm_like_returns):
        model = get_model("gbm").calibrate(gbm_like_returns)
        assert model.mu_ == pytest.approx(0.0004, abs=5e-4)
        assert model.sigma_ == pytest.approx(0.01, rel=0.1)

    def test_simulated_mean_close_to_calibrated(self, gbm_like_returns):
        model = get_model("gbm").calibrate(gbm_like_returns)
        out = model.simulate(2000, 200, np.random.default_rng(1))
        log_out = np.log1p(out)
        assert log_out.mean() == pytest.approx(model.mu_, abs=2e-3)
        assert log_out.std() == pytest.approx(model.sigma_, rel=0.1)


class TestDriftConsistency:
    """Simulated median CAGR should track full-sample historical drift."""

    @pytest.mark.parametrize("name", MC_MODELS)
    def test_median_cagr_near_historical(self, name, gbm_like_returns):
        horizon = 252
        results = run_models(
            gbm_like_returns,
            models=[name],
            sims=500,
            horizon=horizon,
            seed=42,
            periods=252.0,
        )
        res = results[name]
        log_r = np.log1p(gbm_like_returns.to_numpy())
        expected_cagr = float(np.expm1(np.mean(log_r) * 252))
        sim_cagr = res.summary["cagr_median"]
        tol = 0.20 if name in ("heston", "jump_diffusion", "garch") else 0.12
        assert sim_cagr == pytest.approx(expected_cagr, abs=tol)


class TestAnalytics:
    def test_terminal_and_cumulative_consistent(self):
        sim = np.array([[0.1, 0.0], [0.0, 0.1]])
        term = analytics.terminal_values(sim)
        np.testing.assert_allclose(term, [0.1, 0.1])
        cum = analytics.cumulative_paths(sim)
        np.testing.assert_allclose(cum[-1], term)

    def test_max_drawdown_negative(self):
        sim = np.array([[0.1], [-0.5], [0.1]])
        mdd = analytics.max_drawdowns(sim)
        assert mdd[0] < 0

    def test_bust_and_goal_probabilities(self):
        up = np.full((100, 1), 0.01)
        down = np.full((100, 1), -0.05)
        sim = np.concatenate([up, down], axis=1)
        assert analytics.goal_probability(sim, goal=0.5) == pytest.approx(0.5)
        assert analytics.bust_probability(sim, bust=-0.5) == pytest.approx(0.5)

    def test_fan_chart_ordering(self):
        rng = np.random.default_rng(0)
        sim = rng.normal(0.001, 0.01, size=(100, 500))
        median, lower, upper = analytics.fan_chart(sim, level=0.9)
        assert (lower <= median).all()
        assert (median <= upper).all()

    def test_summarize_keys(self):
        rng = np.random.default_rng(0)
        sim = rng.normal(0.001, 0.01, size=(252, 200))
        s = analytics.summarize(sim, bust=-0.25, goal=0.1)
        for key in (
            "cagr_p5",
            "cagr_median",
            "cagr_p95",
            "maxdd_median",
            "maxdd_p95",
            "terminal_median",
            "prob_loss",
            "cvar_5",
            "bust_prob",
            "goal_prob",
        ):
            assert key in s

    def test_cvar_is_tail_mean(self):
        values = np.array([-0.5, -0.3, -0.1, 0.0, 0.1, 0.2, 0.3])
        cvar = analytics.cvar(values, q=0.05)
        assert cvar <= float(np.quantile(values, 0.05))

    def test_cvar_at_one_year_horizon_is_loss_tail(self):
        rng = np.random.default_rng(0)
        sim = rng.normal(-0.001, 0.02, size=(252, 1000))
        s = analytics.summarize(sim, periods=252.0)
        assert s["cvar_5"] <= s["cagr_p5"]

    def test_historical_windows_shape(self, gbm_like_returns):
        windows = analytics.historical_windows(gbm_like_returns, 252)
        n = len(gbm_like_returns)
        assert windows.shape == (252, n - 252 + 1)
        # First column is the first 252 returns; last column the trailing 252.
        arr = gbm_like_returns.to_numpy()
        np.testing.assert_allclose(windows[:, 0], arr[:252])
        np.testing.assert_allclose(windows[:, -1], arr[-252:])

    def test_historical_windows_short_history(self):
        s = pd.Series(np.full(100, 0.001))
        windows = analytics.historical_windows(s, 252)
        assert windows.shape == (100, 1)

    def test_historical_summary_uses_rolling_distribution(self, gbm_like_returns):
        h = analytics.historical_summary(gbm_like_returns, 252, periods=252.0)
        windows = analytics.historical_windows(gbm_like_returns, 252)
        direct = analytics.summarize(windows, periods=252.0)
        assert h["cagr_median"] == pytest.approx(direct["cagr_median"])
        # A real distribution: p5 strictly below p95.
        assert h["cagr_p5"] < h["cagr_p95"]

    def test_realism_percentile_bounds(self):
        rng = np.random.default_rng(0)
        sim = rng.normal(0.001, 0.01, size=(252, 500))
        pct = analytics.realism_percentile(sim, historical_terminal=0.05)
        assert 0.0 <= pct <= 100.0

    def test_conservative_envelope(self, gbm_like_returns):
        results = run_models(
            gbm_like_returns,
            models=["gbm", "bootstrap", "trimmed_1pct"],
            sims=100,
            horizon=120,
            seed=1,
        )
        env = analytics.conservative_envelope(results)
        mc_summaries = [r.summary for r in results.values() if r.category == "montecarlo"]
        assert env["cagr_p5"] == min(s["cagr_p5"] for s in mc_summaries)
        assert env["prob_loss"] == max(s["prob_loss"] for s in mc_summaries)

    def test_model_median_summary(self, gbm_like_returns):
        results = run_models(
            gbm_like_returns,
            models=["gbm", "bootstrap", "trimmed_1pct"],
            sims=100,
            horizon=120,
            seed=1,
        )
        med = analytics.model_median_summary(results)
        mc_summaries = [r.summary for r in results.values() if r.category == "montecarlo"]
        assert med["cagr_p5"] == pytest.approx(
            float(np.median([s["cagr_p5"] for s in mc_summaries]))
        )
        assert med["prob_loss"] == pytest.approx(
            float(np.median([s["prob_loss"] for s in mc_summaries]))
        )

    def test_envelope_attribution(self, gbm_like_returns):
        results = run_models(
            gbm_like_returns,
            models=["gbm", "bootstrap"],
            sims=100,
            horizon=120,
            seed=1,
        )
        env = analytics.conservative_envelope(results)
        attr = analytics.envelope_attribution(results, env)
        assert "cagr_p5" in attr
        assert attr["cagr_p5"] in ("GBM", "Bootstrap")


class TestRunModels:
    def test_runs_all_by_default(self, gbm_like_returns):
        results = run_models(gbm_like_returns, sims=50, horizon=60, seed=42)
        assert set(results) == set(available_models())
        for res in results.values():
            assert res.sim_returns.shape == (60, 50)
            assert "cagr_p5" in res.summary

    def test_subset_and_thresholds(self, gbm_like_returns):
        results = run_models(
            gbm_like_returns,
            models=["gbm", "bootstrap"],
            sims=100,
            horizon=120,
            bust=-0.2,
            goal=0.1,
            seed=1,
        )
        assert list(results) == ["gbm", "bootstrap"]
        for res in results.values():
            s = res.summary
            assert "bust_prob" in s and 0.0 <= s["bust_prob"] <= 1.0
            assert "goal_prob" in s and 0.0 <= s["goal_prob"] <= 1.0

    def test_default_horizon_is_one_year(self, gbm_like_returns):
        results = run_models(gbm_like_returns, models=["gbm"], sims=10, seed=0)
        assert results["gbm"].horizon == 252

    def test_reproducible(self, gbm_like_returns):
        r1 = run_models(gbm_like_returns, models=["gbm"], sims=50, horizon=60, seed=99)
        r2 = run_models(gbm_like_returns, models=["gbm"], sims=50, horizon=60, seed=99)
        np.testing.assert_allclose(r1["gbm"].sim_returns, r2["gbm"].sim_returns)

    def test_result_carries_category(self, gbm_like_returns):
        results = run_models(
            gbm_like_returns, models=["gbm", "trimmed_1pct"], sims=10, horizon=60
        )
        assert results["gbm"].category == "montecarlo"
        assert results["trimmed_1pct"].category == "stress"

    def test_calibration_summary_available(self, gbm_like_returns):
        results = run_models(
            gbm_like_returns, models=["gbm"], sims=10, horizon=60, seed=0
        )
        cal = results["gbm"].calibration_summary()
        assert "Drift (ann.)" in cal


class TestInferPeriods:
    def test_weekday_returns_use_252(self, gbm_like_returns):
        assert infer_periods_per_year(gbm_like_returns) == 252.0

    def test_weekend_returns_use_365(self):
        dates = pd.date_range("2020-01-01", periods=400, freq="D")
        s = pd.Series(np.random.default_rng(0).normal(0, 0.01, 400), index=dates)
        assert infer_periods_per_year(s) == 365.0


class TestDrift:
    def test_zero_drift_lowers_cagr_p5(self, gbm_like_returns):
        hist = run_models(
            gbm_like_returns, models=["gbm"], sims=500, horizon=252,
            seed=5, drift="historical",
        )["gbm"]
        zero = run_models(
            gbm_like_returns, models=["gbm"], sims=500, horizon=252,
            seed=5, drift="zero",
        )["gbm"]
        assert hist.summary["cagr_p5"] > zero.summary["cagr_p5"]

    def test_zero_drift_gbm_mu_near_zero(self, gbm_like_returns):
        from quantstats.montecarlo.core import _apply_drift

        adj = _apply_drift(gbm_like_returns, "zero")
        model = get_model("gbm").calibrate(adj)
        assert model.mu_ == pytest.approx(0.0, abs=1e-9)

    def test_invalid_drift_raises(self, gbm_like_returns):
        with pytest.raises(ValueError):
            run_models(gbm_like_returns, models=["gbm"], sims=10, drift="bogus")


class TestBlockLength:
    def test_politis_white_returns_positive_int(self, gbm_like_returns):
        from quantstats.montecarlo.models.block_bootstrap import (
            politis_white_block_length,
        )

        arr = gbm_like_returns.to_numpy()
        b = politis_white_block_length(arr)
        assert isinstance(b, int)
        assert b >= 1

    def test_block_model_uses_politis_white_by_default(self, gbm_like_returns):
        model = get_model("block_bootstrap").calibrate(gbm_like_returns)
        assert model.block_length_ >= 1
