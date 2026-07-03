"""
Tests for reports.html_montecarlo — the multi-model Monte Carlo tearsheet.
"""

import os
import re
import tempfile

import numpy as np
import pandas as pd
import pytest

from quantstats import reports


@pytest.fixture
def asset_returns():
    rng = np.random.default_rng(3)
    n = 800
    log_r = rng.normal(0.0003, 0.012, n)
    dates = pd.date_range("2018-01-01", periods=n, freq="B")
    return pd.Series(np.expm1(log_r), index=dates, name="QQQ")


def _render(returns, **kwargs):
    with tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False) as f:
        path = f.name
    try:
        reports.html_montecarlo(returns, output=path, **kwargs)
        with open(path) as fh:
            return fh.read()
    finally:
        if os.path.exists(path):
            os.remove(path)


class TestMonteCarloTearsheet:
    def test_generates_valid_html(self, asset_returns):
        content = _render(
            asset_returns, models=["gbm", "bootstrap"], sims=100, horizon=120, seed=1
        )
        assert "<!DOCTYPE html>" in content
        assert "Monte Carlo Tearsheet" in content
        assert not re.search(r"\{\{\s*\w+\s*\}\}", content)

    def test_contains_assessment_and_sections(self, asset_returns):
        content = _render(
            asset_returns, models=["gbm", "bootstrap"], sims=100, horizon=120, seed=1
        )
        assert "Strategy Assessment" in content
        assert "Monte Carlo Models" in content
        assert "Stress Tests" in content
        assert 'id="overlay_fan"' in content
        assert 'id="maxdd_dist"' in content
        assert 'id="cagr_quantiles"' in content
        assert "Verdict:" in content
        assert "GBM" in content
        assert "Bootstrap" in content

    def test_bust_goal_columns_present(self, asset_returns):
        content = _render(
            asset_returns,
            models=["gbm"],
            sims=100,
            horizon=120,
            bust=-0.25,
            goal=0.5,
            seed=1,
        )
        assert "P(Bust)" in content
        assert "P(Goal)" in content

    def test_stress_models_in_stress_section(self, asset_returns):
        content = _render(
            asset_returns,
            models=["gbm", "trimmed_1pct"],
            sims=100,
            horizon=120,
            seed=1,
        )
        assert "Trimmed (top 1%)" in content
        assert "deliberately pessimistic" in content

    def test_runs_all_models_by_default(self, asset_returns):
        content = _render(asset_returns, sims=50, horizon=80, seed=2)
        for label in ("GBM", "GARCH", "Heston", "Jump Diffusion", "Bayesian"):
            assert label in content

    def test_accepts_dataframe_first_column(self, asset_returns):
        df = asset_returns.to_frame()
        content = _render(df, models=["gbm"], sims=50, horizon=80, seed=2)
        assert "<!DOCTYPE html>" in content

    def test_assumptions_footer(self, asset_returns):
        content = _render(
            asset_returns, models=["gbm"], sims=100, horizon=120, seed=42
        )
        assert "Assumptions:" in content
        assert "seed 42" in content

    def test_drift_stress_note_in_header(self, asset_returns):
        content = _render(
            asset_returns,
            models=["gbm"],
            sims=50,
            horizon=80,
            seed=1,
            drift="zero",
        )
        assert "Drift: zero (stress scenario)" in content

    def test_per_model_detail_collapsed(self, asset_returns):
        content = _render(
            asset_returns, models=["gbm", "bootstrap"], sims=50, horizon=80, seed=1
        )
        assert "<details>" in content
        assert 'id="fan_charts"' in content
        assert 'id="terminal_hists"' in content
