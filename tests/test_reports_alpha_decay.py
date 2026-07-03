"""
Tests for reports.html_alpha_decay — the alpha decay tearsheet.
"""

import os
import re
import tempfile

import numpy as np
import pandas as pd
import pytest

from quantstats import reports


def _make_returns():
    rng = np.random.default_rng(3)
    n = 400
    log_r = rng.normal(0.0003, 0.012, n)
    dates = pd.date_range("2018-01-01", periods=n, freq="B")
    return pd.Series(np.expm1(log_r), index=dates, name="QQQ")


@pytest.fixture(scope="module")
def asset_returns():
    return _make_returns()


def _render(returns, **kwargs):
    with tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False) as f:
        path = f.name
    try:
        reports.html_alpha_decay(returns, output=path, **kwargs)
        with open(path) as fh:
            return fh.read()
    finally:
        if os.path.exists(path):
            os.remove(path)


@pytest.fixture(scope="module")
def default_render(asset_returns):
    """Default-parameter tearsheet rendered once and shared across tests."""
    return _render(asset_returns)


class TestAlphaDecayTearsheet:
    def test_generates_valid_html(self, default_render):
        content = default_render
        assert "<!DOCTYPE html>" in content
        assert "Alpha Decay Tearsheet" in content
        assert not re.search(r"\{\{\s*\w+\s*\}\}", content)

    def test_contains_summary_and_sections(self, default_render):
        content = default_render
        assert "Health Summary" in content
        assert "Metric Distributions" in content
        assert "Structural Decay Diagnostics" in content
        assert 'id="summary_table"' in content
        assert 'id="metric_rows"' in content
        assert 'id="cusum_chart"' in content
        assert 'id="time_underwater"' in content

    def test_score_and_badges_present(self, default_render):
        content = default_render
        assert re.search(r"\d+/\d+", content)
        assert 'class="badge' in content
        assert "Excellent" in content or "Good" in content

    def test_metric_rows_for_all_metrics(self, asset_returns):
        content = _render(asset_returns, windows=(7, 15))
        for key in ("cagr", "volatility", "max_drawdown", "win_rate"):
            assert f'id="metric_{key}"' in content
        for key in ("sharpe", "sortino"):
            assert f'id="metric_{key}"' not in content

    def test_custom_windows(self, asset_returns):
        content = _render(asset_returns, windows=(7, 30), title="Custom Decay")
        assert "Custom Decay" in content
        assert "7/30d" in content

    def test_dataframe_input(self, asset_returns):
        df = asset_returns.to_frame("QQQ")
        content = _render(df)
        assert "QQQ" in content

    def test_cusum_section(self, default_render):
        # cusum_k=0.5 / cusum_h=4.0 are the defaults used by default_render
        assert "CUSUM" in default_render
