"""
Tests for reports.html_simple — the simplified tearsheet.

Behaviour is verified through the rendered HTML: presence of kept charts/metrics
and absence of the ones intentionally removed.
"""

import os
import re
import tempfile

import numpy as np
import pandas as pd
import pytest

from quantstats import reports


@pytest.fixture
def sample_returns():
    np.random.seed(42)
    dates = pd.date_range("2019-01-01", periods=750, freq="D")
    return pd.Series(np.random.randn(750) * 0.02, index=dates, name="Strategy")


@pytest.fixture
def sample_benchmark():
    np.random.seed(7)
    dates = pd.date_range("2019-01-01", periods=750, freq="D")
    return pd.Series(np.random.randn(750) * 0.015, index=dates, name="Benchmark")


def _render(returns, **kwargs):
    with tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False) as f:
        path = f.name
    try:
        reports.html_simple(returns, output=path, **kwargs)
        with open(path) as fh:
            return fh.read()
    finally:
        if os.path.exists(path):
            os.remove(path)


class TestSimpleTearsheet:
    def test_generates_valid_html(self, sample_returns):
        content = _render(sample_returns, title="Simple")
        assert "<!DOCTYPE html>" in content
        assert "Simple" in content
        # No leftover template placeholders (CSS braces in @media are fine)
        assert not re.search(r"\{\{\s*\w+\s*\}\}", content)

    def test_drops_unwanted_charts(self, sample_returns):
        content = _render(sample_returns)
        for div_id in (
            'id="rolling_beta"',
            'id="rolling_sharpe"',
            'id="rolling_sortino"',
            'id="monthly_heatmap"',
            'id="returns_dist"',
            'id="daily_returns"',
        ):
            assert div_id not in content

    def test_keeps_core_charts(self, sample_returns):
        content = _render(sample_returns)
        for div_id in (
            'id="log_returns"',
            'id="eoy_returns"',
            'id="daily_dist"',
            'id="rolling_vol"',
            'id="dd_plot"',
        ):
            assert div_id in content

    def test_drops_unwanted_metrics(self, sample_returns):
        content = _render(sample_returns)
        for label in (
            "Max DD Date",
            "Max DD Period Start",
            "Max DD Period End",
            "Risk-Adjusted Return",
            "Risk-Return Ratio",
            "Avg. Return",
            "Avg. Win",
            "Avg. Loss",
            "Win/Loss Ratio",
            "Profit Ratio",
        ):
            assert label not in content

    def test_keeps_core_metrics(self, sample_returns):
        content = _render(sample_returns)
        assert "Sharpe" in content
        assert "Max Drawdown" in content

    def test_worst_drawdowns_is_top5(self, sample_returns):
        content = _render(sample_returns)
        assert "Worst 5 Drawdowns" in content

    def test_works_with_benchmark(self, sample_returns, sample_benchmark):
        content = _render(sample_returns, benchmark=sample_benchmark)
        assert 'id="vol_returns"' in content
        assert "EOY Returns vs Benchmark" in content

    def test_daily_distribution_present(self, sample_returns):
        content = _render(sample_returns)
        assert "Distribution of Daily Returns" in content

    def test_no_consecutive_separator_rows(self, sample_returns):
        """Group separators are kept but never stacked (no large gaps)."""
        content = _render(sample_returns)
        i = content.find("Key Performance Metrics")
        table = content[i : content.find("</table>", i)]
        rows = re.findall(r"<tr>.*?</tr>", table, re.S)
        # Separator rows have only empty cells (now styled as a gray rule).
        blanks = [
            bool(re.fullmatch(r"<tr>(?:<td[^>]*></td>)+</tr>", r, re.S)) for r in rows
        ]
        # at least one separator remains (grouping preserved)
        assert any(blanks)
        # but never two blanks in a row, and none at the edges
        assert not any(a and b for a, b in zip(blanks, blanks[1:]))
        assert not blanks[0] and not blanks[-1]
        # separators are rendered as a subtle gray rule, not empty whitespace
        assert "border-bottom:1px solid" in table
