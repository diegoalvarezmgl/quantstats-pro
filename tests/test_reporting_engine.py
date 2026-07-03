"""
Tests for the quantstats._reporting engine and sections.

These tests exercise the public behaviour of the section-based reporting
engine used by the simple and montecarlo tearsheets, independently of the
legacy monolithic ``reports.html()`` path.
"""

import numpy as np
import pandas as pd
import pytest

from quantstats._reporting import engine, sections


@pytest.fixture
def sample_returns():
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=500, freq="D")
    return pd.Series(np.random.randn(500) * 0.02, index=dates, name="Strategy")


class TestRenderReport:
    def test_fills_known_placeholder(self):
        tpl = "<h1>{{title}}</h1>"
        out = engine.render_report(tpl, {"title": "Hello"})
        assert out == "<h1>Hello</h1>"

    def test_removes_unknown_placeholders(self):
        tpl = "<div>{{known}}</div><div>{{unknown}}</div>"
        out = engine.render_report(tpl, {"known": "X"})
        assert "X" in out
        assert "{{" not in out and "}}" not in out

    def test_leaves_no_placeholder_when_value_empty(self):
        tpl = "<div>{{a}}</div>"
        out = engine.render_report(tpl, {"a": ""})
        assert out == "<div></div>"

    def test_repeated_placeholder_filled_everywhere(self):
        tpl = "{{v}}-{{v}}"
        out = engine.render_report(tpl, {"v": "x"})
        assert out == "x-x"


class TestSections:
    def test_metrics_table_drops_requested_rows(self, sample_returns):
        html = sections.metrics_table(
            sample_returns,
            drop_rows=["Avg. Win %", "Profit Ratio"],
        )
        assert "<table" in html
        assert "Avg. Win" not in html
        assert "Profit Ratio" not in html
        # a metric that should remain
        assert "Sharpe" in html

    def test_dd_table_limits_rows(self, sample_returns):
        html_top5 = sections.dd_table(sample_returns, top=5)
        html_top10 = sections.dd_table(sample_returns, top=10)
        # both render a table; top10 should have at least as many rows
        assert html_top5.count("<tr>") <= html_top10.count("<tr>")
        assert "<table" in html_top5

    def test_plot_section_returns_embedded_svg(self, sample_returns):
        html = sections.plot_section(
            "returns",
            sample_returns,
            benchmark=None,
            figfmt="svg",
        )
        assert "<svg" in html or "svg" in html
