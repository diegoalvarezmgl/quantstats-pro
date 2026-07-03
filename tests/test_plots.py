"""
Tests for quantstats.plots module
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os

import quantstats as qs
from quantstats import plots


@pytest.fixture
def sample_returns():
    """Generate sample daily returns for testing."""
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=500, freq="D")
    returns = pd.Series(np.random.randn(500) * 0.02, index=dates, name="Strategy")
    return returns


@pytest.fixture
def sample_benchmark():
    """Generate sample benchmark returns for testing."""
    np.random.seed(123)
    dates = pd.date_range("2020-01-01", periods=500, freq="D")
    returns = pd.Series(np.random.randn(500) * 0.015, index=dates, name="Benchmark")
    return returns


class TestHistogramXlim:
    """The daily histogram can clip its x-axis to inner quantiles."""

    @pytest.fixture
    def fat_tailed(self):
        np.random.seed(0)
        dates = pd.date_range("2018-01-01", periods=1000, freq="D")
        vals = np.random.randn(1000) * 0.01
        vals[0] = 0.25  # extreme outlier that would stretch the x-axis
        vals[1] = -0.30
        return pd.Series(vals, index=dates, name="Asset")

    def test_quantile_clip_tightens_xaxis(self, fat_tailed):
        wide = plots.histogram(fat_tailed, resample="D", show=False)
        narrow = plots.histogram(
            fat_tailed, resample="D", xlim_quantile=0.01, show=False
        )
        wide_span = np.subtract(*reversed(wide.axes[0].get_xlim()))
        narrow_span = np.subtract(*reversed(narrow.axes[0].get_xlim()))
        assert narrow_span < wide_span

    def test_adaptive_bins_spreads_across_visible_range(self, fat_tailed):
        fig = plots.histogram(
            fat_tailed,
            resample="D",
            xlim_quantile=0.01,
            adaptive_bins=60,
            show=False,
        )
        ax = fig.axes[0]
        xlim = ax.get_xlim()
        visible = []
        for cont in ax.containers:
            for patch in cont.patches:
                cx = patch.get_x() + patch.get_width() / 2
                if xlim[0] <= cx <= xlim[1] and patch.get_height() > 0:
                    visible.append(patch.get_width())
        assert len(visible) >= 40
        widths = {round(w, 8) for w in visible}
        assert len(widths) == 1
        expected_w = (xlim[1] - xlim[0]) / 60
        assert visible[0] == pytest.approx(expected_w, rel=0.05)

    def test_daily_resample_does_not_fabricate_weekend_zeros(self):
        """Calendar resample must not pile comp([])==0 into the centre bin."""
        np.random.seed(1)
        dates = pd.bdate_range("2018-01-01", periods=500)
        returns = pd.Series(np.random.randn(500) * 0.01, index=dates, name="Asset")
        fig = plots.histogram(
            returns,
            resample="D",
            xlim_quantile=0.01,
            adaptive_bins=40,
            show=False,
            prepare_returns=False,
        )
        heights = sorted(
            (p.get_height() for c in fig.axes[0].containers for p in c.patches),
            reverse=True,
        )
        assert heights[0] < heights[1] * 2.5


class TestPlotFunctions:
    """Test that plot functions run without errors."""

    def test_snapshot(self, sample_returns):
        """Test snapshot plot generates without error."""
        # snapshot should work with show=False
        fig = plots.snapshot(sample_returns, show=False)
        assert fig is not None

    def test_returns_plot(self, sample_returns, sample_benchmark):
        """Test returns plot."""
        fig = plots.returns(sample_returns, sample_benchmark, show=False)
        assert fig is not None

    def test_log_returns_plot(self, sample_returns):
        """Test log returns plot."""
        fig = plots.log_returns(sample_returns, show=False)
        assert fig is not None

    def test_yearly_returns(self, sample_returns):
        """Test yearly returns bar chart."""
        fig = plots.yearly_returns(sample_returns, show=False)
        assert fig is not None

    def test_yearly_returns_xaxis_aligned_with_many_years(self, sample_benchmark):
        """EOY bars must stay aligned with year labels (11+ years)."""
        np.random.seed(42)
        dates = pd.date_range("2016-01-01", periods=11 * 252, freq="B")
        returns = pd.DataFrame(
            {
                "QQQ": np.random.randn(len(dates)) * 0.02,
                "FEZ": np.random.randn(len(dates)) * 0.015,
            },
            index=dates,
        )
        fig = plots.yearly_returns(
            returns, sample_benchmark, show=False, prepare_returns=False
        )
        ax = fig.axes[0]
        ticks = ax.get_xticks()
        bar_centers = [
            patch.get_x() + patch.get_width() / 2
            for patch in ax.containers[0].patches
        ]
        assert len(ticks) == len(bar_centers)
        for tick, center in zip(ticks, bar_centers, strict=True):
            assert abs(tick - center) < 0.5

    def test_histogram(self, sample_returns, sample_benchmark):
        """Test histogram plot."""
        fig = plots.histogram(sample_returns, sample_benchmark, show=False)
        assert fig is not None

    def test_daily_returns(self, sample_returns, sample_benchmark):
        """Test daily returns scatter plot."""
        fig = plots.daily_returns(sample_returns, sample_benchmark, show=False)
        assert fig is not None

    def test_drawdown(self, sample_returns):
        """Test drawdown plot."""
        fig = plots.drawdown(sample_returns, show=False)
        assert fig is not None

    def test_drawdowns_periods(self, sample_returns):
        """Test drawdown periods plot."""
        fig = plots.drawdowns_periods(sample_returns, show=False)
        assert fig is not None

    def test_rolling_volatility(self, sample_returns):
        """Test rolling volatility plot."""
        fig = plots.rolling_volatility(sample_returns, show=False)
        assert fig is not None

    def test_rolling_sharpe(self, sample_returns):
        """Test rolling Sharpe plot."""
        fig = plots.rolling_sharpe(sample_returns, show=False)
        assert fig is not None

    def test_rolling_sortino(self, sample_returns):
        """Test rolling Sortino plot."""
        fig = plots.rolling_sortino(sample_returns, show=False)
        assert fig is not None

    def test_rolling_beta(self, sample_returns, sample_benchmark):
        """Test rolling beta plot."""
        fig = plots.rolling_beta(sample_returns, sample_benchmark, show=False)
        assert fig is not None

    def test_monthly_heatmap(self, sample_returns):
        """Test monthly heatmap."""
        fig = plots.monthly_heatmap(sample_returns, show=False)
        assert fig is not None

    def test_distribution(self, sample_returns):
        """Test distribution box plot."""
        fig = plots.distribution(sample_returns, show=False)
        assert fig is not None


class TestPlotSaving:
    """Test plot saving functionality."""

    def test_save_to_file(self, sample_returns):
        """Test saving plot to file."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            output_path = f.name

        try:
            plots.returns(sample_returns, savefig=output_path, show=False)
            assert os.path.exists(output_path)
            assert os.path.getsize(output_path) > 0
        finally:
            if os.path.exists(output_path):
                os.remove(output_path)

    def test_save_with_dict_params(self, sample_returns):
        """Test saving plot with dict parameters."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            output_path = f.name

        try:
            plots.returns(
                sample_returns,
                savefig={"fname": output_path, "dpi": 100},
                show=False,
            )
            assert os.path.exists(output_path)
        finally:
            if os.path.exists(output_path):
                os.remove(output_path)


class TestPlotOptions:
    """Test plot configuration options."""

    def test_grayscale_mode(self, sample_returns):
        """Test grayscale plotting mode."""
        fig = plots.returns(sample_returns, grayscale=True, show=False)
        assert fig is not None

    def test_custom_figsize(self, sample_returns):
        """Test custom figure size."""
        fig = plots.returns(sample_returns, figsize=(12, 8), show=False)
        assert fig is not None

    def test_no_subtitle(self, sample_returns):
        """Test plot without subtitle."""
        fig = plots.returns(sample_returns, subtitle=False, show=False)
        assert fig is not None
