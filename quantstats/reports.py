#!/usr/bin/env python
#
# QuantStats: Portfolio analytics for quants
# https://github.com/ranaroussi/quantstats
#
# Copyright 2019-2025 Ran Aroussi
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re as _regex
from base64 import b64encode as _b64encode
from datetime import datetime as _dt
from math import ceil as _ceil
from math import sqrt as _sqrt

import numpy as _np
import pandas as _pd
from tabulate import tabulate as _tabulate

from . import __version__

# Lazy imports to avoid circular dependency during package initialization
_stats = None
_utils = None
_plots = None


def _get_stats():
    global _stats
    if _stats is None:
        from . import stats
        _stats = stats
    return _stats


def _get_utils():
    global _utils
    if _utils is None:
        from . import utils
        _utils = utils
    return _utils


def _get_plots():
    global _plots
    if _plots is None:
        from . import plots
        _plots = plots
    return _plots
import tempfile
import webbrowser
from io import StringIO
from pathlib import Path

from dateutil.relativedelta import relativedelta

try:
    from IPython.display import HTML as iHTML
    from IPython.display import display as iDisplay
except ImportError:
    pass  # IPython not available, display functions won't be used


def _get_trading_periods(periods_per_year=252):
    """
    Calculate trading periods for different time windows.

    This helper function computes the number of trading periods for full year
    and half year periods, which are commonly used in financial calculations
    for annualization and rolling window analysis.

    Parameters
    ----------
    periods_per_year : int, default 252
        Number of trading periods in a year (e.g., 252 for daily data,
        12 for monthly data)

    Returns
    -------
    tuple
        A tuple containing (periods_per_year, half_year_periods)

    Examples
    --------
    >>> _get_trading_periods(252)  # Daily data
    (252, 126)
    >>> _get_trading_periods(12)   # Monthly data
    (12, 6)
    """
    # Calculate half year periods using ceiling to ensure we get at least half
    half_year = _ceil(periods_per_year / 2)
    return periods_per_year, half_year


def _print_parameters_table(
    benchmark_title=None,
    periods_per_year=252,
    rf=0.0,
    compounded=True,
    match_dates=True,
):
    """
    Print a formatted parameters table for terminal/console output.

    Parameters
    ----------
    benchmark_title : str or None
        Benchmark name/ticker
    periods_per_year : int
        Number of trading periods per year
    rf : float
        Risk-free rate
    compounded : bool
        Whether returns are compounded
    match_dates : bool
        Whether dates are matched with benchmark
    """
    width = 40
    print("=" * width)
    print("                 Parameters")
    print("-" * width)
    if benchmark_title:
        print(f"{'Benchmark':<25}{benchmark_title.upper():>15}")
    print(f"{'Periods/Year':<25}{periods_per_year:>15}")
    print(f"{'Risk-Free Rate':<25}{rf:>14.1%}")
    print(f"{'Compounded':<25}{'Yes' if compounded else 'No':>15}")
    if benchmark_title:
        print(f"{'Match Dates':<25}{'Yes' if match_dates else 'No':>15}")
    print("=" * width)
    print()


def _match_dates(returns, benchmark):
    """
    Align returns and benchmark data to start from the same date.

    This function ensures that both the returns and benchmark series start
    from the same date by finding the latest start date where both series
    have non-zero values. This is crucial for accurate performance comparisons.

    Parameters
    ----------
    returns : pd.Series or pd.DataFrame
        Returns data that may be a Series or DataFrame with multiple columns
    benchmark : pd.Series
        Benchmark returns data

    Returns
    -------
    tuple
        A tuple containing (aligned_returns, aligned_benchmark) both starting
        from the same date

    Examples
    --------
    >>> returns_aligned, bench_aligned = _match_dates(returns, benchmark)
    """
    # Handle different types of returns data (Series vs DataFrame)
    if isinstance(returns, _pd.DataFrame):
        # For DataFrame, use the first column to find the start date
        loc = max(returns[returns.columns[0]].ne(0).idxmax(), benchmark.ne(0).idxmax())
    else:
        # For Series, find the maximum of start dates for both series
        loc = max(returns.ne(0).idxmax(), benchmark.ne(0).idxmax())

    # Slice both series to start from the latest common start date
    returns = returns.loc[loc:]
    benchmark = benchmark.loc[loc:]

    return returns, benchmark


def html(
    returns,
    benchmark=None,
    rf=0.0,
    grayscale=False,
    title="Strategy Tearsheet",
    output=None,
    compounded=True,
    periods_per_year=252,
    download_filename="quantstats-tearsheet.html",
    figfmt="svg",
    template_path=None,
    match_dates=True,
    **kwargs,
):
    """
    Generate an HTML tearsheet report for portfolio performance analysis.

    This function creates a comprehensive HTML report containing performance
    metrics, visualizations, and analysis of investment returns. The report
    includes comparisons with benchmarks, drawdown analysis, and various
    performance charts.

    Parameters
    ----------
    returns : pd.Series or pd.DataFrame
        Daily returns data for the strategy/portfolio
    benchmark : pd.Series, str, or None, default None
        Benchmark returns for comparison. Can be a Series of returns,
        a ticker symbol string, or None for no benchmark
    rf : float, default 0.0
        Risk-free rate for calculations (as decimal, e.g., 0.02 for 2%)
    grayscale : bool, default False
        Whether to generate charts in grayscale instead of color
    title : str, default "Strategy Tearsheet"
        Title to display at the top of the HTML report
    output : str or None, default None
        File path to save the HTML report. If None, downloads in browser
    compounded : bool, default True
        Whether to compound returns for calculations
    periods_per_year : int, default 252
        Number of trading periods per year for annualization
    download_filename : str, default "quantstats-tearsheet.html"
        Filename for browser download if output is None
    figfmt : str, default "svg"
        Format for embedded charts ('svg', 'png', 'jpg')
    template_path : str or None, default None
        Path to custom HTML template file. Uses default if None
    match_dates : bool, default True
        Whether to align returns and benchmark start dates
    **kwargs
        Additional keyword arguments for customization:
        - strategy_title: Custom name for the strategy
        - benchmark_title: Custom name for the benchmark
        - active_returns: Whether to show active returns vs benchmark

    Returns
    -------
    None
        Generates HTML file either as download or saved to specified path

    Examples
    --------
    >>> html(returns, benchmark='^GSPC', title='My Strategy')
    >>> html(returns, output='report.html', grayscale=True)

    Raises
    ------
    FileNotFoundError
        If custom template_path doesn't exist
    """
    # Clean returns data by removing NaN values if date matching is enabled
    if match_dates:
        returns = returns.dropna()

    # Get trading periods for calculations
    win_year, win_half_year = _get_trading_periods(periods_per_year)

    # Secure file path handling for HTML template
    if template_path is None:
        # Use default template path - report.html in same directory
        template_path = Path(__file__).parent / 'report.html'
    else:
        template_path = Path(template_path)

    # Resolve to absolute path and validate template file existence
    template_path = template_path.resolve()

    if not template_path.exists():
        raise FileNotFoundError(f"Template file not found: {template_path}")
    if not template_path.is_file():
        raise ValueError(f"Template path is not a file: {template_path}")

    # Read template securely with UTF-8 encoding
    tpl = template_path.read_text(encoding='utf-8')

    # prepare timeseries
    if match_dates:
        returns = returns.dropna()
    # Clean and prepare returns data for analysis
    returns = _get_utils()._prepare_returns(returns)

    # Handle strategy title - can be single string or list for multiple columns
    strategy_title = kwargs.get("strategy_title", "Strategy")
    if (
        isinstance(returns, _pd.DataFrame)
        and len(returns.columns) > 1
        and isinstance(strategy_title, str)
    ):
        strategy_title = list(returns.columns)

    # Process benchmark data if provided
    if benchmark is not None:
        benchmark_title = kwargs.get("benchmark_title", "Benchmark")
        # Auto-determine benchmark title if not provided
        if kwargs.get("benchmark_title") is None:
            if isinstance(benchmark, str):
                benchmark_title = benchmark
            elif isinstance(benchmark, _pd.Series):
                benchmark_title = benchmark.name if benchmark.name else "Benchmark"
            elif isinstance(benchmark, _pd.DataFrame):
                col_name = benchmark[benchmark.columns[0]].name
                benchmark_title = col_name if col_name else "Benchmark"

        # Ensure benchmark_title is a string for .upper() call
        if benchmark_title is None:
            benchmark_title = "Benchmark"
        # Store original benchmark before any alignment for accurate EOY calculations
        # This preserves the full benchmark data including non-trading days
        if isinstance(benchmark, str):
            # Download the full benchmark data
            benchmark_original = _get_utils().download_returns(benchmark)
            if rf != 0:
                benchmark_original = _get_utils().to_excess_returns(
                    benchmark_original, rf, nperiods=periods_per_year
                )
        elif isinstance(benchmark, _pd.Series):
            benchmark_original = benchmark.copy()
        else:
            benchmark_original = benchmark
        # Prepare benchmark data to match returns index and risk-free rate
        benchmark = _get_utils()._prepare_benchmark(benchmark, returns.index, rf)
        # Align dates between returns and benchmark if requested
        if match_dates is True:
            returns, benchmark = _match_dates(returns, benchmark)
    else:
        benchmark_title = None
        benchmark_original = None

    # Format date range for display in template
    date_range = returns.index.strftime("%e %b, %Y")
    tpl = tpl.replace("{{date_range}}", date_range[0] + " - " + date_range[-1])

    # Build title with compounding indicator (only show if compounded)
    full_title = f"{title} (Compounded)" if compounded else title
    tpl = tpl.replace("{{title}}", full_title)
    tpl = tpl.replace("{{v}}", __version__)
    tpl = tpl.replace(
        "{{generated_at}}",
        _dt.now().strftime("%d %b, %Y %H:%M").lstrip("0"),
    )

    # Build parameters string for subtitle
    params_parts = []

    # Add user-provided parameters first if present
    user_params = kwargs.get("parameters", {})
    if user_params:
        for key, value in user_params.items():
            params_parts.append(f"{key}: {value}")

    # Add auto-detected parameters (always show key params)
    if benchmark_title:
        params_parts.append(f"Benchmark: {benchmark_title.upper()}")
    params_parts.append(f"Periods/Year: {periods_per_year}")
    params_parts.append(f"RF: {rf:.1%}")

    params_str = " &bull; ".join(params_parts)
    if params_str:
        params_str += " | "
    tpl = tpl.replace("{{params}}", params_str)

    # Add matched dates indicator
    matched_dates_str = " (matched dates)" if match_dates and benchmark is not None else ""
    tpl = tpl.replace("{{matched_dates}}", matched_dates_str)

    # Set names for data series to be used in charts and tables
    if benchmark is not None:
        benchmark.name = benchmark_title
    if isinstance(returns, _pd.Series):
        returns.name = strategy_title
    elif isinstance(returns, _pd.DataFrame):
        returns.columns = (
            strategy_title if isinstance(strategy_title, list) else [strategy_title]
        )

    # Generate comprehensive performance metrics table
    mtrx = metrics(
        returns=returns,
        benchmark=benchmark,
        rf=rf,
        display=False,
        mode="full",
        sep=True,
        internal="True",
        compounded=compounded,
        periods_per_year=periods_per_year,
        prepare_returns=False,
        benchmark_title=benchmark_title,
        strategy_title=strategy_title,
    )[2:]

    # Format metrics table for HTML display
    mtrx.index.name = "Metric"
    tpl = tpl.replace("{{metrics}}", _html_table(mtrx))

    # Handle table formatting for multiple columns
    if isinstance(returns, _pd.DataFrame):
        num_cols = len(returns.columns)
        # Replace empty table rows with horizontal rule separators
        for i in reversed(range(num_cols + 1, num_cols + 3)):
            str_td = "<td></td>" * i
            tpl = tpl.replace(
                f"<tr>{str_td}</tr>", f'<tr><td colspan="{i}"><hr></td></tr>'
            )

    # Clean up table formatting with horizontal rules
    tpl = tpl.replace(
        "<tr><td></td><td></td><td></td></tr>", '<tr><td colspan="3"><hr></td></tr>'
    )
    tpl = tpl.replace(
        "<tr><td></td><td></td></tr>", '<tr><td colspan="2"><hr></td></tr>'
    )

    # Generate end-of-year (EOY) returns comparison table
    if benchmark is not None:
        # Use original benchmark for EOY comparison to preserve accurate yearly returns
        # This prevents loss of benchmark returns on non-trading days
        benchmark_for_eoy = benchmark_original if benchmark_original is not None else benchmark
        yoy = _get_stats().compare(
            returns, benchmark_for_eoy, "YE", compounded=compounded, prepare_returns=False
        )
        # Set appropriate column names based on data type
        if isinstance(returns, _pd.Series):
            yoy.columns = [benchmark_title, strategy_title, "Multiplier", "Won"]
        elif isinstance(returns, _pd.DataFrame):
            yoy.columns = list(
                _pd.core.common.flatten([benchmark_title, strategy_title])
            )
        yoy.index.name = "Year"
        tpl = tpl.replace("{{eoy_title}}", "<h3>EOY Returns vs Benchmark</h3>")
        tpl = tpl.replace("{{eoy_table}}", _html_table(yoy))
    else:
        # Generate EOY returns table without benchmark comparison
        # pct multiplier
        yoy = _pd.DataFrame(_get_utils().group_returns(returns, returns.index.year) * 100)
        if isinstance(returns, _pd.Series):
            yoy.columns = ["Return"]
            yoy["Cumulative"] = _get_utils().group_returns(returns, returns.index.year, True) * 100
            # Don't add "%" here - the CSS in report.html handles it via :after pseudo-element
            # Adding "%" in Python causes double "%" display (bug #475)
        elif isinstance(returns, _pd.DataFrame):
            # Don't show cumulative for multiple strategy portfolios
            # just show compounded like when we have a benchmark
            yoy.columns = list(_pd.core.common.flatten(strategy_title))

        yoy.index.name = "Year"
        tpl = tpl.replace("{{eoy_title}}", "<h3>EOY Returns</h3>")
        tpl = tpl.replace("{{eoy_table}}", _html_table(yoy))

    # Generate drawdown analysis table
    if isinstance(returns, _pd.Series):
        # Calculate drawdown series and get worst drawdown periods
        dd = _get_stats().to_drawdown_series(returns)
        dd_info = _get_stats().drawdown_details(dd).sort_values(
            by="max drawdown", ascending=True
        )[:10]
        dd_info = dd_info[["start", "end", "max drawdown", "days"]]
        dd_info.columns = ["Started", "Recovered", "Drawdown", "Days"]
        tpl = tpl.replace("{{dd_info}}", _html_table(dd_info, False))
    elif isinstance(returns, _pd.DataFrame):
        # Handle multiple strategy columns
        dd_info_list = []
        for col in returns.columns:
            dd = _get_stats().to_drawdown_series(returns[col])
            dd_info = _get_stats().drawdown_details(dd).sort_values(
                by="max drawdown", ascending=True
            )[:10]
            dd_info = dd_info[["start", "end", "max drawdown", "days"]]
            dd_info.columns = ["Started", "Recovered", "Drawdown", "Days"]
            dd_info_list.append(_html_table(dd_info, False))

        # Combine all drawdown tables with headers
        dd_html_table = ""
        for html_str, col in zip(dd_info_list, returns.columns, strict=False):
            dd_html_table = (
                dd_html_table + f"<h3>{col}</h3><br>" + StringIO(html_str).read()
            )
        tpl = tpl.replace("{{dd_info}}", dd_html_table)

    # Get active returns setting for plots
    active = kwargs.get("active_returns", False)

    # Generate all the performance plots and embed them in the HTML
    # plots
    figfile = _get_utils()._file_stream()
    _get_plots().returns(
        returns,
        benchmark,
        grayscale=grayscale,
        figsize=(8, 5),
        subtitle=False,
        savefig={"fname": figfile, "format": figfmt},
        show=False,
        ylabel="",
        compound=compounded,
        prepare_returns=False,
    )
    tpl = tpl.replace("{{returns}}", _embed_figure(figfile, figfmt))

    # Log returns plot for better visualization of performance
    figfile = _get_utils()._file_stream()
    _get_plots().log_returns(
        returns,
        benchmark,
        grayscale=grayscale,
        figsize=(8, 4),
        subtitle=False,
        savefig={"fname": figfile, "format": figfmt},
        show=False,
        ylabel="",
        compound=compounded,
        prepare_returns=False,
    )
    tpl = tpl.replace("{{log_returns}}", _embed_figure(figfile, figfmt))

    # Volatility-matched returns plot (only if benchmark exists)
    if benchmark is not None:
        figfile = _get_utils()._file_stream()
        _get_plots().returns(
            returns,
            benchmark,
            match_volatility=True,
            grayscale=grayscale,
            figsize=(8, 4),
            subtitle=False,
            savefig={"fname": figfile, "format": figfmt},
            show=False,
            ylabel="",
            compound=compounded,
            prepare_returns=False,
        )
        tpl = tpl.replace("{{vol_returns}}", _embed_figure(figfile, figfmt))

    # Yearly returns comparison chart
    figfile = _get_utils()._file_stream()
    _get_plots().yearly_returns(
        returns,
        benchmark,
        grayscale=grayscale,
        figsize=(8, 4),
        subtitle=False,
        savefig={"fname": figfile, "format": figfmt},
        show=False,
        ylabel="",
        compounded=compounded,
        prepare_returns=False,
    )
    tpl = tpl.replace("{{eoy_returns}}", _embed_figure(figfile, figfmt))

    # Returns distribution histogram
    figfile = _get_utils()._file_stream()
    _get_plots().histogram(
        returns,
        benchmark,
        grayscale=grayscale,
        figsize=(7, 4),
        subtitle=False,
        savefig={"fname": figfile, "format": figfmt},
        show=False,
        ylabel="",
        compounded=compounded,
        prepare_returns=False,
    )
    tpl = tpl.replace("{{monthly_dist}}", _embed_figure(figfile, figfmt))

    # Daily returns scatter plot
    figfile = _get_utils()._file_stream()
    _get_plots().daily_returns(
        returns,
        benchmark,
        grayscale=grayscale,
        figsize=(8, 3),
        subtitle=False,
        savefig={"fname": figfile, "format": figfmt},
        show=False,
        ylabel="",
        prepare_returns=False,
        active=active,
    )
    tpl = tpl.replace("{{daily_returns}}", _embed_figure(figfile, figfmt))

    # Rolling beta analysis (only if benchmark exists)
    if benchmark is not None:
        figfile = _get_utils()._file_stream()
        _get_plots().rolling_beta(
            returns,
            benchmark,
            grayscale=grayscale,
            figsize=(8, 3),
            subtitle=False,
            window1=win_half_year,
            window2=win_year,
            savefig={"fname": figfile, "format": figfmt},
            show=False,
            ylabel="",
            prepare_returns=False,
        )
        tpl = tpl.replace("{{rolling_beta}}", _embed_figure(figfile, figfmt))

    # Rolling volatility analysis
    figfile = _get_utils()._file_stream()
    _get_plots().rolling_volatility(
        returns,
        benchmark,
        grayscale=grayscale,
        figsize=(8, 3),
        subtitle=False,
        savefig={"fname": figfile, "format": figfmt},
        show=False,
        ylabel="",
        period=win_half_year,
        periods_per_year=win_year,
    )
    tpl = tpl.replace("{{rolling_vol}}", _embed_figure(figfile, figfmt))

    # Rolling Sharpe ratio analysis
    figfile = _get_utils()._file_stream()
    _get_plots().rolling_sharpe(
        returns,
        grayscale=grayscale,
        figsize=(8, 3),
        subtitle=False,
        savefig={"fname": figfile, "format": figfmt},
        show=False,
        ylabel="",
        period=win_half_year,
        periods_per_year=win_year,
    )
    tpl = tpl.replace("{{rolling_sharpe}}", _embed_figure(figfile, figfmt))

    # Rolling Sortino ratio analysis
    figfile = _get_utils()._file_stream()
    _get_plots().rolling_sortino(
        returns,
        grayscale=grayscale,
        figsize=(8, 3),
        subtitle=False,
        savefig={"fname": figfile, "format": figfmt},
        show=False,
        ylabel="",
        period=win_half_year,
        periods_per_year=win_year,
    )
    tpl = tpl.replace("{{rolling_sortino}}", _embed_figure(figfile, figfmt))

    # Drawdown periods analysis
    figfile = _get_utils()._file_stream()
    if isinstance(returns, _pd.Series):
        _get_plots().drawdowns_periods(
            returns,
            grayscale=grayscale,
            figsize=(8, 4),
            subtitle=False,
            title=returns.name,
            savefig={"fname": figfile, "format": figfmt},
            show=False,
            ylabel="",
            compounded=compounded,
            prepare_returns=False,
        )
        tpl = tpl.replace("{{dd_periods}}", _embed_figure(figfile, figfmt))
    elif isinstance(returns, _pd.DataFrame):
        # Handle multiple strategy columns
        embed = []
        for col in returns.columns:
            _get_plots().drawdowns_periods(
                returns[col],
                grayscale=grayscale,
                figsize=(8, 4),
                subtitle=False,
                title=col,
                savefig={"fname": figfile, "format": figfmt},
                show=False,
                ylabel="",
                compounded=compounded,
                prepare_returns=False,
            )
            embed.append(figfile)
        tpl = tpl.replace("{{dd_periods}}", _embed_figure(embed, figfmt))

    # Underwater (drawdown) plot
    figfile = _get_utils()._file_stream()
    _get_plots().drawdown(
        returns,
        grayscale=grayscale,
        figsize=(8, 3),
        subtitle=False,
        savefig={"fname": figfile, "format": figfmt},
        show=False,
        ylabel="",
    )
    tpl = tpl.replace("{{dd_plot}}", _embed_figure(figfile, figfmt))

    # Monthly returns heatmap
    figfile = _get_utils()._file_stream()
    if isinstance(returns, _pd.Series):
        _get_plots().monthly_heatmap(
            returns,
            benchmark,
            grayscale=grayscale,
            figsize=(8, 4),
            cbar=False,
            returns_label=returns.name,
            savefig={"fname": figfile, "format": figfmt},
            show=False,
            ylabel="",
            compounded=compounded,
            active=active,
        )
        tpl = tpl.replace("{{monthly_heatmap}}", _embed_figure(figfile, figfmt))
    elif isinstance(returns, _pd.DataFrame):
        # Handle multiple strategy columns
        embed = []
        for col in returns.columns:
            _get_plots().monthly_heatmap(
                returns[col],
                benchmark,
                grayscale=grayscale,
                figsize=(8, 4),
                cbar=False,
                returns_label=col,
                savefig={"fname": figfile, "format": figfmt},
                show=False,
                ylabel="",
                compounded=compounded,
                active=active,
            )
            embed.append(figfile)
        tpl = tpl.replace("{{monthly_heatmap}}", _embed_figure(embed, figfmt))

    # Returns distribution analysis
    figfile = _get_utils()._file_stream()

    if isinstance(returns, _pd.Series):
        _get_plots().distribution(
            returns,
            grayscale=grayscale,
            figsize=(8, 4),
            subtitle=False,
            title=returns.name,
            savefig={"fname": figfile, "format": figfmt},
            show=False,
            ylabel="",
            compounded=compounded,
            prepare_returns=False,
        )
        tpl = tpl.replace("{{returns_dist}}", _embed_figure(figfile, figfmt))
    elif isinstance(returns, _pd.DataFrame):
        # Handle multiple strategy columns
        embed = []
        for col in returns.columns:
            _get_plots().distribution(
                returns[col],
                grayscale=grayscale,
                figsize=(8, 4),
                subtitle=False,
                title=col,
                savefig={"fname": figfile, "format": figfmt},
                show=False,
                ylabel="",
                compounded=compounded,
                prepare_returns=False,
            )
            embed.append(figfile)
        tpl = tpl.replace("{{returns_dist}}", _embed_figure(embed, figfmt))

    # Clean up any remaining template placeholders
    tpl = _regex.sub(r"\{\{(.*?)\}\}", "", tpl)
    tpl = tpl.replace("white-space:pre;", "")

    # Handle output - either download in browser or save to file
    if output is None:
        if _get_utils()._in_notebook():
            _download_html(tpl, download_filename)
        else:
            # Save to temp file and open in browser
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".html", delete=False, encoding="utf-8"
            ) as f:
                f.write(tpl)
                temp_path = f.name
            webbrowser.open("file://" + temp_path)
        return

    # Write HTML content to specified output file
    with open(output, "w", encoding="utf-8") as f:
        f.write(tpl)


# Metric rows removed from the simple tearsheet (matched by normalized label,
# i.e. case-insensitive and ignoring '%').
_SIMPLE_DROP_METRICS = [
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
]


def html_simple(
    returns,
    benchmark=None,
    rf=0.0,
    grayscale=False,
    title="Strategy Tearsheet",
    output=None,
    compounded=True,
    periods_per_year=252,
    download_filename="quantstats-tearsheet.html",
    figfmt="svg",
    template_path=None,
    match_dates=True,
    **kwargs,
):
    """
    Generate a simplified HTML tearsheet for equity-curve evaluation.

    A leaner variant of :func:`html` that keeps the metrics and charts most
    relevant when judging an equity curve, and drops noisier sections. Relative
    to the full report it removes the rolling beta/Sharpe/Sortino charts, the
    cumulative daily-returns chart, the monthly-returns heatmap and the return
    quantiles chart; the monthly-returns histogram is replaced by a daily one.
    The metrics table drops averages/ratios and max-drawdown dates, and the
    worst-drawdowns table is limited to the top 5.

    Parameters mirror :func:`html`. ``returns`` should be a daily-returns Series
    (or single/multi-column DataFrame). ``benchmark`` is optional.
    """
    from ._reporting import engine, sections

    if match_dates:
        returns = returns.dropna()

    win_year, win_half_year = _get_trading_periods(periods_per_year)

    if template_path is None:
        template_path = Path(__file__).parent / "report_simple.html"
    else:
        template_path = Path(template_path)
    template_path = template_path.resolve()
    if not template_path.exists():
        raise FileNotFoundError(f"Template file not found: {template_path}")
    if not template_path.is_file():
        raise ValueError(f"Template path is not a file: {template_path}")
    tpl = template_path.read_text(encoding="utf-8")

    if match_dates:
        returns = returns.dropna()
    returns = _get_utils()._prepare_returns(returns)

    strategy_title = kwargs.get("strategy_title", "Strategy")
    if (
        isinstance(returns, _pd.DataFrame)
        and len(returns.columns) > 1
        and isinstance(strategy_title, str)
    ):
        strategy_title = list(returns.columns)

    benchmark_original = None
    if benchmark is not None:
        benchmark_title = kwargs.get("benchmark_title")
        if benchmark_title is None:
            if isinstance(benchmark, str):
                benchmark_title = benchmark
            elif isinstance(benchmark, _pd.Series):
                benchmark_title = benchmark.name if benchmark.name else "Benchmark"
            elif isinstance(benchmark, _pd.DataFrame):
                col_name = benchmark[benchmark.columns[0]].name
                benchmark_title = col_name if col_name else "Benchmark"
        if benchmark_title is None:
            benchmark_title = "Benchmark"

        if isinstance(benchmark, str):
            benchmark_original = _get_utils().download_returns(benchmark)
            if rf != 0:
                benchmark_original = _get_utils().to_excess_returns(
                    benchmark_original, rf, nperiods=periods_per_year
                )
        elif isinstance(benchmark, _pd.Series):
            benchmark_original = benchmark.copy()
        else:
            benchmark_original = benchmark

        benchmark = _get_utils()._prepare_benchmark(benchmark, returns.index, rf)
        if match_dates is True:
            returns, benchmark = _match_dates(returns, benchmark)
    else:
        benchmark_title = None

    # Header substitutions
    date_range = returns.index.strftime("%e %b, %Y")
    full_title = f"{title} (Compounded)" if compounded else title

    params_parts = []
    user_params = kwargs.get("parameters", {})
    if user_params:
        for key, value in user_params.items():
            params_parts.append(f"{key}: {value}")
    if benchmark_title:
        params_parts.append(f"Benchmark: {benchmark_title.upper()}")
    params_parts.append(f"Periods/Year: {periods_per_year}")
    params_parts.append(f"RF: {rf:.1%}")
    params_str = " &bull; ".join(params_parts)
    if params_str:
        params_str += " | "

    matched_dates_str = (
        " (matched dates)" if match_dates and benchmark is not None else ""
    )

    if benchmark is not None:
        benchmark.name = benchmark_title
    if isinstance(returns, _pd.Series):
        returns.name = strategy_title
    elif isinstance(returns, _pd.DataFrame):
        returns.columns = (
            strategy_title if isinstance(strategy_title, list) else [strategy_title]
        )

    # Metrics table (full set minus the simple drop list)
    metrics_html = sections.metrics_table(
        returns,
        benchmark=benchmark,
        rf=rf,
        compounded=compounded,
        periods_per_year=periods_per_year,
        drop_rows=_SIMPLE_DROP_METRICS,
        collapse_separators=True,
        style_separators=True,
        strategy_title=strategy_title,
        benchmark_title=benchmark_title,
        prepare_returns=False,
    )

    # EOY table
    eoy_html = sections.eoy_table(
        returns,
        benchmark=benchmark,
        benchmark_original=benchmark_original,
        compounded=compounded,
        strategy_title=strategy_title,
        benchmark_title=benchmark_title,
    )
    eoy_title = (
        "<h3>EOY Returns vs Benchmark</h3>"
        if benchmark is not None
        else "<h3>EOY Returns</h3>"
    )

    # Worst 5 drawdowns table
    if isinstance(returns, _pd.Series):
        dd_info_html = sections.dd_table(returns, top=5)
    else:
        parts = []
        for col in returns.columns:
            parts.append(f"<h3>{col}</h3><br>" + sections.dd_table(returns[col], top=5))
        dd_info_html = "".join(parts)

    fragments = {
        "title": full_title,
        "date_range": date_range[0] + " - " + date_range[-1],
        "matched_dates": matched_dates_str,
        "params": params_str,
        "v": __version__,
        "generated_at": _dt.now().strftime("%d %b, %Y %H:%M").lstrip("0"),
        "metrics": metrics_html,
        "eoy_title": eoy_title,
        "eoy_table": eoy_html,
        "dd_info": dd_info_html,
    }

    # Charts kept in the simple report
    fragments["returns"] = sections.plot_section(
        "returns", returns, benchmark=benchmark, figfmt=figfmt,
        grayscale=grayscale, figsize=(8, 5), subtitle=False, ylabel="",
        compound=compounded, prepare_returns=False,
    )
    fragments["log_returns"] = sections.plot_section(
        "log_returns", returns, benchmark=benchmark, figfmt=figfmt,
        grayscale=grayscale, figsize=(8, 4), subtitle=False, ylabel="",
        compound=compounded, prepare_returns=False,
    )
    if benchmark is not None:
        fragments["vol_returns"] = sections.plot_section(
            "returns", returns, benchmark=benchmark, figfmt=figfmt,
            match_volatility=True, grayscale=grayscale, figsize=(8, 4),
            subtitle=False, ylabel="", compound=compounded, prepare_returns=False,
        )
    fragments["eoy_returns"] = sections.plot_section(
        "yearly_returns", returns, benchmark=benchmark, figfmt=figfmt,
        grayscale=grayscale, figsize=(8, 4), subtitle=False, ylabel="",
        compounded=compounded, prepare_returns=False,
    )
    fragments["daily_dist"] = sections.plot_section(
        "histogram", returns, benchmark=benchmark, figfmt=figfmt,
        resample="D", grayscale=grayscale, figsize=(7, 4), subtitle=False,
        ylabel="", compounded=compounded, xlim_quantile=0.01, adaptive_bins=80,
        prepare_returns=False,
    )
    fragments["rolling_vol"] = sections.plot_section(
        "rolling_volatility", returns, benchmark=benchmark, figfmt=figfmt,
        grayscale=grayscale, figsize=(8, 3), subtitle=False, ylabel="",
        period=win_half_year, periods_per_year=win_year,
    )
    if isinstance(returns, _pd.Series):
        fragments["dd_periods"] = sections.plot_section(
            "drawdowns_periods", returns, benchmark=None, figfmt=figfmt,
            grayscale=grayscale, figsize=(8, 4), subtitle=False,
            title=returns.name, ylabel="", compounded=compounded,
            prepare_returns=False,
        )
    fragments["dd_plot"] = sections.plot_section(
        "drawdown", returns, benchmark=None, figfmt=figfmt,
        grayscale=grayscale, figsize=(8, 3), subtitle=False, ylabel="",
    )

    rendered = engine.render_report(tpl, fragments)

    if output is None:
        if _get_utils()._in_notebook():
            _download_html(rendered, download_filename)
        else:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".html", delete=False, encoding="utf-8"
            ) as f:
                f.write(rendered)
                temp_path = f.name
            webbrowser.open("file://" + temp_path)
        return

    with open(output, "w", encoding="utf-8") as f:
        f.write(rendered)


# Cross-model comparison columns: (summary_key, display_label, is_percent, highlight_risk)
_MC_TABLE_COLUMNS = [
    ("cagr_p5", "CAGR p5", True, True),
    ("cagr_median", "CAGR Median", True, False),
    ("cagr_p95", "CAGR p95", True, False),
    ("maxdd_median", "MaxDD Median", True, False),
    ("maxdd_p95", "MaxDD Worst 5%", True, True),
    ("terminal_median", "Return Median", True, False),
    ("prob_loss", "P(Loss)", True, True),
    ("cvar_5", "CVaR 5%", True, True),
    ("bust_prob", "P(Bust)", True, True),
    ("goal_prob", "P(Goal)", True, False),
    ("realism_pct", "Hist. Percentile", False, False),
]

_MC_KPI_METRICS = [
    ("cagr_p5", "CAGR p5", True),
    ("cagr_median", "CAGR Median", True),
    ("maxdd_p95", "MaxDD Worst 5%", True),
    ("prob_loss", "P(Loss)", True),
    ("cvar_5", "CVaR 5%", True),
    ("bust_prob", "P(Bust)", True),
]

# Visual tone and one-line hint for each KPI card.
_MC_KPI_CARD_META = {
    "cagr_p5": ("risk", "5% worst-case annual return"),
    "cagr_median": ("upside", "central annual return scenario"),
    "maxdd_p95": ("risk", "drawdown exceeded by 5% of paths"),
    "prob_loss": ("risk", "paths ending below break-even"),
    "cvar_5": ("risk", "average loss in the worst 5% tail"),
    "bust_prob": ("risk", "paths hitting the bust drawdown"),
}


def _mc_format_value(key: str, val: float, is_pct: bool) -> str:
    if key == "realism_pct":
        return f"{val:.0f}%"
    if is_pct:
        return f"{val * 100:,.1f}%"
    return f"{val:,.2f}"


def _mc_cvar_label(horizon: int, periods: float) -> str:
    years = horizon / periods if periods else 1.0
    if abs(years - 1.0) < 0.05:
        return "CVaR 5% (1y)"
    if abs(years - round(years)) < 0.05 and years >= 1:
        return f"CVaR 5% ({int(round(years))}y)"
    return f"CVaR 5% ({horizon}p)"


def _mc_build_rows(results, columns, extra: dict[str, dict] | None = None):
    """Build formatted comparison rows keyed by model label."""
    rows = {}
    for res in results.values():
        summary = dict(res.summary)
        if extra and res.label in extra:
            summary.update(extra[res.label])
        row = {}
        for key, label, is_pct, _ in columns:
            if key not in summary:
                continue
            row[label] = _mc_format_value(key, summary[key], is_pct)
        rows[res.label] = row
    return rows


def _mc_comparison_html(rows, columns):
    """Render comparison table with risk-quantile columns highlighted."""
    risk_labels = {label for _, label, _, risk in columns if risk}
    active_cols = [
        (label, risk)
        for _, label, _, risk in columns
        if any(label in row for row in rows.values())
    ]
    parts = [
        "<table><thead><tr><th>Model</th>",
        *[f"<th>{label}</th>" for label, _ in active_cols],
        "</tr></thead><tbody>",
    ]
    for model, row in rows.items():
        row_cls = ' class="historical"' if model.startswith("Historical") else ""
        parts.append(f"<tr{row_cls}><td><strong>{model}</strong></td>")
        for label, risk in active_cols:
            val = row.get(label, "—")
            cls = ' class="risk"' if risk and label in risk_labels else ""
            parts.append(f"<td{cls}>{val}</td>")
        parts.append("</tr>")
    parts.append("</tbody></table>")
    return "".join(parts)


def _mc_kpi_row_html(metrics: dict[str, float], cvar_label: str) -> str:
    """KPI cards for a metrics dict (median or envelope)."""
    label_map = {key: label for key, label, _ in _MC_KPI_METRICS}
    label_map["cvar_5"] = cvar_label

    parts = ['<div class="kpi-grid">']
    for key, label, is_pct in _MC_KPI_METRICS:
        if key not in metrics:
            continue
        display = _mc_format_value(key, metrics[key], is_pct)
        tone, hint = _MC_KPI_CARD_META.get(key, ("neutral", ""))
        if tone == "upside" and metrics[key] < 0:
            tone = "risk"
        hint_html = f'<div class="hint">{hint}</div>' if hint else ""
        parts.append(
            f'<div class="kpi {tone}"><div class="label">{label_map.get(key, label)}</div>'
            f'<div class="value">{display}</div>{hint_html}</div>'
        )
    parts.append("</div>")
    return "".join(parts)


def _mc_risk_summary_html(median: dict[str, float], cvar_label: str) -> str:
    """Neutral KPI cards: median across Montecarlo models."""
    if not median:
        return '<p class="note">No Montecarlo models in this run.</p>'
    return _mc_kpi_row_html(median, cvar_label)


def _mc_envelope_attribution_html(
    envelope: dict[str, float],
    attribution: dict[str, str],
    cvar_label: str,
) -> str:
    """Conservative envelope KPIs and per-metric source model attribution."""
    if not envelope:
        return ""
    label_map = {key: label for key, label, _ in _MC_KPI_METRICS}
    label_map["cvar_5"] = cvar_label

    parts = [
        '<h4>Conservative Envelope</h4>',
        _mc_kpi_row_html(envelope, cvar_label),
        '<p class="note">Envelope extremes across Montecarlo models '
        "(most conservative quantiles).</p>",
    ]
    if attribution:
        parts.append(
            "<table><thead><tr><th>Metric</th><th>Envelope</th>"
            "<th>Source Model</th></tr></thead><tbody>"
        )
        for key, label, is_pct in _MC_KPI_METRICS:
            if key not in envelope:
                continue
            val = _mc_format_value(key, envelope[key], is_pct)
            src = attribution.get(key, "—")
            parts.append(
                f"<tr><td>{label_map.get(key, label)}</td>"
                f"<td>{val}</td><td>{src}</td></tr>"
            )
        parts.append("</tbody></table>")
    return "".join(parts)


def _mc_stress_section_html(
    stress_comparison_html: str,
    envelope: dict[str, float],
    attribution: dict[str, str],
    cvar_label: str,
) -> str:
    """Stress block: conservative envelope plus trimmed-bootstrap models."""
    parts = [_mc_envelope_attribution_html(envelope, attribution, cvar_label)]
    if stress_comparison_html:
        parts.append("<h4>Trimmed Bootstrap</h4>")
        parts.append(stress_comparison_html)
    elif not envelope:
        parts.append('<p class="note">No stress-test models in this run.</p>')
    return "".join(parts)


def _mc_calibration_html(results):
    """Per-model calibration diagnostics: one compact row per model.

    Each model exposes a different parameter set, so a wide matrix would be
    mostly empty. Render Model | inline "key: value" pairs instead.
    """
    rows = []
    for res in results.values():
        cal = res.calibration_summary()
        if not cal:
            continue
        params = " &nbsp;&bull;&nbsp; ".join(
            f"{k}: <strong>{v}</strong>" for k, v in cal.items()
        )
        rows.append(f'<tr><td>{res.label}</td><td class="params">{params}</td></tr>')
    if not rows:
        return '<p class="note">No calibration diagnostics available.</p>'
    return (
        "<table><thead><tr><th>Model</th><th>Calibrated Parameters</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
    )


def html_montecarlo(
    returns,
    models=None,
    horizon=None,
    sims=1000,
    bust=None,
    goal=None,
    seed=None,
    rf=0.0,
    title="Montecarlo Tearsheet",
    output=None,
    periods_per_year=None,
    download_filename="quantstats-montecarlo.html",
    figfmt="svg",
    template_path=None,
    confidence_level=0.95,
    drift="historical",
    match_dates=True,
    **kwargs,
):
    """
    Generate a multi-model Montecarlo HTML tearsheet for a single asset.

    Characterises the asset with several models (GBM, jump-diffusion, GARCH,
    Heston, bootstraps, Bayesian, ...) and simulates forward paths from each,
    producing a cross-model comparison table plus per-model fan charts and
    terminal-return distributions.

    Parameters
    ----------
    returns : pd.Series or pd.DataFrame
        Daily simple returns of a single asset. A DataFrame uses its first
        column.
    models : list[str], optional
        Subset of model names to run. Defaults to all registered models.
    horizon : int, optional
        Periods per simulated path. Defaults to one year (``periods_per_year``
        trading days). Pass ``len(returns)`` for full-history horizon.
    sims : int, default 1000
        Number of simulated paths per model.
    bust : float, optional
        Drawdown threshold for bust probability (e.g. ``-0.25``).
    goal : float, optional
        Terminal-return threshold for goal probability (e.g. ``0.5``).
    seed : int, optional
        Base random seed for reproducibility.
    title : str, default "Montecarlo Tearsheet"
        Report title.
    output : str or None, default None
        Output file path. If None, opens in browser / downloads in notebook.
    periods_per_year : int, optional
        Periods per year for annualisation. When ``None``, inferred from the
        return index (252 for weekdays-only, 365 for 24/7 data).
    figfmt : str, default "svg"
        Embedded figure format.
    confidence_level : float, default 0.95
        Confidence band level for the fan charts.
    drift : {"historical", "zero", "rf"}, default "historical"
        Drift handling before calibration (see ``montecarlo.run_models``).
    """
    import matplotlib.pyplot as _plt

    from . import montecarlo as _mc
    from ._reporting import engine
    from .montecarlo import analytics as _mca
    from .montecarlo import plotting as _mcplt
    from .montecarlo.core import infer_periods_per_year

    if match_dates and hasattr(returns, "dropna"):
        returns = returns.dropna()
    returns = _get_utils()._prepare_returns(returns)
    if isinstance(returns, _pd.DataFrame):
        returns = returns.iloc[:, 0]

    asset_title = kwargs.get("strategy_title")
    if asset_title is None:
        asset_title = returns.name if returns.name else "Asset"

    if template_path is None:
        template_path = Path(__file__).parent / "report_montecarlo.html"
    else:
        template_path = Path(template_path)
    template_path = template_path.resolve()
    if not template_path.exists():
        raise FileNotFoundError(f"Template file not found: {template_path}")
    tpl = template_path.read_text(encoding="utf-8")

    periods = (
        float(periods_per_year)
        if periods_per_year is not None
        else infer_periods_per_year(returns)
    )

    results = _mc.run_models(
        returns,
        models=models,
        horizon=horizon,
        sims=sims,
        bust=bust,
        goal=goal,
        seed=seed,
        periods=periods,
        drift=drift,
        rf=rf,
    )

    mc_results = {k: v for k, v in results.items() if v.category == "montecarlo"}
    stress_results = {k: v for k, v in results.items() if v.category == "stress"}
    mc_list = list(mc_results.values())

    used_horizon = next(iter(results.values())).horizon if results else 0
    # Empirical distribution of realised outcomes: every overlapping
    # horizon-length window over the full history (not just the last year).
    hist_windows = _mca.historical_windows(returns, used_horizon)
    hist_summary = _mca.historical_summary(
        returns, used_horizon, periods=periods, bust=bust, goal=goal
    )
    n_hist_windows = int(hist_windows.shape[1]) if hist_windows.size else 0
    hist_terminal = (
        float(_np.median(_mca.terminal_values(hist_windows)))
        if hist_windows.size
        else 0.0
    )

    realism_extra = {}
    for res in mc_results.values():
        realism_extra[res.label] = {
            "realism_pct": _mca.realism_percentile(
                res.sim_returns, hist_terminal
            )
        }

    cvar_label = _mc_cvar_label(used_horizon, periods)
    table_columns = [
        (k, cvar_label if k == "cvar_5" else lbl, pct, risk)
        for k, lbl, pct, risk in _MC_TABLE_COLUMNS
    ]

    median = _mca.model_median_summary(mc_results)
    envelope = _mca.conservative_envelope(mc_results)
    attribution = _mca.envelope_attribution(mc_results, envelope)
    assessment_html = _mc_risk_summary_html(median, cvar_label)

    mc_rows = _mc_build_rows(mc_results, table_columns, extra=realism_extra)
    if hist_summary:
        hist_row = {}
        for key, label, is_pct, _ in table_columns:
            if key in hist_summary:
                hist_row[label] = _mc_format_value(key, hist_summary[key], is_pct)
        hist_label = f"Historical ({n_hist_windows:,} rolling windows)"
        mc_rows = {hist_label: hist_row, **mc_rows}
    comparison_html = _mc_comparison_html(mc_rows, table_columns)

    if stress_results:
        stress_rows = _mc_build_rows(stress_results, table_columns)
        stress_comparison_html = _mc_comparison_html(stress_rows, table_columns)
    else:
        stress_comparison_html = ""
    stress_html = _mc_stress_section_html(
        stress_comparison_html, envelope, attribution, cvar_label
    )

    calibration_html = _mc_calibration_html(results)
    model_names = list(results.keys())

    def _embed(fig):
        figfile = _get_utils()._file_stream()
        fig.savefig(figfile, format=figfmt, bbox_inches="tight")
        _plt.close(fig)
        return _embed_figure(figfile, figfmt)

    overlay_fig = _mcplt.overlay_fan_figure(
        mc_list, level=confidence_level, historical_windows=hist_windows
    )
    maxdd_fig = _mcplt.maxdd_distribution_figure(
        mc_list, historical_windows=hist_windows
    )
    cagr_fig = _mcplt.cagr_quantiles_figure(
        mc_list, historical_windows=hist_windows
    )

    fan_parts = []
    hist_parts = []
    for res in results.values():
        fan_fig = _mcplt.fan_chart_figure(
            res,
            level=confidence_level,
            title=res.label,
            historical_windows=hist_windows,
        )
        fan_parts.append(f'<div class="cell">{_embed(fan_fig)}</div>')
        hist_fig = _mcplt.terminal_hist_figure(
            res, title=res.label, historical_terminal=hist_terminal
        )
        hist_parts.append(f'<div class="cell">{_embed(hist_fig)}</div>')

    date_range = returns.index.strftime("%e %b, %Y")
    horizon_years = used_horizon / periods if periods else 1.0
    params_parts = []
    if bust is not None:
        params_parts.append(f"Bust: {bust:.0%}")
    if goal is not None:
        params_parts.append(f"Goal: {goal:.0%}")
    params_parts.append(f"Horizon: {used_horizon} periods ({horizon_years:.1f}y)")
    params_parts.append(f"Periods/year: {periods:.0f}")
    params_parts.append(f"Sims: {sims}")
    params_parts.append(f"Asset: {str(asset_title).upper()}")
    if drift != "historical":
        params_parts.append(f"Drift: {drift} (stress scenario)")
    params_str = " &bull; ".join(params_parts)
    if params_str:
        params_str += " | "

    seed_str = str(seed) if seed is not None else "random"
    assumptions = (
        f"Assumptions: {sims:,} simulations per model &bull; seed {seed_str} "
        f"&bull; forward horizon {used_horizon} periods ({horizon_years:.1f}y) "
        f"&bull; {periods:.0f} periods/year &bull; drift={drift} "
        f"&bull; models: {', '.join(model_names)}"
    )

    fragments = {
        "title": title,
        "date_range": date_range[0] + " - " + date_range[-1],
        "params": params_str,
        "v": __version__,
        "generated_at": _dt.now().strftime("%d %b, %Y %H:%M").lstrip("0"),
        "assessment": assessment_html,
        "comparison_table": comparison_html,
        "calibration_table": calibration_html,
        "overlay_fan": _embed(overlay_fig),
        "maxdd_dist": _embed(maxdd_fig),
        "cagr_quantiles": _embed(cagr_fig),
        "stress_section": stress_html,
        "fan_charts": "\n".join(fan_parts),
        "terminal_hists": "\n".join(hist_parts),
        "assumptions": assumptions,
    }
    rendered = engine.render_report(tpl, fragments)

    if output is None:
        if _get_utils()._in_notebook():
            _download_html(rendered, download_filename)
        else:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".html", delete=False, encoding="utf-8"
            ) as f:
                f.write(rendered)
                temp_path = f.name
            webbrowser.open("file://" + temp_path)
        return

    with open(output, "w", encoding="utf-8") as f:
        f.write(rendered)


def _ad_status_counts_html(result):
    counts = {"excellent": 0, "good": 0, "warning": 0, "critical": 0}
    for mr in result.metrics:
        for wr in mr.windows.values():
            counts[wr.status] += 1
    hints = {
        "excellent": "well above historical norm",
        "good": "within acceptable range",
        "warning": "drifting from historical norm",
        "critical": "significant deviation from norm",
    }
    parts = ['<div class="kpi-grid">']
    for key, label in (
        ("excellent", "Excellent"),
        ("good", "Good"),
        ("warning", "Warning"),
        ("critical", "Critical"),
    ):
        parts.append(
            f'<div class="kpi {key}"><div class="label">{label}</div>'
            f'<div class="value">{counts[key]}</div>'
            f'<div class="hint">{hints[key]}</div></div>'
        )
    parts.append("</div>")
    return "".join(parts)


def _ad_score_tone(score_pct: float) -> str:
    """Map aggregate health score to the same status palette as metric cells."""
    if score_pct >= 75:
        return "excellent"
    if score_pct >= 50:
        return "good"
    if score_pct >= 25:
        return "warning"
    return "critical"


def _ad_summary_table_html(result):
    from .alphadecay.core import status_label

    windows = result.windows
    parts = [
        "<table><thead><tr><th>Metric</th>",
        *[f"<th>{w}d</th>" for w in windows],
        "</tr></thead><tbody>",
    ]
    for mr in result.metrics:
        parts.append(f"<tr><td>{mr.spec.label}</td>")
        for w in windows:
            wr = mr.windows[w]
            lbl = status_label(wr.status)
            z_txt = f"z={wr.z_score:+.2f}" if not _np.isnan(wr.z_score) else ""
            parts.append(
                f'<td><span class="badge {wr.status}">{lbl}</span>'
                f'<br><span class="metric-meta">{z_txt}</span></td>'
            )
        parts.append("</tr>")
    parts.append("</tbody></table>")
    return "".join(parts)


def _ad_metric_rows_html(result, embed_fn):
    from .alphadecay import plotting as _adplt
    from .alphadecay.core import format_value, status_label

    parts = []
    for mr in result.metrics:
        parts.append(f'<div class="metric-row" id="metric_{mr.spec.key}">')
        parts.append(f"<h4>{mr.spec.label}</h4>")
        parts.append('<div class="grid">')
        for w in result.windows:
            wr = mr.windows[w]
            fig = _adplt.metric_distribution_figure(wr, mr.spec, title=f"{mr.spec.label} — {w}d")
            meta = (
                f"Obs: {format_value(mr.spec, wr.observed)} &bull; "
                f"Mean: {format_value(mr.spec, wr.mean)} &bull; "
                f"z={wr.z_score:+.2f} &bull; "
                f"P={wr.percentile:.0f}% &bull; "
                f'<span class="badge {wr.status}">{status_label(wr.status)}</span>'
            )
            parts.append(
                f'<div class="cell"><div class="metric-meta">{meta}</div>'
                f"{embed_fn(fig)}</div>"
            )
        parts.append("</div></div>")
    return "\n".join(parts)


def _ad_sota_stats_html(result):
    from .alphadecay.core import status_label

    cusum = result.cusum
    tuw = result.time_underwater
    alarm = "ALARM" if cusum.alarm else "Normal"
    uw = (
        f"{tuw.current_days} days underwater"
        if tuw.is_underwater
        else "At equity peak"
    )
    return (
        '<table><thead><tr><th>Diagnostic</th><th>Current</th><th>Reference</th>'
        "<th>Status</th></tr></thead><tbody>"
        f"<tr><td>CUSUM</td><td>{cusum.current:.4f}</td>"
        f"<td>Threshold {cusum.threshold:.4f} ({cusum.pct_of_threshold:.0%})</td>"
        f"<td>{alarm}</td></tr>"
        f"<tr><td>Time Underwater</td><td>{uw}</td>"
        f"<td>Mean duration {tuw.mean:.0f}d (z={tuw.z_score:+.2f})</td>"
        f'<td><span class="badge {tuw.status}">{status_label(tuw.status)}</span></td></tr>'
        "</tbody></table>"
    )


def html_alpha_decay(
    returns,
    windows=(7, 15, 30),
    rf=0.0,
    title="Alpha Decay Tearsheet",
    output=None,
    periods_per_year=252,
    download_filename="quantstats-alpha-decay.html",
    figfmt="svg",
    template_path=None,
    match_dates=True,
    cusum_k=0.5,
    cusum_h=4.0,
    **kwargs,
):
    """
    Generate an alpha-decay HTML tearsheet for a single asset.

    Computes rolling risk metrics over short windows (7/15/30 days by default),
    compares the latest observation against the historical distribution of each
    metric (z-score traffic light), and adds CUSUM + time-underwater diagnostics.

    Parameters
    ----------
    returns : pd.Series or pd.DataFrame
        Daily simple returns of a single asset. A DataFrame uses its first column.
    windows : tuple[int], default (7, 15, 30)
        Rolling window sizes in trading days.
    rf : float, default 0.0
        Annual risk-free rate for CAGR.
    title : str, default "Alpha Decay Tearsheet"
        Report title.
    output : str or None, default None
        Output file path. If None, opens in browser / downloads in notebook.
    periods_per_year : int, default 252
        Periods per year for annualisation.
    figfmt : str, default "svg"
        Embedded figure format.
    cusum_k : float, default 0.5
        CUSUM slack parameter in units of return standard deviation.
    cusum_h : float, default 4.0
        CUSUM alarm threshold in units of return standard deviation.
    """
    import matplotlib.pyplot as _plt

    from . import alphadecay as _ad
    from ._reporting import engine
    from .alphadecay import plotting as _adplt

    if match_dates and hasattr(returns, "dropna"):
        returns = returns.dropna()
    returns = _get_utils()._prepare_returns(returns)
    if isinstance(returns, _pd.DataFrame):
        returns = returns.iloc[:, 0]

    asset_title = kwargs.get("strategy_title")
    if asset_title is None:
        asset_title = returns.name if returns.name else "Asset"

    if template_path is None:
        template_path = Path(__file__).parent / "report_alpha_decay.html"
    else:
        template_path = Path(template_path)
    template_path = template_path.resolve()
    if not template_path.exists():
        raise FileNotFoundError(f"Template file not found: {template_path}")
    tpl = template_path.read_text(encoding="utf-8")

    result = _ad.analyze(
        returns,
        windows=tuple(windows),
        rf=rf,
        periods=periods_per_year,
        cusum_k=cusum_k,
        cusum_h=cusum_h,
        asset_label=str(asset_title),
    )

    def _embed(fig):
        figfile = _get_utils()._file_stream()
        fig.savefig(figfile, format=figfmt, bbox_inches="tight")
        _plt.close(fig)
        return _embed_figure(figfile, figfmt)

    cusum_fig = _adplt.cusum_figure(result.cusum)
    tuw_fig = _adplt.time_underwater_figure(result.time_underwater)

    date_range = returns.index.strftime("%e %b, %Y")
    windows_label = "/".join(str(w) for w in windows)
    params_parts = [
        f"Windows: {windows_label}d",
        f"Asset: {str(asset_title).upper()}",
        f"Metrics: {len(result.metrics)}",
    ]
    params_str = " &bull; ".join(params_parts) + " | "

    score_tone = _ad_score_tone(result.score_pct)
    score_html = (
        '<div class="kpi-grid score-grid">'
        f'<div class="kpi {score_tone}"><div class="label">Health Score</div>'
        f'<div class="value">{result.score}/{result.total}</div>'
        f'<div class="hint">metric-window cells rated Good or Excellent '
        f"({result.score_pct:.0f}%)</div></div></div>"
    )

    assumptions = (
        f"Assumptions: windows {windows_label}d &bull; "
        f"{len(result.metrics)} metrics &bull; "
        f"CUSUM k={cusum_k}σ h={cusum_h}σ &bull; "
        f"periods/year={periods_per_year} &bull; rf={rf:.2%}"
    )

    fragments = {
        "title": title,
        "date_range": date_range[0] + " - " + date_range[-1],
        "params": params_str,
        "v": __version__,
        "generated_at": _dt.now().strftime("%d %b, %Y %H:%M").lstrip("0"),
        "windows_label": windows_label,
        "score_summary": score_html,
        "status_counts": _ad_status_counts_html(result),
        "summary_table": _ad_summary_table_html(result),
        "metric_rows": _ad_metric_rows_html(result, _embed),
        "cusum_chart": _embed(cusum_fig),
        "time_underwater": _embed(tuw_fig),
        "sota_stats": _ad_sota_stats_html(result),
        "assumptions": assumptions,
    }
    rendered = engine.render_report(tpl, fragments)

    if output is None:
        if _get_utils()._in_notebook():
            _download_html(rendered, download_filename)
        else:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".html", delete=False, encoding="utf-8"
            ) as f:
                f.write(rendered)
                temp_path = f.name
            webbrowser.open("file://" + temp_path)
        return

    with open(output, "w", encoding="utf-8") as f:
        f.write(rendered)


def full(
    returns,
    benchmark=None,
    rf=0.0,
    grayscale=False,
    figsize=(8, 5),
    display=True,
    compounded=True,
    periods_per_year=252,
    match_dates=True,
    **kwargs,
):
    """
    Generate a comprehensive performance analysis report.

    This function creates a full performance analysis including metrics,
    worst drawdowns analysis, and complete visualization suite. It's designed
    for detailed portfolio analysis and can handle both single strategies
    and multiple strategy comparisons.

    Parameters
    ----------
    returns : pd.Series or pd.DataFrame
        Daily returns data for the strategy/portfolio
    benchmark : pd.Series, str, or None, default None
        Benchmark returns for comparison
    rf : float, default 0.0
        Risk-free rate for calculations (as decimal)
    grayscale : bool, default False
        Whether to generate charts in grayscale
    figsize : tuple, default (8, 5)
        Figure size for plots as (width, height)
    display : bool, default True
        Whether to display results in notebook/console
    compounded : bool, default True
        Whether to compound returns for calculations
    periods_per_year : int, default 252
        Number of trading periods per year
    match_dates : bool, default True
        Whether to align returns and benchmark start dates
    **kwargs
        Additional keyword arguments:
        - strategy_title: Custom name for the strategy
        - benchmark_title: Custom name for the benchmark
        - active_returns: Whether to show active returns vs benchmark

    Returns
    -------
    None
        Displays comprehensive analysis including metrics, drawdowns, and plots

    Examples
    --------
    >>> full(returns, benchmark='^GSPC', rf=0.02)
    >>> full(returns, figsize=(10, 6), grayscale=True)
    """
    # prepare timeseries
    if match_dates:
        returns = returns.dropna()
    # Clean and prepare returns data
    returns = _get_utils()._prepare_returns(returns)

    # Process benchmark if provided
    if benchmark is not None:
        benchmark = _get_utils()._prepare_benchmark(benchmark, returns.index, rf)
        if match_dates is True:
            returns, benchmark = _match_dates(returns, benchmark)

    # Extract title parameters from kwargs
    benchmark_title = None
    if benchmark is not None:
        benchmark_title = kwargs.get("benchmark_title", "Benchmark")
    strategy_title = kwargs.get("strategy_title", "Strategy")
    active = kwargs.get("active_returns", False)

    # Handle multiple strategy columns
    if (
        isinstance(returns, _pd.DataFrame)
        and len(returns.columns) > 1
        and isinstance(strategy_title, str)
    ):
        strategy_title = list(returns.columns)

    # Set names for display purposes
    if benchmark is not None:
        benchmark.name = benchmark_title
    if isinstance(returns, _pd.Series):
        returns.name = strategy_title
    elif isinstance(returns, _pd.DataFrame):
        returns.columns = strategy_title

    # Calculate drawdown analysis for worst periods display
    dd = _get_stats().to_drawdown_series(returns)

    # Process drawdown details based on data type
    if isinstance(dd, _pd.Series):
        col = _get_stats().drawdown_details(dd).columns[4]
        dd_info = _get_stats().drawdown_details(dd).sort_values(by=col, ascending=True)[:5]
        if not dd_info.empty:
            dd_info.index = range(1, min(6, len(dd_info) + 1))
            dd_info.columns = map(lambda x: str(x).title(), dd_info.columns)
    elif isinstance(dd, _pd.DataFrame):
        # Handle multiple strategy columns
        col = _get_stats().drawdown_details(dd).columns.get_level_values(1)[4]
        dd_info_dict = {}
        for ptf in dd.columns:
            dd_info = _get_stats().drawdown_details(dd[ptf]).sort_values(
                by=col, ascending=True
            )[:5]
            if not dd_info.empty:
                dd_info.index = range(1, min(6, len(dd_info) + 1))
                dd_info.columns = map(lambda x: str(x).title(), dd_info.columns)
            dd_info_dict[ptf] = dd_info

    # Display results based on environment (notebook vs console)
    if _get_utils()._in_notebook():
        # Display in Jupyter notebook with HTML formatting
        iDisplay(iHTML("<h4>Performance Metrics</h4>"))
        iDisplay(
            metrics(
                returns=returns,
                benchmark=benchmark,
                rf=rf,
                display=display,
                mode="full",
                compounded=compounded,
                periods_per_year=periods_per_year,
                prepare_returns=False,
                benchmark_title=benchmark_title,
                strategy_title=strategy_title,
            )
        )

        # Display worst drawdowns analysis
        if isinstance(dd, _pd.Series):
            iDisplay(iHTML('<h4 style="margin-bottom:20px">Worst 5 Drawdowns</h4>'))
            if dd_info.empty:
                iDisplay(iHTML("<p>(no drawdowns)</p>"))
            else:
                iDisplay(dd_info)
        elif isinstance(dd, _pd.DataFrame):
            # Display drawdowns for each strategy
            for ptf, dd_info in dd_info_dict.items():
                iDisplay(
                    iHTML(
                        f'<h4 style="margin-bottom:20px">{ptf} - Worst 5 Drawdowns</h4>'
                    )
                )
                if dd_info.empty:
                    iDisplay(iHTML("<p>(no drawdowns)</p>"))
                else:
                    iDisplay(dd_info)

        iDisplay(iHTML("<h4>Strategy Visualization</h4>"))
    else:
        # Display in console/terminal environment
        _print_parameters_table(
            benchmark_title=benchmark_title,
            periods_per_year=periods_per_year,
            rf=rf,
            compounded=compounded,
            match_dates=match_dates,
        )
        print("[Performance Metrics]\n")
        metrics(
            returns=returns,
            benchmark=benchmark,
            rf=rf,
            display=display,
            mode="full",
            compounded=compounded,
            periods_per_year=periods_per_year,
            prepare_returns=False,
            benchmark_title=benchmark_title,
            strategy_title=strategy_title,
        )
        print("\n\n")
        print("[Worst 5 Drawdowns]\n")

        # Display drawdowns in tabular format
        if isinstance(dd, _pd.Series):
            if dd_info.empty:
                print("(no drawdowns)")
            else:
                print(
                    _tabulate(
                        dd_info, headers="keys", tablefmt="simple", floatfmt=".2f"
                    )
                )
        elif isinstance(dd, _pd.DataFrame):
            for ptf, dd_info in dd_info_dict.items():
                if dd_info.empty:
                    print("(no drawdowns)")
                else:
                    print(f"{ptf}\n")
                    print(
                        _tabulate(
                            dd_info, headers="keys", tablefmt="simple", floatfmt=".2f"
                        )
                    )

        print("\n\n")
        print("[Strategy Visualization]\nvia Matplotlib")

    # Generate comprehensive plots
    plots(
        returns=returns,
        benchmark=benchmark,
        grayscale=grayscale,
        figsize=figsize,
        mode="full",
        compounded=compounded,
        periods_per_year=periods_per_year,
        prepare_returns=False,
        benchmark_title=benchmark_title,
        strategy_title=strategy_title,
        active=active,
    )


def basic(
    returns,
    benchmark=None,
    rf=0.0,
    grayscale=False,
    figsize=(8, 5),
    display=True,
    compounded=True,
    periods_per_year=252,
    match_dates=True,
    **kwargs,
):
    """
    Generate a basic performance analysis report.

    This function creates a simplified performance analysis with essential
    metrics and basic visualizations. It's designed for quick portfolio
    analysis when detailed analysis is not needed.

    Parameters
    ----------
    returns : pd.Series or pd.DataFrame
        Daily returns data for the strategy/portfolio
    benchmark : pd.Series, str, or None, default None
        Benchmark returns for comparison
    rf : float, default 0.0
        Risk-free rate for calculations (as decimal)
    grayscale : bool, default False
        Whether to generate charts in grayscale
    figsize : tuple, default (8, 5)
        Figure size for plots as (width, height)
    display : bool, default True
        Whether to display results in notebook/console
    compounded : bool, default True
        Whether to compound returns for calculations
    periods_per_year : int, default 252
        Number of trading periods per year
    match_dates : bool, default True
        Whether to align returns and benchmark start dates
    **kwargs
        Additional keyword arguments:
        - strategy_title: Custom name for the strategy
        - benchmark_title: Custom name for the benchmark
        - active_returns: Whether to show active returns vs benchmark

    Returns
    -------
    None
        Displays basic analysis including essential metrics and plots

    Examples
    --------
    >>> basic(returns, benchmark='^GSPC')
    >>> basic(returns, figsize=(10, 6), display=False)
    """
    # prepare timeseries
    if match_dates:
        returns = returns.dropna()
    # Clean and prepare returns data
    returns = _get_utils()._prepare_returns(returns)

    # Process benchmark if provided
    if benchmark is not None:
        benchmark = _get_utils()._prepare_benchmark(benchmark, returns.index, rf)
        if match_dates is True:
            returns, benchmark = _match_dates(returns, benchmark)

    # Extract title parameters from kwargs
    benchmark_title = None
    if benchmark is not None:
        benchmark_title = kwargs.get("benchmark_title", "Benchmark")
    strategy_title = kwargs.get("strategy_title", "Strategy")
    active = kwargs.get("active_returns", False)

    # Handle multiple strategy columns
    if (
        isinstance(returns, _pd.DataFrame)
        and len(returns.columns) > 1
        and isinstance(strategy_title, str)
    ):
        strategy_title = list(returns.columns)

    # Display results based on environment (notebook vs console)
    if _get_utils()._in_notebook():
        # Display in Jupyter notebook with HTML formatting
        iDisplay(iHTML("<h4>Performance Metrics</h4>"))
        metrics(
            returns=returns,
            benchmark=benchmark,
            rf=rf,
            display=display,
            mode="basic",
            compounded=compounded,
            periods_per_year=periods_per_year,
            prepare_returns=False,
            benchmark_title=benchmark_title,
            strategy_title=strategy_title,
        )
        iDisplay(iHTML("<h4>Strategy Visualization</h4>"))
    else:
        # Display in console/terminal environment
        _print_parameters_table(
            benchmark_title=benchmark_title,
            periods_per_year=periods_per_year,
            rf=rf,
            compounded=compounded,
            match_dates=match_dates,
        )
        print("[Performance Metrics]\n")
        metrics(
            returns=returns,
            benchmark=benchmark,
            rf=rf,
            display=display,
            mode="basic",
            compounded=compounded,
            periods_per_year=periods_per_year,
            prepare_returns=False,
            benchmark_title=benchmark_title,
            strategy_title=strategy_title,
        )

        print("\n\n")
        print("[Strategy Visualization]\nvia Matplotlib")

    # Generate basic plots
    plots(
        returns=returns,
        benchmark=benchmark,
        grayscale=grayscale,
        figsize=figsize,
        mode="basic",
        compounded=compounded,
        periods_per_year=periods_per_year,
        prepare_returns=False,
        benchmark_title=benchmark_title,
        strategy_title=strategy_title,
        active=active,
    )


def metrics(
    returns,
    benchmark=None,
    rf=0.0,
    display=True,
    mode="basic",
    sep=False,
    compounded=True,
    periods_per_year=252,
    prepare_returns=True,
    match_dates=True,
    **kwargs,
):
    """
    Calculate comprehensive performance metrics for portfolio analysis.

    This function computes a wide range of performance metrics including
    returns, risk measures, ratios, and statistical measures. It can handle
    both single strategies and multiple strategy comparisons with optional
    benchmark analysis.

    Parameters
    ----------
    returns : pd.Series or pd.DataFrame
        Daily returns data for the strategy/portfolio
    benchmark : pd.Series, str, or None, default None
        Benchmark returns for comparison
    rf : float, default 0.0
        Risk-free rate for calculations (as decimal)
    display : bool, default True
        Whether to display results in formatted table
    mode : str, default "basic"
        Analysis mode - "basic" for essential metrics, "full" for comprehensive
    sep : bool, default False
        Whether to include separator rows in output
    compounded : bool, default True
        Whether to compound returns for calculations
    periods_per_year : int, default 252
        Number of trading periods per year
    prepare_returns : bool, default True
        Whether to prepare/clean returns data
    match_dates : bool, default True
        Whether to align returns and benchmark start dates
    **kwargs
        Additional keyword arguments:
        - strategy_title: Custom name for the strategy
        - benchmark_title: Custom name for the benchmark
        - as_pct: Whether to return percentages
        - internal: Internal calculation flag

    Returns
    -------
    pd.DataFrame or None
        DataFrame with performance metrics if display=False, else None

    Examples
    --------
    >>> metrics_df = metrics(returns, benchmark='^GSPC', display=False)
    >>> metrics(returns, mode="full", rf=0.02)
    """
    # Clean returns data if date matching is enabled
    if match_dates:
        returns = returns.dropna()
    # Remove timezone information from index for consistent processing
    returns.index = returns.index.tz_localize(None)

    # Get trading periods for annualization calculations
    win_year, _ = _get_trading_periods(periods_per_year)

    # Extract column names from kwargs or use defaults
    benchmark_colname = kwargs.get("benchmark_title", "Benchmark")
    strategy_colname = kwargs.get("strategy_title", "Strategy")

    # Handle benchmark column naming
    if benchmark is not None:
        if isinstance(benchmark, str):
            benchmark_colname = f"Benchmark ({benchmark.upper()})"
        elif isinstance(benchmark, _pd.DataFrame) and len(benchmark.columns) > 1:
            raise ValueError(
                "`benchmark` must be a pandas Series, "
                "but a multi-column DataFrame was passed"
            )

    # Handle strategy column naming for multiple strategies
    if isinstance(returns, _pd.DataFrame):
        if len(returns.columns) > 1:
            blank = [""] * len(returns.columns)
            if isinstance(strategy_colname, str):
                strategy_colname = list(returns.columns)
    else:
        blank = [""]

    # if isinstance(returns, _pd.DataFrame):
    #     if len(returns.columns) > 1:
    #         raise ValueError("`returns` needs to be a Pandas Series or one column DataFrame. "
    #                          "multi colums DataFrame was passed")
    #     returns = returns[returns.columns[0]]

    # Prepare returns data if requested
    if prepare_returns:
        df = _get_utils()._prepare_returns(returns)

    # Create main DataFrame for calculations
    if isinstance(returns, _pd.Series):
        df = _pd.DataFrame({"returns": returns})
    elif isinstance(returns, _pd.DataFrame):
        df = _pd.DataFrame(
            {
                "returns_" + str(i + 1): returns[strategy_col]
                for i, strategy_col in enumerate(returns.columns)
            }
        )

    # Process benchmark data if provided
    if benchmark is not None:
        benchmark = _get_utils()._prepare_benchmark(benchmark, returns.index, rf)
        if match_dates is True:
            returns, benchmark = _match_dates(returns, benchmark)
            # Truncate df to the aligned date range to exclude leading zeros
            df = df.loc[returns.index]
        df["benchmark"] = benchmark
        # Update blank list for proper formatting
        if isinstance(returns, _pd.Series):
            blank = ["", ""]
            df["returns"] = returns
        elif isinstance(returns, _pd.DataFrame):
            blank = [""] * len(returns.columns) + [""]
            for i, strategy_col in enumerate(returns.columns):
                df["returns_" + str(i + 1)] = returns[strategy_col]

    # Calculate start and end dates for each series
    if isinstance(returns, _pd.Series):
        s_start = {"returns": df["returns"].index.strftime("%Y-%m-%d")[0]}
        s_end = {"returns": df["returns"].index.strftime("%Y-%m-%d")[-1]}
        s_rf = {"returns": rf}
    elif isinstance(returns, _pd.DataFrame):
        df_strategy_columns = [col for col in df.columns if col != "benchmark"]
        s_start = {
            strategy_col: df[strategy_col].dropna().index.strftime("%Y-%m-%d")[0]
            for strategy_col in df_strategy_columns
        }
        s_end = {
            strategy_col: df[strategy_col].dropna().index.strftime("%Y-%m-%d")[-1]
            for strategy_col in df_strategy_columns
        }
        s_rf = {strategy_col: rf for strategy_col in df_strategy_columns}

    # Add benchmark dates if present
    if "benchmark" in df:
        s_start["benchmark"] = df["benchmark"].index.strftime("%Y-%m-%d")[0]
        s_end["benchmark"] = df["benchmark"].index.strftime("%Y-%m-%d")[-1]
        s_rf["benchmark"] = rf

    # Fill missing values with zeros for calculations
    df = df.fillna(0)

    # Determine percentage multiplier for display
    # pct multiplier
    pct = 100 if display or "internal" in kwargs else 1
    if kwargs.get("as_pct", False):
        pct = 100

    # Initialize metrics DataFrame with basic information
    metrics = _pd.DataFrame()
    metrics["Start Period"] = _pd.Series(s_start)
    metrics["End Period"] = _pd.Series(s_end)
    metrics["Risk-Free Rate %"] = _pd.Series(s_rf) * 100
    metrics["Time in Market %"] = _get_stats().exposure(df, prepare_returns=False) * pct

    # Add separator row
    metrics["~"] = blank

    # Calculate return metrics based on compounding preference
    if compounded:
        metrics["Cumulative Return %"] = (_get_stats().comp(df) * pct).map("{:,.2f}".format)
    else:
        metrics["Total Return %"] = (df.sum() * pct).map("{:,.2f}".format)

    # Calculate annualized return (CAGR)
    metrics["CAGR﹪%"] = _get_stats().cagr(df, rf, compounded, win_year) * pct

    # Add separator row
    metrics["~~~~~~~~~~~~~~"] = blank

    # Calculate risk-adjusted return ratios
    metrics["Sharpe"] = _get_stats().sharpe(df, rf, win_year, True)
    metrics["Prob. Sharpe Ratio %"] = (
        _get_stats().probabilistic_sharpe_ratio(df, rf, win_year, False) * pct
    )

    # Add advanced Sharpe metrics for full mode
    if mode.lower() == "full":
        metrics["Smart Sharpe"] = _get_stats().smart_sharpe(df, rf, win_year, True)
        # metrics['Prob. Smart Sharpe Ratio %'] = _get_stats().probabilistic_sharpe_ratio(df, rf, win_year, False, True) * pct

    # Calculate Sortino ratio (downside deviation-based)
    metrics["Sortino"] = _get_stats().sortino(df, rf, win_year, True)
    if mode.lower() == "full":
        # metrics['Prob. Sortino Ratio %'] = _get_stats().probabilistic_sortino_ratio(df, rf, win_year, False) * pct
        metrics["Smart Sortino"] = _get_stats().smart_sortino(df, rf, win_year, True)
        # metrics['Prob. Smart Sortino Ratio %'] = _get_stats().probabilistic_sortino_ratio(
        #     df, rf, win_year, False, True) * pct

    # Calculate adjusted Sortino ratio
    metrics["Sortino/√2"] = metrics["Sortino"] / _sqrt(2)
    if mode.lower() == "full":
        # metrics['Prob. Sortino/√2 Ratio %'] = _get_stats().probabilistic_adjusted_sortino_ratio(
        #     df, rf, win_year, False) * pct
        metrics["Smart Sortino/√2"] = metrics["Smart Sortino"] / _sqrt(2)
        # metrics['Prob. Smart Sortino/√2 Ratio %'] = _get_stats().probabilistic_adjusted_sortino_ratio(
        #     df, rf, win_year, False, True) * pct

    # Calculate Omega ratio (probability-weighted ratio)
    if isinstance(returns, _pd.Series):
        if "benchmark" in df:
            metrics["Omega"] = [
                _get_stats().omega(df["returns"], rf, 0.0, win_year),
                _get_stats().omega(df["benchmark"], rf, 0.0, win_year),
            ]
        else:
            metrics["Omega"] = _get_stats().omega(df["returns"], rf, 0.0, win_year)
    elif isinstance(returns, _pd.DataFrame):
        omega_values = [
            _get_stats().omega(df[strategy_col], rf, 0.0, win_year)
            for strategy_col in df_strategy_columns
        ]
        if "benchmark" in df:
            omega_values.append(_get_stats().omega(df["benchmark"], rf, 0.0, win_year))
        metrics["Omega"] = omega_values

    # Add separator and prepare for drawdown metrics
    metrics["~~~~~~~~"] = blank
    metrics["Max Drawdown %"] = blank
    metrics["Max DD Date"] = blank
    metrics["Max DD Period Start"] = blank
    metrics["Max DD Period End"] = blank
    metrics["Longest DD Days"] = blank

    # Add detailed volatility and risk metrics for full mode
    if mode.lower() == "full":
        # Calculate annualized volatility
        if isinstance(returns, _pd.Series):
            ret_vol = (
                _get_stats().volatility(df["returns"], win_year, True, prepare_returns=False)
                * pct
            )
        elif isinstance(returns, _pd.DataFrame):
            ret_vol = [
                _get_stats().volatility(
                    df[strategy_col], win_year, True, prepare_returns=False
                )
                * pct
                for strategy_col in df_strategy_columns
            ]

        # Add benchmark volatility if present
        if "benchmark" in df:
            bench_vol = (
                _get_stats().volatility(
                    df["benchmark"], win_year, True, prepare_returns=False
                )
                * pct
            )

            vol_ = [ret_vol, bench_vol]
            if isinstance(ret_vol, list):
                metrics["Volatility (ann.) %"] = list(_pd.core.common.flatten(vol_))
            else:
                metrics["Volatility (ann.) %"] = vol_

            # Calculate benchmark-relative metrics
            if isinstance(returns, _pd.Series):
                metrics["R^2"] = _get_stats().r_squared(
                    df["returns"], df["benchmark"], prepare_returns=False
                )
                metrics["Information Ratio"] = _get_stats().information_ratio(
                    df["returns"], df["benchmark"], prepare_returns=False
                )
            elif isinstance(returns, _pd.DataFrame):
                metrics["R^2"] = (
                    [
                        _get_stats().r_squared(
                            df[strategy_col], df["benchmark"], prepare_returns=False
                        ).round(2)
                        for strategy_col in df_strategy_columns
                    ]
                ) + ["-"]
                metrics["Information Ratio"] = (
                    [
                        _get_stats().information_ratio(
                            df[strategy_col], df["benchmark"], prepare_returns=False
                        ).round(2)
                        for strategy_col in df_strategy_columns
                    ]
                ) + ["-"]
        else:
            # No benchmark case
            if isinstance(returns, _pd.Series):
                metrics["Volatility (ann.) %"] = [ret_vol]
            elif isinstance(returns, _pd.DataFrame):
                metrics["Volatility (ann.) %"] = ret_vol

        # Additional risk and return metrics
        metrics["Calmar"] = _get_stats().calmar(df, prepare_returns=False, periods=win_year)
        metrics["Skew"] = _get_stats().skew(df, prepare_returns=False)
        metrics["Kurtosis"] = _get_stats().kurtosis(df, prepare_returns=False)

        # Additional ratios
        metrics["Ulcer Performance Index"] = _get_stats().ulcer_performance_index(df, rf)
        metrics["Risk-Adjusted Return %"] = _get_stats().rar(df, rf, periods=win_year) * pct
        metrics["Risk-Return Ratio"] = _get_stats().risk_return_ratio(df, prepare_returns=False)

        # Add separator
        metrics["~~~~~~~~~~"] = blank

        # Average return metrics
        metrics["Avg. Return %"] = _get_stats().avg_return(df, prepare_returns=False) * pct
        metrics["Avg. Win %"] = _get_stats().avg_win(df, prepare_returns=False) * pct
        metrics["Avg. Loss %"] = _get_stats().avg_loss(df, prepare_returns=False) * pct
        metrics["Win/Loss Ratio"] = _get_stats().win_loss_ratio(df, prepare_returns=False)
        metrics["Profit Ratio"] = _get_stats().profit_ratio(df, prepare_returns=False)

        # Add separator
        metrics["~~~~~~~~~~~"] = blank

        # Expected returns at different frequencies
        metrics["Expected Daily %%"] = (
            _get_stats().expected_return(df, compounded=compounded, prepare_returns=False)
            * pct
        )
        metrics["Expected Monthly %%"] = (
            _get_stats().expected_return(
                df, compounded=compounded, aggregate="ME", prepare_returns=False
            )
            * pct
        )
        metrics["Expected Yearly %%"] = (
            _get_stats().expected_return(
                df, compounded=compounded, aggregate="YE", prepare_returns=False
            )
            * pct
        )

        # Risk management metrics
        metrics["Kelly Criterion %"] = (
            _get_stats().kelly_criterion(df, prepare_returns=False) * pct
        )
        metrics["Risk of Ruin %"] = _get_stats().risk_of_ruin(df, prepare_returns=False)

        # Value at Risk metrics
        metrics["Daily Value-at-Risk %"] = -abs(
            _get_stats().var(df, prepare_returns=False) * pct
        )
        metrics["Expected Shortfall (cVaR) %"] = -abs(
            _get_stats().cvar(df, prepare_returns=False) * pct
        )

    # Add separator
    metrics["~~~~~~"] = blank

    # Consecutive wins/losses analysis (full mode only)
    if mode.lower() == "full":
        metrics["Max Consecutive Wins *int"] = _get_stats().consecutive_wins(df)
        metrics["Max Consecutive Losses *int"] = _get_stats().consecutive_losses(df)

    # Pain-based metrics (Gain/Pain ratio)
    metrics["Gain/Pain Ratio"] = _get_stats().gain_to_pain_ratio(df, rf)
    metrics["Gain/Pain (1M)"] = _get_stats().gain_to_pain_ratio(df, rf, "ME")
    # if mode.lower() == 'full':
    #     metrics['GPR (3M)'] = _get_stats().gain_to_pain_ratio(df, rf, "QE")
    #     metrics['GPR (6M)'] = _get_stats().gain_to_pain_ratio(df, rf, "2Q")
    #     metrics['GPR (1Y)'] = _get_stats().gain_to_pain_ratio(df, rf, "YE")

    # Add separator
    metrics["~~~~~~~"] = blank

    # Trading-based performance metrics
    metrics["Payoff Ratio"] = _get_stats().payoff_ratio(df, prepare_returns=False)
    metrics["Profit Factor"] = _get_stats().profit_factor(df, prepare_returns=False)
    metrics["Common Sense Ratio"] = _get_stats().common_sense_ratio(df, prepare_returns=False)
    metrics["CPC Index"] = _get_stats().cpc_index(df, prepare_returns=False)
    metrics["Tail Ratio"] = _get_stats().tail_ratio(df, prepare_returns=False)
    metrics["Outlier Win Ratio"] = _get_stats().outlier_win_ratio(df, prepare_returns=False)
    metrics["Outlier Loss Ratio"] = _get_stats().outlier_loss_ratio(df, prepare_returns=False)

    # # returns
    metrics["~~"] = blank

    # Time-based return analysis
    today = df.index[-1]  # _dt.today()
    m3 = today - relativedelta(months=3)
    m6 = today - relativedelta(months=6)
    y1 = today - relativedelta(years=1)

    # Calculate period returns based on compounding preference
    if compounded:
        metrics["MTD %"] = (
            _get_stats().comp(df[df.index >= _dt(today.year, today.month, 1)]) * pct
        )
        metrics["3M %"] = _get_stats().comp(df[df.index >= m3]) * pct
        metrics["6M %"] = _get_stats().comp(df[df.index >= m6]) * pct
        metrics["YTD %"] = _get_stats().comp(df[df.index >= _dt(today.year, 1, 1)]) * pct
        metrics["1Y %"] = _get_stats().comp(df[df.index >= y1]) * pct
    else:
        metrics["MTD %"] = (
            _np.sum(df[df.index >= _dt(today.year, today.month, 1)], axis=0) * pct
        )
        metrics["3M %"] = _np.sum(df[df.index >= m3], axis=0) * pct
        metrics["6M %"] = _np.sum(df[df.index >= m6], axis=0) * pct
        metrics["YTD %"] = _np.sum(df[df.index >= _dt(today.year, 1, 1)], axis=0) * pct
        metrics["1Y %"] = _np.sum(df[df.index >= y1], axis=0) * pct

    # Multi-year annualized returns
    d = today - relativedelta(months=35)
    metrics["3Y (ann.) %"] = (
        _get_stats().cagr(df[df.index >= d], 0.0, compounded, win_year) * pct
    )

    d = today - relativedelta(months=59)
    metrics["5Y (ann.) %"] = (
        _get_stats().cagr(df[df.index >= d], 0.0, compounded, win_year) * pct
    )

    d = today - relativedelta(years=10)
    metrics["10Y (ann.) %"] = (
        _get_stats().cagr(df[df.index >= d], 0.0, compounded, win_year) * pct
    )

    metrics["All-time (ann.) %"] = _get_stats().cagr(df, 0.0, compounded, win_year) * pct

    # Best/worst period analysis (full mode only)
    # best/worst
    if mode.lower() == "full":
        metrics["~~~"] = blank
        metrics["Best Day %"] = (
            _get_stats().best(df, compounded=compounded, prepare_returns=False) * pct
        )
        metrics["Worst Day %"] = _get_stats().worst(df, prepare_returns=False) * pct
        metrics["Best Month %"] = (
            _get_stats().best(
                df, compounded=compounded, aggregate="ME", prepare_returns=False
            )
            * pct
        )
        metrics["Worst Month %"] = (
            _get_stats().worst(df, aggregate="ME", prepare_returns=False) * pct
        )
        metrics["Best Year %"] = (
            _get_stats().best(
                df, compounded=compounded, aggregate="YE", prepare_returns=False
            )
            * pct
        )
        metrics["Worst Year %"] = (
            _get_stats().worst(
                df, compounded=compounded, aggregate="YE", prepare_returns=False
            )
            * pct
        )

    # Calculate and integrate drawdown metrics
    # return drawdown (dd) df
    dd = _calc_dd(
        df,
        display=(display or "internal" in kwargs),
        as_pct=kwargs.get("as_pct", False),
    )

    # Add drawdown metrics to main metrics DataFrame
    # drawdown (dd) detail
    metrics["~~~~"] = blank
    # Properly integrate drawdown data into metrics
    for metric_name in dd.index:
        metrics[metric_name] = dd.loc[metric_name].values

    # Additional drawdown-based metrics
    metrics["Recovery Factor"] = _get_stats().recovery_factor(df)
    metrics["Ulcer Index"] = _get_stats().ulcer_index(df)
    metrics["Serenity Index"] = _get_stats().serenity_index(df, rf)

    # Win rate analysis (full mode only)
    # win rate
    if mode.lower() == "full":
        metrics["~~~~~"] = blank
        metrics["Avg. Up Month %"] = (
            _get_stats().avg_win(
                df, compounded=compounded, aggregate="ME", prepare_returns=False
            )
            * pct
        )
        metrics["Avg. Down Month %"] = (
            _get_stats().avg_loss(
                df, compounded=compounded, aggregate="ME", prepare_returns=False
            )
            * pct
        )
        metrics["Win Days %%"] = _get_stats().win_rate(df, prepare_returns=False) * pct
        metrics["Win Month %%"] = (
            _get_stats().win_rate(
                df, compounded=compounded, aggregate="ME", prepare_returns=False
            )
            * pct
        )
        metrics["Win Quarter %%"] = (
            _get_stats().win_rate(
                df, compounded=compounded, aggregate="QE", prepare_returns=False
            )
            * pct
        )
        metrics["Win Year %%"] = (
            _get_stats().win_rate(
                df, compounded=compounded, aggregate="YE", prepare_returns=False
            )
            * pct
        )

        # Greek letters and correlation analysis (if benchmark exists)
        if "benchmark" in df:
            metrics["~~~~~~~~~~~~"] = blank
            if isinstance(returns, _pd.Series):
                # Calculate Greek letters (Beta, Alpha) for single strategy
                greeks = _get_stats().greeks(
                    df["returns"], df["benchmark"], win_year, prepare_returns=False
                )
                metrics["Beta"] = [str(round(greeks["beta"], 2)), "-"]
                metrics["Alpha"] = [str(round(greeks["alpha"], 2)), "-"]
                metrics["Correlation"] = [
                    str(round(df["benchmark"].corr(df["returns"]) * pct, 2)) + "%",
                    "-",
                ]
                metrics["Treynor Ratio"] = [
                    str(
                        round(
                            _get_stats().treynor_ratio(
                                df["returns"], df["benchmark"], win_year, rf
                            )
                            * pct,
                            2,
                        )
                    )
                    + "%",
                    "-",
                ]
            elif isinstance(returns, _pd.DataFrame):
                # Calculate Greek letters for multiple strategies
                greeks = [
                    _get_stats().greeks(
                        df[strategy_col],
                        df["benchmark"],
                        win_year,
                        prepare_returns=False,
                    )
                    for strategy_col in df_strategy_columns
                ]
                metrics["Beta"] = [str(round(g["beta"], 2)) for g in greeks] + ["-"]
                metrics["Alpha"] = [str(round(g["alpha"], 2)) for g in greeks] + ["-"]
                metrics["Correlation"] = (
                    [
                        str(round(df["benchmark"].corr(df[strategy_col]) * pct, 2))
                        + "%"
                        for strategy_col in df_strategy_columns
                    ]
                ) + ["-"]
                metrics["Treynor Ratio"] = (
                    [
                        str(
                            round(
                                _get_stats().treynor_ratio(
                                    df[strategy_col], df["benchmark"], win_year, rf
                                )
                                * pct,
                                2,
                            )
                        )
                        + "%"
                        for strategy_col in df_strategy_columns
                    ]
                ) + ["-"]

    # Format metrics for display
    # prepare for display
    for col in metrics.columns:
        try:
            # Try to convert to float and round
            metrics[col] = metrics[col].astype(float).round(2)
            if display or "internal" in kwargs:
                metrics[col] = metrics[col].astype(str)
        except (ValueError, TypeError, AttributeError):
            pass
        # Handle integer columns (marked with *int)
        if (display or "internal" in kwargs) and "*int" in col:
            metrics[col] = metrics[col].str.replace(".0", "", regex=False)
            metrics.rename({col: col.replace("*int", "")}, axis=1, inplace=True)
        # Add percentage signs to percentage columns
        if (display or "internal" in kwargs) and "%" in col:
            metrics[col] = metrics[col] + "%"

    # Format drawdown days as integers
    try:
        metrics["Longest DD Days"] = _pd.to_numeric(metrics["Longest DD Days"]).astype(
            "int"
        )
        metrics["Avg. Drawdown Days"] = _pd.to_numeric(
            metrics["Avg. Drawdown Days"]
        ).astype("int")

        if display or "internal" in kwargs:
            metrics["Longest DD Days"] = metrics["Longest DD Days"].astype(str)
            metrics["Avg. Drawdown Days"] = metrics["Avg. Drawdown Days"].astype(str)
    except Exception:
        metrics["Longest DD Days"] = "-"
        metrics["Avg. Drawdown Days"] = "-"
        if display or "internal" in kwargs:
            metrics["Longest DD Days"] = "-"
            metrics["Avg. Drawdown Days"] = "-"

    # Clean up column names (remove separators and percentage signs)
    metrics.columns = [col if "~" not in col else "" for col in metrics.columns]
    metrics.columns = [col[:-1] if "%" in col else col for col in metrics.columns]
    metrics = metrics.T

    # Set appropriate column names
    if "benchmark" in df:
        column_names = [strategy_colname, benchmark_colname]
        if isinstance(strategy_colname, list):
            metrics.columns = list(_pd.core.common.flatten(column_names))
        else:
            metrics.columns = column_names
    else:
        if isinstance(strategy_colname, list):
            metrics.columns = strategy_colname
        else:
            metrics.columns = [strategy_colname]

    # Final data cleaning
    # cleanups
    metrics.replace([-0, "-0"], 0, inplace=True)
    metrics.replace(
        [
            _np.nan,
            -_np.nan,
            _np.inf,
            -_np.inf,
            "-nan%",
            "nan%",
            "-nan",
            "nan",
            "-inf%",
            "inf%",
            "-inf",
            "inf",
        ],
        "-",
        inplace=True,
    )

    # Reorder columns to put benchmark first if present
    # move benchmark to be the first column always if present
    if "benchmark" in df:
        metrics = metrics[
            [benchmark_colname]
            + [col for col in metrics.columns if col != benchmark_colname]
        ]

    # Handle display vs return
    if display:
        # Build and display parameters table (feature #472)
        params_data = {
            "Parameter": ["Risk-Free Rate", "Periods/Year", "Compounded", "Match Dates"],
            "Value": [
                f"{rf:.1%}" if rf != 0 else "0.0%",
                str(periods_per_year),
                "Yes" if compounded else "No",
                "Yes" if match_dates else "No",
            ],
        }
        if benchmark is not None:
            params_data["Parameter"].insert(0, "Benchmark")
            params_data["Value"].insert(0, benchmark_colname)
        params_df = _pd.DataFrame(params_data)
        print("\n" + _tabulate(params_df, headers="keys", tablefmt="simple", showindex=False))
        print("\n")
        print(_tabulate(metrics, headers="keys", tablefmt="simple"))
        return None

    # Remove separator rows if not requested
    if not sep:
        metrics = metrics[metrics.index != ""]

    # Final formatting for programmatic use
    # remove spaces from column names
    metrics = metrics.T
    metrics.columns = [
        c.replace(" %", "").replace(" *int", "").strip() for c in metrics.columns
    ]
    metrics = metrics.T

    return metrics


def plots(
    returns,
    benchmark=None,
    grayscale=False,
    figsize=(8, 5),
    mode="basic",
    compounded=True,
    periods_per_year=252,
    prepare_returns=True,
    match_dates=True,
    **kwargs,
):
    """
    Generate comprehensive visualization plots for portfolio performance.

    This function creates a complete set of performance visualization plots
    including returns, drawdowns, distributions, and rolling metrics. It can
    generate either basic plots or a full comprehensive suite.

    Parameters
    ----------
    returns : pd.Series or pd.DataFrame
        Daily returns data for the strategy/portfolio
    benchmark : pd.Series, str, or None, default None
        Benchmark returns for comparison
    grayscale : bool, default False
        Whether to generate charts in grayscale
    figsize : tuple, default (8, 5)
        Figure size for plots as (width, height)
    mode : str, default "basic"
        Plot mode - "basic" for essential plots, "full" for comprehensive suite
    compounded : bool, default True
        Whether to compound returns for calculations
    periods_per_year : int, default 252
        Number of trading periods per year
    prepare_returns : bool, default True
        Whether to prepare/clean returns data
    match_dates : bool, default True
        Whether to align returns and benchmark start dates
    **kwargs
        Additional keyword arguments:
        - strategy_title: Custom name for the strategy
        - benchmark_title: Custom name for the benchmark
        - active: Whether to show active returns vs benchmark

    Returns
    -------
    None
        Displays various performance plots

    Examples
    --------
    >>> plots(returns, benchmark='^GSPC', mode="full")
    >>> plots(returns, grayscale=True, figsize=(10, 6))
    """
    # Extract title parameters from kwargs
    benchmark_colname = kwargs.get("benchmark_title", "Benchmark")
    strategy_colname = kwargs.get("strategy_title", "Strategy")
    active = kwargs.get("active", False)

    # Handle multiple strategy columns
    if (
        isinstance(returns, _pd.DataFrame)
        and len(returns.columns) > 1
        and isinstance(strategy_colname, str)
    ):
        strategy_colname = list(returns.columns)

    # Get trading periods for rolling window calculations
    win_year, win_half_year = _get_trading_periods(periods_per_year)

    # Clean returns data if date matching is enabled
    if match_dates is True:
        returns = returns.dropna()

    # Prepare returns data if requested
    if prepare_returns:
        returns = _get_utils()._prepare_returns(returns)

    # Set names for display in plots
    if isinstance(returns, _pd.Series):
        returns.name = strategy_colname
    elif isinstance(returns, _pd.DataFrame):
        returns.columns = strategy_colname

    # Generate basic plots (snapshot and heatmap)
    if mode.lower() != "full":
        # Performance snapshot plot
        _get_plots().snapshot(
            returns,
            grayscale=grayscale,
            figsize=(figsize[0], figsize[0]),
            show=True,
            mode=("comp" if compounded else "sum"),
            benchmark_title=benchmark_colname,
            strategy_title=strategy_colname,
        )

        # Monthly returns heatmap
        if isinstance(returns, _pd.Series):
            _get_plots().monthly_heatmap(
                returns,
                benchmark,
                grayscale=grayscale,
                figsize=(figsize[0], figsize[0] * 0.5),
                show=True,
                ylabel="",
                compounded=compounded,
                active=active,
            )
        elif isinstance(returns, _pd.DataFrame):
            # Generate heatmap for each strategy column
            for col in returns.columns:
                _get_plots().monthly_heatmap(
                    returns[col].dropna(),
                    benchmark,
                    grayscale=grayscale,
                    figsize=(figsize[0], figsize[0] * 0.5),
                    show=True,
                    ylabel="",
                    returns_label=col,
                    compounded=compounded,
                    active=active,
                )

        return

    # prepare timeseries
    if benchmark is not None:
        benchmark = _get_utils()._prepare_benchmark(benchmark, returns.index)
        benchmark.name = benchmark_colname
        if match_dates is True:
            returns, benchmark = _match_dates(returns, benchmark)

    # Generate comprehensive plot suite
    # Cumulative returns plot
    _get_plots().returns(
        returns,
        benchmark,
        grayscale=grayscale,
        figsize=(figsize[0], figsize[0] * 0.6),
        show=True,
        ylabel="",
        prepare_returns=False,
        compound=compounded,
    )

    # Log returns plot for better visualization
    _get_plots().log_returns(
        returns,
        benchmark,
        grayscale=grayscale,
        figsize=(figsize[0], figsize[0] * 0.5),
        show=True,
        ylabel="",
        prepare_returns=False,
        compound=compounded,
    )

    # Volatility-matched returns (if benchmark exists)
    if benchmark is not None:
        _get_plots().returns(
            returns,
            benchmark,
            match_volatility=True,
            grayscale=grayscale,
            figsize=(figsize[0], figsize[0] * 0.5),
            show=True,
            ylabel="",
            prepare_returns=False,
            compound=compounded,
        )

    # Yearly returns comparison
    _get_plots().yearly_returns(
        returns,
        benchmark,
        grayscale=grayscale,
        figsize=(figsize[0], figsize[0] * 0.5),
        show=True,
        ylabel="",
        prepare_returns=False,
        compounded=compounded,
    )

    # Returns distribution histogram
    _get_plots().histogram(
        returns,
        benchmark,
        grayscale=grayscale,
        figsize=(figsize[0], figsize[0] * 0.5),
        show=True,
        ylabel="",
        prepare_returns=False,
        compounded=compounded,
    )

    # Calculate figure size for smaller plots
    small_fig_size = (figsize[0], figsize[0] * 0.35)
    if isinstance(returns, _pd.DataFrame) and len(returns.columns) > 1:
        small_fig_size = (
            figsize[0],
            figsize[0] * (0.33 * (len(returns.columns) * 0.66)),
        )

    # Daily returns scatter plot
    _get_plots().daily_returns(
        returns,
        benchmark,
        grayscale=grayscale,
        figsize=small_fig_size,
        show=True,
        ylabel="",
        prepare_returns=False,
        active=active,
    )

    # Rolling beta analysis (if benchmark exists)
    if benchmark is not None:
        _get_plots().rolling_beta(
            returns,
            benchmark,
            grayscale=grayscale,
            window1=win_half_year,
            window2=win_year,
            figsize=small_fig_size,
            show=True,
            ylabel="",
            prepare_returns=False,
        )

    # Rolling volatility analysis
    _get_plots().rolling_volatility(
        returns,
        benchmark,
        grayscale=grayscale,
        figsize=small_fig_size,
        show=True,
        ylabel="",
        period=win_half_year,
    )

    # Rolling Sharpe ratio analysis
    _get_plots().rolling_sharpe(
        returns,
        grayscale=grayscale,
        figsize=small_fig_size,
        show=True,
        ylabel="",
        period=win_half_year,
    )

    # Rolling Sortino ratio analysis
    _get_plots().rolling_sortino(
        returns,
        grayscale=grayscale,
        figsize=small_fig_size,
        show=True,
        ylabel="",
        period=win_half_year,
    )

    # Drawdown periods analysis
    if isinstance(returns, _pd.Series):
        _get_plots().drawdowns_periods(
            returns,
            grayscale=grayscale,
            figsize=(figsize[0], figsize[0] * 0.5),
            show=True,
            ylabel="",
            prepare_returns=False,
            compounded=compounded,
        )
    elif isinstance(returns, _pd.DataFrame):
        # Handle multiple strategy columns
        for col in returns.columns:
            _get_plots().drawdowns_periods(
                returns[col],
                grayscale=grayscale,
                figsize=(figsize[0], figsize[0] * 0.5),
                show=True,
                ylabel="",
                title=col,
                prepare_returns=False,
                compounded=compounded,
            )

    # Underwater (drawdown) plot
    _get_plots().drawdown(
        returns,
        grayscale=grayscale,
        figsize=(figsize[0], figsize[0] * 0.4),
        show=True,
        ylabel="",
        compound=compounded,
    )

    # Monthly returns heatmap
    if isinstance(returns, _pd.Series):
        _get_plots().monthly_heatmap(
            returns,
            benchmark,
            grayscale=grayscale,
            figsize=(figsize[0], figsize[0] * 0.5),
            returns_label=returns.name,
            show=True,
            ylabel="",
            compounded=compounded,
            active=active,
        )
    elif isinstance(returns, _pd.DataFrame):
        # Handle multiple strategy columns
        for col in returns.columns:
            _get_plots().monthly_heatmap(
                returns[col],
                benchmark,
                grayscale=grayscale,
                figsize=(figsize[0], figsize[0] * 0.5),
                show=True,
                ylabel="",
                returns_label=col,
                compounded=compounded,
                active=active,
            )

    # Returns distribution analysis
    if isinstance(returns, _pd.Series):
        _get_plots().distribution(
            returns,
            grayscale=grayscale,
            figsize=(figsize[0], figsize[0] * 0.5),
            show=True,
            title=returns.name,
            ylabel="",
            prepare_returns=False,
            compounded=compounded,
        )
    elif isinstance(returns, _pd.DataFrame):
        # Handle multiple strategy columns
        for col in returns.columns:
            _get_plots().distribution(
                returns[col],
                grayscale=grayscale,
                figsize=(figsize[0], figsize[0] * 0.5),
                show=True,
                title=col,
                ylabel="",
                prepare_returns=False,
                compounded=compounded,
            )


def _calc_dd(df, display=True, as_pct=False):
    """
    Calculate drawdown statistics for performance analysis.

    This helper function computes comprehensive drawdown statistics including
    maximum drawdown, drawdown dates, recovery periods, and average drawdown
    metrics. It handles both single strategy and multiple strategy analysis.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing returns data with columns for strategies
        and optionally benchmark
    display : bool, default True
        Whether the output is for display purposes (affects formatting)
    as_pct : bool, default False
        Whether to return percentages instead of decimals

    Returns
    -------
    pd.DataFrame
        DataFrame with drawdown statistics including:
        - Max Drawdown %: Maximum drawdown percentage
        - Max DD Date: Date of maximum drawdown
        - Max DD Period Start: Start date of worst drawdown period
        - Max DD Period End: End date of worst drawdown period
        - Longest DD Days: Duration of longest drawdown in days
        - Avg. Drawdown %: Average drawdown percentage
        - Avg. Drawdown Days: Average drawdown duration in days

    Examples
    --------
    >>> dd_stats = _calc_dd(returns_df, display=False)
    >>> dd_stats = _calc_dd(returns_df, as_pct=True)
    """
    # Convert returns to drawdown series
    dd = _get_stats().to_drawdown_series(df)
    dd_info = _get_stats().drawdown_details(dd)

    # Return empty DataFrame if no drawdowns found
    if dd_info.empty:
        return _pd.DataFrame()

    # Handle different column structures based on data type
    if "returns" in dd_info:
        ret_dd = dd_info["returns"]
    # to match multiple columns like returns_1, returns_2, ...
    elif (
        any(dd_info.columns.get_level_values(0).str.contains("returns"))
        and dd_info.columns.get_level_values(0).nunique() > 1
    ):
        ret_dd = dd_info.loc[
            :, dd_info.columns.get_level_values(0).str.contains("returns")
        ]
    else:
        ret_dd = dd_info

    # Calculate drawdown statistics based on data structure
    if (
        any(ret_dd.columns.get_level_values(0).str.contains("returns"))
        and ret_dd.columns.get_level_values(0).nunique() > 1
    ):
        # Multiple strategy columns case
        dd_stats = {
            col: {
                "Max Drawdown %": ret_dd[col]
                .sort_values(by="max drawdown", ascending=True)["max drawdown"]
                .values[0]
                / 100,
                "Max DD Date": ret_dd[col]
                .sort_values(by="max drawdown", ascending=True)["valley"]
                .values[0],
                "Max DD Period Start": ret_dd[col]
                .sort_values(by="max drawdown", ascending=True)["start"]
                .values[0],
                "Max DD Period End": ret_dd[col]
                .sort_values(by="max drawdown", ascending=True)["end"]
                .values[0],
                "Longest DD Days": str(
                    _np.round(
                        ret_dd[col]
                        .sort_values(by="days", ascending=False)["days"]
                        .values[0]
                    )
                ),
                "Avg. Drawdown %": ret_dd[col]["max drawdown"].mean() / 100,
                "Avg. Drawdown Days": str(_np.round(ret_dd[col]["days"].mean())),
            }
            for col in ret_dd.columns.get_level_values(0)
        }
    else:
        # Single strategy case. A single-column DataFrame input arrives here
        # with MultiIndex columns (level 0 = the strategy column); flatten it
        # so the per-field access below works on flat column labels.
        if isinstance(ret_dd.columns, _pd.MultiIndex):
            ret_dd = ret_dd.droplevel(0, axis=1)
        max_dd = ret_dd.sort_values(by="max drawdown", ascending=True)
        dd_stats = {
            "returns": {
                "Max Drawdown %": max_dd["max drawdown"].values[0] / 100,
                "Max DD Date": max_dd["valley"].values[0],
                "Max DD Period Start": max_dd["start"].values[0],
                "Max DD Period End": max_dd["end"].values[0],
                "Longest DD Days": str(
                    _np.round(
                        ret_dd.sort_values(by="days", ascending=False)["days"].values[0]
                    )
                ),
                "Avg. Drawdown %": ret_dd["max drawdown"].mean() / 100,
                "Avg. Drawdown Days": str(_np.round(ret_dd["days"].mean())),
            }
        }

    # Add benchmark drawdown statistics if present
    if "benchmark" in df and (dd_info.columns, _pd.MultiIndex):
        bench_dd = dd_info["benchmark"].sort_values(by="max drawdown")
        dd_stats["benchmark"] = {
            "Max Drawdown %": bench_dd.sort_values(by="max drawdown", ascending=True)[
                "max drawdown"
            ].values[0]
            / 100,
            "Max DD Date": bench_dd.sort_values(
                by="max drawdown", ascending=True
            )["valley"].values[0],
            "Max DD Period Start": bench_dd.sort_values(
                by="max drawdown", ascending=True
            )["start"].values[0],
            "Max DD Period End": bench_dd.sort_values(
                by="max drawdown", ascending=True
            )["end"].values[0],
            "Longest DD Days": str(
                _np.round(
                    bench_dd.sort_values(by="days", ascending=False)["days"].values[0]
                )
            ),
            "Avg. Drawdown %": bench_dd["max drawdown"].mean() / 100,
            "Avg. Drawdown Days": str(_np.round(bench_dd["days"].mean())),
        }

    # Apply percentage multiplier based on display settings
    # pct multiplier
    pct = 100 if display or as_pct else 1

    # Convert to DataFrame and apply percentage formatting
    dd_stats = _pd.DataFrame(dd_stats).T
    dd_stats["Max Drawdown %"] = dd_stats["Max Drawdown %"].astype(float) * pct
    dd_stats["Avg. Drawdown %"] = dd_stats["Avg. Drawdown %"].astype(float) * pct

    return dd_stats.T


def _html_table(obj, showindex="default"):
    """
    Convert DataFrame to HTML table format for report generation.

    This helper function converts pandas DataFrames to clean HTML table format
    suitable for embedding in HTML reports. It removes default tabulate styling
    and cleans up spacing for better presentation.

    Parameters
    ----------
    obj : pd.DataFrame
        DataFrame to convert to HTML table
    showindex : str or bool, default "default"
        Whether to show the DataFrame index in the HTML table.
        "default" uses tabulate's default behavior

    Returns
    -------
    str
        HTML string containing the formatted table

    Examples
    --------
    >>> html_str = _html_table(metrics_df)
    >>> html_str = _html_table(metrics_df, showindex=False)
    """
    # Convert DataFrame to HTML table using tabulate
    obj = _tabulate(
        obj, headers="keys", tablefmt="html", floatfmt=".2f", showindex=showindex
    )

    # Remove default tabulate styling attributes
    obj = obj.replace(' style="text-align: right;"', "")
    obj = obj.replace(' style="text-align: left;"', "")
    obj = obj.replace(' style="text-align: center;"', "")

    # Clean up spacing in table cells
    obj = _regex.sub("<td> +", "<td>", obj)
    obj = _regex.sub(" +</td>", "</td>", obj)
    obj = _regex.sub("<th> +", "<th>", obj)
    obj = _regex.sub(" +</th>", "</th>", obj)

    return obj


def _download_html(html, filename="quantstats-tearsheet.html"):
    """
    Generate JavaScript code to download HTML content in browser.

    This helper function creates JavaScript code that triggers a download
    of HTML content in the browser. It's used for downloading tearsheet
    reports directly from Jupyter notebooks.

    Parameters
    ----------
    html : str
        HTML content to be downloaded
    filename : str, default "quantstats-tearsheet.html"
        Filename for the downloaded file

    Returns
    -------
    None
        Displays JavaScript code in notebook to trigger download

    Examples
    --------
    >>> _download_html(html_content, "my_report.html")
    """
    # Create JavaScript code for file download
    jscode = _regex.sub(
        " +",
        " ",
        """<script>
    var bl=new Blob(['{{html}}'],{type:"text/html"});
    var a=document.createElement("a");
    a.href=URL.createObjectURL(bl);
    a.download="{{filename}}";
    a.hidden=true;document.body.appendChild(a);
    a.innerHTML="download report";
    a.click();</script>""".replace(
            "\n", ""
        ),
    )

    # Insert HTML content and clean up formatting
    jscode = jscode.replace("{{html}}", _regex.sub(" +", " ", html.replace("\n", "")))

    # Execute JavaScript in notebook if in notebook environment
    if _get_utils()._in_notebook():
        iDisplay(iHTML(jscode.replace("{{filename}}", filename)))


def _open_html(html):
    """
    Generate JavaScript code to open HTML content in new browser window.

    This helper function creates JavaScript code that opens HTML content
    in a new browser window. It's used for displaying tearsheet reports
    directly in the browser from Jupyter notebooks.

    Parameters
    ----------
    html : str
        HTML content to be displayed in new window

    Returns
    -------
    None
        Displays JavaScript code in notebook to open new window

    Examples
    --------
    >>> _open_html(html_content)
    """
    # Create JavaScript code to open new window with HTML content
    jscode = _regex.sub(
        " +",
        " ",
        """<script>
    var win=window.open();win.document.body.innerHTML='{{html}}';
    </script>""".replace(
            "\n", ""
        ),
    )

    # Insert HTML content and clean up formatting
    jscode = jscode.replace("{{html}}", _regex.sub(" +", " ", html.replace("\n", "")))

    # Execute JavaScript in notebook if in notebook environment
    if _get_utils()._in_notebook():
        iDisplay(iHTML(jscode))


def _embed_figure(figfiles, figfmt):
    """
    Embed matplotlib figures in HTML format for reports.

    This helper function converts matplotlib figure objects to embedded
    HTML format suitable for inclusion in HTML reports. It handles both
    SVG and base64-encoded image formats.

    Parameters
    ----------
    figfiles : io.StringIO or list of io.StringIO
        File-like objects containing figure data. Can be single figure
        or list of figures for multiple plots
    figfmt : str
        Format for the figures ('svg', 'png', 'jpg', etc.)

    Returns
    -------
    str
        HTML string with embedded figure(s) ready for inclusion in report

    Examples
    --------
    >>> embed_str = _embed_figure(figfile, 'svg')
    >>> embed_str = _embed_figure([fig1, fig2], 'png')
    """
    # Handle multiple figures
    if isinstance(figfiles, list):
        embed_string = "\n"
        for figfile in figfiles:
            figbytes = figfile.getvalue()
            if figfmt == "svg":
                # SVG can be embedded directly as text
                return figbytes.decode()
            # For other formats, encode as base64 data URI
            data_uri = _b64encode(figbytes).decode()
            embed_string.join(
                f'<img src="data:image/{figfmt};base64,{data_uri}" />'
            )
    else:
        # Handle single figure
        figbytes = figfiles.getvalue()
        if figfmt == "svg":
            # SVG can be embedded directly as text
            return figbytes.decode()
        # For other formats, encode as base64 data URI
        data_uri = _b64encode(figbytes).decode()
        embed_string = f'<img src="data:image/{figfmt};base64,{data_uri}" />'

    return embed_string
