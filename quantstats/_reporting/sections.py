"""
Reusable report sections.

Each function returns an HTML fragment (string) ready to be injected into a
template by :func:`quantstats._reporting.engine.render_report`. The sections
reuse the low-level helpers already living in :mod:`quantstats.reports`
(``_html_table``, ``_embed_figure``, ``metrics``) so that the new tearsheets
stay visually consistent with the legacy one without duplicating logic.
"""

from __future__ import annotations

import re as _re

import pandas as _pd

# Inline-styled cell used to turn a blank spacer row into a subtle gray rule.
_SEP_CELL = '<td style="border-bottom:1px solid #cfcfcf;padding:0;height:1px"></td>'
_ROW_RE = _re.compile(r"<tr>.*?</tr>", _re.S)
_CELL_RE = _re.compile(r"<td[^>]*>(.*?)</td>", _re.S)


def _style_separator_rows(html: str) -> str:
    """Render blank spacer rows (all-empty cells) as a subtle gray separator."""

    def _repl(match: _re.Match) -> str:
        row = match.group(0)
        cells = _CELL_RE.findall(row)
        if cells and all(c.strip() == "" for c in cells):
            return "<tr>" + _SEP_CELL * len(cells) + "</tr>"
        return row

    return _ROW_RE.sub(_repl, html)


def _reports():
    from .. import reports

    return reports


def _utils():
    from .. import utils

    return utils


def _stats():
    from .. import stats

    return stats


def _plots():
    from .. import plots

    return plots


def _normalize_label(label: str) -> str:
    """Normalize a metric label for robust matching (drop ``%`` and spaces)."""
    return str(label).replace("%", "").strip().lower()


def metrics_table(
    returns,
    benchmark=None,
    rf: float = 0.0,
    compounded: bool = True,
    periods_per_year: int = 252,
    drop_rows: list[str] | None = None,
    collapse_separators: bool = False,
    style_separators: bool = False,
    strategy_title="Strategy",
    benchmark_title=None,
    prepare_returns: bool = True,
) -> str:
    """
    Render the key-performance-metrics table as HTML.

    Mirrors the legacy ``reports.html`` metrics table, with the ability to drop
    specific rows by (normalized) label. When ``collapse_separators`` is True,
    runs of consecutive blank spacer rows (empty index) are collapsed to a
    single one and leading/trailing blanks are removed -- this keeps the visual
    grouping while avoiding the large gaps left after whole metric groups are
    dropped. When ``style_separators`` is True, the remaining blank spacer rows
    are rendered as a subtle gray horizontal rule instead of empty whitespace.
    """
    reports = _reports()
    mtrx = reports.metrics(
        returns=returns,
        benchmark=benchmark,
        rf=rf,
        display=False,
        mode="full",
        sep=True,
        internal="True",
        compounded=compounded,
        periods_per_year=periods_per_year,
        prepare_returns=prepare_returns,
        benchmark_title=benchmark_title,
        strategy_title=strategy_title,
    )[2:]

    if drop_rows:
        drop_norm = {_normalize_label(d) for d in drop_rows}
        keep = [idx for idx in mtrx.index if _normalize_label(idx) not in drop_norm]
        mtrx = mtrx.loc[keep]

    if collapse_separators:
        is_blank = [str(idx).strip() == "" for idx in mtrx.index]
        keep_mask = []
        prev_blank = True  # drop leading blanks
        for blank in is_blank:
            keep_mask.append(not (blank and prev_blank))
            prev_blank = blank
        # Drop trailing blanks too.
        for i in range(len(keep_mask) - 1, -1, -1):
            if is_blank[i] and keep_mask[i]:
                keep_mask[i] = False
            elif keep_mask[i]:
                break
        mtrx = mtrx[keep_mask]

    mtrx.index.name = "Metric"
    html = reports._html_table(mtrx)
    if style_separators:
        html = _style_separator_rows(html)
    return html


def eoy_table(
    returns,
    benchmark=None,
    benchmark_original=None,
    compounded: bool = True,
    strategy_title="Strategy",
    benchmark_title=None,
) -> str:
    """Render the end-of-year returns table as HTML."""
    reports = _reports()
    utils = _utils()
    stats = _stats()

    if benchmark is not None:
        benchmark_for_eoy = (
            benchmark_original if benchmark_original is not None else benchmark
        )
        yoy = stats.compare(
            returns, benchmark_for_eoy, "YE", compounded=compounded, prepare_returns=False
        )
        if isinstance(returns, _pd.Series):
            yoy.columns = [benchmark_title, strategy_title, "Multiplier", "Won"]
        yoy.index.name = "Year"
        return reports._html_table(yoy)

    yoy = _pd.DataFrame(utils.group_returns(returns, returns.index.year) * 100)
    if isinstance(returns, _pd.Series):
        yoy.columns = ["Return"]
        yoy["Cumulative"] = utils.group_returns(returns, returns.index.year, True) * 100
    yoy.index.name = "Year"
    return reports._html_table(yoy)


def dd_table(returns, top: int = 5) -> str:
    """Render the worst-N drawdowns table as HTML (single Series)."""
    reports = _reports()
    stats = _stats()

    dd = stats.to_drawdown_series(returns)
    dd_info = stats.drawdown_details(dd).sort_values(by="max drawdown", ascending=True)[
        :top
    ]
    dd_info = dd_info[["start", "end", "max drawdown", "days"]]
    dd_info.columns = ["Started", "Recovered", "Drawdown", "Days"]
    return reports._html_table(dd_info, False)


def plot_section(
    plot_name: str,
    returns,
    benchmark=None,
    figfmt: str = "svg",
    **plot_kwargs,
) -> str:
    """
    Generate a single plot and return it as an embedded HTML fragment.

    Parameters
    ----------
    plot_name : str
        Name of the plotting function in :mod:`quantstats.plots`.
    returns : pd.Series or pd.DataFrame
        Returns data passed as the first positional argument.
    benchmark : optional
        Passed as second positional argument when not None; some plot
        functions (e.g. rolling_sharpe) do not accept a benchmark, so pass
        ``benchmark=None`` and they will be called with returns only.
    figfmt : str
        Figure format for embedding (``svg`` by default).
    **plot_kwargs
        Extra keyword arguments forwarded to the plotting function.
    """
    reports = _reports()
    utils = _utils()
    plots = _plots()

    plot_fn = getattr(plots, plot_name)
    figfile = utils._file_stream()

    kwargs = dict(
        grayscale=plot_kwargs.pop("grayscale", False),
        savefig={"fname": figfile, "format": figfmt},
        show=False,
    )
    kwargs.update(plot_kwargs)

    if benchmark is not None:
        plot_fn(returns, benchmark, **kwargs)
    else:
        plot_fn(returns, **kwargs)

    return reports._embed_figure(figfile, figfmt)
