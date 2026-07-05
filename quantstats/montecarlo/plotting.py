"""
Plotting helpers for the Montecarlo tearsheet.

These produce matplotlib figures from :class:`ModelResult` objects: overlay fan
charts, drawdown distributions, CAGR quantile strips, and per-model detail
charts for the collapsible appendix.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg", force=False)
import matplotlib.pyplot as plt
import numpy as np

from . import analytics

_PRIMARY = "#348dc1"
_BAND = "#9bc4e2"
_ACCENT = "#003366"
_MEDIAN = "#c1431d"
_HIST = "#1a1a1a"
_PALETTE = [
    "#348dc1",
    "#c1431d",
    "#6B9E00",
    "#7b68ee",
    "#e67e22",
    "#16a085",
    "#8e44ad",
    "#2c3e50",
    "#d35400",
]


def _historical_band(historical_windows: np.ndarray):
    """Median and p5–p95 band of cumulative paths across rolling windows."""
    cum = analytics.cumulative_paths(historical_windows)
    median = np.median(cum, axis=1)
    lower = np.quantile(cum, 0.05, axis=1)
    upper = np.quantile(cum, 0.95, axis=1)
    return median, lower, upper


def fan_chart_figure(
    result,
    level: float = 0.95,
    n_paths: int = 100,
    figsize: tuple[float, float] = (8, 4),
    title: str | None = None,
    historical_windows: np.ndarray | None = None,
):
    """Fan chart: sample paths (faint), confidence band, median, optional history."""
    paths = analytics.cumulative_paths(result.sim_returns) * 100.0
    median, lower, upper = analytics.fan_chart(result.sim_returns, level=level)
    x = np.arange(paths.shape[0])

    fig, ax = plt.subplots(figsize=figsize)
    fig.set_facecolor("white")
    ax.set_facecolor("white")

    sample = min(n_paths, paths.shape[1])
    if sample > 0:
        step = max(1, paths.shape[1] // sample)
        for j in range(0, paths.shape[1], step):
            ax.plot(x, paths[:, j], color=_PRIMARY, alpha=0.05, lw=0.6)

    ax.fill_between(
        x,
        lower * 100.0,
        upper * 100.0,
        color=_BAND,
        alpha=0.5,
        label=f"{int(level * 100)}% band",
    )
    ax.plot(x, median * 100.0, color=_MEDIAN, lw=1.6, label="Median")

    if historical_windows is not None and historical_windows.size:
        h_med, _, _ = _historical_band(historical_windows)
        hx = np.arange(h_med.size)
        ax.plot(
            hx,
            h_med * 100.0,
            color=_HIST,
            lw=2.0,
            ls="--",
            label="Historical median (rolling)",
            zorder=5,
        )

    ax.axhline(0, color="gray", lw=0.8, ls="--")
    ax.set_title(title or result.label, fontweight="bold", color="black")
    ax.set_xlabel("Periods")
    ax.set_ylabel("Cumulative Return (%)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    fig.tight_layout()
    return fig


def terminal_hist_figure(
    result,
    figsize: tuple[float, float] = (8, 3),
    title: str | None = None,
    historical_terminal: float | None = None,
):
    """Histogram of terminal cumulative returns across simulations."""
    terminal = result.terminal_values() * 100.0
    fig, ax = plt.subplots(figsize=figsize)
    fig.set_facecolor("white")
    ax.set_facecolor("white")

    ax.hist(terminal, bins=40, color=_PRIMARY, alpha=0.75, edgecolor="white")
    ax.axvline(
        float(np.median(terminal)), color=_MEDIAN, lw=1.5, label="Median"
    )
    if historical_terminal is not None:
        ax.axvline(
            historical_terminal * 100.0,
            color=_HIST,
            lw=2.0,
            label="Historical",
        )
    ax.axvline(0, color="gray", lw=0.8, ls="--")

    ax.set_title(
        title or f"{result.label} - Terminal Return Distribution",
        fontweight="bold",
        color="black",
    )
    ax.set_xlabel("Terminal Cumulative Return (%)")
    ax.set_ylabel("Frequency")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    fig.tight_layout()
    return fig


def overlay_fan_figure(
    results: list,
    level: float = 0.95,
    figsize: tuple[float, float] = (10, 5),
    title: str = "Model Medians & Envelope",
    historical_windows: np.ndarray | None = None,
):
    """Overlay median paths per model with a pooled confidence envelope."""
    if not results:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No models", ha="center", va="center")
        return fig

    horizon = results[0].horizon
    x = np.arange(horizon)
    pooled = np.concatenate(
        [analytics.cumulative_paths(r.sim_returns) for r in results], axis=1
    )
    median_env = np.median(pooled, axis=1)
    alpha = (1.0 - level) / 2.0
    lower = np.quantile(pooled, alpha, axis=1)
    upper = np.quantile(pooled, 1.0 - alpha, axis=1)

    fig, ax = plt.subplots(figsize=figsize)
    fig.set_facecolor("white")
    ax.set_facecolor("white")

    ax.fill_between(
        x,
        lower * 100.0,
        upper * 100.0,
        color=_BAND,
        alpha=0.35,
        label=f"Pooled {int(level * 100)}% band",
    )
    ax.plot(
        x,
        median_env * 100.0,
        color=_ACCENT,
        lw=2.0,
        ls="--",
        label="Pooled median",
    )

    for i, res in enumerate(results):
        med, _, _ = analytics.fan_chart(res.sim_returns, level=level)
        color = _PALETTE[i % len(_PALETTE)]
        ax.plot(x, med * 100.0, color=color, lw=1.4, label=res.label)

    if historical_windows is not None and historical_windows.size:
        h_med, h_lo, h_hi = _historical_band(historical_windows)
        hx = np.arange(h_med.size)
        ax.fill_between(
            hx,
            h_lo * 100.0,
            h_hi * 100.0,
            color=_HIST,
            alpha=0.08,
            label="Historical 5–95% (rolling windows)",
            zorder=1,
        )
        ax.plot(
            hx,
            h_med * 100.0,
            color=_HIST,
            lw=2.2,
            ls="--",
            label="Historical median (rolling windows)",
            zorder=6,
        )

    ax.axhline(0, color="gray", lw=0.8, ls="--")
    ax.set_title(title, fontweight="bold", color="black")
    ax.set_xlabel("Periods")
    ax.set_ylabel("Cumulative Return (%)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper left", frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    return fig


def maxdd_distribution_figure(
    results: list,
    figsize: tuple[float, float] = (10, 5),
    title: str = "Max Drawdown Distribution by Model",
    historical_windows: np.ndarray | None = None,
):
    """Horizontal boxplot of max drawdown distributions (plus historical)."""
    if not results:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No models", ha="center", va="center")
        return fig

    data = [analytics.max_drawdowns(r.sim_returns) * 100.0 for r in results]
    labels = [r.label for r in results]
    n_hist = 0
    if historical_windows is not None and historical_windows.size:
        data.insert(0, analytics.max_drawdowns(historical_windows) * 100.0)
        labels.insert(0, "Historical (rolling)")
        n_hist = 1

    fig, ax = plt.subplots(figsize=figsize)
    fig.set_facecolor("white")
    ax.set_facecolor("white")

    bp = ax.boxplot(
        data,
        orientation="horizontal",
        tick_labels=labels,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": _MEDIAN, "linewidth": 1.5},
    )
    for i, patch in enumerate(bp["boxes"]):
        if i < n_hist:
            patch.set_facecolor(_HIST)
            patch.set_alpha(0.45)
        else:
            patch.set_facecolor(_PRIMARY)
            patch.set_alpha(0.55)

    ax.set_xlabel("Max Drawdown (%)")
    ax.set_title(title, fontweight="bold", color="black")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig


def cagr_quantiles_figure(
    results: list,
    figsize: tuple[float, float] = (10, 5),
    title: str = "CAGR Quantiles by Model",
    historical_windows: np.ndarray | None = None,
):
    """Horizontal range plot of CAGR p5–median–p95 per model (plus historical)."""
    if not results:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No models", ha="center", va="center")
        return fig

    labels = []
    p5s, medians, p95s = [], [], []
    hist_flags = []
    if historical_windows is not None and historical_windows.size:
        periods = results[0].periods
        h_cagr = analytics.cagr_values(historical_windows, periods) * 100.0
        labels.append("Historical (rolling)")
        p5s.append(float(np.quantile(h_cagr, 0.05)))
        medians.append(float(np.median(h_cagr)))
        p95s.append(float(np.quantile(h_cagr, 0.95)))
        hist_flags.append(True)
    for res in results:
        cagr = analytics.cagr_values(res.sim_returns, res.periods) * 100.0
        labels.append(res.label)
        p5s.append(float(np.quantile(cagr, 0.05)))
        medians.append(float(np.median(cagr)))
        p95s.append(float(np.quantile(cagr, 0.95)))
        hist_flags.append(False)

    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=figsize)
    fig.set_facecolor("white")
    ax.set_facecolor("white")

    for i, (lo, mid, hi, is_hist) in enumerate(
        zip(p5s, medians, p95s, hist_flags, strict=True)
    ):
        color = _HIST if is_hist else _PALETTE[i % len(_PALETTE)]
        ax.plot([lo, hi], [i, i], color=color, lw=2.0, solid_capstyle="round")
        ax.scatter([lo, hi], [i, i], color=color, s=28, zorder=3)
        ax.scatter([mid], [i], color=_MEDIAN, s=48, zorder=4, marker="D")

    ax.axvline(0, color="gray", lw=0.8, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("CAGR (%)")
    ax.set_title(title, fontweight="bold", color="black")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig
