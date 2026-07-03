"""
Plotting helpers for the alpha-decay tearsheet.

Rolling-metric histograms use the analysis (transformed) scale — same as the
z-score — with axis labels naming the transform.  Time-underwater uses raw
days on the chart (easier to read); only its z-score is computed on log(days).
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg", force=False)
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as sp_stats

from .core import (
    CusumResult,
    MetricSpec,
    TimeUnderwaterResult,
    WindowResult,
    format_value,
    transform_axis_label,
)

_EXCELLENT = "#2e7d32"
_GOOD = "#689f38"
_WARNING = "#f9a825"
_CRITICAL = "#c62828"
_PRIMARY = "#348dc1"
_MEDIAN = "#c1431d"
_ACCENT = "#003366"

_STATUS_COLORS = {
    "excellent": _EXCELLENT,
    "good": _GOOD,
    "warning": _WARNING,
    "critical": _CRITICAL,
}


def _underwater_hist_bounds(
    dist: np.ndarray,
    mean: float,
    current_days: int,
    is_underwater: bool,
    *,
    percentile: float = 90.0,
) -> tuple[float, np.ndarray, int, float]:
    """
    Pick a readable histogram x-range for completed spell lengths.

    Completed spells are heavily right-skewed: a single 400-day episode
    would stretch the axis to 400 even when mean ≈ 13 and current ≈ 12.
    We cap the axis at roughly the p90 bulk, always including mean and the
    current spell, and return outlier counts for an annotation.
    """
    markers = max(mean, float(current_days) if is_underwater else 0.0)
    if len(dist) == 0:
        return max(markers * 1.25, 10.0), dist, 0, 0.0

    p = float(np.percentile(dist, percentile))
    upper = max(markers * 1.25, p * 1.05, 12.0)
    mx = float(dist.max())
    if mx <= upper:
        return upper + 1, dist, 0, mx

    in_range = dist[dist <= upper]
    beyond = dist[dist > upper]
    return upper, in_range, int(len(beyond)), float(beyond.max())


def metric_distribution_figure(
    result: WindowResult,
    spec: MetricSpec,
    figsize: tuple[float, float] = (4.2, 3.0),
    title: str | None = None,
):
    """Histogram + KDE on the analysis (transformed) scale with raw-unit legend."""
    dist = result.analysis_distribution
    fig, ax = plt.subplots(figsize=figsize)
    fig.set_facecolor("white")
    ax.set_facecolor("white")

    if len(dist) == 0:
        ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center")
        ax.set_axis_off()
        fig.tight_layout()
        return fig

    color = _STATUS_COLORS.get(result.status, _PRIMARY)
    ax.hist(dist, bins=min(30, max(10, len(dist) // 8)), color=_PRIMARY, alpha=0.55, edgecolor="white")

    if len(dist) > 5 and np.std(dist) > 0:
        xs = np.linspace(dist.min(), dist.max(), 200)
        try:
            kde = sp_stats.gaussian_kde(dist)
            ax.plot(xs, kde(xs) * len(dist) * (dist.max() - dist.min()) / 30, color=_ACCENT, lw=1.2, alpha=0.8)
        except Exception:
            pass

    ax.axvline(
        result.analysis_mean,
        color=_MEDIAN,
        lw=1.4,
        ls="--",
        label=f"Mean ({format_value(spec, result.mean)})",
    )
    ax.axvline(
        result.analysis_observed,
        color=color,
        lw=2.0,
        label=f"Current ({format_value(spec, result.observed)})",
    )

    ttl = title or f"{spec.label} — {result.window}d"
    ax.set_title(ttl, fontweight="bold", color="black", fontsize=10, pad=8)
    ax.set_xlabel(transform_axis_label(spec), fontsize=9)
    ax.set_ylabel("Frequency", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper right", frameon=False, fontsize=7)
    fig.tight_layout()
    return fig


def cusum_figure(
    cusum: CusumResult,
    figsize: tuple[float, float] = (8, 2.8),
    title: str = "CUSUM — Return Decay Detector",
):
    """CUSUM control chart with alarm threshold."""
    fig, ax = plt.subplots(figsize=figsize)
    fig.set_facecolor("white")
    ax.set_facecolor("white")

    x = np.arange(len(cusum.series))
    ax.plot(x, cusum.series.to_numpy(), color=_PRIMARY, lw=1.4, label="CUSUM")
    ax.axhline(cusum.threshold, color=_CRITICAL, lw=1.2, ls="--", label=f"Threshold ({cusum.threshold:.4f})")
    ax.fill_between(x, 0, cusum.threshold, color="#ffebee", alpha=0.3)

    status = "ALARM" if cusum.alarm else f"{cusum.pct_of_threshold:.0%} of threshold"
    ax.set_title(f"{title} — {status}", fontweight="bold", color="black", pad=8)
    ax.set_xlabel("Trading day (0 = start of sample)")
    ax.set_ylabel("Cumulative sum of shortfalls")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    fig.tight_layout()
    return fig


def time_underwater_figure(
    tuw: TimeUnderwaterResult,
    figsize: tuple[float, float] = (8, 2.8),
    title: str = "Time Underwater — Completed Spell Lengths",
):
    """Histogram of completed underwater spells (days); current spell marked separately."""
    fig, ax = plt.subplots(figsize=figsize)
    fig.set_facecolor("white")
    ax.set_facecolor("white")

    dist = tuw.distribution
    if len(dist) == 0:
        ax.text(0.5, 0.5, "No completed drawdown spells yet", ha="center", va="center")
        ax.set_axis_off()
        fig.tight_layout()
        return fig

    color = _STATUS_COLORS.get(tuw.status, _PRIMARY)
    upper, hist_dist, n_beyond, longest_beyond = _underwater_hist_bounds(
        dist, tuw.mean, tuw.current_days, tuw.is_underwater
    )
    width = max(1, int(np.ceil(upper / 30.0)))
    edges = np.arange(0.5, upper + width + 0.5, width)
    ax.hist(hist_dist, bins=edges, color=_PRIMARY, alpha=0.55, edgecolor="white")
    ax.set_xlim(0, upper * 1.03 + 1)
    if n_beyond > 0:
        ax.text(
            0.98,
            0.97,
            f"+{n_beyond} spell{'s' if n_beyond != 1 else ''} > {upper:.0f}d"
            f"\n(longest {longest_beyond:.0f}d)",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=7,
            color="#666",
        )
    ax.axvline(tuw.mean, color=_MEDIAN, lw=1.4, ls="--", label=f"Mean ({tuw.mean:.0f}d)")
    if tuw.is_underwater:
        ax.axvline(
            tuw.current_days,
            color=color,
            lw=2.0,
            label=f"Current spell ({tuw.current_days}d)",
        )
    ax.set_title(title, fontweight="bold", color="black", pad=8)
    ax.set_xlabel("Completed underwater spell (days)")
    ax.set_ylabel("Frequency")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    fig.tight_layout()
    return fig
