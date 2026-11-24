"""Plotting helpers for the backtest-validity tearsheet."""

from __future__ import annotations

import math

import matplotlib

matplotlib.use("Agg", force=False)
import matplotlib.pyplot as plt
import numpy as np

from .core import ValidityResult

_PRIMARY = "#348dc1"
_CRITICAL = "#c62828"
_GOOD = "#2e7d32"
_MUTED = "#9e9e9e"
_OOS_FILL = "#fff3e0"


def _new_axes(figsize: tuple[float, float]):
    fig, ax = plt.subplots(figsize=figsize)
    fig.set_facecolor("white")
    ax.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return fig, ax


def _x(result: ValidityResult):
    return result.returns.index


def sharpe_distribution_figure(
    result: ValidityResult, figsize: tuple[float, float] = (8, 3.0)
):
    """Bootstrap distribution of the annualised Sharpe vs the selection hurdle."""
    fig, ax = _new_axes(figsize)
    samples = result.bootstrap["samples"]["sharpe"]
    samples = samples[np.isfinite(samples)]
    s = result.sharpe
    if len(samples) == 0:
        ax.text(0.5, 0.5, "Sharpe is undefined for this series", ha="center")
        ax.set_axis_off()
        return fig
    ax.hist(samples, bins=50, color=_PRIMARY, alpha=0.55, edgecolor="white")
    ci = result.bootstrap["sharpe"]
    ax.axvspan(ci["low"], ci["high"], color=_PRIMARY, alpha=0.08)
    ax.axvline(0, color="black", lw=1)
    ax.axvline(
        s["sharpe_annual"],
        color=_PRIMARY,
        lw=1.8,
        label=f"Observed {s['sharpe_annual']:.2f}",
    )
    if s["n_trials"] > 1:
        ax.axvline(
            s["sharpe_hurdle_annual"],
            color=_CRITICAL,
            lw=1.6,
            ls="--",
            label=f"Best of {s['n_trials']} skill-less trials "
            f"{s['sharpe_hurdle_annual']:.2f}",
        )
    ax.set_title(
        "Sharpe Ratio — Bootstrap Distribution", fontweight="bold", color="black", pad=8
    )
    ax.set_xlabel(
        f"Annualised Sharpe (shaded: {result.bootstrap['confidence']:.0%} interval)"
    )
    ax.set_ylabel("Replicates")
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    fig.tight_layout()
    return fig


def equity_figure(result: ValidityResult, figsize: tuple[float, float] = (8, 3.0)):
    """Cumulative return with the out-of-sample period shaded."""
    fig, ax = _new_axes(figsize)
    equity = (1.0 + result.returns).cumprod() - 1.0
    x = _x(result)
    ax.plot(x, equity.to_numpy() * 100.0, color=_PRIMARY, lw=1.4)
    ax.axhline(0, color="black", lw=0.8)
    if result.split is not None:
        pos = result.split["split_position"]
        ax.axvspan(x[pos], x[-1], color=_OOS_FILL, alpha=0.8, label="Out-of-sample")
        ax.axvline(x[pos], color=_CRITICAL, lw=1.2, ls="--")
        ax.legend(loc="upper left", frameon=False, fontsize=9)
    ax.set_title("Cumulative Return", fontweight="bold", color="black", pad=8)
    ax.set_ylabel("Cumulative return (%)")
    fig.tight_layout()
    return fig


def rolling_edge_figure(
    result: ValidityResult, figsize: tuple[float, float] = (8, 3.0)
):
    """Rolling annualised alpha (or mean return) with the fitted linear trend."""
    fig, ax = _new_axes(figsize)
    d = result.decay
    roll = d["rolling"]
    name = "Alpha" if d["kind"] == "alpha" else "Mean return"
    if len(roll) == 0:
        ax.text(0.5, 0.5, "Not enough data for a rolling estimate", ha="center")
        ax.set_axis_off()
        return fig
    ax.plot(
        roll.index,
        roll.to_numpy() * 100.0,
        color=_PRIMARY,
        lw=1.3,
        label=f"Rolling {d['window']}-period {name.lower()}",
    )
    ax.axhline(0, color="black", lw=0.8)

    # Trend line from the time-trend regression, centred on the sample mean
    n = d["n"]
    years = np.arange(n) / result.periods
    line = d["edge_annual"] + d["trend_annual_per_year"] * (years - years.mean())
    full_index = d["index"]
    if len(full_index) == n:
        significant = math.isfinite(d["trend_t_stat"]) and abs(d["trend_t_stat"]) > 1.96
        colour = (
            _MUTED
            if not significant
            else (_CRITICAL if d["trend_annual_per_year"] < 0 else _GOOD)
        )
        ax.plot(
            full_index,
            line * 100.0,
            color=colour,
            lw=1.6,
            ls="--",
            label=f"Trend {d['trend_annual_per_year']:+.1%}/yr "
            f"(t={d['trend_t_stat']:.2f})",
        )
    if result.split is not None:
        x = _x(result)
        ax.axvline(x[result.split["split_position"]], color=_CRITICAL, lw=1.0, ls=":")
    ax.set_title(f"{name} Over Time", fontweight="bold", color="black", pad=8)
    ax.set_ylabel(f"Annualised {name.lower()} (%)")
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    fig.tight_layout()
    return fig


def pbo_figure(result: ValidityResult, figsize: tuple[float, float] = (8, 3.0)):
    """Distribution of the out-of-sample rank logit of the in-sample winner."""
    fig, ax = _new_axes(figsize)
    over = result.overfitting
    if over is None:
        ax.set_axis_off()
        return fig
    logits = over["logits"]
    lim = max(abs(float(logits.min())), abs(float(logits.max())), 1.0)
    bins = np.linspace(-lim, lim, 41)
    ax.hist(logits[logits <= 0], bins=bins, color=_CRITICAL, alpha=0.6, edgecolor="white")
    ax.hist(logits[logits > 0], bins=bins, color=_PRIMARY, alpha=0.6, edgecolor="white")
    ax.axvline(0, color="black", lw=1)
    ax.set_title(
        f"Backtest Overfitting — PBO {over['pbo']:.0%}",
        fontweight="bold",
        color="black",
        pad=8,
    )
    ax.set_xlabel(
        "Out-of-sample rank of the in-sample winner (logit; ≤ 0 = bottom half)"
    )
    ax.set_ylabel("Train/test splits")
    fig.tight_layout()
    return fig
