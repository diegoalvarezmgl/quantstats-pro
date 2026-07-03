"""
Analytics over simulated return paths.

All functions take ``sim_returns``: a 2-D array of shape ``(horizon, sims)`` of
simple periodic returns, and derive distributions of outcomes used by the
montecarlo tearsheet and the cross-model comparison table.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from .core import ModelResult


def cumulative_paths(sim_returns: np.ndarray) -> np.ndarray:
    """Cumulative simple return per path: ``prod(1+r) - 1`` along time.

    Returns an array of shape ``(horizon, sims)`` where each column is the
    running cumulative return of that path.
    """
    return np.cumprod(1.0 + sim_returns, axis=0) - 1.0


def terminal_values(sim_returns: np.ndarray) -> np.ndarray:
    """Terminal cumulative simple return for each path (1-D, length ``sims``)."""
    return np.prod(1.0 + sim_returns, axis=0) - 1.0


def max_drawdowns(sim_returns: np.ndarray) -> np.ndarray:
    """Maximum drawdown (negative) of each path (1-D, length ``sims``)."""
    growth = np.cumprod(1.0 + sim_returns, axis=0)
    running_max = np.maximum.accumulate(growth, axis=0)
    drawdowns = growth / running_max - 1.0
    return drawdowns.min(axis=0)


def cagr_values(sim_returns: np.ndarray, periods: float = 252.0) -> np.ndarray:
    """Annualised compound growth rate for each path (1-D, length ``sims``)."""
    horizon = sim_returns.shape[0]
    years = horizon / periods
    terminal = terminal_values(sim_returns)
    # Guard against terminal <= -1 (total wipeout) which would be invalid.
    safe = np.clip(terminal, -0.999999, None)
    return (1.0 + safe) ** (1.0 / years) - 1.0


def cvar(values: np.ndarray, q: float = 0.05) -> float:
    """Expected shortfall (CVaR) of the left tail at quantile ``q``."""
    threshold = float(np.quantile(values, q))
    tail = values[values <= threshold]
    return float(np.mean(tail)) if tail.size > 0 else threshold


def bust_probability(sim_returns: np.ndarray, bust: float) -> float:
    """Fraction of paths whose max drawdown is at or beyond ``bust`` (negative)."""
    mdd = max_drawdowns(sim_returns)
    return float(np.mean(mdd <= bust))


def goal_probability(sim_returns: np.ndarray, goal: float) -> float:
    """Fraction of paths whose terminal cumulative return reaches ``goal``."""
    terminal = terminal_values(sim_returns)
    return float(np.mean(terminal >= goal))


def fan_chart(
    sim_returns: np.ndarray, level: float = 0.95
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Median path and a symmetric confidence band of cumulative returns.

    Returns ``(median, lower, upper)``, each of length ``horizon``.
    """
    paths = cumulative_paths(sim_returns)
    alpha = (1.0 - level) / 2.0
    median = np.median(paths, axis=1)
    lower = np.quantile(paths, alpha, axis=1)
    upper = np.quantile(paths, 1.0 - alpha, axis=1)
    return median, lower, upper


def historical_windows(returns: pd.Series, horizon: int) -> np.ndarray:
    """All overlapping ``horizon``-length return windows from full history.

    Returns an array of shape ``(horizon, n_windows)`` — the same layout as
    ``sim_returns`` — so the empirical distribution of *realised* outcomes over
    the entire history can be compared directly against simulated paths. If the
    history is shorter than ``horizon``, a single truncated window is returned.
    """
    arr = pd.Series(returns).dropna().to_numpy(dtype=float)
    if arr.size == 0:
        return np.empty((0, 0), dtype=float)
    if arr.size <= horizon:
        return arr.reshape(-1, 1)
    windows = np.lib.stride_tricks.sliding_window_view(arr, horizon)
    return np.ascontiguousarray(windows.T)


def historical_window(returns: pd.Series, horizon: int) -> np.ndarray:
    """Trailing ``horizon`` simple returns from history (1-D)."""
    arr = pd.Series(returns).dropna().to_numpy(dtype=float)
    if arr.size == 0:
        return np.array([], dtype=float)
    return arr[-min(horizon, arr.size) :]


def historical_cumulative(returns: pd.Series, horizon: int) -> np.ndarray:
    """Cumulative return path of the trailing historical window."""
    window = historical_window(returns, horizon)
    if window.size == 0:
        return np.array([], dtype=float)
    return cumulative_paths(window.reshape(-1, 1))[:, 0]


def historical_summary(
    returns: pd.Series,
    horizon: int,
    periods: float = 252.0,
    bust: float | None = None,
    goal: float | None = None,
) -> dict[str, float]:
    """Empirical distribution of realised metrics over all rolling windows.

    Uses every overlapping ``horizon``-length window of the full history, so
    quantiles (CAGR p5/median/p95, MaxDD, CVaR, probabilities) describe what
    actually happened across the whole sample — directly comparable with each
    model's simulated distribution.
    """
    windows = historical_windows(returns, horizon)
    if windows.size == 0:
        return {}
    return summarize(windows, periods=periods, bust=bust, goal=goal)


def realism_percentile(sim_returns: np.ndarray, historical_terminal: float) -> float:
    """Percentile rank of the historical terminal within simulated terminals."""
    terminal = terminal_values(sim_returns)
    if terminal.size == 0:
        return float("nan")
    return float(np.mean(terminal <= historical_terminal) * 100.0)


def summarize(
    sim_returns: np.ndarray,
    periods: float = 252.0,
    bust: float | None = None,
    goal: float | None = None,
) -> dict[str, float]:
    """Cross-model comparison row for a set of simulated paths."""
    terminal = terminal_values(sim_returns)
    mdd = max_drawdowns(sim_returns)
    cagr = cagr_values(sim_returns, periods)

    summary: dict[str, float] = {
        "cagr_p5": float(np.quantile(cagr, 0.05)),
        "cagr_median": float(np.median(cagr)),
        "cagr_p95": float(np.quantile(cagr, 0.95)),
        # 5th percentile of max drawdown: severity exceeded by ~5% of paths.
        "maxdd_median": float(np.median(mdd)),
        "maxdd_p95": float(np.quantile(mdd, 0.05)),
        "terminal_median": float(np.median(terminal)),
        "prob_loss": float(np.mean(terminal < 0)),
        # CVaR on horizon terminal return (at 1y horizon = annual return tail).
        "cvar_5": cvar(terminal, 0.05),
    }
    if bust is not None:
        summary["bust_prob"] = bust_probability(sim_returns, bust)
    if goal is not None:
        summary["goal_prob"] = goal_probability(sim_returns, goal)
    return summary


def model_median_summary(
    results: dict[str, ModelResult],
    category: str = "montecarlo",
) -> dict[str, float]:
    """Median metrics across models in a category (robust consensus view)."""
    summaries = [
        res.summary for res in results.values() if res.category == category
    ]
    if not summaries:
        return {}

    keys = (
        "cagr_p5",
        "cagr_median",
        "cagr_p95",
        "maxdd_median",
        "maxdd_p95",
        "terminal_median",
        "prob_loss",
        "cvar_5",
        "bust_prob",
        "goal_prob",
    )
    median: dict[str, float] = {}
    for key in keys:
        vals = [s[key] for s in summaries if key in s]
        if vals:
            median[key] = float(np.median(vals))
    return median


def conservative_envelope(
    results: dict[str, ModelResult],
    category: str = "montecarlo",
) -> dict[str, float]:
    """Worst-case metrics across models in a category (conservative envelope).

    Takes the minimum of upside metrics, maximum of downside probabilities,
    and the most severe drawdown tail across the selected models.
    """
    summaries = [
        (name, res.summary)
        for name, res in results.items()
        if res.category == category
    ]
    if not summaries:
        return {}

    keys_min = (
        "cagr_p5",
        "cagr_median",
        "cagr_p95",
        "maxdd_median",
        "maxdd_p95",
        "terminal_median",
        "cvar_5",
    )
    keys_max = ("prob_loss",)

    envelope: dict[str, float] = {}
    for key in keys_min:
        envelope[key] = min(s[1][key] for s in summaries)
    for key in keys_max:
        envelope[key] = max(s[1][key] for s in summaries)

    if all("bust_prob" in s[1] for s in summaries):
        envelope["bust_prob"] = max(s[1]["bust_prob"] for s in summaries)
    if all("goal_prob" in s[1] for s in summaries):
        envelope["goal_prob"] = min(s[1]["goal_prob"] for s in summaries)
    return envelope


def envelope_attribution(
    results: dict[str, ModelResult],
    envelope: dict[str, float],
    category: str = "montecarlo",
) -> dict[str, str]:
    """Map each envelope metric to the model label that produced the extreme."""
    summaries = [
        (res.label, res.summary)
        for res in results.values()
        if res.category == category
    ]
    if not summaries or not envelope:
        return {}

    keys_min = (
        "cagr_p5",
        "cagr_median",
        "cagr_p95",
        "maxdd_median",
        "maxdd_p95",
        "terminal_median",
        "cvar_5",
        "goal_prob",
    )
    keys_max = ("prob_loss", "bust_prob")

    attr: dict[str, str] = {}
    for key in keys_min:
        if key not in envelope:
            continue
        if key == "goal_prob":
            label = min(summaries, key=lambda s: s[1].get(key, float("inf")))[0]
        else:
            label = min(summaries, key=lambda s: s[1].get(key, float("inf")))[0]
        attr[key] = label

    for key in keys_max:
        if key not in envelope:
            continue
        label = max(summaries, key=lambda s: s[1].get(key, float("-inf")))[0]
        attr[key] = label

    return attr
