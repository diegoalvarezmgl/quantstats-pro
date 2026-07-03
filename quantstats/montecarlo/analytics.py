"""
Analytics over simulated return paths.

All functions take ``sim_returns``: a 2-D array of shape ``(horizon, sims)`` of
simple periodic returns, and derive distributions of outcomes used by the
montecarlo tearsheet and the cross-model comparison table.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

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
        # 95th-percentile drawdown severity: exceeded by only ~5% of paths.
        "maxdd_median": float(np.median(mdd)),
        "maxdd_p95": float(np.quantile(mdd, 0.05)),
        "terminal_median": float(np.median(terminal)),
        "prob_loss": float(np.mean(terminal < 0)),
        "cvar_5": cvar(terminal, 0.05),
    }
    if bust is not None:
        summary["bust_prob"] = bust_probability(sim_returns, bust)
    if goal is not None:
        summary["goal_prob"] = goal_probability(sim_returns, goal)
    return summary


def conservative_envelope(
    results: dict[str, ModelResult],
    category: str = "montecarlo",
) -> dict[str, float]:
    """Worst-case metrics across models in a category (conservative envelope).

    Takes the minimum of upside metrics, maximum of downside probabilities,
    and the most severe drawdown tail across the selected models.
    """
    summaries = [
        res.summary for res in results.values() if res.category == category
    ]
    if not summaries:
        return {}

    envelope: dict[str, float] = {
        "cagr_p5": min(s["cagr_p5"] for s in summaries),
        "cagr_median": min(s["cagr_median"] for s in summaries),
        "cagr_p95": min(s["cagr_p95"] for s in summaries),
        "maxdd_median": min(s["maxdd_median"] for s in summaries),
        "maxdd_p95": min(s["maxdd_p95"] for s in summaries),
        "terminal_median": min(s["terminal_median"] for s in summaries),
        "prob_loss": max(s["prob_loss"] for s in summaries),
        "cvar_5": min(s["cvar_5"] for s in summaries),
    }
    if all("bust_prob" in s for s in summaries):
        envelope["bust_prob"] = max(s["bust_prob"] for s in summaries)
    if all("goal_prob" in s for s in summaries):
        envelope["goal_prob"] = min(s["goal_prob"] for s in summaries)
    return envelope


def evaluate_strategy(
    envelope: dict[str, float],
    max_acceptable_dd: float = -0.30,
    bust: float | None = None,
    bust_tolerance: float = 0.05,
) -> tuple[str, list[tuple[str, str]]]:
    """Return verdict (PASS/CAUTION/FAIL) and list of (level, message) checks."""
    if not envelope:
        return "FAIL", [("fail", "No Monte Carlo models available for assessment.")]

    checks: list[tuple[str, str]] = []

    if envelope["cagr_p5"] <= 0:
        checks.append(
            (
                "fail",
                "CAGR p5 is non-positive — realistic downside loses money.",
            )
        )
    elif envelope["cagr_p5"] < 0.02:
        checks.append(
            ("warn", "CAGR p5 is barely positive — limited upside in bad scenarios.")
        )

    if envelope["prob_loss"] >= 0.5:
        checks.append(
            (
                "fail",
                f"P(loss) = {envelope['prob_loss']:.0%} exceeds 50% threshold.",
            )
        )
    elif envelope["prob_loss"] >= 0.3:
        checks.append(
            ("warn", f"P(loss) = {envelope['prob_loss']:.0%} is elevated.")
        )

    if envelope["maxdd_p95"] < max_acceptable_dd:
        checks.append(
            (
                "fail",
                f"MaxDD p95 ({envelope['maxdd_p95']:.0%}) exceeds tolerance "
                f"({max_acceptable_dd:.0%}).",
            )
        )
    elif envelope["maxdd_p95"] < max_acceptable_dd * 1.1:
        checks.append(
            (
                "warn",
                f"MaxDD p95 ({envelope['maxdd_p95']:.0%}) is close to the "
                f"{max_acceptable_dd:.0%} sizing limit.",
            )
        )

    if bust is not None and "bust_prob" in envelope:
        if envelope["bust_prob"] > bust_tolerance:
            checks.append(
                (
                    "fail",
                    f"P(bust) = {envelope['bust_prob']:.0%} exceeds "
                    f"{bust_tolerance:.0%} tolerance.",
                )
            )
        elif envelope["bust_prob"] > bust_tolerance * 0.5:
            checks.append(
                (
                    "warn",
                    f"P(bust) = {envelope['bust_prob']:.0%} is non-trivial.",
                )
            )

    if any(level == "fail" for level, _ in checks):
        verdict = "FAIL"
    elif any(level == "warn" for level, _ in checks):
        verdict = "CAUTION"
    else:
        verdict = "PASS"

    return verdict, checks
