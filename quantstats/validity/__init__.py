"""
Backtest validity for QuantStats Pro.

Tools for judging whether a backtest's performance is real: deflated and
probabilistic Sharpe ratios, minimum track record length, bootstrap confidence
intervals, in-sample vs out-of-sample comparison, alpha-decay trend tests and
the probability of backtest overfitting.

Start with :func:`validate`.
"""

from .core import (
    Finding,
    ValidityResult,
    bootstrap_confidence_intervals,
    decay_analysis,
    deflated_sharpe_ratio,
    expected_max_sharpe,
    infer_periods_per_year,
    min_track_record_length,
    probabilistic_sharpe_ratio,
    probability_of_backtest_overfitting,
    selection_adjusted_t_threshold,
    sharpe_standard_error,
    split_analysis,
    validate,
)

__all__ = [
    "Finding",
    "ValidityResult",
    "bootstrap_confidence_intervals",
    "decay_analysis",
    "deflated_sharpe_ratio",
    "expected_max_sharpe",
    "infer_periods_per_year",
    "min_track_record_length",
    "probabilistic_sharpe_ratio",
    "probability_of_backtest_overfitting",
    "selection_adjusted_t_threshold",
    "sharpe_standard_error",
    "split_analysis",
    "validate",
]
