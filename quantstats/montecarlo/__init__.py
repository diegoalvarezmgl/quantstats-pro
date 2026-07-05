"""
Multi-model Montecarlo engine for QuantStats Pro.

This package characterises a single asset's return series with several models
(GBM, jump-diffusion, GARCH, stochastic volatility, bootstraps, Bayesian, ...)
and simulates forward paths from each, enabling a cross-model comparison of the
distribution of outcomes (terminal value, CAGR, max drawdown, bust/goal probs).

The legacy shuffle-based Montecarlo (:mod:`quantstats._montecarlo`) remains the
default for ``qs.stats.montecarlo`` and is exposed here as the ``shuffle`` model.
"""

# Importing models registers them in the registry.
from . import (
    analytics,
    base,
    models,  # noqa: E402,F401  (side-effect: registration)
    registry,
)
from .base import SimulationModel
from .core import ModelResult, infer_periods_per_year, run_models
from .registry import available_models, get_model

__all__ = [
    "SimulationModel",
    "ModelResult",
    "run_models",
    "infer_periods_per_year",
    "get_model",
    "available_models",
    "analytics",
    "base",
    "registry",
]
