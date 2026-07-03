"""
Alpha decay monitoring for QuantStats Pro.

Rolling-window risk metrics, historical distributions on an analysis scale
(log / log-severity transforms), z-score traffic lights, CUSUM change
detection, and time-underwater analysis for equity curves.
"""

from . import plotting
from .core import (
    DecayResult,
    MetricResult,
    WindowResult,
    analyze,
    available_metrics,
)

__all__ = [
    "DecayResult",
    "MetricResult",
    "WindowResult",
    "analyze",
    "available_metrics",
    "plotting",
]
