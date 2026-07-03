"""
Trimmed bootstrap models.

Removes the *best* returns from the sample before bootstrapping, producing a
deliberately pessimistic view of outcomes (it strips the luckiest upside while
keeping the downside intact). Only the top 1% variant is registered by default.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..base import SimulationModel
from ..registry import register


class _TrimmedBase(SimulationModel):
    category = "stress"
    trim_fraction: float = 0.01

    def calibrate(self, returns: pd.Series) -> _TrimmedBase:
        arr = self._clean(returns)
        if arr.size == 0:
            raise ValueError("Trimmed model needs at least one return.")
        cutoff = np.quantile(arr, 1.0 - self.trim_fraction)
        trimmed = arr[arr <= cutoff]
        self.returns_ = trimmed if trimmed.size > 0 else arr
        self._fitted = True
        return self

    def simulate(
        self, horizon: int, sims: int, rng: np.random.Generator
    ) -> np.ndarray:
        self._check_fitted()
        idx = rng.integers(0, self.returns_.size, size=(horizon, sims))
        return self.returns_[idx]

    def calibration_summary(self, periods: float = 252.0) -> dict[str, str]:
        return {
            "Trim fraction (top)": f"{self.trim_fraction:.1%}",
            "Sample size (after trim)": f"{self.returns_.size:,}",
        }


@register
class TrimmedTop1(_TrimmedBase):
    name = "trimmed_1pct"
    label = "Trimmed (top 1%)"
    trim_fraction = 0.01
