"""
Trimmed bootstrap models.

Removes the *best* returns from the sample before bootstrapping, producing a
deliberately pessimistic view of outcomes (it strips the luckiest upside while
keeping the downside intact). Two variants are registered: trimming the top 1%
and the top 5% of returns.
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
        # Safety: never end up empty.
        self.returns_ = trimmed if trimmed.size > 0 else arr
        self._fitted = True
        return self

    def simulate(
        self, horizon: int, sims: int, rng: np.random.Generator
    ) -> np.ndarray:
        self._check_fitted()
        idx = rng.integers(0, self.returns_.size, size=(horizon, sims))
        return self.returns_[idx]


@register
class TrimmedTop1(_TrimmedBase):
    name = "trimmed_1pct"
    label = "Trimmed (top 1%)"
    trim_fraction = 0.01


@register
class TrimmedTop5(_TrimmedBase):
    name = "trimmed_5pct"
    label = "Trimmed (top 5%)"
    trim_fraction = 0.05
