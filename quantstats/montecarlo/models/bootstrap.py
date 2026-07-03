"""
IID bootstrap model.

Resamples historical simple returns *with replacement*, one draw per period.
Preserves the empirical marginal distribution (including fat tails and skew)
but assumes independence across time.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..base import SimulationModel
from ..registry import register


@register
class Bootstrap(SimulationModel):
    name = "bootstrap"
    label = "Bootstrap"

    def calibrate(self, returns: pd.Series) -> Bootstrap:
        self.returns_ = self._clean(returns)
        if self.returns_.size == 0:
            raise ValueError("Bootstrap model needs at least one return.")
        self._fitted = True
        return self

    def calibration_summary(self, periods: float = 252.0) -> dict[str, str]:
        return {"Resampled observations": f"{self.returns_.size:,}"}

    def simulate(
        self, horizon: int, sims: int, rng: np.random.Generator
    ) -> np.ndarray:
        self._check_fitted()
        idx = rng.integers(0, self.returns_.size, size=(horizon, sims))
        return self.returns_[idx]
