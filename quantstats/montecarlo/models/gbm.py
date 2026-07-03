"""
Geometric Brownian Motion model.

Calibrates the drift and volatility of log returns by maximum likelihood
(sample mean and standard deviation) and simulates i.i.d. Gaussian log returns,
converting them back to simple returns. This preserves the historical drift
(see the Monte Carlo challenge notes on drift handling).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..base import SimulationModel
from ..registry import register


@register
class GBM(SimulationModel):
    name = "gbm"
    label = "GBM"

    def calibrate(self, returns: pd.Series) -> GBM:
        log_r = self._to_log(self._clean(returns))
        self.mu_ = float(np.mean(log_r))
        self.sigma_ = float(np.std(log_r, ddof=1))
        self._fitted = True
        return self

    def simulate(
        self, horizon: int, sims: int, rng: np.random.Generator
    ) -> np.ndarray:
        self._check_fitted()
        log_paths = rng.normal(self.mu_, self.sigma_, size=(horizon, sims))
        return self._from_log(log_paths)
