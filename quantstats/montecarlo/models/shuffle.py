"""
Shuffle (permutation bootstrap) model.

Resamples the historical returns *without replacement* per path, preserving the
empirical distribution exactly while breaking time ordering. This mirrors the
legacy ``qs.stats.montecarlo`` behaviour, exposed here as a registered model.
When ``horizon`` differs from the history length, it samples with replacement
of full permutations / truncation as needed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..base import SimulationModel
from ..registry import register


@register
class Shuffle(SimulationModel):
    name = "shuffle"
    label = "Shuffle"

    def calibrate(self, returns: pd.Series) -> Shuffle:
        self.returns_ = self._clean(returns)
        if self.returns_.size == 0:
            raise ValueError("Shuffle model needs at least one return.")
        self._fitted = True
        return self

    def calibration_summary(self, periods: float = 252.0) -> dict[str, str]:
        return {"Permuted observations": f"{self.returns_.size:,}"}

    def simulate(
        self, horizon: int, sims: int, rng: np.random.Generator
    ) -> np.ndarray:
        self._check_fitted()
        n = self.returns_.size
        out = np.empty((horizon, sims), dtype=float)
        for j in range(sims):
            if horizon <= n:
                out[:, j] = rng.permutation(self.returns_)[:horizon]
            else:
                # Tile permutations to fill a longer horizon.
                reps = -(-horizon // n)  # ceil division
                chunks = [rng.permutation(self.returns_) for _ in range(reps)]
                out[:, j] = np.concatenate(chunks)[:horizon]
        return out
