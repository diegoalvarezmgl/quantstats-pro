"""
Moving-block bootstrap model.

Resamples contiguous blocks of returns with replacement, preserving short-range
dependence such as volatility clustering. The default block length uses the
Politis-White (2004, 2009) automatic optimal block length (via ``arch``), which
balances bias and variance from the data's dependence structure. A simpler
volatility-cluster proxy is kept available as a fallback / alternative.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..base import SimulationModel
from ..registry import register


def politis_white_block_length(returns: np.ndarray) -> int:
    """Politis-White automatic optimal block length (stationary bootstrap).

    Uses ``arch.bootstrap.optimal_block_length``; falls back to the
    volatility-cluster estimate if it cannot be computed.
    """
    try:
        from arch.bootstrap import optimal_block_length

        result = optimal_block_length(returns)
        value = float(result["stationary"].iloc[0])
        if np.isfinite(value) and value >= 1:
            return max(1, int(round(value)))
    except Exception:
        pass
    return volatility_cluster_block_length(returns)


def volatility_cluster_block_length(returns: np.ndarray) -> int:
    """Average volatility-cluster length (>=1).

    A 'cluster' is a maximal run of periods whose squared return is above the
    median squared return. The mean run length is a simple, data-driven proxy
    for the persistence of volatility.
    """
    if returns.size < 4:
        return 1
    sq = returns**2
    above = sq > np.median(sq)
    runs = []
    count = 0
    for flag in above:
        if flag:
            count += 1
        elif count > 0:
            runs.append(count)
            count = 0
    if count > 0:
        runs.append(count)
    if not runs:
        return 1
    return max(1, int(round(float(np.mean(runs)))))


@register
class BlockBootstrap(SimulationModel):
    name = "block_bootstrap"
    label = "Block Bootstrap"

    def __init__(
        self, block_length: int | None = None, method: str = "politis_white"
    ) -> None:
        super().__init__()
        self._user_block_length = block_length
        self._method = method

    def calibrate(self, returns: pd.Series) -> BlockBootstrap:
        self.returns_ = self._clean(returns)
        if self.returns_.size == 0:
            raise ValueError("Block bootstrap model needs at least one return.")
        if self._user_block_length is not None:
            self.block_length_ = self._user_block_length
        elif self._method == "vol_cluster":
            self.block_length_ = volatility_cluster_block_length(self.returns_)
        else:
            self.block_length_ = politis_white_block_length(self.returns_)
        self._fitted = True
        return self

    def calibration_summary(self, periods: float = 252.0) -> dict[str, str]:
        return {
            "Block length": f"{self.block_length_}",
            "Resampled observations": f"{self.returns_.size:,}",
        }

    def simulate(
        self, horizon: int, sims: int, rng: np.random.Generator
    ) -> np.ndarray:
        self._check_fitted()
        n = self.returns_.size
        b = max(1, min(self.block_length_, n))
        n_blocks = -(-horizon // b)  # ceil
        out = np.empty((horizon, sims), dtype=float)
        max_start = n - b
        for j in range(sims):
            starts = rng.integers(0, max_start + 1, size=n_blocks)
            path = np.concatenate([self.returns_[s : s + b] for s in starts])
            out[:, j] = path[:horizon]
        return out
