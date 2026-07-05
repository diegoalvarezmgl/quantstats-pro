"""
Merton jump-diffusion model.

Log returns are modelled as a Gaussian diffusion plus a compound-Poisson jump
component. Parameters are estimated with a simple, robust threshold method:
periods whose log return deviates from the mean by more than ``k`` robust
standard deviations are flagged as jumps; the diffusion is fit on the remainder
and the jump size distribution on the flagged set. See the Montecarlo challenge
notes on jump identification (threshold vs EM).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..base import SimulationModel
from ..registry import register


@register
class JumpDiffusion(SimulationModel):
    name = "jump_diffusion"
    label = "Jump Diffusion"

    def __init__(self, threshold: float = 3.0) -> None:
        super().__init__()
        self.threshold = threshold

    def calibrate(self, returns: pd.Series) -> JumpDiffusion:
        log_r = self._to_log(self._clean(returns))
        if log_r.size < 4:
            # Degenerate: no jumps, pure diffusion.
            self.mu_ = float(np.mean(log_r)) if log_r.size else 0.0
            self.sigma_ = float(np.std(log_r, ddof=1)) if log_r.size > 1 else 0.0
            self.lambda_ = 0.0
            self.jump_mean_ = 0.0
            self.jump_std_ = 0.0
            self._fitted = True
            return self

        med = np.median(log_r)
        mad = np.median(np.abs(log_r - med))
        robust_std = 1.4826 * mad if mad > 0 else np.std(log_r, ddof=1)
        is_jump = np.abs(log_r - med) > self.threshold * robust_std

        diffusion = log_r[~is_jump]
        jumps = log_r[is_jump]

        self.mu_ = float(np.mean(diffusion)) if diffusion.size else float(med)
        self.sigma_ = (
            float(np.std(diffusion, ddof=1)) if diffusion.size > 1 else float(robust_std)
        )
        self.lambda_ = float(jumps.size / log_r.size)
        self.jump_mean_ = float(np.mean(jumps)) if jumps.size else 0.0
        self.jump_std_ = float(np.std(jumps, ddof=1)) if jumps.size > 1 else 0.0
        self._fitted = True
        return self

    def calibration_summary(self, periods: float = 252.0) -> dict[str, str]:
        ann_mu = self.mu_ * periods
        ann_sigma = self.sigma_ * np.sqrt(periods)
        ann_lambda = self.lambda_ * periods
        return {
            "Drift (ann.)": f"{ann_mu:.2%}",
            "Diffusion vol (ann.)": f"{ann_sigma:.2%}",
            "Jump intensity (ann.)": f"{ann_lambda:.2f}",
            "Jump mean": f"{self.jump_mean_:.4f}",
        }

    def simulate(
        self, horizon: int, sims: int, rng: np.random.Generator
    ) -> np.ndarray:
        self._check_fitted()
        shape = (horizon, sims)
        diffusion = rng.normal(self.mu_, self.sigma_, size=shape)
        if self.lambda_ > 0:
            n_jumps = rng.poisson(self.lambda_, size=shape)
            # Sum of n_jumps gaussian jumps == Normal(n*mean, sqrt(n)*std).
            jump_component = rng.normal(
                n_jumps * self.jump_mean_,
                np.sqrt(np.maximum(n_jumps, 0)) * self.jump_std_,
            )
        else:
            jump_component = 0.0
        return self._from_log(diffusion + jump_component)
