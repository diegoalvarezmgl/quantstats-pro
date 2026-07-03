"""
Bayesian models.

``bayesian`` -- conjugate Normal-Inverse-Gamma posterior for the mean and
variance of log returns under a non-informative (Jeffreys) prior. Parameter
uncertainty is propagated by drawing ``(mu, sigma^2)`` from the posterior once
per path and then generating i.i.d. Gaussian log returns. The per-period
predictive is Student-t, but drawing per path correctly inflates path-to-path
dispersion to reflect estimation uncertainty.

``bayesian_bootstrap`` -- Rubin's Bayesian bootstrap: each path draws Dirichlet
weights over the observed returns and resamples accordingly, a smooth
nonparametric posterior over the empirical distribution.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..base import SimulationModel
from ..registry import register


@register
class BayesianNIG(SimulationModel):
    name = "bayesian"
    label = "Bayesian (NIG)"

    def calibrate(self, returns: pd.Series) -> BayesianNIG:
        log_r = self._to_log(self._clean(returns))
        self.n_ = int(log_r.size)
        if self.n_ < 2:
            raise ValueError("Bayesian model needs at least two returns.")
        self.xbar_ = float(np.mean(log_r))
        self.s2_ = float(np.var(log_r, ddof=1))
        self._fitted = True
        return self

    def calibration_summary(self, periods: float = 252.0) -> dict[str, str]:
        ann_mu = self.xbar_ * periods
        ann_sigma = float(np.sqrt(self.s2_) * np.sqrt(periods))
        return {
            "Drift (ann.)": f"{ann_mu:.2%}",
            "Volatility (ann.)": f"{ann_sigma:.2%}",
            "Sample size": f"{self.n_:,}",
        }

    def simulate(
        self, horizon: int, sims: int, rng: np.random.Generator
    ) -> np.ndarray:
        self._check_fitted()
        n = self.n_
        chi2 = rng.chisquare(df=n - 1, size=sims)
        sigma2 = (n - 1) * self.s2_ / np.maximum(chi2, 1e-12)
        mu = self.xbar_ + rng.standard_normal(sims) * np.sqrt(sigma2 / n)
        sigma = np.sqrt(sigma2)
        shocks = rng.standard_normal((horizon, sims))
        log_paths = mu[np.newaxis, :] + shocks * sigma[np.newaxis, :]
        return self._from_log(log_paths)


@register
class BayesianBootstrap(SimulationModel):
    name = "bayesian_bootstrap"
    label = "Bayesian Bootstrap (Rubin)"

    def calibrate(self, returns: pd.Series) -> BayesianBootstrap:
        self.returns_ = self._clean(returns)
        if self.returns_.size == 0:
            raise ValueError("Bayesian bootstrap needs at least one return.")
        self._fitted = True
        return self

    def calibration_summary(self, periods: float = 252.0) -> dict[str, str]:
        return {"Resampled observations": f"{self.returns_.size:,}"}

    def simulate(
        self, horizon: int, sims: int, rng: np.random.Generator
    ) -> np.ndarray:
        self._check_fitted()
        n = self.returns_.size
        out = np.empty((horizon, sims), dtype=float)
        for j in range(sims):
            w = rng.dirichlet(np.ones(n))
            idx = rng.choice(n, size=horizon, p=w)
            out[:, j] = self.returns_[idx]
        return out
