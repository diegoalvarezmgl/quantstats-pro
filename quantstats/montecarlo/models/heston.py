"""
Heston stochastic-volatility model (first-cut, returns-only calibration).

Heston is usually calibrated to option surfaces; from a return series alone its
five parameters (kappa, theta, xi, rho, v0) are only weakly identified. This
implementation uses a pragmatic method-of-moments calibration on the realised
variance proxy (squared log returns) and simulates with a full-truncation Euler
scheme. It is intended as a reasonable first approximation -- see the Monte
Carlo challenge notes for the identifiability caveats.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..base import SimulationModel
from ..registry import register


@register
class Heston(SimulationModel):
    name = "heston"
    label = "Heston (SV)"

    def calibrate(self, returns: pd.Series) -> Heston:
        log_r = self._to_log(self._clean(returns))
        n = log_r.size
        if n < 8:
            self.mu_ = float(np.mean(log_r)) if n else 0.0
            self.theta_ = float(np.var(log_r)) if n > 1 else 1e-4
            self.v0_ = self.theta_
            self.kappa_ = 0.05
            self.xi_ = 0.1 * np.sqrt(self.theta_)
            self.rho_ = -0.5
            self._fitted = True
            return self

        self.mu_ = float(np.mean(log_r))
        self.theta_ = float(np.var(log_r))
        self.v0_ = self.theta_

        # Variance proxy and its persistence -> kappa.
        v_proxy = (log_r - self.mu_) ** 2
        v_demean = v_proxy - v_proxy.mean()
        denom = float(np.sum(v_demean[:-1] ** 2))
        phi = float(np.sum(v_demean[1:] * v_demean[:-1]) / denom) if denom > 0 else 0.0
        phi = min(max(phi, 1e-3), 0.999)
        self.kappa_ = float(-np.log(phi))

        # Vol-of-vol from stationary variance of the variance proxy:
        # Var(v) ~ xi^2 * theta / (2 kappa)  =>  xi = sqrt(2 kappa Var(v)/theta).
        var_v = float(np.var(v_proxy))
        if self.theta_ > 0 and self.kappa_ > 0:
            self.xi_ = float(np.sqrt(max(2.0 * self.kappa_ * var_v / self.theta_, 0.0)))
        else:
            self.xi_ = float(0.1 * np.sqrt(self.theta_))

        # Leverage: correlation between return and variance innovation.
        dv = np.diff(v_proxy)
        r_lead = log_r[:-1]
        if np.std(dv) > 0 and np.std(r_lead) > 0:
            self.rho_ = float(np.clip(np.corrcoef(r_lead, dv)[0, 1], -0.95, 0.95))
        else:
            self.rho_ = -0.5
        self._fitted = True
        return self

    def calibration_summary(self, periods: float = 252.0) -> dict[str, str]:
        ann_mu = self.mu_ * periods
        ann_vol = float(np.sqrt(self.theta_) * np.sqrt(periods))
        return {
            "Drift (ann.)": f"{ann_mu:.2%}",
            "Volatility (ann.)": f"{ann_vol:.2%}",
            "Mean reversion kappa": f"{self.kappa_:.3f}",
            "Vol-of-vol xi": f"{self.xi_:.4f}",
            "Leverage rho": f"{self.rho_:.2f}",
        }

    def simulate(
        self, horizon: int, sims: int, rng: np.random.Generator
    ) -> np.ndarray:
        self._check_fitted()
        z1 = rng.standard_normal((horizon, sims))
        z2 = rng.standard_normal((horizon, sims))
        # Correlate the two Brownian shocks.
        w_v = z1
        w_s = self.rho_ * z1 + np.sqrt(1.0 - self.rho_**2) * z2

        log_paths = np.empty((horizon, sims), dtype=float)
        v = np.full(sims, max(self.v0_, 1e-12), dtype=float)
        for t in range(horizon):
            v_pos = np.maximum(v, 0.0)
            sqrt_v = np.sqrt(v_pos)
            # mu_ is the mean of log returns; do not subtract variance drag again.
            log_paths[t] = self.mu_ + sqrt_v * w_s[t]
            v = (
                v
                + self.kappa_ * (self.theta_ - v_pos)
                + self.xi_ * sqrt_v * w_v[t]
            )
            v = np.maximum(v, 0.0)
        return self._from_log(log_paths)
