"""
GARCH(1,1) model with Student-t innovations.

Calibrates a constant-mean GARCH(1,1) on log returns using the ``arch`` package
(maximum likelihood, fat-tailed innovations) and simulates forward paths by
running the GARCH recursion with standardized-t shocks. Volatility clustering
and fat tails are captured natively.

Drift is anchored to the sample mean of log returns so simulated CAGR aligns
with history; GARCH MLE ``mu`` can overweight low-volatility (bull) regimes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# arch is a required dependency of quantstats-pro (see pyproject).
from arch import arch_model

from ..base import SimulationModel
from ..registry import register

# arch fits best when data are scaled to roughly unit magnitude.
_SCALE = 100.0


@register
class GARCH(SimulationModel):
    name = "garch"
    label = "GARCH(1,1)-t"

    def calibrate(self, returns: pd.Series) -> GARCH:
        raw_log = self._to_log(self._clean(returns))
        self.log_mean_ = float(np.mean(raw_log))
        log_r = raw_log * _SCALE
        am = arch_model(log_r, mean="Constant", vol="GARCH", p=1, q=1, dist="t")
        res = am.fit(disp="off")
        p = res.params
        self.mu_mle_ = float(p["mu"])
        # Anchor simulated drift to sample mean (scaled space).
        self.mu_ = self.log_mean_ * _SCALE
        self.omega_ = float(p["omega"])
        self.alpha_ = float(p["alpha[1]"])
        self.beta_ = float(p["beta[1]"])
        self.nu_ = float(p.get("nu", 8.0))
        cv = res.conditional_volatility
        last_vol = float(cv.iloc[-1] if hasattr(cv, "iloc") else cv[-1])
        self.h0_ = last_vol**2
        denom = 1.0 - self.alpha_ - self.beta_
        self.uncond_var_ = self.omega_ / denom if denom > 1e-6 else self.h0_
        self._fitted = True
        return self

    def _standardized_t(self, rng, shape):
        """Student-t with unit variance (scaled by sqrt((nu-2)/nu))."""
        nu = max(self.nu_, 2.1)
        raw = rng.standard_t(nu, size=shape)
        return raw * np.sqrt((nu - 2.0) / nu)

    def simulate(
        self, horizon: int, sims: int, rng: np.random.Generator
    ) -> np.ndarray:
        self._check_fitted()
        z = self._standardized_t(rng, (horizon, sims))
        log_paths = np.empty((horizon, sims), dtype=float)
        h = np.full(sims, max(self.h0_, 1e-12), dtype=float)
        for t in range(horizon):
            eps = z[t] * np.sqrt(h)
            log_paths[t] = (self.mu_ + eps) / _SCALE
            h = self.omega_ + self.alpha_ * eps**2 + self.beta_ * h
        return self._from_log(log_paths)

    def calibration_summary(self, periods: float = 252.0) -> dict[str, str]:
        ann_mu = self.log_mean_ * periods
        ann_vol = float(np.sqrt(self.uncond_var_ / _SCALE**2) * np.sqrt(periods))
        return {
            "Drift (ann.)": f"{ann_mu:.2%}",
            "Volatility (ann.)": f"{ann_vol:.2%}",
            "Student-t nu": f"{self.nu_:.1f}",
            "GARCH alpha": f"{self.alpha_:.3f}",
            "GARCH beta": f"{self.beta_:.3f}",
        }
