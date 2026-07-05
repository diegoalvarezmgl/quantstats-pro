"""
Base interface for Montecarlo simulation models.

A simulation model characterises an asset from its historical returns
(``calibrate``) and produces forward paths of *simple periodic returns*
(``simulate``). Keeping a single, narrow interface lets the engine treat every
characterisation method uniformly and compare them side by side.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class SimulationModel(ABC):
    """
    Abstract base class for Montecarlo characterisation models.

    Subclasses must define a unique ``name`` and implement :meth:`calibrate`
    and :meth:`simulate`. ``simulate`` must return an array of shape
    ``(horizon, sims)`` of **simple** periodic returns (not log, not prices),
    so downstream analytics are uniform across models.
    """

    #: Short, unique identifier used by the registry and reports.
    name: str = "base"
    #: Human-readable label for tables and charts.
    label: str = "Base"
    #: Report grouping: ``"montecarlo"`` (neutral simulation) or ``"stress"``.
    category: str = "montecarlo"

    def __init__(self) -> None:
        self._fitted: bool = False

    @abstractmethod
    def calibrate(self, returns: pd.Series) -> SimulationModel:
        """Estimate model parameters from a returns Series. Returns ``self``."""
        raise NotImplementedError

    @abstractmethod
    def simulate(
        self, horizon: int, sims: int, rng: np.random.Generator
    ) -> np.ndarray:
        """
        Generate simulated simple periodic returns.

        Parameters
        ----------
        horizon : int
            Number of periods (e.g. trading days) per simulated path.
        sims : int
            Number of independent simulated paths.
        rng : numpy.random.Generator
            Source of randomness (seeded by the caller for reproducibility).

        Returns
        -------
        numpy.ndarray
            Array of shape ``(horizon, sims)`` of simple periodic returns.
        """
        raise NotImplementedError

    # --- helpers shared by subclasses -------------------------------------

    @staticmethod
    def _clean(returns: pd.Series) -> np.ndarray:
        """Return finite simple returns as a 1-D float array."""
        arr = pd.Series(returns).dropna().to_numpy(dtype=float)
        return arr[np.isfinite(arr)]

    @staticmethod
    def _to_log(simple: np.ndarray) -> np.ndarray:
        """Convert simple returns to log returns (clipped to avoid -inf)."""
        return np.log1p(np.clip(simple, -0.999999, None))

    @staticmethod
    def _from_log(log_returns: np.ndarray) -> np.ndarray:
        """Convert log returns back to simple returns."""
        return np.expm1(log_returns)

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError(
                f"Model '{self.name}' must be calibrated before simulate()."
            )

    def calibration_summary(self, periods: float = 252.0) -> dict[str, str]:
        """Human-readable calibrated parameters for the tearsheet."""
        return {}
