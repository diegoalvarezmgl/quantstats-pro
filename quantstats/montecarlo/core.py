"""
Orchestration: calibrate and simulate multiple models, collect results.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import analytics
from .base import SimulationModel
from .registry import available_models, get_model

_DRIFT_MODES = ("historical", "zero", "rf")


def infer_periods_per_year(returns: pd.Series) -> float:
    """Infer trading days per year from the return index.

    DatetimeIndex with >5% weekend observations → 365 (crypto / 24-7).
    Otherwise → 252 (US equities).
    """
    idx = returns.index
    if isinstance(idx, pd.DatetimeIndex) and len(idx) > 0:
        weekend = idx.dayofweek >= 5
        if float(weekend.mean()) > 0.05:
            return 365.0
    return 252.0


def _apply_drift(
    returns: pd.Series,
    drift: str,
    rf: float = 0.0,
    periods: float = 252.0,
) -> pd.Series:
    """Re-center the return series' drift before calibration.

    ``historical`` keeps the estimated drift; ``zero`` removes it (isolating
    risk/structure); ``rf`` sets the per-period mean to the risk-free rate.
    Adjustment is done on log returns so it is consistent across models.
    """
    if drift not in _DRIFT_MODES:
        raise ValueError(f"drift must be one of {_DRIFT_MODES}, got {drift!r}")
    if drift == "historical":
        return returns

    simple = returns.to_numpy(dtype=float)
    log_r = np.log1p(np.clip(simple, -0.999999, None))
    target = 0.0 if drift == "zero" else np.log1p(rf) / periods
    log_adj = log_r - np.mean(log_r) + target
    return pd.Series(np.expm1(log_adj), index=returns.index, name=returns.name)


@dataclass
class ModelResult:
    """Container for one model's simulated paths and derived analytics."""

    name: str
    label: str
    sim_returns: np.ndarray  # shape (horizon, sims), simple periodic returns
    category: str = "montecarlo"
    periods: float = 252.0
    bust: float | None = None
    goal: float | None = None
    fitted_model: SimulationModel | None = field(default=None, repr=False)

    @property
    def horizon(self) -> int:
        return int(self.sim_returns.shape[0])

    @property
    def sims(self) -> int:
        return int(self.sim_returns.shape[1])

    @property
    def summary(self) -> dict[str, float]:
        return analytics.summarize(
            self.sim_returns, periods=self.periods, bust=self.bust, goal=self.goal
        )

    def terminal_values(self) -> np.ndarray:
        return analytics.terminal_values(self.sim_returns)

    def fan_chart(self, level: float = 0.95):
        return analytics.fan_chart(self.sim_returns, level=level)

    def calibration_summary(self) -> dict[str, str]:
        if self.fitted_model is None:
            return {}
        return self.fitted_model.calibration_summary(periods=self.periods)


def run_models(
    returns: pd.Series,
    models: list[str] | None = None,
    horizon: int | None = None,
    sims: int = 1000,
    bust: float | None = None,
    goal: float | None = None,
    seed: int | None = None,
    periods: float | None = None,
    drift: str = "historical",
    rf: float = 0.0,
) -> dict[str, ModelResult]:
    """
    Calibrate and simulate one or more models on a single asset's returns.

    Parameters
    ----------
    returns : pd.Series
        Daily (periodic) simple returns of a single asset.
    models : list[str], optional
        Model names to run. Defaults to all registered models.
    horizon : int, optional
        Number of periods per path. Defaults to one year
        (``periods_per_year`` trading days).
    sims : int, default 1000
        Number of simulated paths per model.
    bust : float, optional
        Drawdown threshold for bust probability (e.g. ``-0.25``).
    goal : float, optional
        Terminal-return threshold for goal probability (e.g. ``0.5``).
    seed : int, optional
        Base seed; each model receives an independent child stream so results
        are reproducible yet not artificially correlated across models.
    periods : float, optional
        Periods per year for annualisation. When ``None``, inferred from the
        return index (252 for weekdays-only, 365 for 24/7 data).
    drift : {"historical", "zero", "rf"}, default "historical"
        How to treat the estimated drift before calibration. ``historical``
        keeps it; ``zero`` removes it to compare pure risk/structure; ``rf``
        sets the per-period mean to the risk-free rate.
    rf : float, default 0.0
        Annual risk-free rate, used only when ``drift="rf"``.

    Returns
    -------
    dict[str, ModelResult]
        Mapping of model name to its :class:`ModelResult`, in the requested
        order.
    """
    returns = pd.Series(returns).dropna()
    if periods is None:
        periods = infer_periods_per_year(returns)
    if horizon is None:
        horizon = int(round(periods))
    if horizon <= 0:
        raise ValueError("horizon must be a positive integer")

    returns = _apply_drift(returns, drift, rf=rf, periods=periods)

    names = list(models) if models else available_models()
    seed_seq = np.random.SeedSequence(seed)
    children = seed_seq.spawn(len(names))

    results: dict[str, ModelResult] = {}
    for name, child in zip(names, children, strict=True):
        model = get_model(name)
        model.calibrate(returns)
        rng = np.random.default_rng(child)
        sim_returns = model.simulate(horizon, sims, rng)
        results[name] = ModelResult(
            name=name,
            label=getattr(model, "label", name),
            category=getattr(model, "category", "montecarlo"),
            sim_returns=np.asarray(sim_returns, dtype=float),
            periods=periods,
            bust=bust,
            goal=goal,
            fitted_model=model,
        )
    return results
