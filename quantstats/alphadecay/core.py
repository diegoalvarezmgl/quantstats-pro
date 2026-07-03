"""
Core alpha-decay analysis: rolling metrics, z-scores, traffic lights, CUSUM.

Statistical design
------------------
Each rolling metric is compared to its own historical distribution of past
windows.  Raw rolling values are often skewed (volatility is ~lognormal,
drawdowns/VaR are severity metrics with heavy tails, CAGR is bounded below
at −100%).  A plain z-score on raw values mis-calibrates the ±2σ / ±3σ
traffic lights because the sample mean and std are pulled by those tails.

We therefore map each metric to an *analysis scale* where the distribution is
approximately normal, compute z-scores and percentiles there, and plot
histograms on that same scale.  Summary lines (Obs / Mean above each chart)
stay in the original units so the numbers remain interpretable.

See ``MetricSpec.transform``, ``TRANSFORM_RATIONALE``, and ``_transform_values``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

TrafficLight = Literal["excellent", "good", "warning", "critical"]

_STATUS_LABELS = {
    "excellent": "Excellent",
    "good": "Good",
    "warning": "Warning",
    "critical": "Critical",
}


def _stats():
    from quantstats import stats

    return stats


Transform = Literal["none", "log1p", "log", "log_neg"]

# Per-transform rationale (referenced from MetricSpec and plotting axis labels).
TRANSFORM_RATIONALE: dict[Transform, str] = {
    "none": (
        "No transform — the quantity is already roughly symmetric on the "
        "rolling window, or too discrete for a continuous map (win rate over "
        "7–30 days)."
    ),
    "log1p": (
        "log(1 + x) — CAGR is a growth rate bounded below at −100%.  This is "
        "the continuously-compounded rate; if daily log-returns are "
        "~normal, log(1+CAGR) is ~normal and z-scores are well calibrated."
    ),
    "log": (
        "log(x) — strictly positive, right-skewed quantities (realized vol, "
        "downside vol, payoff ratio).  Realized volatility is empirically "
        "~lognormal (Andersen, Bollerslev, Diebold & Ebens, 2001); the log "
        "map turns multiplicative shocks into additive ones."
    ),
    "log_neg": (
        "log(−x) — negative severity metrics (drawdowns, VaR, Expected "
        "Shortfall).  Values are ≤ 0 with heavier left tails; mapping to "
        "log-severity (positive, higher = worse) stabilises variance and fixes "
        "the traffic-light direction (more severe → higher z)."
    ),
}

# Per-metric notes (why this transform for this key).
METRIC_TRANSFORM_NOTES: dict[str, str] = {
    "cagr": "Growth rate with −100% floor → log(1+CAGR).",
    "volatility": "Annualised std of returns; positive and right-skewed → log(vol).",
    "downside_vol": "Same family as volatility (dispersion of losses) → log.",
    "max_drawdown": "Peak-to-trough loss (≤ 0); severity is multiplicative → log(−DD).",
    "mean_drawdown": "Average underwater depth (≤ 0) → log(−mean DD).",
    "win_rate": "Share of winning days in the window; kept raw (discrete, bounded).",
    "var_95": "Left-tail loss quantile (≤ 0) → log-severity.",
    "cvar_95": "Mean loss beyond VaR (≤ 0) → log-severity.",
    "payoff_ratio": "Ratio of two positive means; ratios are log-symmetric → log.",
    "skew": "Third moment; already centred and unbounded → raw.",
}


@dataclass(frozen=True)
class MetricSpec:
    """
    Metric definition for alpha-decay monitoring.

    ``transform`` selects the analysis-scale map (see ``TRANSFORM_RATIONALE``).
    ``higher_is_better`` is evaluated on that *transformed* scale.
    """

    key: str
    label: str
    higher_is_better: bool
    is_pct: bool
    decimals: int = 2
    transform: Transform = "none"


METRIC_SPECS: tuple[MetricSpec, ...] = (
    MetricSpec("cagr", "CAGR", True, True, 2, "log1p"),
    MetricSpec("volatility", "Volatility", False, True, 2, "log"),
    MetricSpec("downside_vol", "Downside Vol", False, True, 2, "log"),
    MetricSpec("max_drawdown", "Max Drawdown", False, True, 2, "log_neg"),
    MetricSpec("mean_drawdown", "Mean Drawdown", False, True, 2, "log_neg"),
    MetricSpec("win_rate", "Win Rate", True, True, 1, "none"),
    MetricSpec("var_95", "VaR 95%", False, True, 2, "log_neg"),
    MetricSpec("cvar_95", "Expected Shortfall 95%", False, True, 2, "log_neg"),
    MetricSpec("payoff_ratio", "Payoff Ratio", True, False, 2, "log"),
    MetricSpec("skew", "Skew", True, False, 2, "none"),
)


def _transform_values(values: np.ndarray, transform: Transform) -> np.ndarray:
    """
    Map raw rolling metric values to the analysis (approximately normal) scale.

    Used for z-scores, percentiles, traffic lights, and histograms.

    Zero-handling (log / log_neg only)
    ----------------------------------
    Some windows produce exact zeros (no losing days → downside vol = 0;
    monotonic-up window → max drawdown = 0).  Zeros are floored to the
    smallest *positive* observation in the sample so they remain the best
    (lowest-severity) values without −inf or distorted moments.
    """
    if transform == "none":
        return values
    if transform == "log1p":
        return np.log1p(np.maximum(values, -1.0 + 1e-12))
    x = values if transform == "log" else -values
    positive = x[x > 0]
    floor = float(positive.min()) if len(positive) else 1e-12
    return np.log(np.clip(x, floor, None))


def transform_axis_label(spec: MetricSpec) -> str:
    """X-axis label for distribution charts (analysis scale)."""
    labels = {
        "none": spec.label,
        "log1p": f"log(1 + {spec.label})",
        "log": f"log({spec.label})",
        "log_neg": f"log(−{spec.label})  [severity]",
    }
    return labels[spec.transform]


def transform_chart_subtitle(spec: MetricSpec) -> str:
    """One-line transform note shown under chart titles (empty if none)."""
    if spec.transform == "none":
        return ""
    return METRIC_TRANSFORM_NOTES.get(spec.key, TRANSFORM_RATIONALE[spec.transform])


def available_metrics() -> list[str]:
    return [spec.key for spec in METRIC_SPECS]


def _downside_volatility(window: pd.Series, periods: float) -> float:
    arr = window.to_numpy(dtype=float)
    if len(arr) == 0:
        return np.nan
    downside = arr[arr < 0]
    if len(downside) == 0:
        return 0.0
    dd = np.sqrt((downside**2).sum() / len(arr))
    return float(dd * np.sqrt(periods))


def _compute_metric(
    window: pd.Series,
    key: str,
    rf: float,
    periods: float,
) -> float:
    """
    Reference (non-vectorised) implementation of one metric on one window.

    Kept as ground truth for the vectorised fast path in
    ``_rolling_metrics_matrix`` — the parity test in tests/test_alphadecay.py
    asserts both produce identical values.
    """
    stats = _stats()
    w = window.dropna()
    if len(w) < 2:
        return np.nan

    if key == "cagr":
        return float(stats.cagr(w, rf=rf, periods=periods))
    if key == "volatility":
        return float(stats.volatility(w, periods=periods, prepare_returns=False))
    if key == "downside_vol":
        return _downside_volatility(w, periods)
    if key == "max_drawdown":
        return float(stats.to_drawdown_series(w).min())
    if key == "mean_drawdown":
        return float(stats.to_drawdown_series(w).mean())
    if key == "win_rate":
        return float(stats.win_rate(w, prepare_returns=False))
    if key == "var_95":
        return float(stats.value_at_risk(w, confidence=0.95, prepare_returns=False))
    if key == "cvar_95":
        return float(stats.cvar(w, confidence=0.95, prepare_returns=False))
    if key == "payoff_ratio":
        return float(stats.payoff_ratio(w, prepare_returns=False))
    if key == "skew":
        return float(stats.skew(w, prepare_returns=False))
    raise KeyError(f"Unknown metric key: {key}")


def _classify_underwater_status(z: float, is_underwater: bool) -> TrafficLight:
    """
    Traffic light for an *ongoing* underwater spell vs completed historical spells.

    Z is on log(duration).  Thresholds match the metric traffic lights
    (lower-is-better convention): being longer than the log-mean spell is
    normal — spell lengths are right-skewed, so under a ~lognormal model
    z ≤ 2 covers ~98% of typical spells.  Only clearly abnormal durations
    escalate: z > 2 → warning, z > 3 → critical.
    """
    if not is_underwater:
        return "excellent"
    if np.isnan(z):
        return "good"
    if z < 0:
        return "excellent"
    if z <= 2:
        return "good"
    if z <= 3:
        return "warning"
    return "critical"


def _classify_status(z: float, higher_is_better: bool) -> TrafficLight:
    if np.isnan(z):
        return "good"
    if higher_is_better:
        if z > 0:
            return "excellent"
        if z >= -2:
            return "good"
        if z >= -3:
            return "warning"
        return "critical"
    if z < 0:
        return "excellent"
    if z <= 2:
        return "good"
    if z <= 3:
        return "warning"
    return "critical"


def _format_metric_value(spec: MetricSpec, value: float) -> str:
    if value is None or np.isnan(value):
        return "—"
    if spec.is_pct:
        return f"{value * 100:.{spec.decimals}f}%"
    return f"{value:.{spec.decimals}f}"


@dataclass
class WindowResult:
    window: int
    distribution: np.ndarray  # raw rolling values (for Obs/Mean labels)
    mean: float
    std: float
    observed: float
    analysis_distribution: np.ndarray  # transformed — used for z-score & charts
    analysis_mean: float
    analysis_observed: float
    z_score: float
    status: TrafficLight
    percentile: float


@dataclass
class MetricResult:
    spec: MetricSpec
    windows: dict[int, WindowResult] = field(default_factory=dict)


@dataclass
class CusumResult:
    series: pd.Series
    threshold: float
    current: float
    alarm: bool
    pct_of_threshold: float
    target_mean: float
    sigma: float


@dataclass
class TimeUnderwaterResult:
    current_days: int
    distribution: np.ndarray  # raw spell lengths in days
    mean: float
    std: float
    analysis_distribution: np.ndarray  # log(duration) — z-score & chart scale
    analysis_mean: float
    analysis_current: float
    z_score: float
    status: TrafficLight
    percentile: float
    is_underwater: bool


@dataclass
class DecayResult:
    returns: pd.Series
    windows: tuple[int, ...]
    metrics: list[MetricResult]
    score: int
    total: int
    cusum: CusumResult
    time_underwater: TimeUnderwaterResult
    asset_label: str = "Asset"

    @property
    def score_pct(self) -> float:
        return 100.0 * self.score / self.total if self.total else 0.0


def _rolling_metrics_matrix(
    returns: pd.Series,
    window: int,
    periods: float,
) -> dict[str, np.ndarray]:
    """
    Compute every metric for all rolling windows of one size in a single
    vectorised pass.

    ``sliding_window_view`` builds an (n−w+1, w) view of the return array (no
    copy), and each metric is evaluated column-wise with numpy.  This replaces
    ``rolling().apply(python_callback)``, which invoked the full quantstats
    metric stack ~n times per metric per window (~12k Python calls per
    ``analyze()``) and dominated runtime.

    Formulas replicate quantstats.stats exactly (see ``_compute_metric``, the
    reference implementation, and the parity test):

    - cagr: compounded total return annualised by w/periods.  Note that
      ``stats.cagr`` ignores ``rf`` internally (it is in the
      ``unnecessary_function_calls`` list of ``_prepare_returns``), so the
      vectorised version takes no rf either.
    - volatility: sample std (ddof=1) × √periods.
    - downside_vol: √(Σ min(r,0)² / w) × √periods (full-window denominator).
    - max/mean drawdown: equity = cumprod(1+r) with a baseline of 1.0
      (matching ``to_drawdown_series``'s phantom start), running max, min/mean.
    - var_95: Gaussian VaR — norm.ppf(0.05, μ, σ) with sample moments.
    - cvar_95: mean of returns strictly below VaR; falls back to VaR when no
      return is below it (this also removes the "Mean of empty slice"
      RuntimeWarning the old path emitted thousands of times).
    - win_rate: wins / non-zero returns (0.0 when no non-zero returns).
    - payoff_ratio: mean(wins) / |mean(losses)|; NaN when there are no losses.
    - skew: pandas' adjusted Fisher–Pearson estimator
      g1 · √(w(w−1))/(w−2).
    """
    from scipy.stats import norm as _norm

    arr = returns.to_numpy(dtype=float)
    w = window
    mat = np.lib.stride_tricks.sliding_window_view(arr, w)
    m = mat.shape[0]
    out: dict[str, np.ndarray] = {}

    if m == 0 or w < 2:
        empty = np.full(max(m, 0), np.nan)
        return {spec.key: empty.copy() for spec in METRIC_SPECS}

    sqrt_periods = np.sqrt(periods)
    mu = mat.mean(axis=1)
    std = mat.std(axis=1, ddof=1)

    total = np.prod(1.0 + mat, axis=1) - 1.0
    years = w / periods
    out["cagr"] = np.abs(total + 1.0) ** (1.0 / years) - 1.0

    out["volatility"] = std * sqrt_periods

    downside = np.where(mat < 0, mat, 0.0)
    out["downside_vol"] = np.sqrt((downside**2).sum(axis=1) / w) * sqrt_periods

    equity = np.cumprod(1.0 + mat, axis=1)
    running_max = np.maximum(np.maximum.accumulate(equity, axis=1), 1.0)
    dd = equity / running_max - 1.0
    out["max_drawdown"] = dd.min(axis=1)
    out["mean_drawdown"] = dd.mean(axis=1)

    nonzero = (mat != 0).sum(axis=1)
    wins = (mat > 0).sum(axis=1)
    out["win_rate"] = np.where(nonzero > 0, wins / np.maximum(nonzero, 1), 0.0)

    var = _norm.ppf(0.05, mu, std)
    below = mat < var[:, None]
    below_cnt = below.sum(axis=1)
    below_sum = np.where(below, mat, 0.0).sum(axis=1)
    out["var_95"] = var
    out["cvar_95"] = np.where(
        below_cnt > 0, below_sum / np.maximum(below_cnt, 1), var
    )

    pos_cnt = wins
    neg_cnt = (mat < 0).sum(axis=1)
    pos_sum = np.where(mat > 0, mat, 0.0).sum(axis=1)
    neg_sum = np.where(mat < 0, mat, 0.0).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        avg_win = np.where(pos_cnt > 0, pos_sum / np.maximum(pos_cnt, 1), np.nan)
        avg_loss = np.where(neg_cnt > 0, neg_sum / np.maximum(neg_cnt, 1), np.nan)
        out["payoff_ratio"] = avg_win / np.abs(avg_loss)

    dev = mat - mu[:, None]
    m2 = (dev**2).mean(axis=1)
    m3 = (dev**3).mean(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        g1 = m3 / m2**1.5
    out["skew"] = g1 * np.sqrt(w * (w - 1.0)) / (w - 2.0)

    return out


def _rolling_metric_series(
    returns: pd.Series,
    window: int,
    key: str,
    rf: float,
    periods: float,
) -> pd.Series:
    """Single-metric rolling series via the reference implementation (slow path).

    Retained for API compatibility and the vectorisation parity test;
    ``analyze()`` uses ``_rolling_metrics_matrix`` instead.
    """

    def _apply(window_series: pd.Series) -> float:
        return _compute_metric(window_series, key, rf, periods)

    return returns.rolling(window).apply(_apply, raw=False)


def _window_result(
    values: np.ndarray | pd.Series,
    window: int,
    spec: MetricSpec,
) -> WindowResult:
    if isinstance(values, pd.Series):
        values = values.to_numpy(dtype=float)
    clean = values[~np.isnan(values)]
    if len(clean) == 0:
        return WindowResult(
            window=window,
            distribution=np.array([]),
            mean=np.nan,
            std=np.nan,
            observed=np.nan,
            analysis_distribution=np.array([]),
            analysis_mean=np.nan,
            analysis_observed=np.nan,
            z_score=np.nan,
            status="good",
            percentile=np.nan,
        )

    observed = float(clean[-1])
    dist = clean
    mean = float(np.mean(dist))
    std = float(np.std(dist, ddof=1)) if len(dist) > 1 else 0.0

    tdist = _transform_values(dist, spec.transform)
    tobs = float(tdist[-1])
    tmean = float(np.mean(tdist))
    tstd = float(np.std(tdist, ddof=1)) if len(tdist) > 1 else 0.0
    z = (tobs - tmean) / tstd if tstd > 0 else 0.0
    status = _classify_status(z, spec.higher_is_better)
    pct = float((tdist <= tobs).mean() * 100.0)
    if not spec.higher_is_better:
        pct = 100.0 - pct
    return WindowResult(
        window=window,
        distribution=dist,
        mean=mean,
        std=std,
        observed=observed,
        analysis_distribution=tdist,
        analysis_mean=tmean,
        analysis_observed=tobs,
        z_score=z,
        status=status,
        percentile=pct,
    )


def _compute_cusum(
    returns: pd.Series,
    k: float = 0.5,
    h: float = 4.0,
) -> CusumResult:
    """
    One-sided CUSUM for detecting sustained *negative* shifts in daily returns.

    For each trading day t (x-axis index 0 … N−1):

        μ = mean(all daily returns)
        σ = std(all daily returns, ddof=1)
        k_abs = k · σ          # slack — ignore small daily shortfalls
        threshold = h · σ      # alarm line on the chart

        S_t = max(0, S_{t−1} + (μ − r_t) − k_abs)

    Interpretation
    --------------
    - ``μ − r_t`` is how much today's return fell *below* the historical average.
    - ``S_t`` accumulates those shortfalls; it resets to 0 whenever performance
      is on target (the inner term would drive it negative).
    - When ``S_t ≥ threshold`` → alarm (possible alpha decay / regime shift).

    The chart x-axis is the trading-day index in the return series (same order
    as the sample period), not calendar time.
    """
    arr = returns.to_numpy(dtype=float)
    mu = float(np.mean(arr))
    sigma = float(np.std(arr, ddof=1))
    if sigma == 0:
        sigma = 1e-12
    k_abs = k * sigma
    threshold = h * sigma

    values = []
    s = 0.0
    for x in arr:
        s = max(0.0, s + (mu - x) - k_abs)
        values.append(s)

    series = pd.Series(values, index=returns.index, name="cusum")
    current = float(values[-1]) if values else 0.0
    pct = current / threshold if threshold > 0 else 0.0
    return CusumResult(
        series=series,
        threshold=threshold,
        current=current,
        alarm=current >= threshold,
        pct_of_threshold=pct,
        target_mean=mu,
        sigma=sigma,
    )


def _underwater_durations(drawdown: pd.Series) -> list[int]:
    """Length (days) of every underwater spell, including an ongoing spell at the end."""
    durations: list[int] = []
    current = 0
    for val in drawdown.to_numpy(dtype=float):
        if val < 0:
            current += 1
        elif current > 0:
            durations.append(current)
            current = 0
    if current > 0:
        durations.append(current)
    return durations


def _completed_underwater_durations(
    drawdown: pd.Series,
) -> tuple[list[int], int, bool]:
    """
    Split drawdown spells into completed history vs the ongoing spell.

    The ongoing spell must be excluded from the reference distribution —
    otherwise a 102-day streak contaminates its own benchmark and the z-score
    is artificially muted.
    """
    all_spells = _underwater_durations(drawdown)
    current, is_uw = _current_underwater_days(drawdown)
    if is_uw and all_spells and all_spells[-1] == current:
        return all_spells[:-1], current, is_uw
    return all_spells, current, is_uw


def _current_underwater_days(drawdown: pd.Series) -> tuple[int, bool]:
    if len(drawdown) == 0:
        return 0, False
    arr = drawdown.to_numpy(dtype=float)
    if arr[-1] >= 0:
        return 0, False
    days = 0
    for val in arr[::-1]:
        if val < 0:
            days += 1
        else:
            break
    return days, True


def _compute_time_underwater(returns: pd.Series) -> TimeUnderwaterResult:
    stats = _stats()
    dd = stats.to_drawdown_series(returns)
    completed, current, is_uw = _completed_underwater_durations(dd)

    if not completed:
        return TimeUnderwaterResult(
            current_days=current,
            distribution=np.array([]),
            mean=0.0,
            std=0.0,
            analysis_distribution=np.array([]),
            analysis_mean=0.0,
            analysis_current=float(np.log(max(current, 1))),
            z_score=np.nan,
            status=_classify_underwater_status(np.nan, is_uw),
            percentile=50.0,
            is_underwater=is_uw,
        )

    dist = np.array(completed, dtype=float)
    mean = float(np.mean(dist))
    std = float(np.std(dist, ddof=1)) if len(dist) > 1 else 0.0
    ldist = np.log(dist)
    lmean = float(np.mean(ldist))
    lcurrent = float(np.log(max(current, 1)))
    lstd = float(np.std(ldist, ddof=1)) if len(ldist) > 1 else 0.0
    z = (lcurrent - lmean) / lstd if is_uw and lstd > 0 else (0.0 if not is_uw else np.nan)
    status = _classify_underwater_status(z, is_uw)
    if is_uw:
        pct = float((dist <= current).mean() * 100.0)
        pct = 100.0 - pct
    else:
        pct = 50.0
    return TimeUnderwaterResult(
        current_days=current,
        distribution=dist,
        mean=mean,
        std=std,
        analysis_distribution=ldist,
        analysis_mean=lmean,
        analysis_current=lcurrent,
        z_score=z,
        status=status,
        percentile=pct,
        is_underwater=is_uw,
    )


def analyze(
    returns: pd.Series,
    windows: tuple[int, ...] = (7, 15, 30),
    rf: float = 0.0,
    periods: float = 252.0,
    cusum_k: float = 0.5,
    cusum_h: float = 4.0,
    asset_label: str | None = None,
) -> DecayResult:
    """
    Run alpha-decay analysis on a return series.

    Computes rolling risk metrics for each window size, compares the latest
    observation against the historical distribution (z-score / traffic light),
    and adds CUSUM + time-underwater diagnostics.
    """
    if isinstance(returns, pd.DataFrame):
        returns = returns.iloc[:, 0]
    returns = returns.dropna()
    if len(returns) < max(windows) + 5:
        raise ValueError(
            f"Need at least {max(windows) + 5} return observations, got {len(returns)}"
        )

    label = asset_label or (returns.name if returns.name else "Asset")
    metrics: list[MetricResult] = []
    score = 0
    total = 0

    # One vectorised pass per window size covers all metrics (rf is unused by
    # the metric formulas — see _rolling_metrics_matrix).
    per_window = {w: _rolling_metrics_matrix(returns, w, periods) for w in windows}

    for spec in METRIC_SPECS:
        mr = MetricResult(spec=spec)
        for window in windows:
            wr = _window_result(per_window[window][spec.key], window, spec)
            mr.windows[window] = wr
            total += 1
            if wr.status in ("excellent", "good"):
                score += 1
        metrics.append(mr)

    cusum = _compute_cusum(returns, k=cusum_k, h=cusum_h)
    tuw = _compute_time_underwater(returns)

    return DecayResult(
        returns=returns,
        windows=windows,
        metrics=metrics,
        score=score,
        total=total,
        cusum=cusum,
        time_underwater=tuw,
        asset_label=str(label),
    )


def status_label(status: TrafficLight) -> str:
    return _STATUS_LABELS[status]


def format_value(spec: MetricSpec, value: float) -> str:
    return _format_metric_value(spec, value)
