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
    baseline_end: int | None = None  # observations used as reference (None = all)


EdgeStatus = Literal["ok", "warning", "critical", "insufficient_data", "no_reference"]


@dataclass
class EdgeTestResult:
    """
    Reference period vs everything after it, on the monitored return stream.

    This is the monitor's statistical statement about decay. The rolling
    metric cells only look at the last few weeks or months, where a change in
    the mean return is far smaller than the noise; this test uses every
    observation since the reference period ended.

    ``p_value`` is the one-sided probability of a Sharpe ratio this far below
    the reference one if nothing had changed. ``status`` is ``warning`` below
    0.05 and ``critical`` below 0.01.
    """

    status: EdgeStatus
    kind: Literal["alpha", "mean_return"]
    message: str
    reference_n: int | None = None
    recent_n: int | None = None
    reference_sharpe_annual: float = np.nan
    recent_sharpe_annual: float = np.nan
    reference_mean_annual: float = np.nan
    recent_mean_annual: float = np.nan
    z_score: float = np.nan
    p_value: float = np.nan

    @property
    def degraded(self) -> bool:
        return self.status in ("warning", "critical")


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
    beta: float | None = None  # set when a benchmark was removed from returns
    edge_test: EdgeTestResult | None = None

    @property
    def score_pct(self) -> float:
        return 100.0 * self.score / self.total if self.total else 0.0

    def to_dict(self) -> dict:
        """Plain, JSON-serialisable summary (no arrays; NaN becomes null)."""

        def num(x):
            x = float(x)
            return x if np.isfinite(x) else None

        metrics = {}
        for mr in self.metrics:
            metrics[mr.spec.key] = {
                "label": mr.spec.label,
                "higher_is_better": mr.spec.higher_is_better,
                "windows": {
                    str(w): {
                        "observed": num(wr.observed),
                        "reference_mean": num(wr.mean),
                        "z_score": num(wr.z_score),
                        "percentile": num(wr.percentile),
                        "status": wr.status,
                    }
                    for w, wr in mr.windows.items()
                },
            }
        tuw = self.time_underwater
        idx = self.returns.index
        dated = isinstance(idx, pd.DatetimeIndex) and len(idx) > 0
        return {
            "label": self.asset_label,
            "n": int(len(self.returns)),
            "start": idx[0].isoformat() if dated else None,
            "end": idx[-1].isoformat() if dated else None,
            "windows": [int(w) for w in self.windows],
            "benchmark_beta": None if self.beta is None else num(self.beta),
            "health": {
                "score": int(self.score),
                "total": int(self.total),
                "score_pct": num(self.score_pct),
            },
            "edge_test": None if self.edge_test is None else {
                "status": self.edge_test.status,
                "degraded": bool(self.edge_test.degraded),
                "kind": self.edge_test.kind,
                "message": self.edge_test.message,
                "reference_observations": self.edge_test.reference_n,
                "recent_observations": self.edge_test.recent_n,
                "reference_sharpe_annual": num(self.edge_test.reference_sharpe_annual),
                "recent_sharpe_annual": num(self.edge_test.recent_sharpe_annual),
                "reference_mean_annual": num(self.edge_test.reference_mean_annual),
                "recent_mean_annual": num(self.edge_test.recent_mean_annual),
                "z_score": num(self.edge_test.z_score),
                "p_value": num(self.edge_test.p_value),
            },
            "metrics": metrics,
            "cusum": {
                "alarm": bool(self.cusum.alarm),
                "pct_of_threshold": num(self.cusum.pct_of_threshold),
                "target_mean": num(self.cusum.target_mean),
                "baseline_observations": self.cusum.baseline_end,
            },
            "time_underwater": {
                "is_underwater": bool(tuw.is_underwater),
                "current_days": int(tuw.current_days),
                "mean_completed_days": num(tuw.mean),
                "z_score": num(tuw.z_score),
                "status": tuw.status,
            },
        }


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
    - cvar_95: Gaussian expected shortfall μ − σ·φ(z)/α, matching
      ``stats.cvar(method="parametric")``.
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
    out["var_95"] = var
    # Gaussian expected shortfall, same estimator as stats.cvar(parametric):
    # μ − σ·φ(z_α)/α. Degenerate windows (σ = 0) collapse to μ.
    es_k = _norm.pdf(_norm.ppf(0.05)) / 0.05
    out["cvar_95"] = np.where(std > 0, mu - std * es_k, mu)

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
    # The latest window shares observations with the `window - 1` rolling
    # values before it. Leave those (and itself) out of the reference, so the
    # current reading is judged against history it is not part of.
    ref_end = len(clean) - window if len(clean) >= 2 * window + 2 else len(clean) - 1
    ref_end = max(ref_end, 1)
    ref = dist[:ref_end]
    mean = float(np.mean(ref))
    std = float(np.std(ref, ddof=1)) if len(ref) > 1 else 0.0

    tdist = _transform_values(dist, spec.transform)
    tobs = float(tdist[-1])
    tref = tdist[:ref_end]
    tmean = float(np.mean(tref))
    tstd = float(np.std(tref, ddof=1)) if len(tref) > 1 else 0.0
    z = (tobs - tmean) / tstd if tstd > 0 else 0.0
    status = _classify_status(z, spec.higher_is_better)
    pct = float((tref <= tobs).mean() * 100.0)
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
    baseline_n: int | None = None,
) -> CusumResult:
    """
    One-sided CUSUM for detecting sustained *negative* shifts in daily returns.

    The target mean and σ come from the first ``baseline_n`` observations (the
    reference period); with ``baseline_n=None`` they come from the whole
    sample, which lets a decayed strategy drag its own target down.

    For each trading day t (x-axis index 0 … N−1):

        μ = mean(reference daily returns)
        σ = std(reference daily returns, ddof=1)
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
    ref = arr if baseline_n is None else arr[: max(2, int(baseline_n))]
    mu = float(np.mean(ref))
    sigma = float(np.std(ref, ddof=1))
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
        baseline_end=None if baseline_n is None else int(len(ref)),
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


def _compute_edge_test(
    returns: pd.Series,
    baseline: float | None,
    periods: float,
    has_benchmark: bool,
) -> EdgeTestResult:
    """Test whether the stream is worse after the reference period than in it."""
    kind: Literal["alpha", "mean_return"] = "alpha" if has_benchmark else "mean_return"
    noun = "Alpha" if has_benchmark else "Performance"
    if baseline is None:
        return EdgeTestResult(
            status="no_reference",
            kind=kind,
            message="No reference period (baseline=None): the edge test was not run.",
        )
    from quantstats.validity import split_analysis

    try:
        res = split_analysis(returns, 1.0 - baseline, periods=periods)
    except ValueError as exc:
        return EdgeTestResult(
            status="insufficient_data",
            kind=kind,
            message=f"Not enough observations to test the edge: {exc}",
        )
    ins, oos = res["in_sample"], res["out_of_sample"]
    ann = float(np.sqrt(periods))
    ref_sr, rec_sr = float(ins["sharpe"]) * ann, float(oos["sharpe"]) * ann
    z, p = float(res["sharpe_change_z"]), float(res["degradation_p_value"])
    if not np.isfinite(p):
        return EdgeTestResult(
            status="insufficient_data",
            kind=kind,
            message="The edge test is undefined for this series (zero variance).",
            reference_n=int(ins["n"]),
            recent_n=int(oos["n"]),
        )
    p_txt = "p<0.001" if p < 0.001 else f"p={p:.3f}"
    status: EdgeStatus = "critical" if p < 0.01 else "warning" if p < 0.05 else "ok"
    if status == "ok":
        message = (
            f"{noun} since the reference period is consistent with it: Sharpe "
            f"{ref_sr:.2f} then, {rec_sr:.2f} since ({p_txt})."
        )
    else:
        message = (
            f"{noun} has deteriorated since the reference period: Sharpe "
            f"{ref_sr:.2f} then, {rec_sr:.2f} since ({p_txt})."
        )
    return EdgeTestResult(
        status=status,
        kind=kind,
        message=message,
        reference_n=int(ins["n"]),
        recent_n=int(oos["n"]),
        reference_sharpe_annual=ref_sr,
        recent_sharpe_annual=rec_sr,
        reference_mean_annual=float(np.mean(returns.to_numpy(dtype=float)[: int(ins["n"])]) * periods),
        recent_mean_annual=float(np.mean(returns.to_numpy(dtype=float)[int(ins["n"]) :]) * periods),
        z_score=z,
        p_value=p,
    )


def analyze(
    returns: pd.Series,
    windows: tuple[int, ...] = (7, 15, 30),
    rf: float = 0.0,
    periods: float = 252.0,
    cusum_k: float = 0.5,
    cusum_h: float = 4.0,
    asset_label: str | None = None,
    benchmark: pd.Series | None = None,
    baseline: float | None = 0.5,
) -> DecayResult:
    """
    Run alpha-decay analysis on a return series.

    Computes rolling risk metrics for each window size, compares the latest
    observation against the historical distribution (z-score / traffic light),
    and adds CUSUM + time-underwater diagnostics.

    ``result.edge_test`` compares the reference period with everything after
    it (one-sided Sharpe test). It is the statistical statement about decay;
    the rolling cells describe only the latest windows and cannot detect a
    change in mean return on their own.

    Parameters
    ----------
    benchmark : pd.Series, optional
        Benchmark returns. When given, the analysis runs on the *active*
        stream ``r − β·benchmark`` (β estimated on the reference period), so
        the monitor tracks alpha rather than total performance.
    baseline : float or None, default 0.5
        Leading fraction of the sample used as the reference period for the
        edge test, the CUSUM target and β. ``None`` uses the whole sample
        (the behaviour before 0.5.0) and skips the edge test.
    """
    if isinstance(returns, pd.DataFrame):
        returns = returns.iloc[:, 0]
    returns = returns.dropna()
    if baseline is not None and not 0 < baseline < 1:
        raise ValueError(f"baseline must be a fraction in (0, 1) or None, got {baseline}")

    beta = None
    if benchmark is not None:
        if isinstance(benchmark, pd.DataFrame):
            benchmark = benchmark.iloc[:, 0]
        name = returns.name
        frame = pd.concat([returns.rename("r"), benchmark.rename("b")], axis=1).dropna()
        ref_n = len(frame) if baseline is None else max(2, int(len(frame) * baseline))
        ref = frame.iloc[:ref_n]
        var_b = float(ref["b"].var())
        beta = float(ref["r"].cov(ref["b"]) / var_b) if var_b > 0 else 0.0
        returns = (frame["r"] - beta * frame["b"]).rename(name)
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

    baseline_n = None if baseline is None else max(2, int(len(returns) * baseline))
    cusum = _compute_cusum(returns, k=cusum_k, h=cusum_h, baseline_n=baseline_n)
    tuw = _compute_time_underwater(returns)
    edge_test = _compute_edge_test(returns, baseline, periods, beta is not None)

    return DecayResult(
        returns=returns,
        windows=windows,
        metrics=metrics,
        score=score,
        total=total,
        cusum=cusum,
        time_underwater=tuw,
        asset_label=str(label),
        beta=beta,
        edge_test=edge_test,
    )


def status_label(status: TrafficLight) -> str:
    return _STATUS_LABELS[status]


def format_value(spec: MetricSpec, value: float) -> str:
    return _format_metric_value(spec, value)
