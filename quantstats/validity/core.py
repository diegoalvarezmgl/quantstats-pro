"""
Backtest validity analytics.

Answers one question about a return series: *how much of this performance
should I believe?*  All Sharpe arithmetic is done in **per-period** units
(the frequency of the input); values are annualised only for display, under
the ``*_annual`` keys.

References
----------
- Bailey & López de Prado (2012), "The Sharpe Ratio Efficient Frontier"
  — probabilistic Sharpe ratio, minimum track record length.
- Bailey & López de Prado (2014), "The Deflated Sharpe Ratio".
- Bailey, Borwein, López de Prado & Zhu (2017), "The Probability of Backtest
  Overfitting" — combinatorially symmetric cross-validation (CSCV).
- Politis & Romano (1994), "The Stationary Bootstrap".
- White (2000), "A Reality Check for Data Snooping"; Romano & Wolf (2005),
  "Stepwise Multiple Testing as Formalized Data Snooping" — max-statistic
  bootstrap behind the selection-adjusted t threshold.
- Newey & West (1987) — HAC standard errors.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm

_EULER_GAMMA = 0.5772156649015329
_MIN_OBS = 30

# Bailey, Borwein, López de Prado & Zhu (2017), section 3.1: "a customary approach
# would be to reject models for which PBO is estimated to be greater than 0.05".
PBO_REJECT_ABOVE = 0.05
# Expected PBO when in-sample rank is unrelated to out-of-sample rank.
PBO_NO_INFORMATION = 0.5


# ---------------------------------------------------------------------------
# Sharpe significance
# ---------------------------------------------------------------------------


def sharpe_standard_error(
    sr: float, n: int, skew: float = 0.0, kurtosis: float = 3.0
) -> float:
    """
    Standard error of a per-period Sharpe ratio estimate.

    ``kurtosis`` is *raw* kurtosis (3 for a normal distribution). Same
    estimator as :func:`quantstats.stats.probabilistic_sharpe_ratio`.
    """
    if n < 2:
        return float("nan")
    var = (1.0 - skew * sr + ((kurtosis - 1.0) / 4.0) * sr**2) / (n - 1)
    return float(math.sqrt(var)) if var > 0 else float("nan")


def probabilistic_sharpe_ratio(
    sr: float,
    n: int,
    skew: float = 0.0,
    kurtosis: float = 3.0,
    sr_benchmark: float = 0.0,
) -> float:
    """P(true Sharpe > ``sr_benchmark``), all Sharpe values per-period."""
    se = sharpe_standard_error(sr, n, skew, kurtosis)
    if not np.isfinite(se) or se == 0:
        return float("nan")
    return float(norm.cdf((sr - sr_benchmark) / se))


def expected_max_sharpe(n_trials: int, trials_sharpe_std: float) -> float:
    """
    Expected maximum per-period Sharpe among ``n_trials`` strategies that all
    have a true Sharpe of zero, given the dispersion of their estimates.

    This is the hurdle a selected backtest must clear before its Sharpe says
    anything: the best of many worthless strategies still looks good.
    """
    if n_trials < 1:
        raise ValueError(f"n_trials must be >= 1, got {n_trials}")
    if n_trials == 1:
        return 0.0
    n = float(n_trials)
    z = (1.0 - _EULER_GAMMA) * norm.ppf(1.0 - 1.0 / n) + _EULER_GAMMA * norm.ppf(
        1.0 - 1.0 / (n * math.e)
    )
    return float(trials_sharpe_std * z)


def min_track_record_length(
    sr: float,
    skew: float = 0.0,
    kurtosis: float = 3.0,
    sr_benchmark: float = 0.0,
    confidence: float = 0.95,
) -> float:
    """
    Observations needed for ``sr`` to exceed ``sr_benchmark`` at ``confidence``.

    Returns ``inf`` when the observed Sharpe does not exceed the benchmark.
    """
    if not np.isfinite(sr) or sr <= sr_benchmark:
        return float("inf")
    z = norm.ppf(confidence)
    var_term = 1.0 - skew * sr + ((kurtosis - 1.0) / 4.0) * sr**2
    return float(1.0 + var_term * (z / (sr - sr_benchmark)) ** 2)


def _moments(arr: np.ndarray) -> dict[str, float]:
    """Sample moments with the same estimators pandas / qs.stats use."""
    s = pd.Series(arr)
    n = int(len(s))
    mean = float(s.mean())
    std = float(s.std(ddof=1)) if n > 1 else float("nan")
    sr = mean / std if std and np.isfinite(std) and std > 0 else float("nan")
    skew = float(s.skew()) if n > 2 else 0.0
    kurt = float(s.kurtosis()) + 3.0 if n > 3 else 3.0
    return {
        "n": n,
        "mean": mean,
        "std": std,
        "sharpe": sr,
        "skew": skew if np.isfinite(skew) else 0.0,
        "kurtosis": kurt if np.isfinite(kurt) else 3.0,
    }


def deflated_sharpe_ratio(
    returns: pd.Series | np.ndarray,
    n_trials: int = 1,
    trials_sharpe_std: float | None = None,
) -> dict[str, float]:
    """
    Deflated Sharpe ratio: the probabilistic Sharpe ratio measured against the
    best Sharpe expected from ``n_trials`` skill-less attempts.

    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Per-period simple returns (already net of the risk-free rate if wanted).
    n_trials : int, default 1
        Number of strategy variants tried before selecting this one. With 1
        the result equals the probabilistic Sharpe ratio.
    trials_sharpe_std : float, optional
        Standard deviation of the per-period Sharpe ratios across the trials.
        When omitted it is assumed equal to the sampling error of a zero-skill
        Sharpe over this track record, ``sqrt(1 / (n - 1))`` — i.e. the trials
        are treated as independent and indistinguishable from noise.

    Returns
    -------
    dict with ``dsr``, ``psr``, ``sharpe``, ``sharpe_hurdle``,
    ``sharpe_std_error``, ``trials_sharpe_std``, ``n_trials``, ``n``.
    """
    arr = np.asarray(returns, dtype=float)
    arr = arr[np.isfinite(arr)]
    m = _moments(arr)
    n = m["n"]
    if trials_sharpe_std is None:
        trials_sharpe_std = math.sqrt(1.0 / (n - 1)) if n > 1 else float("nan")
    hurdle = expected_max_sharpe(int(n_trials), float(trials_sharpe_std))
    return {
        "dsr": probabilistic_sharpe_ratio(
            m["sharpe"], n, m["skew"], m["kurtosis"], sr_benchmark=hurdle
        ),
        "psr": probabilistic_sharpe_ratio(m["sharpe"], n, m["skew"], m["kurtosis"]),
        "sharpe": m["sharpe"],
        "sharpe_hurdle": hurdle,
        "sharpe_std_error": sharpe_standard_error(
            m["sharpe"], n, m["skew"], m["kurtosis"]
        ),
        "trials_sharpe_std": float(trials_sharpe_std),
        "n_trials": int(n_trials),
        "n": n,
    }


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals
# ---------------------------------------------------------------------------


def _stationary_bootstrap_indices(
    n: int, n_boot: int, mean_block: float, rng: np.random.Generator
) -> np.ndarray:
    """Index matrix (n_boot, n) for the Politis–Romano stationary bootstrap."""
    restart = rng.random((n_boot, n)) < (1.0 / mean_block)
    restart[:, 0] = True
    starts = rng.integers(0, n, size=(n_boot, n))
    pos = np.arange(n)[None, :]
    last = np.maximum.accumulate(np.where(restart, pos, 0), axis=1)
    block_start = np.take_along_axis(starts, last, axis=1)
    return (block_start + (pos - last)) % n


def _path_stats(mat: np.ndarray, periods: float) -> dict[str, np.ndarray]:
    """Annualised Sharpe, CAGR, volatility and max drawdown per row."""
    n = mat.shape[1]
    mean = mat.mean(axis=1)
    std = mat.std(axis=1, ddof=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        sharpe = np.where(std > 0, mean / std * math.sqrt(periods), np.nan)
    equity = np.cumprod(1.0 + mat, axis=1)
    total = equity[:, -1]
    with np.errstate(invalid="ignore"):
        cagr = np.where(total > 0, total ** (periods / n) - 1.0, -1.0)
    peak = np.maximum(np.maximum.accumulate(equity, axis=1), 1.0)
    max_dd = (equity / peak - 1.0).min(axis=1)
    return {
        "sharpe": sharpe,
        "cagr": cagr,
        "volatility": std * math.sqrt(periods),
        "max_drawdown": max_dd,
    }


def bootstrap_confidence_intervals(
    returns: pd.Series | np.ndarray,
    periods: float = 252.0,
    n_boot: int = 2000,
    confidence: float = 0.95,
    mean_block: float | None = None,
    seed: int | None = 0,
) -> dict[str, Any]:
    """
    Stationary-bootstrap confidence intervals for the headline statistics.

    Blocks of random length (mean ``mean_block``, default ``n ** (1/3)``)
    preserve short-range dependence such as volatility clustering, which an
    iid bootstrap would destroy.

    Returns a dict with one entry per statistic (``sharpe``, ``cagr``,
    ``volatility``, ``max_drawdown`` — all annualised except drawdown), each
    holding ``observed``, ``low``, ``median``, ``high``; plus
    ``prob_sharpe_below_zero``, ``mean_block``, ``n_boot`` and the raw
    ``samples`` arrays (not included in ``ValidityResult.to_dict``).
    """
    arr = np.asarray(returns, dtype=float)
    arr = arr[np.isfinite(arr)]
    n = len(arr)
    if n < 2:
        raise ValueError("Need at least 2 observations to bootstrap.")
    if mean_block is None:
        mean_block = max(1.0, round(n ** (1.0 / 3.0)))
    rng = np.random.default_rng(seed)

    # Chunk so the index matrix stays around 2M cells whatever the series length
    chunk = max(1, min(n_boot, 2_000_000 // n))
    parts: dict[str, list[np.ndarray]] = {}
    done = 0
    while done < n_boot:
        b = min(chunk, n_boot - done)
        idx = _stationary_bootstrap_indices(n, b, float(mean_block), rng)
        for key, values in _path_stats(arr[idx], periods).items():
            parts.setdefault(key, []).append(values)
        done += b
    samples = {k: np.concatenate(v) for k, v in parts.items()}
    observed = {k: float(v[0]) for k, v in _path_stats(arr[None, :], periods).items()}

    alpha = (1.0 - confidence) / 2.0
    out: dict[str, Any] = {}
    for key, values in samples.items():
        clean = values[np.isfinite(values)]
        if len(clean) == 0:
            low = med = high = float("nan")
        else:
            low, med, high = (
                float(x) for x in np.quantile(clean, [alpha, 0.5, 1.0 - alpha])
            )
        out[key] = {"observed": observed[key], "low": low, "median": med, "high": high}
    sh = samples["sharpe"][np.isfinite(samples["sharpe"])]
    out["prob_sharpe_below_zero"] = float((sh <= 0).mean()) if len(sh) else float("nan")
    out["mean_block"] = float(mean_block)
    out["n_boot"] = int(n_boot)
    out["confidence"] = float(confidence)
    out["samples"] = samples
    return out


# ---------------------------------------------------------------------------
# In-sample / out-of-sample
# ---------------------------------------------------------------------------


def _segment_summary(arr: np.ndarray, periods: float) -> dict[str, float]:
    m = _moments(arr)
    stats = {k: float(v[0]) for k, v in _path_stats(arr[None, :], periods).items()}
    return {
        "n": m["n"],
        "sharpe": m["sharpe"],
        "sharpe_annual": stats["sharpe"],
        "sharpe_std_error": sharpe_standard_error(
            m["sharpe"], m["n"], m["skew"], m["kurtosis"]
        ),
        "psr": probabilistic_sharpe_ratio(m["sharpe"], m["n"], m["skew"], m["kurtosis"]),
        "cagr": stats["cagr"],
        "volatility": stats["volatility"],
        "max_drawdown": stats["max_drawdown"],
        "win_rate": float((arr > 0).mean()),
    }


def _resolve_split(returns: pd.Series, split: Any) -> int:
    """Positional index of the first out-of-sample observation."""
    n = len(returns)
    if isinstance(split, (float, np.floating)) and 0 < float(split) < 1:
        pos = int(round(n * (1.0 - float(split))))
    elif isinstance(split, (int, np.integer)) and not isinstance(split, bool):
        raise ValueError(
            "split must be a date (e.g. '2022-01-01') or the out-of-sample "
            "fraction as a float between 0 and 1 (e.g. 0.3 for the last 30%)."
        )
    else:
        if not isinstance(returns.index, pd.DatetimeIndex):
            raise ValueError(
                "A date split needs a DatetimeIndex. Pass the out-of-sample "
                "fraction instead (e.g. split=0.3)."
            )
        ts = pd.Timestamp(split)
        if returns.index.tz is not None and ts.tzinfo is None:
            ts = ts.tz_localize(returns.index.tz)
        pos = int(returns.index.searchsorted(ts, side="left"))
    if pos < _MIN_OBS or n - pos < _MIN_OBS:
        raise ValueError(
            f"split leaves {pos} in-sample and {n - pos} out-of-sample "
            f"observations; each side needs at least {_MIN_OBS}."
        )
    return pos


def split_analysis(
    returns: pd.Series, split: Any, periods: float = 252.0
) -> dict[str, Any]:
    """
    Compare performance before and after ``split``.

    ``split`` is the first out-of-sample date (anything ``pd.Timestamp``
    accepts) or the out-of-sample fraction as a float in (0, 1). Use the date
    the strategy was frozen or went live: everything before it was available
    while the strategy was being designed.

    The Sharpe difference is tested with a z-statistic built from each
    segment's standard error; ``degradation_p_value`` is the one-sided
    probability of seeing an out-of-sample Sharpe this far below the in-sample
    one if nothing had changed.
    """
    pos = _resolve_split(returns, split)
    is_arr = returns.to_numpy(dtype=float)[:pos]
    oos_arr = returns.to_numpy(dtype=float)[pos:]
    ins = _segment_summary(is_arr, periods)
    oos = _segment_summary(oos_arr, periods)
    se = math.sqrt(ins["sharpe_std_error"] ** 2 + oos["sharpe_std_error"] ** 2)
    z = (oos["sharpe"] - ins["sharpe"]) / se if se > 0 else float("nan")
    retention = (
        oos["sharpe"] / ins["sharpe"] if ins["sharpe"] and ins["sharpe"] > 0 else None
    )
    idx = returns.index
    return {
        "split_position": pos,
        "split_label": str(idx[pos].date()) if hasattr(idx[pos], "date") else str(idx[pos]),
        "in_sample": ins,
        "out_of_sample": oos,
        "sharpe_change_z": float(z),
        "degradation_p_value": float(norm.cdf(z)) if np.isfinite(z) else float("nan"),
        "sharpe_retention": retention,
    }


# ---------------------------------------------------------------------------
# Alpha and decay
# ---------------------------------------------------------------------------


def _hac_ols(y: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """OLS coefficients and Newey–West (Bartlett) standard errors."""
    n, k = x.shape
    xtx_inv = np.linalg.pinv(x.T @ x)
    beta = xtx_inv @ x.T @ y
    resid = y - x @ beta
    lags = int(math.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    xe = x * resid[:, None]
    s = xe.T @ xe
    for lag in range(1, min(lags, n - 1) + 1):
        w = 1.0 - lag / (lags + 1.0)
        g = xe[lag:].T @ xe[:-lag]
        s += w * (g + g.T)
    cov = xtx_inv @ s @ xtx_inv * (n / max(n - k, 1))
    return beta, np.sqrt(np.clip(np.diag(cov), 0.0, None))


def _p_two_sided(t: float) -> float:
    return float(2.0 * (1.0 - norm.cdf(abs(t)))) if np.isfinite(t) else float("nan")


def decay_analysis(
    returns: pd.Series,
    benchmark: pd.Series | None = None,
    periods: float = 252.0,
    window: int | None = None,
) -> dict[str, Any]:
    """
    Estimate the edge and test whether it is shrinking over time.

    With a ``benchmark`` the edge is CAPM alpha: ``r = α + β·b``. Without one
    it is the mean return itself. In both cases a second regression adds a
    linear time term; its coefficient is the change in annualised edge per
    year, tested with Newey–West standard errors. A significantly negative
    trend is the statistical signature of alpha decay.

    ``rolling`` holds the edge re-estimated over a trailing ``window``
    (default: half a year of data, capped at a quarter of the sample).
    """
    r = returns.astype(float)
    has_bench = benchmark is not None
    if has_bench:
        frame = pd.concat([r.rename("r"), benchmark.astype(float).rename("b")], axis=1)
        frame = frame.dropna()
        if len(frame) < _MIN_OBS:
            raise ValueError(
                f"Only {len(frame)} dates are shared by returns and benchmark; "
                f"need at least {_MIN_OBS}. Check that both use the same calendar."
            )
        r, b = frame["r"], frame["b"]
    n = len(r)
    y = r.to_numpy()
    years = np.arange(n) / periods
    years_c = years - years.mean()
    ones = np.ones(n)

    cols = [ones] + ([b.to_numpy()] if has_bench else [])
    coef, se = _hac_ols(y, np.column_stack(cols))
    edge, edge_se = float(coef[0]), float(se[0])
    edge_t = edge / edge_se if edge_se > 0 else float("nan")

    coef_t, se_t = _hac_ols(y, np.column_stack(cols + [years_c]))
    trend, trend_se = float(coef_t[-1]), float(se_t[-1])
    trend_t = trend / trend_se if trend_se > 0 else float("nan")

    if window is None:
        window = int(min(max(periods // 2, 20), max(n // 4, 20)))
    window = int(max(5, min(window, n)))
    if has_bench:
        cov = r.rolling(window).cov(b)
        var = b.rolling(window).var()
        roll_beta = cov / var.where(var > 0)
        roll_edge = (r.rolling(window).mean() - roll_beta * b.rolling(window).mean())
    else:
        roll_beta = None
        roll_edge = r.rolling(window).mean()

    return {
        "kind": "alpha" if has_bench else "mean_return",
        "n": n,
        "edge_annual": edge * periods,
        "edge_t_stat": float(edge_t),
        "edge_p_value": _p_two_sided(edge_t),
        "beta": float(coef[1]) if has_bench else None,
        "trend_annual_per_year": trend * periods,
        "trend_t_stat": float(trend_t),
        "trend_p_value": _p_two_sided(trend_t),
        "window": window,
        "index": r.index,
        "rolling": (roll_edge * periods).dropna(),
        "rolling_beta": roll_beta.dropna() if roll_beta is not None else None,
    }


def _hac_intercept(
    y: np.ndarray, b: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Intercept and its Newey–West standard error for many series at once.

    ``y`` has shape (..., n, k): k series over n periods. ``b`` is the single
    regressor shared by all of them, shape (..., n), or None for a mean-only
    model. Same estimator as :func:`_hac_ols`, vectorised over series and over
    any leading (bootstrap) dimensions.
    """
    n = y.shape[-2]
    if b is None:
        alpha = y.mean(axis=-2)
        resid = y - alpha[..., None, :]
        u = resid / n
        dof = 1
    else:
        bm = b.mean(axis=-1, keepdims=True)
        bc = b - bm
        sbb = (bc**2).sum(axis=-1, keepdims=True)
        sbb = np.where(sbb > 0, sbb, np.nan)
        beta = (bc[..., None] * y).sum(axis=-2) / sbb
        alpha = y.mean(axis=-2) - beta * bm
        resid = y - alpha[..., None, :] - beta[..., None, :] * b[..., None]
        # First row of (X'X)^-1 applied to each x_t
        h = 1.0 / n - bm * bc / sbb
        u = h[..., None] * resid
        dof = 2
    lags = int(math.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    var = (u**2).sum(axis=-2)
    for lag in range(1, min(lags, n - 1) + 1):
        w = 1.0 - lag / (lags + 1.0)
        var = var + 2.0 * w * (u[..., lag:, :] * u[..., :-lag, :]).sum(axis=-2)
    var = var * (n / max(n - dof, 1))
    return alpha, np.sqrt(np.clip(var, 0.0, None))


def selection_adjusted_t_threshold(
    trials: pd.DataFrame | None = None,
    benchmark: pd.Series | None = None,
    *,
    n_trials: int | None = None,
    confidence: float = 0.95,
    n_boot: int = 1000,
    mean_block: float | None = None,
    seed: int | None = 0,
) -> dict[str, Any]:
    """
    Critical value for the t-statistic of the edge of a *selected* strategy.

    A t-statistic is compared with 1.96 when one strategy was tested. When the
    strategy is the best of several, the right comparison is the distribution
    of the largest t-statistic among all of them under the null of no edge.

    With ``trials`` (returns of every variant tried, one column each) that
    distribution is estimated by a studentised max-t stationary bootstrap, in
    the spirit of White (2000) and Romano & Wolf (2005): trials and benchmark
    are resampled with the same time indices, so the dependence between
    variants is kept; each replicate's t-statistics are recentred on the
    observed estimates; the threshold is the upper quantile of their maximum.

    With only ``n_trials`` the Šidák bound for independent tests is used. It
    is conservative when the variants are correlated, which they usually are.

    The test is one-sided at ``(1 - confidence) / 2``, so a single trial gives
    the usual 1.96 at ``confidence=0.95``.

    Returns a dict with ``threshold``, ``method`` (``single_test``, ``sidak``
    or ``max_t_bootstrap``), ``n_trials``, ``level`` and ``n_boot``.
    """
    level = (1.0 - confidence) / 2.0
    if trials is None:
        n_eff = int(n_trials) if n_trials is not None else 1
        if n_eff < 1:
            raise ValueError(f"n_trials must be >= 1, got {n_eff}")
        single = n_eff == 1
        q = 1.0 - level if single else (1.0 - level) ** (1.0 / n_eff)
        return {
            "threshold": float(norm.ppf(q)),
            "method": "single_test" if single else "sidak",
            "n_trials": n_eff,
            "level": level,
            "n_boot": 0,
        }

    if not isinstance(trials, pd.DataFrame) or trials.shape[1] < 1:
        raise ValueError(
            "trials must be a DataFrame with one column of returns per variant."
        )
    frame = trials.astype(float)
    if benchmark is not None:
        frame = pd.concat([frame, benchmark.astype(float).rename("__benchmark__")], axis=1)
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna(how="any")
    if len(frame) < _MIN_OBS:
        raise ValueError(
            f"Only {len(frame)} complete rows across trials"
            f"{' and benchmark' if benchmark is not None else ''}; "
            f"need at least {_MIN_OBS}."
        )
    b = frame.pop("__benchmark__").to_numpy() if benchmark is not None else None
    y = frame.to_numpy()
    y = y[:, y.std(axis=0) > 0]
    n, k = y.shape
    if k == 0:
        raise ValueError("Every trial has zero variance.")
    if k == 1:
        return selection_adjusted_t_threshold(n_trials=1, confidence=confidence)

    alpha_hat, _ = _hac_intercept(y, b)
    if mean_block is None:
        mean_block = max(1.0, round(n ** (1.0 / 3.0)))
    rng = np.random.default_rng(seed)
    chunk = max(1, min(n_boot, 4_000_000 // (n * k)))
    maxima: list[np.ndarray] = []
    done = 0
    while done < n_boot:
        size = min(chunk, n_boot - done)
        idx = _stationary_bootstrap_indices(n, size, float(mean_block), rng)
        a_star, se_star = _hac_intercept(y[idx], b[idx] if b is not None else None)
        with np.errstate(invalid="ignore", divide="ignore"):
            t_star = np.where(se_star > 0, (a_star - alpha_hat) / se_star, np.nan)
        maxima.append(np.nanmax(t_star, axis=1))
        done += size
    dist = np.concatenate(maxima)
    dist = dist[np.isfinite(dist)]
    return {
        "threshold": float(np.quantile(dist, 1.0 - level)),
        "method": "max_t_bootstrap",
        "n_trials": int(k),
        "level": level,
        "n_boot": int(len(dist)),
    }


# ---------------------------------------------------------------------------
# Probability of backtest overfitting (CSCV)
# ---------------------------------------------------------------------------


def probability_of_backtest_overfitting(
    trials: pd.DataFrame, n_splits: int = 16
) -> dict[str, Any]:
    """
    Probability of backtest overfitting via combinatorially symmetric
    cross-validation.

    Parameters
    ----------
    trials : pd.DataFrame
        One column of per-period returns per strategy variant that was tried
        (every variant, not only the good ones), on a common index.
    n_splits : int, default 16
        Number of contiguous time blocks (even). Every way of choosing half of
        them is used once as in-sample, with the other half out-of-sample.

    Returns
    -------
    dict with ``pbo`` (share of splits where the in-sample winner lands in the
    bottom half out-of-sample), ``prob_oos_loss`` (share where the winner has
    a negative out-of-sample Sharpe), ``degradation_slope`` (OOS vs IS Sharpe
    of the winner; below 1 means in-sample rank overstates what follows),
    ``logits``, ``n_trials``, ``n_splits``, ``n_combinations``.
    """
    if not isinstance(trials, pd.DataFrame) or trials.shape[1] < 2:
        raise ValueError(
            "trials must be a DataFrame with one column of returns per "
            "strategy variant (at least 2 columns)."
        )
    mat = trials.dropna(how="any").to_numpy(dtype=float)
    t, n_trials = mat.shape
    n_splits = int(n_splits) - int(n_splits) % 2
    while n_splits > 2 and t // n_splits < 5:
        n_splits -= 2
    if n_splits < 2 or t // n_splits < 2:
        raise ValueError(
            f"Not enough complete rows ({t}) across trials to form time blocks."
        )
    rows = t - t % n_splits
    blocks = mat[:rows].reshape(n_splits, rows // n_splits, n_trials)
    cnt = float(rows // n_splits)
    b_sum = blocks.sum(axis=1)
    b_sq = (blocks**2).sum(axis=1)

    combos = np.array(list(combinations(range(n_splits), n_splits // 2)))
    sel = np.zeros((len(combos), n_splits))
    np.put_along_axis(sel, combos, 1.0, axis=1)

    def _sharpe(mask: np.ndarray) -> np.ndarray:
        n_obs = mask.sum(axis=1, keepdims=True) * cnt
        s = mask @ b_sum
        q = mask @ b_sq
        mean = s / n_obs
        var = (q - n_obs * mean**2) / (n_obs - 1.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(var > 0, mean / np.sqrt(var), 0.0)

    sr_is = _sharpe(sel)
    sr_oos = _sharpe(1.0 - sel)
    best = sr_is.argmax(axis=1)
    pick = np.arange(len(combos))
    best_oos = sr_oos[pick, best]
    best_is = sr_is[pick, best]
    # Relative out-of-sample rank of the in-sample winner, in (0, 1)
    rank = (sr_oos < best_oos[:, None]).sum(axis=1) + 0.5 * (
        (sr_oos == best_oos[:, None]).sum(axis=1) + 1
    )
    omega = rank / (n_trials + 1.0)
    logits = np.log(omega / (1.0 - omega))
    slope = (
        float(np.polyfit(best_is, best_oos, 1)[0])
        if np.ptp(best_is) > 0
        else float("nan")
    )
    return {
        "pbo": float((logits <= 0).mean()),
        "prob_oos_loss": float((best_oos < 0).mean()),
        "degradation_slope": slope,
        "logits": logits,
        "n_trials": int(n_trials),
        "n_splits": int(n_splits),
        "n_combinations": int(len(combos)),
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


@dataclass
class Finding:
    """One observation about the evidence. ``severity``: info, caution, warning."""

    code: str
    severity: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return {"code": self.code, "severity": self.severity, "message": self.message}


@dataclass
class ValidityResult:
    """Output of :func:`validate`. ``to_dict()`` is strict-JSON safe."""

    label: str
    periods: float
    returns: pd.Series = field(repr=False)
    sample: dict[str, Any]
    sharpe: dict[str, Any]
    bootstrap: dict[str, Any] = field(repr=False)
    split: dict[str, Any] | None
    decay: dict[str, Any] = field(repr=False)
    overfitting: dict[str, Any] | None = field(repr=False)
    findings: list[Finding]
    assumptions: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Plain, JSON-serialisable summary (no arrays, NaN/inf become null)."""
        boot = {k: v for k, v in self.bootstrap.items() if k != "samples"}
        decay = {
            k: v for k, v in self.decay.items() if k not in ("rolling", "rolling_beta", "index")
        }
        over = None
        if self.overfitting is not None:
            over = {k: v for k, v in self.overfitting.items() if k != "logits"}
        return _jsonable(
            {
                "label": self.label,
                "periods_per_year": self.periods,
                "sample": self.sample,
                "sharpe": self.sharpe,
                "bootstrap": boot,
                "split": self.split,
                "decay": decay,
                "overfitting": over,
                "findings": [f.to_dict() for f in self.findings],
                "assumptions": self.assumptions,
            }
        )

    def to_json(self, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, allow_nan=False)

    def summary(self) -> str:
        """Short plain-text read-out."""
        s, b = self.sharpe, self.bootstrap
        lines = [
            f"Backtest validity — {self.label}",
            f"  Observations: {self.sample['n']} "
            f"({self.sample['years']:.1f} years at {self.periods:g}/year)",
            f"  Sharpe (annualised): {s['sharpe_annual']:.2f}  "
            f"[{b['sharpe']['low']:.2f}, {b['sharpe']['high']:.2f}] "
            f"{b['confidence']:.0%} bootstrap CI",
            f"  Probabilistic Sharpe: {s['psr']:.1%}   "
            f"Deflated Sharpe ({s['n_trials']} trials): {s['dsr']:.1%}",
            f"  Sharpe hurdle from selection (annualised): "
            f"{s['sharpe_hurdle_annual']:.2f}",
            f"  Track record needed vs that hurdle: "
            f"{_fmt_obs(s['min_track_record_length_deflated'])} obs "
            f"(have {self.sample['n']}; vs zero: "
            f"{_fmt_obs(s['min_track_record_length'])})",
        ]
        if self.split:
            ins, oos = self.split["in_sample"], self.split["out_of_sample"]
            lines.append(
                f"  In-sample Sharpe {ins['sharpe_annual']:.2f} → out-of-sample "
                f"{oos['sharpe_annual']:.2f} from {self.split['split_label']} "
                f"(p={self.split['degradation_p_value']:.3f})"
            )
        d = self.decay
        name = "Alpha" if d["kind"] == "alpha" else "Mean return"
        bar = ""
        if d.get("edge_t_threshold_n_trials", 1) > 1:
            bar = f" vs {d['edge_t_threshold']:.2f} for {d['edge_t_threshold_n_trials']} trials"
        lines.append(
            f"  {name} (annualised): {d['edge_annual']:.2%} (t={d['edge_t_stat']:.2f}{bar}); "
            f"trend {d['trend_annual_per_year']:+.2%}/year (t={d['trend_t_stat']:.2f})"
        )
        if self.overfitting:
            lines.append(
                f"  Probability of backtest overfitting: {self.overfitting['pbo']:.1%} "
                f"({self.overfitting['n_trials']} trials)"
            )
        lines.append("  Findings:")
        lines += [f"    [{f.severity}] {f.code}: {f.message}" for f in self.findings]
        return "\n".join(lines)


def _fmt_obs(x: float) -> str:
    return "∞" if not np.isfinite(x) else f"{math.ceil(x):,}"


def _jsonable(obj: Any) -> Any:
    """Recursively convert to strict-JSON types."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return float(obj) if math.isfinite(float(obj)) else None
    if isinstance(obj, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(obj).isoformat()
    if obj is None or isinstance(obj, str):
        return obj
    if isinstance(obj, np.ndarray):
        return [_jsonable(v) for v in obj.tolist()]
    if isinstance(obj, pd.Series):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    return str(obj)


def _clean_returns(returns: Any, name: str = "returns") -> pd.Series:
    """Coerce to a clean float Series and reject inputs that look like prices."""
    if isinstance(returns, pd.DataFrame):
        if returns.shape[1] != 1:
            raise ValueError(
                f"{name} must be a single series, got a DataFrame with columns "
                f"{list(returns.columns)}. Select one, e.g. {name}['{returns.columns[0]}']."
            )
        returns = returns.iloc[:, 0]
    if not isinstance(returns, pd.Series):
        returns = pd.Series(np.asarray(returns, dtype=float))
    s = pd.to_numeric(returns, errors="coerce").astype(float)
    s = s.replace([np.inf, -np.inf], np.nan).dropna()
    if len(s) and s.min() > 0 and s.median() > 0.5:
        raise ValueError(
            f"{name} looks like a price series (all values positive, median "
            f"{s.median():.2f}). Pass simple returns instead: "
            "qs.utils.to_returns(prices) or prices.pct_change().dropna()."
        )
    if len(s) and s.min() <= -1:
        raise ValueError(
            f"{name} contains a return of {s.min():.2f} (≤ -100%). Expected "
            "simple returns as fractions, e.g. 0.01 for +1%."
        )
    return s


def validate(
    returns: pd.Series,
    benchmark: pd.Series | None = None,
    *,
    n_trials: int | None = None,
    trials: pd.DataFrame | None = None,
    split: Any = None,
    periods: float | None = None,
    rf: float = 0.0,
    confidence: float = 0.95,
    n_boot: int = 2000,
    seed: int | None = 0,
    label: str | None = None,
) -> ValidityResult:
    """
    Assess how much of a backtest's performance is statistically credible.

    Parameters
    ----------
    returns : pd.Series
        Simple per-period returns of the strategy (fractions: 0.01 = +1%),
        ideally with a DatetimeIndex. Prices are rejected.
    benchmark : pd.Series, optional
        Benchmark simple returns on the same calendar. Enables alpha, beta and
        the alpha-decay trend test. Nothing is downloaded; pass the data.
    n_trials : int, optional
        How many strategy variants / parameter sets were tried before this one
        was chosen. Drives the deflated Sharpe ratio. Be honest and generous:
        every backtest you looked at counts. Ignored when ``trials`` is given.
    trials : pd.DataFrame, optional
        Returns of every variant tried, one column each. Enables the
        probability of backtest overfitting and supplies the real dispersion
        of Sharpe ratios across trials.
    split : date-like or float, optional
        First out-of-sample date, or the out-of-sample fraction (0.3 = last
        30%). Enables the in-sample vs out-of-sample comparison.
    periods : float, optional
        Observations per year. Inferred from the index when omitted (252, or
        365 when weekends are present); required for non-date indexes.
    rf : float, default 0.0
        Annual risk-free rate, subtracted from returns before all tests.
    confidence : float, default 0.95
        Confidence level for intervals and the minimum track record length.
    n_boot : int, default 2000
        Bootstrap replicates.
    seed : int or None, default 0
        Seed for the bootstrap. Fixed by default so results are reproducible.

    Returns
    -------
    ValidityResult
        Use ``.summary()`` for text, ``.to_dict()`` / ``.to_json()`` for
        structured output and ``.findings`` for the flagged issues.
    """
    s = _clean_returns(returns)
    if len(s) < _MIN_OBS:
        raise ValueError(
            f"Need at least {_MIN_OBS} return observations, got {len(s)}."
        )
    if periods is None:
        if not isinstance(s.index, pd.DatetimeIndex):
            raise ValueError(
                "periods (observations per year) is required when returns has "
                "no DatetimeIndex, e.g. periods=252 for daily equity data."
            )
        periods = infer_periods_per_year(s)
    periods = float(periods)
    if isinstance(s.index, pd.DatetimeIndex) and not s.index.is_monotonic_increasing:
        s = s.sort_index()
    rf_period = (1.0 + rf) ** (1.0 / periods) - 1.0
    excess = s - rf_period
    arr = excess.to_numpy()
    name = label or (str(returns.name) if getattr(returns, "name", None) else "Strategy")

    trials_std: float | None = None
    trials_source = "assumed"
    if trials is not None:
        tr = trials.dropna(how="any").astype(float) - rf_period
        tr_sd = tr.std(ddof=1)
        tr_sr = (tr.mean() / tr_sd.where(tr_sd > 0)).dropna()
        n_trials_eff = int(trials.shape[1])
        if len(tr_sr) > 1:
            trials_std = float(tr_sr.std(ddof=1))
            trials_source = "trials"
    else:
        n_trials_eff = int(n_trials) if n_trials is not None else 1
    if n_trials_eff < 1:
        raise ValueError(f"n_trials must be >= 1, got {n_trials_eff}")

    dsr = deflated_sharpe_ratio(arr, n_trials_eff, trials_std)
    m = _moments(arr)
    ann = math.sqrt(periods)
    mintrl = min_track_record_length(
        m["sharpe"], m["skew"], m["kurtosis"], confidence=confidence
    )
    mintrl_deflated = min_track_record_length(
        m["sharpe"],
        m["skew"],
        m["kurtosis"],
        sr_benchmark=dsr["sharpe_hurdle"],
        confidence=confidence,
    )
    lag1 = float(pd.Series(arr).autocorr(lag=1)) if len(arr) > 2 else float("nan")
    sharpe = {
        "sharpe": m["sharpe"],
        "sharpe_annual": m["sharpe"] * ann,
        "sharpe_std_error": dsr["sharpe_std_error"],
        "sharpe_std_error_annual": dsr["sharpe_std_error"] * ann,
        "psr": dsr["psr"],
        "dsr": dsr["dsr"],
        "n_trials": n_trials_eff,
        "trials_sharpe_std": dsr["trials_sharpe_std"],
        "sharpe_hurdle": dsr["sharpe_hurdle"],
        "sharpe_hurdle_annual": dsr["sharpe_hurdle"] * ann,
        "min_track_record_length": mintrl,
        "min_track_record_length_deflated": mintrl_deflated,
        "skew": m["skew"],
        "kurtosis": m["kurtosis"],
        "autocorrelation_lag1": lag1,
    }

    boot = bootstrap_confidence_intervals(
        arr, periods=periods, n_boot=n_boot, confidence=confidence, seed=seed
    )
    split_res = split_analysis(excess, split, periods) if split is not None else None
    bench = None
    if benchmark is not None:
        bench = _clean_returns(benchmark, "benchmark") - rf_period
    decay = decay_analysis(excess, bench, periods)
    decay.update(
        _edge_threshold(excess, bench, trials, rf_period, n_trials_eff, confidence, n_boot, seed)
    )
    over = probability_of_backtest_overfitting(trials) if trials is not None else None

    idx = s.index
    is_dt = isinstance(idx, pd.DatetimeIndex)
    sample = {
        "n": int(len(s)),
        "years": float(len(s) / periods),
        "start": idx[0] if is_dt else None,
        "end": idx[-1] if is_dt else None,
    }
    assumptions = {
        "returns": "simple per-period returns, risk-free rate subtracted",
        "rf_annual": float(rf),
        "periods_per_year": periods,
        "confidence": float(confidence),
        "n_trials_provided": n_trials is not None or trials is not None,
        "trials_sharpe_std_source": trials_source,
        "bootstrap": f"stationary, mean block {boot['mean_block']:g}, "
        f"{n_boot} replicates, seed {seed}",
        "sharpe_units": "keys ending in _annual are annualised; others per-period",
    }

    result = ValidityResult(
        label=name,
        periods=periods,
        returns=excess,
        sample=sample,
        sharpe=sharpe,
        bootstrap=boot,
        split=split_res,
        decay=decay,
        overfitting=over,
        findings=[],
        assumptions=assumptions,
    )
    result.findings = _findings(result, confidence)
    return result


def _edge_threshold(
    excess: pd.Series,
    bench: pd.Series | None,
    trials: pd.DataFrame | None,
    rf_period: float,
    n_trials: int,
    confidence: float,
    n_boot: int,
    seed: int | None,
) -> dict[str, Any]:
    """Selection-adjusted critical value for the edge t-statistic, as decay keys."""
    thr = None
    if trials is not None and n_trials > 1:
        # The selected series joins the family so the maximum always covers it;
        # if it is already a column the duplicate leaves the maximum unchanged.
        family = pd.concat(
            [trials.astype(float) - rf_period, excess.rename("__selected__")], axis=1
        )
        try:
            thr = selection_adjusted_t_threshold(
                family, bench, confidence=confidence, n_boot=min(n_boot, 1000), seed=seed
            )
        except ValueError:
            thr = None
    if thr is None:
        thr = selection_adjusted_t_threshold(n_trials=n_trials, confidence=confidence)
    return {
        "edge_t_threshold": thr["threshold"],
        "edge_t_threshold_method": thr["method"],
        "edge_t_threshold_n_trials": int(n_trials),
    }


def infer_periods_per_year(s: pd.Series) -> float:
    """
    Observations per year from the spacing of a DatetimeIndex.

    Daily data gives 252 (or 365 when weekends are present); weekly 52,
    monthly 12, quarterly 4, yearly 1; intraday scales the daily figure by
    the median number of bars per day. Non-date indexes give 252.
    """
    from ..montecarlo.core import infer_periods_per_year as _daily

    daily_default = _daily(s)
    if not isinstance(s.index, pd.DatetimeIndex) or len(s) < 3:
        return daily_default
    days = float(np.median(np.diff(s.index.values).astype("timedelta64[s]").astype(float)))
    days /= 86400.0
    if days <= 0:
        return daily_default
    if days < 0.9:  # intraday: scale the daily convention by bars per day
        per_day = pd.Series(1, index=s.index).groupby(s.index.normalize()).sum().median()
        return float(daily_default * per_day)
    if days <= 4:
        return daily_default
    if days <= 10:
        return 52.0
    if days <= 45:
        return 12.0
    if days <= 135:
        return 4.0
    return 1.0


def _findings(res: ValidityResult, confidence: float) -> list[Finding]:
    out: list[Finding] = []
    s, n = res.sharpe, res.sample["n"]
    add = lambda code, sev, msg: out.append(Finding(code, sev, msg))  # noqa: E731

    if not res.assumptions["n_trials_provided"]:
        add(
            "N_TRIALS_NOT_PROVIDED",
            "caution",
            "No n_trials or trials given, so the deflated Sharpe equals the "
            "probabilistic Sharpe and ignores selection bias. Pass the number "
            "of variants that were tried.",
        )
    elif res.assumptions["trials_sharpe_std_source"] == "assumed" and s["n_trials"] > 1:
        add(
            "TRIAL_DISPERSION_ASSUMED",
            "info",
            "Dispersion of Sharpe ratios across trials was not observed; it is "
            "assumed equal to pure sampling noise (independent zero-skill "
            "trials). Pass trials= with every variant's returns to measure it.",
        )

    need = s["min_track_record_length_deflated"]
    below_hurdle = not np.isfinite(need)
    if np.isfinite(s["dsr"]) and not below_hurdle:
        if s["dsr"] < confidence:
            sev = "warning" if s["dsr"] < 0.5 else "caution"
            add(
                "SHARPE_NOT_SIGNIFICANT_AFTER_SELECTION",
                sev,
                f"Deflated Sharpe is {s['dsr']:.1%}: after accounting for "
                f"{s['n_trials']} trial(s) there is less than {confidence:.0%} "
                "confidence that the true Sharpe is positive.",
            )
        else:
            add(
                "SHARPE_SIGNIFICANT_AFTER_SELECTION",
                "info",
                f"Deflated Sharpe is {s['dsr']:.1%} with {s['n_trials']} trial(s).",
            )

    if below_hurdle and s["n_trials"] > 1 and s["sharpe"] > 0:
        add(
            "SHARPE_BELOW_SELECTION_HURDLE",
            "warning",
            f"Observed Sharpe ({s['sharpe_annual']:.2f} annualised) does not exceed "
            f"the {s['sharpe_hurdle_annual']:.2f} expected from the best of "
            f"{s['n_trials']} skill-less trials (deflated Sharpe {s['dsr']:.1%}); "
            "no track record length would make it significant.",
        )
    elif below_hurdle:
        add(
            "SHARPE_NOT_POSITIVE",
            "warning",
            "Observed Sharpe is not positive (or undefined); there is no edge "
            "to validate.",
        )
    elif need > n:
        add(
            "TRACK_RECORD_TOO_SHORT",
            "caution",
            f"About {math.ceil(need):,} observations are needed to confirm this "
            f"Sharpe at {confidence:.0%}; the sample has {n:,}.",
        )

    ci = res.bootstrap["sharpe"]
    if np.isfinite(ci["low"]) and ci["low"] <= 0:
        add(
            "SHARPE_CI_INCLUDES_ZERO",
            "caution",
            f"The {confidence:.0%} bootstrap interval for the annualised Sharpe "
            f"is [{ci['low']:.2f}, {ci['high']:.2f}], which includes zero.",
        )

    lag1 = s["autocorrelation_lag1"]
    if np.isfinite(lag1) and abs(lag1) > 0.1:
        direction = "understates" if lag1 > 0 else "overstates"
        add(
            "RETURNS_AUTOCORRELATED",
            "caution",
            f"Lag-1 autocorrelation is {lag1:+.2f}; the analytic Sharpe standard "
            f"error {direction} the true uncertainty. Prefer the bootstrap interval.",
        )
    if s["skew"] < -1 or s["kurtosis"] > 10:
        add(
            "FAT_TAILS",
            "info",
            f"Skew {s['skew']:.2f}, kurtosis {s['kurtosis']:.1f}: Sharpe hides "
            "tail risk here; read it together with drawdown and the Montecarlo report.",
        )

    if res.split is None:
        add(
            "NO_OUT_OF_SAMPLE_SPLIT",
            "info",
            "No split given, so in-sample and out-of-sample performance are "
            "not separated. Pass split=<date the strategy was frozen>.",
        )
    else:
        sp = res.split
        p = sp["degradation_p_value"]
        oos = sp["out_of_sample"]
        if np.isfinite(p) and p < 1 - confidence:
            add(
                "OUT_OF_SAMPLE_DEGRADATION",
                "warning",
                f"Sharpe fell from {sp['in_sample']['sharpe_annual']:.2f} in-sample "
                f"to {oos['sharpe_annual']:.2f} out-of-sample (p={p:.3f}).",
            )
        if oos["sharpe"] <= 0:
            add(
                "OUT_OF_SAMPLE_SHARPE_NOT_POSITIVE",
                "warning",
                f"Out-of-sample Sharpe is {oos['sharpe_annual']:.2f}.",
            )
        elif oos["psr"] < confidence:
            add(
                "OUT_OF_SAMPLE_INCONCLUSIVE",
                "caution",
                f"Out-of-sample Sharpe is positive ({oos['sharpe_annual']:.2f}) but "
                f"its probabilistic Sharpe is {oos['psr']:.1%} over {oos['n']} "
                "observations — not enough on its own to confirm the edge.",
            )

    d = res.decay
    what = "Alpha" if d["kind"] == "alpha" else "Mean return"
    t_edge = d["edge_t_stat"]
    t_single = float(norm.ppf(1.0 - (1.0 - confidence) / 2.0))
    t_bar = d.get("edge_t_threshold", t_single)
    if d["kind"] == "alpha" and np.isfinite(t_edge) and abs(t_edge) < t_single:
        add(
            "ALPHA_NOT_SIGNIFICANT",
            "caution",
            f"Alpha of {d['edge_annual']:.2%}/year has a t-statistic of "
            f"{d['edge_t_stat']:.2f}; performance is not distinguishable from "
            f"benchmark exposure (beta {d['beta']:.2f}).",
        )
    elif d["kind"] == "alpha" and np.isfinite(t_edge) and t_single <= t_edge < t_bar:
        how = (
            "max-t bootstrap over the trials"
            if d.get("edge_t_threshold_method") == "max_t_bootstrap"
            else "Šidák bound for independent trials, conservative if they are correlated"
        )
        add(
            "ALPHA_NOT_SIGNIFICANT_AFTER_SELECTION",
            "caution",
            f"Alpha of {d['edge_annual']:.2%}/year has a t-statistic of "
            f"{t_edge:.2f}. That clears the single-test {t_single:.2f} but not the "
            f"{t_bar:.2f} required for the best of {s['n_trials']} trials ({how}).",
        )
    if np.isfinite(d["trend_t_stat"]) and d["trend_t_stat"] < -t_single:
        add(
            "EDGE_DECAYING",
            "warning",
            f"{what} is falling by {abs(d['trend_annual_per_year']):.2%} per year "
            f"(t={d['trend_t_stat']:.2f}).",
        )

    if res.overfitting is not None:
        pbo = res.overfitting["pbo"]
        base = (
            f"PBO is {pbo:.1%}: in that share of train/test splits the best "
            "in-sample variant ranked in the bottom half out-of-sample."
        )
        if pbo >= PBO_NO_INFORMATION:
            sev = "warning"
            base += " In-sample rank carries no information about out-of-sample rank."
        elif pbo > PBO_REJECT_ABOVE:
            sev = "caution"
            base += (
                f" Inconclusive: above the {PBO_REJECT_ABOVE:.0%} rejection level of "
                "Bailey et al. (2017), and a single-sample PBO in this range is "
                "common both with and without a skilled variant."
            )
        else:
            sev = "info"
        add("PROBABILITY_OF_BACKTEST_OVERFITTING", sev, base)
    return out
