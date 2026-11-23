"""
Command-line interface for QuantStats Pro.

Designed for scripts and AI agents: every analysis command reads a CSV (or
Parquet) file, prints one JSON object to stdout and never opens a browser or
downloads data. Errors are printed to stderr as ``{"error": {...}}`` with a
non-zero exit code.

    quantstats guide
    quantstats validate returns.csv --n-trials 50 --split 2023-01-01
    quantstats metrics returns.csv --benchmark spy.csv
    quantstats decay returns.csv --windows 21,63,126
    quantstats montecarlo returns.csv --sims 1000 --bust -0.25
    quantstats report returns.csv --kind validity --output report.html
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_GUIDE = Path(__file__).parent / "AGENT_GUIDE.md"
_REPORT_KINDS = ("full", "simple", "montecarlo", "alpha-decay", "validity")


class CLIError(Exception):
    """A user-facing input problem."""


def guide_text() -> str:
    """Return the agent guide shipped with the package."""
    return _GUIDE.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------


def _read_frame(path: str):
    import pandas as pd

    p = Path(path)
    if not p.exists():
        raise CLIError(f"File not found: {path}")
    if p.suffix.lower() in (".parquet", ".pq"):
        frame = pd.read_parquet(p)
        if not isinstance(frame.index, pd.DatetimeIndex):
            frame = _index_by_first_date_column(frame)
        return frame
    return _index_by_first_date_column(pd.read_csv(p))


def _index_by_first_date_column(frame):
    """Use the first column as a DatetimeIndex when it parses as dates."""
    import pandas as pd

    if frame.shape[1] < 2:
        return frame
    first = frame.iloc[:, 0]
    if pd.api.types.is_numeric_dtype(first):
        return frame
    try:
        idx = pd.to_datetime(first, errors="raise")
    except (ValueError, TypeError):
        return frame
    out = frame.iloc[:, 1:].copy()
    out.index = pd.DatetimeIndex(idx)
    out.index.name = None
    return out.sort_index()


def load_frame(path: str, prices: bool = False):
    """Load every numeric column of a file as returns."""
    import pandas as pd

    frame = _read_frame(path)
    numeric = frame.select_dtypes("number")
    if numeric.shape[1] == 0:
        raise CLIError(
            f"{path} has no numeric columns. Expected a date column followed by "
            f"one or more columns of returns; found {list(frame.columns)}."
        )
    numeric = numeric.astype(float)
    if prices:
        numeric = numeric.pct_change().iloc[1:]
    if isinstance(numeric.index, pd.DatetimeIndex) and numeric.index.tz is not None:
        numeric.index = numeric.index.tz_convert("UTC").tz_localize(None)
    return numeric


def load_series(path: str, column: str | None = None, prices: bool = False):
    """Load one column of a file as a returns Series."""
    frame = load_frame(path, prices=prices)
    if column is not None:
        if column not in frame.columns:
            raise CLIError(
                f"Column {column!r} not found in {path}. "
                f"Numeric columns: {list(frame.columns)}."
            )
        series = frame[column]
    elif frame.shape[1] == 1:
        series = frame.iloc[:, 0]
    else:
        raise CLIError(
            f"{path} has several numeric columns {list(frame.columns)}; "
            "choose one with --column."
        )
    series = series.dropna()
    if len(series) == 0:
        raise CLIError(f"Column {series.name!r} in {path} has no numeric values.")
    return series


def _parse_split(value: str | None) -> Any:
    if value is None:
        return None
    try:
        frac = float(value)
    except ValueError:
        return value
    if not 0 < frac < 1:
        raise CLIError(
            "--split must be a date (2023-01-01) or the out-of-sample fraction "
            "between 0 and 1 (0.3)."
        )
    return frac


def _strategy(args):
    return load_series(args.file, args.column, args.prices)


def _periods(args, series) -> float:
    """--periods if given, else inferred from the index (one rule everywhere)."""
    from .validity import infer_periods_per_year

    return float(args.periods) if args.periods else infer_periods_per_year(series)


def _benchmark(args):
    if not getattr(args, "benchmark", None):
        return None
    return load_series(args.benchmark, args.benchmark_column, args.prices)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def _cmd_guide(args) -> int:
    sys.stdout.write(guide_text())
    return 0


def _cmd_validate(args) -> int:
    from . import validity
    from .validity import report

    trials = load_frame(args.trials, prices=args.prices) if args.trials else None
    result = validity.validate(
        _strategy(args),
        _benchmark(args),
        n_trials=args.n_trials,
        trials=trials,
        split=_parse_split(args.split),
        periods=args.periods,
        rf=args.rf,
        confidence=args.confidence,
        n_boot=args.n_boot,
        seed=args.seed,
    )
    payload = result.to_dict()
    if args.html:
        Path(args.html).write_text(report.render_html(result), encoding="utf-8")
        payload["html_report"] = str(Path(args.html).resolve())
    if args.text:
        print(result.summary())
    else:
        _emit(payload)
    return 0


def _cmd_metrics(args) -> int:
    _emit(compute_metrics(_strategy(args), _benchmark(args), args.periods, args.rf))
    return 0


def compute_metrics(returns, benchmark=None, periods: float | None = None, rf=0.0):
    """Headline performance metrics as a flat, JSON-safe dict (raw fractions)."""
    import pandas as pd

    from . import stats
    from .validity.core import _clean_returns, _jsonable, infer_periods_per_year

    r = _clean_returns(returns)
    p = float(periods) if periods else infer_periods_per_year(r)
    dated = isinstance(r.index, pd.DatetimeIndex)
    out: dict[str, Any] = {
        "n": len(r),
        "start": r.index[0] if dated else None,
        "end": r.index[-1] if dated else None,
        "periods_per_year": p,
        "rf_annual": rf,
        "total_return": stats.comp(r),
        "cagr": stats.cagr(r, periods=p),
        "volatility": stats.volatility(r, periods=p),
        "sharpe": stats.sharpe(r, rf=rf, periods=p),
        "sortino": stats.sortino(r, rf=rf, periods=p),
        "calmar": stats.calmar(r, periods=p),
        "max_drawdown": stats.max_drawdown(r),
        "value_at_risk_95": stats.value_at_risk(r),
        "expected_shortfall_95": stats.cvar(r),
        "skew": stats.skew(r),
        "kurtosis_excess": stats.kurtosis(r),
        "win_rate": stats.win_rate(r),
        "profit_factor": stats.profit_factor(r),
        "payoff_ratio": stats.payoff_ratio(r),
        "best_period": stats.best(r),
        "worst_period": stats.worst(r),
        "probabilistic_sharpe_ratio": stats.probabilistic_sharpe_ratio(r, rf=rf, periods=p),
    }
    if benchmark is not None:
        b = _clean_returns(benchmark, "benchmark")
        frame = pd.concat([r.rename("r"), b.rename("b")], axis=1).dropna()
        if len(frame) < 30:
            raise CLIError(
                f"Only {len(frame)} dates are shared by the strategy and the benchmark."
            )
        greeks = stats.greeks(frame["r"], frame["b"], periods=p)
        out["benchmark"] = {
            "n_shared": len(frame),
            "alpha_annual": greeks["alpha"],
            "beta": greeks["beta"],
            "information_ratio": stats.information_ratio(frame["r"], frame["b"]),
            "r_squared": stats.r_squared(frame["r"], frame["b"]),
            "benchmark_cagr": stats.cagr(frame["b"], periods=p),
            "benchmark_sharpe": stats.sharpe(frame["b"], rf=rf, periods=p),
        }
    return _jsonable(out)


def _cmd_decay(args) -> int:
    from . import alphadecay

    windows = tuple(int(w) for w in args.windows.split(","))
    returns = _strategy(args)
    result = alphadecay.analyze(
        returns,
        windows=windows,
        periods=_periods(args, returns),
        benchmark=_benchmark(args),
        baseline=args.baseline,
    )
    _emit(result.to_dict())
    return 0


def _cmd_montecarlo(args) -> int:
    from .montecarlo import analytics, run_models
    from .validity.core import _jsonable

    models = args.models.split(",") if args.models else None
    returns = _strategy(args)
    results = run_models(
        returns,
        models=models,
        horizon=args.horizon,
        sims=args.sims,
        bust=args.bust,
        goal=args.goal,
        seed=args.seed,
        periods=_periods(args, returns),
    )
    first = next(iter(results.values()))
    _emit(
        _jsonable(
            {
                "horizon": first.horizon,
                "sims": first.sims,
                "periods_per_year": first.periods,
                "bust": args.bust,
                "goal": args.goal,
                "seed": args.seed,
                "models": {name: res.summary for name, res in results.items()},
                "median_across_models": analytics.model_median_summary(results),
                "conservative_envelope": analytics.conservative_envelope(results),
            }
        )
    )
    return 0


def _cmd_report(args) -> int:
    from . import reports

    returns = _strategy(args)
    bench = _benchmark(args)
    out = str(Path(args.output))
    kind = args.kind
    periods = _periods(args, returns)
    if kind == "full":
        reports.html(returns, bench, rf=args.rf, output=out,
                     periods_per_year=int(periods))
    elif kind == "simple":
        reports.html_simple(returns, bench, rf=args.rf, output=out,
                            periods_per_year=int(periods))
    elif kind == "montecarlo":
        reports.html_montecarlo(returns, rf=args.rf, output=out, seed=args.seed,
                                periods_per_year=periods)
    elif kind == "alpha-decay":
        reports.html_alpha_decay(returns, rf=args.rf, output=out, benchmark=bench,
                                 periods_per_year=int(periods))
    else:
        reports.html_validity(
            returns,
            bench,
            n_trials=args.n_trials,
            split=_parse_split(args.split),
            rf=args.rf,
            output=out,
            periods_per_year=periods,
            seed=args.seed,
        )
    _emit(
        {"kind": kind, "output": str(Path(out).resolve()), "periods_per_year": periods}
    )
    return 0


def _emit(payload: dict) -> None:
    sys.stdout.write(json.dumps(payload, indent=2, allow_nan=False) + "\n")


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def _add_input(p: argparse.ArgumentParser, benchmark: bool = True) -> None:
    p.add_argument("file", help="CSV or Parquet: a date column, then returns")
    p.add_argument("--column", help="column to analyse (needed if several)")
    p.add_argument(
        "--prices",
        action="store_true",
        help="the files hold prices; convert to simple returns first",
    )
    p.add_argument(
        "--periods", type=float, default=None, help="observations per year (default: inferred)"
    )
    p.add_argument("--rf", type=float, default=0.0, help="annual risk-free rate (0.03 = 3%%)")
    if benchmark:
        p.add_argument("--benchmark", help="CSV/Parquet with benchmark returns")
        p.add_argument("--benchmark-column", help="benchmark column (needed if several)")


def build_parser() -> argparse.ArgumentParser:
    from .version import version

    parser = argparse.ArgumentParser(
        prog="quantstats",
        description="QuantStats Pro — portfolio analytics and backtest validation. "
        "Analysis commands print JSON; run `quantstats guide` for usage notes.",
    )
    parser.add_argument("--version", action="version", version=f"quantstats-pro {version}")
    sub = parser.add_subparsers(dest="command", required=True)

    g = sub.add_parser("guide", help="print the usage guide for AI agents")
    g.set_defaults(func=_cmd_guide)

    v = sub.add_parser("validate", help="how credible is this backtest? (JSON)")
    _add_input(v)
    v.add_argument("--n-trials", type=int, help="strategy variants tried before choosing")
    v.add_argument("--trials", help="CSV/Parquet with one returns column per variant tried")
    v.add_argument("--split", help="first out-of-sample date, or OOS fraction (0.3)")
    v.add_argument("--confidence", type=float, default=0.95)
    v.add_argument("--n-boot", type=int, default=2000)
    v.add_argument("--seed", type=int, default=0)
    v.add_argument("--html", help="also write the validity tearsheet to this path")
    v.add_argument("--text", action="store_true", help="print a text summary, not JSON")
    v.set_defaults(func=_cmd_validate)

    m = sub.add_parser("metrics", help="headline performance metrics (JSON)")
    _add_input(m)
    m.set_defaults(func=_cmd_metrics)

    d = sub.add_parser("decay", help="rolling drift / alpha-decay monitor (JSON)")
    _add_input(d)
    d.add_argument("--windows", default="21,63,126", help="rolling windows, comma-separated")
    d.add_argument(
        "--baseline", type=float, default=0.5, help="leading fraction used as reference"
    )
    d.set_defaults(func=_cmd_decay)

    mc = sub.add_parser("montecarlo", help="multi-model forward simulation (JSON)")
    _add_input(mc, benchmark=False)
    mc.add_argument("--models", help="comma-separated model names (default: all)")
    mc.add_argument("--horizon", type=int, help="periods to simulate (default: one year)")
    mc.add_argument("--sims", type=int, default=1000)
    mc.add_argument("--bust", type=float, help="drawdown threshold, e.g. -0.25")
    mc.add_argument("--goal", type=float, help="terminal return target, e.g. 0.5")
    mc.add_argument("--seed", type=int, default=0)
    mc.set_defaults(func=_cmd_montecarlo)

    r = sub.add_parser("report", help="write an HTML tearsheet to a file")
    _add_input(r)
    r.add_argument("--kind", choices=_REPORT_KINDS, default="full")
    r.add_argument("--output", required=True, help="path of the HTML file to write")
    r.add_argument("--n-trials", type=int, help="(validity) variants tried")
    r.add_argument("--split", help="(validity) first out-of-sample date or fraction")
    r.add_argument("--seed", type=int, default=0)
    r.set_defaults(func=_cmd_report)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command != "guide":
        import matplotlib

        matplotlib.use("Agg", force=True)
    try:
        return int(args.func(args))
    except Exception as exc:  # the contract is JSON on stderr, never a traceback
        message = exc.args[0] if isinstance(exc, KeyError) and exc.args else str(exc)
        err = {"error": {"type": type(exc).__name__, "message": str(message)}}
        sys.stderr.write(json.dumps(err) + "\n")
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
