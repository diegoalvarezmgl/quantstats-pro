# QuantStats Pro — guide for AI agents

You are using `quantstats-pro` (import name `quantstats`) to analyse the
returns of a trading strategy, fund or asset on someone's behalf. This guide
tells you which call answers which question, what the inputs must look like,
and how to report the results without overstating them.

Print this guide any time with `python -m quantstats guide`.

## 1. Input contract

Almost every mistake with this library is an input mistake. Check these
before calling anything.

| Rule | Detail |
| --- | --- |
| Returns, not prices | Functions expect **simple per-period returns as fractions** (`0.01` = +1%). Convert prices with `prices.pct_change().dropna()` or `qs.utils.to_returns(prices)`. `quantstats.validity` rejects input that is obviously prices, but cannot catch every case (e.g. sub-$1 prices), and the legacy `qs.stats.*` functions guess and can guess wrong. Always convert first. |
| One strategy per Series | Pass a `pd.Series` with a `DatetimeIndex`, sorted ascending, no duplicate dates. |
| Net of costs | Use returns after fees, slippage and financing. Gross backtest returns make every statistic here optimistic. Ask the user if you do not know. |
| Frequency | Annualisation uses `periods` / `periods_per_year`: 252 daily equities, 365 daily crypto, 52 weekly, 12 monthly. `quantstats.validity` and every CLI command infer it from the index spacing (`quantstats.validity.infer_periods_per_year`). `qs.stats.*`, `qs.reports.html` and `alphadecay.analyze` default to 252, and the Montecarlo engine only tells 252 from 365, so pass it explicitly in Python for anything that is not daily. |
| Benchmark | Pass benchmark **returns** as a Series on the same calendar. Passing a ticker string to `qs.reports.html` triggers a Yahoo Finance download; avoid that unless the user asked for it. |
| No trade data | Everything is computed from period returns. "Win rate" is the share of positive periods, not of winning trades. |

## 2. Which call answers which question

| The user wants to know… | Call | Output |
| --- | --- | --- |
| Is this backtest real or overfit? | `quantstats.validity.validate(returns, benchmark, n_trials=…, split=…)` | `ValidityResult` → `.to_dict()`, `.findings`, `.summary()` |
| Same, as a shareable page | `qs.reports.html_validity(returns, …, output="validity.html")` | HTML file, returns the `ValidityResult` |
| Headline performance numbers | `quantstats.cli.compute_metrics(returns, benchmark)` or individual `qs.stats.*` | flat dict of floats |
| Is live performance drifting from its history? | `quantstats.alphadecay.analyze(returns, benchmark=…, windows=(21, 63, 126))` | `DecayResult` → `.to_dict()` |
| What could the next year look like? | `quantstats.montecarlo.run_models(returns, bust=-0.25, goal=0.5, seed=0)` | dict of `ModelResult`, each with `.summary` |
| A full performance tearsheet | `qs.reports.html(returns, benchmark, output="report.html")` | HTML file |

Always pass `output=` to report functions. Without it they open a browser.

## 3. Validating a backtest

```python
import pandas as pd
from quantstats import validity

returns = pd.read_csv("strategy.csv", index_col=0, parse_dates=True)["ret"]
bench = pd.read_csv("spy.csv", index_col=0, parse_dates=True)["ret"]

res = validity.validate(
    returns,
    bench,
    n_trials=50,          # variants tried before choosing this one
    split="2023-01-01",   # date the strategy was frozen / went live
)
print(res.summary())
data = res.to_dict()      # strict JSON: no NaN, no numpy types
```

Arguments that change the answer — get them from the user, do not invent them:

- **`n_trials`** — how many parameter sets, signals or variants were tested
  before this one was picked. It is the single most important input. If the
  user does not know, ask for an order of magnitude and say that you used it.
  With no value, the deflated Sharpe ignores selection bias and the result
  carries the finding `N_TRIALS_NOT_PROVIDED`.
- **`trials`** — a DataFrame with the returns of *every* variant tried, one
  column each. Better than `n_trials`: it measures the real dispersion between
  variants and enables the probability of backtest overfitting.
- **`split`** — the first date the strategy could not have been tuned on. Never
  choose it after looking at the results.

### Reading the result

| Key (`res.to_dict()`) | Meaning | How to read it |
| --- | --- | --- |
| `sharpe.sharpe_annual` | Observed annualised Sharpe | A point estimate. Always quote it with `bootstrap.sharpe.low/high`. |
| `sharpe.psr` | P(true Sharpe > 0), one trial | Ignores selection. Use `dsr` when variants were tried. |
| `sharpe.dsr` | P(true Sharpe > best of `n_trials` worthless strategies) | ≥ 0.95 is the conventional bar. Below it the Sharpe is not distinguishable from luck. |
| `sharpe.sharpe_hurdle_annual` | Sharpe the best of N skill-less trials would show | If the observed Sharpe is below this, no amount of data rescues it. |
| `sharpe.min_track_record_length_deflated` | Observations needed for significance | Compare with `sample.n`. `null` means infinite. |
| `bootstrap.<stat>.low/high` | Confidence interval (stationary bootstrap) | An interval for max drawdown is usually much wider than users expect; say so. |
| `split.degradation_p_value` | P(seeing this OOS drop by chance) | < 0.05 → performance broke after the split. |
| `decay.edge_annual`, `decay.edge_t_stat` | Alpha (with benchmark) or mean return | Compare with `decay.edge_t_threshold`, not with 2. Below it the edge is not distinguishable from zero / from benchmark exposure. |
| `decay.edge_t_threshold`, `decay.edge_t_threshold_method` | Critical value for `edge_t_stat` given the trials: `single_test` (1.96), `sidak` (only `n_trials` known; conservative) or `max_t_bootstrap` (`trials` passed) | With 60 trials it is near 3, so t = 2.3 on a selected strategy is not evidence of alpha. |
| `decay.trend_annual_per_year`, `decay.trend_t_stat` | Change in the edge per year | t < −2 → the edge is decaying. |
| `overfitting.pbo` | Share of train/test splits where the in-sample winner is below median out-of-sample | ≤ 0.05 → passes the rejection level of Bailey et al. (2017). ≥ 0.5 → selection adds no information. Anything in between is inconclusive: do not report it as a pass. |
| `findings[]` | `{code, severity, message}` with severity `info` / `caution` / `warning` | Report every `warning` and `caution`. Codes are stable identifiers. |

Keys ending in `_annual` are annualised; other Sharpe values are per-period.

### Rules for reporting

1. Lead with the findings, not the Sharpe. A Sharpe of 2 with
   `SHARPE_NOT_SIGNIFICANT_AFTER_SELECTION` is not a good result.
2. Never describe a strategy as "validated", "robust" or "statistically
   significant" unless `dsr ≥ confidence` **and** there is no `warning`.
3. State the assumptions you supplied (`n_trials`, `split`, costs, benchmark).
   They are echoed in `assumptions`.
4. These tests cannot detect look-ahead bias, survivorship bias, unrealistic
   fills or data errors. A clean result means "not obviously luck", nothing
   more. Say this when the user is about to act on the numbers.
5. Do not give investment advice. Describe the evidence.

## 4. Monitoring a live strategy

```python
from quantstats import alphadecay

res = alphadecay.analyze(returns, benchmark=bench, windows=(21, 63, 126))
d = res.to_dict()
d["edge_test"]                           # reference period vs since: status, p_value
d["health"]["score_pct"]                 # share of metric-window cells rated good
d["metrics"]["volatility"]["windows"]    # per-window observed value, z-score, status
```

- With `benchmark`, the monitor runs on `r − β·benchmark`, i.e. on alpha.
  Without it, it tracks total performance and a market sell-off will light it up.
- The default windows `(7, 15, 30)` are very noisy; prefer `(21, 63, 126)`
  for daily data.
- 10 metrics × 3 windows are 30 comparisons. A few red cells are expected by
  chance; look for a cluster that persists.
- **Read `d["edge_test"]` first.** It compares the reference period (first
  half by default, `baseline=0.5`) with everything since, one-sided on the
  Sharpe ratio: `status` is `ok`, `warning` (p < 0.05) or `critical`
  (p < 0.01). This is the monitor's statistical statement about decay.
- A health score of 100% does **not** mean the edge is intact. The cells look
  at the latest windows only; they react to changes in risk and are too short
  to detect a change in average return.
- `edge_test` needs time: a decay that began in the last 10–20% of the sample
  is usually not detected yet. `ok` means "no evidence so far".
- Do not report `cusum.alarm` as evidence of decay: with the default
  parameters it reflects only the latest day and has little power on daily
  returns.

## 5. Forward simulation

```python
from quantstats.montecarlo import run_models, analytics

results = run_models(returns, sims=1000, bust=-0.25, goal=0.5, seed=0)
results["garch"].summary                     # cagr_p5, maxdd_p95, bust_prob, …
analytics.model_median_summary(results)      # consensus across models
analytics.conservative_envelope(results)     # worst case across models
```

Simulations extrapolate the historical return distribution. If the validity
check says the history is not credible, neither are the simulations. Pass
`seed` so results are reproducible.

## 6. Command line

Every command prints one JSON object to stdout. Errors go to stderr as
`{"error": {"type", "message"}}` with exit code 2. Nothing opens a browser or
touches the network.

```bash
quantstats validate returns.csv --n-trials 50 --split 2023-01-01 --benchmark spy.csv
quantstats validate returns.csv --trials all_variants.csv --html validity.html
quantstats metrics returns.csv --benchmark spy.csv --periods 252
quantstats decay returns.csv --benchmark spy.csv --windows 21,63,126
quantstats montecarlo returns.csv --sims 1000 --bust -0.25 --goal 0.5
quantstats report returns.csv --kind full --output report.html
```

Input files: CSV or Parquet with a date column first, then one or more numeric
columns. Use `--column` when there are several, and `--prices` when the file
holds prices rather than returns. `python -m quantstats …` is equivalent.

## 7. Common mistakes

- Feeding prices, cumulative returns or percentages (`1.0` meaning 1%) as returns.
- Leaving `n_trials` out and quoting the deflated Sharpe as if selection had
  been accounted for.
- Picking `split` to make the out-of-sample period look good.
- Annualising crypto, weekly or monthly data with the 252 default.
- Calling a report function without `output=` in a headless session.
- Comparing a strategy with a benchmark on a different calendar; align on
  shared dates first.
- Installing alongside the original `quantstats` package: both provide the
  `quantstats` import name. Uninstall one.
