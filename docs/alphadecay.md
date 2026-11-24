# Alpha Decay Monitor

QuantStats Pro includes a short-horizon **alpha decay** monitor for single assets.
It tracks whether recent rolling risk metrics are drifting from their historical
distribution — useful for spotting strategy degradation before it shows up in
long-horizon tearsheets.

## Quick Start

```python
import quantstats as qs
from quantstats.alphadecay import analyze, available_metrics

returns = qs.utils.download_returns("QQQ")

print(available_metrics())
# ['cagr', 'volatility', 'downside_vol', 'max_drawdown', ...]

result = analyze(returns, windows=(7, 15, 30), rf=0.0, periods=252)

print(f"Health: {result.score}/{result.total} ({result.score_pct:.0f}%)")
for metric in result.metrics:
    wr = metric.windows[30]
    print(f"{metric.spec.label} (30d): z={wr.z_score:+.2f} → {wr.status}")
```

## HTML Tearsheet

```python
qs.reports.html_alpha_decay(
    returns,
    windows=(7, 15, 30),
    output="qqq_alpha_decay.html",
)
```

The tearsheet includes:

- **Health Summary** — aggregate score and traffic-light counts (Excellent / Good / Warning / Critical)
- **Summary table** — z-scores per metric and window
- **Metric distributions** — historical rolling distribution with current observation marked
- **CUSUM** return-decay detector
- **Time underwater** duration analysis

## Metrics

| Key | Label | Higher is better |
|-----|-------|------------------|
| `cagr` | CAGR | Yes |
| `volatility` | Volatility | No |
| `downside_vol` | Downside Vol | No |
| `max_drawdown` | Max Drawdown | No |
| `mean_drawdown` | Mean Drawdown | No |
| `win_rate` | Win Rate | Yes |
| `var_95` | VaR 95% | No |
| `cvar_95` | Expected Shortfall 95% | No |
| `payoff_ratio` | Payoff Ratio | Yes |
| `skew` | Skew | Yes |

Z-scores are computed on a per-metric **analysis scale** (log / log-severity
transforms for skewed metrics) so traffic lights are statistically meaningful.

The reference distribution for each cell excludes the rolling values that
overlap the current window, so the latest reading is not compared with itself.

## Reference period and benchmark (0.5.0+)

```python
result = analyze(returns, benchmark=bench, baseline=0.5, windows=(21, 63, 126))
result.to_dict()   # JSON-serialisable summary
```

- `baseline` — leading fraction of the sample used as the reference period for
  the edge test and the CUSUM target mean and σ (default `0.5`). `None` uses the full sample
  (behaviour before 0.5.0), which lets a decayed strategy lower its own target.
- `benchmark` — benchmark returns. The monitor then runs on the active stream
  `r − β·benchmark` (β estimated on the reference period), so it tracks alpha
  instead of total performance.
- With 10 metrics × 3 windows, a few Warning cells are expected by chance.
  Look for clusters that persist. The CUSUM chart is a visual aid: with the
  default `k`/`h` its alarm flag has little power on daily returns. Windows of 7–15 days are noisy;
  `(21, 63, 126)` is a steadier choice for daily data.

## Edge test (0.5.1+)

```python
et = result.edge_test
et.status        # "ok", "warning" (p < 0.05), "critical" (p < 0.01)
et.message       # one-sentence reading
et.reference_sharpe_annual, et.recent_sharpe_annual, et.p_value
```

The rolling cells look only at the latest windows. Over 126 days a strategy
with 10% volatility has a standard error of about 14% a year on its mean
return, so the cells cannot tell a lost edge from noise; a strategy whose
alpha vanished years ago can score 30/30.

The edge test answers that question directly. It compares the Sharpe ratio of
the reference period with the Sharpe ratio of everything after it (one-sided
test, same statistic as `validity.split_analysis`), on the monitored stream —
alpha when a benchmark is given. It is shown at the top of the tearsheet.

Measured on simulated daily strategies (10 years, 9.5% active volatility,
20%/year alpha):

| Scenario | Flagged (p < 0.05) |
| --- | --- |
| No change | 4–6% |
| Alpha drops to −5%/year at 50–60% of the sample | 96–99% |
| Alpha drops to 7.5%/year at 50% of the sample | 68% |
| Alpha drops to −5%/year in the last 20% | 53% |
| Alpha drops to −5%/year in the last 10% | 19% |

Recent decay needs time to show. `ok` means no evidence so far, not that the
edge is intact. With `baseline=None` or fewer than 30 observations on either
side the test is not run (`status` is `no_reference` / `insufficient_data`).

β is fitted on the reference period only. If the exposure to the benchmark
changes afterwards, the monitored stream carries benchmark return and the
verdict can be wrong in either direction.

For the trend of alpha over time, see [Backtest validity](validity.md)
(`validity.decay_analysis`).

## Traffic lights

| Status | Meaning |
|--------|---------|
| Excellent | Well above historical norm (for higher-is-better metrics) or well below severity (for risk metrics) |
| Good | Within acceptable range |
| Warning | Drifting from historical norm |
| Critical | Significant deviation |

## API Reference

### `alphadecay.analyze(returns, windows=(7, 15, 30), rf=0.0, periods=252, ...)`

Returns a `DecayResult` with:

- `score`, `total`, `score_pct` — health score (Good + Excellent cells)
- `metrics` — list of `MetricResult`, each with per-window `WindowResult`
- `cusum` — CUSUM decay detector state
- `time_underwater` — underwater duration diagnostics

### `reports.html_alpha_decay(returns, windows=(7, 15, 30), ...)`

Generate the full HTML tearsheet. See `help(qs.reports.html_alpha_decay)` for parameters.
