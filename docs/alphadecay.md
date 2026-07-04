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
