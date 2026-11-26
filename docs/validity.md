# Backtest Validity

`quantstats.validity` answers one question about a return series: **how much of
this performance should I believe?** A backtest's Sharpe ratio is an estimate,
and usually the best of many attempts. This module measures the uncertainty of
the estimate, discounts it for the number of variants tried, and checks whether
the edge survives out-of-sample and over time.

## Quick start

```python
import quantstats as qs
from quantstats import validity

res = validity.validate(
    returns,                 # simple per-period returns (prices are rejected)
    benchmark,               # optional: benchmark returns → alpha and alpha decay
    n_trials=50,             # variants tried before choosing this one
    split="2023-01-01",      # first out-of-sample date (or 0.3 = last 30%)
)

print(res.summary())
res.to_dict()                # strict-JSON summary
res.findings                 # list of Finding(code, severity, message)
```

HTML tearsheet:

```python
qs.reports.html_validity(
    returns, benchmark, n_trials=50, split="2023-01-01", output="validity.html"
)
```

Command line:

```bash
quantstats validate returns.csv --benchmark spy.csv --n-trials 50 --split 2023-01-01
```

## What it computes

| Block | Question | Needs |
| --- | --- | --- |
| Probabilistic Sharpe ratio | Is the true Sharpe above zero, given sample length, skew and fat tails? | — |
| Deflated Sharpe ratio | …and above the best Sharpe expected from N worthless trials? | `n_trials` or `trials` |
| Minimum track record length | How many observations would make this Sharpe significant? | — |
| Bootstrap confidence intervals | How wide is the plausible range of Sharpe, CAGR, volatility and max drawdown? | — |
| In-sample vs out-of-sample | Did performance break after the strategy was frozen? | `split` |
| Alpha and decay trend | Is there alpha beyond benchmark exposure, and is it shrinking? | `benchmark` (without it: mean return) |
| Probability of backtest overfitting | How often does the in-sample winner end up below median out-of-sample? | `trials` |

### The two inputs that matter most

**`n_trials`** is the number of parameter sets, signals or variants you looked
at before settling on this one. The best of 100 random strategies has an
impressive Sharpe by construction; the deflated Sharpe ratio subtracts that
effect. Be generous: every backtest you looked at counts. When it is omitted
the result carries the finding `N_TRIALS_NOT_PROVIDED`.

**`trials`** is better still: a DataFrame with the returns of every variant,
one column each. It supplies the real dispersion of Sharpe ratios across
variants (instead of assuming it), enables the probability of backtest
overfitting, and lets the alpha t-statistic be compared with the largest
t-statistic the whole family would produce by chance.

### Alpha of a selected strategy

A Newey–West t-statistic of alpha is compared with 1.96 when one strategy was
tested. For the best of several, `decay.edge_t_threshold` holds the critical
value that keeps the same one-sided 2.5% error rate:

| Input | `edge_t_threshold_method` | How |
| --- | --- | --- |
| one trial | `single_test` | Normal quantile, 1.96 at 95% confidence |
| `n_trials=N` | `sidak` | Normal quantile at `0.975^(1/N)`; 3.34 for N = 60. Assumes independent trials, so it is conservative when they are correlated |
| `trials=df` | `max_t_bootstrap` | Studentised max-t stationary bootstrap in the spirit of White (2000) and Romano & Wolf (2005): trials and benchmark are resampled with the same time indices, each replicate's t-statistics are recentred on the observed estimates, and the threshold is the 97.5% quantile of their maximum |

`ALPHA_NOT_SIGNIFICANT_AFTER_SELECTION` is raised when the t-statistic clears
1.96 but not that threshold.

### Reading PBO

Bailey, Borwein, López de Prado and Zhu (2017) treat PBO as a type I error
probability and reject above 0.05. A value of 0.5 is what an uninformative
in-sample ranking gives on average, but a single-sample PBO is widely
dispersed: in the [simulation study](articles/selection-bias.md), searches
with no skilled rule produce a PBO below 0.25 in 16% of markets and below 0.05
in 1.4%. The finding is therefore `info` at or below 0.05, `caution`
(inconclusive) between 0.05 and 0.5, and `warning` from 0.5.

When only `n_trials` is given, the dispersion of Sharpe ratios across trials is
assumed to equal the sampling noise of a zero-skill strategy over the same
track record (independent trials). Correlated variants make this conservative.

## Individual functions

```python
from quantstats import validity as v

v.deflated_sharpe_ratio(returns, n_trials=100)
v.min_track_record_length(sr=0.08, skew=-0.5, kurtosis=5)   # per-period Sharpe
v.bootstrap_confidence_intervals(returns, periods=252, n_boot=2000, seed=0)
v.split_analysis(returns, "2023-01-01", periods=252)
v.decay_analysis(returns, benchmark, periods=252)
v.probability_of_backtest_overfitting(trials_df, n_splits=16)
v.selection_adjusted_t_threshold(trials_df, benchmark)        # or n_trials=60
```

All Sharpe arithmetic is per-period (the frequency of the input). In
`ValidityResult.to_dict()`, keys ending in `_annual` are annualised.

## Findings

`res.findings` is a list of stable codes with a severity of `info`, `caution`
or `warning`:

| Code | Meaning |
| --- | --- |
| `N_TRIALS_NOT_PROVIDED` | Selection bias was not accounted for |
| `TRIAL_DISPERSION_ASSUMED` | Cross-trial Sharpe dispersion was assumed, not measured |
| `SHARPE_SIGNIFICANT_AFTER_SELECTION` | Deflated Sharpe ≥ confidence |
| `SHARPE_NOT_SIGNIFICANT_AFTER_SELECTION` | Deflated Sharpe below confidence |
| `SHARPE_BELOW_SELECTION_HURDLE` | Observed Sharpe is below what luck alone would produce |
| `SHARPE_NOT_POSITIVE` | Nothing to validate |
| `TRACK_RECORD_TOO_SHORT` | More observations needed for significance |
| `SHARPE_CI_INCLUDES_ZERO` | Bootstrap interval for the Sharpe spans zero |
| `RETURNS_AUTOCORRELATED` | Analytic standard errors are unreliable; prefer the bootstrap |
| `FAT_TAILS` | Sharpe understates tail risk |
| `NO_OUT_OF_SAMPLE_SPLIT` | No `split` supplied |
| `OUT_OF_SAMPLE_DEGRADATION` | Sharpe dropped significantly after the split |
| `OUT_OF_SAMPLE_SHARPE_NOT_POSITIVE` | Out-of-sample Sharpe ≤ 0 |
| `OUT_OF_SAMPLE_INCONCLUSIVE` | Out-of-sample Sharpe positive but too short to confirm |
| `ALPHA_NOT_SIGNIFICANT` | Alpha t-statistic below the single-test critical value: performance is explained by benchmark exposure |
| `ALPHA_NOT_SIGNIFICANT_AFTER_SELECTION` | Alpha t-statistic clears the single-test value but not the one required for the best of the trials |
| `EDGE_DECAYING` | Significant negative time trend in alpha / mean return |
| `PROBABILITY_OF_BACKTEST_OVERFITTING` | PBO value: `info` at ≤ 5%, `caution` (inconclusive) up to 50%, `warning` at ≥ 50% |

## What it cannot tell you

These are statistical tests on a return series. They cannot detect look-ahead
bias, survivorship bias, unrealistic fills, missing costs or data errors. A
clean report means "not obviously luck" — not "safe to trade".

## References

- Bailey, D. H. & López de Prado, M. (2012). *The Sharpe Ratio Efficient Frontier.* Journal of Risk.
- Bailey, D. H. & López de Prado, M. (2014). *The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting and Non-Normality.* Journal of Portfolio Management.
- Bailey, D. H., Borwein, J., López de Prado, M. & Zhu, Q. J. (2017). *The Probability of Backtest Overfitting.* Journal of Computational Finance.
- Politis, D. N. & Romano, J. P. (1994). *The Stationary Bootstrap.* JASA.
- Newey, W. K. & West, K. D. (1987). *A Simple, Positive Semi-Definite, Heteroskedasticity and Autocorrelation Consistent Covariance Matrix.* Econometrica.
