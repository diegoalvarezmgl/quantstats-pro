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
variants (instead of assuming it) and enables the probability of backtest
overfitting.

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
| `ALPHA_NOT_SIGNIFICANT` | Performance is explained by benchmark exposure |
| `EDGE_DECAYING` | Significant negative time trend in alpha / mean return |
| `PROBABILITY_OF_BACKTEST_OVERFITTING` | PBO value (warning at ≥ 50%) |

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
