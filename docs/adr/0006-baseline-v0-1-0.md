# Baseline v0.1.0 — upstream issue triage

QuantStats Pro v0.1.0 is the starting baseline: upstream code at v0.0.81 plus fork identity. Before hunting new bugs, we triaged the issues previously tracked from upstream.

## Triage summary

| Issue | Title | Status in v0.1.0 | Verdict |
|-------|-------|------------------|---------|
| #514 | `information_ratio()` arithmetic vs geometric | Open upstream | **Not a bug.** Current code implements standard IR: `mean(Rp−Rb)/std(Rp−Rb)`. Geometric IR is a different metric (Bacon). |
| #516 | 3Y/5Y/10Y ann. return inconsistencies | Open upstream | **Not fixed.** Still uses `months=35`, `months=59`, `years=10`. Design inconsistency (~2.92y / ~4.92y / 10y), not confirmed wrong. Deferred. |
| #518 | EOY chart bug when range > 10y | Open upstream | **Unconfirmed.** Programmatic test with 11y data generates plot without error. Visual/HTML bug not reproduced. Deferred. |
| #510 | `compounded` flag on `rar()` | Open upstream | **Not fixed.** Feature request, not a bug. |
| #507 | `compounded` flag on `calmar()` | Open upstream | **Not fixed.** Feature request, not a bug. |
| #521 | Time-series risk-free rate in `html()` | Open upstream | **Not fixed.** Feature request. |
| #520 | Heatmap broken on 2nd `reports.full()` in VS Code | Open upstream | **Not fixed.** Environment-specific; needs reproduction in VS Code. |
| #493 | Trade-analysis metrics from returns | Open upstream | **By design.** Documented as period-based metrics in README. |

## Baseline health (local, 2026-06-20)

- **Tests:** 125 passed
- **Python:** 3.10–3.13 (CI matrix)
- **Version:** `0.1.0`
- **Lint:** ruff reports ~153 pre-existing issues from upstream — cleanup deferred, not part of baseline

## Bug discovery policy going forward

Do not treat upstream open issues as our backlog. For each candidate bug:

1. Reproduce with a minimal script or failing test
2. Confirm expected behaviour against a reference (textbook, paper, another library)
3. Fix only with a regression test
4. One issue = one branch = one commit

**Considered:** Continue fixing upstream issues #514→#518 in order — rejected after triage showed none are confirmed bugs ready to fix.
