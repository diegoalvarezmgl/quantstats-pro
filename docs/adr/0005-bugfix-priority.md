# Bugfix priority for v0.1.x

First bugfixes after the identity push, in order. Each gets its own branch, regression test, and commit.

1. **#514** — `information_ratio()` uses arithmetic mean instead of geometric
2. **#516** — Inconsistencies in 3Y/5Y/10Y annualized return calculations
3. **#518** — EOY returns vs benchmark on time ranges longer than 10y

Deferred: #521 (variable risk-free rate), #510 (compounded flag on rar), #520 (heatmap VS Code).
