# Drop-in replacement with `quantstats` import

QuantStats Pro is a drop-in replacement for upstream QuantStats. Users swap `pip install quantstats` → `pip install quantstats-pro` and keep `import quantstats as qs` unchanged. The PyPI package name is `quantstats-pro` to avoid conflicting with the original on PyPI.

The reports and plots layers may diverge visually; the stats layer keeps stable function signatures. Formula corrections that change computed values are allowed as patch releases with CHANGELOG notes.

**Considered:** Separate import (`quantstats_pro`) for stronger product identity — rejected because migration friction outweighs the benefit when reports/plots can already change freely under drop-in semantics.
