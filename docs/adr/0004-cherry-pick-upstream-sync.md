# Selective cherry-pick sync with upstream

QuantStats Pro syncs with upstream via selective cherry-picks, not periodic merges. When upstream publishes a fix worth adopting, fetch, evaluate, and cherry-pick it as an explicit commit. Blind merges are avoided to protect our diverging reports/plots work.

**Considered:** Independent (miss useful upstream fixes) or periodic merge (high conflict cost as we diverge).
