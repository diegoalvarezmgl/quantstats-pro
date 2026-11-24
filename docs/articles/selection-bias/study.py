"""
Selection bias in a moving-average crossover search: a simulation study.

Reproduces every number and figure in ``docs/articles/selection-bias.md``.

    python docs/articles/selection-bias/study.py            # 1,000 markets
    python docs/articles/selection-bias/study.py --markets 200

Each simulated market is 2,520 daily returns drawn from a scaled Student-t(5).
On each market 60 long/flat moving-average crossovers are evaluated, the one
with the best Sharpe ratio on the first 70% of the sample is selected, and the
selected rule is then examined with the tests in ``quantstats.validity``.

Three scenarios:

``null``     zero drift. No rule has a non-zero true Sharpe ratio.
``drift``    drift of 3 bp/day. Long/flat rules earn part of the market
             premium but have no timing ability.
``skill``    zero drift, plus one extra rule with genuine predictive power
             (true annualised Sharpe close to 1) among the 60.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from quantstats import validity

N_OBS = 2520
PERIODS = 252.0
IS_FRACTION = 0.7
FAST = (3, 5, 8, 10, 15, 20)
SLOW = (30, 40, 50, 60, 80, 100, 120, 150, 200, 250)
SKILL_NOISE = 12.5  # noise-to-signal ratio giving a true annual Sharpe near 1
FIXED_RULE = "ma_10_40"  # control rule, chosen before looking at any data
HERE = Path(__file__).parent

INK, SIGNAL, MUTED, RULE, WASH = "#14171a", "#c8321e", "#5c646c", "#d9dcde", "#f1f2f0"


def simulate_market(rng: np.random.Generator, drift: float) -> pd.Series:
    idx = pd.bdate_range("2015-01-01", periods=N_OBS)
    return pd.Series(rng.standard_t(5, N_OBS) * 0.008 + drift, index=idx, name="market")


def crossover_trials(market: pd.Series) -> pd.DataFrame:
    """Returns of the 60 long/flat crossovers; positions use yesterday's signal."""
    price = (1.0 + market).cumprod()
    out = {}
    for fast in FAST:
        fast_ma = price.rolling(fast).mean()
        for slow in SLOW:
            signal = (fast_ma > price.rolling(slow).mean()).astype(float)
            out[f"ma_{fast}_{slow}"] = signal.shift(1).fillna(0.0) * market
    return pd.DataFrame(out)


def skilled_rule(market: pd.Series, rng: np.random.Generator) -> pd.Series:
    """Long/short rule whose position is a noisy observation of today's return."""
    noise = rng.normal(0.0, SKILL_NOISE * market.std(), len(market))
    return pd.Series(np.sign(market.to_numpy() + noise) * market.to_numpy(), index=market.index)


def annual_sharpe(x: pd.Series | np.ndarray) -> float:
    arr = np.asarray(x, dtype=float)
    sd = arr.std(ddof=1)
    return float(arr.mean() / sd * np.sqrt(PERIODS)) if sd > 0 else 0.0


def examine(market: pd.Series, trials: pd.DataFrame) -> dict[str, float]:
    split = int(len(market) * IS_FRACTION)
    is_sharpe = trials.iloc[:split].mean() / trials.iloc[:split].std(ddof=1)
    best = is_sharpe.idxmax()
    chosen = trials[best]

    per_period = trials.mean() / trials.std(ddof=1)
    dsr = validity.deflated_sharpe_ratio(
        chosen, n_trials=trials.shape[1], trials_sharpe_std=float(per_period.std(ddof=1))
    )
    sp = validity.split_analysis(chosen, market.index[split], PERIODS)
    edge = validity.decay_analysis(chosen, market, PERIODS)
    pbo = validity.probability_of_backtest_overfitting(trials)

    # Variant: select on the full sample, so DSR is computed on the data used to choose.
    full_winner = trials[per_period.idxmax()]
    dsr_full = validity.deflated_sharpe_ratio(
        full_winner, n_trials=trials.shape[1], trials_sharpe_std=float(per_period.std(ddof=1))
    )
    corr = trials.corr().to_numpy()
    mean_corr = float(corr[np.triu_indices_from(corr, k=1)].mean())

    # Control: one rule fixed in advance, so no selection takes place.
    fixed = trials[FIXED_RULE]
    fixed_psr = validity.deflated_sharpe_ratio(fixed, n_trials=1)["psr"]
    fixed_split = validity.split_analysis(fixed, market.index[split], PERIODS)
    return {
        "selected": best,
        "dsr_full_sample_selection": dsr_full["dsr"],
        "hurdle_annual": dsr["sharpe_hurdle"] * np.sqrt(PERIODS),
        "mean_pairwise_corr": mean_corr,
        "fixed_psr": fixed_psr,
        "fixed_alpha_t": validity.decay_analysis(fixed, market, PERIODS)["edge_t_stat"],
        "fixed_sharpe_is": fixed_split["in_sample"]["sharpe_annual"],
        "fixed_sharpe_oos": fixed_split["out_of_sample"]["sharpe_annual"],
        "fixed_degradation_p": fixed_split["degradation_p_value"],
        "selected_is_skilled": float(best == "skilled"),
        "sharpe_is": sp["in_sample"]["sharpe_annual"],
        "sharpe_oos": sp["out_of_sample"]["sharpe_annual"],
        "sharpe_full": annual_sharpe(chosen),
        "market_sharpe_full": annual_sharpe(market),
        "psr": dsr["psr"],
        "dsr": dsr["dsr"],
        "degradation_p": sp["degradation_p_value"],
        "alpha_t": edge["edge_t_stat"],
        "pbo": pbo["pbo"],
    }


def run_scenario(name: str, markets: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(markets):
        market = simulate_market(rng, drift=0.0003 if name == "drift" else 0.0)
        trials = crossover_trials(market)
        if name == "skill":
            # Replace one crossover so the search still covers 60 rules.
            trials = trials.drop(columns=trials.columns[-1])
            trials["skilled"] = skilled_rule(market, rng)
        rows.append(examine(market, trials))
    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame) -> dict[str, float]:
    skilled = df[df["selected_is_skilled"] == 1.0]
    conditional = (
        {
            "share_dsr_ge_95_given_skilled_selected": float((skilled["dsr"] >= 0.95).mean()),
            "share_degradation_p_lt_05_given_skilled_selected": float(
                (skilled["degradation_p"] < 0.05).mean()
            ),
        }
        if len(skilled)
        else {}
    )
    return {
        **conditional,
        "markets": int(len(df)),
        "selected_is_skilled": float(df["selected_is_skilled"].mean()),
        "median_sharpe_is": float(df["sharpe_is"].median()),
        "median_sharpe_oos": float(df["sharpe_oos"].median()),
        "median_sharpe_full": float(df["sharpe_full"].median()),
        "median_market_sharpe": float(df["market_sharpe_full"].median()),
        "share_oos_below_is": float((df["sharpe_oos"] < df["sharpe_is"]).mean()),
        "share_psr_ge_95": float((df["psr"] >= 0.95).mean()),
        "share_dsr_ge_95": float((df["dsr"] >= 0.95).mean()),
        "share_alpha_t_gt_196": float((df["alpha_t"] > 1.96).mean()),
        "share_degradation_p_lt_05": float((df["degradation_p"] < 0.05).mean()),
        "fixed_share_psr_ge_95": float((df["fixed_psr"] >= 0.95).mean()),
        "fixed_share_alpha_t_gt_196": float((df["fixed_alpha_t"] > 1.96).mean()),
        "fixed_share_degradation_p_lt_05": float((df["fixed_degradation_p"] < 0.05).mean()),
        "fixed_median_sharpe_is": float(df["fixed_sharpe_is"].median()),
        "fixed_median_sharpe_oos": float(df["fixed_sharpe_oos"].median()),
        "share_dsr_ge_95_full_sample_selection": float((df["dsr_full_sample_selection"] >= 0.95).mean()),
        "median_hurdle_annual": float(df["hurdle_annual"].median()),
        "median_pairwise_corr": float(df["mean_pairwise_corr"].median()),
        "median_pbo": float(df["pbo"].median()),
        "share_pbo_ge_50": float((df["pbo"] >= 0.5).mean()),
    }


def figures(results: dict[str, pd.DataFrame]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def style(ax):
        ax.grid(False)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(RULE)
        ax.tick_params(colors=MUTED, labelsize=9, length=0)
        ax.xaxis.label.set_color(MUTED)
        ax.yaxis.label.set_color(MUTED)

    null = results["null"]
    fig, ax = plt.subplots(figsize=(8, 3.2))
    bins = np.linspace(-2.0, 2.5, 46)
    ax.hist(null["sharpe_is"], bins=bins, color=WASH, label="In-sample (first 70%)")
    ax.hist(null["sharpe_is"], bins=bins, histtype="step", color=INK, lw=1.4)
    ax.hist(null["sharpe_oos"], bins=bins, histtype="step", color=SIGNAL, lw=1.4, label="Out-of-sample (last 30%)")
    ax.axvline(0, color=MUTED, lw=1)
    ax.set_xlabel("Annualised Sharpe of the selected rule, null scenario")
    ax.set_ylabel("Markets")
    ax.legend(frameon=False, fontsize=9)
    style(ax)
    fig.tight_layout()
    fig.savefig(HERE / "fig_is_vs_oos.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 3.2))
    bins = np.linspace(0, 1, 41)
    ax.hist(null["pbo"], bins=bins, color=WASH, label="Null: no rule has an edge")
    ax.hist(null["pbo"], bins=bins, histtype="step", color=INK, lw=1.4)
    ax.hist(results["skill"]["pbo"], bins=bins, histtype="step", color=MUTED, lw=1.4, ls="--", label="Positive control: one rule with real skill")
    ax.axvline(0.5, color=SIGNAL, lw=1.2)
    ax.set_xlabel("Probability of backtest overfitting (CSCV, 16 blocks)")
    ax.set_ylabel("Markets")
    ax.legend(frameon=False, fontsize=9)
    style(ax)
    fig.tight_layout()
    fig.savefig(HERE / "fig_pbo.png", dpi=200)
    plt.close(fig)


def published_example() -> dict[str, float | str]:
    """The single market behind docs/tearsheet_validity.html (seed 3, 3 bp drift)."""
    market = simulate_market(np.random.default_rng(3), drift=0.0003)
    out = examine(market, crossover_trials(market))
    keep = ("selected", "sharpe_is", "sharpe_oos", "sharpe_full", "market_sharpe_full", "psr", "dsr", "alpha_t", "pbo")
    return {k: out[k] for k in keep}


def skilled_true_sharpe(n: int = 5_000_000) -> float:
    """Sharpe of the skilled rule over a very long sample."""
    rng = np.random.default_rng(0)
    m = rng.standard_t(5, n) * 0.008
    return annual_sharpe(np.sign(m + rng.normal(0.0, SKILL_NOISE * m.std(), n)) * m)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--markets", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20261124)
    args = parser.parse_args()

    results = {
        name: run_scenario(name, args.markets, args.seed + i)
        for i, name in enumerate(("null", "drift", "skill"))
    }
    summary = {name: summarise(df) for name, df in results.items()}
    extra = {"published_example": published_example(), "skilled_true_sharpe": skilled_true_sharpe()}
    (HERE / "results.json").write_text(json.dumps({**summary, **extra}, indent=2) + "\n")
    figures(results)
    print(pd.DataFrame(summary).round(3).to_string())
    print(json.dumps(extra, indent=2))


if __name__ == "__main__":
    main()
