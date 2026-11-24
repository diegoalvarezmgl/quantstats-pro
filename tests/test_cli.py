"""Tests for the quantstats command-line interface."""

import json

import numpy as np
import pandas as pd
import pytest

from quantstats import cli


@pytest.fixture
def files(tmp_path):
    rng = np.random.default_rng(3)
    n = 600
    idx = pd.bdate_range("2021-01-01", periods=n)
    bench = pd.Series(rng.normal(0.0004, 0.011, n), index=idx)
    strat = 0.0004 + 0.5 * bench + rng.normal(0, 0.006, n)
    paths = {k: tmp_path / f"{k}.csv" for k in ("strat", "bench", "prices", "trials")}
    pd.DataFrame({"date": idx, "ret": strat.values}).to_csv(paths["strat"], index=False)
    pd.DataFrame({"date": idx, "spy": bench.values}).to_csv(paths["bench"], index=False)
    pd.DataFrame({"date": idx, "px": 100 * (1 + strat).cumprod().values}).to_csv(
        paths["prices"], index=False
    )
    trials = pd.DataFrame(rng.normal(0, 0.008, (n, 6)), columns=list("abcdef"))
    trials["a"] = strat.values
    trials.insert(0, "date", idx)
    trials.to_csv(paths["trials"], index=False)
    return {k: str(v) for k, v in paths.items()} | {"dir": tmp_path}


def _run(capsys, *argv):
    code = cli.main(list(argv))
    out = capsys.readouterr()
    return code, out.out, out.err


def test_guide_is_packaged(capsys):
    code, out, _ = _run(capsys, "guide")
    assert code == 0
    assert "Input contract" in out
    assert out == cli.guide_text()


def test_validate_json(capsys, files):
    code, out, _ = _run(
        capsys, "validate", files["strat"], "--benchmark", files["bench"],
        "--n-trials", "25", "--split", "0.3", "--n-boot", "100",
    )
    assert code == 0
    data = json.loads(out)
    assert data["sharpe"]["n_trials"] == 25
    assert data["decay"]["kind"] == "alpha"
    assert data["split"]["out_of_sample"]["n"] == 180
    assert data["sample"]["start"].startswith("2021-01-01")


def test_validate_with_trials_and_html(capsys, files):
    html = files["dir"] / "v.html"
    code, out, _ = _run(
        capsys, "validate", files["trials"], "--column", "a",
        "--trials", files["trials"], "--n-boot", "100", "--html", str(html),
    )
    assert code == 0
    data = json.loads(out)
    assert data["overfitting"]["n_trials"] == 6
    assert html.exists() and "Backtest Validity" in html.read_text()


def test_prices_are_rejected_unless_flagged(capsys, files):
    code, out, err = _run(capsys, "validate", files["prices"])
    assert code == 2 and out == ""
    assert "price series" in json.loads(err)["error"]["message"]

    code, out, _ = _run(capsys, "validate", files["prices"], "--prices", "--n-boot", "50")
    assert code == 0
    assert json.loads(out)["sample"]["n"] == 599


def test_ambiguous_column_and_missing_file(capsys, files):
    code, _, err = _run(capsys, "metrics", files["trials"])
    assert code == 2
    assert "--column" in json.loads(err)["error"]["message"]

    code, _, err = _run(capsys, "metrics", "nope.csv")
    assert code == 2
    assert json.loads(err)["error"]["type"] == "CLIError"


def test_metrics_matches_stats(capsys, files):
    import quantstats as qs

    code, out, _ = _run(capsys, "metrics", files["strat"], "--benchmark", files["bench"])
    assert code == 0
    data = json.loads(out)
    r = cli.load_series(files["strat"])
    assert data["sharpe"] == pytest.approx(qs.stats.sharpe(r))
    assert data["max_drawdown"] == pytest.approx(qs.stats.max_drawdown(r))
    assert data["benchmark"]["beta"] == pytest.approx(0.5, abs=0.06)


def test_decay_and_montecarlo_json(capsys, files):
    code, out, _ = _run(capsys, "decay", files["strat"], "--benchmark", files["bench"])
    assert code == 0
    data = json.loads(out)
    assert data["windows"] == [21, 63, 126]
    assert data["benchmark_beta"] == pytest.approx(0.5, abs=0.08)

    code, out, _ = _run(
        capsys, "montecarlo", files["strat"], "--models", "gbm,bootstrap",
        "--sims", "50", "--bust", "-0.2",
    )
    assert code == 0
    data = json.loads(out)
    assert set(data["models"]) == {"gbm", "bootstrap"}
    assert "bust_prob" in data["models"]["gbm"]


def test_montecarlo_is_reproducible(capsys, files):
    args = ("montecarlo", files["strat"], "--models", "bootstrap", "--sims", "40")
    assert _run(capsys, *args)[1] == _run(capsys, *args)[1]


@pytest.mark.parametrize("kind", ["simple", "validity", "alpha-decay"])
def test_report_writes_file(capsys, files, kind):
    out_path = files["dir"] / f"{kind}.html"
    code, out, _ = _run(
        capsys, "report", files["strat"], "--benchmark", files["bench"],
        "--kind", kind, "--output", str(out_path),
    )
    assert code == 0
    assert json.loads(out)["kind"] == kind
    assert out_path.stat().st_size > 10_000


def test_monthly_data_uses_one_annualisation_everywhere(capsys, tmp_path):
    rng = np.random.default_rng(5)
    idx = pd.date_range("2012-01-01", periods=120, freq="MS")
    path = tmp_path / "monthly.csv"
    pd.DataFrame({"date": idx, "ret": rng.normal(0.01, 0.04, 120)}).to_csv(
        path, index=False
    )
    _, out, _ = _run(capsys, "metrics", str(path))
    metrics = json.loads(out)
    _, out, _ = _run(capsys, "validate", str(path), "--n-boot", "50")
    valid = json.loads(out)
    _, out, _ = _run(capsys, "montecarlo", str(path), "--models", "gbm", "--sims", "20")
    mc = json.loads(out)
    assert metrics["periods_per_year"] == valid["periods_per_year"] == 12
    assert mc["periods_per_year"] == 12 and mc["horizon"] == 12
    assert metrics["sharpe"] == pytest.approx(valid["sharpe"]["sharpe_annual"])


def test_unexpected_errors_are_json(capsys, files, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("kaboom")

    monkeypatch.setattr(cli, "compute_metrics", boom)
    code, out, err = _run(capsys, "metrics", files["strat"])
    assert code == 2 and out == ""
    assert json.loads(err)["error"] == {"type": "RuntimeError", "message": "kaboom"}
