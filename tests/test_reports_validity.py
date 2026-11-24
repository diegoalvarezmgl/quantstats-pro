"""Tests for reports.html_validity."""

import numpy as np
import pandas as pd
import pytest

import quantstats as qs


@pytest.fixture
def data():
    rng = np.random.default_rng(12)
    n = 800
    idx = pd.bdate_range("2020-01-01", periods=n)
    bench = pd.Series(rng.normal(0.0004, 0.01, n), index=idx, name="bench")
    strat = pd.Series(0.0005 + 0.4 * bench + rng.normal(0, 0.007, n), name="strat")
    trials = pd.DataFrame(rng.normal(0, 0.008, (n, 10)), index=idx)
    trials[0] = strat
    return strat, bench, trials


def test_html_validity_full(tmp_path, data):
    strat, bench, trials = data
    out = tmp_path / "validity.html"
    res = qs.reports.html_validity(
        strat, bench, trials=trials, split="2022-01-03", output=str(out), n_boot=200
    )
    html = out.read_text(encoding="utf-8")
    assert "{{" not in html
    for section in (
        "Sharpe Ratio Significance",
        "In-Sample vs Out-of-Sample",
        "Alpha and Alpha Decay",
        "Probability of Backtest Overfitting",
        "Deflated Sharpe",
    ):
        assert section in html
    assert html.count("<svg") == 4
    for finding in res.findings:
        assert finding.code in html
    assert res.sharpe["n_trials"] == 10


def test_html_validity_minimal(tmp_path, data):
    strat, _, _ = data
    out = tmp_path / "validity.html"
    res = qs.reports.html_validity(strat, output=str(out), n_boot=100)
    html = out.read_text(encoding="utf-8")
    assert "Edge and Decay" in html
    assert "No split supplied" in html
    assert "Probability of Backtest Overfitting" not in html
    assert html.count("<svg") == 3
    assert res.overfitting is None


def test_html_validity_rejects_ticker_benchmark(data):
    strat, _, _ = data
    with pytest.raises(ValueError, match="does not download"):
        qs.reports.html_validity(strat, benchmark="SPY", output="unused.html")


def test_html_validity_escapes_title(tmp_path, data):
    strat, _, _ = data
    out = tmp_path / "v.html"
    qs.reports.html_validity(
        strat, output=str(out), n_boot=50, title="<script>x</script>"
    )
    assert "<script>x</script>" not in out.read_text(encoding="utf-8")
