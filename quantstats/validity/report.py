"""HTML rendering of a :class:`ValidityResult` (used by ``reports.html_validity``)."""

from __future__ import annotations

import html as _html
import math
from datetime import datetime
from pathlib import Path

from .core import ValidityResult, _fmt_obs

_SEVERITY_ORDER = {"warning": 0, "caution": 1, "info": 2}


def _pct(x: float | None, digits: int = 1) -> str:
    return "–" if x is None or not math.isfinite(x) else f"{x * 100:.{digits}f}%"


def _num(x: float | None, digits: int = 2) -> str:
    return "–" if x is None or not math.isfinite(x) else f"{x:,.{digits}f}"


def _kpi(label: str, value: str, hint: str, tone: str = "") -> str:
    return (
        f'<div class="kpi {tone}"><div class="label">{label}</div>'
        f'<div class="value">{value}</div><div class="hint">{hint}</div></div>'
    )


def _prob_tone(p: float, confidence: float) -> str:
    if not math.isfinite(p):
        return ""
    if p >= confidence:
        return "excellent"
    if p >= 0.8:
        return "good"
    if p >= 0.5:
        return "warning"
    return "critical"


def _table(header: list[str], rows: list[list[str]]) -> str:
    head = "".join(f"<th>{h}</th>" for h in header)
    body = "".join(
        "<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>" for row in rows
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def _kpis_html(res: ValidityResult) -> str:
    s, b = res.sharpe, res.bootstrap
    conf = b["confidence"]
    ci = b["sharpe"]
    n = res.sample["n"]
    need = s["min_track_record_length_deflated"]
    if not math.isfinite(need):
        trl_tone = "critical"
    elif need <= n:
        trl_tone = "excellent"
    else:
        trl_tone = "warning"
    cards = [
        _kpi(
            "Sharpe (annualised)",
            _num(s["sharpe_annual"]),
            f"{conf:.0%} CI {_num(ci['low'])} to {_num(ci['high'])}",
            "critical" if math.isfinite(ci["low"]) and ci["high"] <= 0 else "",
        ),
        _kpi(
            "Probabilistic Sharpe",
            _pct(s["psr"]),
            "P(true Sharpe &gt; 0), one trial",
            _prob_tone(s["psr"], conf),
        ),
        _kpi(
            "Deflated Sharpe",
            _pct(s["dsr"]),
            f"after {s['n_trials']:,} trial(s); hurdle "
            f"{_num(s['sharpe_hurdle_annual'])}",
            _prob_tone(s["dsr"], conf),
        ),
        _kpi(
            "Track record needed",
            _fmt_obs(need),
            f"observations at {conf:.0%}; sample has {n:,}",
            trl_tone,
        ),
    ]
    if res.split is not None:
        oos = res.split["out_of_sample"]
        ins = res.split["in_sample"]
        p = res.split["degradation_p_value"]
        tone = (
            "critical"
            if oos["sharpe"] <= 0
            else ("warning" if math.isfinite(p) and p < 1 - conf else "good")
        )
        cards.append(
            _kpi(
                "Out-of-sample Sharpe",
                _num(oos["sharpe_annual"]),
                f"in-sample {_num(ins['sharpe_annual'])}; p={_num(p, 3)}",
                tone,
            )
        )
    if res.overfitting is not None:
        pbo = res.overfitting["pbo"]
        tone = "critical" if pbo >= 0.5 else ("warning" if pbo >= 0.25 else "excellent")
        cards.append(
            _kpi(
                "Prob. of overfitting",
                _pct(pbo, 0),
                f"CSCV over {res.overfitting['n_trials']} trials",
                tone,
            )
        )
    return '<div class="kpi-grid">' + "".join(cards) + "</div>"


def _findings_html(res: ValidityResult) -> str:
    items = sorted(res.findings, key=lambda f: _SEVERITY_ORDER.get(f.severity, 3))
    if not items:
        return '<p class="note">No findings.</p>'
    lis = "".join(
        f'<li class="{f.severity}"><span class="badge {_badge(f.severity)}">'
        f"{f.severity}</span> <code>{f.code}</code>{_html.escape(f.message)}</li>"
        for f in items
    )
    return f'<ul class="findings">{lis}</ul>'


def _badge(severity: str) -> str:
    return {"warning": "critical", "caution": "caution", "info": "info"}.get(
        severity, "info"
    )


def _sharpe_table(res: ValidityResult) -> str:
    s = res.sharpe
    rows = [
        ["Sharpe (annualised)", _num(s["sharpe_annual"])],
        ["Standard error (annualised)", _num(s["sharpe_std_error_annual"])],
        ["Probabilistic Sharpe ratio", _pct(s["psr"])],
        ["Trials accounted for", f"{s['n_trials']:,}"],
        ["Selection hurdle (annualised)", _num(s["sharpe_hurdle_annual"])],
        ["Deflated Sharpe ratio", _pct(s["dsr"])],
        ["Min. track record (vs 0)", _fmt_obs(s["min_track_record_length"])],
        [
            "Min. track record (vs hurdle)",
            _fmt_obs(s["min_track_record_length_deflated"]),
        ],
        ["Observations", f"{res.sample['n']:,}"],
        ["Skew / kurtosis", f"{_num(s['skew'])} / {_num(s['kurtosis'], 1)}"],
        ["Lag-1 autocorrelation", _num(s["autocorrelation_lag1"])],
    ]
    return _table(["Sharpe significance", "Value"], rows)


def _bootstrap_table(res: ValidityResult) -> str:
    b = res.bootstrap
    conf = b["confidence"]

    def row(label: str, key: str, pct: bool) -> list[str]:
        fmt = _pct if pct else _num
        e = b[key]
        return [label, fmt(e["observed"]), fmt(e["low"]), fmt(e["median"]), fmt(e["high"])]

    lo, hi = (1 - conf) / 2, 1 - (1 - conf) / 2
    rows = [
        row("Sharpe (annualised)", "sharpe", False),
        row("CAGR", "cagr", True),
        row("Volatility (annualised)", "volatility", True),
        row("Max drawdown", "max_drawdown", True),
        ["P(Sharpe ≤ 0)", _pct(b["prob_sharpe_below_zero"]), "", "", ""],
    ]
    return _table(
        ["Bootstrap", "Observed", f"P{lo * 100:g}", "Median", f"P{hi * 100:g}"], rows
    )


def _split_table(res: ValidityResult) -> str:
    sp = res.split
    if sp is None:
        return ""
    a, b = sp["in_sample"], sp["out_of_sample"]

    def chg(x: float, y: float) -> str:
        return _num(y - x) if math.isfinite(x) and math.isfinite(y) else "–"

    rows = [
        ["Observations", f"{a['n']:,}", f"{b['n']:,}", ""],
        [
            "Sharpe (annualised)",
            _num(a["sharpe_annual"]),
            _num(b["sharpe_annual"]),
            chg(a["sharpe_annual"], b["sharpe_annual"]),
        ],
        ["Probabilistic Sharpe", _pct(a["psr"]), _pct(b["psr"]), ""],
        ["CAGR", _pct(a["cagr"]), _pct(b["cagr"]), ""],
        ["Volatility", _pct(a["volatility"]), _pct(b["volatility"]), ""],
        ["Max drawdown", _pct(a["max_drawdown"]), _pct(b["max_drawdown"]), ""],
        ["Win rate", _pct(a["win_rate"]), _pct(b["win_rate"]), ""],
        [
            "Sharpe change z / p-value",
            "",
            "",
            f"{_num(sp['sharpe_change_z'])} / {_num(sp['degradation_p_value'], 3)}",
        ],
    ]
    return _table(
        ["", "In-sample", f"Out-of-sample (from {sp['split_label']})", "Change"], rows
    )


def _edge_table(res: ValidityResult) -> str:
    d = res.decay
    name = "Alpha" if d["kind"] == "alpha" else "Mean return"
    rows = [
        [f"{name} (annualised)", _pct(d["edge_annual"], 2)],
        ["t-statistic (Newey–West)", _num(d["edge_t_stat"])],
        ["p-value", _num(d["edge_p_value"], 3)],
    ]
    if d["beta"] is not None:
        rows.append(["Beta", _num(d["beta"])])
    rows += [
        ["Trend (annualised, per year)", _pct(d["trend_annual_per_year"], 2)],
        ["Trend t-statistic", _num(d["trend_t_stat"])],
        ["Trend p-value", _num(d["trend_p_value"], 3)],
    ]
    return _table(["Edge and decay", "Value"], rows)


def _pbo_section(res: ValidityResult, embed) -> str:
    over = res.overfitting
    if over is None:
        return ""
    from . import plotting

    rows = [
        ["Probability of backtest overfitting", _pct(over["pbo"])],
        ["P(out-of-sample Sharpe &lt; 0) for the winner", _pct(over["prob_oos_loss"])],
        ["OOS vs IS Sharpe slope of the winner", _num(over["degradation_slope"])],
        ["Trials / time blocks / splits",
         f"{over['n_trials']} / {over['n_splits']} / {over['n_combinations']:,}"],
    ]
    return (
        "<h3>Probability of Backtest Overfitting</h3>"
        '<p class="note">Combinatorially symmetric cross-validation: the history '
        "is cut into blocks, every half/half combination is used as train/test, "
        "and each time the best variant in-sample is ranked out-of-sample. PBO is "
        "the share of splits where that winner lands in the bottom half.</p>"
        f'<div id="pbo_chart" class="cell full">{embed(plotting.pbo_figure(res))}</div>'
        + _table(["Overfitting", "Value"], rows)
    )


def render_html(
    result: ValidityResult,
    title: str = "Backtest Validity Tearsheet",
    figfmt: str = "svg",
    template_path: str | Path | None = None,
) -> str:
    """Render the validity tearsheet to an HTML string."""
    import matplotlib.pyplot as plt

    from .. import __version__, reports, utils
    from .._reporting import engine
    from . import plotting

    if template_path is None:
        template_path = Path(__file__).parent.parent / "report_validity.html"
    template_path = Path(template_path).resolve()
    if not template_path.exists():
        raise FileNotFoundError(f"Template file not found: {template_path}")
    tpl = template_path.read_text(encoding="utf-8")

    def embed(fig) -> str:
        stream = utils._file_stream()
        fig.savefig(stream, format=figfmt, bbox_inches="tight")
        plt.close(fig)
        return reports._embed_figure(stream, figfmt)

    s, d, a = result.sharpe, result.decay, result.assumptions
    sample = result.sample
    if sample["start"] is not None:
        date_range = (
            f"{sample['start'].strftime('%e %b, %Y').strip()} - "
            f"{sample['end'].strftime('%e %b, %Y').strip()}"
        )
    else:
        date_range = f"{sample['n']:,} observations"

    is_alpha = d["kind"] == "alpha"
    edge_note = (
        "Alpha is the return left after removing benchmark exposure "
        "(r = α + β·benchmark). "
        if is_alpha
        else "No benchmark was supplied, so the edge is the mean excess return. "
    ) + (
        "The dashed line is a linear time trend fitted to the full sample with "
        "Newey–West standard errors: a significantly negative slope means the "
        "edge is decaying."
    )
    split_note = (
        ""
        if result.split is not None
        else '<p class="note">No split supplied. Pass <code>split=</code> with the '
        "date the strategy was frozen (or the out-of-sample fraction) to compare "
        "the period used to design it with the period that followed.</p>"
    )
    params = " &bull; ".join(
        [
            f"Strategy: {_html.escape(result.label)}",
            f"Trials: {s['n_trials']:,}",
            f"Observations: {sample['n']:,} ({sample['years']:.1f}y)",
        ]
    )
    assumptions = (
        f"Assumptions: {a['returns']} (rf={a['rf_annual']:.2%}) &bull; "
        f"periods/year={a['periods_per_year']:g} &bull; "
        f"confidence={a['confidence']:.0%} &bull; bootstrap: {a['bootstrap']} &bull; "
        f"cross-trial Sharpe dispersion: {a['trials_sharpe_std_source']}. "
        "Methods: Bailey &amp; López de Prado (2012, 2014); Bailey, Borwein, "
        "López de Prado &amp; Zhu (2017); Politis &amp; Romano (1994). "
        "Informational only — not investment advice."
    )

    fragments = {
        "title": _html.escape(title),
        "date_range": date_range,
        "params": params + " | ",
        "v": __version__,
        "generated_at": datetime.now().strftime("%d %b, %Y %H:%M").lstrip("0"),
        "kpis": _kpis_html(result),
        "findings": _findings_html(result),
        "n_trials": f"{s['n_trials']:,}",
        "sharpe_chart": embed(plotting.sharpe_distribution_figure(result)),
        "sharpe_table": _sharpe_table(result),
        "bootstrap_table": _bootstrap_table(result),
        "split_note": split_note,
        "equity_chart": embed(plotting.equity_figure(result)),
        "split_table": _split_table(result),
        "edge_title": "Alpha and Alpha Decay" if is_alpha else "Edge and Decay",
        "edge_note": edge_note,
        "edge_chart": embed(plotting.rolling_edge_figure(result)),
        "edge_table": _edge_table(result),
        "pbo_section": _pbo_section(result, embed),
        "assumptions": assumptions,
    }
    return engine.render_report(tpl, fragments)
