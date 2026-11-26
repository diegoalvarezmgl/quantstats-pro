"""HTML rendering of a :class:`ValidityResult` (used by ``reports.html_validity``)."""

from __future__ import annotations

import html as _html
import math
from datetime import datetime
from pathlib import Path

from .core import PBO_NO_INFORMATION, PBO_REJECT_ABOVE, ValidityResult, _fmt_obs

_SEVERITY_ORDER = {"warning": 0, "caution": 1, "info": 2}


def _minus(text: str) -> str:
    return text.replace("-", "−")


def _pct(x: float | None, digits: int = 1) -> str:
    if x is None or not math.isfinite(x):
        return "–"
    return _minus(f"{x * 100:.{digits}f}%")


def _num(x: float | None, digits: int = 2) -> str:
    if x is None or not math.isfinite(x):
        return "–"
    return _minus(f"{x:,.{digits}f}")


def _clamp(x: float) -> float:
    return max(0.0, min(100.0, x))


def _gauge(
    fill: tuple[float, float] | None = None,
    bar: float | None = None,
    ref: float | None = None,
    obs: float | None = None,
) -> str:
    """A track with a filled range and up to three ticks; positions in percent."""
    parts = []
    if fill is not None and all(math.isfinite(v) for v in fill):
        left = _clamp(min(fill))
        width = _clamp(max(fill)) - left
        parts.append(
            f'<span class="g-fill" style="left:{left:.1f}%;width:{width:.1f}%"></span>'
        )
    for cls, pos in (("g-obs", obs), ("g-ref", ref), ("g-bar", bar)):
        if pos is not None and math.isfinite(pos):
            parts.append(f'<span class="{cls}" style="left:{_clamp(pos):.1f}%"></span>')
    return '<div class="gauge">' + "".join(parts) + "</div>"


def _axis(lo: float, hi: float, pad: float = 0.1):
    """Map a value to a percentage of an axis spanning lo..hi plus padding."""
    span = hi - lo
    if not math.isfinite(span) or span <= 0:
        return lambda x: float("nan")
    lo, span = lo - pad * span, span * (1 + 2 * pad)
    return lambda x: (x - lo) / span * 100.0


def _kpi(label: str, value: str, hint: str, tone: str = "", gauge: str = "") -> str:
    return (
        f'<div class="kpi {tone}"><div class="label">{label}</div>'
        f'<div class="value">{value}</div>{gauge}<div class="hint">{hint}</div></div>'
    )


def _table(header: list[str], rows: list[list[str]], cls: str = "") -> str:
    head = "".join(f"<th>{h}</th>" for h in header)
    body = "".join(
        "<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>" for row in rows
    )
    attr = f' class="{cls}"' if cls else ""
    return f"<table{attr}><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def _kpis_html(res: ValidityResult) -> str:
    """Key figures, each drawn against the bar it has to clear, most telling first."""
    s, b, d = res.sharpe, res.bootstrap, res.decay
    conf = b["confidence"]
    ci = b["sharpe"]
    n = res.sample["n"]
    figures = []

    dsr = s["dsr"]
    if not math.isfinite(dsr) or dsr >= conf:
        tone = ""
    else:
        tone = "warning" if dsr >= 0.5 else "critical"
    selection = (
        f"after {s['n_trials']:,} trials, hurdle {_num(s['sharpe_hurdle_annual'])}"
        if s["n_trials"] > 1
        else "one trial, no selection discount"
    )
    figures.append(
        _kpi(
            "Deflated Sharpe",
            _pct(dsr),
            f"Bar at {conf:.0%} &middot; {selection}",
            tone,
            _gauge(fill=(0.0, dsr * 100.0), bar=conf * 100.0),
        )
    )

    if res.split is not None:
        oos = res.split["out_of_sample"]["sharpe_annual"]
        ins = res.split["in_sample"]["sharpe_annual"]
        p = res.split["degradation_p_value"]
        if math.isfinite(oos) and oos <= 0:
            tone = "critical"
        elif math.isfinite(p) and p < 1 - conf:
            tone = "warning"
        else:
            tone = ""
        reach = max(
            [abs(v) for v in (ins, oos) if math.isfinite(v)] + [0.5]
        )
        at = _axis(-reach, reach)
        figures.append(
            _kpi(
                "Out-of-sample Sharpe",
                _num(oos),
                f"In-sample {_num(ins)} &middot; degradation p = {_num(p, 3)}",
                tone,
                _gauge(fill=(50.0, at(oos)), bar=50.0, ref=at(ins)),
            )
        )

    if res.overfitting is not None:
        pbo = res.overfitting["pbo"]
        if pbo >= PBO_NO_INFORMATION:
            tone, verdict = "critical", "selection uninformative"
        elif pbo > PBO_REJECT_ABOVE:
            tone, verdict = "warning", "inconclusive"
        else:
            tone, verdict = "", "passes"
        figures.append(
            _kpi(
                "Probability of overfitting",
                _pct(pbo, 0),
                f"Reject above {PBO_REJECT_ABOVE:.0%} &middot; {verdict} &middot; "
                f"{res.overfitting['n_trials']} trials",
                tone,
                _gauge(
                    fill=(0.0, pbo * 100.0),
                    bar=PBO_REJECT_ABOVE * 100.0,
                    ref=PBO_NO_INFORMATION * 100.0,
                ),
            )
        )

    low, high = ci["low"], ci["high"]
    if math.isfinite(high) and high <= 0:
        tone = "critical"
    elif math.isfinite(low) and low <= 0:
        tone = "warning"
    else:
        tone = ""
    hint = f"{conf:.0%} interval {_num(low)} to {_num(high)}"
    if tone == "warning":
        hint += " &middot; includes zero"
    gauge = _gauge()
    if math.isfinite(low) and math.isfinite(high):
        at = _axis(min(low, 0.0), max(high, 0.0))
        gauge = _gauge(fill=(at(low), at(high)), bar=at(0.0), obs=at(s["sharpe_annual"]))
    figures.append(
        _kpi("Sharpe, annualised", _num(s["sharpe_annual"]), hint, tone, gauge)
    )

    need = s["min_track_record_length_deflated"]
    edge = "Alpha" if d["kind"] == "alpha" else "Mean return"
    edge_detail = f"t {_num(d['edge_t_stat'])}"
    if d.get("edge_t_threshold_n_trials", 1) > 1:
        edge_detail += f" vs {_num(d['edge_t_threshold'])} required"
    if d["beta"] is not None:
        edge_detail += f", &beta; {_num(d['beta'])}"
    sub = [
        ("Probabilistic Sharpe", _pct(s["psr"]), "one trial"),
        ("Track record needed", f"{_fmt_obs(need)} obs", f"of {n:,} held"),
        (edge, f"{_pct(d['edge_annual'], 2)}/yr", edge_detail),
        ("Skew", _num(s["skew"]), f"kurtosis {_num(s['kurtosis'], 1)}"),
    ]
    strip = "".join(
        f'<div><span class="k">{k}</span><b>{v}</b> <span class="x">{x}</span></div>'
        for k, v, x in sub
    )
    return (
        '<div class="figures">' + "".join(figures) + "</div>"
        f'<div class="sub">{strip}</div>'
    )


_SEVERITY_NOUN = {"warning": "warning", "caution": "caution", "info": "note"}


def _summary_html(res: ValidityResult) -> str:
    """Headline: how many findings of each severity, worst first."""
    counts = {
        sev: sum(1 for f in res.findings if f.severity == sev)
        for sev in _SEVERITY_NOUN
    }
    parts = [
        f"{c} {_SEVERITY_NOUN[sev]}{'' if c == 1 else 's'}"
        for sev, c in counts.items()
        if c
    ]
    if not parts:
        return (
            '<div class="summary"><div class="head">No findings</div>'
            '<div class="sub">Nothing was flagged at the configured thresholds.'
            "</div></div>"
        )
    worst = next(sev for sev, c in counts.items() if c)
    rest = "".join(f" &middot; {p}" for p in parts[1:])
    return (
        f'<div class="summary"><div class="head"><span class="sq {worst}"></span>'
        f"{parts[0]}<span>{rest}</span></div></div>"
    )


def _findings_html(res: ValidityResult) -> str:
    items = sorted(res.findings, key=lambda f: _SEVERITY_ORDER.get(f.severity, 3))
    if not items:
        return '<p class="note">No findings.</p>'
    rows = "".join(
        f'<tr class="{f.severity}"><td class="s"><span class="sq {f.severity}">'
        f'</span></td><td class="l">{f.severity}</td>'
        f"<td>{_html.escape(f.message)}</td>"
        f'<td class="c">{f.code}</td></tr>'
        for f in items
    )
    return f'<table class="findings"><tbody>{rows}</tbody></table>'


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
    half = (len(rows) + 1) // 2
    return (
        '<div class="two-col">'
        + _table(["Sharpe significance", "Value"], rows[:half], "kv")
        + _table(["Sample", "Value"], rows[half:], "kv")
        + "</div>"
    )


def _bootstrap_table(res: ValidityResult) -> str:
    b = res.bootstrap
    conf = b["confidence"]

    def row(label: str, key: str, pct: bool) -> list[str]:
        fmt = _pct if pct else _num
        e = b[key]
        low, high, obs = e["low"], e["high"], e["observed"]
        bar = _gauge()
        if math.isfinite(low) and math.isfinite(high):
            at = _axis(min(low, 0.0), max(high, 0.0))
            zero = at(0.0) if low < 0 < high else None
            bar = _gauge(fill=(at(low), at(high)), bar=zero, obs=at(obs))
        return [label, fmt(obs), fmt(low), bar, fmt(high), fmt(e["median"])]

    lo, hi = (1 - conf) / 2, 1 - (1 - conf) / 2
    rows = [
        row("Sharpe (annualised)", "sharpe", False),
        row("CAGR", "cagr", True),
        row("Volatility (annualised)", "volatility", True),
        row("Max drawdown", "max_drawdown", True),
        ["P(Sharpe ≤ 0)", _pct(b["prob_sharpe_below_zero"]), "", "", "", ""],
    ]
    cls = ["", "", ' class="lo"', ' class="bar"', ' class="hi"', ""]
    body = "".join(
        "<tr>" + "".join(f"<td{c}>{v}</td>" for c, v in zip(cls, r, strict=True)) + "</tr>"
        for r in rows
    )
    head = "".join(
        f"<th{c}>{h}</th>"
        for c, h in zip(
            cls,
            ["Bootstrap", "Observed", f"P{lo * 100:g}", "Interval", f"P{hi * 100:g}", "Median"],
            strict=True,
        )
    )
    return (
        f'<table class="ivl"><thead><tr>{head}</tr></thead>'
        f"<tbody>{body}</tbody></table>"
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
        ["p-value, single test", _num(d["edge_p_value"], 3)],
    ]
    if d.get("edge_t_threshold_n_trials", 1) > 1:
        how = (
            "max-t bootstrap"
            if d["edge_t_threshold_method"] == "max_t_bootstrap"
            else "Šidák"
        )
        rows.insert(
            2,
            [
                f"t required for the best of {d['edge_t_threshold_n_trials']:,} trials ({how})",
                _num(d["edge_t_threshold"]),
            ],
        )
    if d["beta"] is not None:
        rows.append(["Beta", _num(d["beta"])])
    rows += [
        ["Trend (annualised, per year)", _pct(d["trend_annual_per_year"], 2)],
        ["Trend t-statistic", _num(d["trend_t_stat"])],
        ["Trend p-value", _num(d["trend_p_value"], 3)],
    ]
    return _table(["Edge and decay", "Value"], rows, "kv")


def _edge_method_note(res: ValidityResult) -> str:
    d = res.decay
    n = d.get("edge_t_threshold_n_trials", 1)
    if n <= 1:
        return "One trial: compared with the single-test critical value."
    if d["edge_t_threshold_method"] == "max_t_bootstrap":
        return (
            f"Compared with {_num(d['edge_t_threshold'])}, the upper quantile of the "
            f"largest t-statistic among the {n:,} trials under no edge (studentised "
            "max-t stationary bootstrap, trials and benchmark resampled jointly)."
        )
    return (
        f"Compared with {_num(d['edge_t_threshold'])}, the Šidák critical value for "
        f"{n:,} independent trials. Conservative when trials are correlated; pass "
        "the trials' returns to measure it."
    )


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
        '<section class="sec"><div class="sec-side">'
        "<h3>Probability of Backtest Overfitting</h3>"
        '<p class="note">Every half/half combination of time blocks is used as '
        "train/test. PBO is the share of splits where the best variant in-sample "
        "lands in the bottom half out-of-sample. Bailey et al. (2017) reject above "
        f"{PBO_REJECT_ABOVE:.0%}; {PBO_NO_INFORMATION:.0%} is what an uninformative "
        "in-sample rank gives on average.</p></div>"
        '<div class="sec-main">'
        f'<div id="pbo_chart" class="cell full">{embed(plotting.pbo_figure(res))}</div>'
        + _table(["Overfitting", "Value"], rows, "kv")
        + "</div></section>"
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
        "Return left after removing benchmark exposure (r = α + β·benchmark). "
        if is_alpha
        else "No benchmark was supplied, so the edge is the mean excess return. "
    ) + (
        "The dashed line is a fitted time trend: a significantly negative slope "
        "means the edge is decaying."
    )
    if result.split is not None:
        split_side = (
            f"The shaded period, from {result.split['split_label']}, is the part "
            "the strategy was not tuned on."
        )
        split_note = ""
    else:
        split_side = (
            "Compares the period used to design the strategy with the period "
            "that followed."
        )
        split_note = (
            '<p class="note">No split supplied. Pass <code>split=</code> with the '
            "date the strategy was frozen (or the out-of-sample fraction) to "
            "compare the two periods.</p>"
        )
    params = " &bull; ".join(
        [
            f"Strategy: {_html.escape(result.label)}",
            f"Trials: {s['n_trials']:,}",
            f"Observations: {sample['n']:,} ({sample['years']:.1f}y)",
        ]
    )
    method = [
        (
            "Probabilistic Sharpe",
            "Probability that the true Sharpe is above zero given sample length, "
            "skew and fat tails.",
        ),
        (
            "Deflated Sharpe",
            "The same probability against the best Sharpe expected from "
            f"{s['n_trials']:,} skill-less trial(s). Cross-trial dispersion: "
            f"{a['trials_sharpe_std_source']}.",
        ),
        (
            "Minimum track record",
            "Observations needed for the observed Sharpe to be significant at "
            f"{a['confidence']:.0%}.",
        ),
        ("Bootstrap", f"{a['bootstrap'].capitalize()}. Newey–West errors for the edge."),
        (
            "Edge t-statistic",
            _edge_method_note(result),
        ),
        (
            "Returns",
            f"{a['returns'].capitalize()} (rf={a['rf_annual']:.2%}); "
            f"{a['periods_per_year']:g} periods per year.",
        ),
        (
            "Not covered",
            "Look-ahead bias, survivorship bias, unrealistic fills and data errors.",
        ),
    ]
    assumptions = (
        "<dl>"
        + "".join(f"<div><dt>{k}</dt><dd>{v}</dd></div>" for k, v in method)
        + "</dl>"
        "Methods: Bailey &amp; López de Prado (2012, 2014); Bailey, Borwein, "
        "López de Prado &amp; Zhu (2017); Politis &amp; Romano (1994); "
        "White (2000); Romano &amp; Wolf (2005). "
        "Informational only — not investment advice."
    )

    fragments = {
        "title": _html.escape(title),
        "date_range": date_range,
        "params": params,
        "v": __version__,
        "generated_at": datetime.now().strftime("%d %b, %Y %H:%M").lstrip("0"),
        "summary": _summary_html(result),
        "kpis": _kpis_html(result),
        "findings": _findings_html(result),
        "n_trials": f"{s['n_trials']:,}",
        "sharpe_chart": embed(plotting.sharpe_distribution_figure(result)),
        "sharpe_table": _sharpe_table(result),
        "bootstrap_table": _bootstrap_table(result),
        "split_side": split_side,
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
