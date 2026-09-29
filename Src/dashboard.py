"""Local dashboard shell, descriptive analytics, and saved model evaluation view."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi.responses import HTMLResponse

from Src.dataset1_integration import load_dataset1_transaction_table
from Src.ingestion import load_dataset2

DATASET_IDS = ("dataset1", "dataset2")
EVALUATION_REPORT = Path(__file__).resolve().parents[1] / "reports" / "evaluation" / "model_comparison.md"


class DashboardDataError(RuntimeError):
    """Raised when local dataset analytics are not available."""


@lru_cache(maxsize=2)
def get_overview(dataset_id: str) -> dict:
    """Load and cache an independent, PII-free overview for one dataset."""
    if dataset_id == "dataset1":
        frame = load_dataset1_transaction_table().frame
        target, amount = "FraudIndicator", "Amount"
        label = "Synthetic Financial Fraud Dataset"
        synthetic = True
        category_column = "Category"
        time_column = "Timestamp"
        time_kind = "calendar"
        limitation = "Synthetic fraud labels are randomly generated and do not represent real financial behavior."
    elif dataset_id == "dataset2":
        frame = load_dataset2().frame
        target, amount = "Class", "Amount"
        label = "Credit Card Fraud Dataset"
        synthetic = False
        category_column = None
        time_column = "Time"
        time_kind = "elapsed"
        limitation = "V1–V28 are anonymized PCA components. Time is elapsed seconds, not a calendar timestamp."
    else:
        raise DashboardDataError("Unknown dataset selection")

    if frame.empty or target not in frame or amount not in frame:
        raise DashboardDataError("The selected dataset has no usable transaction rows")
    y = pd.to_numeric(frame[target], errors="coerce")
    values = pd.to_numeric(frame[amount], errors="coerce")
    valid = y.isin((0, 1)) & values.notna() & np.isfinite(values)
    if not bool(valid.all()):
        raise DashboardDataError("The selected dataset contains invalid target or amount values")
    fraud_mask = y.eq(1)
    fraud_count = int(fraud_mask.sum())
    rows = len(frame)

    overview = {
        "dataset": {"id": dataset_id, "label": label, "synthetic": synthetic, "limitation": limitation},
        "kpis": {
            "total_transactions": rows,
            "fraud_transactions": fraud_count,
            "fraud_rate": fraud_count / rows,
            "total_amount": float(values.sum()),
            "average_amount": float(values.mean()),
            "high_risk_transactions": None,
        },
        "fraud_split": {"legitimate": int((~fraud_mask).sum()), "fraud": fraud_count},
        "trend": _trend(frame, y, time_column, time_kind),
        "amount_distribution": _amount_distribution(values, fraud_mask),
        "category_fraud": _category_summary(frame, y, category_column),
        "amount_note": "Source amount units; no currency is assumed.",
    }
    return overview


def clear_overview_cache() -> None:
    """Clear local dataset summaries (primarily useful after data refresh)."""
    get_overview.cache_clear()


@lru_cache(maxsize=2)
def get_performance(partition: str = "test") -> dict:
    """Read the already-computed Dataset 2 evaluation report for one split."""
    if partition not in {"validation", "test"}:
        raise DashboardDataError("Partition must be validation or test")
    try:
        report = EVALUATION_REPORT.read_text(encoding="utf-8")
    except OSError:
        raise DashboardDataError("The validated model evaluation report is unavailable") from None

    lines = report.splitlines()
    target = _metadata_value(lines, "- Target:")
    split_strategy = _metadata_value(lines, "- Split:")
    section = f"## {partition.title()} results"
    try:
        section_start = next(index for index, line in enumerate(lines) if line.strip() == section)
        table_header = next(index for index in range(section_start + 1, len(lines)) if lines[index].startswith("| Model |"))
    except StopIteration:
        raise DashboardDataError("The evaluation report is missing its expected model metrics table") from None

    headers = _markdown_cells(lines[table_header])
    expected_headers = ["Model", "Score type", "Decision rule", "Prevalence", "Alert rate", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC", "TN", "FP", "FN", "TP"]
    if headers != expected_headers:
        raise DashboardDataError("The evaluation report metrics table has an unexpected schema")

    models = []
    for line in lines[table_header + 2:]:
        if not line.startswith("|"):
            break
        cells = _markdown_cells(line)
        if len(cells) != len(expected_headers):
            raise DashboardDataError("The evaluation report contains a malformed metrics row")
        model, score_kind, decision_rule = cells[:3]
        if score_kind not in {"probability", "anomaly"}:
            raise DashboardDataError("The evaluation report contains an unsupported score kind")
        prevalence = _report_percent(cells[3])
        alert_rate = _report_percent(cells[4])
        metrics = {name.lower().replace("-", "_"): _report_float(value) for name, value in zip(expected_headers[5:10], cells[5:10])}
        counts = {name: _report_int(value) for name, value in zip(("tn", "fp", "fn", "tp"), cells[10:14])}
        models.append({"name": model, "score_kind": score_kind, "decision_rule": decision_rule,
                       "rows": sum(counts.values()), "prevalence": prevalence, "alert_rate": alert_rate,
                       "metrics": metrics, "confusion_matrix": counts})
    if not models:
        raise DashboardDataError("The evaluation report contains no model rows")
    return {
        "dataset": "Credit Card Fraud Dataset",
        "dataset1_note": "Dataset 1 is synthetic and has no trained fraud model; it is not combined with these results.",
        "partition": partition,
        "target": target,
        "split_strategy": split_strategy,
        "models": models,
        "curves": {"roc": "/dashboard/performance/roc.png", "precision_recall": "/dashboard/performance/precision-recall.png"},
        "limitations": [
            "Metrics use the reported fixed decision rules; no threshold was tuned using the test partition.",
            "Isolation Forest scores are anomaly rankings, not fraud probabilities; its labels use its native cutoff.",
            "These results describe the saved evaluation snapshot, not a live retraining or monitoring job.",
        ],
    }


def clear_performance_cache() -> None:
    """Clear cached evaluation-report summaries after a local artifact refresh."""
    get_performance.cache_clear()


def _metadata_value(lines: list[str], prefix: str) -> str:
    try:
        return next(line[len(prefix):].strip().strip("`") for line in lines if line.startswith(prefix))
    except StopIteration:
        raise DashboardDataError(f"The evaluation report is missing {prefix.removesuffix(':')}") from None


def _markdown_cells(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _report_float(value: str) -> float:
    try:
        number = float(value.rstrip("%"))
    except ValueError:
        raise DashboardDataError("The evaluation report contains a non-numeric metric") from None
    if not np.isfinite(number) or not 0 <= number <= 1:
        raise DashboardDataError("The evaluation report contains an out-of-range metric")
    return number


def _report_percent(value: str) -> float:
    if not value.endswith("%"):
        return _report_float(value)
    try:
        percent = float(value[:-1])
    except ValueError:
        raise DashboardDataError("The evaluation report contains an invalid percentage") from None
    if not np.isfinite(percent) or not 0 <= percent <= 100:
        raise DashboardDataError("The evaluation report contains an out-of-range percentage")
    return percent / 100


def _report_int(value: str) -> int:
    try:
        number = int(value.replace(",", ""))
    except ValueError:
        raise DashboardDataError("The evaluation report contains an invalid confusion count") from None
    if number < 0:
        raise DashboardDataError("The evaluation report contains a negative confusion count")
    return number


def _trend(frame: pd.DataFrame, target: pd.Series, column: str, kind: str) -> list[dict]:
    times = frame[column]
    if kind == "calendar":
        parsed = pd.to_datetime(times, errors="coerce")
        if parsed.isna().any():
            raise DashboardDataError("Dataset 1 contains timestamps that cannot be summarized")
        work = pd.DataFrame({"bucket": parsed.dt.floor("D").dt.strftime("%Y-%m-%d"), "fraud": target})
        grouped = work.groupby("bucket", sort=True)["fraud"].agg(total="size", fraud="sum")
        return [{"label": str(label), "total": int(row.total), "fraud": int(row.fraud)} for label, row in grouped.iterrows()]

    numeric = pd.to_numeric(times, errors="coerce")
    if numeric.isna().any() or not np.isfinite(numeric).all() or (numeric < 0).any():
        raise DashboardDataError("Dataset 2 elapsed time values are invalid")
    bucket = (numeric // 7200).astype(int)
    work = pd.DataFrame({"bucket": bucket, "fraud": target})
    grouped = work.groupby("bucket", sort=True)["fraud"].agg(total="size", fraud="sum")
    return [
        {"label": f"{int(index) * 2:02d}–{int(index) * 2 + 2:02d} h", "total": int(row.total), "fraud": int(row.fraud)}
        for index, row in grouped.iterrows()
    ]


def _amount_distribution(values: pd.Series, fraud_mask: pd.Series) -> dict:
    upper = float(values.quantile(0.95))
    if not np.isfinite(upper) or upper <= 0:
        upper = float(values.max()) if float(values.max()) > 0 else 1.0
    edges = np.linspace(0, upper, 11)
    clipped = values.clip(upper=upper)
    bins = np.minimum(np.digitize(clipped, edges[1:-1], right=False), 9)
    fraud = np.bincount(bins[fraud_mask.to_numpy()], minlength=10)
    legitimate = np.bincount(bins[(~fraud_mask).to_numpy()], minlength=10)
    labels = [f"{edges[i]:g}–{edges[i + 1]:g}" for i in range(10)]
    labels[-1] = f"{edges[-2]:g}+"
    return {"labels": labels, "legitimate": legitimate.astype(int).tolist(), "fraud": fraud.astype(int).tolist(), "upper_percentile": 95}


def _category_summary(frame: pd.DataFrame, target: pd.Series, column: str | None) -> list[dict]:
    if column is None or column not in frame:
        return []
    work = pd.DataFrame({"category": frame[column].astype("string").fillna("Unknown"), "fraud": target})
    grouped = work.groupby("category", dropna=False, sort=True)["fraud"].agg(total="size", fraud="sum")
    return [
        {"category": str(category), "total": int(row.total), "fraud": int(row.fraud), "fraud_rate": float(row.fraud / row.total)}
        for category, row in grouped.iterrows()
    ]


def dashboard_shell() -> HTMLResponse:
    """Return the directly accessible, login-free dashboard shell."""
    return HTMLResponse(_DASHBOARD_HTML)


_DASHBOARD_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="theme-color" content="#101821">
  <title>Fraudwatch — Analytics</title>
  <style>
    :root { color-scheme: dark; --bg:#101821; --panel:#18232e; --panel2:#1d2a36; --line:#2a3946; --muted:#91a0ad; --text:#e9f0f4; --mint:#67e0b1; --blue:#80b8ff; --amber:#f3bd66; }
    * { box-sizing:border-box; }
    [hidden] { display:none !important; }
    body { margin:0; background:radial-gradient(ellipse at 72% -15%,#1e3841 0,transparent 42%),var(--bg); color:var(--text); font:14px/1.5 Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif; }
    a { color:inherit; text-decoration:none; }
    .app { min-height:100vh; display:grid; grid-template-columns:248px minmax(0,1fr); }
    aside { padding:25px 16px; border-right:1px solid var(--line); background:#111b24cc; }
    .brand { display:flex; align-items:center; gap:11px; padding:0 10px 29px; font-size:17px; font-weight:700; letter-spacing:-.4px; }
    .mark { width:32px;height:32px;border-radius:11px;background:linear-gradient(145deg,#6ce4b6,#4eaaa8);display:grid;place-items:center;color:#10201e;font-weight:900; }
    .eyebrow { color:var(--muted); text-transform:uppercase; letter-spacing:1.2px; font-size:10px; font-weight:750; padding:0 11px; margin:10px 0 9px; }
    nav { display:grid; gap:5px; }
    nav a { padding:10px 11px; display:flex; align-items:center; gap:11px; border-radius:9px; color:#aab7c0; transition:.16s ease; }
    nav a:hover,nav a.active { color:#effbf6;background:#20342f; }
    nav a.active { box-shadow:inset 2px 0 var(--mint); }
    nav .icon { width:18px;text-align:center;color:#8ba0ad;font-size:15px; }
    .side-note { margin:35px 5px 0; padding:14px; border:1px solid var(--line); border-radius:12px; color:var(--muted); font-size:12px; }
    .side-note strong { display:block;color:#dce7ec;margin-bottom:5px;font-size:12px; }
    main { min-width:0;padding:26px clamp(20px,4vw,56px) 56px;max-width:1600px;width:100%;margin:0 auto; }
    .topline { display:flex;align-items:center;justify-content:space-between;gap:18px;padding-bottom:26px;border-bottom:1px solid var(--line); }
    .crumb { color:var(--muted);font-size:12px; }
    .local { display:flex;align-items:center;gap:8px;color:#b3c0c9;font-size:12px; }
    .dot { width:7px;height:7px;background:var(--mint);border-radius:50%;box-shadow:0 0 12px #67e0b188; }
    .hero { padding:38px 0 27px;display:flex;justify-content:space-between;align-items:end;gap:20px; }
    .kicker { color:var(--mint);font-weight:700;font-size:11px;letter-spacing:1.4px;text-transform:uppercase; }
    h1 { margin:8px 0 8px;font-size:clamp(29px,4vw,43px);line-height:1.1;letter-spacing:-1.6px; }
    .sub { color:var(--muted);margin:0;max-width:600px;font-size:14px; }
    .selector { padding:10px 13px;color:#b7c5cd;background:var(--panel);border:1px solid var(--line);border-radius:9px;font-size:12px;white-space:nowrap; }
    .selector b { color:var(--text);font-weight:600; }
    select { margin-left:8px;padding:7px 25px 7px 9px;border-radius:7px;border:1px solid #435462;background:#18232e;color:var(--text);font:inherit;max-width:235px; }
    .notice { border:1px solid #3b564e;background:linear-gradient(100deg,#1b332e,#1a2a30);border-radius:13px;padding:17px 19px;display:flex;align-items:flex-start;gap:13px;margin:4px 0 22px; }
    .notice-icon { width:28px;height:28px;display:grid;place-items:center;border-radius:9px;color:var(--mint);background:#28483e;flex:0 0 auto; }
    .notice strong { display:block;margin-bottom:3px;font-size:13px; }
    .notice p { color:#aabcb8;margin:0;font-size:12px; }
    .grid { display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px; }
    .card { background:linear-gradient(145deg,#1a2631,#17222c);border:1px solid var(--line);border-radius:13px;padding:19px;min-height:151px; }
    .card-head { display:flex;justify-content:space-between;align-items:center;color:#c5d0d7;font-weight:650;font-size:13px; }
    .tag { font-size:10px;letter-spacing:.4px;color:#a7b6be;border:1px solid #394956;border-radius:20px;padding:3px 8px;font-weight:550; }
    .card p { color:var(--muted);font-size:12px;line-height:1.55;margin:13px 0 0; }
    .placeholder { height:33px;margin-top:18px;border-radius:6px;background:repeating-linear-gradient(135deg,#22313d,#22313d 8px,#1e2b36 8px,#1e2b36 16px);opacity:.7; }
    .bottom { margin-top:22px;padding:16px 18px;border:1px solid var(--line);border-radius:12px;color:var(--muted);font-size:12px;display:flex;justify-content:space-between;gap:20px; }
    .bottom a { color:var(--blue); }
    .kpis { display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px;margin:0 0 17px; }
    .kpi { min-width:0;padding:16px;background:linear-gradient(145deg,#1a2631,#17222c);border:1px solid var(--line);border-radius:12px; }
    .kpi-label { color:var(--muted);font-size:11px; }
    .kpi-value { margin-top:9px;font-size:clamp(18px,2.2vw,26px);font-weight:700;letter-spacing:-.6px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis; }
    .kpi-note { color:#84949f;font-size:10px;margin-top:5px; }
    .charts { display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px; }
    .chart-card { min-width:0;background:linear-gradient(145deg,#1a2631,#17222c);border:1px solid var(--line);border-radius:13px;padding:18px; }
    .chart-card.wide { grid-column:span 2; }
    .chart-title { display:flex;align-items:center;justify-content:space-between;gap:12px;font-weight:650;font-size:13px; }
    .chart-caption { color:var(--muted);font-size:11px;margin:4px 0 13px; }
    .chart { display:block;width:100%;height:220px; }
    .bar-list { display:grid;gap:14px;padding:9px 0; }
    .bar-label { display:flex;justify-content:space-between;gap:12px;color:#c4d0d7;font-size:11px;margin-bottom:6px; }
    .track { height:8px;background:#2a3946;border-radius:10px;overflow:hidden; }
    .fill { height:100%;background:var(--mint);border-radius:10px; }
    .fill.fraud { background:var(--amber); }
    .legend { display:flex;gap:15px;color:var(--muted);font-size:10px;margin-top:9px; }
    .legend i { display:inline-block;width:8px;height:8px;border-radius:2px;margin-right:5px;background:var(--mint); }
    .legend .fraud-key { background:var(--amber); }
    .state { color:var(--muted);font-size:12px;padding:23px 8px;text-align:center; }
    .dataset-note { border-color:#4c4536;background:#29271f; }
    .dataset-note .notice-icon { color:var(--amber);background:#493b25; }
    .planned { margin-top:25px; }
    .planned h2 { font-size:15px;margin:0 0 12px; }
    .performance-section { margin:34px 0 0;scroll-margin-top:20px; }
    .section-head { display:flex;align-items:end;justify-content:space-between;gap:20px;margin-bottom:16px; }
    .section-head h2 { margin:6px 0 0;font-size:23px;letter-spacing:-.5px; }
    .partition-select { color:var(--muted);font-size:11px; }
    .performance-note { background:#242b31;border-color:#3a4852; }
    .model-cards { display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:13px; }
    .model-card { min-width:0;padding:17px;background:linear-gradient(145deg,#1a2631,#17222c);border:1px solid var(--line);border-radius:12px; }
    .model-name { font-weight:700;font-size:14px; }
    .model-meta { color:var(--muted);font-size:10px;margin-top:4px;line-height:1.5; }
    .metric-grid { display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:8px;margin:15px 0; }
    .metric { padding:9px 8px;background:#121c25;border-radius:8px;min-width:0; }
    .metric-label { color:var(--muted);font-size:9px; }
    .metric-value { margin-top:4px;font-weight:700;font-size:13px; }
    .matrix-title { color:#bbc8cf;font-size:10px;margin-bottom:6px; }
    .matrix { display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:1px;border:1px solid var(--line);border-radius:7px;overflow:hidden; }
    .matrix div { background:#1c2933;padding:7px 5px;text-align:center;color:#dce5ea;font-size:10px; }
    .matrix small { display:block;color:var(--muted);font-size:8px;margin-bottom:2px; }
    .performance-charts { display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:13px;margin-top:14px; }
    .curve-card { min-width:0;padding:15px;background:linear-gradient(145deg,#1a2631,#17222c);border:1px solid var(--line);border-radius:12px; }
    .curve-card h3 { margin:0;font-size:12px; }
    .curve-card p { color:var(--muted);font-size:10px;margin:4px 0 10px; }
    .curve-card img { display:block;width:100%;height:auto;border-radius:6px;background:#fff; }
    .limitations { color:var(--muted);font-size:11px;line-height:1.6;padding-left:18px; }
    .limitations li+li { margin-top:3px; }
    section { scroll-margin-top:20px; }
    @media(max-width:1000px){.kpis{grid-template-columns:repeat(3,minmax(0,1fr))}}
    @media(max-width:850px){.app{grid-template-columns:76px minmax(0,1fr)}aside{padding:20px 10px}.brand{justify-content:center;padding:0 0 28px}.brand-name,.eyebrow,.nav-label,.side-note{display:none}nav a{justify-content:center;padding:12px 6px}.grid{grid-template-columns:1fr 1fr}}
    @media(max-width:650px){.charts{grid-template-columns:1fr}.chart-card.wide{grid-column:span 1}.kpis{grid-template-columns:repeat(2,minmax(0,1fr))}}
    @media(max-width:700px){.model-cards,.performance-charts{grid-template-columns:1fr}.section-head{align-items:flex-start;flex-direction:column}.metric-grid{gap:5px}.metric{padding:8px 5px}}
    @media(max-width:560px){.app{display:block}aside{border-right:0;border-bottom:1px solid var(--line);padding:10px 14px}.brand{display:none}nav{display:flex;overflow-x:auto}nav a{flex:0 0 auto;padding:9px 11px}.nav-label{display:inline}.hero{align-items:flex-start;flex-direction:column;padding-top:27px}.grid{grid-template-columns:1fr}.card{min-height:125px}.topline{padding-bottom:16px}.bottom{flex-direction:column}}
  </style>
</head>
<body>
  <div class="app">
    <aside>
      <div class="brand"><span class="mark">F</span><span class="brand-name">fraudwatch</span></div>
      <div class="eyebrow">Workspace</div>
      <nav aria-label="Dashboard navigation">
        <a class="active" href="#overview" aria-current="page"><span class="icon">◫</span><span class="nav-label">Overview</span></a>
        <a href="#performance"><span class="icon">⌁</span><span class="nav-label">Model performance</span></a>
        <a href="#investigations"><span class="icon">⌕</span><span class="nav-label">Investigations</span></a>
        <a href="#data-quality"><span class="icon">▦</span><span class="nav-label">Data quality</span></a>
      </nav>
      <div class="side-note"><strong>Local research workspace</strong>Built for demonstration and academic analysis. Dataset results will remain separate.</div>
    </aside>
    <main id="overview">
      <div class="topline"><div class="crumb">Workspace <span aria-hidden="true">/</span> Overview</div><div class="local"><span class="dot"></span> Local environment</div></div>
      <header class="hero">
        <div><div class="kicker">Fraud analytics</div><h1>See the signal.<br>Understand the risk.</h1><p class="sub">A focused workspace for transaction patterns, model performance, and explainable fraud investigation.</p></div>
        <label class="selector"><b>Dataset</b><select id="dataset-select" aria-label="Select dataset"><option value="dataset2">Credit Card Fraud Dataset</option><option value="dataset1">Synthetic Financial Fraud Dataset</option></select></label>
      </header>
      <div id="dataset-notice" class="notice" hidden><div class="notice-icon">!</div><div><strong>Synthetic sample data</strong><p>Fraud labels in this dataset are randomly generated. These charts are for integration and UI demonstration only.</p></div></div>
      <div id="load-state" class="state" role="status" aria-live="polite">Loading validated dataset summary…</div>
      <div class="kpis" id="kpis" hidden>
        <article class="kpi"><div class="kpi-label">Total transactions</div><div class="kpi-value" id="kpi-total">—</div><div class="kpi-note">Validated source rows</div></article>
        <article class="kpi"><div class="kpi-label">Known fraud labels</div><div class="kpi-value" id="kpi-fraud">—</div><div class="kpi-note">Observed target labels</div></article>
        <article class="kpi"><div class="kpi-label">Label rate</div><div class="kpi-value" id="kpi-rate">—</div><div class="kpi-note">Fraud labels / transactions</div></article>
        <article class="kpi"><div class="kpi-label">Total amount</div><div class="kpi-value" id="kpi-sum">—</div><div class="kpi-note" id="amount-note">Source units, no currency assumed</div></article>
        <article class="kpi"><div class="kpi-label">Average amount</div><div class="kpi-value" id="kpi-average">—</div><div class="kpi-note">Per transaction</div></article>
        <article class="kpi"><div class="kpi-label">High-risk scored</div><div class="kpi-value" id="kpi-highrisk">—</div><div class="kpi-note" id="highrisk-note">Predictive scores are outside this phase</div></article>
      </div>
      <div class="charts" id="charts" hidden>
        <section class="chart-card"><div class="chart-title">Fraud vs. legitimate</div><div class="chart-caption">Counts from the selected dataset’s target labels</div><div id="fraud-split" class="bar-list"></div><div class="legend"><span><i></i>Legitimate</span><span><i class="fraud-key"></i>Fraud</span></div></section>
        <section class="chart-card"><div class="chart-title">Fraud trend</div><div class="chart-caption" id="trend-caption">Fraud labels over time</div><canvas class="chart" id="trend-chart" aria-label="Fraud labels over time"></canvas></section>
        <section class="chart-card wide"><div class="chart-title">Amount distribution</div><div class="chart-caption">Share of each label class per amount band; top 5% grouped in the final band</div><canvas class="chart" id="amount-chart" aria-label="Amount distribution by fraud label"></canvas><div class="legend"><span><i></i>Legitimate</span><span><i class="fraud-key"></i>Fraud</span></div></section>
        <section class="chart-card wide" id="category-card"><div class="chart-title">Fraud by category</div><div class="chart-caption">Dataset 1 category labels; synthetic target correlations are not real-world evidence</div><div id="category-chart" class="bar-list"></div></section>
      </div>
      <div class="notice" id="error-state" role="alert" hidden><div class="notice-icon">!</div><div><strong>Dataset summary unavailable</strong><p id="error-message">Check local dataset configuration and try again.</p></div></div>
      <section class="performance-section" id="performance" aria-labelledby="performance-title">
        <div class="section-head"><div><div class="kicker">Validated evaluation</div><h2 id="performance-title">Model performance</h2></div><label class="partition-select">Evaluation partition<select id="partition-select" aria-label="Evaluation partition"><option value="test">Test</option><option value="validation">Validation</option></select></label></div>
        <div class="notice performance-note"><div class="notice-icon">i</div><div><strong>Dataset 2 evaluation only</strong><p id="split-description">Dataset 1 is synthetic and is intentionally not modeled. These metrics are not combined with Dataset 1.</p></div></div>
        <div id="performance-state" class="state" role="status" aria-live="polite">Loading saved evaluation results…</div>
        <div id="performance-results" hidden>
          <div id="model-cards" class="model-cards"></div>
          <div class="performance-charts">
            <article class="curve-card"><h3>ROC curves</h3><p>Validation and test ranking curves from the saved evaluation.</p><img id="roc-curve" alt="ROC curves for each model on validation and test partitions"></article>
            <article class="curve-card"><h3>Precision–recall curves</h3><p>Average precision (PR-AUC) is important for this highly imbalanced dataset.</p><img id="pr-curve" alt="Precision-recall curves for each model on validation and test partitions"></article>
          </div>
          <ul id="performance-limitations" class="limitations"></ul>
        </div>
      </section>
      <div class="planned"><h2>More analysis views</h2><div class="grid">
        <section class="card" id="investigations"><div class="card-head">Transaction investigations <span class="tag">PLANNED</span></div><p>Submit a transaction, inspect its risk score, and review local SHAP contributors where supported.</p><div class="placeholder" aria-hidden="true"></div></section>
        <section class="card" id="data-quality"><div class="card-head">Data quality <span class="tag">PLANNED</span></div><p>Explore each dataset independently, with clear quality checks and the synthetic dataset labelled accordingly.</p><div class="placeholder" aria-hidden="true"></div></section>
      </div></div>
      </div>
      <footer class="bottom"><span>Local demo · No sign-in required · No customer personal information displayed</span><span>Model scores are decision support, not calibrated guarantees.</span></footer>
    </main>
  </div>
  <script>
    document.querySelectorAll('nav a').forEach(link => link.addEventListener('click', () => {
      document.querySelectorAll('nav a').forEach(item => { item.classList.remove('active'); item.removeAttribute('aria-current'); });
      link.classList.add('active'); link.setAttribute('aria-current', 'page');
    }));

    const formatCount = value => new Intl.NumberFormat().format(value);
    const formatAmount = value => new Intl.NumberFormat(undefined, {maximumFractionDigits: 2, notation: value >= 1e7 ? 'compact' : 'standard'}).format(value);
    const drawLine = (canvas, points, color) => {
      const ctx = canvas.getContext('2d'), ratio = window.devicePixelRatio || 1, width = canvas.clientWidth, height = canvas.clientHeight;
      canvas.width = width * ratio; canvas.height = height * ratio; ctx.scale(ratio, ratio);
      const pad = {l:38,r:12,t:14,b:32}, w=width-pad.l-pad.r, h=height-pad.t-pad.b;
      ctx.clearRect(0,0,width,height); ctx.strokeStyle='#30404c'; ctx.fillStyle='#91a0ad'; ctx.font='10px system-ui';
      for(let i=0;i<4;i++){const y=pad.t+h*i/3;ctx.beginPath();ctx.moveTo(pad.l,y);ctx.lineTo(width-pad.r,y);ctx.stroke();ctx.fillText(String(Math.round((Math.max(...points,1))*(1-i/3))),2,y+3)}
      const max=Math.max(...points,1); ctx.strokeStyle=color;ctx.lineWidth=2;ctx.beginPath();
      points.forEach((v,i)=>{const x=pad.l+(points.length===1?w/2:w*i/(points.length-1)),y=pad.t+h*(1-v/max);i?ctx.lineTo(x,y):ctx.moveTo(x,y)});ctx.stroke();
    };
    const drawAmounts = data => {
      const canvas=document.getElementById('amount-chart'),ctx=canvas.getContext('2d'),ratio=window.devicePixelRatio||1,width=canvas.clientWidth,height=canvas.clientHeight;
      canvas.width=width*ratio;canvas.height=height*ratio;ctx.scale(ratio,ratio);ctx.clearRect(0,0,width,height);
      const pad={l:35,r:8,t:12,b:31},w=width-pad.l-pad.r,h=height-pad.t-pad.b, bins=data.labels.length;
      const legitTotal=data.legitimate.reduce((a,b)=>a+b,0)||1,fraudTotal=data.fraud.reduce((a,b)=>a+b,0)||1;
      const peak=Math.max(...data.legitimate.map((n)=>n/legitTotal*100),...data.fraud.map((n)=>n/fraudTotal*100),1);
      const group=w/bins,bar=Math.max(2,group*.32);ctx.font='9px system-ui';ctx.fillStyle='#91a0ad';ctx.strokeStyle='#30404c';
      for(let i=0;i<3;i++){const y=pad.t+h*i/2;ctx.beginPath();ctx.moveTo(pad.l,y);ctx.lineTo(width-pad.r,y);ctx.stroke();ctx.fillText(`${Math.round(peak*(1-i/2))}%`,2,y+3)}
      data.labels.forEach((label,i)=>{const l=data.legitimate[i]/legitTotal*100,f=data.fraud[i]/fraudTotal*100,x=pad.l+group*i+group*.16;ctx.fillStyle='#67e0b1';ctx.fillRect(x,pad.t+h*(1-l/peak),bar,h*l/peak);ctx.fillStyle='#f3bd66';ctx.fillRect(x+bar+2,pad.t+h*(1-f/peak),bar,h*f/peak);if(i%2===0||i===bins-1)ctx.fillStyle='#91a0ad',ctx.fillText(label.split('–')[0],x-3,height-9)});
    };
    const drawBars = (target, rows, labelKey, countKey, className) => {
      const root=document.getElementById(target);root.replaceChildren();const max=Math.max(...rows.map(row=>row[countKey]),1);
      rows.forEach(row=>{const wrap=document.createElement('div'),line=document.createElement('div'),lab=document.createElement('span'),val=document.createElement('span'),track=document.createElement('div'),fill=document.createElement('div');wrap.className='bar-row';line.className='bar-label';lab.textContent=row[labelKey];val.textContent=formatCount(row[countKey]);track.className='track';fill.className=`fill ${className||''}`;fill.style.width=`${Math.max(row[countKey]/max*100,row[countKey]?1:0)}%`;line.append(lab,val);track.append(fill);wrap.append(line,track);root.append(wrap)});
    };
    const drawCategories = rows => {
      const root=document.getElementById('category-chart');root.replaceChildren();const max=Math.max(...rows.map(row=>row.fraud_rate),.01);
      rows.forEach(row=>{const wrap=document.createElement('div'),line=document.createElement('div'),lab=document.createElement('span'),val=document.createElement('span'),track=document.createElement('div'),fill=document.createElement('div');wrap.className='bar-row';line.className='bar-label';lab.textContent=row.category;val.textContent=`${(row.fraud_rate*100).toFixed(1)}% · ${formatCount(row.fraud)} labelled`;track.className='track';fill.className='fill fraud';fill.style.width=`${Math.max(row.fraud_rate/max*100,row.fraud_rate?1:0)}%`;line.append(lab,val);track.append(fill);wrap.append(line,track);root.append(wrap)});
    };
    async function loadOverview(){
      const dataset=document.getElementById('dataset-select').value,state=document.getElementById('load-state'),error=document.getElementById('error-state');
      state.hidden=false;state.textContent='Loading validated dataset summary…';error.hidden=true;document.getElementById('kpis').hidden=true;document.getElementById('charts').hidden=true;
      try{
        const response=await fetch(`/dashboard/overview?dataset=${encodeURIComponent(dataset)}`);if(!response.ok)throw new Error((await response.json()).detail||'Dataset summary unavailable');const data=await response.json(),k=data.kpis;
        document.getElementById('kpi-total').textContent=formatCount(k.total_transactions);document.getElementById('kpi-fraud').textContent=formatCount(k.fraud_transactions);document.getElementById('kpi-rate').textContent=`${(k.fraud_rate*100).toFixed(dataset==='dataset2'?3:2)}%`;document.getElementById('kpi-sum').textContent=formatAmount(k.total_amount);document.getElementById('kpi-average').textContent=formatAmount(k.average_amount);document.getElementById('kpi-highrisk').textContent=k.high_risk_transactions==null?'—':formatCount(k.high_risk_transactions);
        document.getElementById('dataset-notice').hidden=!data.dataset.synthetic;document.getElementById('category-card').hidden=!data.category_fraud.length;document.getElementById('amount-note').textContent=data.amount_note;state.hidden=true;document.getElementById('kpis').hidden=false;document.getElementById('charts').hidden=false;
        drawBars('fraud-split',[{label:'Legitimate',count:data.fraud_split.legitimate},{label:'Fraud',count:data.fraud_split.fraud}],'label','count');document.querySelectorAll('#fraud-split .bar-row')[1]?.querySelector('.fill')?.classList.add('fraud');
        document.getElementById('trend-caption').textContent=data.dataset.id==='dataset1'?'Known fraud labels by calendar day':'Known fraud labels by two-hour elapsed-time interval (source Time field)';drawLine(document.getElementById('trend-chart'),data.trend.map(row=>row.fraud),'#f3bd66');drawAmounts(data.amount_distribution);
        drawCategories(data.category_fraud);
        document.querySelector('.crumb').innerHTML=`Workspace <span aria-hidden="true">/</span> ${data.dataset.label}`;
      }catch(err){state.hidden=true;error.hidden=false;document.getElementById('error-message').textContent=err.message}
    }
    const metricLabels=[['precision','Precision'],['recall','Recall'],['f1','F1'],['roc_auc','ROC-AUC'],['pr_auc','PR-AUC']];
    function renderModels(models){
      const root=document.getElementById('model-cards');root.replaceChildren();
      models.forEach(model=>{
        const card=document.createElement('article');card.className='model-card';
        const header=document.createElement('div');header.className='card-head';const name=document.createElement('span');name.className='model-name';name.textContent=model.name;const type=document.createElement('span');type.className='tag';type.textContent=model.score_kind==='anomaly'?'ANOMALY SCORE':'PROBABILITY';header.append(name,type);
        const rule=document.createElement('div');rule.className='model-meta';rule.textContent=`Fixed decision: ${model.decision_rule} · ${formatCount(model.rows)} rows · prevalence ${(model.prevalence*100).toFixed(3)}% · alert rate ${(model.alert_rate*100).toFixed(3)}%`;
        const metrics=document.createElement('div');metrics.className='metric-grid';metricLabels.forEach(([key,label])=>{const item=document.createElement('div');item.className='metric';const metricName=document.createElement('div');metricName.className='metric-label';metricName.textContent=label;const value=document.createElement('div');value.className='metric-value';value.textContent=(model.metrics[key]*100).toFixed(2)+'%';item.append(metricName,value);metrics.append(item)});
        const matrixTitle=document.createElement('div');matrixTitle.className='matrix-title';matrixTitle.textContent='Confusion matrix · actual × predicted';
        const matrix=document.createElement('div');matrix.className='matrix';[['TN','True negative','tn'],['FP','False positive','fp'],['FN','False negative','fn'],['TP','True positive','tp']].forEach(([short,long,key])=>{const cell=document.createElement('div'),label=document.createElement('small');label.textContent=short;cell.setAttribute('aria-label',`${long}: ${formatCount(model.confusion_matrix[key])}`);cell.append(label,document.createTextNode(formatCount(model.confusion_matrix[key])));matrix.append(cell)});
        card.append(header,rule,metrics,matrixTitle,matrix);root.append(card);
      });
    }
    async function loadPerformance(){
      const state=document.getElementById('performance-state'),results=document.getElementById('performance-results');state.hidden=false;state.textContent='Loading saved evaluation results…';results.hidden=true;
      try{
        const partition=document.getElementById('partition-select').value,response=await fetch(`/dashboard/performance?partition=${partition}`);if(!response.ok)throw new Error((await response.json()).detail||'Evaluation results unavailable');const data=await response.json();
        document.getElementById('split-description').textContent=`${data.dataset1_note} Split: ${data.split_strategy}`;renderModels(data.models);document.getElementById('roc-curve').src=data.curves.roc;document.getElementById('pr-curve').src=data.curves.precision_recall;
        const list=document.getElementById('performance-limitations');list.replaceChildren();data.limitations.forEach(note=>{const item=document.createElement('li');item.textContent=note;list.append(item)});
        state.hidden=true;results.hidden=false;
      }catch(err){state.textContent=err.message;}
    }
    document.getElementById('dataset-select').addEventListener('change',loadOverview);document.getElementById('partition-select').addEventListener('change',loadPerformance);window.addEventListener('resize',()=>{if(!document.getElementById('charts').hidden)loadOverview()});loadOverview();loadPerformance();
  </script>
</body>
</html>"""
