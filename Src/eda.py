"""Descriptive EDA summaries and reports for the two separate fraud datasets."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from Src.dataset1_integration import load_dataset1_transaction_table
from Src.dataset2 import EXPECTED_COLUMNS as DATASET2_COLUMNS
from Src.ingestion import PROJECT_ROOT, load_dataset2

LOGGER = logging.getLogger(__name__)
CLASS_LABEL = "FraudIndicator"
DATASET2_TARGET = "Class"
PCA_COLUMNS = tuple(f"V{i}" for i in range(1, 29))


class EDAError(ValueError):
    """Raised when the provided dataset cannot support the requested summaries."""


@dataclass(frozen=True)
class EDAArtifacts:
    dataset1_report: Path
    dataset2_report: Path
    dataset1_chart: Path
    dataset2_chart: Path
    dataset2_pca_chart: Path


def profile_dataset1(frame: pd.DataFrame) -> dict:
    """Summarize the synthetic Dataset 1 analytical table without causal claims."""
    _require_frame_columns(frame, (CLASS_LABEL, "Amount", "TransactionAmount", "Timestamp", "Category"), "Dataset 1")
    target_counts = _target_counts(frame, CLASS_LABEL)
    timestamp = pd.to_datetime(frame["Timestamp"], errors="coerce")
    daily = frame.assign(_day=timestamp.dt.floor("D")).dropna(subset=["_day"])
    daily_summary = (
        daily.groupby("_day", sort=True)[CLASS_LABEL]
        .agg(transaction_count="size", fraud_count="sum")
        .reset_index()
    )
    category_summary = (
        frame.groupby("Category", dropna=False)[CLASS_LABEL]
        .agg(transaction_count="size", fraud_count="sum", fraud_rate="mean")
        .reset_index()
        .sort_values("Category", kind="stable")
    )
    return {
        "dataset_label": "Synthetic Financial Fraud Dataset",
        "synthetic": True,
        "rows": len(frame),
        "columns": len(frame.columns),
        "fraud_counts": target_counts,
        "fraud_rate": target_counts.get(1, 0) / len(frame) if len(frame) else 0.0,
        "missing_values": {str(c): int(n) for c, n in frame.isna().sum().items() if n},
        "duplicate_rows": int(frame.duplicated().sum()),
        "time_min": timestamp.min(),
        "time_max": timestamp.max(),
        "unparsed_timestamps": int(timestamp.isna().sum()),
        "daily": daily_summary,
        "category": category_summary,
        "amount_by_class": {
            column: _numeric_summary_by_target(frame, column, CLASS_LABEL)
            for column in ("Amount", "TransactionAmount")
        },
        "column_names": tuple(frame.columns),
        "_frame": frame,
    }


def profile_dataset2(frame: pd.DataFrame) -> dict:
    """Summarize the credit-card benchmark; PCA components remain uninterpreted."""
    required = (DATASET2_TARGET, "Amount", "Time", *PCA_COLUMNS)
    _require_frame_columns(frame, required, "Dataset 2")
    target_counts = _target_counts(frame, DATASET2_TARGET)
    amount_by_class = _numeric_summary_by_target(frame, "Amount", DATASET2_TARGET)
    time_by_class = _numeric_summary_by_target(frame, "Time", DATASET2_TARGET)
    pca_summary: dict[str, dict] = {}
    for component in PCA_COLUMNS:
        pca_summary[component] = {}
        for label in (0, 1):
            values = pd.to_numeric(frame.loc[frame[DATASET2_TARGET] == label, component], errors="coerce")
            pca_summary[component][label] = {
                "count": int(values.count()),
                "mean": _finite_float(values.mean()),
                "std": _finite_float(values.std()),
                "median": _finite_float(values.median()),
            }
    return {
        "dataset_label": "Credit Card Fraud Dataset",
        "synthetic": False,
        "rows": len(frame),
        "columns": len(frame.columns),
        "fraud_counts": target_counts,
        "fraud_rate": target_counts.get(1, 0) / len(frame) if len(frame) else 0.0,
        "missing_values": {str(c): int(n) for c, n in frame.isna().sum().items() if n},
        "duplicate_rows": int(frame.duplicated().sum()),
        "amount_by_class": amount_by_class,
        "time_by_class": time_by_class,
        "pca_summary": pca_summary,
    }


def generate_eda_reports(
    output_dir: str | Path | None = None,
    *,
    dataset1_root: str | Path | None = None,
    dataset2_project_root: str | Path | None = None,
    downloader=None,
) -> EDAArtifacts:
    """Load each dataset separately and write report Markdown and static charts."""
    out = Path(output_dir) if output_dir else PROJECT_ROOT / "reports" / "eda"
    out = out.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    dataset1 = load_dataset1_transaction_table(dataset1_root)
    dataset2 = load_dataset2(project_root=dataset2_project_root, downloader=downloader)
    summary1 = profile_dataset1(dataset1.frame)
    summary2 = profile_dataset2(dataset2.frame)

    d1_chart = out / "dataset1_overview.png"
    d2_chart = out / "dataset2_overview.png"
    pca_chart = out / "dataset2_pca_components.png"
    _plot_dataset1(summary1, d1_chart)
    _plot_dataset2(dataset2.frame, d2_chart)
    _plot_pca_means(summary2, pca_chart)

    d1_report = out / "dataset1_eda.md"
    d2_report = out / "dataset2_eda.md"
    d1_report.write_text(_render_dataset1(summary1, dataset1.report), encoding="utf-8")
    d2_report.write_text(_render_dataset2(summary2, dataset2.report), encoding="utf-8")
    LOGGER.info(
        "eda_reports_generated",
        extra={
            "event_type": "eda_reports_generated",
            "dataset1_rows": summary1["rows"],
            "dataset2_rows": summary2["rows"],
            "output_directory": str(out),
        },
    )
    return EDAArtifacts(d1_report, d2_report, d1_chart, d2_chart, pca_chart)


def _target_counts(frame: pd.DataFrame, target: str) -> dict[object, int]:
    if frame.empty:
        raise EDAError(f"Cannot profile empty dataset (target {target})")
    values = frame[target]
    if values.isna().any() or not values.isin({0, 1}).all():
        raise EDAError(f"Target {target!r} must contain only non-null 0/1 values")
    return {int(label): int(count) for label, count in values.value_counts(sort=False).items()}


def _numeric_summary_by_target(frame: pd.DataFrame, column: str, target: str) -> dict[int, dict[str, float | int | None]]:
    result: dict[int, dict[str, float | int | None]] = {}
    for label in (0, 1):
        values = pd.to_numeric(frame.loc[frame[target] == label, column], errors="coerce").dropna()
        result[label] = {
            "count": int(values.count()),
            "mean": _finite_float(values.mean()),
            "median": _finite_float(values.median()),
            "p95": _finite_float(values.quantile(0.95)) if not values.empty else None,
            "max": _finite_float(values.max()) if not values.empty else None,
        }
    return result


def _require_frame_columns(frame: pd.DataFrame, columns: tuple[str, ...], dataset_name: str) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise EDAError(f"{dataset_name} profile requires a pandas DataFrame")
    missing = [column for column in columns if column not in frame]
    if missing:
        raise EDAError(f"{dataset_name} is missing columns needed for EDA: {missing}")


def _finite_float(value) -> float | None:
    if pd.isna(value) or not np.isfinite(value):
        return None
    return float(value)


def _render_dataset1(summary: dict, join_report) -> str:
    lines = [
        "# Dataset 1 EDA — Synthetic Financial Fraud Dataset", "",
        "> **Synthetic data notice:** fraud and suspicious indicators are randomly generated. These results do not represent real financial behavior or causal relationships.", "",
        "## Data profile", "",
        f"- Transactions: {summary['rows']:,}; analytical columns: {summary['columns']}",
        f"- Join row-count check: {join_report.input_transaction_rows:,} input → {join_report.output_rows:,} output",
        f"- FraudIndicator: {summary['fraud_counts'].get(1, 0):,} positive / {summary['fraud_counts'].get(0, 0):,} legitimate ({summary['fraud_rate']:.2%} positive)",
        f"- Missing values: {sum(summary['missing_values'].values()):,}; exact duplicate rows: {summary['duplicate_rows']:,}",
        f"- Timestamp span: {_date(summary['time_min'])} to {_date(summary['time_max'])}; unparsed timestamps: {summary['unparsed_timestamps']}",
        "", "## Amounts by generated target", "",
        "`Amount` and `TransactionAmount` are separately generated fields and are summarized independently.", "",
        "| Field | Target | Count | Mean | Median | 95th percentile | Maximum |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for column, classes in summary["amount_by_class"].items():
        for label, stats in classes.items():
            lines.append(_amount_row(column, label, stats))
    lines.extend(["", "## Fraud counts by category", "", "| Category | Transactions | FraudIndicator=1 | Synthetic positive rate |", "|---|---:|---:|---:|"])
    for row in summary["category"].itertuples(index=False):
        lines.append(f"| {_cell(row.Category)} | {row.transaction_count} | {row.fraud_count} | {row.fraud_rate:.2%} |")
    lines.extend(["", "## Time overview", "", f"Daily bins with parseable timestamps: {len(summary['daily'])}.", "The plotted daily counts are descriptive only; this synthetic label was sampled independently of transaction behavior.", "", "![Dataset 1 synthetic overview](dataset1_overview.png)", "", "## Limitations", "", "The result is suitable for integration and analytics demonstrations only. Fraud rates and category/time patterns are synthetic observations, not financial-industry estimates. Customer name, address, age, account-balance, and login-time fields were excluded from the analytical table.", ""])
    return "\n".join(lines)


def _render_dataset2(summary: dict, load_report) -> str:
    lines = [
        "# Dataset 2 EDA — Credit Card Fraud Dataset", "",
        "## Data profile", "",
        f"- Transactions: {summary['rows']:,}; columns: {summary['columns']}",
        f"- Class 1 fraud: {summary['fraud_counts'].get(1, 0):,}; Class 0 legitimate: {summary['fraud_counts'].get(0, 0):,}",
        f"- Positive rate: {summary['fraud_rate']:.4%}; class imbalance (legitimate:fraud): {_imbalance(summary['fraud_counts'])}",
        f"- Missing values: {sum(summary['missing_values'].values()):,}; exact duplicate rows: {summary['duplicate_rows']:,} (preserved)",
        f"- Loader encoding: {load_report.encoding}",
        "", "## Amount by class", "",
        "Amount is reported in the dataset's source units; no currency assumption is added.", "",
        "| Class | Count | Mean | Median | 95th percentile | Maximum |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for label, stats in summary["amount_by_class"].items():
        lines.append(_amount_row("", label, stats, include_name=False))
    lines.extend(["", "## Time field by class", "", "`Time` is summarized in the source's numeric scale and is not converted to calendar timestamps.", "", "| Class | Count | Mean | Median | 95th percentile | Maximum |", "|---:|---:|---:|---:|---:|---:|"])
    for label, stats in summary["time_by_class"].items():
        lines.append(_amount_row("", label, stats, include_name=False))
    lines.extend(["", "## V1–V28 PCA component summaries", "", "V1–V28 are anonymized PCA components. Their names have no disclosed business interpretation; class-wise means below are descriptive summaries, not feature importance or causal effects.", "", "| Component | Class 0 mean | Class 1 mean | Class 0 std | Class 1 std |", "|---|---:|---:|---:|---:|"])
    for component, classes in summary["pca_summary"].items():
        lines.append(f"| {component} | {_number(classes[0]['mean'])} | {_number(classes[1]['mean'])} | {_number(classes[0]['std'])} | {_number(classes[1]['std'])} |")
    lines.extend(["", "![Dataset 2 overview](dataset2_overview.png)", "", "![PCA component class-wise means](dataset2_pca_components.png)", "", "## Limitations", "", "These are descriptive benchmark-dataset statistics only. No model has been fitted, no threshold selected, and no fraud effectiveness claimed. Exact duplicate rows are retained; downstream data preparation must decide and document their treatment.", ""])
    return "\n".join(lines)


def _plot_dataset1(summary: dict, destination: Path) -> None:
    plt = _pyplot()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    counts = summary["fraud_counts"]
    axes[0].bar(["Legitimate", "Fraud"], [counts.get(0, 0), counts.get(1, 0)], color=["#4778a8", "#c55252"])
    axes[0].set_title("Synthetic label counts")
    axes[0].set_ylabel("Transactions")
    frame = summary["_frame"]
    amount_values = pd.to_numeric(frame["Amount"], errors="coerce").dropna()
    log_amount = bool((amount_values >= 0).all())
    x_label = "log1p(Amount)" if log_amount else "Amount (source scale; negative values present)"
    for label, color, name in ((0, "#4778a8", "Legitimate"), (1, "#c55252", "Fraud")):
        values = pd.to_numeric(frame.loc[frame[CLASS_LABEL] == label, "Amount"], errors="coerce").dropna()
        transformed = np.log1p(values) if log_amount else values
        if not transformed.empty:
            axes[1].hist(transformed, bins=25, alpha=0.6, color=color, label=name, density=True)
    axes[1].set_title("Amount by generated label")
    axes[1].set_xlabel(x_label)
    axes[1].legend()
    daily = summary["daily"]
    if not daily.empty:
        axes[2].plot(daily["_day"], daily["transaction_count"], color="#4778a8", label="Transactions")
        axes[2].plot(daily["_day"], daily["fraud_count"], color="#c55252", label="FraudIndicator=1")
    axes[2].set_title("Daily synthetic counts")
    axes[2].tick_params(axis="x", rotation=45)
    axes[2].legend()
    fig.suptitle("Dataset 1 — synthetic analytics only")
    fig.savefig(destination, dpi=150)
    plt.close(fig)


def _plot_dataset2(frame: pd.DataFrame, destination: Path) -> None:
    plt = _pyplot()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    counts = frame[DATASET2_TARGET].value_counts().reindex([0, 1], fill_value=0)
    axes[0].bar(["Legitimate", "Fraud"], counts.values, color=["#4778a8", "#c55252"])
    axes[0].set_title("Observed class counts")
    axes[0].set_ylabel("Transactions")
    amount_values = pd.to_numeric(frame["Amount"], errors="coerce").dropna()
    log_amount = bool((amount_values >= 0).all())
    x_label = "log1p(Amount)" if log_amount else "Amount (source scale; negative values present)"
    for label, color, name in ((0, "#4778a8", "Legitimate"), (1, "#c55252", "Fraud")):
        values = pd.to_numeric(frame.loc[frame[DATASET2_TARGET] == label, "Amount"], errors="coerce").dropna()
        transformed = np.log1p(values) if log_amount else values
        if not transformed.empty:
            axes[1].hist(transformed, bins=40, alpha=0.6, color=color, label=name, density=True)
        time_values = pd.to_numeric(frame.loc[frame[DATASET2_TARGET] == label, "Time"], errors="coerce").dropna()
        if not time_values.empty:
            axes[2].hist(time_values, bins=40, alpha=0.6, color=color, label=name, density=True)
    axes[1].set_title("Amount by class")
    axes[1].set_xlabel(x_label)
    axes[1].legend()
    axes[2].set_title("Time by class")
    axes[2].set_xlabel("Time (source scale)")
    axes[2].legend()
    fig.suptitle("Dataset 2 — descriptive benchmark profile")
    fig.savefig(destination, dpi=150)
    plt.close(fig)


def _plot_pca_means(summary: dict, destination: Path) -> None:
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
    components = list(summary["pca_summary"])
    for label, color, name in ((0, "#4778a8", "Class 0"), (1, "#c55252", "Class 1")):
        values = [summary["pca_summary"][component][label]["mean"] for component in components]
        values = [np.nan if value is None else value for value in values]
        ax.plot(components, values, marker="o", markersize=3, color=color, label=name)
    ax.set_title("Anonymized PCA component means by class (descriptive, not importance)")
    ax.set_ylabel("Mean component value")
    ax.set_xlabel("Anonymized PCA component")
    ax.tick_params(axis="x", rotation=45)
    ax.legend()
    fig.savefig(destination, dpi=150)
    plt.close(fig)


def _pyplot():
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


def _amount_row(name: str, label: int, stats: Mapping, *, include_name: bool = True) -> str:
    cells = [name] if include_name else []
    cells.extend([str(label), str(stats["count"]), _number(stats["mean"]), _number(stats["median"]), _number(stats["p95"]), _number(stats["max"])])
    return "| " + " | ".join(cells) + " |"


def _number(value) -> str:
    return "—" if value is None else f"{value:,.4f}"


def _date(value) -> str:
    return "—" if pd.isna(value) else value.isoformat(sep=" ")


def _cell(value) -> str:
    return "(missing)" if pd.isna(value) else str(value).replace("|", "\\|")


def _imbalance(counts: Mapping[int, int]) -> str:
    positive = counts.get(1, 0)
    negative = counts.get(0, 0)
    return "undefined" if positive == 0 else f"{negative / positive:,.1f}:1"
