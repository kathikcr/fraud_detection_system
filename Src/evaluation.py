"""Unified holdout evaluation and comparison artifacts for fitted baselines."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_recall_curve, roc_curve

from Src.model_metrics import BinaryMetrics, MetricsError, measure_binary_probabilities, measure_binary_scores, validate_prepared_splits
from Src.preprocessing import PreparedSplits

LOGGER = logging.getLogger(__name__)
ScoreKind = Literal["probability", "anomaly"]


class EvaluationError(ValueError):
    """Raised when a fitted estimator or evaluation input is invalid."""


@dataclass(frozen=True)
class EvaluationSpec:
    name: str
    estimator: object
    score_kind: ScoreKind
    threshold: float = 0.5


@dataclass(frozen=True)
class PartitionEvaluation:
    metrics: BinaryMetrics
    prevalence: float
    alert_rate: float
    roc_fpr: tuple[float, ...]
    roc_tpr: tuple[float, ...]
    pr_precision: tuple[float, ...]
    pr_recall: tuple[float, ...]


@dataclass(frozen=True)
class ModelEvaluation:
    name: str
    score_kind: ScoreKind
    decision_rule: str
    validation: PartitionEvaluation
    test: PartitionEvaluation


@dataclass(frozen=True)
class EvaluationReport:
    target_column: str
    split_strategy: str
    models: tuple[ModelEvaluation, ...]


@dataclass(frozen=True)
class EvaluationArtifacts:
    report: Path
    roc_chart: Path
    precision_recall_chart: Path
    confusion_matrix_chart: Path


def evaluate_models(splits: PreparedSplits, specs: Sequence[EvaluationSpec]) -> EvaluationReport:
    """Measure already-fitted models on validation and test without fitting/tuning.

    Supervised models use positive-class probabilities and each spec's fixed
    threshold. Isolation Forest uses its native `predict` decision for confusion
    metrics and negated `score_samples` for ranking metrics. No score threshold
    is selected in this function.
    """
    try:
        validate_prepared_splits(splits, expected_target="Class")
    except MetricsError as exc:
        raise EvaluationError(str(exc)) from exc
    if not specs:
        raise EvaluationError("At least one fitted model specification is required")
    names = [spec.name.strip() for spec in specs]
    if any(not name for name in names) or len(set(names)) != len(names):
        raise EvaluationError("Model names must be non-empty and unique")

    results: list[ModelEvaluation] = []
    for spec in specs:
        if spec.score_kind not in ("probability", "anomaly"):
            raise EvaluationError(f"Unsupported score kind for model {spec.name!r}: {spec.score_kind}")
        if spec.score_kind == "probability":
            try:
                threshold = float(spec.threshold)
            except (TypeError, ValueError) as exc:
                raise EvaluationError(f"Probability threshold for {spec.name!r} must be numeric") from exc
            if not np.isfinite(threshold) or not 0 <= threshold <= 1:
                raise EvaluationError(f"Probability threshold for {spec.name!r} must be between 0 and 1")
        _validate_estimator_features(spec.estimator, splits.feature_names, spec.name)
        if spec.score_kind == "probability" and not hasattr(spec.estimator, "predict_proba"):
            raise EvaluationError(f"Model {spec.name!r} has no predict_proba method")
        if spec.score_kind == "anomaly" and not all(
            hasattr(spec.estimator, method) for method in ("score_samples", "predict")
        ):
            raise EvaluationError(f"Anomaly model {spec.name!r} must implement score_samples and predict")

        validation = _evaluate_partition(
            spec, splits.X_validation, splits.y_validation
        )
        test = _evaluate_partition(spec, splits.X_test, splits.y_test)
        if spec.score_kind == "probability":
            rule = f"probability >= {float(spec.threshold):.4g} (fixed)"
        else:
            estimator_name = type(spec.estimator).__name__
            params = spec.estimator.get_params(deep=False) if hasattr(spec.estimator, "get_params") else {}
            contamination = params.get("contamination", "unspecified")
            rule = f"{estimator_name} native predict cutoff (contamination={contamination})"
        results.append(ModelEvaluation(spec.name, spec.score_kind, rule, validation, test))
    LOGGER.info(
        "model_evaluation_complete",
        extra={"event_type": "model_evaluation_complete", "models": names,
               "validation_rows": len(splits.y_validation), "test_rows": len(splits.y_test),
               "thresholds_tuned": False},
    )
    return EvaluationReport(splits.target_column, splits.split_strategy, tuple(results))


def write_evaluation_artifacts(
    report: EvaluationReport,
    output_dir: str | Path,
) -> EvaluationArtifacts:
    """Write the compact comparison report, ROC/PR plots, and test confusion grids."""
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    report_path = destination / "model_comparison.md"
    roc_path = destination / "roc_curves.png"
    pr_path = destination / "precision_recall_curves.png"
    confusion_path = destination / "test_confusion_matrices.png"
    report_path.write_text(_render_report(report), encoding="utf-8")
    _plot_curves(report, roc_path, curve="roc")
    _plot_curves(report, pr_path, curve="pr")
    _plot_confusion_matrices(report, confusion_path)
    return EvaluationArtifacts(report_path, roc_path, pr_path, confusion_path)


def _evaluate_partition(spec: EvaluationSpec, features: pd.DataFrame, target: pd.Series) -> PartitionEvaluation:
    if spec.score_kind == "probability":
        estimator_classes = list(getattr(spec.estimator, "classes_", ()))
        if 1 not in estimator_classes:
            raise EvaluationError(f"Model {spec.name!r} was not fitted with fraud class 1")
        probabilities = np.asarray(spec.estimator.predict_proba(features), dtype=float)
        if probabilities.ndim != 2 or probabilities.shape[0] != len(features):
            raise EvaluationError(f"Model {spec.name!r} returned malformed probability output")
        scores = probabilities[:, estimator_classes.index(1)]
        threshold = float(spec.threshold)
        metrics = measure_binary_probabilities(target, scores, threshold=threshold)
        predictions = (scores >= threshold).astype(int)
    else:
        estimator = spec.estimator
        labels = np.asarray(estimator.predict(features))
        if labels.ndim != 1 or len(labels) != len(features) or not np.isin(labels, [-1, 1]).all():
            raise EvaluationError(f"Anomaly model {spec.name!r} must return -1/+1 outlier decisions")
        scores = -np.asarray(estimator.score_samples(features), dtype=float)
        predictions = (labels == -1).astype(int)
        metrics = measure_binary_scores(target, scores, predictions)

    fpr, tpr, _ = roc_curve(target, scores, pos_label=1, drop_intermediate=True)
    precision, recall, _ = precision_recall_curve(target, scores, pos_label=1)
    return PartitionEvaluation(
        metrics=metrics,
        prevalence=float(target.mean()),
        alert_rate=float(predictions.mean()),
        roc_fpr=tuple(float(value) for value in fpr),
        roc_tpr=tuple(float(value) for value in tpr),
        pr_precision=tuple(float(value) for value in precision),
        pr_recall=tuple(float(value) for value in recall),
    )


def _validate_estimator_features(estimator, feature_names: tuple[str, ...], name: str) -> None:
    if not hasattr(estimator, "n_features_in_"):
        raise EvaluationError(f"Model {name!r} is not fitted")
    if int(estimator.n_features_in_) != len(feature_names):
        raise EvaluationError(f"Model {name!r} expects {estimator.n_features_in_} features, got {len(feature_names)}")
    fitted_names = getattr(estimator, "feature_names_in_", None)
    if fitted_names is not None and tuple(fitted_names) != feature_names:
        raise EvaluationError(f"Model {name!r} fitted feature names do not match prepared split columns")


def _render_report(report: EvaluationReport) -> str:
    lines = [
        "# Dataset 2 Model Evaluation", "",
        "All estimators were fitted before this evaluation. The utility did not refit models, tune thresholds, or select a winner.", "",
        f"- Target: `{report.target_column}` (fraud class 1)",
        f"- Split: {report.split_strategy}",
        "- Dataset 1 is synthetic and is not combined with Dataset 2 model metrics.",
        "- PR-AUC is average precision. Precision/recall/F1 and confusion counts use each model's stated fixed decision rule.", "",
        "## Validation results", "",
        _metrics_table(report.models, "validation"), "",
        "## Test results", "",
        _metrics_table(report.models, "test"), "",
        "![ROC curves for validation and test](roc_curves.png)", "",
        "![Precision-recall curves for validation and test](precision_recall_curves.png)", "",
        "![Test confusion matrices](test_confusion_matrices.png)", "",
        "## Interpretation limits", "",
        "These values describe the single observed chronological dataset snapshot. A model may rank cases well while its fixed cutoff produces an unsuitable alert rate. Isolation Forest's anomaly scores are not fraud probabilities, and its outlier decision uses `contamination=auto`; do not compare that operating rule to probability thresholds as if they were equivalent. No test-derived threshold or model selection decision is made here. The final test partition remains an evaluation set, not a tuning set.", "",
    ]
    return "\n".join(lines)


def _metrics_table(models: tuple[ModelEvaluation, ...], partition_name: str) -> str:
    lines = [
        "| Model | Score type | Decision rule | Prevalence | Alert rate | Precision | Recall | F1 | ROC-AUC | PR-AUC | TN | FP | FN | TP |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model in models:
        partition = getattr(model, partition_name)
        metrics = partition.metrics
        lines.append(
            f"| {model.name} | {model.score_kind} | {model.decision_rule} | {partition.prevalence:.4%} | "
            f"{partition.alert_rate:.4%} | {metrics.precision:.6f} | {metrics.recall:.6f} | {metrics.f1:.6f} | "
            f"{metrics.roc_auc:.6f} | {metrics.pr_auc:.6f} | {metrics.true_negative:,} | "
            f"{metrics.false_positive:,} | {metrics.false_negative:,} | {metrics.true_positive:,} |"
        )
    return "\n".join(lines)


def _plot_curves(report: EvaluationReport, destination: Path, *, curve: Literal["roc", "pr"]) -> None:
    plt = _pyplot()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for ax, partition_name, title in zip(axes, ("validation", "test"), ("Validation", "Test")):
        for model in report.models:
            result = getattr(model, partition_name)
            if curve == "roc":
                ax.plot(result.roc_fpr, result.roc_tpr, label=f"{model.name} (AUC {result.metrics.roc_auc:.3f})")
            else:
                ax.plot(result.pr_recall, result.pr_precision, label=f"{model.name} (AP {result.metrics.pr_auc:.3f})")
        if curve == "roc":
            ax.plot([0, 1], [0, 1], linestyle="--", color="#777777", linewidth=1, label="Chance")
            ax.set_xlabel("False positive rate")
            ax.set_ylabel("True positive rate")
            ax.set_title(f"{title} ROC")
        else:
            prevalence = getattr(report.models[0], partition_name).prevalence
            ax.axhline(prevalence, linestyle="--", color="#777777", linewidth=1, label=f"Prevalence {prevalence:.3%}")
            ax.set_xlabel("Recall")
            ax.set_ylabel("Precision")
            ax.set_ylim(0, 1.02)
            ax.set_title(f"{title} precision-recall")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)
    fig.suptitle("Dataset 2 models — fixed holdout evaluation")
    fig.savefig(destination, dpi=150)
    plt.close(fig)


def _plot_confusion_matrices(report: EvaluationReport, destination: Path) -> None:
    plt = _pyplot()
    count = len(report.models)
    columns = 2
    rows = (count + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(9, 4.2 * rows), squeeze=False, constrained_layout=True)
    for ax, model in zip(axes.flat, report.models):
        metrics = model.test.metrics
        matrix = np.array([[metrics.true_negative, metrics.false_positive],
                           [metrics.false_negative, metrics.true_positive]])
        ax.imshow(matrix, cmap="Blues")
        ax.set_xticks([0, 1], labels=["Legitimate", "Fraud"])
        ax.set_yticks([0, 1], labels=["Legitimate", "Fraud"])
        ax.set_xlabel("Predicted class")
        ax.set_ylabel("Actual class")
        ax.set_title(f"{model.name} — test")
        midpoint = matrix.max() / 2 if matrix.size else 0
        for (row, col), value in np.ndenumerate(matrix):
            ax.text(col, row, f"{value:,}", ha="center", va="center",
                    color="white" if value > midpoint else "black")
    for ax in axes.flat[count:]:
        ax.axis("off")
    fig.suptitle("Test confusion matrices (fixed model decision rules)")
    fig.savefig(destination, dpi=150)
    plt.close(fig)


def _pyplot():
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt
