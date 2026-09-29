"""First supervised benchmark: class-weighted Logistic Regression on Dataset 2."""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score

from Src.preprocessing import PreparedSplits

LOGGER = logging.getLogger(__name__)
DEFAULT_THRESHOLD = 0.5
RANDOM_STATE = 42
MAX_ITER = 2000


class BaselineError(ValueError):
    """Raised when baseline inputs do not meet the model's contract."""


@dataclass(frozen=True)
class BinaryMetrics:
    rows: int
    precision: float
    recall: float
    f1: float
    roc_auc: float
    pr_auc: float
    true_negative: int
    false_positive: int
    false_negative: int
    true_positive: int
    predicted_positive: int


@dataclass(frozen=True)
class LogisticBaselineResult:
    model: LogisticRegression
    validation: BinaryMetrics
    test: BinaryMetrics
    threshold: float
    random_state: int
    class_weight: str
    max_iter: int
    feature_names: tuple[str, ...]
    split_strategy: str
    train_rows: int


def run_logistic_regression_baseline(splits: PreparedSplits) -> LogisticBaselineResult:
    """Fit only on training data and report fixed-threshold holdout metrics.

    The default 0.5 threshold is not tuned. Validation and test are evaluated
    independently after the fit; the final test set is never used to train.
    """
    if splits.target_column != "Class":
        raise BaselineError("The first Logistic Regression baseline is scoped to Dataset 2 target 'Class'")
    _validate_splits(splits)
    model = LogisticRegression(
        class_weight="balanced", max_iter=MAX_ITER, random_state=RANDOM_STATE, solver="lbfgs"
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(splits.X_train, splits.y_train)
    convergence = [warning for warning in caught if issubclass(warning.category, ConvergenceWarning)]
    if convergence:
        raise BaselineError(f"Logistic Regression did not converge after {MAX_ITER} iterations")

    validation = _measure(model, splits.X_validation, splits.y_validation)
    test = _measure(model, splits.X_test, splits.y_test)
    LOGGER.info(
        "logistic_regression_baseline_complete",
        extra={
            "event_type": "logistic_regression_baseline_complete",
            "dataset": "Dataset 2",
            "train_rows": len(splits.X_train),
            "validation_pr_auc": validation.pr_auc,
            "test_pr_auc": test.pr_auc,
            "threshold": DEFAULT_THRESHOLD,
            "class_weight": "balanced",
        },
    )
    return LogisticBaselineResult(
        model=model, validation=validation, test=test, threshold=DEFAULT_THRESHOLD,
        random_state=RANDOM_STATE, class_weight="balanced", max_iter=MAX_ITER,
        feature_names=splits.feature_names, split_strategy=splits.split_strategy,
        train_rows=len(splits.X_train),
    )


def write_baseline_report(result: LogisticBaselineResult, path: str | Path) -> Path:
    """Write measured metrics and reproducibility details as Markdown."""
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Logistic Regression Baseline — Dataset 2", "",
        "This is the first supervised baseline for the credit-card benchmark. Dataset 1 is synthetic and is not pooled with this result.", "",
        "## Configuration", "",
        f"- Features: {len(result.feature_names)} numeric inputs from Time, Amount, and anonymized V1–V28",
        f"- Split: {result.split_strategy}",
        f"- Training rows: {result.train_rows:,}",
        f"- Class weight: `{result.class_weight}`",
        f"- Solver / max iterations / random state: `lbfgs` / {result.max_iter} / {result.random_state}",
        f"- Decision threshold: {result.threshold:.2f} (fixed default; not tuned)",
        "- Preprocessing: train-fitted median imputation and standard scaling from `Src.preprocessing`",
        "- Validation and test partitions were not used during fitting.", "",
        "## Results", "",
        "Metrics use fraud (`Class=1`) as the positive class. PR-AUC is average precision; no accuracy optimization was used.", "",
        "| Partition | Rows | Precision | Recall | F1 | ROC-AUC | PR-AUC | TN | FP | FN | TP | Predicted fraud |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        _metrics_row("Validation", result.validation),
        _metrics_row("Test", result.test), "",
        "## Limitations", "",
        "These are measured results for this dataset snapshot and chronological split. The threshold remains at 0.5; validation-based threshold selection, calibration, alternative imbalance strategies, and model comparison are separate work. V1–V28 are anonymized PCA components, not business-interpretable fields. This baseline does not establish deployment performance.", "",
    ]
    destination.write_text("\n".join(lines), encoding="utf-8")
    return destination


def _measure(model: LogisticRegression, features: pd.DataFrame, target: pd.Series) -> BinaryMetrics:
    probabilities = model.predict_proba(features)[:, 1]
    predictions = (probabilities >= DEFAULT_THRESHOLD).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        target, predictions, average="binary", pos_label=1, zero_division=0
    )
    tn, fp, fn, tp = confusion_matrix(target, predictions, labels=[0, 1]).ravel()
    return BinaryMetrics(
        rows=len(target), precision=float(precision), recall=float(recall), f1=float(f1),
        roc_auc=float(roc_auc_score(target, probabilities)),
        pr_auc=float(average_precision_score(target, probabilities)),
        true_negative=int(tn), false_positive=int(fp), false_negative=int(fn), true_positive=int(tp),
        predicted_positive=int(predictions.sum()),
    )


def _validate_splits(splits: PreparedSplits) -> None:
    partitions = (
        ("train", splits.X_train, splits.y_train),
        ("validation", splits.X_validation, splits.y_validation),
        ("test", splits.X_test, splits.y_test),
    )
    expected_features = tuple(splits.X_train.columns)
    if expected_features != splits.feature_names or not expected_features:
        raise BaselineError("Training features do not match declared preprocessed feature names")
    for name, features, target in partitions:
        if features.empty or len(features) != len(target):
            raise BaselineError(f"Dataset 2 {name} partition is empty or feature/target row counts differ")
        if tuple(features.columns) != expected_features:
            raise BaselineError(f"Dataset 2 {name} feature columns do not match training columns")
        values = features.to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise BaselineError(f"Dataset 2 {name} features contain NaN or infinite values")
        if target.isna().any() or not target.isin({0, 1}).all() or target.nunique() != 2:
            raise BaselineError(f"Dataset 2 {name} target must contain both non-null classes 0 and 1")


def _metrics_row(name: str, metrics: BinaryMetrics) -> str:
    return (
        f"| {name} | {metrics.rows:,} | {metrics.precision:.6f} | {metrics.recall:.6f} | "
        f"{metrics.f1:.6f} | {metrics.roc_auc:.6f} | {metrics.pr_auc:.6f} | "
        f"{metrics.true_negative:,} | {metrics.false_positive:,} | {metrics.false_negative:,} | "
        f"{metrics.true_positive:,} | {metrics.predicted_positive:,} |"
    )
