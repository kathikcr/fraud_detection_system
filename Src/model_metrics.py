"""Shared binary fraud metrics used by one-model-at-a-time benchmarks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score

from Src.preprocessing import PreparedSplits


class MetricsError(ValueError):
    """Raised when binary metric inputs are invalid."""


def validate_prepared_splits(
    splits: PreparedSplits,
    *,
    expected_target: str = "Class",
    validate_training_target: bool = True,
) -> None:
    """Validate model-ready train/validation/test arrays before fitting or scoring."""
    if splits.target_column != expected_target:
        raise MetricsError(f"Model expects target {expected_target!r}, received {splits.target_column!r}")
    partitions = (
        ("train", splits.X_train, splits.y_train),
        ("validation", splits.X_validation, splits.y_validation),
        ("test", splits.X_test, splits.y_test),
    )
    expected_features = tuple(splits.X_train.columns)
    if expected_features != splits.feature_names or not expected_features:
        raise MetricsError("Training features do not match declared preprocessed feature names")
    for name, features, target in partitions:
        if features.empty or len(features) != len(target):
            raise MetricsError(f"Dataset 2 {name} partition is empty or feature/target row counts differ")
        if tuple(features.columns) != expected_features:
            raise MetricsError(f"Dataset 2 {name} feature columns do not match training columns")
        if not np.isfinite(features.to_numpy(dtype=float)).all():
            raise MetricsError(f"Dataset 2 {name} features contain NaN or infinite values")
        if name == "train" and not validate_training_target:
            continue
        if target.isna().any() or not target.isin({0, 1}).all() or target.nunique() != 2:
            raise MetricsError(f"Dataset 2 {name} target must contain both non-null classes 0 and 1")


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


def measure_binary_probabilities(target: pd.Series, probabilities, *, threshold: float = 0.5) -> BinaryMetrics:
    """Compute fraud-positive metrics from probabilities at a fixed threshold."""
    probabilities = np.asarray(probabilities, dtype=float)
    if target.isna().any() or not target.isin({0, 1}).all() or target.nunique() != 2:
        raise MetricsError("Metrics target must contain both non-null classes 0 and 1")
    if probabilities.ndim != 1 or len(probabilities) != len(target):
        raise MetricsError("Probability vector must be one-dimensional and align with target rows")
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise MetricsError("Probabilities must be finite values between 0 and 1")
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise MetricsError("Decision threshold must be a finite value between 0 and 1")

    predictions = (probabilities >= threshold).astype(int)
    return measure_binary_scores(target, probabilities, predictions)


def measure_binary_scores(target: pd.Series, scores, predictions) -> BinaryMetrics:
    """Compute ranking metrics from any finite score plus fixed binary decisions."""
    scores = np.asarray(scores, dtype=float)
    predictions = np.asarray(predictions)
    if target.isna().any() or not target.isin({0, 1}).all() or target.nunique() != 2:
        raise MetricsError("Metrics target must contain both non-null classes 0 and 1")
    if scores.ndim != 1 or len(scores) != len(target) or not np.isfinite(scores).all():
        raise MetricsError("Score vector must be finite, one-dimensional, and aligned with target rows")
    if predictions.ndim != 1 or len(predictions) != len(target) or not np.isin(predictions, [0, 1]).all():
        raise MetricsError("Predictions must be aligned binary values containing only 0/1")
    predictions = predictions.astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        target, predictions, average="binary", pos_label=1, zero_division=0
    )
    tn, fp, fn, tp = confusion_matrix(target, predictions, labels=[0, 1]).ravel()
    return BinaryMetrics(
        rows=len(target), precision=float(precision), recall=float(recall), f1=float(f1),
        roc_auc=float(roc_auc_score(target, scores)),
        pr_auc=float(average_precision_score(target, scores)),
        true_negative=int(tn), false_positive=int(fp), false_negative=int(fn), true_positive=int(tp),
        predicted_positive=int(predictions.sum()),
    )


def metrics_markdown_row(name: str, metrics: BinaryMetrics) -> str:
    return (
        f"| {name} | {metrics.rows:,} | {metrics.precision:.6f} | {metrics.recall:.6f} | "
        f"{metrics.f1:.6f} | {metrics.roc_auc:.6f} | {metrics.pr_auc:.6f} | "
        f"{metrics.true_negative:,} | {metrics.false_positive:,} | {metrics.false_negative:,} | "
        f"{metrics.true_positive:,} | {metrics.predicted_positive:,} |"
    )
