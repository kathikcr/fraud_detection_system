"""First supervised benchmark: class-weighted Logistic Regression on Dataset 2."""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

from Src.model_metrics import BinaryMetrics, MetricsError, measure_binary_probabilities, metrics_markdown_row, validate_prepared_splits
from Src.preprocessing import PreparedSplits

LOGGER = logging.getLogger(__name__)
DEFAULT_THRESHOLD = 0.5
RANDOM_STATE = 42
MAX_ITER = 2000


class BaselineError(ValueError):
    """Raised when baseline inputs do not meet the model's contract."""


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
    try:
        validate_prepared_splits(splits, expected_target="Class")
    except MetricsError as exc:
        raise BaselineError(str(exc)) from exc
    model = LogisticRegression(
        class_weight="balanced", max_iter=MAX_ITER, random_state=RANDOM_STATE, solver="lbfgs"
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(splits.X_train, splits.y_train)
    convergence = [warning for warning in caught if issubclass(warning.category, ConvergenceWarning)]
    if convergence:
        raise BaselineError(f"Logistic Regression did not converge after {MAX_ITER} iterations")

    validation = measure_binary_probabilities(
        splits.y_validation, model.predict_proba(splits.X_validation)[:, 1], threshold=DEFAULT_THRESHOLD
    )
    test = measure_binary_probabilities(
        splits.y_test, model.predict_proba(splits.X_test)[:, 1], threshold=DEFAULT_THRESHOLD
    )
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


_metrics_row = metrics_markdown_row
