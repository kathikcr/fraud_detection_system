"""Second supervised benchmark: bounded class-weighted Random Forest on Dataset 2."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier

from Src.model_metrics import BinaryMetrics, MetricsError, measure_binary_probabilities, metrics_markdown_row, validate_prepared_splits
from Src.preprocessing import PreparedSplits

LOGGER = logging.getLogger(__name__)
DEFAULT_THRESHOLD = 0.5
RANDOM_STATE = 42
N_ESTIMATORS = 200
MAX_DEPTH = 20
MIN_SAMPLES_LEAF = 2
MAX_SAMPLES = 0.8
N_JOBS = 2


class RandomForestBaselineError(ValueError):
    """Raised when the Random Forest baseline cannot run safely."""


@dataclass(frozen=True)
class RandomForestBaselineResult:
    model: RandomForestClassifier
    validation: BinaryMetrics
    test: BinaryMetrics
    threshold: float
    random_state: int
    class_weight: str
    n_estimators: int
    max_depth: int
    min_samples_leaf: int
    max_samples: float
    n_jobs: int
    feature_names: tuple[str, ...]
    split_strategy: str
    train_rows: int


def run_random_forest_baseline(splits: PreparedSplits) -> RandomForestBaselineResult:
    """Fit one deterministic, cost-sensitive forest on Dataset 2 training rows only."""
    try:
        validate_prepared_splits(splits, expected_target="Class")
    except MetricsError as exc:
        raise RandomForestBaselineError(str(exc)) from exc
    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        max_samples=MAX_SAMPLES,
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
    )
    model.fit(splits.X_train, splits.y_train)
    validation = measure_binary_probabilities(
        splits.y_validation, model.predict_proba(splits.X_validation)[:, 1], threshold=DEFAULT_THRESHOLD
    )
    test = measure_binary_probabilities(
        splits.y_test, model.predict_proba(splits.X_test)[:, 1], threshold=DEFAULT_THRESHOLD
    )
    LOGGER.info(
        "random_forest_baseline_complete",
        extra={
            "event_type": "random_forest_baseline_complete",
            "dataset": "Dataset 2",
            "train_rows": len(splits.X_train),
            "validation_pr_auc": validation.pr_auc,
            "test_pr_auc": test.pr_auc,
            "threshold": DEFAULT_THRESHOLD,
            "class_weight": "balanced_subsample",
        },
    )
    return RandomForestBaselineResult(
        model=model, validation=validation, test=test, threshold=DEFAULT_THRESHOLD,
        random_state=RANDOM_STATE, class_weight="balanced_subsample",
        n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH, min_samples_leaf=MIN_SAMPLES_LEAF,
        max_samples=MAX_SAMPLES, n_jobs=N_JOBS, feature_names=splits.feature_names,
        split_strategy=splits.split_strategy, train_rows=len(splits.X_train),
    )


def write_random_forest_report(result: RandomForestBaselineResult, path: str | Path) -> Path:
    """Write measured holdout metrics and exact forest configuration."""
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Random Forest Baseline — Dataset 2", "",
        "This is the second supervised benchmark for the credit-card dataset. Dataset 1 remains separate and is not modeled because its label is synthetic.", "",
        "## Configuration", "",
        f"- Features: {len(result.feature_names)} numeric inputs from Time, Amount, and anonymized V1–V28",
        f"- Split: {result.split_strategy}",
        f"- Training rows: {result.train_rows:,}",
        f"- Trees / maximum depth / minimum leaf rows: {result.n_estimators} / {result.max_depth} / {result.min_samples_leaf}",
        f"- Bootstrap sample fraction per tree: {result.max_samples:.2f}",
        f"- Class weight / random state / jobs: `{result.class_weight}` / {result.random_state} / {result.n_jobs}",
        f"- Decision threshold: {result.threshold:.2f} (fixed default; not tuned)",
        "- Preprocessing: train-fitted median imputation and standard scaling from `Src.preprocessing`",
        "- Validation and test partitions were not used during fitting.", "",
        "## Results", "",
        "Metrics use fraud (`Class=1`) as the positive class. PR-AUC is average precision; no accuracy optimization was used.", "",
        "| Partition | Rows | Precision | Recall | F1 | ROC-AUC | PR-AUC | TN | FP | FN | TP | Predicted fraud |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        metrics_markdown_row("Validation", result.validation),
        metrics_markdown_row("Test", result.test), "",
        "## Limitations", "",
        "These are measured results for this dataset snapshot and chronological split. The 0.5 threshold is fixed and not operationally optimized. Compare the false-positive burden with recall before interpreting the result; threshold selection is a later phase and must use validation, not test. V1–V28 are anonymized PCA components without disclosed business meanings. No model artifact is persisted in this phase.", "",
    ]
    destination.write_text("\n".join(lines), encoding="utf-8")
    return destination
