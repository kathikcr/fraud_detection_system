"""Third supervised benchmark: bounded, imbalance-weighted XGBoost on Dataset 2."""

from __future__ import annotations

import logging
import platform
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import xgboost
from xgboost import XGBClassifier

from Src.model_metrics import BinaryMetrics, MetricsError, measure_binary_probabilities, metrics_markdown_row, validate_prepared_splits
from Src.preprocessing import PreparedSplits

LOGGER = logging.getLogger(__name__)
DEFAULT_THRESHOLD = 0.5
RANDOM_STATE = 42
N_ESTIMATORS = 300
MAX_DEPTH = 4
LEARNING_RATE = 0.05
SUBSAMPLE = 0.8
COLSAMPLE_BYTREE = 0.8
MAX_DELTA_STEP = 1
N_JOBS = 2


class XGBoostBaselineError(ValueError):
    """Raised when the XGBoost baseline cannot run safely."""


@dataclass(frozen=True)
class XGBoostBaselineResult:
    model: XGBClassifier
    validation: BinaryMetrics
    test: BinaryMetrics
    threshold: float
    random_state: int
    scale_pos_weight: float
    n_estimators: int
    max_depth: int
    learning_rate: float
    subsample: float
    colsample_bytree: float
    max_delta_step: int
    n_jobs: int
    feature_names: tuple[str, ...]
    split_strategy: str
    train_rows: int
    xgboost_version: str
    python_version: str


def run_xgboost_baseline(splits: PreparedSplits) -> XGBoostBaselineResult:
    """Fit only on train; derive the imbalance weight from train labels only."""
    try:
        validate_prepared_splits(splits, expected_target="Class")
    except MetricsError as exc:
        raise XGBoostBaselineError(str(exc)) from exc
    counts = splits.y_train.value_counts()
    negatives = int(counts.get(0, 0))
    positives = int(counts.get(1, 0))
    if not positives or not negatives:
        raise XGBoostBaselineError("Training partition must contain both target classes")
    positive_weight = negatives / positives
    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        learning_rate=LEARNING_RATE,
        subsample=SUBSAMPLE,
        colsample_bytree=COLSAMPLE_BYTREE,
        scale_pos_weight=positive_weight,
        max_delta_step=MAX_DELTA_STEP,
        tree_method="hist",
        device="cpu",
        n_jobs=N_JOBS,
        random_state=RANDOM_STATE,
        verbosity=0,
    )
    model.fit(splits.X_train, splits.y_train)
    validation = measure_binary_probabilities(
        splits.y_validation, model.predict_proba(splits.X_validation)[:, 1], threshold=DEFAULT_THRESHOLD
    )
    test = measure_binary_probabilities(
        splits.y_test, model.predict_proba(splits.X_test)[:, 1], threshold=DEFAULT_THRESHOLD
    )
    LOGGER.info(
        "xgboost_baseline_complete",
        extra={
            "event_type": "xgboost_baseline_complete",
            "dataset": "Dataset 2",
            "train_rows": len(splits.X_train),
            "scale_pos_weight": positive_weight,
            "validation_pr_auc": validation.pr_auc,
            "test_pr_auc": test.pr_auc,
            "threshold": DEFAULT_THRESHOLD,
        },
    )
    return XGBoostBaselineResult(
        model=model, validation=validation, test=test, threshold=DEFAULT_THRESHOLD,
        random_state=RANDOM_STATE, scale_pos_weight=positive_weight,
        n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH, learning_rate=LEARNING_RATE,
        subsample=SUBSAMPLE, colsample_bytree=COLSAMPLE_BYTREE,
        max_delta_step=MAX_DELTA_STEP, n_jobs=N_JOBS, feature_names=splits.feature_names,
        split_strategy=splits.split_strategy, train_rows=len(splits.X_train),
        xgboost_version=xgboost.__version__, python_version=platform.python_version(),
    )


def write_xgboost_report(result: XGBoostBaselineResult, path: str | Path) -> Path:
    """Write measured holdout metrics and full baseline configuration."""
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# XGBoost Baseline — Dataset 2", "",
        "This is the third supervised benchmark for the credit-card dataset. Dataset 1 remains separate and is not modeled because its label is synthetic.", "",
        "## Configuration", "",
        f"- Features: {len(result.feature_names)} numeric inputs from Time, Amount, and anonymized V1–V28",
        f"- Split: {result.split_strategy}",
        f"- Training rows: {result.train_rows:,}",
        f"- XGBoost / Python versions: {result.xgboost_version} / {result.python_version}",
        f"- Trees / depth / learning rate: {result.n_estimators} / {result.max_depth} / {result.learning_rate:.2f}",
        f"- Row / column subsampling: {result.subsample:.2f} / {result.colsample_bytree:.2f}",
        f"- `scale_pos_weight`: {result.scale_pos_weight:.4f} (training negatives / training positives only)",
        f"- `max_delta_step` / random state / jobs: {result.max_delta_step} / {result.random_state} / {result.n_jobs}",
        f"- Tree method / device / metric: `hist` / CPU / `aucpr`",
        f"- Decision threshold: {result.threshold:.2f} (fixed default; not tuned)",
        "- Preprocessing: train-fitted median imputation and standard scaling from `Src.preprocessing`",
        "- Validation and test partitions were not used during fitting or class-weight calculation.", "",
        "## Results", "",
        "Metrics use fraud (`Class=1`) as the positive class. PR-AUC is average precision; no accuracy optimization was used.", "",
        "| Partition | Rows | Precision | Recall | F1 | ROC-AUC | PR-AUC | TN | FP | FN | TP | Predicted fraud |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        metrics_markdown_row("Validation", result.validation),
        metrics_markdown_row("Test", result.test), "",
        "## Limitations", "",
        "These are measured results for this dataset snapshot and chronological split. The 0.5 threshold is fixed; any threshold selection must use validation and leave test untouched. Weighting can affect probability calibration and alert volume. V1–V28 are anonymized PCA components without disclosed business meanings. No model artifact is persisted in this phase.", "",
    ]
    destination.write_text("\n".join(lines), encoding="utf-8")
    return destination
