"""Fourth required model: unsupervised Isolation Forest for Dataset 2."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from sklearn.ensemble import IsolationForest

from Src.model_metrics import BinaryMetrics, MetricsError, measure_binary_scores, metrics_markdown_row, validate_prepared_splits
from Src.preprocessing import PreparedSplits

LOGGER = logging.getLogger(__name__)
RANDOM_STATE = 42
N_ESTIMATORS = 300
MAX_SAMPLES = 256
CONTAMINATION = "auto"
N_JOBS = 2


class IsolationForestBaselineError(ValueError):
    """Raised when the Isolation Forest baseline cannot run safely."""


@dataclass(frozen=True)
class IsolationForestBaselineResult:
    model: IsolationForest
    validation: BinaryMetrics
    test: BinaryMetrics
    random_state: int
    n_estimators: int
    max_samples: int
    contamination: str
    offset: float
    n_jobs: int
    feature_names: tuple[str, ...]
    split_strategy: str
    train_rows: int
    validation_anomaly_rate: float
    test_anomaly_rate: float


def run_isolation_forest_baseline(splits: PreparedSplits) -> IsolationForestBaselineResult:
    """Fit on training features only; fraud labels are used for holdout metrics only.

    `contamination="auto"` provides the estimator's fixed native outlier cutoff.
    Ranking metrics use negated `score_samples` (higher means more anomalous);
    threshold metrics use `predict`'s native outlier decision. Scores are not
    probabilities and are not calibrated as fraud risk.
    """
    try:
        validate_prepared_splits(
            splits, expected_target="Class", validate_training_target=False
        )
    except MetricsError as exc:
        raise IsolationForestBaselineError(str(exc)) from exc
    model = IsolationForest(
        n_estimators=N_ESTIMATORS,
        max_samples=min(MAX_SAMPLES, len(splits.X_train)),
        contamination=CONTAMINATION,
        n_jobs=N_JOBS,
        random_state=RANDOM_STATE,
    )
    # Deliberately pass only X_train: no target labels, target-derived weights,
    # validation rows, or test rows are used to fit this unsupervised estimator.
    model.fit(splits.X_train)
    validation_scores, validation_predictions = _score_and_predict(model, splits.X_validation)
    test_scores, test_predictions = _score_and_predict(model, splits.X_test)
    validation = measure_binary_scores(splits.y_validation, validation_scores, validation_predictions)
    test = measure_binary_scores(splits.y_test, test_scores, test_predictions)
    LOGGER.info(
        "isolation_forest_baseline_complete",
        extra={
            "event_type": "isolation_forest_baseline_complete",
            "dataset": "Dataset 2",
            "train_rows": len(splits.X_train),
            "validation_anomaly_rate": validation.predicted_positive / validation.rows,
            "test_anomaly_rate": test.predicted_positive / test.rows,
            "validation_pr_auc": validation.pr_auc,
            "test_pr_auc": test.pr_auc,
            "contamination": CONTAMINATION,
        },
    )
    return IsolationForestBaselineResult(
        model=model, validation=validation, test=test, random_state=RANDOM_STATE,
        n_estimators=N_ESTIMATORS, max_samples=min(MAX_SAMPLES, len(splits.X_train)), contamination=CONTAMINATION,
        offset=float(model.offset_), n_jobs=N_JOBS, feature_names=splits.feature_names,
        split_strategy=splits.split_strategy, train_rows=len(splits.X_train),
        validation_anomaly_rate=validation.predicted_positive / validation.rows,
        test_anomaly_rate=test.predicted_positive / test.rows,
    )


def write_isolation_forest_report(result: IsolationForestBaselineResult, path: str | Path) -> Path:
    """Write measured holdout results, clearly distinguishing anomaly scores."""
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Isolation Forest Baseline — Dataset 2", "",
        "Isolation Forest is fitted as an unsupervised anomaly detector on Dataset 2 training features only. Dataset 1 is synthetic and is not pooled into these results.", "",
        "## Configuration", "",
        f"- Features: {len(result.feature_names)} numeric inputs from Time, Amount, and anonymized V1–V28",
        f"- Split: {result.split_strategy}",
        f"- Training rows: {result.train_rows:,}",
        f"- Trees / max samples per tree: {result.n_estimators} / {result.max_samples}",
        f"- Contamination / native score offset / random state / jobs: `{result.contamination}` / {result.offset:.6f} / {result.random_state} / {result.n_jobs}",
        "- Fit inputs: training feature matrix only; class labels are not passed to `fit`.",
        "- `-score_samples` ranks observations (higher means more anomalous); `predict` supplies binary outlier decisions using the estimator's fixed native cutoff.",
        "- Scores are not calibrated fraud probabilities. No validation threshold tuning or resampling was used.", "",
        "## Results", "",
        "Fraud (`Class=1`) is the positive class for evaluation only. PR-AUC is average precision computed from anomaly-score ranking.", "",
        "| Partition | Rows | Anomaly rate | Precision | Recall | F1 | ROC-AUC | PR-AUC | TN | FP | FN | TP |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        _metrics_row("Validation", result.validation, result.validation_anomaly_rate),
        _metrics_row("Test", result.test, result.test_anomaly_rate), "",
        "## Limitations", "",
        "The outlier cutoff is the Isolation Forest `contamination=auto` default, not the known fraud prevalence. Low anomaly recall or precision reflects this unsupervised cutoff and features, not a probability threshold. Results describe one chronological dataset snapshot. `V1`–`V28` remain anonymized PCA components, not interpretable business fields. No model artifact is persisted in this phase.", "",
    ]
    destination.write_text("\n".join(lines), encoding="utf-8")
    return destination


def _score_and_predict(model: IsolationForest, features) -> tuple:
    scores = -model.score_samples(features)
    predictions = (model.predict(features) == -1).astype(int)
    return scores, predictions


def _metrics_row(name: str, metrics: BinaryMetrics, anomaly_rate: float) -> str:
    return (
        f"| {name} | {metrics.rows:,} | {anomaly_rate:.4%} | {metrics.precision:.6f} | "
        f"{metrics.recall:.6f} | {metrics.f1:.6f} | {metrics.roc_auc:.6f} | {metrics.pr_auc:.6f} | "
        f"{metrics.true_negative:,} | {metrics.false_positive:,} | {metrics.false_negative:,} | {metrics.true_positive:,} |"
    )
