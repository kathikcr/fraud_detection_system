import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression

from Src.evaluation import EvaluationSpec, evaluate_models, write_evaluation_artifacts
from Src.preprocessing import prepare_dataset2


def test_writes_comparison_report_roc_pr_and_confusion_artifacts(tmp_path):
    row = np.arange(400)
    frame = pd.DataFrame({"Time": row // 4, "Amount": row % 31})
    for index in range(1, 29):
        frame[f"V{index}"] = np.cos(row / (index + 2))
    frame["Class"] = (row % 20 == 0).astype(int)
    splits = prepare_dataset2(frame)
    logistic = LogisticRegression(class_weight="balanced", random_state=42).fit(splits.X_train, splits.y_train)
    isolation = IsolationForest(n_estimators=30, max_samples=128, contamination="auto", random_state=42).fit(splits.X_train)
    report = evaluate_models(splits, [
        EvaluationSpec("Logistic Regression", logistic, "probability", 0.5),
        EvaluationSpec("Isolation Forest", isolation, "anomaly"),
    ])

    artifacts = write_evaluation_artifacts(report, tmp_path / "evaluation")
    text = artifacts.report.read_text(encoding="utf-8")

    assert "did not refit models" in text
    assert "not fraud probabilities" in text
    assert "## Validation results" in text and "## Test results" in text
    for path in (artifacts.roc_chart, artifacts.precision_recall_chart, artifacts.confusion_matrix_chart):
        assert path.is_file() and path.stat().st_size > 1000
