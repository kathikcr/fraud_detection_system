import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression

from Src.evaluation import EvaluationError, EvaluationSpec, evaluate_models
from Src.preprocessing import prepare_dataset2


def toy_dataset2(rows=320):
    row = np.arange(rows)
    frame = pd.DataFrame({"Time": row // 4, "Amount": (row % 17).astype(float)})
    for index in range(1, 29):
        frame[f"V{index}"] = ((row * index) % 23).astype(float) / (index + 1)
    frame["Class"] = (row % 8 == 0).astype(int)
    return frame


def fitted_estimators(splits):
    logistic = LogisticRegression(random_state=42).fit(splits.X_train, splits.y_train)
    isolation = IsolationForest(n_estimators=20, max_samples=128, contamination="auto", random_state=42).fit(splits.X_train)
    return logistic, isolation


def test_evaluates_probability_and_anomaly_models_with_consistent_holdout_metrics():
    splits = prepare_dataset2(toy_dataset2())
    logistic, isolation = fitted_estimators(splits)
    coef = logistic.coef_.copy()
    offset = isolation.offset_

    results = evaluate_models(splits, [
        EvaluationSpec("Logistic", logistic, "probability", 0.4),
        EvaluationSpec("Isolation Forest", isolation, "anomaly"),
    ])

    assert results.target_column == "Class"
    assert len(results.models) == 2
    assert results.models[0].decision_rule == "probability >= 0.4 (fixed)"
    assert "native predict cutoff" in results.models[1].decision_rule
    for model in results.models:
        for partition_name in ("validation", "test"):
            partition = getattr(model, partition_name)
            assert partition.metrics.rows == len(getattr(splits, f"y_{partition_name}"))
            assert partition.prevalence == pytest.approx(getattr(splits, f"y_{partition_name}").mean())
            assert 0 <= partition.alert_rate <= 1
            assert len(partition.roc_fpr) == len(partition.roc_tpr)
            assert len(partition.pr_precision) == len(partition.pr_recall)
            assert partition.roc_fpr[0] == pytest.approx(0)
            assert partition.roc_tpr[-1] == pytest.approx(1)
    np.testing.assert_array_equal(logistic.coef_, coef)
    assert isolation.offset_ == offset


def test_evaluator_rejects_empty_duplicate_unfitted_and_bad_threshold_specs():
    splits = prepare_dataset2(toy_dataset2())
    logistic, _ = fitted_estimators(splits)
    with pytest.raises(EvaluationError, match="At least one"):
        evaluate_models(splits, [])
    with pytest.raises(EvaluationError, match="unique"):
        evaluate_models(splits, [EvaluationSpec("same", logistic, "probability"), EvaluationSpec("same", logistic, "probability")])
    with pytest.raises(EvaluationError, match="not fitted"):
        evaluate_models(splits, [EvaluationSpec("new", LogisticRegression(), "probability")])
    with pytest.raises(EvaluationError, match="threshold"):
        evaluate_models(splits, [EvaluationSpec("bad", logistic, "probability", 1.5)])


def test_evaluator_rejects_score_kind_and_feature_contract_errors():
    splits = prepare_dataset2(toy_dataset2())
    logistic, isolation = fitted_estimators(splits)
    with pytest.raises(EvaluationError, match="Unsupported score kind"):
        evaluate_models(splits, [EvaluationSpec("bad-kind", logistic, "other")])
    with pytest.raises(EvaluationError, match="must implement"):
        evaluate_models(splits, [EvaluationSpec("bad-if", logistic, "anomaly")])
    isolation.feature_names_in_ = np.array(["bad"] * len(splits.feature_names))
    with pytest.raises(EvaluationError, match="feature names"):
        evaluate_models(splits, [EvaluationSpec("bad-features", isolation, "anomaly")])
