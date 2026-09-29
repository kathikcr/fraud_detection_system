from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from Src.isolation_forest_baseline import IsolationForestBaselineError, run_isolation_forest_baseline
from Src.preprocessing import prepare_dataset2


def toy_dataset2(rows=240):
    row = np.arange(rows)
    frame = pd.DataFrame({"Time": row // 4, "Amount": (row % 17).astype(float)})
    for index in range(1, 29):
        frame[f"V{index}"] = ((row * index) % 23).astype(float) / (index + 1)
    frame["Class"] = (row % 4 == 0).astype(int)
    return frame


def test_isolation_forest_fits_features_only_and_scores_fraud_anomaly_ranking():
    splits = prepare_dataset2(toy_dataset2())
    result = run_isolation_forest_baseline(splits)

    assert result.train_rows == len(splits.X_train)
    assert result.contamination == "auto"
    assert result.offset == pytest.approx(result.model.offset_)
    assert result.model.n_features_in_ == 30
    assert result.validation.predicted_positive == result.validation.true_positive + result.validation.false_positive
    assert result.test.predicted_positive == result.test.true_positive + result.test.false_positive
    for metrics, expected_rows in ((result.validation, len(splits.y_validation)), (result.test, len(splits.y_test))):
        assert metrics.rows == expected_rows
        assert 0 <= metrics.precision <= 1
        assert 0 <= metrics.recall <= 1
        assert 0 <= metrics.f1 <= 1
        assert 0 <= metrics.roc_auc <= 1
        assert 0 <= metrics.pr_auc <= 1
        assert metrics.true_negative + metrics.false_positive + metrics.false_negative + metrics.true_positive == expected_rows

    arbitrary_labels = replace(splits, y_train=pd.Series(0, index=splits.y_train.index, dtype="int8"))
    changed_result = run_isolation_forest_baseline(arbitrary_labels)
    np.testing.assert_allclose(
        result.model.score_samples(splits.X_train),
        changed_result.model.score_samples(splits.X_train),
    )


def test_isolation_forest_rejects_wrong_target_bad_columns_and_non_finite_values():
    splits = prepare_dataset2(toy_dataset2())
    with pytest.raises(IsolationForestBaselineError, match="expects target"):
        run_isolation_forest_baseline(replace(splits, target_column="FraudIndicator"))
    with pytest.raises(IsolationForestBaselineError, match="feature columns"):
        run_isolation_forest_baseline(replace(splits, X_test=splits.X_test.rename(columns={"Time": "bad"})))

    features = splits.X_test.copy()
    features.iloc[0, 0] = np.inf
    with pytest.raises(IsolationForestBaselineError, match="NaN or infinite"):
        run_isolation_forest_baseline(replace(splits, X_test=features))
