from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from Src.preprocessing import prepare_dataset2
from Src.random_forest_baseline import RandomForestBaselineError, run_random_forest_baseline


def toy_dataset2(rows=240):
    row = np.arange(rows)
    frame = pd.DataFrame({"Time": row // 4, "Amount": (row % 17).astype(float)})
    for index in range(1, 29):
        frame[f"V{index}"] = ((row * index) % 23).astype(float) / (index + 1)
    frame["Class"] = (row % 4 == 0).astype(int)
    return frame


def test_random_forest_fits_training_partition_and_reports_metrics_deterministically():
    splits = prepare_dataset2(toy_dataset2())
    result = run_random_forest_baseline(splits)
    repeated = run_random_forest_baseline(splits)

    assert result.train_rows == len(splits.X_train)
    assert result.class_weight == "balanced_subsample"
    assert result.threshold == 0.5
    assert result.n_estimators == len(result.model.estimators_) == 200
    assert result.model.n_features_in_ == 30
    assert result.test == repeated.test
    for metrics, expected_rows in ((result.validation, len(splits.y_validation)), (result.test, len(splits.y_test))):
        assert metrics.rows == expected_rows
        assert 0 <= metrics.precision <= 1
        assert 0 <= metrics.recall <= 1
        assert 0 <= metrics.f1 <= 1
        assert 0 <= metrics.roc_auc <= 1
        assert 0 <= metrics.pr_auc <= 1
        assert metrics.true_negative + metrics.false_positive + metrics.false_negative + metrics.true_positive == expected_rows

    altered = replace(splits, X_validation=splits.X_validation * 10_000, X_test=splits.X_test * -10_000)
    altered_result = run_random_forest_baseline(altered)
    for original_tree, altered_tree in zip(result.model.estimators_, altered_result.model.estimators_):
        np.testing.assert_array_equal(original_tree.tree_.threshold, altered_tree.tree_.threshold)


def test_random_forest_rejects_wrong_target_column_order_and_infinite_values():
    splits = prepare_dataset2(toy_dataset2())
    with pytest.raises(RandomForestBaselineError, match="expects target"):
        run_random_forest_baseline(replace(splits, target_column="FraudIndicator"))

    with pytest.raises(RandomForestBaselineError, match="feature columns"):
        run_random_forest_baseline(replace(splits, X_test=splits.X_test.rename(columns={"Time": "bad"})))

    features = splits.X_test.copy()
    features.iloc[0, 0] = np.inf
    with pytest.raises(RandomForestBaselineError, match="NaN or infinite"):
        run_random_forest_baseline(replace(splits, X_test=features))
