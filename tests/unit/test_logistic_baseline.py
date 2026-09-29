from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from Src.preprocessing import PreprocessingError, prepare_dataset2
from Src.logistic_baseline import BaselineError, run_logistic_regression_baseline


def toy_dataset2(rows=400):
    values = np.arange(rows)
    frame = pd.DataFrame({"Time": values // 4, "Amount": (values % 17).astype(float)})
    for index in range(1, 29):
        frame[f"V{index}"] = ((values * index) % 23).astype(float) / (index + 1)
    frame["Class"] = (values % 4 == 0).astype(int)
    return frame


def test_baseline_fits_train_only_and_reports_fraud_metrics():
    splits = prepare_dataset2(toy_dataset2())
    result = run_logistic_regression_baseline(splits)

    assert result.train_rows == len(splits.X_train)
    assert result.class_weight == "balanced"
    assert result.threshold == 0.5
    assert result.model.n_features_in_ == 30
    for metrics, expected_rows in ((result.validation, len(splits.y_validation)), (result.test, len(splits.y_test))):
        assert metrics.rows == expected_rows
        assert 0 <= metrics.precision <= 1
        assert 0 <= metrics.recall <= 1
        assert 0 <= metrics.f1 <= 1
        assert 0 <= metrics.roc_auc <= 1
        assert 0 <= metrics.pr_auc <= 1
        assert metrics.true_negative + metrics.false_positive + metrics.false_negative + metrics.true_positive == expected_rows
        assert metrics.predicted_positive == metrics.false_positive + metrics.true_positive

    altered_holdouts = replace(
        splits,
        X_validation=splits.X_validation * 100_000,
        X_test=splits.X_test * -100_000,
    )
    altered_result = run_logistic_regression_baseline(altered_holdouts)
    np.testing.assert_allclose(result.model.coef_, altered_result.model.coef_)
    np.testing.assert_allclose(result.model.intercept_, altered_result.model.intercept_)


def test_baseline_rejects_wrong_target_misaligned_columns_and_invalid_features():
    splits = prepare_dataset2(toy_dataset2())
    with pytest.raises(BaselineError, match="scoped to Dataset 2"):
        run_logistic_regression_baseline(replace(splits, target_column="FraudIndicator"))

    with pytest.raises(BaselineError, match="feature columns"):
        run_logistic_regression_baseline(replace(splits, X_test=splits.X_test.rename(columns={"Time": "wrong"})))

    invalid = splits.X_validation.copy()
    invalid.iloc[0, 0] = np.inf
    with pytest.raises(BaselineError, match="NaN or infinite"):
        run_logistic_regression_baseline(replace(splits, X_validation=invalid))


def test_preprocessing_contract_rejects_missing_class_in_any_partition():
    frame = toy_dataset2()
    with pytest.raises(PreprocessingError, match="both target classes"):
        frame.loc[frame["Time"] >= 85, "Class"] = 0
        prepare_dataset2(frame)
