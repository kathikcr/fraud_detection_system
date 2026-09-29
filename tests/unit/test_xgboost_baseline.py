from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from Src.preprocessing import prepare_dataset2
from Src.xgboost_baseline import XGBoostBaselineError, run_xgboost_baseline


def toy_dataset2(rows=240):
    row = np.arange(rows)
    frame = pd.DataFrame({"Time": row // 4, "Amount": (row % 17).astype(float)})
    for index in range(1, 29):
        frame[f"V{index}"] = ((row * index) % 23).astype(float) / (index + 1)
    frame["Class"] = (row % 4 == 0).astype(int)
    return frame


def test_xgboost_uses_training_only_class_ratio_and_reports_fraud_metrics():
    splits = prepare_dataset2(toy_dataset2())
    result = run_xgboost_baseline(splits)
    expected_weight = (splits.y_train == 0).sum() / (splits.y_train == 1).sum()

    assert result.scale_pos_weight == pytest.approx(expected_weight)
    assert result.threshold == 0.5
    assert result.train_rows == len(splits.y_train)
    assert result.model.get_params()["n_estimators"] == result.n_estimators == 300
    assert result.model.get_params()["tree_method"] == "hist"
    for metrics, expected_rows in ((result.validation, len(splits.y_validation)), (result.test, len(splits.y_test))):
        assert metrics.rows == expected_rows
        assert 0 <= metrics.precision <= 1
        assert 0 <= metrics.recall <= 1
        assert 0 <= metrics.f1 <= 1
        assert 0 <= metrics.roc_auc <= 1
        assert 0 <= metrics.pr_auc <= 1
        assert metrics.true_negative + metrics.false_positive + metrics.false_negative + metrics.true_positive == expected_rows

    changed_holdouts = replace(
        splits,
        X_validation=splits.X_validation * -50_000,
        X_test=splits.X_test * 50_000,
        y_validation=1 - splits.y_validation,
        y_test=1 - splits.y_test,
    )
    changed_result = run_xgboost_baseline(changed_holdouts)
    assert changed_result.scale_pos_weight == pytest.approx(result.scale_pos_weight)
    np.testing.assert_allclose(
        result.model.predict_proba(splits.X_train),
        changed_result.model.predict_proba(splits.X_train),
    )


def test_xgboost_rejects_wrong_target_feature_order_and_non_finite_input():
    splits = prepare_dataset2(toy_dataset2())
    with pytest.raises(XGBoostBaselineError, match="expects target"):
        run_xgboost_baseline(replace(splits, target_column="FraudIndicator"))
    with pytest.raises(XGBoostBaselineError, match="feature columns"):
        run_xgboost_baseline(replace(splits, X_test=splits.X_test.rename(columns={"Time": "bad"})))

    features = splits.X_validation.copy()
    features.iloc[0, 0] = np.inf
    with pytest.raises(XGBoostBaselineError, match="NaN or infinite"):
        run_xgboost_baseline(replace(splits, X_validation=features))


def test_xgboost_rejects_missing_training_class():
    splits = prepare_dataset2(toy_dataset2())
    invalid = replace(splits, y_train=pd.Series(np.zeros(len(splits.y_train), dtype=int), index=splits.y_train.index))
    with pytest.raises(XGBoostBaselineError, match="both non-null classes"):
        run_xgboost_baseline(invalid)
