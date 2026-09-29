import numpy as np
import pandas as pd
import pytest

from Src.ingestion import DATASET2_SCHEMA
from Src.preprocessing import PreprocessingError, prepare_dataset1, prepare_dataset2


def synthetic_dataset1(rows=120):
    labels = np.tile([0, 1], rows // 2)
    return pd.DataFrame({
        "TransactionID": np.arange(rows),
        "CustomerID": np.arange(rows) % 13,
        "MerchantID": np.arange(rows) % 7,
        "Amount": np.arange(rows, dtype=float),
        "Timestamp": pd.date_range("2025-01-01", periods=rows, freq="h").astype(str),
        "FraudIndicator": labels,
        "Category": np.where(np.arange(rows) % 4 == 0, "Online", "Retail"),
        "TransactionAmount": np.arange(rows, dtype=float) * 2,
        "AnomalyScore": np.arange(rows, dtype=float),
        "SuspiciousFlag": labels,
        "MerchantName": "merchant",
        "Location": "location",
    })


def synthetic_dataset2(rows=200):
    frame = pd.DataFrame({column: np.zeros(rows, dtype=float) for column in DATASET2_SCHEMA.columns})
    frame["Time"] = np.repeat(np.arange(rows // 2, dtype=float), 2)
    frame["Amount"] = np.arange(rows, dtype=float)
    frame["Class"] = np.tile([0, 1], rows // 2)
    return frame


def test_dataset1_uses_seeded_stratified_splits_and_excludes_leakage_fields():
    source = synthetic_dataset1()
    prepared = prepare_dataset1(source)
    repeated = prepare_dataset1(source)

    assert len(prepared.X_train) == 84
    assert len(prepared.X_validation) == 18
    assert len(prepared.X_test) == 18
    assert prepared.class_counts == {"train": {0: 42, 1: 42}, "validation": {0: 9, 1: 9}, "test": {0: 9, 1: 9}}
    assert not set(prepared.feature_names) & {"FraudIndicator", "TransactionID", "SuspiciousFlag", "AnomalyScore", "MerchantID", "CustomerID"}
    assert prepared.X_train.equals(repeated.X_train)
    assert prepared.split_strategy.startswith("stratified random 70/15/15")
    assert prepared.excluded_columns == ("TransactionID", "CustomerID", "MerchantID", "MerchantName", "Location", "FraudIndicator", "SuspiciousFlag", "AnomalyScore")


def test_dataset1_fits_scaler_only_on_training_rows():
    source = synthetic_dataset1()
    prepared = prepare_dataset1(source)
    scaler = prepared.preprocessor.named_transformers_["numeric"].named_steps["scaler"]
    train_mean = source.loc[prepared.X_train.index, "Amount"].mean()
    assert scaler.mean_[0] == pytest.approx(train_mean)


def test_dataset1_imputes_numeric_and_missing_category_values():
    source = synthetic_dataset1()
    source.loc[0, "Amount"] = np.nan
    source.loc[1, "Category"] = None
    prepared = prepare_dataset1(source)

    assert np.isfinite(prepared.X_train.to_numpy()).all()
    assert np.isfinite(prepared.X_validation.to_numpy()).all()
    assert np.isfinite(prepared.X_test.to_numpy()).all()
    assert prepared.X_train.shape[1] == len(prepared.feature_names)


def test_dataset1_rejects_invalid_target_timestamp_and_non_numeric():
    source = synthetic_dataset1()
    source.loc[0, "FraudIndicator"] = 2
    with pytest.raises(PreprocessingError, match="0/1"):
        prepare_dataset1(source)

    source = synthetic_dataset1()
    source.loc[0, "Timestamp"] = "not-a-date"
    with pytest.raises(PreprocessingError, match="Timestamp"):
        prepare_dataset1(source)

    source = synthetic_dataset1()
    source["Amount"] = source["Amount"].astype(object)
    source.loc[0, "Amount"] = "bad"
    with pytest.raises(PreprocessingError, match="non-numeric"):
        prepare_dataset1(source)


def test_dataset2_uses_forward_chronological_splits_and_keeps_tied_times_together():
    source = synthetic_dataset2()
    prepared = prepare_dataset2(source)

    train_max = source.loc[prepared.X_train.index, "Time"].max()
    validation_times = source.loc[prepared.X_validation.index, "Time"]
    test_min = source.loc[prepared.X_test.index, "Time"].min()
    assert train_max < validation_times.min() <= validation_times.max() < test_min
    assert len(prepared.X_train) + len(prepared.X_validation) + len(prepared.X_test) == len(source)
    for partition in (prepared.X_train, prepared.X_validation, prepared.X_test):
        assert partition.index.to_series().map(source["Time"]).duplicated(keep=False).sum() % 2 == 0
    assert prepared.split_strategy.startswith("chronological 70/15/15")
    assert "Class" not in prepared.feature_names
    assert len(prepared.X_train) > len(prepared.X_validation) > 0


def test_dataset2_fits_scaler_on_train_and_rejects_unsorted_or_single_class_windows():
    source = synthetic_dataset2()
    prepared = prepare_dataset2(source)
    scaler = prepared.preprocessor.named_transformers_["numeric"].named_steps["scaler"]
    train_amount_mean = source.loc[prepared.X_train.index, "Amount"].mean()
    amount_position = list(prepared.feature_names).index("Amount")
    assert scaler.mean_[amount_position] == pytest.approx(train_amount_mean)

    source.loc[[0, 1], "Time"] = [2.0, 0.0]
    with pytest.raises(PreprocessingError, match="non-decreasing"):
        prepare_dataset2(source)


def test_dataset2_rejects_missing_columns_invalid_targets_and_infinite_values():
    source = synthetic_dataset2()
    with pytest.raises(PreprocessingError, match="missing required"):
        prepare_dataset2(source.drop(columns="V28"))

    source = synthetic_dataset2()
    source.loc[0, "Class"] = np.nan
    with pytest.raises(PreprocessingError, match="0/1"):
        prepare_dataset2(source)

    source = synthetic_dataset2()
    source.loc[0, "Amount"] = np.inf
    with pytest.raises(PreprocessingError, match="infinite"):
        prepare_dataset2(source)


def test_dataset2_imputes_missing_numeric_values_without_dropping_rows():
    source = synthetic_dataset2()
    source.loc[0, "Amount"] = np.nan
    prepared = prepare_dataset2(source)
    assert len(prepared.X_train) + len(prepared.X_validation) + len(prepared.X_test) == len(source)
    assert np.isfinite(prepared.X_train.to_numpy()).all()
