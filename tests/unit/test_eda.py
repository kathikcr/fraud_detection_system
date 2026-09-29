import pandas as pd
import pytest

from Src.dataset1_integration import build_dataset1_transaction_table
from Src.eda import EDAError, PCA_COLUMNS, profile_dataset1, profile_dataset2
from Src.ingestion import DATASET2_SCHEMA


def small_dataset2():
    rows = []
    for label, amount, time_value, offset in ((0, 12.0, 0.0, 0.0), (0, 30.0, 60.0, 0.2), (1, 120.0, 120.0, 0.4)):
        row = {column: 0.0 for column in DATASET2_SCHEMA.columns}
        row.update({"Time": time_value, "Amount": amount, "Class": label})
        for index, component in enumerate(PCA_COLUMNS):
            row[component] = offset + index / 100.0
        rows.append(row)
    return pd.DataFrame(rows, columns=DATASET2_SCHEMA.columns)


def test_profiles_dataset1_as_synthetic_with_separate_amount_fields(dataset1_csv_root):
    from Src.ingestion import load_dataset1

    raw_tables = load_dataset1(dataset1_csv_root)
    table = build_dataset1_transaction_table(raw_tables).frame
    summary = profile_dataset1(table)

    assert summary["synthetic"] is True
    assert summary["fraud_counts"] == {0: 2, 1: 1}
    assert summary["columns"] == 12
    assert set(summary["amount_by_class"]) == {"Amount", "TransactionAmount"}
    assert summary["unparsed_timestamps"] == 0
    assert summary["duplicate_rows"] == 0


def test_profiles_dataset2_imbalance_and_anonymized_components():
    summary = profile_dataset2(small_dataset2())

    assert summary["fraud_counts"] == {0: 2, 1: 1}
    assert summary["fraud_rate"] == pytest.approx(1 / 3)
    assert set(summary["pca_summary"]) == set(PCA_COLUMNS)
    assert summary["pca_summary"]["V1"][1]["mean"] == pytest.approx(0.4)
    assert summary["amount_by_class"][1]["median"] == 120.0


def test_reports_duplicate_rows_and_does_not_change_input():
    frame = small_dataset2()
    frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    before = frame.copy(deep=True)

    summary = profile_dataset2(frame)

    assert summary["duplicate_rows"] == 1
    pd.testing.assert_frame_equal(frame, before)


def test_profiles_reject_invalid_target_and_missing_eda_columns(dataset1_csv_root):
    from Src.ingestion import load_dataset1

    table = build_dataset1_transaction_table(load_dataset1(dataset1_csv_root)).frame
    invalid = table.copy()
    invalid.loc[0, "FraudIndicator"] = 2
    with pytest.raises(EDAError, match="only non-null 0/1"):
        profile_dataset1(invalid)

    with pytest.raises(EDAError, match="missing columns"):
        profile_dataset2(pd.DataFrame({"Class": [0], "Amount": [1], "Time": [0]}))
