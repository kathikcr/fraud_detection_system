import pandas as pd
import pytest

from Src.dataset1_integration import (
    Dataset1IntegrationError,
    build_dataset1_transaction_table,
)


def sample_tables():
    return {
        "transaction_records": pd.DataFrame({
            "TransactionID": [1, 2, 3], "Amount": [10.0, 20.0, 30.0], "CustomerID": [10, 10, 20],
        }),
        "transaction_metadata": pd.DataFrame({
            "TransactionID": [1, 2, 3], "Timestamp": ["t1", "t2", "t3"], "MerchantID": [100, 100, 200],
        }),
        "fraud_indicators": pd.DataFrame({"TransactionID": [1, 2, 3], "FraudIndicator": [0, 1, 0]}),
        "suspicious_activity": pd.DataFrame({"CustomerID": [10, 20, 30], "SuspiciousFlag": [0, 1, 0]}),
        "customer_data": pd.DataFrame({
            "CustomerID": [10, 20, 30], "Name": ["A", "B", "C"],
            "Address": ["a", "b", "c"], "Age": [30, 40, 50],
        }),
        "account_activity": pd.DataFrame({
            "CustomerID": [10, 20], "AccountBalance": [1000.0, 2000.0], "LastLogin": ["x", "y"],
        }),
        "merchant_data": pd.DataFrame({
            "MerchantID": [100, 200, 300], "MerchantName": ["M1", "M2", "M3"], "Location": ["L1", "L2", "L3"],
        }),
        "transaction_category_labels": pd.DataFrame({"TransactionID": [1, 2, 3], "Category": ["A", "B", "C"]}),
        "amount_data": pd.DataFrame({"TransactionID": [1, 2, 3], "TransactionAmount": [110.0, 120.0, 130.0]}),
        "anomaly_scores": pd.DataFrame({"TransactionID": [1, 2, 3], "AnomalyScore": [0.1, 0.2, 0.3]}),
    }


def test_builds_transaction_grain_table_and_preserves_target_and_amount_fields():
    tables = sample_tables()

    result = build_dataset1_transaction_table(tables)

    assert len(result.frame) == 3
    assert result.frame["TransactionID"].tolist() == [1, 2, 3]
    assert result.frame["FraudIndicator"].tolist() == tables["fraud_indicators"]["FraudIndicator"].tolist()
    assert result.report.target_distribution == {0: 2, 1: 1}
    assert result.report.input_transaction_rows == result.report.output_rows == 3
    assert result.frame["Amount"].tolist() == [10.0, 20.0, 30.0]
    assert result.frame["TransactionAmount"].tolist() == [110.0, 120.0, 130.0]


def test_excludes_customer_personal_and_account_activity_fields():
    result = build_dataset1_transaction_table(sample_tables())

    assert not {"Name", "Address", "Age", "AccountBalance", "LastLogin"} & set(result.frame.columns)
    assert result.report.excluded_personal_columns == ("Name", "Address", "Age", "AccountBalance", "LastLogin")


def test_audits_many_to_one_relationships_and_unused_dimension_rows():
    result = build_dataset1_transaction_table(sample_tables())
    by_target = {(item.target_table, item.key): item for item in result.report.relationships}

    assert by_target[("customer_data", "CustomerID")].cardinality == "many-to-one"
    assert by_target[("customer_data", "CustomerID")].unreferenced_target_keys == 1
    assert by_target[("merchant_data", "MerchantID")].unreferenced_target_keys == 1
    assert by_target[("fraud_indicators", "TransactionID")].cardinality == "one-to-one"


def test_rejects_duplicate_transaction_keys_before_join():
    tables = sample_tables()
    tables["transaction_records"] = pd.concat(
        [tables["transaction_records"], tables["transaction_records"].iloc[[0]]], ignore_index=True
    )

    with pytest.raises(Dataset1IntegrationError, match="transaction_records key TransactionID has .* duplicate"):
        build_dataset1_transaction_table(tables)


def test_rejects_duplicate_satellite_keys_to_prevent_join_explosion():
    tables = sample_tables()
    tables["fraud_indicators"] = pd.concat(
        [tables["fraud_indicators"], tables["fraud_indicators"].iloc[[0]]], ignore_index=True
    )

    with pytest.raises(Dataset1IntegrationError, match="fraud_indicators key TransactionID has .* duplicate"):
        build_dataset1_transaction_table(tables)


def test_rejects_unmatched_customer_foreign_key():
    tables = sample_tables()
    tables["transaction_records"].loc[0, "CustomerID"] = 999

    with pytest.raises(Dataset1IntegrationError, match=r"unmatched value\(s\).*customer_data"):
        build_dataset1_transaction_table(tables)


def test_rejects_null_foreign_key():
    tables = sample_tables()
    tables["transaction_metadata"].loc[0, "MerchantID"] = None

    with pytest.raises(Dataset1IntegrationError, match="null values in key MerchantID"):
        build_dataset1_transaction_table(tables)


def test_rejects_unexpected_orphan_transaction_rows():
    tables = sample_tables()
    tables["amount_data"] = pd.concat([
        tables["amount_data"], pd.DataFrame({"TransactionID": [4], "TransactionAmount": [99.0]})
    ], ignore_index=True)

    with pytest.raises(Dataset1IntegrationError, match="Unexpected orphan key.*amount_data"):
        build_dataset1_transaction_table(tables)


def test_rejects_invalid_fraud_target():
    tables = sample_tables()
    tables["fraud_indicators"].loc[0, "FraudIndicator"] = 3

    with pytest.raises(Dataset1IntegrationError, match="FraudIndicator must be non-null"):
        build_dataset1_transaction_table(tables)


def test_rejects_missing_table():
    tables = sample_tables()
    del tables["anomaly_scores"]

    with pytest.raises(Dataset1IntegrationError, match="Missing Dataset 1 tables"):
        build_dataset1_transaction_table(tables)
