from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture
def dataset1_csv_root(tmp_path: Path) -> Path:
    """Small synthetic Dataset 1 fixture; tests never need raw CSVs committed."""
    root = tmp_path / "Data"
    tables = {
        "Customer Profiles/account_activity.csv": pd.DataFrame({
            "CustomerID": [10, 20, 30], "AccountBalance": [1000.0, 2000.0, 3000.0],
            "LastLogin": ["2022-01-01", "2022-01-02", "2022-01-03"],
        }),
        "Customer Profiles/customer_data.csv": pd.DataFrame({
            "CustomerID": [10, 20, 30], "Name": ["Test A", "Test B", "Test C"],
            "Age": [31, 42, 53], "Address": ["A", "B", "C"],
        }),
        "Fraudulent Patterns/fraud_indicators.csv": pd.DataFrame({
            "TransactionID": [1, 2, 3], "FraudIndicator": [0, 1, 0],
        }),
        "Fraudulent Patterns/suspicious_activity.csv": pd.DataFrame({
            "CustomerID": [10, 20, 30], "SuspiciousFlag": [0, 1, 0],
        }),
        "Merchant Information/merchant_data.csv": pd.DataFrame({
            "MerchantID": [100, 200, 300], "MerchantName": ["M1", "M2", "M3"],
            "Location": ["L1", "L2", "L3"],
        }),
        "Merchant Information/transaction_category_labels.csv": pd.DataFrame({
            "TransactionID": [1, 2, 3], "Category": ["Food", "Travel", "Retail"],
        }),
        "Transaction Amounts/amount_data.csv": pd.DataFrame({
            "TransactionID": [1, 2, 3], "TransactionAmount": [110.0, 120.0, 130.0],
        }),
        "Transaction Amounts/anomaly_scores.csv": pd.DataFrame({
            "TransactionID": [1, 2, 3], "AnomalyScore": [0.1, 0.2, 0.3],
        }),
        "Transaction Data/transaction_metadata.csv": pd.DataFrame({
            "TransactionID": [1, 2, 3], "Timestamp": ["t1", "t2", "t3"],
            "MerchantID": [100, 100, 200],
        }),
        "Transaction Data/transaction_records.csv": pd.DataFrame({
            "TransactionID": [1, 2, 3], "Amount": [10.0, 20.0, 30.0],
            "CustomerID": [10, 10, 20],
        }),
    }
    for relative_path, frame in tables.items():
        destination = root / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(destination, index=False)
    return root
