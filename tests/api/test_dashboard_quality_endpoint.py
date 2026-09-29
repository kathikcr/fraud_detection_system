from pathlib import Path
from types import SimpleNamespace

import pandas as pd
from fastapi.testclient import TestClient

from Src import dashboard
from Src.api import create_app
from Src.ingestion import IngestionError


def _loaded(frame, *, rows, columns, missing=None, duplicates=0, target=None):
    report = SimpleNamespace(
        rows=rows,
        columns=tuple(columns),
        missing_values=missing or {},
        duplicate_rows=duplicates,
        target_distribution=target or {},
        path=Path("ignored.csv"),
    )
    return SimpleNamespace(frame=frame, report=report)


def test_dataset2_quality_reports_observed_values_and_duplicates_without_paths(monkeypatch):
    frame = pd.DataFrame({"Time": [0.0, 1.0, 2.0], "Amount": [10.0, 20.0, 30.0], "Class": [0, 1, 0]})
    loaded = _loaded(frame, rows=3, columns=frame.columns, duplicates=1, target={0: 2, 1: 1})
    monkeypatch.setattr(dashboard, "load_dataset2", lambda: loaded)
    dashboard.clear_quality_cache()

    with TestClient(create_app()) as client:
        response = client.get("/dashboard/quality?dataset=dataset2")

    assert response.status_code == 200
    result = response.json()
    assert result["dataset"]["synthetic"] is False
    assert result["summary"]["transaction_rows"] == 3
    assert result["summary"]["target_distribution"] == {"0": 2, "1": 1}
    assert result["summary"]["fraud_rate"] == 1 / 3
    assert result["summary"]["duplicate_rows"] == 1
    assert next(item for item in result["checks"] if item["name"] == "Exact duplicate rows")["status"] == "warning"
    assert "ignored.csv" not in response.text
    assert result["summary"]["time_range"] == [0, 2]


def test_dataset1_quality_is_separate_privacy_conscious_and_checks_join_counts(monkeypatch):
    tables = {
        "transaction_records": _loaded(pd.DataFrame(), rows=5, columns=["TransactionID", "Amount", "CustomerID"]),
        "fraud_indicators": _loaded(pd.DataFrame(), rows=5, columns=["TransactionID", "FraudIndicator"], target={0: 4, 1: 1}),
    }
    relationship = SimpleNamespace(
        source_table="transaction_records", target_table="fraud_indicators", cardinality="one-to-one",
        matched_reference_keys=5, unmatched_reference_keys=0, unreferenced_target_keys=0,
    )
    audit = SimpleNamespace(input_transaction_rows=5, output_rows=5, relationships=(relationship,))
    monkeypatch.setattr(dashboard, "load_dataset1", lambda: tables)
    monkeypatch.setattr(dashboard, "build_dataset1_transaction_table", lambda _: SimpleNamespace(report=audit))
    dashboard.clear_quality_cache()

    with TestClient(create_app()) as client:
        response = client.get("/dashboard/quality?dataset=dataset1")

    assert response.status_code == 200
    result = response.json()
    assert result["dataset"] == {"id": "dataset1", "label": "Synthetic Financial Fraud Dataset", "synthetic": True}
    assert result["summary"]["transaction_rows"] == 5
    assert result["summary"]["target_distribution"] == {"0": 4, "1": 1}
    assert result["relationships"][0]["status"] == "passed"
    assert next(item for item in result["checks"] if item["name"] == "Label interpretation")["status"] == "info"
    assert "CustomerID" not in response.text


def test_dataset1_quality_reports_join_row_loss_and_unavailable_data(monkeypatch):
    tables = {"fraud_indicators": _loaded(pd.DataFrame(), rows=2, columns=["TransactionID", "FraudIndicator"])}
    monkeypatch.setattr(dashboard, "load_dataset1", lambda: tables)
    monkeypatch.setattr(dashboard, "build_dataset1_transaction_table", lambda _: (_ for _ in ()).throw(dashboard.Dataset1IntegrationError("bad join")))
    dashboard.clear_quality_cache()

    with TestClient(create_app()) as client:
        join_result = client.get("/dashboard/quality?dataset=dataset1")
        monkeypatch.setattr(dashboard, "load_dataset1", lambda: (_ for _ in ()).throw(IngestionError("local path detail")))
        dashboard.clear_quality_cache()
        unavailable = client.get("/dashboard/quality?dataset=dataset1")

    assert join_result.status_code == 200
    assert next(item for item in join_result.json()["checks"] if item["name"] == "Transaction joins")["status"] == "failed"
    assert unavailable.status_code == 503
    assert "local path detail" not in unavailable.text


def test_dataset2_quality_flags_non_numeric_and_negative_values(monkeypatch):
    frame = pd.DataFrame({"Time": [0.0, -1.0], "Amount": [10.0, "not-a-number"], "Class": [0, 1]})
    loaded = _loaded(frame, rows=2, columns=frame.columns, target={0: 1, 1: 1})
    monkeypatch.setattr(dashboard, "load_dataset2", lambda: loaded)
    dashboard.clear_quality_cache()

    with TestClient(create_app()) as client:
        response = client.get("/dashboard/quality?dataset=dataset2")

    assert response.status_code == 200
    checks = {item["name"]: item["status"] for item in response.json()["checks"]}
    assert checks["Finite numeric values"] == "failed"
    assert checks["Time and Amount ranges"] == "failed"
