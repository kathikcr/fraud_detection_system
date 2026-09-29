from types import SimpleNamespace

import pandas as pd
from fastapi.testclient import TestClient

from Src import dashboard
from Src.api import create_app


def test_dataset1_overview_is_separate_and_labels_synthetic_data(monkeypatch):
    frame = pd.DataFrame({
        "TransactionID": ["t1", "t2", "t3"],
        "Amount": [10.0, 20.0, 30.0],
        "FraudIndicator": [0, 1, 0],
        "Timestamp": ["2022-01-01", "2022-01-01", "2022-01-02"],
        "Category": ["Retail", "Travel", "Retail"],
        "Name": ["private-a", "private-b", "private-c"],
    })
    monkeypatch.setattr(dashboard, "load_dataset1_transaction_table", lambda: SimpleNamespace(frame=frame))
    dashboard.clear_overview_cache()

    with TestClient(create_app()) as client:
        response = client.get("/dashboard/overview?dataset=dataset1")

    assert response.status_code == 200
    result = response.json()
    assert result["dataset"]["synthetic"] is True
    assert result["kpis"]["total_transactions"] == 3
    assert result["kpis"]["fraud_transactions"] == 1
    assert result["kpis"]["total_amount"] == 60
    assert result["kpis"]["high_risk_transactions"] is None
    assert len(result["trend"]) == 2
    assert {item["category"] for item in result["category_fraud"]} == {"Retail", "Travel"}
    assert "private-a" not in response.text
    assert "Name" not in response.text


def test_dataset2_overview_uses_elapsed_time_and_has_no_categories(monkeypatch):
    frame = pd.DataFrame({"Amount": [12.0, 40.0, 5.0], "Class": [0, 1, 0], "Time": [0, 7200, 14400]})
    monkeypatch.setattr(dashboard, "load_dataset2", lambda: SimpleNamespace(frame=frame))
    dashboard.clear_overview_cache()

    with TestClient(create_app()) as client:
        response = client.get("/dashboard/overview?dataset=dataset2")

    assert response.status_code == 200
    result = response.json()
    assert result["dataset"]["synthetic"] is False
    assert result["kpis"]["total_transactions"] == 3
    assert result["trend"][0]["label"] == "00–02 h"
    assert result["trend"][1]["fraud"] == 1
    assert result["category_fraud"] == []


def test_dashboard_overview_rejects_unknown_dataset_and_reports_unavailable_data(monkeypatch):
    with TestClient(create_app()) as client:
        invalid = client.get("/dashboard/overview?dataset=combined")
        monkeypatch.setattr(dashboard, "load_dataset1_transaction_table", lambda: (_ for _ in ()).throw(FileNotFoundError("local path")))
        unavailable = client.get("/dashboard/overview?dataset=dataset1")

    assert invalid.status_code == 422
    assert unavailable.status_code == 503
    assert "local path" not in unavailable.text
