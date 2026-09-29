import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression

import Src.api as api_module
from Src.api import create_app
from Src.artifacts import save_model_artifact
from Src.preprocessing import DATASET2_FEATURES, prepare_dataset2
from Src.risk_scoring import build_risk_scorer


@pytest.fixture
def investigation_setup(tmp_path):
    row = np.arange(400)
    frame = pd.DataFrame({"Time": row // 4, "Amount": row % 31})
    for index in range(1, 29):
        frame[f"V{index}"] = np.cos(row / (index + 2))
    frame["Class"] = (row % 20 == 0).astype(int)
    splits = prepare_dataset2(frame)
    estimator = LogisticRegression(class_weight="balanced", random_state=42).fit(splits.X_train, splits.y_train)
    scorer = build_risk_scorer("Logistic Regression", estimator, score_kind="probability",
                               feature_names=splits.feature_names)
    artifact_dir = save_model_artifact(
        tmp_path / "lr-v1", model_name="Logistic Regression", estimator=estimator,
        preprocessor=splits.preprocessor, risk_scorer=scorer, dataset_id="dataset2",
        split_strategy=splits.split_strategy, train_rows=len(splits.X_train), decision_threshold=0.5,
    )
    background_path = tmp_path / "training-background.csv"
    splits.X_train.sample(n=8, random_state=42).to_csv(background_path, index=False)
    raw = frame.iloc[splits.X_test.index[0]]
    transaction = {feature: float(raw[feature]) for feature in DATASET2_FEATURES}
    return artifact_dir, background_path, transaction


def test_investigation_endpoint_returns_prediction_and_attributions(investigation_setup, monkeypatch):
    artifact_dir, background_path, transaction = investigation_setup
    original_loader = api_module.load_model_artifact
    original_read_csv = api_module.pd.read_csv
    loader_calls, background_reads = [], []

    def counted_loader(path):
        loader_calls.append(path)
        return original_loader(path)

    def counted_read_csv(path, *args, **kwargs):
        background_reads.append(path)
        return original_read_csv(path, *args, **kwargs)

    monkeypatch.setattr(api_module, "load_model_artifact", counted_loader)
    monkeypatch.setattr(api_module.pd, "read_csv", counted_read_csv)
    application = create_app(artifact_dir=artifact_dir, background_path=background_path)
    with TestClient(application) as client:
        prediction = client.post("/predict", json=transaction)
        response = client.post("/investigate", json={
            "transaction_id": "review-case-17", "transaction": transaction,
        })
        cached = client.post("/investigate", json={
            "transaction_id": "review-case-18", "transaction": transaction,
        })

    assert prediction.status_code == 200
    assert response.status_code == 200
    result = response.json()
    assert result["transaction_id"] == "review-case-17"
    assert result["time"] == transaction["Time"]
    assert result["amount"] == transaction["Amount"]
    assert result["prediction"]["score"] == pytest.approx(prediction.json()["score"])
    assert result["explanation"]["score"] == pytest.approx(result["prediction"]["score"])
    assert result["explanation"]["score_kind"] == "probability"
    assert len(result["explanation"]["attributions"]) == len(DATASET2_FEATURES)
    assert len(result["top_contributors"]) == 5
    assert result["investigation_ms"] >= 0
    reconstructed = result["explanation"]["baseline_score"] + sum(
        item["shap_value"] for item in result["explanation"]["attributions"]
    )
    assert reconstructed == pytest.approx(result["prediction"]["score"], abs=1e-4)
    assert cached.status_code == 200
    assert len(loader_calls) == 1
    assert len(background_reads) == 1


@pytest.mark.parametrize("change", [
    lambda body: body.update(transaction_id="   "),
    lambda body: body["transaction"].update(Class=0),
    lambda body: body["transaction"].pop("V28"),
    lambda body: body.update(background_path="C:/user/model.csv"),
])
def test_investigation_endpoint_rejects_invalid_request_without_echoing_values(
    investigation_setup, change
):
    artifact_dir, background_path, transaction = investigation_setup
    payload = {"transaction_id": "case-1", "transaction": transaction.copy()}
    change(payload)
    app = create_app(artifact_dir=artifact_dir, background_path=background_path)

    with TestClient(app) as client:
        response = client.post("/investigate", json=payload)

    assert response.status_code == 422
    assert "C:/user/model.csv" not in response.text


def test_investigation_endpoint_requires_configured_training_background(investigation_setup):
    artifact_dir, _, transaction = investigation_setup
    app = create_app(artifact_dir=artifact_dir)

    with TestClient(app) as client:
        response = client.post("/investigate", json={
            "transaction_id": 17, "transaction": transaction,
        })

    assert response.status_code == 503
    assert response.json() == {"detail": "Investigation service is unavailable"}


def test_investigation_endpoint_rejects_background_with_wrong_schema(investigation_setup, tmp_path):
    artifact_dir, _, transaction = investigation_setup
    invalid_background = tmp_path / "wrong-background.csv"
    pd.DataFrame({"Time": [0.0], "V1": [0.0]}).to_csv(invalid_background, index=False)
    app = create_app(artifact_dir=artifact_dir, background_path=invalid_background)

    with TestClient(app) as client:
        response = client.post("/investigate", json={
            "transaction_id": 17, "transaction": transaction,
        })

    assert response.status_code == 503
    assert str(invalid_background) not in response.text


@pytest.mark.parametrize("background", [
    pd.DataFrame(np.zeros((101, len(DATASET2_FEATURES))), columns=DATASET2_FEATURES),
    pd.DataFrame([[np.inf] * len(DATASET2_FEATURES)], columns=DATASET2_FEATURES),
])
def test_investigation_endpoint_rejects_oversized_or_nonfinite_background(
    investigation_setup, tmp_path, background
):
    artifact_dir, _, transaction = investigation_setup
    background_path = tmp_path / "invalid-background.csv"
    background.to_csv(background_path, index=False)
    app = create_app(artifact_dir=artifact_dir, background_path=background_path)

    with TestClient(app) as client:
        response = client.post("/investigate", json={
            "transaction_id": "case-17", "transaction": transaction,
        })

    assert response.status_code == 503
    assert str(background_path) not in response.text
