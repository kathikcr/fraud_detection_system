import json

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression

import Src.api as api_module
from Src.api import PredictionRequest, create_app
from Src.artifacts import load_model_artifact, save_model_artifact
from Src.inference import FraudInference
from Src.preprocessing import DATASET2_FEATURES, prepare_dataset2
from Src.risk_scoring import build_risk_scorer


@pytest.fixture
def trained_artifact(tmp_path):
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
    raw = frame.iloc[splits.X_test.index[0]]
    request = {feature: float(raw[feature]) for feature in DATASET2_FEATURES}
    return artifact_dir, load_model_artifact(artifact_dir), request


def test_prediction_endpoint_uses_cached_artifact_and_returns_typed_score(trained_artifact, monkeypatch):
    artifact_dir, artifact, payload = trained_artifact
    expected = FraudInference(artifact).predict_transaction(payload)
    original_loader = api_module.load_model_artifact
    calls = []

    def counted_loader(path):
        calls.append(path)
        return original_loader(path)

    monkeypatch.setattr(api_module, "load_model_artifact", counted_loader)
    application = create_app(artifact_dir=artifact_dir)
    with TestClient(application) as client:
        first = client.post("/predict", json=payload)
        second = client.post("/predict", json=payload)

    assert first.status_code == 200
    assert first.json()["model_name"] == expected.model_name
    assert first.json()["score"] == pytest.approx(expected.score)
    assert first.json()["score_kind"] == "probability"
    assert first.json()["score_basis"] == "uncalibrated model probability × 100"
    assert first.json()["risk_level"] == expected.risk_level
    assert first.json()["model_alert"] is expected.model_alert
    assert first.json()["decision_method"] == expected.decision_method
    assert first.json()["inference_ms"] >= 0
    assert second.status_code == 200
    assert len(calls) == 1


@pytest.mark.parametrize("mutation", [
    lambda body: body.pop("V1"),
    lambda body: body.update(Class=0),
    lambda body: body.update(Amount="12.0"),
    lambda body: body.update(Amount=-1.0),
    lambda body: body.update(V2=float("nan")),
    lambda body: body.update(V3=float("inf")),
    lambda body: body.update(artifact_dir="C:/model"),
])
def test_prediction_endpoint_rejects_invalid_payload_without_echoing_values(
    trained_artifact, mutation
):
    _, _, payload = trained_artifact
    bad_payload = payload.copy()
    mutation(bad_payload)
    app = create_app(inference_service=FraudInference(load_model_artifact(trained_artifact[0])))

    with TestClient(app) as client:
        response = client.post(
            "/predict",
            content=json.dumps(bad_payload, allow_nan=True),
            headers={"Content-Type": "application/json"},
        )

    assert response.status_code == 422
    assert "C:/model" not in response.text


def test_prediction_endpoint_returns_unavailable_without_artifact_configuration(
    trained_artifact, monkeypatch
):
    monkeypatch.delenv("FRAUD_MODEL_ARTIFACT_DIR", raising=False)
    app = create_app()

    with TestClient(app) as client:
        response = client.post("/predict", json=trained_artifact[2])

    assert response.status_code == 503
    assert response.json() == {"detail": "Prediction service is unavailable"}


def test_prediction_schema_validation_runs_before_model_configuration(trained_artifact, monkeypatch):
    monkeypatch.delenv("FRAUD_MODEL_ARTIFACT_DIR", raising=False)
    payload = trained_artifact[2].copy()
    del payload["V1"]

    with TestClient(create_app()) as client:
        response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_prediction_endpoint_hides_artifact_loader_errors(trained_artifact):
    app = create_app(artifact_dir=trained_artifact[0] / "missing")

    with TestClient(app) as client:
        response = client.post("/predict", json=trained_artifact[2])

    assert response.status_code == 503
    assert str(trained_artifact[0]) not in response.text


@pytest.mark.parametrize("field, value", [("Time", -1), ("Amount", -0.1)])
def test_prediction_request_model_enforces_nonnegative_fields(field, value, trained_artifact):
    payload = trained_artifact[2].copy()
    payload[field] = value

    with pytest.raises(ValueError, match="cannot be negative"):
        PredictionRequest.model_validate(payload)
