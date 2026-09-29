import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import IsolationForest

from Src.artifacts import load_model_artifact, save_model_artifact
from Src.inference import FraudInference, InferenceError
from Src.preprocessing import DATASET2_FEATURES, prepare_dataset2
from Src.risk_scoring import build_risk_scorer


def _frame(rows=400):
    row = np.arange(rows)
    frame = pd.DataFrame({"Time": row // 4, "Amount": row % 31})
    for index in range(1, 29):
        frame[f"V{index}"] = np.cos(row / (index + 2))
    frame["Class"] = (row % 20 == 0).astype(int)
    return frame


def _raw_record(row):
    return {feature: float(row[feature]) for feature in DATASET2_FEATURES}


def _engine(tmp_path, model_type="logistic"):
    frame = _frame()
    splits = prepare_dataset2(frame)
    if model_type == "logistic":
        estimator = LogisticRegression(class_weight="balanced", random_state=42).fit(splits.X_train, splits.y_train)
        model_name, score_kind, threshold = "Logistic Regression", "probability", 0.5
        reference_features = None
    else:
        estimator = IsolationForest(n_estimators=12, max_samples=128, random_state=42).fit(splits.X_train)
        model_name, score_kind, threshold = "Isolation Forest", "anomaly", None
        reference_features = splits.X_train
    scorer = build_risk_scorer(model_name, estimator, score_kind=score_kind,
                               feature_names=splits.feature_names, reference_features=reference_features)
    path = save_model_artifact(
        tmp_path / f"{model_type}-v1", model_name=model_name, estimator=estimator,
        preprocessor=splits.preprocessor, risk_scorer=scorer, dataset_id="dataset2",
        split_strategy=splits.split_strategy, train_rows=len(splits.X_train), decision_threshold=threshold,
    )
    return FraudInference(load_model_artifact(path)), frame, splits, scorer


def test_single_and_batch_inference_match_saved_artifact_scores(tmp_path):
    engine, frame, splits, scorer = _engine(tmp_path)
    record = _raw_record(frame.iloc[splits.X_test.index[0]])
    batch = engine.predict_transactions([record, record])
    single = engine.predict_transaction(record)
    expected = scorer.score_transaction(splits.X_test.iloc[[0]])

    assert len(batch) == 2
    assert batch[0].score == pytest.approx(expected.score)
    assert single.score == pytest.approx(expected.score)
    assert batch[0].score_kind == "probability"
    assert batch[0].decision_method == "positive-class probability >= 0.5"
    assert batch[0].model_alert is (expected.score >= 50.0)
    assert batch[0].inference_ms >= 0
    assert batch[0].score_basis == "uncalibrated model probability × 100"


@pytest.mark.parametrize("mutation, message", [
    (lambda row: row.pop("Amount"), "missing required"),
    (lambda row: row.update(Class=0), "unsupported fields"),
    (lambda row: row.update(V1=np.nan), "must be finite"),
    (lambda row: row.update(Amount=-1), "cannot be negative"),
])
def test_inference_rejects_invalid_request_before_model_scoring(tmp_path, mutation, message):
    engine, frame, splits, _ = _engine(tmp_path)
    record = _raw_record(frame.iloc[splits.X_test.index[0]])
    mutation(record)

    with pytest.raises(InferenceError, match=message):
        engine.predict_transaction(record)


def test_inference_requires_project_loaded_artifact():
    with pytest.raises(InferenceError, match="LoadedArtifact"):
        FraudInference("C:/some/arbitrary/model")


def test_inference_handles_extremely_large_finite_amount_without_nonfinite_score(tmp_path):
    engine, frame, splits, _ = _engine(tmp_path)
    record = _raw_record(frame.iloc[splits.X_test.index[0]])
    record["Amount"] = 1e300

    result = engine.predict_transaction(record)

    assert 0 <= result.score <= 100
    assert np.isfinite(result.inference_ms)


def test_isolation_inference_uses_native_anomaly_decision(tmp_path):
    engine, frame, splits, scorer = _engine(tmp_path, model_type="isolation")
    record = _raw_record(frame.iloc[splits.X_test.index[0]])
    result = engine.predict_transaction(record)
    transformed = splits.X_test.iloc[[0]]
    expected_alert = scorer.estimator.predict(transformed)[0] == -1

    assert result.score_kind == "anomaly"
    assert result.model_alert is bool(expected_alert)
    assert result.decision_method == "Isolation Forest native predict == -1"
