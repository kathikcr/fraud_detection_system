import json

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from Src.artifacts import ArtifactError, load_model_artifact, save_model_artifact
from Src.preprocessing import prepare_dataset2
from Src.risk_scoring import build_risk_scorer


def _raw_rows(indices):
    row = np.asarray(indices)
    frame = pd.DataFrame({"Time": row // 4, "Amount": row % 31}, index=row)
    for index in range(1, 29):
        frame[f"V{index}"] = np.cos(row / (index + 2))
    return frame


@pytest.fixture(scope="module")
def splits():
    row = np.arange(400)
    frame = pd.DataFrame({"Time": row // 4, "Amount": row % 31})
    for index in range(1, 29):
        frame[f"V{index}"] = np.cos(row / (index + 2))
    frame["Class"] = (row % 20 == 0).astype(int)
    return prepare_dataset2(frame)


@pytest.mark.parametrize("model_type", ["logistic", "forest", "xgboost", "isolation"])
def test_artifacts_round_trip_prediction_and_metadata(tmp_path, splits, model_type):
    if model_type == "logistic":
        estimator, name, kind = LogisticRegression(class_weight="balanced", random_state=42), "Logistic Regression", "probability"
    elif model_type == "forest":
        estimator, name, kind = RandomForestClassifier(n_estimators=12, max_depth=3, random_state=42), "Random Forest", "probability"
    elif model_type == "xgboost":
        estimator = XGBClassifier(n_estimators=8, max_depth=2, learning_rate=0.1, n_jobs=1,
                                  eval_metric="logloss", random_state=42)
        name, kind = "XGBoost", "probability"
    else:
        estimator, name, kind = IsolationForest(n_estimators=12, max_samples=128, random_state=42), "Isolation Forest", "anomaly"
    estimator.fit(splits.X_train, splits.y_train) if kind == "probability" else estimator.fit(splits.X_train)
    scorer = build_risk_scorer(
        name, estimator, score_kind=kind, feature_names=splits.feature_names,
        reference_features=splits.X_train if kind == "anomaly" else None,
    )
    expected = scorer.score_transaction(splits.X_test.iloc[[0]])
    artifact_path = save_model_artifact(
        tmp_path / model_type, model_name=name, estimator=estimator,
        preprocessor=splits.preprocessor, risk_scorer=scorer,
        dataset_id="kaggle-credit-card-fraud", split_strategy="forward-chronological",
        train_rows=len(splits.X_train), decision_threshold=0.5 if kind == "probability" else None,
    )

    loaded = load_model_artifact(artifact_path)
    raw = _raw_rows([splits.X_test.index[0]])
    transformed = loaded.preprocessor.transform(raw)
    actual = loaded.risk_scorer.score_transaction(pd.DataFrame(transformed, columns=splits.feature_names))
    assert actual == expected
    assert loaded.manifest["model_name"] == name
    assert loaded.manifest["feature_names"] == list(splits.feature_names)
    assert loaded.manifest["train_rows"] == len(splits.X_train)
    if kind == "anomaly":
        assert loaded.risk_scorer.sorted_reference_anomaly_scores == scorer.sorted_reference_anomaly_scores


def test_artifact_detects_bundle_tampering(tmp_path, splits):
    estimator = LogisticRegression(random_state=42).fit(splits.X_train, splits.y_train)
    scorer = build_risk_scorer("Logistic Regression", estimator, score_kind="probability", feature_names=splits.feature_names)
    root = save_model_artifact(
        tmp_path / "model", model_name="Logistic Regression", estimator=estimator,
        preprocessor=splits.preprocessor, risk_scorer=scorer, dataset_id="dataset2",
        split_strategy="forward-chronological", train_rows=len(splits.X_train), decision_threshold=0.5,
    )
    with (root / "bundle.skops").open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(ArtifactError, match="checksum mismatch"):
        load_model_artifact(root)


def test_artifact_rejects_manifest_score_basis_mismatch(tmp_path, splits):
    estimator = LogisticRegression(random_state=42).fit(splits.X_train, splits.y_train)
    scorer = build_risk_scorer("Logistic Regression", estimator, score_kind="probability", feature_names=splits.feature_names)
    root = save_model_artifact(
        tmp_path / "model", model_name="Logistic Regression", estimator=estimator,
        preprocessor=splits.preprocessor, risk_scorer=scorer, dataset_id="dataset2",
        split_strategy="forward-chronological", train_rows=len(splits.X_train), decision_threshold=0.5,
    )
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["score_basis"] = "fraud probability"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ArtifactError, match="score_basis"):
        load_model_artifact(root)


def test_artifacts_are_immutable(tmp_path, splits):
    estimator = LogisticRegression(random_state=42).fit(splits.X_train, splits.y_train)
    scorer = build_risk_scorer("Logistic Regression", estimator, score_kind="probability", feature_names=splits.feature_names)
    args = dict(model_name="Logistic Regression", estimator=estimator, preprocessor=splits.preprocessor,
                risk_scorer=scorer, dataset_id="dataset2", split_strategy="forward-chronological",
                train_rows=len(splits.X_train), decision_threshold=0.5)
    root = tmp_path / "model"
    save_model_artifact(root, **args)
    with pytest.raises(ArtifactError, match="immutable"):
        save_model_artifact(root, **args)
