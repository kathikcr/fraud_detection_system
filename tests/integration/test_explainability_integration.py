import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from Src.artifacts import load_model_artifact, save_model_artifact
from Src.explainability import ExplainabilityError, FraudExplainer
from Src.preprocessing import prepare_dataset2
from Src.risk_scoring import build_risk_scorer


def _frame(rows=400):
    row = np.arange(rows)
    frame = pd.DataFrame({"Time": row // 4, "Amount": row % 31})
    for index in range(1, 29):
        frame[f"V{index}"] = np.cos(row / (index + 2))
    frame["Class"] = (row % 20 == 0).astype(int)
    return frame


def _loaded(tmp_path, splits, model_type):
    if model_type == "logistic":
        estimator = LogisticRegression(class_weight="balanced", random_state=42).fit(splits.X_train, splits.y_train)
        name, kind, threshold = "Logistic Regression", "probability", 0.5
        reference = None
    elif model_type == "forest":
        estimator = RandomForestClassifier(n_estimators=8, max_depth=3, random_state=42).fit(splits.X_train, splits.y_train)
        name, kind, threshold = "Random Forest", "probability", 0.5
        reference = None
    elif model_type == "xgboost":
        estimator = XGBClassifier(n_estimators=8, max_depth=2, learning_rate=0.1, n_jobs=1,
                                  eval_metric="logloss", random_state=42).fit(splits.X_train, splits.y_train)
        name, kind, threshold = "XGBoost", "probability", 0.5
        reference = None
    else:
        estimator = IsolationForest(n_estimators=8, max_samples=128, random_state=42).fit(splits.X_train)
        name, kind, threshold = "Isolation Forest", "anomaly", None
        reference = splits.X_train
    scorer = build_risk_scorer(name, estimator, score_kind=kind, feature_names=splits.feature_names,
                               reference_features=reference)
    path = save_model_artifact(
        tmp_path / f"{model_type}-artifact", model_name=name, estimator=estimator,
        preprocessor=splits.preprocessor, risk_scorer=scorer, dataset_id="dataset2",
        split_strategy=splits.split_strategy, train_rows=len(splits.X_train), decision_threshold=threshold,
    )
    return load_model_artifact(path)


@pytest.fixture(scope="module")
def splits():
    return prepare_dataset2(_frame())


@pytest.mark.parametrize("model_type", ["logistic", "forest", "xgboost", "isolation"])
def test_shap_explains_saved_model_risk_score_additively(tmp_path, splits, model_type):
    artifact = _loaded(tmp_path, splits, model_type)
    explainer = FraudExplainer(artifact, splits.X_train, seed=42, background_rows=8)
    features = splits.X_test.iloc[[0, 1]]

    results = explainer.explain_transactions(features, max_evals=61)

    expected = artifact.risk_scorer.score_transactions(features)
    assert len(results) == 2
    for explanation, score in zip(results, expected, strict=True):
        assert explanation.model_name == artifact.manifest["model_name"]
        assert explanation.score_kind == score.score_kind
        assert explanation.score_basis == score.score_basis
        assert explanation.score == pytest.approx(score.score)
        assert explanation.baseline_score + sum(item.shap_value for item in explanation.attributions) == pytest.approx(
            explanation.score, abs=1e-4
        )
        assert len(explanation.attributions) == len(splits.feature_names)
        assert tuple(item.feature_name for item in explanation.attributions) == splits.feature_names


def test_explainer_rejects_wrong_schema_and_too_few_evaluations(tmp_path, splits):
    artifact = _loaded(tmp_path, splits, "logistic")
    explainer = FraudExplainer(artifact, splits.X_train, background_rows=8)
    with pytest.raises(ExplainabilityError, match="columns and order"):
        explainer.explain_transactions(splits.X_test.iloc[[0]].rename(columns={"Time": "unknown"}), max_evals=61)
    with pytest.raises(ExplainabilityError, match="max_evals"):
        explainer.explain_transactions(splits.X_test.iloc[[0]], max_evals=60)


def test_explainer_requires_training_background_feature_contract(tmp_path, splits):
    artifact = _loaded(tmp_path, splits, "logistic")
    with pytest.raises(ExplainabilityError, match="columns and order"):
        FraudExplainer(artifact, splits.X_train.rename(columns={"Time": "wrong"}))
    with pytest.raises(ExplainabilityError, match="finite"):
        FraudExplainer(artifact, splits.X_train.assign(Time=np.inf))
