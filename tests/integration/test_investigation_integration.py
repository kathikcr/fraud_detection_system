import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression

from Src.artifacts import load_model_artifact, save_model_artifact
from Src.investigation import InvestigationError, TransactionInvestigator
from Src.preprocessing import DATASET2_FEATURES, prepare_dataset2
from Src.risk_scoring import build_risk_scorer


def _frame(rows=400):
    row = np.arange(rows)
    frame = pd.DataFrame({"Time": row // 4, "Amount": row % 31})
    for index in range(1, 29):
        frame[f"V{index}"] = np.cos(row / (index + 2))
    frame["Class"] = (row % 20 == 0).astype(int)
    return frame


def _investigator(tmp_path, model_type):
    frame = _frame()
    splits = prepare_dataset2(frame)
    if model_type == "logistic":
        estimator = LogisticRegression(class_weight="balanced", random_state=42).fit(splits.X_train, splits.y_train)
        name, kind, threshold, reference = "Logistic Regression", "probability", 0.5, None
    else:
        estimator = IsolationForest(n_estimators=8, max_samples=128, random_state=42).fit(splits.X_train)
        name, kind, threshold, reference = "Isolation Forest", "anomaly", None, splits.X_train
    scorer = build_risk_scorer(name, estimator, score_kind=kind,
                               feature_names=splits.feature_names, reference_features=reference)
    path = save_model_artifact(
        tmp_path / f"{model_type}-artifact", model_name=name, estimator=estimator,
        preprocessor=splits.preprocessor, risk_scorer=scorer, dataset_id="dataset2",
        split_strategy=splits.split_strategy, train_rows=len(splits.X_train), decision_threshold=threshold,
    )
    artifact = load_model_artifact(path)
    investigator = TransactionInvestigator(
        artifact, splits.X_train, seed=42, background_rows=8, max_evals=61, top_contributors=5
    )
    raw = frame.iloc[splits.X_test.index[0]]
    transaction = {name: float(raw[name]) for name in DATASET2_FEATURES}
    return investigator, transaction


@pytest.mark.parametrize("model_type, score_kind", [("logistic", "probability"), ("isolation", "anomaly")])
def test_investigation_bundles_prediction_and_ranked_explanation(tmp_path, model_type, score_kind):
    investigator, transaction = _investigator(tmp_path, model_type)

    result = investigator.investigate(" case-2026-001 ", transaction)

    assert result.transaction_id == "case-2026-001"
    assert result.time == transaction["Time"]
    assert result.amount == transaction["Amount"]
    assert result.prediction.score_kind == score_kind
    assert result.explanation.score_kind == score_kind
    assert result.explanation.score == pytest.approx(result.prediction.score)
    assert len(result.top_contributors) == 5
    assert [abs(item.shap_value) for item in result.top_contributors] == sorted(
        [abs(item.shap_value) for item in result.top_contributors], reverse=True
    )
    assert all(item.feature_name in DATASET2_FEATURES for item in result.top_contributors)
    assert result.investigation_ms >= result.prediction.inference_ms


def test_investigation_rejects_ids_and_payload_fields_that_could_be_misused(tmp_path):
    investigator, transaction = _investigator(tmp_path, "logistic")
    transaction_with_id = transaction | {"transaction_id": "should-not-be-a-feature"}

    with pytest.raises(InvestigationError, match="transaction_id"):
        investigator.investigate("  ", transaction)
    with pytest.raises(InvestigationError, match="unsupported fields"):
        investigator.investigate("case-2", transaction_with_id)
