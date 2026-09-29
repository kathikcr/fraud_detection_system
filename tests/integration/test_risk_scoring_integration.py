import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression

from Src.preprocessing import prepare_dataset2
from Src.risk_scoring import build_risk_scorer


def test_risk_scoring_works_with_training_only_preprocessing_and_independent_models():
    row = np.arange(400)
    frame = pd.DataFrame({"Time": row // 4, "Amount": row % 31})
    for index in range(1, 29):
        frame[f"V{index}"] = np.cos(row / (index + 2))
    frame["Class"] = (row % 20 == 0).astype(int)
    splits = prepare_dataset2(frame)
    logistic = LogisticRegression(class_weight="balanced", random_state=42).fit(splits.X_train, splits.y_train)
    isolation = IsolationForest(n_estimators=30, max_samples=128, contamination="auto", random_state=42).fit(splits.X_train)
    probability_scorer = build_risk_scorer(
        "Logistic Regression", logistic, score_kind="probability", feature_names=splits.feature_names
    )
    anomaly_scorer = build_risk_scorer(
        "Isolation Forest", isolation, score_kind="anomaly", feature_names=splits.feature_names,
        reference_features=splits.X_train,
    )

    probability_result = probability_scorer.score_transaction(splits.X_test.iloc[[0]])
    anomaly_result = anomaly_scorer.score_transaction(splits.X_test.iloc[[0]])

    assert probability_result.score_kind == "probability"
    assert anomaly_result.score_kind == "anomaly"
    assert probability_result.score_basis != anomaly_result.score_basis
    assert 0 <= probability_result.score <= 100
    assert 0 <= anomaly_result.score <= 100
