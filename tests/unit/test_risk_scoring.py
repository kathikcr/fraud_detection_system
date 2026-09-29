import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression

from Src.preprocessing import prepare_dataset2
from Src.risk_scoring import RiskBandPolicy, RiskScoringError, build_risk_scorer


def toy_dataset2(rows=400):
    row = np.arange(rows)
    frame = pd.DataFrame({"Time": row // 4, "Amount": (row % 17).astype(float)})
    for index in range(1, 29):
        frame[f"V{index}"] = ((row * index) % 23).astype(float) / (index + 1)
    frame["Class"] = (row % 20 == 0).astype(int)
    return frame


def fitted_models(splits):
    logistic = LogisticRegression(class_weight="balanced", random_state=42).fit(splits.X_train, splits.y_train)
    isolation = IsolationForest(n_estimators=30, max_samples=128, contamination="auto", random_state=42).fit(splits.X_train)
    return logistic, isolation


def test_probability_scores_map_to_zero_to_hundred_with_display_bands():
    splits = prepare_dataset2(toy_dataset2())
    logistic, _ = fitted_models(splits)
    scorer = build_risk_scorer(
        "LR", logistic, score_kind="probability", feature_names=splits.feature_names,
        policy=RiskBandPolicy(low_upper=25, medium_upper=65),
    )
    batch = scorer.score_transactions(splits.X_test.iloc[:5])
    expected = logistic.predict_proba(splits.X_test.iloc[:5])[:, list(logistic.classes_).index(1)] * 100

    assert len(batch) == 5
    np.testing.assert_allclose([item.score for item in batch], expected)
    assert all(item.score_kind == "probability" for item in batch)
    assert all(item.score_basis == "uncalibrated model probability × 100" for item in batch)
    assert all(item.risk_level in {"low", "medium", "high"} for item in batch)
    assert all(0 <= item.score <= 100 for item in batch)
    single = scorer.score_transaction(splits.X_test.iloc[[0]])
    assert single.score == pytest.approx(batch[0].score)
    assert single.risk_level == batch[0].risk_level


def test_anomaly_scores_use_training_reference_percentiles_without_labels():
    splits = prepare_dataset2(toy_dataset2())
    _, isolation = fitted_models(splits)
    scorer = build_risk_scorer(
        "IF", isolation, score_kind="anomaly", feature_names=splits.feature_names,
        reference_features=splits.X_train,
    )
    batch = scorer.score_transactions(splits.X_test.iloc[:10])
    raw = -isolation.score_samples(splits.X_test.iloc[:10])
    expected = np.searchsorted(np.asarray(scorer.sorted_reference_anomaly_scores), raw, side="right") / len(splits.X_train) * 100

    np.testing.assert_allclose([item.score for item in batch], expected)
    assert all(item.score_kind == "anomaly" for item in batch)
    assert all(item.score_basis == "percentile of training-reference anomaly scores" for item in batch)
    assert all(0 <= item.score <= 100 for item in batch)

    assert scorer.sorted_reference_anomaly_scores == tuple(sorted(scorer.sorted_reference_anomaly_scores))
    score_order = np.argsort(raw)
    ordered_risk = np.asarray([item.score for item in batch])[score_order]
    assert np.all(np.diff(ordered_risk) >= 0)


def test_risk_scorer_rejects_bad_estimator_scale_features_and_batches():
    splits = prepare_dataset2(toy_dataset2())
    logistic, isolation = fitted_models(splits)
    with pytest.raises(RiskScoringError, match="already-fitted"):
        build_risk_scorer("new", LogisticRegression(), score_kind="probability", feature_names=splits.feature_names)
    with pytest.raises(RiskScoringError, match="training-only reference"):
        build_risk_scorer("if", isolation, score_kind="anomaly", feature_names=splits.feature_names)
    with pytest.raises(RiskScoringError, match="only used for anomaly"):
        build_risk_scorer("lr", logistic, score_kind="probability", feature_names=splits.feature_names, reference_features=splits.X_train)
    with pytest.raises(RiskScoringError, match="low_upper"):
        RiskBandPolicy(low_upper=70, medium_upper=30)

    scorer = build_risk_scorer("lr", logistic, score_kind="probability", feature_names=splits.feature_names)
    with pytest.raises(RiskScoringError, match="one-row"):
        scorer.score_transaction(splits.X_test.iloc[:2])
    with pytest.raises(RiskScoringError, match="columns and order"):
        scorer.score_transactions(splits.X_test.rename(columns={"Time": "not-time"}).iloc[:1])
    non_finite = splits.X_test.iloc[:1].copy()
    non_finite.iloc[0, 0] = np.inf
    with pytest.raises(RiskScoringError, match="must be finite"):
        scorer.score_transaction(non_finite)
