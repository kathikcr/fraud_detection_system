# Risk Scoring

`Src.risk_scoring` wraps an already-fitted Dataset 2 estimator and converts its outputs to a display score in `[0, 100]`. The API expects rows already transformed by the train-fitted preprocessor. Raw transaction validation and transformation belong to the later inference layer.

## Supervised models

For Logistic Regression, Random Forest, or XGBoost, the positive-class estimator output is multiplied by 100. The result is explicitly marked `score_kind="probability"` and `score_basis="uncalibrated model probability × 100"`. The class-weighted training configurations can affect calibration; this number must not be presented as a calibrated fraud likelihood or combined across models.

## Isolation Forest

Isolation Forest has no fraud probability. It is configured as an anomaly score by passing the already-fitted detector plus **training features only** as `reference_features`. The scorer computes `-score_samples` for the training reference, sorts those values, then maps a new anomaly score to its empirical percentile `[0, 100]`. Higher means more anomalous relative to the training feature distribution. No training labels are used in this normalization. The result is marked `score_kind="anomaly"` and `score_basis="percentile of training-reference anomaly scores"`.

## Display bands

The default presentation bands are low `<30`, medium `30–<70`, and high `70–100`; callers can supply another `RiskBandPolicy`. These are display labels only, not fraud decision thresholds, tuned operating points, or service-level policies. Do not compare the numerical risk values across model types or datasets as if they shared a calibrated meaning.

## Example

```python
from Src.risk_scoring import build_risk_scorer

probability_scorer = build_risk_scorer(
    "XGBoost", xgb_result.model, score_kind="probability",
    feature_names=prepared.feature_names,
)
score = probability_scorer.score_transaction(prepared.X_test.iloc[[0]])

anomaly_scorer = build_risk_scorer(
    "Isolation Forest", isolation_result.model, score_kind="anomaly",
    feature_names=prepared.feature_names,
    reference_features=prepared.X_train,
)
anomaly_score = anomaly_scorer.score_transaction(prepared.X_test.iloc[[0]])
```

Both methods validate column names/order, numeric finite values, and row count. `score_transaction()` requires exactly one row; `score_transactions()` supports batches. Results contain only the score, score type/basis, model name, and display risk level; the caller retains responsibility for associating transaction identifiers.
