# Local SHAP explainability

`Src.explainability.FraudExplainer` returns local SHAP attributions for the 0–100 score already produced by a loaded Dataset 2 artifact. It supports the four approved estimators through SHAP's model-agnostic permutation explainer.

```python
from Src.artifacts import load_model_artifact
from Src.explainability import FraudExplainer

artifact = load_model_artifact("artifacts/xgboost-v1")
# Supply preprocessed rows from the training partition only.
explainer = FraudExplainer(artifact, X_train, seed=42, background_rows=100)
explanation = explainer.explain_transactions(X_new.iloc[[0]])[0]

print(explanation.score, explanation.baseline_score)
for item in sorted(explanation.attributions, key=lambda item: abs(item.shap_value), reverse=True)[:5]:
    print(item.feature_name, item.feature_value, item.shap_value)
```

The background must contain the estimator's exact ordered transformed features and should come only from the training partition. At most 100 rows are retained; larger backgrounds are sampled reproducibly using `seed`. Training examples are not embedded in model artifacts, so callers must retain an appropriate training-only background under their data-access policy.

Attributions are signed score points: positive values raise the score from the explainer's baseline and negative values lower it. Their sum with `baseline_score` reconstructs `score`. `max_evals` controls the number of permutation evaluations (default 500; minimum is `2 × feature_count + 1`). This is an approximate, model-agnostic explanation; higher values generally improve stability at greater inference cost. Explanations are checked for additivity before return.

All values and features are in the model's transformed input space. For this dataset the scaler centers and scales `Time`, `Amount`, and the anonymized PCA columns. `V1`–`V28` have no known business interpretation. Feature attributions describe this model and its background assumptions; they are not causal claims or standalone proof of fraud. For Isolation Forest, the explained output is its training-reference anomaly percentile and must not be described as fraud probability.

The independent masker varies features separately, which can produce combinations that are uncommon in the source data. The result is a local model explanation under this masking assumption, not a causal explanation. The current layer produces data objects only; plotting and HTTP endpoints are separate features.
