# Transaction-level investigation

`Src.investigation.TransactionInvestigator` combines the existing raw transaction inference and SHAP layers into one reviewer-facing record for a single Dataset 2 transaction.

```python
from Src.artifacts import load_model_artifact
from Src.investigation import TransactionInvestigator

artifact = load_model_artifact("artifacts/logistic-regression-v1")
# X_train is the already-preprocessed training partition and is used only as
# the SHAP masker background.
investigator = TransactionInvestigator(artifact, X_train)
case = investigator.investigate(
    transaction_id="case-2026-001",  # caller-supplied reference
    transaction=raw_transaction,
)

print(case.prediction.model_alert, case.prediction.score, case.prediction.risk_level)
for item in case.top_contributors:
    print(item.feature_name, item.shap_value)
```

The Kaggle Dataset 2 schema has no transaction identifier. A caller-supplied case reference is therefore required and kept separate from the exact model feature payload. It is not passed into the estimator or included in logs. The investigation record returns the reference, source `Time` and `Amount`, model alert and typed score, full local explanation, top absolute SHAP contributors, and elapsed time. It does not assert an actual fraud label or persist the transaction.

The prediction and explanation must match before the investigation is returned. The distinction between a supervised uncalibrated probability score and Isolation Forest anomaly percentile is preserved. Display risk bands remain separate from the model's alert decision. SHAP contributions are transformed-feature model explanations, not causal evidence; see [EXPLAINABILITY.md](EXPLAINABILITY.md).
