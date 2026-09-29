# Dataset 2 transaction inference

`Src.inference.FraudInference` validates raw transaction objects, applies the saved training-fitted preprocessor, and returns typed model risk scores plus the model's alert decision. It is a Python service layer; the HTTP endpoint is a later feature.

```python
from Src.artifacts import load_model_artifact
from Src.inference import FraudInference

artifact = load_model_artifact("artifacts/logistic-regression-v1")
inference = FraudInference(artifact)
result = inference.predict_transaction({
    "Time": 1200.0,
    "V1": -1.2,
    # Include V2 through V28 as finite numeric values.
    "V2": 0.2,
    # ...
    "V28": 0.04,
    "Amount": 35.5,
})
```

Inputs must contain exactly `Time`, `V1` through `V28`, and `Amount`. Every value must be a finite numeric value; `Time` and `Amount` must be non-negative. Zero amount is accepted. Missing values, numeric strings, booleans, the target `Class`, and extra fields are rejected. Batch inference validates every record before scoring and returns no partial batch result.

`TransactionPrediction.model_alert` follows the saved decision rule: positive-class probability at or above the artifact threshold for supervised models, or the Isolation Forest native `predict == -1` decision. It is separate from the scorer's presentation-only low/medium/high band. `score_kind` and `score_basis` retain the distinction between uncalibrated supervised model probability and anomaly percentile. `inference_ms` is elapsed time for the whole call.

Logs record model identity, score type, row count, elapsed time, and error class; they never include transaction values. There is no arbitrary amount ceiling because the dataset contract defines non-negative finite amount rather than a universal operational maximum. Extremely large finite values are processed if transformation and scoring remain numerically valid; this layer does not claim out-of-distribution detection.
