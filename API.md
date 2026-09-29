# HTTP API (incremental)

The local app currently serves the dashboard and four API routes:

| Route | Purpose | Success response |
|---|---|---|
| `GET /` | Open the login-free local dashboard shell | `200` HTML page |
| `GET /dashboard/overview?dataset=dataset1` or `dataset2` | Load independent cached KPIs and chart data for one validated dataset | `200` JSON overview |
| `GET /health` | Confirm the API process can serve requests | `200 {"status":"ok"}` |
| `POST /predict` | Score one raw Dataset 2 transaction | `200` prediction response |
| `POST /investigate` | Return one case's prediction and local SHAP explanation | `200` investigation response |

`/health` is a process liveness check. It does not claim that a model, dataset, or prediction service is ready. `/predict` requires a versioned artifact directory from `Src.artifacts` and uses the existing `FraudInference` service. The artifact path is configured through `FRAUD_MODEL_ARTIFACT_DIR` or passed to `create_app()` by the host application. It is never accepted from a request. On the first valid prediction request the API loads and caches the artifact for the process lifetime; a restart is required to switch artifact versions.

The `POST /predict` JSON must contain exactly `Time`, `V1` through `V28`, and `Amount`, all as strict finite JSON numbers. `Time` and `Amount` must be non-negative. Missing features, strings, booleans, target `Class`, and extra fields are rejected with HTTP 422. Validation responses include field locations and error types but omit submitted values. Missing or invalid artifact configuration returns HTTP 503; a scoring failure returns a generic HTTP 500. The response includes `model_name`, `score`, `score_kind`, `score_basis`, `risk_level`, `model_alert`, `decision_method`, and `inference_ms`. Probability and anomaly semantics stay distinct; see [INFERENCE.md](INFERENCE.md).

`POST /investigate` accepts an external `transaction_id` separately from the nested `transaction` feature object. It returns the prediction, an additive local SHAP explanation, and the five largest absolute contributors. Dataset 2 has no transaction ID of its own; the caller's reference is metadata and is never scored or logged. The route requires both `FRAUD_MODEL_ARTIFACT_DIR` and `FRAUD_SHAP_BACKGROUND_PATH`. The latter must point to a CSV containing 1–100 finite preprocessed training rows with the exact model feature names/order. Keep this background restricted to the training partition. It is loaded and cached on the first investigation request, subject to a 1 MB file limit. The output preserves the distinction between supervised scores and Isolation Forest anomaly percentiles; explanation caveats are in [EXPLAINABILITY.md](EXPLAINABILITY.md).

Create the background from the existing fitted preprocessing result's training partition only:

```python
background = splits.X_train.sample(n=min(100, len(splits.X_train)), random_state=42)
background.to_csv("<configured-training-background.csv>", index=False)
```

The request shape is `{"transaction_id":"case-123","transaction":{"Time":...,"V1":...,"V28":...,"Amount":...}}`; the nested transaction must include all 30 input features. The external case reference is not a model feature.

Configure a local artifact and run from the repository root in PowerShell:

```powershell
$env:FRAUD_MODEL_ARTIFACT_DIR = "<path-to-versioned-artifact-directory>"
$env:FRAUD_SHAP_BACKGROUND_PATH = "<path-to-training-background.csv>"
python -m uvicorn Src.api:app --host 127.0.0.1 --port 8000
```

Then open `http://127.0.0.1:8000/` for the dashboard shell, request `http://127.0.0.1:8000/health`, post one transaction to `http://127.0.0.1:8000/predict`, or submit a case to `http://127.0.0.1:8000/investigate`. Generated API docs are disabled. This project is for local development, demonstration, portfolio presentation, and academic use. User authentication, accounts, sessions, access control, rate limiting, and deployment infrastructure are outside the current scope; the local dashboard is directly accessible.

The overview defaults to Dataset 2 and accepts only `dataset1` or `dataset2`. Summaries use validated loaders and are cached for the process lifetime. Dataset 1 analytics use the joined transaction table, exclude personal fields, and are labelled synthetic. Dataset 2 `Time` is summarized as elapsed-time intervals, and `Amount` is shown in source units without a currency assumption. High-risk model counts are unavailable in this view because the overview displays observed target labels rather than recomputing model scores.

The API uses FastAPI and Uvicorn; route tests use FastAPI's `TestClient` with HTTPX. See `tests/api/`.
