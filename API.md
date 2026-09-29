# HTTP API (incremental)

The API currently exposes two routes:

| Route | Purpose | Success response |
|---|---|---|
| `GET /health` | Confirm the API process can serve requests | `200 {"status":"ok"}` |
| `POST /predict` | Score one raw Dataset 2 transaction | `200` prediction response |

`/health` is a process liveness check. It does not claim that a model, dataset, or prediction service is ready. `/predict` requires a versioned artifact directory from `Src.artifacts` and uses the existing `FraudInference` service. The artifact path is configured through `FRAUD_MODEL_ARTIFACT_DIR` or passed to `create_app()` by the host application. It is never accepted from a request. On the first valid prediction request the API loads and caches the artifact for the process lifetime; a restart is required to switch artifact versions.

The prediction JSON must contain exactly `Time`, `V1` through `V28`, and `Amount`, all as strict finite JSON numbers. `Time` and `Amount` must be non-negative. Missing features, strings, booleans, target `Class`, and extra fields are rejected with HTTP 422. Validation responses include field locations and error types but omit submitted values. Missing or invalid artifact configuration returns HTTP 503; a scoring failure returns a generic HTTP 500. The response includes `model_name`, `score`, `score_kind`, `score_basis`, `risk_level`, `model_alert`, `decision_method`, and `inference_ms`. Probability and anomaly semantics stay distinct; see [INFERENCE.md](INFERENCE.md).

Configure a local artifact and run from the repository root in PowerShell:

```powershell
$env:FRAUD_MODEL_ARTIFACT_DIR = "<path-to-versioned-artifact-directory>"
python -m uvicorn Src.api:app --host 127.0.0.1 --port 8000
```

Then request `http://127.0.0.1:8000/health` or post one transaction to `http://127.0.0.1:8000/predict`. The route surface is intentionally limited to these two endpoints; generated API docs are disabled until the API feature set is further along. Authentication and rate limiting are not implemented yet, so keep this endpoint bound to a trusted local network for now.

The API uses FastAPI and Uvicorn; route tests use FastAPI's `TestClient` with HTTPX. See `tests/api/`.
