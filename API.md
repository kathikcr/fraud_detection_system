# HTTP API (incremental)

This checkpoint adds only one route: `GET /health`.

| Route | Purpose | Success response |
|---|---|---|
| `GET /health` | Confirm the API process can serve requests | `200 {"status":"ok"}` |

This is a process liveness check. It does not claim that a model, dataset, or prediction service is ready. Importing/creating the application does not load artifacts or datasets. Other routes and generated API documentation are disabled in this checkpoint so the API remains one endpoint at a time.

Run locally from the repository root:

```powershell
python -m uvicorn Src.api:app --host 127.0.0.1 --port 8000
```

Then request `http://127.0.0.1:8000/health`. The API uses FastAPI and Uvicorn; endpoint tests use FastAPI's `TestClient` with HTTPX. See `tests/api/test_health_endpoint.py`.
