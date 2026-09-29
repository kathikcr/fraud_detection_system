# Implementation Progress

- **Completed:** Data pipeline, Dataset 1 integration, EDA, leakage-safe preprocessing, four Dataset 2 baselines, unified evaluation, typed risk scoring, safe artifact save/load, transaction inference, SHAP explainability, transaction-level investigation, `GET /health`, and single-transaction `POST /predict`.
- **Current state:** `/predict` accepts the exact finite numeric Dataset 2 schema, rejects extra/missing fields and negative Time/Amount, and returns explicit probability/anomaly score semantics. The service loads the artifact from `FRAUD_MODEL_ARTIFACT_DIR` on first valid prediction and caches it under a lock. The request cannot choose a model path.
- **Tests/checks:** 128 tests PASS, including API valid request/response parity, input/error edge cases, no-config/incompatible-artifact behavior, load-once caching, and the previous full regression suites.
- **API constraints:** `/health` remains liveness only. `/predict` is not authenticated or rate-limited yet; local/trusted-network use only. Validation error responses do not echo transaction values.
- **Docs:** `API.md`; routes in `Src/api.py`; API tests in `tests/api/`.
- **Next:** Add one further API endpoint (transaction investigation) only after this prediction route checkpoint passes.
