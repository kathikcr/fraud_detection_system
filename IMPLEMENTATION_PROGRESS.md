# Implementation Progress

- **Completed:** Data pipeline, Dataset 1 integration, EDA, leakage-safe preprocessing, four Dataset 2 baselines, unified evaluation, risk scoring, safe artifact persistence, transaction inference, SHAP explainability, transaction investigation, and three HTTP endpoints (`GET /health`, `POST /predict`, `POST /investigate`).
- **Current state:** `/investigate` joins prediction and additive local SHAP results for one case. It accepts the caller's case ID separately from model features. It loads a bounded (max 100 rows, max 1 MB) transformed training-only SHAP background from `FRAUD_SHAP_BACKGROUND_PATH`; model and background are cached per process. Neither path is accepted in request data.
- **Tests/checks:** 137 tests PASS, including API investigation/prediction parity, SHAP reconstruction, invalid request and background configuration, oversized/non-finite background rejection, model/background load-once caching, and the full previous regression suites.
- **Environment warnings:** Two transitive Starlette/AnyIO and python-multipart deprecation warnings remain under TestClient; no application warnings or failures.
- **API constraints:** `/health` is liveness only. `/predict` and `/investigate` remain unauthenticated and unrate-limited; trusted local-network use only. Validation responses omit submitted values.
- **Docs:** `API.md`; API code `Src/api.py`; API tests under `tests/api/`.
- **Next:** Add authentication/access control before making model endpoints available outside a trusted local network.
