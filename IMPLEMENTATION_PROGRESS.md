# Implementation Progress

- **Completed:** Data pipeline, Dataset 1 integration, EDA, leakage-safe preprocessing, four Dataset 2 baselines, unified evaluation, risk scoring, safe artifact persistence, transaction inference, SHAP explainability, transaction investigation, three API endpoints, and the directly accessible local dashboard shell/navigation.
- **Current state:** `/investigate` joins prediction and additive local SHAP results for one case. It accepts the caller's case ID separately from model features. It loads a bounded (max 100 rows, max 1 MB) transformed training-only SHAP background from `FRAUD_SHAP_BACKGROUND_PATH`; model and background are cached per process. Neither path is accepted in request data.
- **Tests/checks:** 138 tests PASS, including direct dashboard access, API investigation/prediction parity, SHAP reconstruction, invalid request and background configuration, oversized/non-finite background rejection, model/background load-once caching, and the full regression suite.
- **Environment warnings:** Two transitive Starlette/AnyIO and python-multipart deprecation warnings remain under TestClient; no application warnings or failures.
- **API constraints:** `/health` is liveness only. This local application intentionally has no login, accounts, sessions, protected routes, per-user authorization, or deployment infrastructure. Validation responses omit submitted values.
- **Dashboard phase:** Shell/navigation served at `/` by `Src/api.py`, implemented in `Src/dashboard.py`; no user sign-in or fake live metrics.
- **Docs:** `API.md`; dashboard code `Src/dashboard.py`; API tests under `tests/api/`.
- **Next:** Implement the dashboard overview with dataset selection and data-backed KPIs/charts. Keep the application directly accessible with no authentication; deployment remains out of scope.
