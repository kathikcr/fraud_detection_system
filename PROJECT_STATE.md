# Project State (Required Local Dashboard Complete)

- **Stack:** Python, pandas, scikit-learn, XGBoost, Matplotlib, skops, SHAP, FastAPI, Uvicorn; dashboard shell served from the Python app with no separate frontend framework.
- **Datasets:** Dataset 1 is synthetic and its fraud target is randomized; it is not modeled. Dataset 2 is the Kaggle credit-card benchmark. Raw CSVs are excluded from Git.
- **ML service layer:** Four Dataset 2 baselines, evaluation, typed risk scoring, safe artifact persistence, validated transaction inference, local SHAP, and transaction investigation.
- **HTTP API:** `GET /health`; cached `GET /dashboard/overview`, `/dashboard/performance`, and `/dashboard/quality`; two evaluation chart assets; `POST /predict`; `POST /investigate` single case plus local SHAP evidence. See `API.md`.
- **Dashboard:** Login-free shell at `GET /` with independent dataset analytics, validation/test model-performance, transaction investigation, and data-quality views. Dataset 1 is labelled synthetic and not modeled; no customer personal fields are exposed. See `API.md`.
- **Project scope:** Local development, demonstration, portfolio presentation, and academic use. No authentication, accounts, sessions, protected routes, per-user authorization, or deployment infrastructure.
- **Tests:** 148 tests pass, including Data Quality API behavior, privacy constraints, and dataset-specific validation checks.
- **Next feature:** No required local dashboard phases remain. Optional future work can be scoped separately. Authentication and deployment remain excluded by current instructions.
