# Project State (Local Dashboard Shell Complete)

- **Stack:** Python, pandas, scikit-learn, XGBoost, Matplotlib, skops, SHAP, FastAPI, Uvicorn; dashboard shell served from the Python app with no separate frontend framework.
- **Datasets:** Dataset 1 is synthetic and its fraud target is randomized; it is not modeled. Dataset 2 is the Kaggle credit-card benchmark. Raw CSVs are excluded from Git.
- **ML service layer:** Four Dataset 2 baselines, evaluation, typed risk scoring, safe artifact persistence, validated transaction inference, local SHAP, and transaction investigation.
- **HTTP API:** `GET /health` liveness; `POST /predict` single prediction; `POST /investigate` single case plus local SHAP evidence. The investigation route loads a separate bounded training-only background from runtime config. See `API.md`.
- **Dashboard:** Login-free local shell and navigation at `GET /`; analytic pages are the next implementation phases.
- **Project scope:** Local development, demonstration, portfolio presentation, and academic use. No authentication, accounts, sessions, protected routes, per-user authorization, or deployment infrastructure.
- **Tests:** 138 tests pass, including dashboard access, API routes, and previous service/model regression coverage.
- **Next feature:** Implement the dashboard overview with dataset selection and data-backed KPIs/charts. The dashboard should remain directly accessible without login.
