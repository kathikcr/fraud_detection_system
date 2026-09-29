# Project State (Dashboard Overview Complete)

- **Stack:** Python, pandas, scikit-learn, XGBoost, Matplotlib, skops, SHAP, FastAPI, Uvicorn; dashboard shell served from the Python app with no separate frontend framework.
- **Datasets:** Dataset 1 is synthetic and its fraud target is randomized; it is not modeled. Dataset 2 is the Kaggle credit-card benchmark. Raw CSVs are excluded from Git.
- **ML service layer:** Four Dataset 2 baselines, evaluation, typed risk scoring, safe artifact persistence, validated transaction inference, local SHAP, and transaction investigation.
- **HTTP API:** `GET /health`; cached `GET /dashboard/overview`; `POST /predict`; `POST /investigate` single case plus local SHAP evidence. See `API.md`.
- **Dashboard:** Login-free shell and a data-backed overview at `GET /`, with a separate dataset selector, KPI summaries, fraud split, amount distribution, trend, and available category breakdown. No customer personal fields are exposed; Dataset 1 is labelled synthetic. See `API.md`.
- **Project scope:** Local development, demonstration, portfolio presentation, and academic use. No authentication, accounts, sessions, protected routes, per-user authorization, or deployment infrastructure.
- **Tests:** 141 tests pass, including dataset-specific overview analytics, unavailable-data handling, dashboard access, API routes, and previous service/model regression coverage.
- **Next feature:** Implement the model-performance view using validated metrics and evaluation artifacts. The dashboard remains directly accessible without login.
