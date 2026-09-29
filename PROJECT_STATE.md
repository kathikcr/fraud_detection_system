# Project State (Health API Complete)

- **Stack:** Python, pandas, scikit-learn, XGBoost, Matplotlib, skops, SHAP, FastAPI, Uvicorn; no frontend.
- **Datasets:** Dataset 1 is synthetic and its fraud target is randomized; it is not modeled. Dataset 2 is the Kaggle credit-card benchmark. Raw CSVs are excluded from Git.
- **ML service layer:** Four Dataset 2 baselines, evaluation, typed risk scoring, safe artifacts, validated transaction inference, local SHAP, and transaction investigation.
- **HTTP API:** `Src/api.py` currently exposes only `GET /health`, a process liveness check. No model is loaded and no prediction route is available yet. See `API.md`.
- **Tests:** 115 tests pass, including API route tests and all earlier model/service coverage.
- **Next feature:** Single-transaction prediction endpoint using `FraudInference`.
