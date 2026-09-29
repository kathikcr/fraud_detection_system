# Project State (Single Prediction API Complete)

- **Stack:** Python, pandas, scikit-learn, XGBoost, Matplotlib, skops, SHAP, FastAPI, Uvicorn; no frontend.
- **Datasets:** Dataset 1 is synthetic and its fraud target is randomized; it is not modeled. Dataset 2 is the Kaggle credit-card benchmark. Raw CSVs are excluded from Git.
- **ML service layer:** Four Dataset 2 baselines, evaluation, typed risk scoring, safe artifact persistence, validated transaction inference, local SHAP, and transaction investigation.
- **HTTP API:** `GET /health` is process liveness; `POST /predict` validates and scores one Dataset 2 transaction using an artifact configured outside the request. Model load is cached. Authentication/rate limiting are pending. See `API.md`.
- **Tests:** 128 tests pass, including API request validation, response parity with the inference service, artifact caching, and previous regression coverage.
- **Next feature:** Transaction investigation API endpoint.
