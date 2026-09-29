# Project State (Investigation API Complete)

- **Stack:** Python, pandas, scikit-learn, XGBoost, Matplotlib, skops, SHAP, FastAPI, Uvicorn; no frontend.
- **Datasets:** Dataset 1 is synthetic and its fraud target is randomized; it is not modeled. Dataset 2 is the Kaggle credit-card benchmark. Raw CSVs are excluded from Git.
- **ML service layer:** Four Dataset 2 baselines, evaluation, typed risk scoring, safe artifact persistence, validated transaction inference, local SHAP, and transaction investigation.
- **HTTP API:** `GET /health` liveness; `POST /predict` single prediction; `POST /investigate` single case plus local SHAP evidence. The investigation route loads a separate bounded training-only background from runtime config. See `API.md`.
- **API security:** Model endpoints are not authenticated or rate-limited yet and are documented for trusted local-network use only.
- **Tests:** 137 tests pass, including all API routes and previous service/model regression coverage.
- **Next feature:** API authentication and access control.
