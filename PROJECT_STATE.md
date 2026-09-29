# Project State (Transaction Inference Complete)

- **Stack:** Python, pandas, scikit-learn, XGBoost, Matplotlib, skops; no app framework.
- **Datasets:** Dataset 1 is synthetic and its fraud target is randomized; it is not modeled. Dataset 2 is the Kaggle credit-card benchmark. Raw CSVs are excluded from Git.
- **Pipelines and models:** Independent validated loaders, Dataset 1 integration, EDA, leakage-safe preprocessing, four Dataset 2 model baselines, and unified holdout evaluation.
- **Risk scoring and artifacts:** Typed probability/anomaly scoring plus safe save/load for four estimator families, with preprocessor and feature metadata verification.
- **Inference:** `Src/inference.py` validates raw Dataset 2 requests and runs batch/single predictions via loaded artifacts. It reports model alerts separately from display risk tiers; no HTTP endpoint exists yet. See `INFERENCE.md`.
- **Tests:** 94 tests pass, including inference schema validation, regression checks, and model artifact round trips.
- **Next feature:** SHAP explainability for model predictions.
