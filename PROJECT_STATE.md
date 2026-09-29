# Project State (Local Explainability Complete)

- **Stack:** Python, pandas, scikit-learn, XGBoost, Matplotlib, skops, SHAP; no app framework.
- **Datasets:** Dataset 1 is synthetic and its fraud target is randomized; it is not modeled. Dataset 2 is the Kaggle credit-card benchmark. Raw CSVs are excluded from Git.
- **Pipelines and models:** Independent validated loaders, Dataset 1 integration, EDA, leakage-safe preprocessing, four Dataset 2 model baselines, and unified holdout evaluation.
- **Risk scoring, artifacts, inference:** Typed probability/anomaly scoring; safe save/load; validated Dataset 2 single/batch inference via loaded artifacts.
- **Explainability:** `Src/explainability.py` returns local SHAP contributions for the score output of each supported model, with bounded training-only background data and additivity validation. Explanations are not causal and use transformed features. See `EXPLAINABILITY.md`.
- **Tests:** 100 tests pass, including SHAP score reconstruction for all four estimator families.
- **Next feature:** Transaction-level investigation workflow.
