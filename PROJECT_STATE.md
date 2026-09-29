# Project State (Risk Scoring Complete)

- **Stack:** Python, pandas, scikit-learn, XGBoost, Matplotlib; no app framework.
- **Datasets:** Dataset 1 is synthetic and its fraud target is randomized; it is not modeled. Dataset 2 is the Kaggle credit-card benchmark. Raw CSVs are excluded from Git.
- **Pipelines and models:** Independent validated loaders, Dataset 1 integration, EDA, leakage-safe preprocessing, four Dataset 2 model baselines, and unified holdout evaluation.
- **Risk scoring:** `Src/risk_scoring.py` wraps fitted models for one/batch scoring on preprocessed rows. Supervised outputs map uncalibrated positive-class probability to 0–100. Isolation Forest maps anomaly scores to percentiles against training features only and never labels them fraud probabilities. Configurable low/medium/high bands are presentation labels, not operating thresholds. See `RISK_SCORING.md`.
- **Model comparison:** `reports/evaluation/model_comparison.md` and accompanying plots; fixed test metrics from the local snapshot. No model winner or deployment threshold selected.
- **Tests:** 68 tests pass, including single/batch risk scoring, score basis distinctions, training-reference normalization, invalid feature handling, and the previous pipeline/model/evaluation suites.
- **Next feature:** Save/load artifacts for preprocessors/models/scorers, preserving configuration and score basis.
