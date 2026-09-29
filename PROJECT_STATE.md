# Project State (XGBoost Baseline Complete)

- **Stack:** Python, pandas, scikit-learn, XGBoost, Matplotlib; no app framework.
- **Datasets:** Dataset 1 is synthetic and has a randomized fraud target; it is not modeled. Dataset 2 is the Kaggle credit-card benchmark. Raw CSVs are excluded from Git.
- **Pipelines:** Independent validated loaders, Dataset 1 integration, separate EDA, leakage-safe preprocessing, and shared fraud-positive binary metrics. Dataset 2 uses chronological 70/15/15 splits, with tied times grouped and preprocessing fit on train only.
- **Models:** Dataset 2 Logistic Regression, Random Forest, and XGBoost baselines are implemented separately. All use the same split and threshold 0.5; no threshold tuning, resampling, or artifact persistence yet.
- **XGBoost setup:** Version 3.4.1 / Python 3.13.5; 300 rounds, depth 4, learning rate 0.05, `hist` CPU method, two workers, fixed seed 42, and `scale_pos_weight=518.1771` derived from training labels only.
- **Observed XGBoost test metrics:** Precision 0.4756, recall 0.7500, F1 0.5821, ROC-AUC 0.9844, PR-AUC 0.7681; TN/FP/FN/TP = 42,626/43/13/39. Results describe this dataset snapshot and fixed default threshold only; see `reports/models/xgboost_baseline.md`.
- **Other model reports:** `reports/models/logistic_regression_baseline.md` and `reports/models/random_forest_baseline.md` preserve their measured validation/test results.
- **Tests:** 57 tests pass; all three local model reports were generated from measured runs. No convergence or runtime failures.
- **Next feature:** Isolation Forest as a separate anomaly-detection increment.
