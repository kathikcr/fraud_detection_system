# Project State (Required Model Baselines Complete)

- **Stack:** Python, pandas, scikit-learn, XGBoost, Matplotlib; no app framework.
- **Datasets:** Dataset 1 is synthetic and has a randomized fraud target; it is not modeled. Dataset 2 is the Kaggle credit-card benchmark. Raw CSVs are excluded from Git.
- **Pipelines:** Independent validated loaders, Dataset 1 integration, separate EDA, leakage-safe preprocessing, and shared binary metrics. Dataset 2 uses chronological 70/15/15 splits, equal times grouped, transformations fit on train only.
- **Models:** Logistic Regression, Random Forest, XGBoost, and Isolation Forest are each implemented as independent Dataset 2 baselines. The first three use train labels; Isolation Forest receives only training features and uses labels only for holdout evaluation. No threshold tuning, resampling, or artifact persistence yet.
- **Isolation Forest test result:** Precision 0.0231, recall 0.7115, F1 0.0447, ROC-AUC 0.9339, PR-AUC 0.0461; TN/FP/FN/TP = 41,101/1,568/15/37. `contamination="auto"` flagged 3.7569% of test rows; scores are anomaly rankings, not probabilities. See `reports/models/isolation_forest_baseline.md`.
- **Supervised model reports:** `reports/models/logistic_regression_baseline.md`, `random_forest_baseline.md`, `xgboost_baseline.md`; results are local snapshot metrics at fixed threshold 0.5.
- **Tests:** 60 tests pass, including proof that Isolation Forest's fitted result is unchanged when training labels are changed.
- **Next feature:** Unified model evaluation/comparison utilities, then risk scoring.
