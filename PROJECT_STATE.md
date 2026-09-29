# Project State (Random Forest Baseline Complete)

- **Stack:** Python, pandas, scikit-learn, Matplotlib; no app framework.
- **Datasets:** Dataset 1 is synthetic and its fraud target is randomized. Dataset 2 is the Kaggle credit-card benchmark. Raw CSVs are excluded from Git.
- **Pipelines:** Independent validated loaders, safe Dataset 1 integration, separate EDA, and train-only leakage-safe preprocessing. Dataset 2 uses chronological 70/15/15 splits with equal timestamps grouped.
- **Models:** Dataset 2 Logistic Regression and Random Forest baselines are implemented as isolated modules using the same chronological splits and common binary metrics. Logistic Regression uses class-balanced weights; Random Forest uses 200 trees, depth cap 20, minimum leaf size 2, 0.8 bootstrap samples, and balanced subsample weights. Both use seed 42 and fixed 0.5 thresholds; neither tunes on test data. No resampling.
- **Observed test result:** Precision 0.0570, recall 0.8269, F1 0.1067, ROC-AUC 0.9772, PR-AUC 0.7069; 711 false positives and 9 false negatives. High false-positive burden means this default threshold is only a baseline.
- **Report:** `reports/models/logistic_regression_baseline.md` contains validation and test metrics for the local snapshot. Dataset 1 is not modeled.
- **Random Forest observed test result:** Precision 0.9730, recall 0.6923, F1 0.8090, ROC-AUC 0.9470, PR-AUC 0.7755; 1 false positive and 16 false negatives. See `reports/models/random_forest_baseline.md`. Single-snapshot measurement only; recall is lower than the Logistic Regression baseline at the same threshold.
- **Tests:** 53 tests pass; both full Dataset 2 model runs completed on the same local snapshot.
- **Next feature:** XGBoost as the next isolated model increment, after this checkpoint.
