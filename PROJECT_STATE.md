# Project State (Logistic Regression Baseline Complete)

- **Stack:** Python, pandas, scikit-learn, Matplotlib; no app framework.
- **Datasets:** Dataset 1 is synthetic and its fraud target is randomized. Dataset 2 is the Kaggle credit-card benchmark. Raw CSVs are excluded from Git.
- **Pipelines:** Independent validated loaders, safe Dataset 1 integration, separate EDA, and train-only leakage-safe preprocessing. Dataset 2 uses chronological 70/15/15 splits with equal timestamps grouped.
- **Baseline:** A single Dataset 2 Logistic Regression model uses class-balanced weights, `lbfgs`, max_iter 2,000, seed 42, and a fixed 0.5 threshold. It fits only the training partition; validation/test report precision, recall, F1, ROC-AUC, PR-AUC, and confusion counts. No resampling or threshold tuning.
- **Observed test result:** Precision 0.0570, recall 0.8269, F1 0.1067, ROC-AUC 0.9772, PR-AUC 0.7069; 711 false positives and 9 false negatives. High false-positive burden means this default threshold is only a baseline.
- **Report:** `reports/models/logistic_regression_baseline.md` contains validation and test metrics for the local snapshot. Dataset 1 is not modeled.
- **Tests:** 50 tests pass; local full Dataset 2 baseline completed without convergence warnings.
- **Next feature:** Random Forest as the next isolated model increment.
