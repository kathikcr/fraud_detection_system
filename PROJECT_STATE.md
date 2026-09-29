# Project State (Unified Evaluation Complete)

- **Stack:** Python, pandas, scikit-learn, XGBoost, Matplotlib; no app framework.
- **Datasets:** Dataset 1 is synthetic and its fraud target is randomized; it is not modeled. Dataset 2 is the Kaggle credit-card benchmark. Raw CSVs are excluded from Git.
- **Pipelines:** Independent validated loaders, Dataset 1 integration, separate EDA, leakage-safe preprocessing, and shared model metrics. Dataset 2 uses chronological 70/15/15 splits, ties grouped, transforms fit on train only.
- **Models:** Logistic Regression, Random Forest, XGBoost, and Isolation Forest baselines. Unified evaluation accepts fitted estimators and does not refit or tune thresholds. It handles supervised probability scores separately from Isolation Forest anomaly rankings.
- **Unified evaluation:** `Src/evaluation.py` reports validation/test precision, recall, F1, ROC-AUC, PR-AUC, prevalence, alert rate, and confusion counts; it also creates ROC/PR curves and test confusion matrices. The four models were freshly fitted on the same local snapshot for comparison. No winner or operating threshold was selected.
- **Observed test comparison:** LR PR-AUC 0.7069 (711 FP, 9 FN); RF 0.7755 (1 FP, 16 FN); XGBoost 0.7681 (43 FP, 13 FN); Isolation Forest 0.0461 (1,568 FP, 15 FN). Fixed probability thresholds were 0.5; Isolation Forest used native `contamination=auto` decisions. See `reports/evaluation/model_comparison.md` for all metrics and plots.
- **Tests:** 64 tests pass, including multi-model scoring, curve endpoints, fixed rules, invalid estimator/feature contracts, and artifact generation.
- **Next feature:** Dataset 2 risk scoring; no threshold tuning or deployment policy has been selected.
