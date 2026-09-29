# Implementation Progress

- **Completed:** Dataset/data pipeline phases; EDA; leakage-safe preprocessing; Logistic Regression, Random Forest, XGBoost, and Isolation Forest baselines for Dataset 2.
- **Current feature:** Isolation Forest complete. Dataset 1 remains unmodeled because its label is randomly generated.
- **Tests/checks:** 60 tests PASS. Isolation Forest unit/integration tests cover train-feature-only fitting, label independence, score orientation, native outlier decisions, metric bounds, bad input rejection, and report output. Existing ingestion, preprocessing, and three supervised model suites also pass. Full local Dataset 2 run succeeded.
- **Configuration:** IsolationForest with 300 trees, max_samples=256, contamination=`auto`, seed 42, two workers. Trained using X_train only; target labels did not affect fit or threshold. Fraud labels are used for validation/test metrics.
- **Measured Dataset 2 test metrics:** Precision 0.0231, recall 0.7115, F1 0.0447, ROC-AUC 0.9339, PR-AUC 0.0461; TN/FP/FN/TP = 41,101/1,568/15/37. Native detector flagged 3.7569% of test transactions as anomalies versus 0.1217% fraud prevalence. These are anomaly rankings/native binary decisions, not probability predictions.
- **Artifacts:** `Src/isolation_forest_baseline.py`, `tests/unit/test_isolation_forest_baseline.py`, `tests/integration/test_isolation_forest_integration.py`, and `reports/models/isolation_forest_baseline.md`; shared `Src/model_metrics.py` now supports continuous ranking scores and fixed binary decisions.
- **Next:** Unified evaluation utilities to compare measured model outputs, then risk scoring. No model winner or deployable threshold has been selected.
