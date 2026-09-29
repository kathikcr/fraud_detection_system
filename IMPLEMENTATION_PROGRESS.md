# Implementation Progress

- **Completed:** Dataset/data phases, EDA, leakage-safe preprocessing, four Dataset 2 model baselines, and unified evaluation utilities.
- **Current feature:** Unified evaluation complete; no model winner or tuned threshold selected.
- **Tests/checks:** 64 tests PASS. Unit/integration tests cover probability and anomaly score paths, metrics and curve data, fixed decision rules, estimator/schema validation, and report/chart output. Full suite passed with no warnings. Fresh full-data runs of all four estimators were evaluated on the shared local chronological splits; ROC, PR, and confusion charts were manually reviewed.
- **Measured comparison (test):** LR PR-AUC 0.7069, FP/FN 711/9; RF 0.7755, FP/FN 1/16; XGBoost 0.7681, FP/FN 43/13; Isolation Forest 0.0461, FP/FN 1,568/15. Each supervised threshold is fixed at 0.5; Isolation Forest reports its native cutoff and anomaly rate. See `reports/evaluation/model_comparison.md` for all measured validation/test metrics.
- **Artifacts:** `Src/evaluation.py`; `tests/unit/test_evaluation.py`; `tests/integration/test_evaluation_artifacts.py`; `reports/evaluation/model_comparison.md`, `roc_curves.png`, `precision_recall_curves.png`, `test_confusion_matrices.png`. Shared `Src/model_metrics.py` now separates arbitrary ranking scores from binary decisions.
- **Next:** Dataset 2 risk scoring. Model comparison remains descriptive on one chronological snapshot; threshold/operating-point choice remains unresolved.
