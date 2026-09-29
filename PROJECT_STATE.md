# Project State (Transaction Investigation Complete)

- **Stack:** Python, pandas, scikit-learn, XGBoost, Matplotlib, skops, SHAP; no app framework.
- **Datasets:** Dataset 1 is synthetic and its fraud target is randomized; it is not modeled. Dataset 2 is the Kaggle credit-card benchmark. Raw CSVs are excluded from Git.
- **Pipelines and models:** Independent validated loaders, Dataset 1 integration, EDA, leakage-safe preprocessing, four Dataset 2 model baselines, and unified holdout evaluation.
- **Risk, artifacts, inference, explainability:** Typed probability/anomaly risk scoring; safe save/load; validated raw Dataset 2 scoring; local SHAP explanations for all four models.
- **Transaction investigation:** `Src/investigation.py` combines a single transaction's prediction, alert, score semantics, and top local contributors. A caller-supplied case reference is metadata only. No actual fraud label is claimed and no case is persisted. See `INVESTIGATION.md`.
- **Tests:** 112 tests pass, including supervised/anomaly investigation integration and prior artifact, inference, explainability, and regression coverage.
- **Next feature:** HTTP API, implemented endpoint by endpoint.
