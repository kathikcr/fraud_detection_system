# Project State (Artifact Persistence Complete)

- **Stack:** Python, pandas, scikit-learn, XGBoost, Matplotlib, skops; no app framework.
- **Datasets:** Dataset 1 is synthetic and its fraud target is randomized; it is not modeled. Dataset 2 is the Kaggle credit-card benchmark. Raw CSVs are excluded from Git.
- **Pipelines and models:** Independent validated loaders, Dataset 1 integration, EDA, leakage-safe preprocessing, four Dataset 2 model baselines, and unified holdout evaluation.
- **Risk scoring:** Supervised outputs map uncalibrated positive-class probability to 0–100. Isolation Forest maps anomaly scores to percentiles against training features only. Configurable risk bands are presentation labels, not operating thresholds. See `RISK_SCORING.md`.
- **Artifact persistence:** `Src/artifacts.py` safely saves and loads approved estimator/preprocessor bundles with skops. It verifies checksums, model family, runtime versions, score type, feature order, model/preprocessing configuration, and anomaly training reference metadata. Artifacts are immutable directories; see `ARTIFACTS.md`.
- **Tests:** 75 tests pass, including round-trip scoring for all four model families and artifact tamper detection.
- **Next feature:** Raw transaction request validation and inference flow using restored model artifacts.
