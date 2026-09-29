# Implementation Progress

- **Completed:** Data pipeline phases; EDA; leakage-safe preprocessing; four Dataset 2 model baselines; unified evaluation; typed model-specific risk scoring; constrained model artifact save/load.
- **Current state:** The platform can persist and restore the four supported fitted estimator families with their preprocessor and risk scorer. Artifacts use skops, immutable directories, checksums, runtime compatibility checks, a serialized-type allowlist, and exact model/preprocessor/feature metadata validation.
- **Tests/checks:** 75 tests PASS, including round-trip risk score parity for Logistic Regression, Random Forest, XGBoost, and Isolation Forest; anomaly reference preservation; checksum and manifest tampering; and immutable destination behavior.
- **Scoring decisions:** Supervised estimators map positive-class output ×100 and explicitly mark it uncalibrated. Isolation Forest uses negative `score_samples` ranked against training-feature anomaly scores; no labels or validation/test rows are used as the reference. Default low/medium/high bands `<30`, `30–<70`, `>=70` are display-only, not decision thresholds.
- **Artifact docs:** `ARTIFACTS.md`; implementation in `Src/artifacts.py`; integration coverage in `tests/integration/test_artifact_persistence.py`.
- **Next:** Build raw transaction request validation and prediction-facing inference flow on top of the saved artifact interface.
