# Implementation Progress

- **Completed:** Data pipeline, Dataset 1 integration, EDA, leakage-safe preprocessing, four Dataset 2 model baselines, unified evaluation, typed risk scoring, safe model artifact save/load, and raw Dataset 2 transaction inference.
- **Current state:** `Src.inference.FraudInference` accepts only a validated `LoadedArtifact`, validates the exact raw Dataset 2 numeric schema, uses the stored preprocessor, and returns risk score plus the model-specific alert decision. It logs model, score kind, count, duration, and error class without logging transaction values. No API endpoint is included yet.
- **Tests/checks:** 94 tests PASS, including malformed input cases, float overflow, extremely large finite amount handling, model alert threshold behavior, and Logistic Regression/Isolation Forest inference through serialized artifacts. Full regression suite passes.
- **Scoring decisions:** Supervised scores remain uncalibrated positive-class model probabilities scaled to 0–100. Isolation Forest remains an anomaly percentile against training-only scores. Display bands are separate from the alert decision.
- **Inference docs:** `INFERENCE.md`; implementation in `Src/inference.py`; unit and integration coverage in `tests/unit/test_inference.py` and `tests/integration/test_inference_integration.py`.
- **Next:** Implement explainability using SHAP as its own tested phase before transaction investigation or API endpoints.
