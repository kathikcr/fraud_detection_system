# Implementation Progress

- **Completed:** Data pipeline, Dataset 1 integration, EDA, leakage-safe preprocessing, four Dataset 2 baselines, unified evaluation, typed risk scoring, safe artifact save/load, transaction inference, and local SHAP explainability.
- **Current state:** `Src.explainability.FraudExplainer` explains the existing 0–100 score for all four supported estimator types with SHAP permutation attributions. It requires a bounded, ordered, training-only transformed background, verifies local additivity, and returns score basis plus signed per-feature values. No plots, investigation workflow, or API endpoints are included yet.
- **Tests/checks:** 100 tests PASS, including SHAP additivity and score parity for Logistic Regression, Random Forest, XGBoost, and Isolation Forest; invalid schema/background/evaluation settings; and full prior regression suites.
- **Explainability decisions:** Attributions use transformed input features and independent masking, are model-specific approximations rather than causal claims, and reconstruct the current display score. Isolation Forest explanations remain anomaly percentiles, never fraud probabilities. V1–V28 are anonymized components.
- **Dependency:** SHAP pinned to `0.52.0`; model artifacts do not embed training examples, so the explainer caller supplies a training-only background.
- **Docs:** `EXPLAINABILITY.md`; implementation in `Src/explainability.py`; integration tests in `tests/integration/test_explainability_integration.py`.
- **Next:** Transaction-level investigation, before beginning HTTP endpoints or dashboard features.
