# Implementation Progress

- **Completed:** Data pipeline phases; EDA; leakage-safe preprocessing; four Dataset 2 models; unified evaluation; typed model-specific risk scoring.
- **Current feature:** Risk scoring complete. Raw request parsing, prediction endpoints, and model artifact persistence are still later features.
- **Tests/checks:** 68 tests PASS. Risk-score unit/integration coverage verifies probability-to-score mapping, single and batch inputs, configurable display tiers, anomaly percentile normalization using training features, score-basis labeling, invalid schema/NaN/infinity, and both existing model families. Full suite passes.
- **Scoring decisions:** Supervised estimators map positive-class output ×100 and explicitly mark it uncalibrated. Isolation Forest uses negative `score_samples` ranked against training-feature anomaly scores; no labels or validation/test rows are used as the reference. Default low/medium/high bands `<30`, `30–<70`, `>=70` are display-only, not decision thresholds.
- **Artifacts:** `Src/risk_scoring.py`, `tests/unit/test_risk_scoring.py`, `tests/integration/test_risk_scoring_integration.py`, and `RISK_SCORING.md`; README/project state updated.
- **Next:** Save/load model and preprocessing artifacts with enough metadata to preserve the score type, model configuration, and feature order.
