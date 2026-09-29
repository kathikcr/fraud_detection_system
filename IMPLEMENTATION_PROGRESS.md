# Implementation Progress

- **Completed:** Repository inspection; Dataset 1 profile; Dataset 2 KaggleHub acquisition and reuse; reusable CSV ingestion; validated Dataset 1 joins; separate EDA reports/charts; leakage-safe preprocessing for both datasets.
- **Current feature:** Preprocessing complete. No models trained and no resampling performed.
- **Tests/checks:** 46 tests PASS with no warnings. New preprocessing coverage verifies split reproducibility/chronology, tied timestamp boundaries, train-only scaling, excluded leakage/ID fields, missing-value imputation, invalid targets, malformed features, infinity, and missing schema. Manual preparation against both local datasets succeeded.
- **Dataset 1:** Synthetic only. Uses seeded stratified 70/15/15 splits (seed 42), with 700/150/150 rows and fraud counts 32/7/6. Inputs: two separate amount columns, derived timestamp calendar values, and category. IDs, target, suspicious/anomaly indicators, merchant descriptors, and PII are excluded.
- **Dataset 2:** Uses forward chronological 70/15/15 partitions, keeping tied `Time` values together. The observed splits are 199,364/42,722/42,721 with fraud counts 384/56/52. Median imputation/scaling is fit on train only. `V1`–`V28` remain uninterpreted PCA dimensions. Details and snapshot cut points are in `PREPROCESSING.md`.
- **Artifacts:** `Src/preprocessing.py`, `tests/unit/test_preprocessing.py`, `PREPROCESSING.md`; README and project state updated; scikit-learn pinned in requirements.
- **Next:** Logistic Regression baseline, after this preprocessing checkpoint.
