# Leakage-Safe Preprocessing

`Src.preprocessing` prepares Dataset 1 and Dataset 2 independently. Call `prepare_dataset1()` with the validated Dataset 1 transaction table and `prepare_dataset2()` with the separately loaded Dataset 2 frame. Each returns prepared train, validation, and test features and labels, the fitted train-only transformer, feature names, split details, and class counts. The module does not train models, resample rows, tune thresholds, or fit on validation/test data.

## Dataset 1 — synthetic

The reproducible random split is stratified 70/15/15 (default seed 42). Candidate inputs are `Amount`, `TransactionAmount`, hour/day-of-week/month derived from `Timestamp`, and one-hot encoded `Category`. Numeric values are median-imputed and standardized; category values are most-frequent-imputed and one-hot encoded, with unseen categories ignored. All imputation and scaling parameters are fit on training rows.

Excluded fields: `FraudIndicator` is the target; `SuspiciousFlag` and `AnomalyScore` are fraud/post-event indicators and potential direct leakage; transaction/customer/merchant IDs are identifiers; merchant name/location are omitted because this synthetic source does not establish a justified predictor for them. Customer PII/profile fields were already excluded from the integrated analytical table. Dataset 1's randomly generated target means its split is a pipeline demonstration, not evidence of real-world predictive performance.

## Dataset 2 — credit-card benchmark

The observed CSV is ordered non-decreasing by `Time`, so the pipeline uses forward chronological partitions nearest to 70/15/15 by row count. Rows with the same `Time` value stay in one partition. This avoids training on later observations to predict earlier ones and avoids splitting tied transaction times across boundaries. It is intentionally not stratified; class prevalence is measured separately in each partition. Input must already be time-ordered; the pipeline rejects unsorted data rather than silently reordering it. `Time`, `Amount`, and `V1`–`V28` are numeric inputs; `Class` is kept only as the target. All numeric features are median-imputed and standardized using training rows only. `V1`–`V28` remain anonymized PCA components without business-level interpretation.

On the observed local snapshot, partition sizes were 199,364 / 42,722 / 42,721 (train/validation/test), with fraud counts 384 / 56 / 52. The time cut points were train through 132,928, validation 132,929–151,328, and test from 151,329 onward. Exact duplicate source rows are retained; tied timestamps are not divided. These values describe the local snapshot and are recomputed from the data when preparing splits.

No oversampling or class rebalancing occurs in preprocessing. If evaluated later, any resampling must be restricted to training data and the validation/test partitions left unchanged. See the later Logistic Regression baseline report for results measured on these partitions.
