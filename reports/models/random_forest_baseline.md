# Random Forest Baseline — Dataset 2

This is the second supervised benchmark for the credit-card dataset. Dataset 1 remains separate and is not modeled because its label is synthetic.

## Configuration

- Features: 30 numeric inputs from Time, Amount, and anonymized V1–V28
- Split: chronological 70/15/15 by row count; equal Time values kept together; no shuffling
- Training rows: 199,364
- Trees / maximum depth / minimum leaf rows: 200 / 20 / 2
- Bootstrap sample fraction per tree: 0.80
- Class weight / random state / jobs: `balanced_subsample` / 42 / 2
- Decision threshold: 0.50 (fixed default; not tuned)
- Preprocessing: train-fitted median imputation and standard scaling from `Src.preprocessing`
- Validation and test partitions were not used during fitting.

## Results

Metrics use fraud (`Class=1`) as the positive class. PR-AUC is average precision; no accuracy optimization was used.

| Partition | Rows | Precision | Recall | F1 | ROC-AUC | PR-AUC | TN | FP | FN | TP | Predicted fraud |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Validation | 42,722 | 1.000000 | 0.714286 | 0.833333 | 0.961397 | 0.868153 | 42,666 | 0 | 16 | 40 | 40 |
| Test | 42,721 | 0.972973 | 0.692308 | 0.808989 | 0.946956 | 0.775480 | 42,668 | 1 | 16 | 36 | 37 |

## Limitations

These are measured results for this dataset snapshot and chronological split. The 0.5 threshold is fixed and not operationally optimized. Compare the false-positive burden with recall before interpreting the result; threshold selection is a later phase and must use validation, not test. V1–V28 are anonymized PCA components without disclosed business meanings. No model artifact is persisted in this phase.
