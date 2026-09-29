# Logistic Regression Baseline — Dataset 2

This is the first supervised baseline for the credit-card benchmark. Dataset 1 is synthetic and is not pooled with this result.

## Configuration

- Features: 30 numeric inputs from Time, Amount, and anonymized V1–V28
- Split: chronological 70/15/15 by row count; equal Time values kept together; no shuffling
- Training rows: 199,364
- Class weight: `balanced`
- Solver / max iterations / random state: `lbfgs` / 2000 / 42
- Decision threshold: 0.50 (fixed default; not tuned)
- Preprocessing: train-fitted median imputation and standard scaling from `Src.preprocessing`
- Validation and test partitions were not used during fitting.

## Results

Metrics use fraud (`Class=1`) as the positive class. PR-AUC is average precision; no accuracy optimization was used.

| Partition | Rows | Precision | Recall | F1 | ROC-AUC | PR-AUC | TN | FP | FN | TP | Predicted fraud |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Validation | 42,722 | 0.053007 | 0.928571 | 0.100289 | 0.982762 | 0.839355 | 41,737 | 929 | 4 | 52 | 981 |
| Test | 42,721 | 0.057029 | 0.826923 | 0.106700 | 0.977236 | 0.706915 | 41,958 | 711 | 9 | 43 | 754 |

## Limitations

These are measured results for this dataset snapshot and chronological split. At the fixed 0.5 threshold, recall is high but precision is low: the test partition has 711 false positives for 43 true positives. Validation-based threshold selection, calibration, alternative imbalance strategies, and model comparison are separate work. V1–V28 are anonymized PCA components, not business-interpretable fields. This baseline does not establish deployment performance.
