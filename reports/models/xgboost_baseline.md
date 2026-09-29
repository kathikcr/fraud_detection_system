# XGBoost Baseline — Dataset 2

This is the third supervised benchmark for the credit-card dataset. Dataset 1 remains separate and is not modeled because its label is synthetic.

## Configuration

- Features: 30 numeric inputs from Time, Amount, and anonymized V1–V28
- Split: chronological 70/15/15 by row count; equal Time values kept together; no shuffling
- Training rows: 199,364
- XGBoost / Python versions: 3.4.1 / 3.13.5
- Trees / depth / learning rate: 300 / 4 / 0.05
- Row / column subsampling: 0.80 / 0.80
- `scale_pos_weight`: 518.1771 (training negatives / training positives only)
- `max_delta_step` / random state / jobs: 1 / 42 / 2
- Tree method / device / metric: `hist` / CPU / `aucpr`
- Decision threshold: 0.50 (fixed default; not tuned)
- Preprocessing: train-fitted median imputation and standard scaling from `Src.preprocessing`
- Validation and test partitions were not used during fitting or class-weight calculation.

## Results

Metrics use fraud (`Class=1`) as the positive class. PR-AUC is average precision; no accuracy optimization was used.

| Partition | Rows | Precision | Recall | F1 | ROC-AUC | PR-AUC | TN | FP | FN | TP | Predicted fraud |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Validation | 42,722 | 0.545455 | 0.857143 | 0.666667 | 0.987024 | 0.851849 | 42,626 | 40 | 8 | 48 | 88 |
| Test | 42,721 | 0.475610 | 0.750000 | 0.582090 | 0.984437 | 0.768073 | 42,626 | 43 | 13 | 39 | 82 |

## Limitations

These are measured results for this dataset snapshot and chronological split. The 0.5 threshold is fixed; any threshold selection must use validation and leave test untouched. Weighting can affect probability calibration and alert volume. V1–V28 are anonymized PCA components without disclosed business meanings. No model artifact is persisted in this phase.
