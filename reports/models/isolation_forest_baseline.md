# Isolation Forest Baseline — Dataset 2

Isolation Forest is fitted as an unsupervised anomaly detector on Dataset 2 training features only. Dataset 1 is synthetic and is not pooled into these results.

## Configuration

- Features: 30 numeric inputs from Time, Amount, and anonymized V1–V28
- Split: chronological 70/15/15 by row count; equal Time values kept together; no shuffling
- Training rows: 199,364
- Trees / max samples per tree: 300 / 256
- Contamination / native score offset / random state / jobs: `auto` / -0.500000 / 42 / 2
- Fit inputs: training feature matrix only; class labels are not passed to `fit`.
- `-score_samples` ranks observations (higher means more anomalous); `predict` supplies binary outlier decisions using the estimator's fixed native cutoff.
- Scores are not calibrated fraud probabilities. No validation threshold tuning or resampling was used.

## Results

Fraud (`Class=1`) is the positive class for evaluation only. PR-AUC is average precision computed from anomaly-score ranking.

| Partition | Rows | Anomaly rate | Precision | Recall | F1 | ROC-AUC | PR-AUC | TN | FP | FN | TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Validation | 42,722 | 5.1238% | 0.022385 | 0.875000 | 0.043653 | 0.939653 | 0.032430 | 40,526 | 2,140 | 7 | 49 |
| Test | 42,721 | 3.7569% | 0.023053 | 0.711538 | 0.044659 | 0.933870 | 0.046077 | 41,101 | 1,568 | 15 | 37 |

## Limitations

The outlier cutoff is the Isolation Forest `contamination=auto` default, not the known fraud prevalence. Low anomaly recall or precision reflects this unsupervised cutoff and features, not a probability threshold. Results describe one chronological dataset snapshot. `V1`–`V28` remain anonymized PCA components, not interpretable business fields. No model artifact is persisted in this phase.
