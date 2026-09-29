# Fraud Detection Dataset

> **Synthetic data notice:** Dataset 1 is generated sample data. Its fraud and suspicious-activity indicators are randomly sampled; observed patterns do not represent real financial behavior.

🔒 Dataset Description
The Financial Fraud Detection Dataset contains data related to financial transactions and fraudulent patterns. It is designed for the purpose of training and evaluating machine learning models for fraud detection.

📁 Dataset Structure
The dataset is organized within the "data" folder and consists of several subfolders, each containing CSV files with specific information related to financial transactions, customer profiles, fraudulent patterns, transaction amounts, and merchant information. The dataset structure is as follows:

- 📂 data
  - 📂 Transaction Data
    - transaction_records.csv: Contains transaction records with details such as transaction ID, date, amount, and customer ID.
    - transaction_metadata.csv: Contains additional metadata for each transaction.

  - 📂 Customer Profiles
    - customer_data.csv: Includes customer profiles with information such as name, age, address, and contact details.
    - account_activity.csv: Provides details of customer account activity, including account balance, transaction history, and account status.

  - 📂 Fraudulent Patterns
    - fraud_indicators.csv: Contains indicators of fraudulent patterns and suspicious activities.
    - suspicious_activity.csv: Provides specific details of transactions flagged as suspicious.

  - 📂 Transaction Amounts
    - amount_data.csv: Includes transaction amounts for each transaction.
    - anomaly_scores.csv: Provides anomaly scores for transaction amounts, indicating potential fraudulence.

  - 📂 Merchant Information
    - merchant_data.csv: Contains information about merchants involved in transactions.
    - transaction_category_labels.csv: Provides category labels for different transaction types.

📂 src
- data.py: Python file containing code to generate the synthetic sample tables and random indicators.

💡 Usage
This dataset can be used for various purposes, including:

- Developing and evaluating machine learning models for financial fraud detection.
- Conducting research on fraud detection algorithms and techniques.
- Training data analysts and data scientists on fraud detection methodologies.

Feel free to use this dataset in your projects, experiments, or research. You are encouraged to create notebooks or other analysis tools to explore and visualize the data. If you find the dataset useful, please consider upvoting to show your support.

## Dataset 2: Credit Card Fraud

The project fetches the Kaggle `mlg-ulb/creditcardfraud` dataset through KaggleHub and reuses a validated local copy:

```powershell
python -m pip install -r requirements.txt
python -c "from Src.dataset2 import resolve_dataset2; print(resolve_dataset2())"
```

The downloaded raw CSV is stored under `Data/Dataset 2/` and excluded from Git. If Kaggle requires authentication in your environment, configure KaggleHub using `KAGGLE_API_TOKEN` or its user-level login/configuration; do not store credentials in source. See the [KaggleHub authentication instructions](https://github.com/Kaggle/kagglehub#authenticate).

Raw dataset CSVs are not checked into this repository. Place Dataset 1's files below `Data/` or set `FRAUD_DATASET1_DIR`; Dataset 2 is fetched automatically when it is not already available.

The observed file contains 284,807 rows, 31 columns, 492 positive `Class` labels, no missing values, and 1,081 exact duplicate rows. Duplicates are preserved and reported. Dataset 1 is synthetic and should not be presented as real-world fraud evidence.

## Validated CSV ingestion

The reusable loaders validate and report each dataset independently; they do not join data or train models:

```python
from Src.ingestion import load_dataset1, load_dataset2

dataset1_tables = load_dataset1()
dataset2 = load_dataset2()
```

Set `FRAUD_DATASET1_DIR` to change the Dataset 1 discovery root. Set `FRAUD_DATASET2_DIR` to a custom Dataset 2 CSV or directory. Install test dependencies with `python -m pip install -r requirements-dev.txt`, then run `python -m pytest -q`.

Dataset 1 relationship checks and the privacy-conscious transaction table are available via:

```python
from Src.dataset1_integration import load_dataset1_transaction_table

result = load_dataset1_transaction_table()
transactions = result.frame
audit = result.report
```

The integration step validates key coverage and join cardinality before merging. It preserves transaction grain and omits customer and account-profile fields from the output.

## Exploratory data analysis

Generate separate descriptive reports and charts after installing project dependencies and resolving the datasets:

```python
from Src.eda import generate_eda_reports

artifacts = generate_eda_reports()
```

Artifacts are written to `reports/eda/`. Dataset 1 is explicitly treated as synthetic. Dataset 2's severe class imbalance, amount/time distributions, and anonymized PCA component summaries are described independently. The EDA does not train models or claim PCA mean differences are feature importance.

## Leakage-safe preprocessing

`Src.preprocessing.prepare_dataset1()` and `prepare_dataset2()` create independent train/validation/test partitions and fit imputers/scalers on training data only. Dataset 1 uses a reproducible stratified split and excludes IDs and fraud-derived indicators. Dataset 2 uses forward chronological splits with tied timestamps kept together. See [PREPROCESSING.md](PREPROCESSING.md) for feature exclusions, split reasoning, and measured partition counts.

## Logistic Regression baseline

The first Dataset 2 baseline is available through `Src.logistic_baseline.run_logistic_regression_baseline(prepared_splits)`. It uses class weights and a fixed 0.5 threshold; it does not tune on test data. Measured validation/test results are in [reports/models/logistic_regression_baseline.md](reports/models/logistic_regression_baseline.md). At this threshold, recall is high but precision is low, with hundreds of false positives; the score is a baseline, not a deployment-ready operating point. No Dataset 1 model is trained because its target is randomly generated.

The second model is `Src.random_forest_baseline.run_random_forest_baseline(prepared_splits)`. It uses the same Dataset 2 splits and fixed threshold, with a bounded 200-tree configuration and balanced subsample weights. Results are in [reports/models/random_forest_baseline.md](reports/models/random_forest_baseline.md). On this snapshot it produced fewer false positives than Logistic Regression but lower recall; no test-based threshold tuning was done.

The third model is `Src.xgboost_baseline.run_xgboost_baseline(prepared_splits)`. It calculates `scale_pos_weight` from training labels only and uses a bounded CPU histogram-tree configuration. Results and exact settings are in [reports/models/xgboost_baseline.md](reports/models/xgboost_baseline.md). All three model reports use the same chronological partitions and default 0.5 threshold; these are benchmark measurements, not tuned operating points.

The fourth required model is `Src.isolation_forest_baseline.run_isolation_forest_baseline(prepared_splits)`. It is unsupervised: only training features are passed to fit. Its native outlier cutoff is fixed to `contamination="auto"`; continuous anomaly scores are ranking signals, not fraud probabilities. The measured test results show a substantial false-alert burden; see [reports/models/isolation_forest_baseline.md](reports/models/isolation_forest_baseline.md).

## Unified model evaluation

`Src.evaluation.evaluate_models(prepared_splits, specs)` evaluates already-fitted estimators on validation and test without fitting or tuning them. Provide an `EvaluationSpec` for each estimator to indicate whether it supplies fraud probabilities or anomaly scores. The output includes precision, recall, F1, ROC-AUC, PR-AUC, alert rate, confusion counts, ROC/precision-recall curves, and confusion-matrix charts. The local four-model comparison is in [reports/evaluation/model_comparison.md](reports/evaluation/model_comparison.md). Isolation Forest rankings are kept distinct from calibrated probabilities.

## Risk scoring

`Src.risk_scoring.build_risk_scorer()` wraps an already-fitted model to score one or more preprocessed transaction rows. Supervised scores are marked uncalibrated model probabilities scaled to 0–100; Isolation Forest uses a training-feature reference percentile, never a fraud probability. Default low/medium/high bands are display-only. See [RISK_SCORING.md](RISK_SCORING.md) for the distinctions and example.

## Model artifacts

`Src.artifacts.save_model_artifact()` and `load_model_artifact()` persist and restore supported fitted models with their training-fitted preprocessor and risk-scoring metadata. Bundles use skops serialization, checksums, a fixed model-family allowlist, and runtime/feature-order validation. Each output directory is immutable; write a new versioned directory for a changed model. See [ARTIFACTS.md](ARTIFACTS.md) for the format and usage.

## Raw transaction inference

`Src.inference.FraudInference` validates raw Dataset 2 transaction objects, transforms them with the loaded artifact's preprocessor, and returns a typed risk score and model alert decision for single transactions or batches. The local HTTP API exposes prediction and investigation routes; see [INFERENCE.md](INFERENCE.md) and [API.md](API.md) for details.

## Explainability

`Src.explainability.FraudExplainer` provides local SHAP attributions for a model score using a bounded, deterministic sample of training-only transformed features. It supports all four saved model families and preserves probability/anomaly score semantics. These are model explanations, not causal claims; `V1`–`V28` remain anonymized components. See [EXPLAINABILITY.md](EXPLAINABILITY.md).

## Transaction investigation

`Src.investigation.TransactionInvestigator` bundles inference and local explanation into a reviewer-facing record for one transaction. Since Dataset 2 has no transaction ID, callers provide an external case reference that stays separate from model inputs. See [INVESTIGATION.md](INVESTIGATION.md).

## HTTP API

The local dashboard is at `/`; its dataset selector loads an independent, cached overview for each validated dataset from `GET /dashboard/overview`. The views show observed target-label counts, amount summaries, time trends, and Dataset 1 category summaries. Dataset 1 is clearly marked synthetic, Dataset 2 time remains elapsed time, and personal customer fields are excluded. Predictive high-risk counts are not computed in this descriptive overview. The Model Performance section reads the saved Dataset 2 evaluation report, with a validation/test selector, confusion matrices, and ROC/precision-recall curves; it does not retrain or tune thresholds.

The local API also exposes `GET /health`, `POST /predict`, and `POST /investigate` for one validated Dataset 2 case with local SHAP evidence. Configure `FRAUD_MODEL_ARTIFACT_DIR` and, for investigations, `FRAUD_SHAP_BACKGROUND_PATH` before running locally with `python -m uvicorn Src.api:app --host 127.0.0.1 --port 8000`. The application is directly accessible for local use; authentication and deployment are out of scope. See [API.md](API.md).

