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

