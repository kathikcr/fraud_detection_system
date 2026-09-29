# Dataset 1 Profile

**Scope:** Read-only inspection of the ten CSV files under `Data/`. No joins were materialized, no records were changed, and no model was trained. Dataset 1 is synthetic: `Src/data.py` samples `FraudIndicator`, `SuspiciousFlag`, and `AnomalyScore` independently using random draws. Its patterns are not evidence about real financial behavior.

## File inventory and schema

All CSVs loaded as UTF-8 with pandas. Each has 1,000 rows; no file is empty, has missing values, or has exact duplicate rows.

| File | Shape | Columns and inferred types | Unique-column candidates |
|---|---:|---|---|
| `Customer Profiles/account_activity.csv` | 1000 × 3 | `CustomerID` int64; `AccountBalance` float64; `LastLogin` string | `CustomerID`, `AccountBalance`, `LastLogin` |
| `Customer Profiles/customer_data.csv` | 1000 × 4 | `CustomerID` int64; `Name` string; `Age` int64; `Address` string | `CustomerID`, `Name`, `Address` |
| `Fraudulent Patterns/fraud_indicators.csv` | 1000 × 2 | `TransactionID` int64; `FraudIndicator` int64 | `TransactionID` |
| `Fraudulent Patterns/suspicious_activity.csv` | 1000 × 2 | `CustomerID` int64; `SuspiciousFlag` int64 | `CustomerID` |
| `Merchant Information/merchant_data.csv` | 1000 × 3 | `MerchantID` int64; `MerchantName` string; `Location` string | `MerchantID`, `MerchantName`, `Location` |
| `Merchant Information/transaction_category_labels.csv` | 1000 × 2 | `TransactionID` int64; `Category` string | `TransactionID` |
| `Transaction Amounts/amount_data.csv` | 1000 × 2 | `TransactionID` int64; `TransactionAmount` float64 | `TransactionID`, `TransactionAmount` |
| `Transaction Amounts/anomaly_scores.csv` | 1000 × 2 | `TransactionID` int64; `AnomalyScore` float64 | `TransactionID`, `AnomalyScore` |
| `Transaction Data/transaction_metadata.csv` | 1000 × 3 | `TransactionID` int64; `Timestamp` string; `MerchantID` int64 | `TransactionID`, `Timestamp` |
| `Transaction Data/transaction_records.csv` | 1000 × 3 | `TransactionID` int64; `Amount` float64; `CustomerID` int64 | `TransactionID`, `Amount` |

No missing values, exact duplicate rows, or all-null rows were found in any file. The ID columns listed above are unique within their respective tables, except `CustomerID` in `transaction_records` and `MerchantID` in `transaction_metadata`, which repeat as expected for transaction-to-entity references. Unique non-key values are observed in these particular samples and should not be assumed to be keys.

## Keys and relationship candidates

| Candidate relationship | Cardinality in files | Key coverage |
|---|---|---|
| `transaction_records` → `transaction_metadata` on `TransactionID` | 1:1 (1,000 each) | 1,000 overlap; no unmatched IDs either side |
| `transaction_records` → `fraud_indicators` on `TransactionID` | 1:1 | 1,000 overlap; no unmatched IDs either side |
| `transaction_records` → `amount_data` on `TransactionID` | 1:1 | 1,000 overlap; no unmatched IDs either side |
| `transaction_records` → `anomaly_scores` on `TransactionID` | 1:1 | 1,000 overlap; no unmatched IDs either side |
| `transaction_records` → `transaction_category_labels` on `TransactionID` | 1:1 | 1,000 overlap; no unmatched IDs either side |
| `transaction_records` → `customer_data` / `account_activity` on `CustomerID` | many:1 | All 636 distinct transaction customer IDs resolve; each customer table has 364 unused IDs |
| `transaction_records` → `suspicious_activity` on `CustomerID` | many:1 | All 636 distinct transaction customer IDs resolve; 364 customer IDs are not referenced by transactions |
| `transaction_metadata` → `merchant_data` on `MerchantID` | many:1 | All 651 distinct transaction merchant IDs resolve; 349 merchant IDs are not referenced by transactions |

The ID coverage/cardinalities make these plausible joins, but join construction and post-join row-count assertions belong to the later integration feature. `Amount` and `TransactionAmount` are separately generated random values in the source; they are not interchangeable measurements. `Timestamp` and `LastLogin` parse as dates: observed timestamp span is 2022-01-01 00:00 through 2022-02-11 15:00; observed last-login span is 2022-01-01 through 2024-09-26.

## Target and sensitive/derived field review

- **Target candidate:** `fraud_indicators.csv::FraudIndicator`; observed values are `{0: 955, 1: 45}` (4.5% positive), with no nulls and binary values only. Source code confirms it is independently sampled with probabilities 0.95 and 0.05, not derived from transaction behavior.
- **Other fraud/suspicion candidates:** `suspicious_activity.csv::SuspiciousFlag` has `{0: 977, 1: 23}` and is sampled independently at customer level. `anomaly_scores.csv::AnomalyScore` is a random uniform value, not a measured anomaly output. Treat these as synthetic demonstration/flag fields, not validated predictors.
- **PII:** `customer_data.csv` contains `Name` and `Address`. Do not expose these in an analytics view or use them as features by default.
- No second copy of `FraudIndicator` was found by schema inspection. Exclude target and suspicious/fraud/anomaly fields from any future predictor matrix pending explicit leakage review; this profile does not authorize model training.

## Validation performed

- Discovered exactly ten CSVs with the expected filenames under `Data/`.
- Read every CSV successfully as UTF-8; recorded shapes, inferred dtypes, null counts, exact duplicate counts, and unique-column/key candidates.
- Checked target values/class counts and key uniqueness/coverage for likely relationships.
- Parsed existing `Src/data.py` source as Python syntax without executing it (the script writes generated files when run).
- No project automated test suite is present. Git checkpoint unavailable because the workspace has no `.git` repository.

## Next feature

Implement the Dataset 2 KaggleHub downloader, reusing any existing local copy, validating the expected CSV, and keeping the downloaded raw data out of Git. No modeling until the required data-preparation stages are complete.
