# Dataset 1 Safe Integration

`Src/dataset1_integration.py` audits relationships and returns a transaction-grain analytical table only after all key/cardinality checks pass.

```python
from Src.dataset1_integration import load_dataset1_transaction_table

result = load_dataset1_transaction_table()
transactions = result.frame
audit = result.report
```

## Verified relationships

- `TransactionID` is unique in `transaction_records`; metadata, fraud indicators, category labels, amount data, and anomaly scores each join 1:1 on `TransactionID`.
- `CustomerID` in transaction records is many-to-one to customer profiles, account activity, and suspicious-activity records. Every referenced customer is present; each of those tables has 364 unreferenced dimension IDs in the current sample.
- `MerchantID` in transaction metadata is many-to-one to merchant data. Every referenced merchant is present; 349 merchant records are unreferenced in the current sample.
- The current join keeps 1,000 input transactions as 1,000 output rows and preserves the 955/45 `FraudIndicator` distribution.
- Duplicate/null keys, unresolved foreign keys, unexpected transaction-key orphan rows, invalid target values, or any row multiplication fail with a clear error.

The analytical table retains transaction, timestamp, merchant, category, amount, target, suspicious flag, and synthetic anomaly-score fields. `Amount` and independently generated `TransactionAmount` remain separate. Customer names, addresses, ages, account balances, and login timestamps are key-checked where relevant but excluded from the output. Fraud/suspicion/anomaly columns are for synthetic analytics and are not approved model predictors.
