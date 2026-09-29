# Exploratory Data Analysis

EDA is generated independently for the synthetic Dataset 1 and the Kaggle credit-card Dataset 2. Run `generate_eda_reports()` from `Src.eda` after configuring dataset access. Markdown summaries and static charts are saved under `reports/eda/`.

## Dataset 1

Dataset 1's fraud flags are random synthetic indicators. Its fraud counts, category breakdowns, timestamp summaries, and amount distributions are useful for validating the analytics pipeline only; they are not estimates of real fraud behavior. The analytical view omits customer name, address, age, account balance, and login-time fields.

## Dataset 2

The benchmark profile describes the observed class imbalance, amount and source-scale time distributions, and per-class descriptive statistics for anonymized `V1`–`V28` components. The component means are descriptive summaries, not feature-importance estimates. The source's exact duplicate rows are preserved and counted.

No model is trained in this phase. The generated files reflect the local dataset version available when the report was generated; raw CSV files and credentials are excluded from version control.
