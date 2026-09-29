# Project State (EDA Complete)

- **Stack:** Python and pandas data pipeline; no application framework is configured. Reports and charts use Matplotlib.
- **Datasets:** Dataset 1 is synthetic and its generated fraud/suspicious flags are random. Dataset 2 is the Kaggle credit-card benchmark fetched or reused through KaggleHub. Raw CSVs are ignored by Git.
- **Validated modules:** `Src/dataset2.py` resolves Dataset 2; `Src/ingestion.py` loads and validates both datasets independently; `Src/dataset1_integration.py` verifies Dataset 1 relationships and creates a transaction-grain view without customer/account profile fields; `Src/eda.py` produces separate descriptive EDA reports and charts.
- **EDA outputs:** `reports/eda/dataset1_eda.md`, `dataset2_eda.md`, `dataset1_overview.png`, `dataset2_overview.png`, and `dataset2_pca_components.png`. Dataset 1 results are synthetic-only. Dataset 2 summaries include severe imbalance, amount/time distributions, anonymized PCA component summaries, and duplicate counts. No model was trained.
- **Dataset observations:** Dataset 1 has 1,000 transactions, 45 generated positive labels, and no missing or exact duplicate rows. The observed Dataset 2 CSV has 284,807 rows, 492 positive labels, no missing values, and 1,081 exact duplicate rows; duplicates remain unchanged.
- **Tests:** 38 tests passed, including EDA unit and report-generation tests. The suite uses small generated fixtures and does not require raw datasets.
- **Next feature:** Leakage-safe preprocessing design, kept separate by dataset and preserving the synthetic-data limitation. Do not train models until the preprocessing phase is validated.
