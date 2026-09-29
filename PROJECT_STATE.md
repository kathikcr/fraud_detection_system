# Project State (Leakage-Safe Preprocessing Complete)

- **Stack:** Python, pandas, scikit-learn, and Matplotlib; no application framework is configured.
- **Datasets:** Dataset 1 is synthetic with independently randomized fraud/suspicious labels. Dataset 2 is the Kaggle credit-card benchmark fetched or reused through KaggleHub. Raw CSVs are ignored by Git.
- **Validated modules:** Dataset 2 acquisition (`Src/dataset2.py`); independent validated loaders (`Src/ingestion.py`); safe Dataset 1 joins (`Src/dataset1_integration.py`); separate descriptive EDA (`Src/eda.py`); independent leakage-safe partitions and train-only transformations (`Src/preprocessing.py`).
- **Preprocessing:** Dataset 1 uses seeded stratified 70/15/15 partitions and excludes target-derived indicators, identifiers, merchant descriptors, and PII. Dataset 2 uses forward chronological partitions with tied `Time` values grouped, and records each partition's class balance. Imputers/scalers fit on training rows only. No resampling or models yet; see `PREPROCESSING.md`.
- **Observed Dataset 2 split:** 199,364 / 42,722 / 42,721 rows, containing 384 / 56 / 52 fraud cases, respectively. Train time ends at 132,928; validation spans 132,929–151,328; test starts at 151,329. Exact duplicate rows remain preserved.
- **Tests:** 46 tests pass. The preprocessing module was also manually run against both locally available datasets.
- **Next feature:** Logistic Regression baseline, using prepared Dataset 2 splits. Dataset 1 remains synthetic demonstration data.
