# Dataset 2 Acquisition

- **Source:** KaggleHub dataset handle `mlg-ulb/creditcardfraud`.
- **Resolved CSV:** `Data/Dataset 2/creditcard.csv`; the resolved path and row count are recorded in ignored `Data/.dataset2_location.json`.
- **Acquisition:** Downloaded into the project-relative raw-data directory. The resolver checks configured and existing local copies first, validates the CSV, records the path, and reuses it on later calls.
- **Observed after loading:** 284,807 rows × 31 columns; expected ordered columns (`Time`, `V1`–`V28`, `Amount`, `Class`) present; `Class` values `{0, 1}` with counts `{0: 284315, 1: 492}`; zero missing values; 1,081 exact duplicate rows.
- **Data handling:** Raw data is excluded via `.gitignore`; duplicate rows were measured but not removed or altered. Their treatment belongs to the later ingestion/data-quality feature.
- **Dependency:** `kagglehub==1.0.2` is declared in `requirements.txt`. KaggleHub authentication is needed only if the resource requires user consent or private access. If required, configure `KAGGLE_API_TOKEN` in the process environment or use KaggleHub's user-level login/configuration; never put credentials in source.
- **Verification:** Eight focused unit/integration tests pass. The real KaggleHub download and subsequent local reuse were manually verified. Dataset 1 checks still pass. There is no Git repository, so no Git checkpoint could be made.

KaggleHub authentication setup details: [KaggleHub README](https://github.com/Kaggle/kagglehub#authenticate).
