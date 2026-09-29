# Data Ingestion

`Src/ingestion.py` provides reusable CSV discovery, loading, and validation. Dataset 1 tables and Dataset 2 are loaded independently; this module does not join them or train models.

```python
from Src.ingestion import load_dataset1, load_dataset2

dataset1_tables = load_dataset1()  # mapping by table name; values contain frame and report
dataset2 = load_dataset2()         # one LoadedCsv with frame and report
```

## Configuration and validation

- Dataset 1 defaults to the project's `Data/` directory; override with `FRAUD_DATASET1_DIR` pointing to a root containing the expected CSVs (discovery is recursive).
- Dataset 2 resolution uses `FRAUD_DATASET2_DIR` or the existing KaggleHub/downloader resolver.
- UTF-8 with BOM is attempted first; Windows-1252 is the fallback.
- Expected columns, target presence/domain, non-empty row counts, field counts, missing values, and exact duplicate rows are checked. A schema may request an exact row count.
- Missing values, unexpected columns, malformed rows, and invalid target values raise `IngestionError` with a dataset/path-specific message.
- Exact duplicate rows are counted and reported by default, never dropped. Pass `duplicate_policy="error"` to reject them. Dataset 2 currently has 1,081 duplicates; this loader preserves them for explicit later data-quality handling.
- Successful discovery and loads emit structured log records with paths, dimensions, encoding, quality counts, and target distribution.

Run the focused suite with `python -m pytest -q`.
