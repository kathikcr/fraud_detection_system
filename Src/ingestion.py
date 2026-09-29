"""Validated, separate CSV loaders for the two fraud datasets."""

from __future__ import annotations

import csv
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Mapping

import pandas as pd

from Src.dataset2 import EXPECTED_COLUMNS as DATASET2_COLUMNS
from Src.dataset2 import resolve_dataset2

LOGGER = logging.getLogger(__name__)
DATASET1_ROOT_ENV = "FRAUD_DATASET1_DIR"
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET1_FILENAMES = (
    "account_activity.csv",
    "customer_data.csv",
    "fraud_indicators.csv",
    "suspicious_activity.csv",
    "merchant_data.csv",
    "transaction_category_labels.csv",
    "amount_data.csv",
    "anomaly_scores.csv",
    "transaction_metadata.csv",
    "transaction_records.csv",
)


class IngestionError(ValueError):
    """Raised when a source CSV fails discovery or validation."""


@dataclass(frozen=True)
class CsvSchema:
    name: str
    columns: tuple[str, ...]
    target_column: str | None = None
    target_values: frozenset[object] | None = None
    min_rows: int = 1
    expected_rows: int | None = None
    allow_missing: bool = False


@dataclass(frozen=True)
class CsvLoadReport:
    path: Path
    rows: int
    columns: tuple[str, ...]
    encoding: str
    missing_values: Mapping[str, int]
    duplicate_rows: int
    target_distribution: Mapping[object, int]


@dataclass(frozen=True)
class LoadedCsv:
    frame: pd.DataFrame
    report: CsvLoadReport


DATASET1_SCHEMAS: Mapping[str, CsvSchema] = {
    "account_activity": CsvSchema("Dataset 1 / account_activity", ("CustomerID", "AccountBalance", "LastLogin")),
    "customer_data": CsvSchema("Dataset 1 / customer_data", ("CustomerID", "Name", "Age", "Address")),
    "fraud_indicators": CsvSchema(
        "Dataset 1 / fraud_indicators", ("TransactionID", "FraudIndicator"),
        target_column="FraudIndicator", target_values=frozenset({0, 1}),
    ),
    "suspicious_activity": CsvSchema("Dataset 1 / suspicious_activity", ("CustomerID", "SuspiciousFlag")),
    "merchant_data": CsvSchema("Dataset 1 / merchant_data", ("MerchantID", "MerchantName", "Location")),
    "transaction_category_labels": CsvSchema("Dataset 1 / transaction_category_labels", ("TransactionID", "Category")),
    "amount_data": CsvSchema("Dataset 1 / amount_data", ("TransactionID", "TransactionAmount")),
    "anomaly_scores": CsvSchema("Dataset 1 / anomaly_scores", ("TransactionID", "AnomalyScore")),
    "transaction_metadata": CsvSchema("Dataset 1 / transaction_metadata", ("TransactionID", "Timestamp", "MerchantID")),
    "transaction_records": CsvSchema("Dataset 1 / transaction_records", ("TransactionID", "Amount", "CustomerID")),
}

DATASET2_SCHEMA = CsvSchema(
    "Dataset 2 / creditcard", tuple(DATASET2_COLUMNS), target_column="Class", target_values=frozenset({0, 1})
)


def discover_csv_files(
    root: str | Path,
    expected_filenames: tuple[str, ...] | list[str],
) -> dict[str, Path]:
    """Find each requested CSV below root and reject missing or ambiguous names."""
    base = Path(root).expanduser()
    if not base.is_dir():
        raise IngestionError(f"Dataset directory does not exist: {base}")

    wanted = set(expected_filenames)
    matches: dict[str, list[Path]] = {filename: [] for filename in wanted}
    for candidate in base.rglob("*.csv"):
        if candidate.name in wanted:
            matches[candidate.name].append(candidate)

    missing = sorted(name for name, paths in matches.items() if not paths)
    ambiguous = {name: paths for name, paths in matches.items() if len(paths) > 1}
    if missing:
        raise IngestionError(f"Missing expected CSV files below {base}: {', '.join(missing)}")
    if ambiguous:
        details = "; ".join(f"{name}: {paths}" for name, paths in ambiguous.items())
        raise IngestionError(f"Ambiguous CSV discovery below {base}: {details}")

    result = {name: paths[0] for name, paths in matches.items()}
    LOGGER.info(
        "csv_discovery_complete",
        extra={"event_type": "csv_discovery_complete", "dataset_root": str(base), "file_count": len(result)},
    )
    return result


def load_csv(
    path: str | Path,
    schema: CsvSchema,
    *,
    duplicate_policy: Literal["report", "error"] = "report",
) -> LoadedCsv:
    """Read and validate one CSV; duplicates are reported without being removed by default."""
    csv_path = Path(path).expanduser()
    if duplicate_policy not in {"report", "error"}:
        raise ValueError("duplicate_policy must be 'report' or 'error'")
    if not csv_path.is_file():
        raise IngestionError(f"CSV file does not exist for {schema.name}: {csv_path}")

    frame, encoding = _read_csv_with_encoding(csv_path, schema.name)
    if schema.target_column and schema.target_column not in frame.columns:
        raise IngestionError(
            f"Missing target column {schema.target_column!r} in {schema.name} ({csv_path})"
        )

    actual = set(frame.columns)
    expected = set(schema.columns)
    missing_columns = sorted(expected - actual)
    extra_columns = sorted(actual - expected)
    if missing_columns or extra_columns:
        problems = []
        if missing_columns:
            problems.append(f"missing columns: {missing_columns}")
        if extra_columns:
            problems.append(f"unexpected columns: {extra_columns}")
        raise IngestionError(f"Column validation failed for {schema.name} ({csv_path}): {'; '.join(problems)}")

    rows = len(frame)
    if rows < schema.min_rows:
        raise IngestionError(
            f"Row-count validation failed for {schema.name}: expected at least {schema.min_rows}, got {rows}"
        )
    if schema.expected_rows is not None and rows != schema.expected_rows:
        raise IngestionError(
            f"Row-count validation failed for {schema.name}: expected {schema.expected_rows}, got {rows}"
        )

    missing_values = {str(column): int(count) for column, count in frame.isna().sum().items() if count}
    if missing_values and not schema.allow_missing:
        raise IngestionError(f"Missing values in {schema.name}: {missing_values}")

    target_distribution: dict[object, int] = {}
    if schema.target_column:
        target = frame[schema.target_column]
        if schema.target_values is not None:
            invalid = target.loc[~target.isin(schema.target_values)].drop_duplicates().tolist()
            if invalid:
                raise IngestionError(
                    f"Invalid values in target {schema.target_column!r} for {schema.name}: {invalid}"
                )
        target_distribution = {
            value.item() if hasattr(value, "item") else value: int(count)
            for value, count in target.value_counts(dropna=False, sort=False).items()
        }

    duplicate_rows = int(frame.duplicated().sum())
    if duplicate_rows and duplicate_policy == "error":
        raise IngestionError(f"Found {duplicate_rows} exact duplicate rows in {schema.name} ({csv_path})")

    report = CsvLoadReport(
        path=csv_path.resolve(),
        rows=rows,
        columns=tuple(frame.columns),
        encoding=encoding,
        missing_values=missing_values,
        duplicate_rows=duplicate_rows,
        target_distribution=target_distribution,
    )
    LOGGER.info(
        "csv_loaded",
        extra={
            "event_type": "csv_loaded",
            "dataset": schema.name,
            "dataset_path": str(report.path),
            "row_count": report.rows,
            "column_count": len(report.columns),
            "encoding": report.encoding,
            "missing_value_columns": len(report.missing_values),
            "duplicate_rows": report.duplicate_rows,
            "target_distribution": dict(report.target_distribution),
        },
    )
    return LoadedCsv(frame=frame, report=report)


def load_dataset1(
    data_root: str | Path | None = None,
    *,
    duplicate_policy: Literal["report", "error"] = "report",
) -> dict[str, LoadedCsv]:
    """Discover and load Dataset 1 tables independently; does not join them."""
    configured = os.environ.get(DATASET1_ROOT_ENV)
    root = Path(data_root or configured or PROJECT_ROOT / "Data").expanduser()
    paths = discover_csv_files(root, DATASET1_FILENAMES)
    loaded: dict[str, LoadedCsv] = {}
    for filename in DATASET1_FILENAMES:
        stem = Path(filename).stem
        loaded[stem] = load_csv(paths[filename], DATASET1_SCHEMAS[stem], duplicate_policy=duplicate_policy)
    return loaded


def load_dataset2(
    project_root: str | Path | None = None,
    *,
    duplicate_policy: Literal["report", "error"] = "report",
    downloader=None,
) -> LoadedCsv:
    """Resolve Dataset 2 through KaggleHub/local cache and load it separately."""
    csv_path = resolve_dataset2(project_root=project_root, downloader=downloader)
    return load_csv(csv_path, DATASET2_SCHEMA, duplicate_policy=duplicate_policy)


def _read_csv_with_encoding(path: Path, dataset_name: str) -> tuple[pd.DataFrame, str]:
    for encoding in ("utf-8-sig", "cp1252"):
        try:
            frame = pd.read_csv(path, encoding=encoding, on_bad_lines="error")
            _validate_csv_structure(path, encoding, dataset_name)
            return frame, encoding
        except UnicodeDecodeError:
            if encoding == "cp1252":
                break
        except pd.errors.EmptyDataError as exc:
            raise IngestionError(f"CSV is empty or has no header for {dataset_name}: {path}") from exc
        except pd.errors.ParserError as exc:
            raise IngestionError(f"Malformed CSV for {dataset_name} ({path}): {exc}") from exc
        except OSError as exc:
            raise IngestionError(f"Could not read CSV for {dataset_name} ({path}): {exc}") from exc
    raise IngestionError(f"Could not decode CSV as UTF-8 or Windows-1252 for {dataset_name}: {path}")


def _validate_csv_structure(path: Path, encoding: str, dataset_name: str) -> None:
    """Catch short rows too; pandas otherwise pads them with NaN values."""
    try:
        with path.open("r", encoding=encoding, newline="") as stream:
            reader = csv.reader(stream, strict=True)
            header = next(reader, None)
            if not header:
                raise IngestionError(f"CSV is empty or has no header for {dataset_name}: {path}")
            for row in reader:
                if row and len(row) != len(header):
                    raise IngestionError(
                        f"Malformed CSV for {dataset_name} ({path}): line {reader.line_num} "
                        f"has {len(row)} fields; expected {len(header)}"
                    )
    except csv.Error as exc:
        raise IngestionError(f"Malformed CSV for {dataset_name} ({path}): {exc}") from exc
    except OSError as exc:
        raise IngestionError(f"Could not read CSV for {dataset_name} ({path}): {exc}") from exc
