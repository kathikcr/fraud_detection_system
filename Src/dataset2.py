"""Locate, download, and validate the Kaggle credit-card fraud dataset."""

from __future__ import annotations

import csv
import json
import logging
import os
from pathlib import Path
from typing import Callable

DATASET_HANDLE = "mlg-ulb/creditcardfraud"
CSV_NAME = "creditcard.csv"
DATASET_DIR_ENV = "FRAUD_DATASET2_DIR"
EXPECTED_COLUMNS = ["Time", *(f"V{i}" for i in range(1, 29)), "Amount", "Class"]
LOGGER = logging.getLogger(__name__)


class Dataset2Error(RuntimeError):
    """Raised when Dataset 2 cannot be found, downloaded, or validated."""


def validate_creditcard_csv(path: str | Path, *, scan_rows: bool = True) -> int:
    """Validate the expected CSV header and row structure; return data row count."""
    csv_path = Path(path)
    if not csv_path.is_file():
        raise Dataset2Error(f"Expected Dataset 2 CSV does not exist: {csv_path}")
    if csv_path.stat().st_size == 0:
        raise Dataset2Error(f"Dataset 2 CSV is empty: {csv_path}")

    try:
        with csv_path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.reader(stream, strict=True)
            header = next(reader, None)
            if header != EXPECTED_COLUMNS:
                raise Dataset2Error(
                    f"Unexpected columns in {csv_path}; expected {EXPECTED_COLUMNS}, got {header}"
                )

            row_count = 0
            for row in reader:
                if not row:
                    continue
                if len(row) != len(EXPECTED_COLUMNS):
                    raise Dataset2Error(
                        f"Malformed row {reader.line_num} in {csv_path}: "
                        f"expected {len(EXPECTED_COLUMNS)} fields, got {len(row)}"
                    )
                row_count += 1
                if not scan_rows:
                    break
    except (OSError, UnicodeError, csv.Error, StopIteration) as exc:
        raise Dataset2Error(f"Could not read Dataset 2 CSV {csv_path}: {exc}") from exc

    if row_count == 0:
        raise Dataset2Error(f"Dataset 2 CSV has a valid header but no data rows: {csv_path}")
    return row_count


def _configured_candidates(root: Path) -> list[Path]:
    candidates: list[Path] = []
    configured = os.environ.get(DATASET_DIR_ENV)
    if configured:
        location = Path(configured).expanduser()
        candidates.append(location if location.name.lower() == CSV_NAME else location / CSV_NAME)

    # A persisted location supports KaggleHub's cache and avoids downloading again.
    location_record = root / "Data" / ".dataset2_location.json"
    if location_record.is_file():
        try:
            saved = json.loads(location_record.read_text(encoding="utf-8"))
            saved_path = saved.get("csv_path")
            if saved_path:
                candidates.append(Path(saved_path))
        except (OSError, json.JSONDecodeError, AttributeError):
            LOGGER.warning("Ignoring unreadable Dataset 2 location record: %s", location_record)

    default_dir = root / "Data" / "Dataset 2"
    candidates.append(default_dir / CSV_NAME)
    # Reuse a user-provided copy already somewhere in the project tree.
    candidates.extend(
        path for path in root.rglob(CSV_NAME)
        if not any(part.lower() in {".git", ".venv", "venv", "node_modules"} for part in path.parts)
    )

    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate.resolve()).casefold()
        if key not in seen:
            seen.add(key)
            unique.append(candidate)
    return unique


def resolve_dataset2(
    project_root: str | Path | None = None,
    *,
    downloader: Callable[..., str] | None = None,
) -> Path:
    """Return a validated local CSV, downloading it through KaggleHub if absent.

    Set ``FRAUD_DATASET2_DIR`` to a custom local directory or CSV file path.
    Otherwise downloads are placed under the project's ignored ``Data/Dataset 2``.
    """
    root = Path(project_root).resolve() if project_root else Path(__file__).resolve().parents[1]
    candidates = _configured_candidates(root)
    for candidate in candidates:
        if candidate.is_file():
            resolved = candidate.resolve()
            cached_rows = _cached_row_count(root, resolved)
            if cached_rows is None:
                rows = validate_creditcard_csv(candidate)
            else:
                validate_creditcard_csv(candidate, scan_rows=False)
                rows = cached_rows
            _record_location(root, resolved, rows)
            LOGGER.info("Reusing Dataset 2 at %s", resolved)
            return resolved

    output_dir = Path(os.environ.get(DATASET_DIR_ENV, root / "Data" / "Dataset 2")).expanduser()
    if output_dir.name.lower() == CSV_NAME:
        output_dir = output_dir.parent
    output_dir = output_dir.resolve()

    if downloader is None:
        try:
            import kagglehub
        except ImportError as exc:
            raise Dataset2Error(
                "KaggleHub is required to download Dataset 2. Install project requirements first."
            ) from exc
        downloader = kagglehub.dataset_download

    try:
        downloaded = Path(downloader(DATASET_HANDLE, output_dir=str(output_dir))).expanduser()
    except Exception as exc:
        raise Dataset2Error(
            "KaggleHub could not download mlg-ulb/creditcardfraud. "
            "If Kaggle requires authentication, configure KaggleHub credentials in the "
            "environment (KAGGLE_API_TOKEN) or user-level Kaggle configuration; never add "
            f"credentials to source. Failure type: {type(exc).__name__}."
        ) from None

    if downloaded.is_file() and downloaded.name.lower() == CSV_NAME:
        csv_path = downloaded
    elif downloaded.is_dir():
        matches = list(downloaded.rglob(CSV_NAME))
        if len(matches) != 1:
            raise Dataset2Error(
                f"KaggleHub returned {downloaded}, but expected exactly one {CSV_NAME}; found {len(matches)}"
            )
        csv_path = matches[0]
    else:
        raise Dataset2Error(f"KaggleHub returned an invalid dataset path: {downloaded}")

    row_count = validate_creditcard_csv(csv_path)
    resolved = csv_path.resolve()
    _record_location(root, resolved, row_count)
    LOGGER.info("Downloaded and validated Dataset 2 at %s (%d rows)", resolved, row_count)
    return resolved


def _record_location(root: Path, csv_path: Path, row_count: int) -> None:
    record = root / "Data" / ".dataset2_location.json"
    record.parent.mkdir(parents=True, exist_ok=True)
    record.write_text(
        json.dumps(
            {
                "dataset": DATASET_HANDLE,
                "csv_path": str(csv_path),
                "row_count": row_count,
                "file_size": csv_path.stat().st_size,
                "modified_ns": csv_path.stat().st_mtime_ns,
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )


def _cached_row_count(root: Path, csv_path: Path) -> int | None:
    record = root / "Data" / ".dataset2_location.json"
    try:
        saved = json.loads(record.read_text(encoding="utf-8"))
        stat = csv_path.stat()
        if (
            Path(saved["csv_path"]).resolve() == csv_path
            and saved.get("file_size") == stat.st_size
            and saved.get("modified_ns") == stat.st_mtime_ns
            and isinstance(saved.get("row_count"), int)
        ):
            return saved["row_count"]
    except (OSError, KeyError, ValueError, json.JSONDecodeError, TypeError):
        return None
    return None
