import csv
from pathlib import Path

import pytest

from Src.dataset2 import (
    CSV_NAME,
    DATASET_HANDLE,
    EXPECTED_COLUMNS,
    Dataset2Error,
    resolve_dataset2,
    validate_creditcard_csv,
)


def write_csv(path: Path, *, rows=None, header=None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(header or EXPECTED_COLUMNS)
        for row in rows if rows is not None else [["0"] * len(EXPECTED_COLUMNS)]:
            writer.writerow(row)
    return path


def test_reuses_existing_local_dataset_without_downloading(tmp_path):
    csv_path = write_csv(tmp_path / "Data" / "Dataset 2" / CSV_NAME)

    def unexpected_download(*args, **kwargs):
        pytest.fail("Downloader must not be called when a valid local file exists")

    resolved = resolve_dataset2(tmp_path, downloader=unexpected_download)

    assert resolved == csv_path.resolve()
    assert (tmp_path / "Data" / ".dataset2_location.json").is_file()


def test_validates_header_and_counts_rows(tmp_path):
    csv_path = write_csv(tmp_path / CSV_NAME, rows=[["0"] * 31, ["1"] * 31])

    assert validate_creditcard_csv(csv_path) == 2


def test_rejects_unexpected_columns(tmp_path):
    csv_path = write_csv(tmp_path / CSV_NAME, header=["Time", "Class"])

    with pytest.raises(Dataset2Error, match="Unexpected columns"):
        validate_creditcard_csv(csv_path)


def test_rejects_malformed_row(tmp_path):
    csv_path = write_csv(tmp_path / CSV_NAME, rows=[["0", "1"]])

    with pytest.raises(Dataset2Error, match="Malformed row"):
        validate_creditcard_csv(csv_path)


def test_rejects_header_only_or_empty_csv(tmp_path):
    header_only = write_csv(tmp_path / "header_only.csv", rows=[])
    with pytest.raises(Dataset2Error, match="no data rows"):
        validate_creditcard_csv(header_only)

    empty = tmp_path / "empty.csv"
    empty.touch()
    with pytest.raises(Dataset2Error, match="empty"):
        validate_creditcard_csv(empty)


def test_rejects_missing_csv(tmp_path):
    with pytest.raises(Dataset2Error, match="does not exist"):
        validate_creditcard_csv(tmp_path / CSV_NAME)


def test_errors_when_download_result_lacks_expected_csv(tmp_path):
    downloaded_dir = tmp_path / "download"
    downloaded_dir.mkdir()
    (downloaded_dir / "other.csv").write_text("a,b\n1,2\n", encoding="utf-8")

    def downloader(handle, *, output_dir):
        assert handle == DATASET_HANDLE
        return str(downloaded_dir)

    with pytest.raises(Dataset2Error, match="expected exactly one creditcard.csv"):
        resolve_dataset2(tmp_path / "project", downloader=downloader)
