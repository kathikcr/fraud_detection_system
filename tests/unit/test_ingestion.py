import csv
import logging
from pathlib import Path

import pytest

from Src.ingestion import (
    CsvSchema,
    IngestionError,
    discover_csv_files,
    load_csv,
)


SCHEMA = CsvSchema(
    "test transactions",
    ("id", "amount", "target"),
    target_column="target",
    target_values=frozenset({0, 1}),
)


def write_rows(path: Path, header, rows, *, encoding="utf-8") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding=encoding, newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)
    return path


def test_loads_csv_and_emits_structured_log(tmp_path, caplog):
    path = write_rows(tmp_path / "transactions.csv", SCHEMA.columns, [[1, 8.5, 0], [2, 3.1, 1]])

    with caplog.at_level(logging.INFO, logger="Src.ingestion"):
        result = load_csv(path, SCHEMA)

    assert result.frame.shape == (2, 3)
    assert result.report.rows == 2
    assert result.report.target_distribution == {0: 1, 1: 1}
    record = next(r for r in caplog.records if getattr(r, "event_type", None) == "csv_loaded")
    assert record.row_count == 2
    assert record.dataset == "test transactions"


def test_reads_windows_1252_when_utf8_decode_fails(tmp_path):
    schema = CsvSchema("encoded", ("id", "label"))
    path = write_rows(tmp_path / "encoded.csv", schema.columns, [[1, "café"]], encoding="cp1252")

    result = load_csv(path, schema)

    assert result.frame.loc[0, "label"] == "café"
    assert result.report.encoding == "cp1252"


def test_rejects_missing_csv(tmp_path):
    with pytest.raises(IngestionError, match="does not exist"):
        load_csv(tmp_path / "missing.csv", SCHEMA)


def test_rejects_empty_and_header_only_csv(tmp_path):
    empty = tmp_path / "empty.csv"
    empty.touch()
    with pytest.raises(IngestionError, match="empty or has no header"):
        load_csv(empty, SCHEMA)

    header_only = write_rows(tmp_path / "header.csv", SCHEMA.columns, [])
    with pytest.raises(IngestionError, match="at least 1"):
        load_csv(header_only, SCHEMA)


def test_rejects_malformed_csv(tmp_path):
    path = tmp_path / "malformed.csv"
    path.write_text("id,amount,target\n1,4,0\n2,8\n", encoding="utf-8")

    with pytest.raises(IngestionError, match="Malformed CSV"):
        load_csv(path, SCHEMA)


def test_rejects_unexpected_columns(tmp_path):
    path = write_rows(tmp_path / "extra.csv", (*SCHEMA.columns, "unexpected"), [[1, 2, 0, "x"]])

    with pytest.raises(IngestionError, match="unexpected columns"):
        load_csv(path, SCHEMA)


def test_rejects_missing_target(tmp_path):
    path = write_rows(tmp_path / "no_target.csv", ("id", "amount"), [[1, 2]])

    with pytest.raises(IngestionError, match="Missing target column"):
        load_csv(path, SCHEMA)


def test_rejects_invalid_target_values(tmp_path):
    path = write_rows(tmp_path / "bad_target.csv", SCHEMA.columns, [[1, 2, 2]])

    with pytest.raises(IngestionError, match="Invalid values in target"):
        load_csv(path, SCHEMA)


def test_rejects_missing_values(tmp_path):
    path = write_rows(tmp_path / "null.csv", SCHEMA.columns, [[1, "", 0]])

    with pytest.raises(IngestionError, match="Missing values"):
        load_csv(path, SCHEMA)


def test_reports_duplicates_without_deleting_and_can_reject_them(tmp_path):
    rows = [[1, 2.0, 0], [1, 2.0, 0], [2, 3.0, 1]]
    path = write_rows(tmp_path / "duplicates.csv", SCHEMA.columns, rows)

    result = load_csv(path, SCHEMA)

    assert result.report.duplicate_rows == 1
    assert len(result.frame) == 3
    with pytest.raises(IngestionError, match="1 exact duplicate rows"):
        load_csv(path, SCHEMA, duplicate_policy="error")


def test_validates_expected_row_count(tmp_path):
    path = write_rows(tmp_path / "rows.csv", SCHEMA.columns, [[1, 2, 0]])
    schema = CsvSchema("fixed", SCHEMA.columns, expected_rows=2)

    with pytest.raises(IngestionError, match="expected 2, got 1"):
        load_csv(path, schema)


def test_discovers_named_csvs_and_errors_for_missing_or_ambiguous_files(tmp_path):
    write_rows(tmp_path / "a" / "one.csv", ("x",), [[1]])
    assert discover_csv_files(tmp_path, ["one.csv"]) == {"one.csv": tmp_path / "a" / "one.csv"}

    with pytest.raises(IngestionError, match="Missing expected CSV"):
        discover_csv_files(tmp_path, ["absent.csv"])

    write_rows(tmp_path / "b" / "one.csv", ("x",), [[2]])
    with pytest.raises(IngestionError, match="Ambiguous CSV discovery"):
        discover_csv_files(tmp_path, ["one.csv"])
