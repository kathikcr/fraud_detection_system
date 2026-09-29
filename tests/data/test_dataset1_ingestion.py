from Src.ingestion import DATASET1_FILENAMES, DATASET1_ROOT_ENV, load_dataset1


def test_dataset1_tables_load_separately_with_target_and_quality_reports(monkeypatch, dataset1_csv_root):
    monkeypatch.setenv(DATASET1_ROOT_ENV, str(dataset1_csv_root))
    tables = load_dataset1()

    assert set(f"{name}.csv" for name in tables) == set(DATASET1_FILENAMES)
    assert all(table.report.rows == 3 for table in tables.values())
    assert all(table.report.duplicate_rows == 0 for table in tables.values())
    assert all(not table.report.missing_values for table in tables.values())
    assert tables["fraud_indicators"].report.target_distribution == {0: 2, 1: 1}
