import csv

from Src.dataset2 import CSV_NAME, EXPECTED_COLUMNS
from Src.ingestion import load_dataset2


def test_loads_dataset2_through_local_resolver_and_reports_duplicate_rows(tmp_path):
    project = tmp_path / "project"
    downloaded = tmp_path / "download"
    downloaded.mkdir()
    csv_path = downloaded / CSV_NAME
    duplicate = ["0"] * len(EXPECTED_COLUMNS)
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(EXPECTED_COLUMNS)
        writer.writerow(duplicate)
        writer.writerow(duplicate)
        writer.writerow(["1"] * len(EXPECTED_COLUMNS))

    result = load_dataset2(project, downloader=lambda handle, **kwargs: str(downloaded))

    assert result.frame.shape == (3, 31)
    assert result.report.target_distribution == {0: 2, 1: 1}
    assert result.report.duplicate_rows == 1
    assert len(result.frame) == 3
