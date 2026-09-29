import csv

from Src.dataset2 import CSV_NAME, EXPECTED_COLUMNS
from Src.eda import generate_eda_reports


def test_generates_separate_reports_and_charts_without_pii(dataset1_csv_root, tmp_path):
    downloaded = tmp_path / "download"
    downloaded.mkdir()
    csv_path = downloaded / CSV_NAME
    rows = []
    for label, amount, time_value in ((0, 12.0, 0), (0, 30.0, 60), (1, 120.0, 120)):
        rows.append([time_value, *[0.1 * label for _ in range(28)], amount, label])
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(EXPECTED_COLUMNS)
        writer.writerows(rows)

    output = tmp_path / "eda"
    artifacts = generate_eda_reports(
        output,
        dataset1_root=dataset1_csv_root,
        dataset2_project_root=tmp_path / "project",
        downloader=lambda handle, **kwargs: str(downloaded),
    )

    assert artifacts.dataset1_report.is_file()
    assert artifacts.dataset2_report.is_file()
    assert all(path.is_file() and path.stat().st_size > 1000 for path in (
        artifacts.dataset1_chart, artifacts.dataset2_chart, artifacts.dataset2_pca_chart,
    ))
    dataset1_text = artifacts.dataset1_report.read_text(encoding="utf-8")
    dataset2_text = artifacts.dataset2_report.read_text(encoding="utf-8")
    assert "Synthetic data notice" in dataset1_text
    assert "no disclosed business interpretation" in dataset2_text
    assert "284,807" not in dataset1_text
    assert "Test A" not in dataset1_text and "Test B" not in dataset1_text
