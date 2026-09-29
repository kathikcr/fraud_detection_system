import csv
import json
from pathlib import Path

from Src.dataset2 import CSV_NAME, DATASET_HANDLE, EXPECTED_COLUMNS, resolve_dataset2


def test_downloads_validates_records_and_reuses_resolved_location(tmp_path):
    project = tmp_path / "project"
    download_dir = tmp_path / "kaggle-cache"
    returned = download_dir / "version" / CSV_NAME
    returned.parent.mkdir(parents=True)
    with returned.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(EXPECTED_COLUMNS)
        writer.writerow(["0"] * len(EXPECTED_COLUMNS))
        writer.writerow(["1"] * len(EXPECTED_COLUMNS))

    calls = []

    def downloader(handle, *, output_dir):
        calls.append((handle, output_dir))
        return str(download_dir)

    first = resolve_dataset2(project, downloader=downloader)
    second = resolve_dataset2(project, downloader=lambda *a, **k: calls.append("unexpected"))

    assert first == returned.resolve()
    assert second == first
    assert len(calls) == 1
    assert calls[0][0] == DATASET_HANDLE
    record = json.loads(
        (project / "Data" / ".dataset2_location.json").read_text(encoding="utf-8")
    )
    assert record["csv_path"] == str(first)
    assert record["row_count"] == 2
