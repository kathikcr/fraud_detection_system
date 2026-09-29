import numpy as np
import pandas as pd

from Src.isolation_forest_baseline import run_isolation_forest_baseline, write_isolation_forest_report
from Src.preprocessing import prepare_dataset2


def test_isolation_forest_integrates_and_reports_scores_as_non_probabilities(tmp_path):
    row = np.arange(400)
    frame = pd.DataFrame({"Time": row // 4, "Amount": row % 31})
    for index in range(1, 29):
        frame[f"V{index}"] = np.cos(row / (index + 2))
    frame["Class"] = (row % 20 == 0).astype(int)

    result = run_isolation_forest_baseline(prepare_dataset2(frame))
    report = write_isolation_forest_report(result, tmp_path / "isolation-forest.md")
    content = report.read_text(encoding="utf-8")

    assert report.is_file()
    assert "unsupervised" in content
    assert "not calibrated fraud probabilities" in content
    assert "PR-AUC" in content and "Anomaly rate" in content
    assert result.validation_anomaly_rate == result.validation.predicted_positive / result.validation.rows
